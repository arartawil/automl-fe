# MRMR (Minimum Redundancy Maximum Relevance) feature selection
"""
MRMR algorithm for feature selection.

This module provides the mRMRSelector class which implements the Minimum
Redundancy Maximum Relevance algorithm for feature selection. mRMR selects
features that are highly relevant to the target while minimizing redundancy
among selected features.

References
----------
Peng, H., Long, F., & Ding, C. (2005). Feature selection based on mutual
information: criteria of max-dependency, max-relevance, and min-redundancy.
IEEE Transactions on Pattern Analysis and Machine Intelligence, 27(8), 1226-1238.
"""

from typing import Dict, List, Optional, Union
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder


class mRMRSelector(BaseEstimator, TransformerMixin):
    """
    Minimum Redundancy Maximum Relevance (mRMR) feature selection.

    mRMR selects features that have high mutual information with the target
    (relevance) while having low mutual information with already selected
    features (redundancy). The algorithm uses a greedy forward selection
    approach to maximize (relevance - alpha * redundancy).

    Parameters
    ----------
    n_features : int or float, optional
        Number of features to select. If float between 0 and 1,
        interpreted as fraction of total features. If None, selects
        half of the features.
    alpha : float, default=1.0
        Trade-off parameter between relevance and redundancy.
        Higher values give more weight to reducing redundancy.
    relevance_method : str, default='mutual_info'
        Method for calculating relevance:
        - 'mutual_info': Mutual information with target
        - 'correlation': Absolute correlation with target
    redundancy_method : str, default='mutual_info'
        Method for calculating redundancy:
        - 'mutual_info': Mutual information between features
        - 'correlation': Absolute correlation between features
    discrete_target : bool or None, default=None
        Whether the target is discrete (classification) or continuous
        (regression). If None, automatically detected.
    random_state : int, optional
        Random state for reproducibility.

    Attributes
    ----------
    selected_features_ : list
        Names of selected features after fitting.
    feature_importances_ : dict
        mRMR scores for each feature.
    relevance_scores_ : dict
        Relevance scores (MI with target) for each feature.
    redundancy_matrix_ : pd.DataFrame
        Pairwise redundancy (MI) between features.
    selection_order_ : list
        Order in which features were selected.

    Examples
    --------
    >>> from automl_fe.selection import mRMRSelector
    >>> import pandas as pd
    >>> from sklearn.datasets import load_iris
    >>> 
    >>> data = load_iris()
    >>> X = pd.DataFrame(data.data, columns=data.feature_names)
    >>> y = pd.Series(data.target)
    >>> 
    >>> selector = mRMRSelector(n_features=3, alpha=1.0)
    >>> X_selected = selector.fit_transform(X, y)
    >>> print(selector.selected_features_)
    >>> print(selector.selection_order_)
    """

    def __init__(
        self,
        n_features: Optional[Union[int, float]] = None,
        alpha: float = 1.0,
        relevance_method: str = 'mutual_info',
        redundancy_method: str = 'mutual_info',
        discrete_target: Optional[bool] = None,
        random_state: Optional[int] = None,
    ):
        self.n_features = n_features
        self.alpha = alpha
        self.relevance_method = relevance_method
        self.redundancy_method = redundancy_method
        self.discrete_target = discrete_target
        self.random_state = random_state

        # Initialize state
        self.selected_features_: List[str] = []
        self.feature_importances_: Dict[str, float] = {}
        self.relevance_scores_: Dict[str, float] = {}
        self.redundancy_matrix_: Optional[pd.DataFrame] = None
        self.selection_order_: List[str] = []
        self._feature_names: List[str] = []

    def _get_n_features(self, total_features: int) -> int:
        """Calculate number of features to select."""
        if self.n_features is None:
            return max(1, total_features // 2)
        elif isinstance(self.n_features, float) and 0 < self.n_features < 1:
            return max(1, int(total_features * self.n_features))
        else:
            return min(int(self.n_features), total_features)

    def _detect_discrete_target(self, y: np.ndarray) -> bool:
        """Detect if target is discrete (classification) or continuous."""
        if y.dtype in ['object', 'category', 'bool']:
            return True
        n_unique = len(np.unique(y))
        return n_unique <= 20 and n_unique / len(y) < 0.05

    def _calculate_relevance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate relevance scores (mutual information with target).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target array.

        Returns
        -------
        np.ndarray
            Relevance scores for each feature.
        """
        if self.relevance_method == 'mutual_info':
            if self.discrete_target:
                return mutual_info_classif(
                    X, y, 
                    discrete_features=False,
                    random_state=self.random_state
                )
            else:
                return mutual_info_regression(
                    X, y,
                    discrete_features=False,
                    random_state=self.random_state
                )
        elif self.relevance_method == 'correlation':
            scores = np.zeros(X.shape[1])
            for i in range(X.shape[1]):
                if np.std(X[:, i]) > 0:
                    scores[i] = np.abs(np.corrcoef(X[:, i], y)[0, 1])
            return np.nan_to_num(scores)
        else:
            raise ValueError(f"Unknown relevance method: {self.relevance_method}")

    def _calculate_redundancy_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise redundancy between features.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Symmetric redundancy matrix.
        """
        n_features = X.shape[1]
        redundancy = np.zeros((n_features, n_features))

        if self.redundancy_method == 'mutual_info':
            # Calculate MI between each pair of features
            for i in range(n_features):
                # MI of feature with itself is its entropy (not needed)
                for j in range(i + 1, n_features):
                    # Use mutual_info_regression treating one feature as target
                    mi = mutual_info_regression(
                        X[:, i].reshape(-1, 1),
                        X[:, j],
                        discrete_features=False,
                        random_state=self.random_state
                    )[0]
                    redundancy[i, j] = mi
                    redundancy[j, i] = mi

        elif self.redundancy_method == 'correlation':
            corr = np.corrcoef(X.T)
            redundancy = np.abs(corr)
            np.fill_diagonal(redundancy, 0)

        else:
            raise ValueError(f"Unknown redundancy method: {self.redundancy_method}")

        return redundancy

    def _greedy_selection(
        self,
        relevance: np.ndarray,
        redundancy: np.ndarray,
        n_to_select: int,
    ) -> List[int]:
        """
        Greedy forward selection maximizing mRMR criterion.

        Parameters
        ----------
        relevance : np.ndarray
            Relevance scores for each feature.
        redundancy : np.ndarray
            Pairwise redundancy matrix.
        n_to_select : int
            Number of features to select.

        Returns
        -------
        list
            Indices of selected features in selection order.
        """
        n_features = len(relevance)
        selected = []
        remaining = list(range(n_features))

        # Select first feature with highest relevance
        first_idx = np.argmax(relevance)
        selected.append(first_idx)
        remaining.remove(first_idx)

        # Greedy selection of remaining features
        while len(selected) < n_to_select and remaining:
            best_score = float('-inf')
            best_feature = None

            for feature in remaining:
                # Calculate mRMR score
                rel = relevance[feature]
                
                # Average redundancy with already selected features
                if len(selected) > 0:
                    red = np.mean([redundancy[feature, s] for s in selected])
                else:
                    red = 0

                mrmr_score = rel - self.alpha * red

                if mrmr_score > best_score:
                    best_score = mrmr_score
                    best_feature = feature

            if best_feature is not None:
                selected.append(best_feature)
                remaining.remove(best_feature)
            else:
                break

        return selected

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'mRMRSelector':
        """
        Fit the mRMR selector to the data.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : pd.Series
            Target variable.

        Returns
        -------
        mRMRSelector
            Fitted selector.
        """
        self._feature_names = list(X.columns)
        X_array = X.values.astype(np.float64)
        y_array = y.values.ravel()

        # Handle categorical target
        if y_array.dtype == object:
            le = LabelEncoder()
            y_array = le.fit_transform(y_array)

        # Detect target type
        if self.discrete_target is None:
            self.discrete_target = self._detect_discrete_target(y_array)

        # Calculate relevance scores
        relevance = self._calculate_relevance(X_array, y_array)
        relevance = np.nan_to_num(relevance, nan=0.0)

        # Store relevance scores
        self.relevance_scores_ = dict(zip(self._feature_names, relevance))

        # Calculate redundancy matrix
        redundancy = self._calculate_redundancy_matrix(X_array)
        self.redundancy_matrix_ = pd.DataFrame(
            redundancy,
            index=self._feature_names,
            columns=self._feature_names
        )

        # Determine number of features to select
        n_to_select = self._get_n_features(len(self._feature_names))

        # Greedy mRMR selection
        selected_indices = self._greedy_selection(relevance, redundancy, n_to_select)

        # Store results
        self.selection_order_ = [self._feature_names[i] for i in selected_indices]
        self.selected_features_ = self.selection_order_.copy()

        # Calculate mRMR scores for selected features
        for i, idx in enumerate(selected_indices):
            rel = relevance[idx]
            if i > 0:
                red = np.mean([redundancy[idx, selected_indices[j]] 
                              for j in range(i)])
            else:
                red = 0
            mrmr_score = rel - self.alpha * red
            self.feature_importances_[self._feature_names[idx]] = mrmr_score

        # Add zero scores for non-selected features
        for name in self._feature_names:
            if name not in self.feature_importances_:
                self.feature_importances_[name] = 0.0

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by selecting features.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.

        Returns
        -------
        pd.DataFrame
            DataFrame with selected features only.
        """
        if not self.selected_features_:
            raise ValueError("Selector has not been fitted. Call fit() first.")
        
        return X[self.selected_features_].copy()

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)

    def get_relevance_scores(self) -> pd.DataFrame:
        """
        Get relevance scores for all features.

        Returns
        -------
        pd.DataFrame
            DataFrame with feature names and relevance scores.
        """
        return pd.DataFrame([
            {'feature': k, 'relevance': v}
            for k, v in self.relevance_scores_.items()
        ]).sort_values('relevance', ascending=False).reset_index(drop=True)

    def get_redundancy_matrix(self) -> pd.DataFrame:
        """
        Get pairwise redundancy matrix.

        Returns
        -------
        pd.DataFrame
            Symmetric DataFrame of pairwise feature redundancies.
        """
        if self.redundancy_matrix_ is None:
            raise ValueError("Selector has not been fitted.")
        return self.redundancy_matrix_.copy()

    def get_support(self, indices: bool = False) -> Union[np.ndarray, List[int]]:
        """
        Get mask or indices of selected features.

        Parameters
        ----------
        indices : bool, default=False
            If True, return indices instead of mask.

        Returns
        -------
        array or list
            Boolean mask or list of indices.
        """
        mask = np.array([f in self.selected_features_ for f in self._feature_names])
        
        if indices:
            return np.where(mask)[0].tolist()
        return mask
