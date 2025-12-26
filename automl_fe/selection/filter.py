# Filter-based feature selection methods
"""
Filter methods for feature selection based on statistical measures.

This module provides the FilterSelector class which implements various
filter-based feature selection techniques including mutual information,
chi-square, f-statistic, variance threshold, and correlation filtering.
"""

from typing import Dict, List, Optional, Union, Any
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
    chi2,
    f_classif,
    f_regression,
    VarianceThreshold,
    SelectKBest,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler


class FilterSelector(BaseEstimator, TransformerMixin):
    """
    Filter-based feature selection using statistical measures.

    This class implements multiple filter methods for feature selection,
    which evaluate features independently of any machine learning model.
    Filter methods are computationally efficient and suitable for high-
    dimensional datasets.

    Parameters
    ----------
    method : str, default='mutual_info'
        Selection method to use:
        - 'mutual_info': Mutual information
        - 'chi2': Chi-square test (classification only)
        - 'f_statistic': F-statistic (ANOVA)
        - 'variance': Variance threshold
        - 'correlation': Correlation with target
    n_features : int or float, optional
        Number of features to select. If float between 0 and 1,
        interpreted as fraction of total features.
    task : str, default='classification'
        Task type: 'classification' or 'regression'.
    variance_threshold : float, default=0.01
        Minimum variance for variance filtering.
    correlation_threshold : float, default=0.95
        Maximum correlation between features to remove redundancy.
    random_state : int, optional
        Random state for reproducibility.

    Attributes
    ----------
    selected_features_ : list
        Names of selected features after fitting.
    feature_importances_ : dict
        Feature importance scores from selection.
    feature_scores_ : pd.DataFrame
        DataFrame with all feature scores and rankings.

    Examples
    --------
    >>> from automl_fe.selection import FilterSelector
    >>> import pandas as pd
    >>> from sklearn.datasets import load_iris
    >>> 
    >>> data = load_iris()
    >>> X = pd.DataFrame(data.data, columns=data.feature_names)
    >>> y = pd.Series(data.target)
    >>> 
    >>> selector = FilterSelector(method='mutual_info', n_features=3)
    >>> X_selected = selector.fit_transform(X, y)
    >>> print(selector.selected_features_)
    """

    def __init__(
        self,
        method: str = 'mutual_info',
        n_features: Optional[Union[int, float]] = None,
        task: str = 'classification',
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        random_state: Optional[int] = None,
    ):
        self.method = method
        self.n_features = n_features
        self.task = task
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.random_state = random_state

        # Initialize state
        self.selected_features_: List[str] = []
        self.feature_importances_: Dict[str, float] = {}
        self.feature_scores_: Optional[pd.DataFrame] = None
        self._feature_names: List[str] = []

    def _get_n_features(self, total_features: int) -> int:
        """Calculate number of features to select."""
        if self.n_features is None:
            return total_features
        elif isinstance(self.n_features, float) and 0 < self.n_features < 1:
            return max(1, int(total_features * self.n_features))
        else:
            return min(int(self.n_features), total_features)

    def _mutual_info_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate mutual information scores."""
        if self.task == 'classification':
            return mutual_info_classif(
                X, y, random_state=self.random_state
            )
        else:
            return mutual_info_regression(
                X, y, random_state=self.random_state
            )

    def _chi2_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate chi-square scores (classification only)."""
        if self.task != 'classification':
            warnings.warn("Chi-square is only for classification. Using f_statistic instead.")
            return self._f_statistic_scores(X, y)
        
        # Chi2 requires non-negative features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        scores, _ = chi2(X_scaled, y)
        return scores

    def _f_statistic_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate F-statistic scores."""
        if self.task == 'classification':
            scores, _ = f_classif(X, y)
        else:
            scores, _ = f_regression(X, y)
        return scores

    def _variance_scores(self, X: np.ndarray) -> np.ndarray:
        """Calculate variance scores."""
        return np.var(X, axis=0)

    def _correlation_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate correlation with target."""
        scores = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            # Handle constant columns
            if np.std(X[:, i]) == 0:
                scores[i] = 0
            else:
                scores[i] = np.abs(np.corrcoef(X[:, i], y)[0, 1])
        return np.nan_to_num(scores)

    def _relief_scores(self, X: np.ndarray, y: np.ndarray, n_iterations: int = 100) -> np.ndarray:
        """
        Calculate Relief algorithm scores.
        
        Relief evaluates features by sampling instances and comparing 
        nearest neighbors of same/different class.
        """
        n_samples, n_features = X.shape
        scores = np.zeros(n_features)
        
        # Normalize features for distance calculation
        X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-10)
        
        rng = np.random.RandomState(self.random_state)
        
        for _ in range(min(n_iterations, n_samples)):
            # Sample random instance
            idx = rng.randint(n_samples)
            instance = X_norm[idx]
            instance_class = y[idx]
            
            # Find nearest hit (same class) and miss (different class)
            same_class_mask = y == instance_class
            diff_class_mask = ~same_class_mask
            
            if not any(same_class_mask) or not any(diff_class_mask):
                continue
                
            # Calculate distances
            same_class_distances = np.sum((X_norm[same_class_mask] - instance) ** 2, axis=1)
            diff_class_distances = np.sum((X_norm[diff_class_mask] - instance) ** 2, axis=1)
            
            # Find nearest (excluding self for same class)
            same_class_distances[same_class_distances == 0] = np.inf
            nearest_hit_idx = np.argmin(same_class_distances)
            nearest_miss_idx = np.argmin(diff_class_distances)
            
            nearest_hit = X_norm[same_class_mask][nearest_hit_idx]
            nearest_miss = X_norm[diff_class_mask][nearest_miss_idx]
            
            # Update scores
            scores -= (instance - nearest_hit) ** 2
            scores += (instance - nearest_miss) ** 2
        
        return scores / n_iterations

    def _remove_correlated_features(
        self, 
        X: pd.DataFrame, 
        scores: Dict[str, float]
    ) -> List[str]:
        """Remove highly correlated features, keeping higher-scored ones."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return list(X.columns)
        
        corr_matrix = X[numeric_cols].corr().abs()
        
        # Track features to remove
        to_remove = set()
        
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                if corr_matrix.iloc[i, j] > self.correlation_threshold:
                    # Remove the one with lower score
                    col_i = numeric_cols[i]
                    col_j = numeric_cols[j]
                    
                    if col_i in to_remove or col_j in to_remove:
                        continue
                    
                    score_i = scores.get(col_i, 0)
                    score_j = scores.get(col_j, 0)
                    
                    if score_i < score_j:
                        to_remove.add(col_i)
                    else:
                        to_remove.add(col_j)
        
        return [col for col in X.columns if col not in to_remove]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FilterSelector':
        """
        Fit the filter selector to the data.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : pd.Series
            Target variable.

        Returns
        -------
        FilterSelector
            Fitted selector.
        """
        self._feature_names = list(X.columns)
        X_array = X.values.astype(np.float64)
        y_array = y.values.ravel()

        # Calculate scores based on method
        if self.method == 'mutual_info':
            scores = self._mutual_info_scores(X_array, y_array)
        elif self.method == 'chi2':
            scores = self._chi2_scores(X_array, y_array)
        elif self.method == 'f_statistic':
            scores = self._f_statistic_scores(X_array, y_array)
        elif self.method == 'variance':
            scores = self._variance_scores(X_array)
        elif self.method == 'correlation':
            scores = self._correlation_scores(X_array, y_array)
        elif self.method == 'relief':
            scores = self._relief_scores(X_array, y_array)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Handle NaN scores
        scores = np.nan_to_num(scores, nan=0.0)

        # Create feature scores DataFrame
        self.feature_scores_ = pd.DataFrame({
            'feature': self._feature_names,
            'score': scores,
            'rank': pd.Series(scores).rank(ascending=False).astype(int)
        }).sort_values('score', ascending=False)

        # Store feature importances
        self.feature_importances_ = dict(zip(self._feature_names, scores))

        # Determine number of features to select
        n_to_select = self._get_n_features(len(self._feature_names))

        # Select top features
        top_features = self.feature_scores_.head(n_to_select)['feature'].tolist()

        # Optionally remove correlated features
        if self.correlation_threshold < 1.0:
            top_features = self._remove_correlated_features(
                X[top_features], 
                self.feature_importances_
            )

        self.selected_features_ = top_features

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

    def get_feature_scores(self) -> pd.DataFrame:
        """
        Get feature scores and rankings.

        Returns
        -------
        pd.DataFrame
            DataFrame with feature names, scores, and rankings.
        """
        if self.feature_scores_ is None:
            raise ValueError("Selector has not been fitted.")
        return self.feature_scores_.copy()

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
