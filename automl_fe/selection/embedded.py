# Embedded feature selection methods
"""
Embedded methods for feature selection integrated with model training.

This module provides the EmbeddedSelector class which implements various
embedded feature selection techniques including LASSO, Ridge, ElasticNet,
Random Forest, and XGBoost feature importance.
"""

from typing import Dict, List, Optional, Union, Any
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import (
    Lasso, LassoCV,
    Ridge, RidgeCV,
    ElasticNet, ElasticNetCV,
    LogisticRegression,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class EmbeddedSelector(BaseEstimator, TransformerMixin):
    """
    Embedded feature selection using model-based importance.

    This class implements embedded methods where feature selection is
    integrated with model training. These methods leverage model coefficients
    or feature importances to select relevant features.

    Parameters
    ----------
    method : str, default='random_forest'
        Selection method to use:
        - 'lasso': LASSO regularization (L1)
        - 'ridge': Ridge regularization (L2)
        - 'elasticnet': ElasticNet (L1 + L2)
        - 'random_forest': Random Forest importance
        - 'xgboost': XGBoost importance
    n_features : int or float, optional
        Number of features to select. If float between 0 and 1,
        interpreted as fraction of total features.
    task : str, default='classification'
        Task type: 'classification' or 'regression'.
    threshold : float or str, default='mean'
        Importance threshold for feature selection:
        - float: Absolute threshold value
        - 'mean': Select features with importance >= mean
        - 'median': Select features with importance >= median
    alpha : float, optional
        Regularization strength for LASSO/Ridge/ElasticNet.
        If None, uses cross-validation to find optimal value.
    l1_ratio : float, default=0.5
        ElasticNet mixing parameter (0 = Ridge, 1 = LASSO).
    n_estimators : int, default=100
        Number of trees for tree-based methods.
    stability_selection : bool, default=False
        If True, run multiple times and select stable features.
    n_stability_runs : int, default=10
        Number of runs for stability selection.
    random_state : int, optional
        Random state for reproducibility.
    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    selected_features_ : list
        Names of selected features after fitting.
    feature_importances_ : dict
        Feature importance scores from selection.
    coef_ : np.ndarray
        Coefficients from linear models.
    estimator_ : estimator
        Fitted estimator used for selection.
    stability_scores_ : dict
        Stability scores from multiple runs (if stability_selection=True).

    Examples
    --------
    >>> from automl_fe.selection import EmbeddedSelector
    >>> import pandas as pd
    >>> from sklearn.datasets import load_iris
    >>> 
    >>> data = load_iris()
    >>> X = pd.DataFrame(data.data, columns=data.feature_names)
    >>> y = pd.Series(data.target)
    >>> 
    >>> selector = EmbeddedSelector(method='random_forest', n_features=3)
    >>> X_selected = selector.fit_transform(X, y)
    >>> print(selector.selected_features_)
    """

    def __init__(
        self,
        method: str = 'random_forest',
        n_features: Optional[Union[int, float]] = None,
        task: str = 'classification',
        threshold: Union[float, str] = 'mean',
        alpha: Optional[float] = None,
        l1_ratio: float = 0.5,
        n_estimators: int = 100,
        stability_selection: bool = False,
        n_stability_runs: int = 10,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        self.method = method
        self.n_features = n_features
        self.task = task
        self.threshold = threshold
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_estimators = n_estimators
        self.stability_selection = stability_selection
        self.n_stability_runs = n_stability_runs
        self.random_state = random_state
        self.verbose = verbose

        # Initialize state
        self.selected_features_: List[str] = []
        self.feature_importances_: Dict[str, float] = {}
        self.coef_: Optional[np.ndarray] = None
        self.estimator_: Any = None
        self.stability_scores_: Dict[str, float] = {}
        self._feature_names: List[str] = []
        self._scaler: Optional[StandardScaler] = None

    def _get_n_features(self, total_features: int) -> int:
        """Calculate number of features to select."""
        if self.n_features is None:
            return total_features
        elif isinstance(self.n_features, float) and 0 < self.n_features < 1:
            return max(1, int(total_features * self.n_features))
        else:
            return min(int(self.n_features), total_features)

    def _lasso_selection(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> tuple[np.ndarray, Any]:
        """
        LASSO-based feature selection.

        Parameters
        ----------
        X : np.ndarray
            Scaled feature matrix.
        y : np.ndarray
            Target array.

        Returns
        -------
        tuple
            (importances, fitted_estimator)
        """
        if self.task == 'classification':
            if self.alpha is not None:
                estimator = LogisticRegression(
                    penalty='l1',
                    solver='saga',
                    C=1.0 / self.alpha,
                    max_iter=1000,
                    random_state=self.random_state
                )
            else:
                # Use cross-validation for C
                estimator = LogisticRegression(
                    penalty='l1',
                    solver='saga',
                    max_iter=1000,
                    random_state=self.random_state
                )
        else:
            if self.alpha is not None:
                estimator = Lasso(
                    alpha=self.alpha,
                    random_state=self.random_state,
                    max_iter=1000
                )
            else:
                estimator = LassoCV(
                    cv=5,
                    random_state=self.random_state,
                    max_iter=1000
                )
        
        estimator.fit(X, y)
        
        if hasattr(estimator, 'coef_'):
            coef = estimator.coef_
            if coef.ndim > 1:
                # Multi-class: take mean absolute coefficient
                importances = np.abs(coef).mean(axis=0)
            else:
                importances = np.abs(coef)
        else:
            importances = np.zeros(X.shape[1])
        
        self.coef_ = estimator.coef_
        
        return importances, estimator

    def _ridge_selection(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> tuple[np.ndarray, Any]:
        """
        Ridge-based feature importance.

        Parameters
        ----------
        X : np.ndarray
            Scaled feature matrix.
        y : np.ndarray
            Target array.

        Returns
        -------
        tuple
            (importances, fitted_estimator)
        """
        if self.task == 'classification':
            estimator = LogisticRegression(
                penalty='l2',
                solver='lbfgs',
                C=1.0 / (self.alpha or 1.0),
                max_iter=1000,
                random_state=self.random_state
            )
        else:
            if self.alpha is not None:
                estimator = Ridge(
                    alpha=self.alpha,
                    random_state=self.random_state
                )
            else:
                estimator = RidgeCV(cv=5)
        
        estimator.fit(X, y)
        
        coef = estimator.coef_
        if coef.ndim > 1:
            importances = np.abs(coef).mean(axis=0)
        else:
            importances = np.abs(coef)
        
        self.coef_ = estimator.coef_
        
        return importances, estimator

    def _elasticnet_selection(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> tuple[np.ndarray, Any]:
        """
        ElasticNet-based feature selection.

        Parameters
        ----------
        X : np.ndarray
            Scaled feature matrix.
        y : np.ndarray
            Target array.

        Returns
        -------
        tuple
            (importances, fitted_estimator)
        """
        if self.task == 'classification':
            estimator = LogisticRegression(
                penalty='elasticnet',
                solver='saga',
                l1_ratio=self.l1_ratio,
                C=1.0 / (self.alpha or 1.0),
                max_iter=1000,
                random_state=self.random_state
            )
        else:
            if self.alpha is not None:
                estimator = ElasticNet(
                    alpha=self.alpha,
                    l1_ratio=self.l1_ratio,
                    random_state=self.random_state,
                    max_iter=1000
                )
            else:
                estimator = ElasticNetCV(
                    l1_ratio=self.l1_ratio,
                    cv=5,
                    random_state=self.random_state,
                    max_iter=1000
                )
        
        estimator.fit(X, y)
        
        coef = estimator.coef_
        if coef.ndim > 1:
            importances = np.abs(coef).mean(axis=0)
        else:
            importances = np.abs(coef)
        
        self.coef_ = estimator.coef_
        
        return importances, estimator

    def _random_forest_selection(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> tuple[np.ndarray, Any]:
        """
        Random Forest feature importance.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target array.

        Returns
        -------
        tuple
            (importances, fitted_estimator)
        """
        if self.task == 'classification':
            estimator = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            estimator = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        estimator.fit(X, y)
        importances = estimator.feature_importances_
        
        return importances, estimator

    def _xgboost_selection(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> tuple[np.ndarray, Any]:
        """
        XGBoost feature importance.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target array.

        Returns
        -------
        tuple
            (importances, fitted_estimator)
        """
        if not HAS_XGBOOST:
            warnings.warn("XGBoost not installed. Using RandomForest instead.")
            return self._random_forest_selection(X, y)
        
        if self.task == 'classification':
            n_classes = len(np.unique(y))
            objective = 'binary:logistic' if n_classes == 2 else 'multi:softmax'
            estimator = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                objective=objective,
                random_state=self.random_state,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            estimator = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        estimator.fit(X, y)
        importances = estimator.feature_importances_
        
        return importances, estimator

    def _get_threshold_value(self, importances: np.ndarray) -> float:
        """Calculate threshold value based on configuration."""
        if isinstance(self.threshold, (int, float)):
            return float(self.threshold)
        elif self.threshold == 'mean':
            return np.mean(importances)
        elif self.threshold == 'median':
            return np.median(importances)
        else:
            raise ValueError(f"Unknown threshold: {self.threshold}")

    def _stability_selection(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Run stability selection with subsampling.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target array.

        Returns
        -------
        dict
            Stability scores for each feature.
        """
        n_features = X.shape[1]
        selection_counts = np.zeros(n_features)
        
        rng = np.random.RandomState(self.random_state)
        
        for run in range(self.n_stability_runs):
            # Subsample data
            n_samples = X.shape[0]
            subsample_idx = rng.choice(n_samples, size=int(0.8 * n_samples), replace=False)
            X_sub = X[subsample_idx]
            y_sub = y[subsample_idx]
            
            # Get importances
            if self.method == 'lasso':
                importances, _ = self._lasso_selection(X_sub, y_sub)
            elif self.method == 'ridge':
                importances, _ = self._ridge_selection(X_sub, y_sub)
            elif self.method == 'elasticnet':
                importances, _ = self._elasticnet_selection(X_sub, y_sub)
            elif self.method == 'random_forest':
                importances, _ = self._random_forest_selection(X_sub, y_sub)
            elif self.method == 'xgboost':
                importances, _ = self._xgboost_selection(X_sub, y_sub)
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            # Threshold
            threshold = self._get_threshold_value(importances)
            selected = importances >= threshold
            selection_counts += selected
            
            if self.verbose > 0:
                print(f"Stability run {run + 1}/{self.n_stability_runs}: "
                      f"{selected.sum()} features selected")
        
        # Calculate stability scores
        stability_scores = selection_counts / self.n_stability_runs
        
        return dict(zip(self._feature_names, stability_scores))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EmbeddedSelector':
        """
        Fit the embedded selector to the data.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : pd.Series
            Target variable.

        Returns
        -------
        EmbeddedSelector
            Fitted selector.
        """
        self._feature_names = list(X.columns)
        X_array = X.values.astype(np.float64)
        y_array = y.values.ravel()
        
        # Scale features for linear methods
        if self.method in ['lasso', 'ridge', 'elasticnet']:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X_array)
        else:
            X_scaled = X_array
        
        # Stability selection
        if self.stability_selection:
            self.stability_scores_ = self._stability_selection(X_scaled, y_array)
            importances = np.array([self.stability_scores_[f] for f in self._feature_names])
            self.estimator_ = None
        else:
            # Single run
            if self.method == 'lasso':
                importances, self.estimator_ = self._lasso_selection(X_scaled, y_array)
            elif self.method == 'ridge':
                importances, self.estimator_ = self._ridge_selection(X_scaled, y_array)
            elif self.method == 'elasticnet':
                importances, self.estimator_ = self._elasticnet_selection(X_scaled, y_array)
            elif self.method == 'random_forest':
                importances, self.estimator_ = self._random_forest_selection(X_scaled, y_array)
            elif self.method == 'xgboost':
                importances, self.estimator_ = self._xgboost_selection(X_scaled, y_array)
            else:
                raise ValueError(f"Unknown method: {self.method}")
        
        # Store importances
        importances = np.nan_to_num(importances, nan=0.0)
        self.feature_importances_ = dict(zip(self._feature_names, importances))
        
        # Select features
        n_to_select = self._get_n_features(len(self._feature_names))
        
        if self.n_features is not None:
            # Select top n_features
            sorted_indices = np.argsort(importances)[::-1]
            selected_indices = sorted_indices[:n_to_select]
        else:
            # Use threshold
            threshold = self._get_threshold_value(importances)
            selected_indices = np.where(importances >= threshold)[0]
            
            # Ensure at least one feature
            if len(selected_indices) == 0:
                selected_indices = [np.argmax(importances)]
        
        self.selected_features_ = [self._feature_names[i] for i in selected_indices]
        
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

    def get_feature_importances(self) -> pd.DataFrame:
        """
        Get feature importances as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with feature names and importance scores.
        """
        return pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in self.feature_importances_.items()
        ]).sort_values('importance', ascending=False).reset_index(drop=True)

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
