# Core functionality for AutoML Feature Engineering
"""
Core module containing the main feature engineering pipeline.

This module provides the FeatureEngineering class which orchestrates
preprocessing, feature selection, and evaluation in an automated pipeline.
"""

import logging
import pickle
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .preprocessing import (
    smart_encode_categorical,
    adaptive_impute,
    generate_polynomial_features,
    scale_features,
    detect_and_handle_outliers,
    optimize_dtypes,
)
from .selection import FilterSelector, WrapperSelector, EmbeddedSelector, mRMRSelector


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Automated feature engineering pipeline with advanced feature selection.

    This class provides a comprehensive pipeline for feature engineering,
    including preprocessing, feature selection, and evaluation. It supports
    multiple feature selection methods and can automatically recommend
    the best approach based on data characteristics.

    Parameters
    ----------
    selection_method : str, default='filter'
        Feature selection method to use. Options: 'filter', 'wrapper',
        'embedded', 'mrmr', 'auto'.
    selection_params : dict, optional
        Parameters to pass to the selection method.
    n_features : int or float, optional
        Number of features to select. If float between 0 and 1, interpreted
        as a fraction of total features.
    task : str, default='auto'
        Task type: 'classification', 'regression', or 'auto' to detect.
    preprocessing : bool, default=True
        Whether to apply preprocessing transformations.
    scaling : str, default='standard'
        Scaling method: 'standard', 'minmax', 'robust', or None.
    handle_missing : bool, default=True
        Whether to handle missing values.
    handle_outliers : bool, default=False
        Whether to detect and handle outliers.
    polynomial_features : bool, default=False
        Whether to generate polynomial features.
    polynomial_degree : int, default=2
        Degree for polynomial feature generation.
    random_state : int, optional
        Random state for reproducibility.
    verbose : int, default=1
        Verbosity level (0=silent, 1=progress, 2=detailed).

    Attributes
    ----------
    fitted_ : bool
        Whether the pipeline has been fitted.
    feature_names_in_ : list
        Original feature names.
    feature_names_out_ : list
        Feature names after transformation.
    selected_features_ : list
        Names of selected features.
    feature_importances_ : dict
        Feature importance scores from selection.
    transformers_ : dict
        Fitted transformers for each step.
    selection_report_ : dict
        Detailed report of feature selection.

    Examples
    --------
    >>> from automl_fe import FeatureEngineering
    >>> import pandas as pd
    >>> from sklearn.datasets import load_iris
    >>> 
    >>> # Load data
    >>> data = load_iris()
    >>> X = pd.DataFrame(data.data, columns=data.feature_names)
    >>> y = pd.Series(data.target)
    >>> 
    >>> # Create and fit pipeline
    >>> fe = FeatureEngineering(selection_method='filter', n_features=3)
    >>> X_transformed = fe.fit_transform(X, y)
    >>> 
    >>> # Get selected features
    >>> print(fe.selected_features_)
    """

    def __init__(
        self,
        selection_method: str = 'filter',
        selection_params: Optional[Dict] = None,
        n_features: Optional[Union[int, float]] = None,
        task: str = 'auto',
        preprocessing: bool = True,
        scaling: str = 'standard',
        handle_missing: bool = True,
        handle_outliers: bool = False,
        polynomial_features: bool = False,
        polynomial_degree: int = 2,
        random_state: Optional[int] = None,
        verbose: int = 1,
    ):
        self.selection_method = selection_method
        self.selection_params = selection_params or {}
        self.n_features = n_features
        self.task = task
        self.preprocessing = preprocessing
        self.scaling = scaling
        self.handle_missing = handle_missing
        self.handle_outliers = handle_outliers
        self.polynomial_features = polynomial_features
        self.polynomial_degree = polynomial_degree
        self.random_state = random_state
        self.verbose = verbose

        # Initialize state
        self.fitted_ = False
        self.feature_names_in_: List[str] = []
        self.feature_names_out_: List[str] = []
        self.selected_features_: List[str] = []
        self.feature_importances_: Dict[str, float] = {}
        self.transformers_: Dict[str, Any] = {}
        self.selection_report_: Dict[str, Any] = {}
        self._task_type: str = 'classification'
        self._selector: Any = None

    def _log(self, message: str, level: int = 1):
        """Log a message based on verbosity level."""
        if self.verbose >= level:
            logger.info(message)

    def _detect_task(self, y: pd.Series) -> str:
        """
        Automatically detect task type based on target variable.

        Parameters
        ----------
        y : pd.Series
            Target variable.

        Returns
        -------
        str
            'classification' or 'regression'.
        """
        if y.dtype in ['object', 'category', 'bool']:
            return 'classification'
        
        n_unique = y.nunique()
        if n_unique <= 20 and n_unique / len(y) < 0.05:
            return 'classification'
        return 'regression'

    def _preprocess(self, X: pd.DataFrame, y: Optional[pd.Series] = None, fit: bool = True) -> pd.DataFrame:
        """
        Apply preprocessing transformations.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : pd.Series, optional
            Target variable (used for supervised imputation).
        fit : bool, default=True
            Whether to fit transformers or use existing ones.

        Returns
        -------
        pd.DataFrame
            Preprocessed features.
        """
        X_processed = X.copy()

        # Handle missing values
        if self.handle_missing:
            self._log("Handling missing values...", level=2)
            X_processed, imputer = adaptive_impute(
                X_processed,
                fitted_imputers=self.transformers_.get('imputer') if not fit else None
            )
            if fit:
                self.transformers_['imputer'] = imputer

        # Optimize data types
        self._log("Optimizing data types...", level=2)
        X_processed = optimize_dtypes(X_processed)

        # Handle outliers
        if self.handle_outliers:
            self._log("Handling outliers...", level=2)
            X_processed = detect_and_handle_outliers(X_processed)

        # Encode categorical variables
        self._log("Encoding categorical variables...", level=2)
        X_processed, encoders = smart_encode_categorical(
            X_processed,
            fitted_encoders=self.transformers_.get('encoders') if not fit else None
        )
        if fit:
            self.transformers_['encoders'] = encoders

        # Generate polynomial features
        if self.polynomial_features:
            self._log(f"Generating polynomial features (degree={self.polynomial_degree})...", level=2)
            X_processed, poly_transformer = generate_polynomial_features(
                X_processed,
                degree=self.polynomial_degree,
                fitted_transformer=self.transformers_.get('polynomial') if not fit else None
            )
            if fit:
                self.transformers_['polynomial'] = poly_transformer

        # Scale features
        if self.scaling:
            self._log(f"Scaling features ({self.scaling})...", level=2)
            X_processed, scaler = scale_features(
                X_processed,
                method=self.scaling,
                fitted_scaler=self.transformers_.get('scaler') if not fit else None
            )
            if fit:
                self.transformers_['scaler'] = scaler

        return X_processed

    def _get_selector(self) -> Any:
        """
        Get the appropriate feature selector based on configuration.

        Returns
        -------
        Selector object
            Configured feature selector.
        """
        n_features = self.n_features
        params = self.selection_params.copy()
        params['random_state'] = self.random_state

        if self.selection_method == 'filter':
            return FilterSelector(
                n_features=n_features,
                task=self._task_type,
                **params
            )
        elif self.selection_method == 'wrapper':
            return WrapperSelector(
                n_features=n_features,
                task=self._task_type,
                **params
            )
        elif self.selection_method == 'embedded':
            return EmbeddedSelector(
                n_features=n_features,
                task=self._task_type,
                **params
            )
        elif self.selection_method == 'mrmr':
            return mRMRSelector(
                n_features=n_features,
                **params
            )
        elif self.selection_method == 'auto':
            return self._auto_select_method()
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")

    def _auto_select_method(self) -> Any:
        """
        Automatically select the best feature selection method.

        Returns
        -------
        Selector object
            Recommended feature selector.
        """
        # Use mRMR for smaller datasets, embedded for larger
        return mRMRSelector(
            n_features=self.n_features,
            random_state=self.random_state
        )

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEngineering':
        """
        Fit the feature engineering pipeline.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : pd.Series, optional
            Target variable (required for supervised selection).

        Returns
        -------
        FeatureEngineering
            Fitted pipeline.
        """
        self._log("Starting feature engineering pipeline...")
        
        # Store original feature names
        self.feature_names_in_ = list(X.columns)
        
        # Detect task type
        if self.task == 'auto' and y is not None:
            self._task_type = self._detect_task(y)
            self._log(f"Detected task type: {self._task_type}")
        else:
            self._task_type = self.task if self.task != 'auto' else 'classification'

        # Apply preprocessing
        if self.preprocessing:
            self._log("Applying preprocessing...")
            X_processed = self._preprocess(X, y, fit=True)
        else:
            X_processed = X.copy()

        # Store preprocessed feature names
        self.feature_names_out_ = list(X_processed.columns)

        # Apply feature selection
        if y is not None:
            self._log(f"Applying {self.selection_method} feature selection...")
            self._selector = self._get_selector()
            self._selector.fit(X_processed, y)
            
            self.selected_features_ = self._selector.selected_features_
            self.feature_importances_ = self._selector.feature_importances_
            self.selection_report_ = {
                'method': self.selection_method,
                'n_features_in': len(self.feature_names_out_),
                'n_features_out': len(self.selected_features_),
                'selected_features': self.selected_features_,
                'feature_scores': self.feature_importances_,
            }
            
            self._log(f"Selected {len(self.selected_features_)} features")
        else:
            self.selected_features_ = self.feature_names_out_

        self.fitted_ = True
        self._log("Feature engineering pipeline fitted successfully!")
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using the fitted pipeline.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.

        Returns
        -------
        pd.DataFrame
            Transformed and selected features.

        Raises
        ------
        ValueError
            If the pipeline has not been fitted.
        """
        if not self.fitted_:
            raise ValueError("Pipeline has not been fitted. Call fit() first.")

        # Apply preprocessing
        if self.preprocessing:
            X_processed = self._preprocess(X, fit=False)
        else:
            X_processed = X.copy()

        # Apply feature selection
        if self._selector is not None:
            X_processed = self._selector.transform(X_processed)
        elif self.selected_features_:
            X_processed = X_processed[self.selected_features_]

        return X_processed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : pd.Series, optional
            Target variable.

        Returns
        -------
        pd.DataFrame
            Transformed and selected features.
        """
        self.fit(X, y)
        return self.transform(X)

    def get_feature_importances(self) -> pd.DataFrame:
        """
        Get feature importance scores as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with feature names and importance scores.
        """
        if not self.feature_importances_:
            raise ValueError("No feature importances available. Fit the pipeline first.")
        
        df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in self.feature_importances_.items()
        ])
        return df.sort_values('importance', ascending=False).reset_index(drop=True)

    def get_feature_names(self) -> List[str]:
        """
        Get names of selected features.

        Returns
        -------
        list
            List of selected feature names.
        """
        return self.selected_features_.copy()

    def feature_engineering_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive feature engineering report.

        Returns
        -------
        dict
            Dictionary containing transformation and selection details.
        """
        if not self.fitted_:
            raise ValueError("Pipeline has not been fitted.")
        
        report = {
            'task_type': self._task_type,
            'original_features': len(self.feature_names_in_),
            'preprocessed_features': len(self.feature_names_out_),
            'selected_features': len(self.selected_features_),
            'preprocessing_steps': list(self.transformers_.keys()),
            'selection_method': self.selection_method,
            'selection_details': self.selection_report_,
            'top_features': self.get_feature_importances().head(10).to_dict('records')
            if self.feature_importances_ else [],
        }
        return report

    def auto_select(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        methods: List[str] = None,
        cv: int = 5,
        scoring: str = None,
    ) -> Dict[str, Any]:
        """
        Try multiple selection methods and pick the best.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : pd.Series
            Target variable.
        methods : list, optional
            Methods to try. Default: ['filter', 'wrapper', 'embedded', 'mrmr'].
        cv : int, default=5
            Number of cross-validation folds.
        scoring : str, optional
            Scoring metric. Default based on task type.

        Returns
        -------
        dict
            Results including best method and scores.
        """
        if methods is None:
            methods = ['filter', 'embedded', 'mrmr']  # Skip wrapper for speed

        # Detect task type
        task = self._detect_task(y)
        if scoring is None:
            scoring = 'accuracy' if task == 'classification' else 'neg_mean_squared_error'

        results = {}
        best_score = float('-inf')
        best_method = None

        for method in methods:
            self._log(f"Trying {method} selection...")
            
            # Create temporary pipeline
            temp_fe = FeatureEngineering(
                selection_method=method,
                n_features=self.n_features,
                task=task,
                preprocessing=self.preprocessing,
                random_state=self.random_state,
                verbose=0,
            )
            
            try:
                X_selected = temp_fe.fit_transform(X, y)
                
                # Evaluate with simple model
                model = (RandomForestClassifier(n_estimators=50, random_state=self.random_state)
                        if task == 'classification'
                        else RandomForestRegressor(n_estimators=50, random_state=self.random_state))
                
                scores = cross_val_score(model, X_selected, y, cv=cv, scoring=scoring)
                mean_score = scores.mean()
                
                results[method] = {
                    'mean_score': mean_score,
                    'std_score': scores.std(),
                    'n_features': len(temp_fe.selected_features_),
                    'features': temp_fe.selected_features_,
                }
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_method = method
                    
            except Exception as e:
                self._log(f"Method {method} failed: {e}")
                results[method] = {'error': str(e)}

        # Fit with best method
        if best_method:
            self._log(f"Best method: {best_method} (score: {best_score:.4f})")
            self.selection_method = best_method
            self.fit(X, y)

        return {
            'best_method': best_method,
            'best_score': best_score,
            'all_results': results,
        }

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the fitted pipeline to disk.

        Parameters
        ----------
        path : str or Path
            Path to save the pipeline.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        self._log(f"Pipeline saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'FeatureEngineering':
        """
        Load a fitted pipeline from disk.

        Parameters
        ----------
        path : str or Path
            Path to the saved pipeline.

        Returns
        -------
        FeatureEngineering
            Loaded pipeline.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
