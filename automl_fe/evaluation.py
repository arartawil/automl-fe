# Evaluation metrics for AutoML Feature Engineering
"""
Evaluation module for feature quality assessment.

This module provides the SelectionComparator class for comparing multiple
feature selection methods using cross-validation and statistical testing.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error,
)
from scipy import stats

from .selection import FilterSelector, WrapperSelector, EmbeddedSelector, mRMRSelector


class SelectionComparator:
    """
    Compare multiple feature selection methods.

    This class evaluates and compares different feature selection methods
    using cross-validation, statistical significance testing, and stability
    analysis.

    Parameters
    ----------
    methods : list of str or dict, optional
        Selection methods to compare. Can be:
        - List of method names: ['filter', 'mrmr', 'embedded']
        - List of dicts with method and params: [{'method': 'filter', 'params': {...}}]
        If None, uses all available methods.
    n_features : int or float, optional
        Number of features to select for each method.
    task : str, default='auto'
        Task type: 'classification', 'regression', or 'auto'.
    cv : int, default=5
        Number of cross-validation folds.
    scoring : list of str, optional
        Scoring metrics to evaluate. Default based on task.
    estimator : estimator object, optional
        Model to use for evaluation. Default: RandomForest.
    n_jobs : int, default=1
        Number of parallel jobs.
    random_state : int, optional
        Random state for reproducibility.
    verbose : int, default=1
        Verbosity level.

    Attributes
    ----------
    results_ : pd.DataFrame
        Detailed comparison results.
    best_method_ : str
        Name of best performing method.
    feature_overlap_ : pd.DataFrame
        Pairwise feature overlap between methods.
    stability_scores_ : dict
        Selection stability across CV folds.
    timing_ : dict
        Execution time for each method.

    Examples
    --------
    >>> from automl_fe.evaluation import SelectionComparator
    >>> import pandas as pd
    >>> from sklearn.datasets import load_iris
    >>> 
    >>> data = load_iris()
    >>> X = pd.DataFrame(data.data, columns=data.feature_names)
    >>> y = pd.Series(data.target)
    >>> 
    >>> comparator = SelectionComparator(n_features=3)
    >>> results = comparator.compare(X, y)
    >>> print(comparator.best_method_)
    >>> print(comparator.results_)
    """

    def __init__(
        self,
        methods: Optional[List[Union[str, Dict]]] = None,
        n_features: Optional[Union[int, float]] = None,
        task: str = 'auto',
        cv: int = 5,
        scoring: Optional[List[str]] = None,
        estimator: Optional[Any] = None,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        verbose: int = 1,
    ):
        self.methods = methods
        self.n_features = n_features
        self.task = task
        self.cv = cv
        self.scoring = scoring
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        # Initialize state
        self.results_: Optional[pd.DataFrame] = None
        self.best_method_: Optional[str] = None
        self.feature_overlap_: Optional[pd.DataFrame] = None
        self.stability_scores_: Dict[str, float] = {}
        self.timing_: Dict[str, float] = {}
        self._selected_features: Dict[str, List[str]] = {}
        self._task_type: str = 'classification'

    def _detect_task(self, y: pd.Series) -> str:
        """Detect task type from target."""
        if y.dtype in ['object', 'category', 'bool']:
            return 'classification'
        n_unique = y.nunique()
        if n_unique <= 20 and n_unique / len(y) < 0.05:
            return 'classification'
        return 'regression'

    def _get_methods(self) -> List[Dict]:
        """Get list of methods with parameters."""
        if self.methods is None:
            return [
                {'name': 'filter_mi', 'method': 'filter', 'params': {'method': 'mutual_info'}},
                {'name': 'filter_f', 'method': 'filter', 'params': {'method': 'f_statistic'}},
                {'name': 'mrmr', 'method': 'mrmr', 'params': {}},
                {'name': 'embedded_rf', 'method': 'embedded', 'params': {'method': 'random_forest'}},
                {'name': 'embedded_lasso', 'method': 'embedded', 'params': {'method': 'lasso'}},
            ]
        
        methods_list = []
        for m in self.methods:
            if isinstance(m, str):
                methods_list.append({'name': m, 'method': m, 'params': {}})
            else:
                name = m.get('name', m.get('method', 'unknown'))
                methods_list.append({
                    'name': name,
                    'method': m.get('method', m.get('name')),
                    'params': m.get('params', {})
                })
        
        return methods_list

    def _get_selector(self, method_info: Dict) -> Any:
        """Create selector instance from method info."""
        method = method_info['method']
        params = method_info['params'].copy()
        params['n_features'] = self.n_features
        params['random_state'] = self.random_state

        if method == 'filter':
            return FilterSelector(task=self._task_type, **params)
        elif method == 'wrapper':
            return WrapperSelector(task=self._task_type, **params)
        elif method == 'embedded':
            return EmbeddedSelector(task=self._task_type, **params)
        elif method == 'mrmr':
            return mRMRSelector(**params)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _get_estimator(self) -> Any:
        """Get estimator for evaluation."""
        if self.estimator is not None:
            return clone(self.estimator)
        
        if self._task_type == 'classification':
            return RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        else:
            return RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )

    def _get_scoring_metrics(self) -> List[str]:
        """Get scoring metrics based on task."""
        if self.scoring is not None:
            return self.scoring
        
        if self._task_type == 'classification':
            return ['accuracy', 'f1_weighted', 'roc_auc_ovr']
        else:
            return ['neg_mean_squared_error', 'r2', 'neg_mean_absolute_error']

    def _get_cv_splitter(self, y: np.ndarray):
        """Get cross-validation splitter."""
        if self._task_type == 'classification':
            return StratifiedKFold(
                n_splits=self.cv, 
                shuffle=True, 
                random_state=self.random_state
            )
        return KFold(
            n_splits=self.cv, 
            shuffle=True, 
            random_state=self.random_state
        )

    def _evaluate_method(
        self,
        method_info: Dict,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, Any]:
        """
        Evaluate a single selection method.

        Parameters
        ----------
        method_info : dict
            Method configuration.
        X : pd.DataFrame
            Input features.
        y : pd.Series
            Target variable.

        Returns
        -------
        dict
            Evaluation results.
        """
        method_name = method_info['name']
        
        if self.verbose > 0:
            print(f"Evaluating {method_name}...")
        
        start_time = time.time()
        
        try:
            # Create and fit selector
            selector = self._get_selector(method_info)
            X_selected = selector.fit_transform(X, y)
            
            elapsed_time = time.time() - start_time
            self.timing_[method_name] = elapsed_time
            
            # Store selected features
            self._selected_features[method_name] = selector.selected_features_
            
            # Evaluate with cross-validation
            estimator = self._get_estimator()
            cv = self._get_cv_splitter(y.values)
            
            results = {
                'method': method_name,
                'n_features': len(selector.selected_features_),
                'time': elapsed_time,
            }
            
            # Calculate scores for each metric
            for metric in self._get_scoring_metrics():
                try:
                    scores = cross_val_score(
                        estimator, X_selected, y,
                        cv=cv, scoring=metric, n_jobs=self.n_jobs
                    )
                    results[f'{metric}_mean'] = scores.mean()
                    results[f'{metric}_std'] = scores.std()
                    results[f'{metric}_scores'] = scores.tolist()
                except Exception as e:
                    if self.verbose > 0:
                        print(f"  Warning: {metric} failed: {e}")
                    results[f'{metric}_mean'] = np.nan
                    results[f'{metric}_std'] = np.nan
            
            # Calculate stability (feature overlap across CV folds)
            stability = self._calculate_stability(selector, X, y)
            results['stability'] = stability
            self.stability_scores_[method_name] = stability
            
            return results
            
        except Exception as e:
            if self.verbose > 0:
                print(f"  Error: {e}")
            return {
                'method': method_name,
                'error': str(e),
            }

    def _calculate_stability(
        self,
        selector: Any,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> float:
        """
        Calculate selection stability across CV folds.

        Uses Jaccard similarity between feature sets selected in each fold.

        Parameters
        ----------
        selector : Selector
            Feature selector.
        X : pd.DataFrame
            Input features.
        y : pd.Series
            Target variable.

        Returns
        -------
        float
            Stability score (0 to 1, higher is more stable).
        """
        cv = self._get_cv_splitter(y.values)
        selected_sets = []
        
        for train_idx, _ in cv.split(X, y):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            
            try:
                fold_selector = clone(selector)
                fold_selector.fit(X_train, y_train)
                selected_sets.append(set(fold_selector.selected_features_))
            except Exception:
                continue
        
        if len(selected_sets) < 2:
            return 0.0
        
        # Calculate average Jaccard similarity
        similarities = []
        for i in range(len(selected_sets)):
            for j in range(i + 1, len(selected_sets)):
                intersection = len(selected_sets[i] & selected_sets[j])
                union = len(selected_sets[i] | selected_sets[j])
                if union > 0:
                    similarities.append(intersection / union)
        
        return np.mean(similarities) if similarities else 0.0

    def _calculate_feature_overlap(self) -> pd.DataFrame:
        """
        Calculate pairwise feature overlap between methods.

        Returns
        -------
        pd.DataFrame
            Matrix of Jaccard similarities.
        """
        methods = list(self._selected_features.keys())
        n_methods = len(methods)
        
        overlap = np.zeros((n_methods, n_methods))
        
        for i in range(n_methods):
            for j in range(n_methods):
                set_i = set(self._selected_features[methods[i]])
                set_j = set(self._selected_features[methods[j]])
                
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                
                overlap[i, j] = intersection / union if union > 0 else 0
        
        return pd.DataFrame(overlap, index=methods, columns=methods)

    def _statistical_comparison(self) -> pd.DataFrame:
        """
        Perform statistical significance tests between methods.

        Uses paired t-test on CV scores.

        Returns
        -------
        pd.DataFrame
            P-values for pairwise comparisons.
        """
        if self.results_ is None:
            return pd.DataFrame()
        
        primary_metric = self._get_scoring_metrics()[0]
        scores_col = f'{primary_metric}_scores'
        
        methods = self.results_['method'].tolist()
        n_methods = len(methods)
        
        pvalues = np.ones((n_methods, n_methods))
        
        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                try:
                    scores_i = self.results_.iloc[i][scores_col]
                    scores_j = self.results_.iloc[j][scores_col]
                    
                    if isinstance(scores_i, list) and isinstance(scores_j, list):
                        _, pvalue = stats.ttest_rel(scores_i, scores_j)
                        pvalues[i, j] = pvalue
                        pvalues[j, i] = pvalue
                except Exception:
                    continue
        
        return pd.DataFrame(pvalues, index=methods, columns=methods)

    def compare(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        """
        Compare all feature selection methods.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : pd.Series
            Target variable.

        Returns
        -------
        pd.DataFrame
            Comparison results.
        """
        # Detect task type
        if self.task == 'auto':
            self._task_type = self._detect_task(y)
        else:
            self._task_type = self.task
        
        if self.verbose > 0:
            print(f"Task type: {self._task_type}")
            print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
            print("-" * 50)
        
        # Get methods to compare
        methods = self._get_methods()
        
        # Evaluate each method
        results = []
        for method_info in methods:
            result = self._evaluate_method(method_info, X, y)
            results.append(result)
        
        # Create results DataFrame
        self.results_ = pd.DataFrame(results)
        
        # Determine best method
        primary_metric = self._get_scoring_metrics()[0]
        score_col = f'{primary_metric}_mean'
        
        if score_col in self.results_.columns:
            valid_results = self.results_[self.results_[score_col].notna()]
            if len(valid_results) > 0:
                best_idx = valid_results[score_col].idxmax()
                self.best_method_ = valid_results.loc[best_idx, 'method']
        
        # Calculate feature overlap
        if self._selected_features:
            self.feature_overlap_ = self._calculate_feature_overlap()
        
        if self.verbose > 0:
            print("-" * 50)
            print(f"Best method: {self.best_method_}")
        
        return self.results_

    def get_comparison_summary(self) -> pd.DataFrame:
        """
        Get a summary of comparison results.

        Returns
        -------
        pd.DataFrame
            Summary with key metrics for each method.
        """
        if self.results_ is None:
            raise ValueError("No comparison results. Call compare() first.")
        
        summary_cols = ['method', 'n_features', 'time', 'stability']
        
        # Add mean scores
        for metric in self._get_scoring_metrics():
            col = f'{metric}_mean'
            if col in self.results_.columns:
                summary_cols.append(col)
        
        summary = self.results_[
            [c for c in summary_cols if c in self.results_.columns]
        ].copy()
        
        return summary.sort_values(
            f'{self._get_scoring_metrics()[0]}_mean',
            ascending=False
        )

    def get_feature_overlap(self) -> pd.DataFrame:
        """
        Get feature overlap matrix.

        Returns
        -------
        pd.DataFrame
            Pairwise Jaccard similarity of selected features.
        """
        if self.feature_overlap_ is None:
            raise ValueError("No overlap data. Call compare() first.")
        return self.feature_overlap_.copy()

    def get_common_features(self, min_methods: int = 2) -> List[str]:
        """
        Get features selected by multiple methods.

        Parameters
        ----------
        min_methods : int, default=2
            Minimum number of methods that must select a feature.

        Returns
        -------
        list
            Features selected by at least min_methods methods.
        """
        if not self._selected_features:
            raise ValueError("No selection data. Call compare() first.")
        
        feature_counts = defaultdict(int)
        for features in self._selected_features.values():
            for f in features:
                feature_counts[f] += 1
        
        return [f for f, count in feature_counts.items() if count >= min_methods]

    def statistical_tests(self) -> pd.DataFrame:
        """
        Get statistical significance tests between methods.

        Returns
        -------
        pd.DataFrame
            P-values for pairwise comparisons.
        """
        return self._statistical_comparison()

    def get_best_features(self) -> List[str]:
        """
        Get features selected by the best method.

        Returns
        -------
        list
            Selected features from best method.
        """
        if self.best_method_ is None:
            raise ValueError("No best method. Call compare() first.")
        return self._selected_features.get(self.best_method_, [])
