# Wrapper-based feature selection methods
"""
Wrapper methods for feature selection using model performance.

This module provides the WrapperSelector class which implements various
wrapper-based feature selection techniques including RFE, forward selection,
backward elimination, and genetic algorithm-based selection.
"""

from typing import Dict, List, Optional, Union, Any, Callable
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge


class WrapperSelector(BaseEstimator, TransformerMixin):
    """
    Wrapper-based feature selection using model performance.

    This class implements wrapper methods that evaluate feature subsets
    by training a model and measuring performance. Wrapper methods are
    more computationally expensive but can capture feature interactions.

    Parameters
    ----------
    method : str, default='rfe'
        Selection method to use:
        - 'rfe': Recursive Feature Elimination
        - 'rfecv': RFE with cross-validation
        - 'forward': Forward stepwise selection
        - 'backward': Backward elimination
        - 'genetic': Genetic algorithm selection
    n_features : int or float, optional
        Number of features to select. If float between 0 and 1,
        interpreted as fraction of total features.
    estimator : estimator object, optional
        Model to use for evaluation. If None, uses RandomForest.
    task : str, default='classification'
        Task type: 'classification' or 'regression'.
    cv : int, default=5
        Number of cross-validation folds.
    scoring : str, optional
        Scoring metric for evaluation. Default based on task.
    n_jobs : int, default=1
        Number of parallel jobs.
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
    cv_scores_ : dict
        Cross-validation scores during selection.
    estimator_ : estimator
        Fitted estimator used for selection.

    Examples
    --------
    >>> from automl_fe.selection import WrapperSelector
    >>> import pandas as pd
    >>> from sklearn.datasets import load_iris
    >>> 
    >>> data = load_iris()
    >>> X = pd.DataFrame(data.data, columns=data.feature_names)
    >>> y = pd.Series(data.target)
    >>> 
    >>> selector = WrapperSelector(method='rfe', n_features=3)
    >>> X_selected = selector.fit_transform(X, y)
    >>> print(selector.selected_features_)
    """

    def __init__(
        self,
        method: str = 'rfe',
        n_features: Optional[Union[int, float]] = None,
        estimator: Optional[Any] = None,
        task: str = 'classification',
        cv: int = 5,
        scoring: Optional[str] = None,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        self.method = method
        self.n_features = n_features
        self.estimator = estimator
        self.task = task
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        # Initialize state
        self.selected_features_: List[str] = []
        self.feature_importances_: Dict[str, float] = {}
        self.cv_scores_: Dict[str, float] = {}
        self.estimator_: Any = None
        self._feature_names: List[str] = []

    def _get_n_features(self, total_features: int) -> int:
        """Calculate number of features to select."""
        if self.n_features is None:
            return max(1, total_features // 2)
        elif isinstance(self.n_features, float) and 0 < self.n_features < 1:
            return max(1, int(total_features * self.n_features))
        else:
            return min(int(self.n_features), total_features)

    def _get_estimator(self) -> Any:
        """Get the estimator to use for selection."""
        if self.estimator is not None:
            return clone(self.estimator)
        
        if self.task == 'classification':
            return RandomForestClassifier(
                n_estimators=50,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        else:
            return RandomForestRegressor(
                n_estimators=50,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )

    def _get_scoring(self) -> str:
        """Get the scoring metric."""
        if self.scoring is not None:
            return self.scoring
        return 'accuracy' if self.task == 'classification' else 'neg_mean_squared_error'

    def _get_cv(self, y: np.ndarray):
        """Get cross-validation splitter."""
        if self.task == 'classification':
            return StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        return KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

    def _rfe_selection(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        n_to_select: int
    ) -> List[int]:
        """
        Recursive Feature Elimination.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target array.
        n_to_select : int
            Number of features to select.

        Returns
        -------
        list
            Indices of selected features.
        """
        estimator = self._get_estimator()
        
        rfe = RFE(
            estimator=estimator,
            n_features_to_select=n_to_select,
            step=1,
            verbose=self.verbose
        )
        rfe.fit(X, y)
        
        self.estimator_ = rfe.estimator_
        
        # Store rankings as importances (inverse)
        rankings = rfe.ranking_
        max_rank = rankings.max()
        importances = (max_rank - rankings + 1) / max_rank
        
        for i, name in enumerate(self._feature_names):
            self.feature_importances_[name] = importances[i]
        
        return np.where(rfe.support_)[0].tolist()

    def _rfecv_selection(self, X: np.ndarray, y: np.ndarray) -> List[int]:
        """
        RFE with cross-validation for automatic n_features selection.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target array.

        Returns
        -------
        list
            Indices of selected features.
        """
        estimator = self._get_estimator()
        
        rfecv = RFECV(
            estimator=estimator,
            step=1,
            cv=self._get_cv(y),
            scoring=self._get_scoring(),
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )
        rfecv.fit(X, y)
        
        self.estimator_ = rfecv.estimator_
        self.cv_scores_['optimal_n_features'] = rfecv.n_features_
        
        # Store rankings as importances
        rankings = rfecv.ranking_
        max_rank = rankings.max()
        importances = (max_rank - rankings + 1) / max_rank
        
        for i, name in enumerate(self._feature_names):
            self.feature_importances_[name] = importances[i]
        
        return np.where(rfecv.support_)[0].tolist()

    def _forward_selection(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        n_to_select: int
    ) -> List[int]:
        """
        Forward stepwise selection.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target array.
        n_to_select : int
            Number of features to select.

        Returns
        -------
        list
            Indices of selected features.
        """
        n_features = X.shape[1]
        selected = []
        remaining = list(range(n_features))
        
        scoring = self._get_scoring()
        cv = self._get_cv(y)
        
        best_score = float('-inf')
        
        while len(selected) < n_to_select and remaining:
            best_feature = None
            best_feature_score = float('-inf')
            
            for feature in remaining:
                current_features = selected + [feature]
                X_subset = X[:, current_features]
                
                estimator = self._get_estimator()
                try:
                    scores = cross_val_score(
                        estimator, X_subset, y,
                        cv=cv, scoring=scoring, n_jobs=self.n_jobs
                    )
                    mean_score = scores.mean()
                except Exception:
                    mean_score = float('-inf')
                
                if mean_score > best_feature_score:
                    best_feature_score = mean_score
                    best_feature = feature
            
            if best_feature is not None:
                selected.append(best_feature)
                remaining.remove(best_feature)
                self.cv_scores_[self._feature_names[best_feature]] = best_feature_score
                
                if self.verbose > 0:
                    print(f"Selected feature {len(selected)}: "
                          f"{self._feature_names[best_feature]} "
                          f"(score: {best_feature_score:.4f})")
            else:
                break
        
        # Calculate importances based on selection order
        for i, idx in enumerate(selected):
            self.feature_importances_[self._feature_names[idx]] = 1.0 - (i / len(selected))
        
        for idx in remaining:
            self.feature_importances_[self._feature_names[idx]] = 0.0
        
        return selected

    def _backward_elimination(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        n_to_select: int
    ) -> List[int]:
        """
        Backward elimination.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target array.
        n_to_select : int
            Number of features to select.

        Returns
        -------
        list
            Indices of selected features.
        """
        n_features = X.shape[1]
        selected = list(range(n_features))
        
        scoring = self._get_scoring()
        cv = self._get_cv(y)
        elimination_order = []
        
        while len(selected) > n_to_select:
            worst_feature = None
            best_remaining_score = float('-inf')
            
            for feature in selected:
                # Try removing this feature
                current_features = [f for f in selected if f != feature]
                X_subset = X[:, current_features]
                
                estimator = self._get_estimator()
                try:
                    scores = cross_val_score(
                        estimator, X_subset, y,
                        cv=cv, scoring=scoring, n_jobs=self.n_jobs
                    )
                    mean_score = scores.mean()
                except Exception:
                    mean_score = float('-inf')
                
                if mean_score > best_remaining_score:
                    best_remaining_score = mean_score
                    worst_feature = feature
            
            if worst_feature is not None:
                selected.remove(worst_feature)
                elimination_order.append(worst_feature)
                
                if self.verbose > 0:
                    print(f"Eliminated: {self._feature_names[worst_feature]} "
                          f"(remaining score: {best_remaining_score:.4f})")
            else:
                break
        
        # Calculate importances (eliminated last = more important)
        for i, idx in enumerate(elimination_order):
            self.feature_importances_[self._feature_names[idx]] = i / n_features
        
        for idx in selected:
            self.feature_importances_[self._feature_names[idx]] = 1.0
        
        return selected

    def _genetic_selection(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        n_to_select: int,
        population_size: int = 50,
        n_generations: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
    ) -> List[int]:
        """
        Genetic algorithm feature selection.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target array.
        n_to_select : int
            Target number of features.
        population_size : int
            Size of population.
        n_generations : int
            Number of generations.
        mutation_rate : float
            Probability of mutation.
        crossover_rate : float
            Probability of crossover.

        Returns
        -------
        list
            Indices of selected features.
        """
        n_features = X.shape[1]
        rng = np.random.RandomState(self.random_state)
        
        scoring = self._get_scoring()
        cv = self._get_cv(y)
        
        def fitness(chromosome: np.ndarray) -> float:
            """Evaluate fitness of a chromosome."""
            selected_idx = np.where(chromosome)[0]
            if len(selected_idx) == 0:
                return float('-inf')
            
            X_subset = X[:, selected_idx]
            estimator = self._get_estimator()
            
            try:
                scores = cross_val_score(
                    estimator, X_subset, y,
                    cv=cv, scoring=scoring, n_jobs=self.n_jobs
                )
                score = scores.mean()
                
                # Penalize for deviating from target n_features
                n_selected = len(selected_idx)
                penalty = abs(n_selected - n_to_select) * 0.01
                
                return score - penalty
            except Exception:
                return float('-inf')
        
        # Initialize population
        population = []
        for _ in range(population_size):
            chromosome = np.zeros(n_features, dtype=bool)
            selected_indices = rng.choice(n_features, size=n_to_select, replace=False)
            chromosome[selected_indices] = True
            population.append(chromosome)
        
        best_chromosome = None
        best_fitness = float('-inf')
        
        for gen in range(n_generations):
            # Evaluate fitness
            fitness_scores = [fitness(chrom) for chrom in population]
            
            # Track best
            gen_best_idx = np.argmax(fitness_scores)
            if fitness_scores[gen_best_idx] > best_fitness:
                best_fitness = fitness_scores[gen_best_idx]
                best_chromosome = population[gen_best_idx].copy()
            
            if self.verbose > 0:
                print(f"Generation {gen + 1}: best fitness = {best_fitness:.4f}")
            
            # Selection (tournament)
            new_population = []
            for _ in range(population_size):
                # Tournament selection
                tournament_idx = rng.choice(population_size, size=3, replace=False)
                tournament_fitness = [fitness_scores[i] for i in tournament_idx]
                winner_idx = tournament_idx[np.argmax(tournament_fitness)]
                new_population.append(population[winner_idx].copy())
            
            # Crossover
            for i in range(0, population_size - 1, 2):
                if rng.random() < crossover_rate:
                    # Single-point crossover
                    point = rng.randint(1, n_features)
                    new_population[i][:point], new_population[i + 1][:point] = \
                        new_population[i + 1][:point].copy(), new_population[i][:point].copy()
            
            # Mutation
            for i in range(population_size):
                for j in range(n_features):
                    if rng.random() < mutation_rate:
                        new_population[i][j] = not new_population[i][j]
            
            population = new_population
        
        # Use best chromosome
        if best_chromosome is None:
            best_chromosome = population[np.argmax([fitness(c) for c in population])]
        
        selected = np.where(best_chromosome)[0].tolist()
        
        # Store importances
        for i, name in enumerate(self._feature_names):
            self.feature_importances_[name] = 1.0 if i in selected else 0.0
        
        return selected

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'WrapperSelector':
        """
        Fit the wrapper selector to the data.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : pd.Series
            Target variable.

        Returns
        -------
        WrapperSelector
            Fitted selector.
        """
        self._feature_names = list(X.columns)
        X_array = X.values.astype(np.float64)
        y_array = y.values.ravel()
        
        n_to_select = self._get_n_features(len(self._feature_names))
        
        if self.method == 'rfe':
            selected_indices = self._rfe_selection(X_array, y_array, n_to_select)
        elif self.method == 'rfecv':
            selected_indices = self._rfecv_selection(X_array, y_array)
        elif self.method == 'forward':
            selected_indices = self._forward_selection(X_array, y_array, n_to_select)
        elif self.method == 'backward':
            selected_indices = self._backward_elimination(X_array, y_array, n_to_select)
        elif self.method == 'genetic':
            selected_indices = self._genetic_selection(X_array, y_array, n_to_select)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
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
