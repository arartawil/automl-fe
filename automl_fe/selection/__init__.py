# Feature Selection Module
"""
Feature selection methods including filter, wrapper, embedded, and mRMR approaches.

This module provides various feature selection algorithms:
- FilterSelector: Statistical filter methods (mutual info, chi2, f-statistic, etc.)
- WrapperSelector: Model-based methods (RFE, forward/backward selection, genetic)
- EmbeddedSelector: Regularization and tree-based methods (LASSO, RF, XGBoost)
- mRMRSelector: Minimum Redundancy Maximum Relevance algorithm
"""

from .filter import FilterSelector
from .wrapper import WrapperSelector
from .embedded import EmbeddedSelector
from .mrmr import mRMRSelector

__all__ = [
    'FilterSelector',
    'WrapperSelector',
    'EmbeddedSelector',
    'mRMRSelector',
]
