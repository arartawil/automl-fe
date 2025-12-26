# AutoML Feature Engineering Library
"""
automl_fe - Automated Machine Learning Feature Engineering
"""

from .core import FeatureEngineering
from .evaluation import SelectionComparator
from .visualization import SelectionVisualizer

__version__ = "0.1.0"
__all__ = ["FeatureEngineering", "SelectionComparator", "SelectionVisualizer"]
