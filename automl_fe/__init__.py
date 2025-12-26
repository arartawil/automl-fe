# AutoML Feature Engineering Library
"""
automl_fe - Automated Machine Learning Feature Engineering
"""

from .core import FeatureEngineering
from .evaluation import SelectionComparator
from .visualization import SelectionVisualizer
from .quality import DataQualityChecker, DataQualityReport, generate_quality_report_summary
from .pipeline_manager import FeaturePipelineManager, PipelineExporter, quick_save_pipeline, quick_load_pipeline
from .categorical import (
    TargetEncoder, 
    FrequencyEncoder, 
    BinaryEncoder, 
    ComprehensiveCategoricalEncoder
)

__version__ = "1.0.1"
__all__ = [
    "FeatureEngineering", 
    "SelectionComparator", 
    "SelectionVisualizer",
    "DataQualityChecker",
    "DataQualityReport", 
    "generate_quality_report_summary",
    "FeaturePipelineManager",
    "PipelineExporter",
    "quick_save_pipeline",
    "quick_load_pipeline",
    "TargetEncoder",
    "FrequencyEncoder", 
    "BinaryEncoder",
    "ComprehensiveCategoricalEncoder"
]
