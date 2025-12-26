# Pipeline Serialization and Export Module
"""
Pipeline serialization functionality for saving and loading trained
feature engineering pipelines.

This module enables saving trained feature selection pipelines to disk
and loading them for application to new datasets, ensuring reproducibility
and deployment capabilities.
"""

import json
import pickle
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PipelineExporter:
    """
    Export and import functionality for feature engineering pipelines.
    
    Supports multiple serialization formats including JSON for configuration
    and pickle/joblib for trained models.
    
    Parameters
    ----------
    format : str, default='joblib'
        Serialization format: 'joblib', 'pickle', or 'json'.
    compress : bool, default=True
        Whether to compress serialized files.
    """
    
    def __init__(self, format: str = 'joblib', compress: bool = True):
        self.format = format
        self.compress = compress
        self.supported_formats = ['joblib', 'pickle', 'json']
        
        if format not in self.supported_formats:
            raise ValueError(f"Format must be one of {self.supported_formats}")
    
    def export_pipeline(
        self,
        pipeline_obj: Any,
        filepath: Union[str, Path],
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Export a trained feature engineering pipeline.
        
        Parameters
        ----------
        pipeline_obj : object
            The trained pipeline object to export.
        filepath : str or Path
            Path where to save the pipeline.
        metadata : dict, optional
            Additional metadata to save with the pipeline.
            
        Returns
        -------
        bool
            True if export successful, False otherwise.
        """
        try:
            filepath = Path(filepath)
            
            # Prepare export package
            export_package = {
                'pipeline': pipeline_obj,
                'metadata': metadata or {},
                'format_version': '1.0.1',
                'export_timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Add pipeline-specific metadata
            if hasattr(pipeline_obj, '__dict__'):
                export_package['pipeline_config'] = {
                    k: v for k, v in pipeline_obj.__dict__.items()
                    if isinstance(v, (str, int, float, bool, list, dict, type(None)))
                }
            
            # Export based on format
            if self.format == 'joblib':
                compression = 'gzip' if self.compress else None
                joblib.dump(export_package, filepath.with_suffix('.joblib'), compress=compression)
                
            elif self.format == 'pickle':
                mode = 'wb'
                with open(filepath.with_suffix('.pkl'), mode) as f:
                    pickle.dump(export_package, f)
                    
            elif self.format == 'json':
                # For JSON, we can only save configuration, not trained objects
                json_data = {
                    'metadata': export_package['metadata'],
                    'pipeline_config': export_package.get('pipeline_config', {}),
                    'format_version': export_package['format_version'],
                    'export_timestamp': export_package['export_timestamp']
                }
                
                with open(filepath.with_suffix('.json'), 'w') as f:
                    json.dump(json_data, f, indent=2, default=str)
                    
            return True
            
        except Exception as e:
            print(f"Export failed: {str(e)}")
            return False
    
    def import_pipeline(self, filepath: Union[str, Path]) -> Optional[Dict]:
        """
        Import a saved feature engineering pipeline.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the saved pipeline.
            
        Returns
        -------
        dict or None
            Loaded pipeline package or None if loading failed.
        """
        try:
            filepath = Path(filepath)
            
            # Determine format from file extension
            if filepath.suffix == '.joblib':
                return joblib.load(filepath)
            elif filepath.suffix == '.pkl':
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            elif filepath.suffix == '.json':
                with open(filepath, 'r') as f:
                    return json.load(f)
            else:
                # Try to detect format
                if self.format == 'joblib':
                    return joblib.load(filepath)
                elif self.format == 'pickle':
                    with open(filepath, 'rb') as f:
                        return pickle.load(f)
                        
        except Exception as e:
            print(f"Import failed: {str(e)}")
            return None


class FeaturePipelineManager:
    """
    Manager for feature engineering pipeline lifecycle.
    
    Provides higher-level interface for saving, loading, and managing
    multiple feature engineering pipelines with versioning and metadata.
    """
    
    def __init__(self, base_dir: Union[str, Path] = "feature_pipelines"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.exporter = PipelineExporter()
        
    def save_pipeline(
        self,
        pipeline_obj: Any,
        name: str,
        version: str = "1.0",
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Save a feature engineering pipeline with metadata.
        
        Parameters
        ----------
        pipeline_obj : object
            The trained pipeline to save.
        name : str
            Pipeline name.
        version : str, default="1.0"
            Pipeline version.
        description : str, default=""
            Pipeline description.
        tags : list of str, optional
            Tags for categorizing the pipeline.
            
        Returns
        -------
        bool
            True if save successful.
        """
        # Create pipeline directory
        pipeline_dir = self.base_dir / name / version
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare metadata
        metadata = {
            'name': name,
            'version': version,
            'description': description,
            'tags': tags or [],
            'created_at': pd.Timestamp.now().isoformat(),
            'feature_count': getattr(pipeline_obj, 'n_features_', 'unknown'),
            'selection_method': getattr(pipeline_obj, 'selection_method', 'unknown')
        }
        
        # Save pipeline
        filepath = pipeline_dir / f"{name}_v{version}"
        success = self.exporter.export_pipeline(pipeline_obj, filepath, metadata)
        
        if success:
            # Update registry
            self._update_registry(name, version, metadata)
            
        return success
    
    def load_pipeline(self, name: str, version: str = "latest") -> Optional[Any]:
        """
        Load a saved feature engineering pipeline.
        
        Parameters
        ----------
        name : str
            Pipeline name.
        version : str, default="latest"
            Pipeline version to load.
            
        Returns
        -------
        object or None
            Loaded pipeline object.
        """
        if version == "latest":
            version = self._get_latest_version(name)
            
        if version is None:
            return None
            
        pipeline_dir = self.base_dir / name / version
        filepath = pipeline_dir / f"{name}_v{version}.joblib"
        
        if not filepath.exists():
            print(f"Pipeline not found: {filepath}")
            return None
            
        package = self.exporter.import_pipeline(filepath)
        return package['pipeline'] if package else None
    
    def list_pipelines(self) -> pd.DataFrame:
        """
        List all available pipelines with metadata.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing pipeline information.
        """
        registry_file = self.base_dir / "registry.json"
        
        if not registry_file.exists():
            return pd.DataFrame(columns=['name', 'version', 'created_at', 'description'])
            
        with open(registry_file, 'r') as f:
            registry = json.load(f)
            
        return pd.DataFrame(registry)
    
    def delete_pipeline(self, name: str, version: str = "all") -> bool:
        """
        Delete a pipeline or all versions of a pipeline.
        
        Parameters
        ----------
        name : str
            Pipeline name.
        version : str, default="all"
            Version to delete or "all" for all versions.
            
        Returns
        -------
        bool
            True if deletion successful.
        """
        try:
            if version == "all":
                pipeline_dir = self.base_dir / name
                if pipeline_dir.exists():
                    import shutil
                    shutil.rmtree(pipeline_dir)
            else:
                version_dir = self.base_dir / name / version
                if version_dir.exists():
                    import shutil
                    shutil.rmtree(version_dir)
                    
            # Update registry
            self._remove_from_registry(name, version)
            return True
            
        except Exception as e:
            print(f"Deletion failed: {str(e)}")
            return False
    
    def _get_latest_version(self, name: str) -> Optional[str]:
        """Get the latest version of a pipeline."""
        pipeline_dir = self.base_dir / name
        if not pipeline_dir.exists():
            return None
            
        versions = [d.name for d in pipeline_dir.iterdir() if d.is_dir()]
        if not versions:
            return None
            
        # Simple version sorting (assumes semantic versioning)
        versions.sort(key=lambda x: [int(i) for i in x.split('.')])
        return versions[-1]
    
    def _update_registry(self, name: str, version: str, metadata: Dict):
        """Update the pipeline registry."""
        registry_file = self.base_dir / "registry.json"
        
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                registry = json.load(f)
        else:
            registry = []
            
        # Remove existing entry if it exists
        registry = [r for r in registry if not (r['name'] == name and r['version'] == version)]
        
        # Add new entry
        registry.append(metadata)
        
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=2, default=str)
    
    def _remove_from_registry(self, name: str, version: str):
        """Remove pipeline from registry."""
        registry_file = self.base_dir / "registry.json"
        
        if not registry_file.exists():
            return
            
        with open(registry_file, 'r') as f:
            registry = json.load(f)
            
        if version == "all":
            registry = [r for r in registry if r['name'] != name]
        else:
            registry = [r for r in registry if not (r['name'] == name and r['version'] == version)]
            
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=2, default=str)


# Utility functions for pipeline serialization
def quick_save_pipeline(pipeline_obj: Any, filepath: Union[str, Path]) -> bool:
    """Quick save function for feature engineering pipelines."""
    exporter = PipelineExporter()
    return exporter.export_pipeline(pipeline_obj, filepath)


def quick_load_pipeline(filepath: Union[str, Path]) -> Optional[Any]:
    """Quick load function for feature engineering pipelines."""
    exporter = PipelineExporter()
    package = exporter.import_pipeline(filepath)
    return package['pipeline'] if package else None