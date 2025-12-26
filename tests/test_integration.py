"""
Integration tests for automl-fe.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from automl_fe import FeatureEngineering, SelectionComparator, SelectionVisualizer
from automl_fe.selection import FilterSelector, mRMRSelector, EmbeddedSelector


@pytest.fixture
def full_pipeline_data():
    """Full dataset for integration tests."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""
    
    def test_complete_pipeline(self, full_pipeline_data):
        """Test complete feature engineering pipeline."""
        X_train, X_test, y_train, y_test = full_pipeline_data
        
        # Create pipeline
        fe = FeatureEngineering(
            selection_method='mrmr',
            n_features=10,
            preprocessing=True,
            scaling='standard',
            random_state=42
        )
        
        # Fit and transform
        X_train_selected = fe.fit_transform(X_train, y_train)
        X_test_selected = fe.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train_selected, y_train)
        
        # Evaluate
        accuracy = model.score(X_test_selected, y_test)
        
        assert X_train_selected.shape[1] == 10
        assert X_test_selected.shape[1] == 10
        assert accuracy > 0.8  # Should be reasonable accuracy
    
    def test_method_comparison_workflow(self, full_pipeline_data):
        """Test complete method comparison workflow."""
        X_train, X_test, y_train, y_test = full_pipeline_data
        
        # Compare methods
        comparator = SelectionComparator(
            methods=['filter', 'mrmr', 'embedded'],
            n_features=10,
            cv=3,
            random_state=42,
            verbose=0
        )
        
        results = comparator.compare(X_train, y_train)
        
        # Check results
        assert len(results) == 3
        assert comparator.best_method_ is not None
        assert len(comparator.get_common_features(min_methods=2)) >= 0
    
    def test_auto_select_workflow(self, full_pipeline_data):
        """Test auto-select workflow."""
        X_train, X_test, y_train, y_test = full_pipeline_data
        
        fe = FeatureEngineering(n_features=10, random_state=42)
        results = fe.auto_select(
            X_train, y_train, 
            methods=['filter', 'mrmr'],
            cv=3
        )
        
        assert results['best_method'] is not None
        assert results['best_score'] > 0
        
        # Should be fitted with best method
        X_train_selected = fe.transform(X_train)
        assert X_train_selected.shape[1] == 10


class TestPipelineIntegration:
    """Tests for sklearn pipeline integration."""
    
    def test_with_sklearn_estimators(self, full_pipeline_data):
        """Test that selected features work with sklearn estimators."""
        X_train, X_test, y_train, y_test = full_pipeline_data
        
        # Feature selection
        selector = FilterSelector(n_features=10, random_state=42)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Multiple estimators
        estimators = [
            RandomForestClassifier(n_estimators=50, random_state=42),
        ]
        
        for est in estimators:
            est.fit(X_train_selected, y_train)
            accuracy = est.score(X_test_selected, y_test)
            assert accuracy > 0.7


class TestCrossValidationIntegration:
    """Tests for cross-validation integration."""
    
    def test_cv_with_selection(self, full_pipeline_data):
        """Test cross-validation with feature selection."""
        X_train, X_test, y_train, y_test = full_pipeline_data
        
        # Select features
        selector = mRMRSelector(n_features=10, random_state=42)
        X_selected = selector.fit_transform(X_train, y_train)
        
        # Cross-validation
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        scores = cross_val_score(model, X_selected, y_train, cv=5)
        
        assert len(scores) == 5
        assert scores.mean() > 0.8


class TestVisualizationIntegration:
    """Tests for visualization integration."""
    
    def test_visualization_with_selector(self, full_pipeline_data):
        """Test visualization with selector results."""
        X_train, X_test, y_train, y_test = full_pipeline_data
        
        # Fit selector
        selector = FilterSelector(n_features=10, random_state=42)
        selector.fit(X_train, y_train)
        
        # Create visualizer
        viz = SelectionVisualizer()
        
        # Test plots (just check they don't error)
        try:
            ax = viz.plot_feature_importance(selector.feature_importances_, top_n=10)
            assert ax is not None
        except ImportError:
            pytest.skip("matplotlib not available")
    
    def test_visualization_with_comparator(self, full_pipeline_data):
        """Test visualization with comparator results."""
        X_train, X_test, y_train, y_test = full_pipeline_data
        
        # Compare methods
        comparator = SelectionComparator(
            methods=['filter', 'mrmr'],
            n_features=10,
            cv=3,
            random_state=42,
            verbose=0
        )
        results = comparator.compare(X_train, y_train)
        
        # Create visualizer
        try:
            viz = SelectionVisualizer()
            ax = viz.plot_method_comparison(results)
            assert ax is not None
        except ImportError:
            pytest.skip("matplotlib not available")


class TestErrorPropagation:
    """Tests for error handling and propagation."""
    
    def test_invalid_method_raises(self, full_pipeline_data):
        """Test that invalid method raises error."""
        X_train, X_test, y_train, y_test = full_pipeline_data
        
        fe = FeatureEngineering(selection_method='invalid_method')
        
        with pytest.raises(ValueError, match="Unknown"):
            fe.fit(X_train, y_train)
    
    def test_transform_before_fit_raises(self, full_pipeline_data):
        """Test that transform before fit raises error."""
        X_train, X_test, y_train, y_test = full_pipeline_data
        
        fe = FeatureEngineering()
        
        with pytest.raises(ValueError, match="not been fitted"):
            fe.transform(X_train)
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrame."""
        X = pd.DataFrame()
        y = pd.Series(dtype=float)
        
        fe = FeatureEngineering()
        
        with pytest.raises(Exception):  # Should raise some error
            fe.fit(X, y)


class TestReproducibility:
    """Tests for reproducibility."""
    
    def test_random_state_reproducibility(self, full_pipeline_data):
        """Test that random_state produces reproducible results."""
        X_train, X_test, y_train, y_test = full_pipeline_data
        
        # Run twice with same random state
        fe1 = FeatureEngineering(n_features=10, selection_method='mrmr', random_state=42)
        X1 = fe1.fit_transform(X_train, y_train)
        
        fe2 = FeatureEngineering(n_features=10, selection_method='mrmr', random_state=42)
        X2 = fe2.fit_transform(X_train, y_train)
        
        pd.testing.assert_frame_equal(X1, X2)
        assert fe1.selected_features_ == fe2.selected_features_
    
    def test_different_random_states(self, full_pipeline_data):
        """Test that different random states may produce different results."""
        X_train, X_test, y_train, y_test = full_pipeline_data
        
        fe1 = FeatureEngineering(n_features=10, selection_method='mrmr', random_state=42)
        fe1.fit(X_train, y_train)
        
        fe2 = FeatureEngineering(n_features=10, selection_method='mrmr', random_state=123)
        fe2.fit(X_train, y_train)
        
        # May or may not be different depending on method
        # Just check both ran successfully
        assert len(fe1.selected_features_) == 10
        assert len(fe2.selected_features_) == 10


class TestMulticlassSupport:
    """Tests for multiclass classification support."""
    
    def test_multiclass_classification(self):
        """Test with multiclass target."""
        data = load_iris()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        
        fe = FeatureEngineering(
            selection_method='mrmr',
            n_features=3,
            random_state=42
        )
        
        X_selected = fe.fit_transform(X, y)
        
        assert X_selected.shape[1] == 3
        assert fe._task_type == 'classification'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
