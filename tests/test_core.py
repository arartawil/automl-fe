"""
Tests for core FeatureEngineering class.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes
import tempfile
import os

from automl_fe.core import FeatureEngineering


@pytest.fixture
def classification_data():
    """Binary classification dataset."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y


@pytest.fixture
def multiclass_data():
    """Multiclass classification dataset."""
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y


@pytest.fixture
def regression_data():
    """Regression dataset."""
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y


@pytest.fixture
def mixed_data():
    """Dataset with mixed types and missing values."""
    np.random.seed(42)
    n_samples = 100
    
    X = pd.DataFrame({
        'numeric1': np.random.randn(n_samples),
        'numeric2': np.random.randn(n_samples) * 10,
        'category1': np.random.choice(['A', 'B', 'C'], n_samples),
        'category2': np.random.choice(['X', 'Y'], n_samples),
        'with_missing': np.where(
            np.random.random(n_samples) > 0.1,
            np.random.randn(n_samples),
            np.nan
        ),
    })
    y = pd.Series(np.random.randint(0, 2, n_samples))
    return X, y


class TestFeatureEngineeringInit:
    """Tests for FeatureEngineering initialization."""
    
    def test_default_init(self):
        """Test default initialization."""
        fe = FeatureEngineering()
        assert fe.selection_method == 'filter'
        assert fe.preprocessing == True
        assert fe.scaling == 'standard'
        assert fe.fitted_ == False
    
    def test_custom_init(self):
        """Test custom initialization."""
        fe = FeatureEngineering(
            selection_method='mrmr',
            n_features=5,
            task='classification',
            random_state=42
        )
        assert fe.selection_method == 'mrmr'
        assert fe.n_features == 5
        assert fe.task == 'classification'
        assert fe.random_state == 42


class TestFeatureEngineeringFit:
    """Tests for fitting FeatureEngineering."""
    
    def test_fit_classification(self, classification_data):
        """Test fitting on classification data."""
        X, y = classification_data
        fe = FeatureEngineering(
            n_features=10, 
            selection_method='mrmr',  # Use mrmr to avoid correlation filtering
            random_state=42
        )
        fe.fit(X, y)
        
        assert fe.fitted_ == True
        assert len(fe.selected_features_) == 10
        assert len(fe.feature_importances_) > 0
    
    def test_fit_regression(self, regression_data):
        """Test fitting on regression data."""
        X, y = regression_data
        fe = FeatureEngineering(
            n_features=5, 
            task='regression',
            random_state=42
        )
        fe.fit(X, y)
        
        assert fe.fitted_ == True
        assert len(fe.selected_features_) == 5
    
    def test_fit_transform(self, classification_data):
        """Test fit_transform method."""
        X, y = classification_data
        fe = FeatureEngineering(
            n_features=10, 
            selection_method='mrmr',  # Use mrmr to avoid correlation filtering
            random_state=42
        )
        X_transformed = fe.fit_transform(X, y)
        
        assert X_transformed.shape[1] == 10
        assert X_transformed.shape[0] == X.shape[0]
    
    def test_task_detection(self, classification_data, regression_data):
        """Test automatic task detection."""
        X_clf, y_clf = classification_data
        X_reg, y_reg = regression_data
        
        fe_clf = FeatureEngineering(task='auto', random_state=42)
        fe_clf.fit(X_clf, y_clf)
        assert fe_clf._task_type == 'classification'
        
        fe_reg = FeatureEngineering(task='auto', random_state=42)
        fe_reg.fit(X_reg, y_reg)
        assert fe_reg._task_type == 'regression'


class TestFeatureEngineeringTransform:
    """Tests for transform functionality."""
    
    def test_transform_after_fit(self, classification_data):
        """Test transform after fitting."""
        X, y = classification_data
        fe = FeatureEngineering(
            n_features=10, 
            selection_method='mrmr',  # Use mrmr to avoid correlation filtering
            random_state=42
        )
        fe.fit(X, y)
        
        X_new = X.iloc[:50]
        X_transformed = fe.transform(X_new)
        
        assert X_transformed.shape == (50, 10)
    
    def test_transform_before_fit_raises(self, classification_data):
        """Test that transform before fit raises error."""
        X, y = classification_data
        fe = FeatureEngineering()
        
        with pytest.raises(ValueError, match="not been fitted"):
            fe.transform(X)
    
    def test_transform_consistency(self, classification_data):
        """Test that transform produces consistent results."""
        X, y = classification_data
        fe = FeatureEngineering(
            n_features=10, 
            selection_method='mrmr',  # Use mrmr to avoid correlation filtering
            random_state=42
        )
        
        X_fit = fe.fit_transform(X, y)
        X_transform = fe.transform(X)
        
        pd.testing.assert_frame_equal(X_fit, X_transform)


class TestFeatureEngineeringMethods:
    """Tests for various selection methods."""
    
    @pytest.mark.parametrize("method", ['filter', 'mrmr', 'embedded'])
    def test_selection_methods(self, classification_data, method):
        """Test different selection methods."""
        X, y = classification_data
        # Add correlation_threshold=1.0 for filter method to avoid correlation removal
        params = {'correlation_threshold': 1.0} if method == 'filter' else {}
        fe = FeatureEngineering(
            selection_method=method,
            n_features=10,
            selection_params=params,
            random_state=42
        )
        X_selected = fe.fit_transform(X, y)
        
        assert X_selected.shape[1] == 10
        assert len(fe.selected_features_) == 10
    
    def test_auto_select(self, classification_data):
        """Test auto_select method."""
        X, y = classification_data
        fe = FeatureEngineering(n_features=10, random_state=42)
        
        results = fe.auto_select(X, y, methods=['filter', 'mrmr'])
        
        assert 'best_method' in results
        assert 'best_score' in results
        assert results['best_method'] is not None


class TestFeatureEngineeringPreprocessing:
    """Tests for preprocessing functionality."""
    
    def test_with_preprocessing(self, classification_data):
        """Test with preprocessing enabled."""
        X, y = classification_data
        fe = FeatureEngineering(
            preprocessing=True,
            scaling='standard',
            n_features=10,
            selection_method='mrmr',  # Use mrmr to avoid correlation filtering
            random_state=42
        )
        X_transformed = fe.fit_transform(X, y)
        
        assert X_transformed.shape[1] == 10
    
    def test_without_preprocessing(self, classification_data):
        """Test with preprocessing disabled."""
        X, y = classification_data
        fe = FeatureEngineering(
            preprocessing=False,
            n_features=10,
            selection_method='mrmr',  # Use mrmr to avoid correlation filtering
            random_state=42
        )
        X_transformed = fe.fit_transform(X, y)
        
        assert X_transformed.shape[1] == 10
    
    def test_mixed_data_preprocessing(self, mixed_data):
        """Test preprocessing with mixed data types."""
        X, y = mixed_data
        fe = FeatureEngineering(
            preprocessing=True,
            handle_missing=True,
            n_features=3,
            random_state=42
        )
        X_transformed = fe.fit_transform(X, y)
        
        # Should not have NaN values after imputation
        assert not X_transformed.isna().any().any()


class TestFeatureEngineeringSerialization:
    """Tests for save/load functionality."""
    
    def test_save_load(self, classification_data):
        """Test saving and loading pipeline."""
        X, y = classification_data
        fe = FeatureEngineering(
            n_features=10, 
            selection_method='mrmr',  # Use mrmr to avoid correlation filtering
            random_state=42
        )
        fe.fit(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'pipeline.pkl')
            fe.save(path)
            
            fe_loaded = FeatureEngineering.load(path)
            
            assert fe_loaded.fitted_ == True
            assert fe_loaded.selected_features_ == fe.selected_features_
    
    def test_loaded_transform(self, classification_data):
        """Test that loaded pipeline transforms correctly."""
        X, y = classification_data
        fe = FeatureEngineering(
            n_features=10, 
            selection_method='mrmr',  # Use mrmr to avoid correlation filtering
            random_state=42
        )
        X_original = fe.fit_transform(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'pipeline.pkl')
            fe.save(path)
            fe_loaded = FeatureEngineering.load(path)
            
            X_loaded = fe_loaded.transform(X)
            pd.testing.assert_frame_equal(X_original, X_loaded)


class TestFeatureEngineeringReports:
    """Tests for reporting functionality."""
    
    def test_get_feature_importances(self, classification_data):
        """Test getting feature importances."""
        X, y = classification_data
        fe = FeatureEngineering(
            n_features=10, 
            selection_method='mrmr',  # Use mrmr to avoid correlation filtering
            random_state=42
        )
        fe.fit(X, y)
        
        importance_df = fe.get_feature_importances()
        
        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
    
    def test_feature_engineering_report(self, classification_data):
        """Test generating report."""
        X, y = classification_data
        fe = FeatureEngineering(
            n_features=10, 
            selection_method='mrmr',  # Use mrmr to avoid correlation filtering
            random_state=42
        )
        fe.fit(X, y)
        
        report = fe.feature_engineering_report()
        
        assert 'task_type' in report
        assert 'selected_features' in report
        assert 'selection_method' in report
    
    def test_get_feature_names(self, classification_data):
        """Test getting feature names."""
        X, y = classification_data
        fe = FeatureEngineering(
            n_features=10, 
            selection_method='mrmr',  # Use mrmr to avoid correlation filtering
            random_state=42
        )
        fe.fit(X, y)
        
        names = fe.get_feature_names()
        
        assert len(names) == 10
        assert names == fe.selected_features_


class TestFeatureEngineeringEdgeCases:
    """Tests for edge cases."""
    
    def test_n_features_larger_than_total(self, classification_data):
        """Test when n_features > total features."""
        X, y = classification_data
        fe = FeatureEngineering(n_features=100, random_state=42)
        X_transformed = fe.fit_transform(X, y)
        
        # Should not select more than available
        assert X_transformed.shape[1] <= X.shape[1]
    
    def test_n_features_as_fraction(self, classification_data):
        """Test n_features as fraction."""
        X, y = classification_data
        fe = FeatureEngineering(
            n_features=0.5, 
            selection_method='mrmr',  # Use mrmr to avoid correlation filtering
            random_state=42
        )
        X_transformed = fe.fit_transform(X, y)
        
        expected_n = int(X.shape[1] * 0.5)
        assert X_transformed.shape[1] == expected_n
    
    def test_reproducibility(self, classification_data):
        """Test that random_state ensures reproducibility."""
        X, y = classification_data
        
        fe1 = FeatureEngineering(
            n_features=10, 
            selection_method='mrmr',  # Use mrmr for consistency
            random_state=42
        )
        X1 = fe1.fit_transform(X, y)
        
        fe2 = FeatureEngineering(
            n_features=10, 
            selection_method='mrmr',  # Use mrmr for consistency
            random_state=42
        )
        X2 = fe2.fit_transform(X, y)
        
        pd.testing.assert_frame_equal(X1, X2)
        assert fe1.selected_features_ == fe2.selected_features_


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
