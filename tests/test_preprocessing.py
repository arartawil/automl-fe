"""
Tests for preprocessing functions.
"""

import pytest
import numpy as np
import pandas as pd

from automl_fe.preprocessing import (
    smart_encode_categorical,
    adaptive_impute,
    generate_polynomial_features,
    scale_features,
    detect_and_handle_outliers,
    optimize_dtypes,
    remove_low_variance_features,
    remove_highly_correlated_features,
)


@pytest.fixture
def categorical_data():
    """DataFrame with categorical columns."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'binary': np.random.choice(['yes', 'no'], n),
        'low_card': np.random.choice(['A', 'B', 'C', 'D'], n),
        'high_card': [f'cat_{i % 20}' for i in range(n)],
        'numeric': np.random.randn(n),
    })


@pytest.fixture
def missing_data():
    """DataFrame with missing values."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'low_missing': np.random.randn(n),
        'med_missing': np.random.randn(n),
        'high_missing': np.random.randn(n),
        'categorical': np.random.choice(['A', 'B', 'C'], n),
    })
    
    # Add missing values
    df.loc[np.random.choice(n, 5, replace=False), 'low_missing'] = np.nan
    df.loc[np.random.choice(n, 20, replace=False), 'med_missing'] = np.nan
    df.loc[np.random.choice(n, 60, replace=False), 'high_missing'] = np.nan
    df.loc[np.random.choice(n, 10, replace=False), 'categorical'] = np.nan
    
    return df


@pytest.fixture
def numeric_data():
    """DataFrame with numeric columns."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'a': np.random.randn(n),
        'b': np.random.randn(n) * 10,
        'c': np.random.randn(n) + 5,
    })


@pytest.fixture
def correlated_data():
    """DataFrame with correlated features."""
    np.random.seed(42)
    n = 100
    base = np.random.randn(n)
    return pd.DataFrame({
        'a': base,
        'b': base * 2 + np.random.randn(n) * 0.1,  # Highly correlated with a
        'c': base * -1 + np.random.randn(n) * 0.1,  # Highly correlated with a
        'd': np.random.randn(n),  # Independent
    })


class TestSmartEncodeCategorical:
    """Tests for smart_encode_categorical function."""
    
    def test_binary_encoding(self, categorical_data):
        """Test binary column encoding."""
        df_encoded, encoders = smart_encode_categorical(categorical_data)
        
        assert 'binary' in df_encoded.columns
        assert df_encoded['binary'].dtype in [np.int64, np.int32, np.float64]
        assert encoders['binary']['type'] == 'label'
    
    def test_low_cardinality_encoding(self, categorical_data):
        """Test low cardinality one-hot encoding."""
        df_encoded, encoders = smart_encode_categorical(categorical_data)
        
        # low_card should be one-hot encoded
        assert 'low_card' not in df_encoded.columns
        assert any('low_card_' in col for col in df_encoded.columns)
        assert encoders['low_card']['type'] == 'onehot'
    
    def test_high_cardinality_encoding(self, categorical_data):
        """Test high cardinality label encoding."""
        df_encoded, encoders = smart_encode_categorical(categorical_data)
        
        # high_card should be label encoded
        assert 'high_card' in df_encoded.columns
        assert df_encoded['high_card'].dtype in [np.int64, np.int32, np.float64]
        assert encoders['high_card']['type'] == 'label'
    
    def test_transform_with_fitted_encoders(self, categorical_data):
        """Test transforming with fitted encoders."""
        df_encoded, encoders = smart_encode_categorical(categorical_data)
        
        # Transform new data
        new_data = categorical_data.head(10).copy()
        df_new, _ = smart_encode_categorical(new_data, fitted_encoders=encoders)
        
        assert df_new.shape[0] == 10
    
    def test_numeric_columns_unchanged(self, categorical_data):
        """Test that numeric columns are unchanged."""
        df_encoded, encoders = smart_encode_categorical(categorical_data)
        
        pd.testing.assert_series_equal(
            df_encoded['numeric'],
            categorical_data['numeric'],
            check_names=False
        )


class TestAdaptiveImpute:
    """Tests for adaptive_impute function."""
    
    def test_imputation_removes_nan(self, missing_data):
        """Test that imputation removes all NaN values."""
        df_imputed, imputers = adaptive_impute(missing_data)
        
        assert not df_imputed.isna().any().any()
    
    def test_imputation_preserves_shape(self, missing_data):
        """Test that imputation preserves shape."""
        df_imputed, imputers = adaptive_impute(missing_data)
        
        assert df_imputed.shape == missing_data.shape
    
    def test_imputation_with_fitted_imputers(self, missing_data):
        """Test transforming with fitted imputers."""
        df_imputed, imputers = adaptive_impute(missing_data)
        
        new_data = missing_data.head(10).copy()
        df_new, _ = adaptive_impute(new_data, fitted_imputers=imputers)
        
        assert not df_new.isna().any().any()
    
    def test_no_imputation_needed(self, numeric_data):
        """Test with no missing values."""
        df_imputed, imputers = adaptive_impute(numeric_data)
        
        pd.testing.assert_frame_equal(df_imputed, numeric_data)


class TestGeneratePolynomialFeatures:
    """Tests for generate_polynomial_features function."""
    
    def test_polynomial_generation(self, numeric_data):
        """Test polynomial feature generation."""
        df_poly, transformer = generate_polynomial_features(numeric_data, degree=2)
        
        # Should have more features
        assert df_poly.shape[1] >= numeric_data.shape[1]
        assert transformer is not None
    
    def test_interaction_only(self, numeric_data):
        """Test interaction-only features."""
        df_poly, transformer = generate_polynomial_features(
            numeric_data, degree=2, interaction_only=True
        )
        
        assert df_poly.shape[1] < numeric_data.shape[1] ** 2
    
    def test_transform_with_fitted(self, numeric_data):
        """Test transforming with fitted transformer."""
        df_poly, transformer = generate_polynomial_features(numeric_data, degree=2)
        
        new_data = numeric_data.head(10)
        df_new, _ = generate_polynomial_features(
            new_data, degree=2, fitted_transformer=transformer
        )
        
        assert df_new.shape[1] == df_poly.shape[1]


class TestScaleFeatures:
    """Tests for scale_features function."""
    
    @pytest.mark.parametrize("method", ['standard', 'minmax', 'robust'])
    def test_scaling_methods(self, numeric_data, method):
        """Test different scaling methods."""
        df_scaled, scaler = scale_features(numeric_data, method=method)
        
        assert df_scaled.shape == numeric_data.shape
        assert scaler is not None
    
    def test_standard_scaling_properties(self, numeric_data):
        """Test standard scaling produces ~mean=0, std=1."""
        df_scaled, _ = scale_features(numeric_data, method='standard')
        
        means = df_scaled.mean()
        stds = df_scaled.std()
        
        assert all(means.abs() < 0.1)  # Mean close to 0
        assert all((stds - 1).abs() < 0.1)  # Std close to 1
    
    def test_minmax_scaling_properties(self, numeric_data):
        """Test minmax scaling produces values in [0, 1]."""
        df_scaled, _ = scale_features(numeric_data, method='minmax')
        
        assert (df_scaled.min() >= 0).all()
        assert (df_scaled.max() <= 1).all()
    
    def test_transform_with_fitted(self, numeric_data):
        """Test transforming with fitted scaler."""
        df_scaled, scaler = scale_features(numeric_data, method='standard')
        
        new_data = numeric_data.head(10)
        df_new, _ = scale_features(new_data, method='standard', fitted_scaler=scaler)
        
        assert df_new.shape == new_data.shape


class TestDetectAndHandleOutliers:
    """Tests for detect_and_handle_outliers function."""
    
    def test_outlier_clipping(self):
        """Test outlier clipping."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 100, 4, 5],
            'b': [10, 20, 30, 40, 50, 60],
        })
        
        df_clean = detect_and_handle_outliers(df, method='iqr', handling='clip')
        
        assert df_clean['a'].max() < 100
    
    def test_outlier_nan(self):
        """Test replacing outliers with NaN."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 100, 4, 5],
        })
        
        df_clean = detect_and_handle_outliers(df, method='iqr', handling='nan')
        
        assert df_clean['a'].isna().sum() > 0
    
    @pytest.mark.parametrize("method", ['iqr', 'zscore'])
    def test_detection_methods(self, method):
        """Test different detection methods."""
        df = pd.DataFrame({
            'a': np.concatenate([np.random.randn(100), [10, -10]]),
        })
        
        df_clean = detect_and_handle_outliers(df, method=method, handling='clip')
        
        assert df_clean['a'].max() < 10


class TestOptimizeDtypes:
    """Tests for optimize_dtypes function."""
    
    def test_downcasting(self):
        """Test dtype downcasting."""
        df = pd.DataFrame({
            'ints': np.array([1, 2, 3], dtype=np.int64),
            'floats': np.array([1.0, 2.0, 3.0], dtype=np.float64),
        })
        
        df_opt = optimize_dtypes(df)
        
        # Should be smaller or same size
        assert df_opt.memory_usage().sum() <= df.memory_usage().sum()


class TestRemoveLowVarianceFeatures:
    """Tests for remove_low_variance_features function."""
    
    def test_low_variance_removal(self):
        """Test removal of low variance features."""
        df = pd.DataFrame({
            'constant': [1, 1, 1, 1, 1],
            'variable': [1, 2, 3, 4, 5],
        })
        
        df_filtered, removed = remove_low_variance_features(df, threshold=0.01)
        
        assert 'constant' in removed
        assert 'variable' not in removed
        assert 'variable' in df_filtered.columns


class TestRemoveHighlyCorrelatedFeatures:
    """Tests for remove_highly_correlated_features function."""
    
    def test_correlation_removal(self, correlated_data):
        """Test removal of highly correlated features."""
        df_filtered, removed = remove_highly_correlated_features(
            correlated_data, threshold=0.95
        )
        
        # Should have removed b or c (correlated with a)
        assert len(removed) >= 1
        assert df_filtered.shape[1] < correlated_data.shape[1]
    
    def test_independent_features_kept(self, correlated_data):
        """Test that independent features are kept."""
        df_filtered, removed = remove_highly_correlated_features(
            correlated_data, threshold=0.95
        )
        
        # d should be kept as it's independent
        assert 'd' in df_filtered.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
