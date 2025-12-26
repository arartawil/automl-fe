"""
Tests for feature selection methods.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes

from automl_fe.selection import (
    FilterSelector,
    WrapperSelector,
    EmbeddedSelector,
    mRMRSelector,
)


@pytest.fixture
def classification_data():
    """Binary classification dataset."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data[:100], columns=data.feature_names)  # Subset for speed
    y = pd.Series(data.target[:100])
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
    X = pd.DataFrame(data.data[:100], columns=data.feature_names)
    y = pd.Series(data.target[:100])
    return X, y


class TestFilterSelector:
    """Tests for FilterSelector class."""
    
    @pytest.mark.parametrize("method", [
        'mutual_info', 'f_statistic', 'chi2', 'variance', 'correlation'
    ])
    def test_filter_methods(self, classification_data, method):
        """Test different filter methods."""
        X, y = classification_data
        selector = FilterSelector(
            method=method,
            n_features=5,
            task='classification',
            correlation_threshold=1.0,  # Disable correlation filtering for exact count
            random_state=42
        )
        X_selected = selector.fit_transform(X, y)
        
        assert X_selected.shape[1] == 5
        assert len(selector.selected_features_) == 5
    
    def test_feature_scores(self, classification_data):
        """Test feature scores output."""
        X, y = classification_data
        selector = FilterSelector(method='mutual_info', n_features=5, random_state=42)
        selector.fit(X, y)
        
        scores = selector.get_feature_scores()
        
        assert isinstance(scores, pd.DataFrame)
        assert 'feature' in scores.columns
        assert 'score' in scores.columns
        assert 'rank' in scores.columns
    
    def test_correlation_filtering(self, classification_data):
        """Test correlation threshold filtering."""
        X, y = classification_data
        selector = FilterSelector(
            method='mutual_info',
            n_features=10,
            correlation_threshold=0.9,
            random_state=42
        )
        selector.fit(X, y)
        
        # Should have removed some correlated features
        assert len(selector.selected_features_) <= 10
    
    def test_get_support(self, classification_data):
        """Test get_support method."""
        X, y = classification_data
        selector = FilterSelector(n_features=5, correlation_threshold=1.0, random_state=42)
        selector.fit(X, y)
        
        mask = selector.get_support(indices=False)
        indices = selector.get_support(indices=True)
        
        assert mask.sum() == 5
        assert len(indices) == 5
    
    def test_regression_task(self, regression_data):
        """Test with regression task."""
        X, y = regression_data
        selector = FilterSelector(
            method='f_statistic',
            n_features=5,
            task='regression',
            random_state=42
        )
        X_selected = selector.fit_transform(X, y)
        
        assert X_selected.shape[1] == 5


class TestmRMRSelector:
    """Tests for mRMRSelector class."""
    
    def test_basic_selection(self, classification_data):
        """Test basic mRMR selection."""
        X, y = classification_data
        selector = mRMRSelector(n_features=5, random_state=42)
        X_selected = selector.fit_transform(X, y)
        
        assert X_selected.shape[1] == 5
        assert len(selector.selected_features_) == 5
        assert len(selector.selection_order_) == 5
    
    def test_relevance_scores(self, classification_data):
        """Test relevance scores."""
        X, y = classification_data
        selector = mRMRSelector(n_features=5, random_state=42)
        selector.fit(X, y)
        
        relevance = selector.get_relevance_scores()
        
        assert isinstance(relevance, pd.DataFrame)
        assert 'relevance' in relevance.columns
    
    def test_redundancy_matrix(self, classification_data):
        """Test redundancy matrix."""
        X, y = classification_data
        selector = mRMRSelector(n_features=5, random_state=42)
        selector.fit(X, y)
        
        redundancy = selector.get_redundancy_matrix()
        
        assert isinstance(redundancy, pd.DataFrame)
        assert redundancy.shape[0] == redundancy.shape[1]
    
    def test_alpha_parameter(self, classification_data):
        """Test different alpha values."""
        X, y = classification_data
        
        # Low alpha = more focus on relevance
        selector_low = mRMRSelector(n_features=5, alpha=0.1, random_state=42)
        selector_low.fit(X, y)
        
        # High alpha = more focus on reducing redundancy
        selector_high = mRMRSelector(n_features=5, alpha=2.0, random_state=42)
        selector_high.fit(X, y)
        
        # Both should work, may select different features
        assert len(selector_low.selected_features_) == 5
        assert len(selector_high.selected_features_) == 5
    
    def test_discrete_target_detection(self, classification_data, regression_data):
        """Test automatic target type detection."""
        X_clf, y_clf = classification_data
        X_reg, y_reg = regression_data
        
        selector_clf = mRMRSelector(n_features=3, random_state=42)
        selector_clf.fit(X_clf, y_clf)
        assert selector_clf.discrete_target == True
        
        selector_reg = mRMRSelector(n_features=3, random_state=42)
        selector_reg.fit(X_reg, y_reg)
        assert selector_reg.discrete_target == False


class TestWrapperSelector:
    """Tests for WrapperSelector class."""
    
    def test_rfe_selection(self, classification_data):
        """Test RFE selection."""
        X, y = classification_data
        selector = WrapperSelector(
            method='rfe',
            n_features=5,
            task='classification',
            random_state=42
        )
        X_selected = selector.fit_transform(X, y)
        
        assert X_selected.shape[1] == 5
    
    def test_forward_selection(self, classification_data):
        """Test forward selection."""
        X, y = classification_data
        # Use small subset for speed
        X_small = X.iloc[:50, :10]
        y_small = y.iloc[:50]
        
        selector = WrapperSelector(
            method='forward',
            n_features=3,
            cv=2,
            random_state=42
        )
        X_selected = selector.fit_transform(X_small, y_small)
        
        assert X_selected.shape[1] == 3
    
    def test_backward_selection(self, classification_data):
        """Test backward elimination."""
        X, y = classification_data
        # Use small subset for speed
        X_small = X.iloc[:50, :10]
        y_small = y.iloc[:50]
        
        selector = WrapperSelector(
            method='backward',
            n_features=3,
            cv=2,
            random_state=42
        )
        X_selected = selector.fit_transform(X_small, y_small)
        
        assert X_selected.shape[1] == 3
    
    def test_custom_estimator(self, classification_data):
        """Test with custom estimator."""
        from sklearn.linear_model import LogisticRegression
        
        X, y = classification_data
        selector = WrapperSelector(
            method='rfe',
            n_features=5,
            estimator=LogisticRegression(max_iter=1000),
            random_state=42
        )
        X_selected = selector.fit_transform(X, y)
        
        assert X_selected.shape[1] == 5


class TestEmbeddedSelector:
    """Tests for EmbeddedSelector class."""
    
    @pytest.mark.parametrize("method", [
        'random_forest', 'lasso', 'ridge', 'elasticnet'
    ])
    def test_embedded_methods(self, classification_data, method):
        """Test different embedded methods."""
        X, y = classification_data
        selector = EmbeddedSelector(
            method=method,
            n_features=5,
            task='classification',
            random_state=42
        )
        X_selected = selector.fit_transform(X, y)
        
        assert X_selected.shape[1] == 5
    
    def test_feature_importances(self, classification_data):
        """Test feature importance output."""
        X, y = classification_data
        selector = EmbeddedSelector(
            method='random_forest',
            n_features=5,
            random_state=42
        )
        selector.fit(X, y)
        
        importances = selector.get_feature_importances()
        
        assert isinstance(importances, pd.DataFrame)
        assert 'importance' in importances.columns
    
    def test_stability_selection(self, classification_data):
        """Test stability selection."""
        X, y = classification_data
        selector = EmbeddedSelector(
            method='random_forest',
            n_features=5,
            stability_selection=True,
            n_stability_runs=5,
            random_state=42
        )
        selector.fit(X, y)
        
        assert len(selector.stability_scores_) > 0
    
    def test_threshold_selection(self, classification_data):
        """Test threshold-based selection."""
        X, y = classification_data
        selector = EmbeddedSelector(
            method='random_forest',
            threshold='mean',
            task='classification',
            random_state=42
        )
        selector.fit(X, y)
        
        # Should select features above mean importance
        assert len(selector.selected_features_) > 0
    
    def test_regression_task(self, regression_data):
        """Test with regression task."""
        X, y = regression_data
        selector = EmbeddedSelector(
            method='lasso',
            n_features=5,
            task='regression',
            random_state=42
        )
        X_selected = selector.fit_transform(X, y)
        
        assert X_selected.shape[1] == 5


class TestSelectorInteroperability:
    """Tests for selector interoperability."""
    
    def test_transform_new_data(self, classification_data):
        """Test transforming new data after fitting."""
        X, y = classification_data
        X_train = X.iloc[:80]
        X_test = X.iloc[80:]
        y_train = y.iloc[:80]
        
        selector = FilterSelector(n_features=5, correlation_threshold=1.0, random_state=42)
        selector.fit(X_train, y_train)
        
        X_test_transformed = selector.transform(X_test)
        
        assert X_test_transformed.shape[1] == 5
        assert list(X_test_transformed.columns) == selector.selected_features_
    
    def test_consistent_feature_order(self, classification_data):
        """Test that feature order is consistent."""
        X, y = classification_data
        
        selector = mRMRSelector(n_features=5, random_state=42)
        X_fit = selector.fit_transform(X, y)
        X_transform = selector.transform(X)
        
        assert list(X_fit.columns) == list(X_transform.columns)
    
    def test_handle_missing_features(self, classification_data):
        """Test handling when some features are missing."""
        X, y = classification_data
        
        selector = FilterSelector(n_features=5, correlation_threshold=1.0, random_state=42)
        selector.fit(X, y)
        
        # Remove a non-selected feature
        X_partial = X.drop(columns=[X.columns[0]])
        
        # Should still work if all selected features present
        if X.columns[0] not in selector.selected_features_:
            X_transformed = selector.transform(X_partial)
            assert X_transformed.shape[1] == 5


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_feature_selection(self, classification_data):
        """Test selecting single feature."""
        X, y = classification_data
        
        selector = FilterSelector(n_features=1, correlation_threshold=1.0, random_state=42)
        X_selected = selector.fit_transform(X, y)
        
        assert X_selected.shape[1] == 1
    
    def test_all_features_selection(self, classification_data):
        """Test selecting all features."""
        X, y = classification_data
        
        selector = FilterSelector(n_features=X.shape[1], correlation_threshold=1.0, random_state=42)
        X_selected = selector.fit_transform(X, y)
        
        assert X_selected.shape[1] == X.shape[1]
    
    def test_fractional_n_features(self, classification_data):
        """Test fractional n_features."""
        X, y = classification_data
        
        selector = FilterSelector(n_features=0.5, correlation_threshold=1.0, random_state=42)
        X_selected = selector.fit_transform(X, y)
        
        expected = max(1, int(X.shape[1] * 0.5))
        assert X_selected.shape[1] == expected


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
