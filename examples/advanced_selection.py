"""
Advanced Feature Selection Example

This script demonstrates advanced feature selection techniques
and comparison between different methods.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

from automl_fe.selection import (
    FilterSelector,
    WrapperSelector,
    EmbeddedSelector,
    mRMRSelector,
)
from automl_fe.evaluation import SelectionComparator


def main():
    """Main function demonstrating advanced selection."""
    
    print("=" * 60)
    print("AutoML-FE - Advanced Feature Selection Example")
    print("=" * 60)
    
    # Load dataset
    print("\n1. Loading dataset...")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    print(f"   Shape: {X.shape}")
    
    # Number of features to select
    n_features = 10
    
    # =========================================
    # Filter-based Selection
    # =========================================
    print("\n" + "-" * 40)
    print("2. Filter-based Selection Methods")
    print("-" * 40)
    
    filter_methods = ['mutual_info', 'f_statistic', 'chi2', 'correlation']
    
    for method in filter_methods:
        selector = FilterSelector(
            method=method,
            n_features=n_features,
            task='classification',
            random_state=42
        )
        X_selected = selector.fit_transform(X, y)
        
        # Evaluate
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        scores = cross_val_score(model, X_selected, y, cv=5)
        
        print(f"\n   {method.upper()}:")
        print(f"   Selected: {selector.selected_features_[:5]}...")
        print(f"   CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    # =========================================
    # mRMR Selection
    # =========================================
    print("\n" + "-" * 40)
    print("3. mRMR Selection")
    print("-" * 40)
    
    mrmr_selector = mRMRSelector(
        n_features=n_features,
        alpha=1.0,
        random_state=42
    )
    X_mrmr = mrmr_selector.fit_transform(X, y)
    
    print(f"\n   Selection order:")
    for i, feature in enumerate(mrmr_selector.selection_order_[:5], 1):
        relevance = mrmr_selector.relevance_scores_.get(feature, 0)
        print(f"   {i}. {feature} (relevance: {relevance:.4f})")
    
    # Evaluate
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    scores = cross_val_score(model, X_mrmr, y, cv=5)
    print(f"\n   CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    # =========================================
    # Embedded Selection with Stability
    # =========================================
    print("\n" + "-" * 40)
    print("4. Embedded Selection with Stability Analysis")
    print("-" * 40)
    
    embedded_selector = EmbeddedSelector(
        method='random_forest',
        n_features=n_features,
        stability_selection=True,
        n_stability_runs=10,
        random_state=42,
        verbose=1
    )
    X_embedded = embedded_selector.fit_transform(X, y)
    
    print(f"\n   Selected features with stability scores:")
    stable_features = sorted(
        embedded_selector.stability_scores_.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    for feature, stability in stable_features:
        if stability > 0.5:
            print(f"   * {feature}: {stability:.2f}")
    
    # =========================================
    # Method Comparison
    # =========================================
    print("\n" + "-" * 40)
    print("5. Comprehensive Method Comparison")
    print("-" * 40)
    
    comparator = SelectionComparator(
        methods=[
            {'name': 'filter_mi', 'method': 'filter', 'params': {'method': 'mutual_info'}},
            {'name': 'filter_f', 'method': 'filter', 'params': {'method': 'f_statistic'}},
            {'name': 'mrmr', 'method': 'mrmr', 'params': {}},
            {'name': 'rf_importance', 'method': 'embedded', 'params': {'method': 'random_forest'}},
            {'name': 'lasso', 'method': 'embedded', 'params': {'method': 'lasso'}},
        ],
        n_features=n_features,
        cv=5,
        random_state=42,
        verbose=1
    )
    
    results = comparator.compare(X, y)
    
    print("\n   Comparison Results:")
    summary = comparator.get_comparison_summary()
    print(summary[['method', 'n_features', 'accuracy_mean', 'stability', 'time']].to_string())
    
    print(f"\n   Best method: {comparator.best_method_}")
    
    # Feature overlap
    print("\n   Feature overlap between methods:")
    overlap = comparator.get_feature_overlap()
    print(overlap.round(2).to_string())
    
    # Common features
    common = comparator.get_common_features(min_methods=3)
    print(f"\n   Features selected by 3+ methods: {common}")
    
    # =========================================
    # Wrapper Methods (slower, more accurate)
    # =========================================
    print("\n" + "-" * 40)
    print("6. Wrapper-based Selection (RFE)")
    print("-" * 40)
    
    wrapper_selector = WrapperSelector(
        method='rfe',
        n_features=n_features,
        task='classification',
        cv=3,
        random_state=42,
        verbose=0
    )
    X_wrapper = wrapper_selector.fit_transform(X, y)
    
    print(f"   RFE Selected: {wrapper_selector.selected_features_[:5]}...")
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    scores = cross_val_score(model, X_wrapper, y, cv=5)
    print(f"   CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    print("\n" + "=" * 60)
    print("Advanced selection example completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
