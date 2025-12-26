"""
Visualization Examples for AutoML Feature Engineering

This script demonstrates various visualization capabilities
for feature selection analysis.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

from automl_fe.selection import FilterSelector, mRMRSelector
from automl_fe.evaluation import SelectionComparator
from automl_fe.visualization import SelectionVisualizer


def main():
    """Main function demonstrating visualization capabilities."""
    
    print("=" * 60)
    print("AutoML-FE - Visualization Examples")
    print("=" * 60)
    
    # Load dataset
    print("\n1. Loading dataset...")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    # Create visualizer
    viz = SelectionVisualizer(figsize=(10, 6), dpi=100)
    
    # =========================================
    # Feature Importance Plot
    # =========================================
    print("\n2. Creating feature importance plot...")
    
    selector = FilterSelector(
        method='mutual_info',
        n_features=15,
        task='classification',
        random_state=42
    )
    selector.fit(X, y)
    
    fig = viz.plot_feature_importance(
        selector.feature_importances_,
        top_n=15,
        title='Top 15 Features by Mutual Information'
    )
    print("   Feature importance plot created")
    
    # =========================================
    # Correlation Matrix
    # =========================================
    print("\n3. Creating correlation matrix...")
    
    selected_features = selector.selected_features_[:10]
    fig = viz.plot_correlation_matrix(
        X,
        features=selected_features,
        title='Correlation Matrix of Selected Features'
    )
    print("   Correlation matrix created")
    
    # =========================================
    # Method Comparison
    # =========================================
    print("\n4. Comparing selection methods...")
    
    comparator = SelectionComparator(
        methods=['filter', 'mrmr', 'embedded'],
        n_features=10,
        cv=5,
        random_state=42,
        verbose=0
    )
    results = comparator.compare(X, y)
    
    fig = viz.plot_method_comparison(
        results,
        metric='accuracy_mean',
        error_metric='accuracy_std',
        title='Feature Selection Method Comparison'
    )
    print("   Method comparison plot created")
    
    # =========================================
    # Feature Overlap
    # =========================================
    print("\n5. Creating feature overlap heatmap...")
    
    fig = viz.plot_feature_overlap(
        comparator.feature_overlap_,
        title='Feature Overlap Between Methods'
    )
    print("   Feature overlap heatmap created")
    
    # =========================================
    # Stability Plot
    # =========================================
    print("\n6. Creating stability plot...")
    
    fig = viz.plot_stability(
        comparator.stability_scores_,
        title='Selection Stability Across Methods'
    )
    print("   Stability plot created")
    
    # =========================================
    # Selected vs Unselected
    # =========================================
    print("\n7. Creating selected vs unselected comparison...")
    
    fig = viz.plot_selected_vs_unselected(
        X,
        selector.selected_features_,
        selector.feature_importances_,
        title='Importance: Selected vs Unselected Features'
    )
    print("   Selected vs unselected plot created")
    
    # =========================================
    # Feature Distribution
    # =========================================
    print("\n8. Creating feature distribution plot...")
    
    top_feature = selector.selected_features_[0]
    fig = viz.plot_feature_distribution(
        X,
        feature=top_feature,
        target=y,
        title=f'Distribution of {top_feature} by Class'
    )
    print("   Feature distribution plot created")
    
    # =========================================
    # Comprehensive Report
    # =========================================
    print("\n9. Creating comprehensive report figure...")
    
    fig = viz.create_report_figure(
        feature_importances=selector.feature_importances_,
        selected_features=selector.selected_features_,
        comparison_df=results,
        correlation_df=X[selected_features]
    )
    print("   Report figure created")
    
    # Save figures
    print("\n10. Saving figures...")
    # Uncomment to save:
    # viz.save_figure(fig, 'feature_selection_report.png', dpi=150)
    # print("    Saved to 'feature_selection_report.png'")
    
    print("\n" + "=" * 60)
    print("Visualization examples completed!")
    print("Note: Use plt.show() to display plots interactively")
    print("=" * 60)
    
    # Uncomment to show plots:
    # import matplotlib.pyplot as plt
    # plt.show()


if __name__ == '__main__':
    main()
