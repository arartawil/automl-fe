"""
Real-World Example: Complete ML Workflow with Feature Engineering

This script demonstrates a complete machine learning workflow
using automl-fe for feature engineering on a real dataset.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)
import warnings
warnings.filterwarnings('ignore')

from automl_fe import FeatureEngineering
from automl_fe.evaluation import SelectionComparator


def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    """Evaluate a model and return metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    return {
        'model': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba),
    }


def main():
    """Complete ML workflow with feature engineering."""
    
    print("=" * 70)
    print("Complete ML Workflow with AutoML Feature Engineering")
    print("=" * 70)
    
    # =========================================
    # Step 1: Load and Prepare Data
    # =========================================
    print("\n" + "=" * 50)
    print("STEP 1: Data Loading and Exploration")
    print("=" * 50)
    
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    
    print(f"\nDataset: Breast Cancer Wisconsin")
    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Target distribution: {dict(y.value_counts())}")
    print(f"\nFeature statistics:")
    print(X.describe().T[['mean', 'std', 'min', 'max']].head(10))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain/Test split: {len(X_train)}/{len(X_test)}")
    
    # =========================================
    # Step 2: Baseline (No Feature Selection)
    # =========================================
    print("\n" + "=" * 50)
    print("STEP 2: Baseline Performance (All Features)")
    print("=" * 50)
    
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
    }
    
    # Scale data for baseline
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    baseline_results = []
    for name, model in models.items():
        result = evaluate_model(
            model, X_train_scaled, X_test_scaled, y_train, y_test, name
        )
        result['n_features'] = X_train.shape[1]
        baseline_results.append(result)
    
    baseline_df = pd.DataFrame(baseline_results)
    print("\nBaseline Results:")
    print(baseline_df.to_string(index=False))
    
    # =========================================
    # Step 3: Feature Selection Comparison
    # =========================================
    print("\n" + "=" * 50)
    print("STEP 3: Feature Selection Method Comparison")
    print("=" * 50)
    
    comparator = SelectionComparator(
        methods=[
            {'name': 'mutual_info', 'method': 'filter', 'params': {'method': 'mutual_info'}},
            {'name': 'f_statistic', 'method': 'filter', 'params': {'method': 'f_statistic'}},
            {'name': 'mrmr', 'method': 'mrmr', 'params': {'alpha': 1.0}},
            {'name': 'random_forest', 'method': 'embedded', 'params': {'method': 'random_forest'}},
            {'name': 'lasso', 'method': 'embedded', 'params': {'method': 'lasso'}},
        ],
        n_features=10,
        cv=5,
        random_state=42,
        verbose=1
    )
    
    comparison_results = comparator.compare(X_train, y_train)
    
    print("\nSelection Method Comparison:")
    summary = comparator.get_comparison_summary()
    print(summary[['method', 'n_features', 'accuracy_mean', 'accuracy_std', 'stability']].to_string(index=False))
    
    print(f"\nBest selection method: {comparator.best_method_}")
    
    # Common features across methods
    common_features = comparator.get_common_features(min_methods=3)
    print(f"Features selected by 3+ methods ({len(common_features)}): {common_features[:5]}...")
    
    # =========================================
    # Step 4: Apply Best Feature Selection
    # =========================================
    print("\n" + "=" * 50)
    print("STEP 4: Apply Feature Engineering Pipeline")
    print("=" * 50)
    
    fe = FeatureEngineering(
        selection_method=comparator.best_method_.split('_')[0] if '_' in comparator.best_method_ else comparator.best_method_,
        n_features=10,
        preprocessing=True,
        scaling='standard',
        random_state=42,
        verbose=1
    )
    
    X_train_selected = fe.fit_transform(X_train, y_train)
    X_test_selected = fe.transform(X_test)
    
    print(f"\nSelected {len(fe.selected_features_)} features:")
    importance_df = fe.get_feature_importances()
    print(importance_df.head(10).to_string())
    
    # =========================================
    # Step 5: Evaluate with Selected Features
    # =========================================
    print("\n" + "=" * 50)
    print("STEP 5: Model Evaluation with Selected Features")
    print("=" * 50)
    
    selected_results = []
    for name, model in models.items():
        result = evaluate_model(
            model, X_train_selected, X_test_selected, y_train, y_test, name
        )
        result['n_features'] = X_train_selected.shape[1]
        selected_results.append(result)
    
    selected_df = pd.DataFrame(selected_results)
    print("\nResults with Selected Features:")
    print(selected_df.to_string(index=False))
    
    # =========================================
    # Step 6: Compare Baseline vs Selected
    # =========================================
    print("\n" + "=" * 50)
    print("STEP 6: Comparison Summary")
    print("=" * 50)
    
    print("\nBaseline (30 features) vs Selected (10 features):")
    print("-" * 60)
    
    for i, model_name in enumerate(models.keys()):
        base_acc = baseline_df.iloc[i]['accuracy']
        sel_acc = selected_df.iloc[i]['accuracy']
        diff = sel_acc - base_acc
        
        base_auc = baseline_df.iloc[i]['auc']
        sel_auc = selected_df.iloc[i]['auc']
        auc_diff = sel_auc - base_auc
        
        print(f"{model_name}:")
        print(f"  Accuracy: {base_acc:.4f} -> {sel_acc:.4f} ({diff:+.4f})")
        print(f"  AUC:      {base_auc:.4f} -> {sel_auc:.4f} ({auc_diff:+.4f})")
    
    # Best model
    best_idx = selected_df['f1'].idxmax()
    best_model_name = selected_df.loc[best_idx, 'model']
    
    print(f"\nBest model: {best_model_name}")
    print(f"  Accuracy: {selected_df.loc[best_idx, 'accuracy']:.4f}")
    print(f"  F1 Score: {selected_df.loc[best_idx, 'f1']:.4f}")
    print(f"  AUC: {selected_df.loc[best_idx, 'auc']:.4f}")
    
    # Feature reduction
    reduction = (1 - 10/30) * 100
    print(f"\nFeature reduction: {reduction:.0f}% (30 -> 10 features)")
    
    # =========================================
    # Step 7: Generate Report
    # =========================================
    print("\n" + "=" * 50)
    print("STEP 7: Feature Engineering Report")
    print("=" * 50)
    
    report = fe.feature_engineering_report()
    print("\nPipeline Report:")
    for key, value in report.items():
        if key != 'top_features':
            print(f"  {key}: {value}")
    
    print("\nTop 5 features by importance:")
    for i, feat in enumerate(report.get('top_features', [])[:5], 1):
        print(f"  {i}. {feat['feature']}: {feat['importance']:.4f}")
    
    # Save pipeline
    fe.save('production_pipeline.pkl')
    print("\nPipeline saved to 'production_pipeline.pkl'")
    
    print("\n" + "=" * 70)
    print("Workflow completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
