"""
Basic Usage Example for AutoML Feature Engineering

This script demonstrates the basic usage of the automl-fe library
for feature selection and preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from automl_fe import FeatureEngineering


def main():
    """Main function demonstrating basic usage."""
    
    print("=" * 60)
    print("AutoML Feature Engineering - Basic Usage Example")
    print("=" * 60)
    
    # Load dataset
    print("\n1. Loading breast cancer dataset...")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    print(f"   Dataset shape: {X.shape}")
    print(f"   Number of features: {X.shape[1]}")
    print(f"   Number of samples: {X.shape[0]}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create feature engineering pipeline
    print("\n2. Creating feature engineering pipeline...")
    fe = FeatureEngineering(
        selection_method='mrmr',  # Use mRMR for selection
        n_features=10,            # Select top 10 features
        preprocessing=True,       # Apply preprocessing
        scaling='standard',       # Use standard scaling
        verbose=1,
        random_state=42
    )
    
    # Fit and transform
    print("\n3. Fitting and transforming data...")
    X_train_selected = fe.fit_transform(X_train, y_train)
    X_test_selected = fe.transform(X_test)
    
    print(f"\n   Original features: {X_train.shape[1]}")
    print(f"   Selected features: {X_train_selected.shape[1]}")
    
    # Show selected features
    print("\n4. Selected features:")
    for i, feature in enumerate(fe.selected_features_, 1):
        importance = fe.feature_importances_.get(feature, 0)
        print(f"   {i:2d}. {feature}: {importance:.4f}")
    
    # Get feature importance DataFrame
    print("\n5. Feature importance ranking:")
    importance_df = fe.get_feature_importances()
    print(importance_df.head(10).to_string())
    
    # Train model on original features
    print("\n6. Comparing model performance...")
    
    # Original features
    rf_original = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_original.fit(X_train, y_train)
    acc_original = accuracy_score(y_test, rf_original.predict(X_test))
    
    # Selected features
    rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_selected.fit(X_train_selected, y_train)
    acc_selected = accuracy_score(y_test, rf_selected.predict(X_test_selected))
    
    print(f"   Accuracy with all {X_train.shape[1]} features: {acc_original:.4f}")
    print(f"   Accuracy with {X_train_selected.shape[1]} selected features: {acc_selected:.4f}")
    print(f"   Feature reduction: {(1 - X_train_selected.shape[1]/X_train.shape[1])*100:.1f}%")
    
    # Generate report
    print("\n7. Feature engineering report:")
    report = fe.feature_engineering_report()
    for key, value in report.items():
        if key != 'top_features':
            print(f"   {key}: {value}")
    
    # Save pipeline
    print("\n8. Saving pipeline...")
    fe.save('feature_pipeline.pkl')
    print("   Pipeline saved to 'feature_pipeline.pkl'")
    
    # Load and verify
    print("\n9. Loading and verifying pipeline...")
    fe_loaded = FeatureEngineering.load('feature_pipeline.pkl')
    X_test_verify = fe_loaded.transform(X_test)
    assert X_test_verify.equals(X_test_selected), "Loaded pipeline produces different results!"
    print("   Pipeline loaded and verified successfully!")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
