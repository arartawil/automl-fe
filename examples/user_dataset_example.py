"""
Example: Feature Selection on User-Uploaded Dataset

This example shows how to use automl-fe when a user uploads
their own CSV dataset (like Iris) and wants to perform feature selection.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Import automl-fe
from automl_fe import FeatureEngineering, SelectionComparator


def load_user_dataset(file_path: str, target_column: str):
    """
    Load a CSV file uploaded by the user.
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file
    target_column : str
        Name of the target/label column
    
    Returns
    -------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable
    """
    df = pd.read_csv(file_path)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {list(X.columns)}")
    print(f"Target: {target_column} ({y.nunique()} classes)")
    
    return X, y


def feature_selection_workflow(X, y, n_features=None, method='mrmr'):
    """
    Perform feature selection on the dataset.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input features
    y : pd.Series
        Target variable
    n_features : int, optional
        Number of features to select. If None, selects half.
    method : str
        Selection method: 'filter', 'mrmr', 'embedded', 'wrapper'
    
    Returns
    -------
    fe : FeatureEngineering
        Fitted feature engineering pipeline
    X_selected : pd.DataFrame
        Dataset with selected features
    """
    # Default: select half of the features
    if n_features is None:
        n_features = max(1, X.shape[1] // 2)
    
    print(f"\n{'='*50}")
    print(f"Feature Selection using '{method}' method")
    print(f"Selecting {n_features} features from {X.shape[1]}")
    print(f"{'='*50}")
    
    # Create feature engineering pipeline
    fe = FeatureEngineering(
        selection_method=method,
        n_features=n_features,
        preprocessing=True,
        scaling='standard',
        random_state=42
    )
    
    # Fit and transform
    X_selected = fe.fit_transform(X, y)
    
    # Show results
    print(f"\n✓ Selected Features ({len(fe.selected_features_)}):")
    for i, feat in enumerate(fe.selected_features_, 1):
        importance = fe.feature_importances_.get(feat, 0)
        print(f"  {i}. {feat} (importance: {importance:.4f})")
    
    return fe, X_selected


def compare_methods(X, y, n_features=None):
    """
    Compare different feature selection methods.
    """
    if n_features is None:
        n_features = max(1, X.shape[1] // 2)
    
    print(f"\n{'='*50}")
    print("Comparing Feature Selection Methods")
    print(f"{'='*50}")
    
    comparator = SelectionComparator(
        methods=['filter', 'mrmr', 'embedded'],
        n_features=n_features,
        cv=5,
        random_state=42,
        verbose=1
    )
    
    results = comparator.compare(X, y)
    
    # Print comparison - results is a DataFrame
    print("\nResults:")
    print("-" * 40)
    for _, row in results.iterrows():
        method = row['method']
        score = row.get('accuracy_mean', row.get('r2_mean', 0))
        std = row.get('accuracy_std', row.get('r2_std', 0))
        print(f"{method:12} | Score: {score:.4f} ± {std:.4f}")
    
    print(f"\n✓ Best method: {comparator.best_method_}")
    print(f"✓ Common features (selected by 2+ methods):")
    common = comparator.get_common_features(min_methods=2)
    for feat in common:
        print(f"  - {feat}")
    
    return comparator, results


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train a model and evaluate on test set."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy, model


# =============================================================================
# MAIN EXAMPLE
# =============================================================================

if __name__ == "__main__":
    
    # -------------------------------------------------------------------------
    # OPTION 1: Load from CSV file (user-uploaded dataset)
    # -------------------------------------------------------------------------
    # Uncomment and modify the path to your CSV file:
    # 
    # X, y = load_user_dataset(
    #     file_path="path/to/your/dataset.csv",
    #     target_column="target"  # Name of your target column
    # )
    
    # -------------------------------------------------------------------------
    # OPTION 2: Use sklearn's Iris dataset as example
    # -------------------------------------------------------------------------
    from sklearn.datasets import load_iris
    
    print("Loading Iris dataset...")
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='species')
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {list(X.columns)}")
    print(f"Target classes: {y.nunique()}")
    
    # -------------------------------------------------------------------------
    # STEP 1: Split data
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)} samples | Test: {len(X_test)} samples")
    
    # -------------------------------------------------------------------------
    # STEP 2: Feature Selection (select 2 features from 4)
    # -------------------------------------------------------------------------
    fe, X_train_selected = feature_selection_workflow(
        X_train, y_train, 
        n_features=2,      # Select 2 best features
        method='mrmr'      # Use mRMR algorithm
    )
    
    # Transform test set using the fitted pipeline
    X_test_selected = fe.transform(X_test)
    
    # -------------------------------------------------------------------------
    # STEP 3: Train and evaluate
    # -------------------------------------------------------------------------
    print(f"\n{'='*50}")
    print("Model Training & Evaluation")
    print(f"{'='*50}")
    
    # With all features
    acc_all, _ = train_and_evaluate(X_train, X_test, y_train, y_test)
    print(f"Accuracy with ALL {X.shape[1]} features: {acc_all:.4f}")
    
    # With selected features
    acc_selected, _ = train_and_evaluate(
        X_train_selected, X_test_selected, y_train, y_test
    )
    print(f"Accuracy with {len(fe.selected_features_)} selected features: {acc_selected:.4f}")
    
    # -------------------------------------------------------------------------
    # STEP 4: Compare different selection methods (optional)
    # -------------------------------------------------------------------------
    print("\n")
    comparator, results = compare_methods(X_train, y_train, n_features=2)
    
    # -------------------------------------------------------------------------
    # STEP 5: Save the pipeline for later use (optional)
    # -------------------------------------------------------------------------
    fe.save("feature_pipeline.pkl")
    print("\n✓ Pipeline saved to 'feature_pipeline.pkl'")
    
    # To load later:
    # fe_loaded = FeatureEngineering.load("feature_pipeline.pkl")
    # X_new_selected = fe_loaded.transform(X_new)
