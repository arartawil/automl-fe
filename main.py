#!/usr/bin/env python3
"""
AutoML-FE Demo Script

This script demonstrates the core functionality of AutoML-FE for automated
feature engineering and selection.

Run with: python main.py
"""

import sys
import random
import numpy as np
import pandas as pd
from pathlib import Path

# Add current directory to path for local development
sys.path.insert(0, str(Path(__file__).parent))

from automl_fe import FeatureEngineering
from sklearn.datasets import load_breast_cancer, load_diabetes


def demo_feature_selection():
    """Demonstrate basic feature selection functionality."""
    print("\n🎯 DEMO 1: Feature Selection")
    print("=" * 50)
    
    # Load sample data
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    print(f"📊 Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Create feature engineering pipeline
    fe = FeatureEngineering(
        selection_method='mrmr',
        n_features=10,
        preprocessing=True,
        scaling='standard'
    )
    
    print("🔄 Running feature selection...")
    
    # Fit and transform
    X_selected = fe.fit_transform(X, y)
    
    print(f"✅ Selected {X_selected.shape[1]} features from {X.shape[1]} original features")
    
    # Display results
    if hasattr(fe, 'selected_features_'):
        print(f"\n📈 Selected Features:")
        for i, feature in enumerate(fe.selected_features_[:5]):  # Show first 5
            print(f"   {i+1}. {feature}")
        if len(fe.selected_features_) > 5:
            print(f"   ... and {len(fe.selected_features_) - 5} more")
    
    return fe, X_selected


def demo_method_comparison():
    """Demonstrate comparing different feature selection methods."""
    print("\n🔍 DEMO 2: Method Comparison")
    print("=" * 50)
    
    # Load regression dataset
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    print(f"📊 Loaded regression dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    methods = ['filter', 'mrmr', 'wrapper', 'embedded']
    results = {}
    
    for method in methods:
        print(f"⚡ Testing {method} method...")
        
        try:
            fe = FeatureEngineering(
                selection_method=method,
                n_features=5,
                preprocessing=True
            )
            
            X_selected = fe.fit_transform(X, y)
            results[method] = {
                'success': True,
                'n_features': X_selected.shape[1],
                'selected_features': getattr(fe, 'selected_features_', [])
            }
            print(f"   ✅ Success: {X_selected.shape[1]} features selected")
            
        except Exception as e:
            results[method] = {'success': False, 'error': str(e)}
            print(f"   ❌ Error: {str(e)[:50]}...")
    
    # Summary
    print(f"\n📊 Method Comparison Summary:")
    for method, result in results.items():
        if result['success']:
            print(f"   {method.upper()}: ✅ {result['n_features']} features")
        else:
            print(f"   {method.upper()}: ❌ Failed")
    
    return results


def demo_preprocessing_pipeline():
    """Demonstrate preprocessing capabilities."""
    print("\n🛠️ DEMO 3: Preprocessing Pipeline")
    print("=" * 50)
    
    # Create sample data with missing values and mixed types
    np.random.seed(42)
    n_samples = 100
    
    # Generate mixed data as lists first, then convert to pandas
    numeric_1 = np.random.randn(n_samples).tolist()
    numeric_2 = np.random.randn(n_samples).tolist()
    categorical = np.random.choice(['A', 'B', 'C'], n_samples).tolist()
    with_missing = np.random.randn(n_samples).tolist()
    
    # Add missing values
    missing_idx = np.random.choice(n_samples, 20, replace=False)
    for idx in missing_idx:
        with_missing[idx] = None  # Use None instead of np.nan for missing
    
    # Create DataFrame
    X = pd.DataFrame({
        'numeric_1': numeric_1,
        'numeric_2': numeric_2,
        'categorical': categorical,
        'with_missing': with_missing
    })
    
    y = pd.Series(np.random.randn(n_samples))
    
    print(f"📊 Created sample dataset:")
    print(f"   Shape: {X.shape}")
    print(f"   Missing values: {X.isnull().sum().sum()}")
    print(f"   Categorical columns: {len(X.select_dtypes(include='object').columns)}")
    
    # Apply preprocessing
    fe = FeatureEngineering(
        selection_method='filter',
        n_features=3,
        preprocessing=True,
        handle_missing=True,
        scaling='standard'
    )
    
    print("🔄 Applying preprocessing pipeline...")
    X_processed = fe.fit_transform(X, y)
    
    print(f"✅ Preprocessing completed:")
    print(f"   Final shape: {X_processed.shape}")
    print(f"   Missing values after processing: {pd.DataFrame(X_processed).isnull().sum().sum()}")
    
    return X_processed


def main():
    """Main demo function."""
    print("�📊 AutoML-FE: Automated Machine Learning Feature Engineering")
    print("=" * 60)
    print("This demo showcases the core functionality of AutoML-FE.")
    print("Advanced feature selection and engineering made simple.")
    print()
    
    try:
        # Run demonstrations
        fe, X_selected = demo_feature_selection()
        results = demo_method_comparison()
        X_processed = demo_preprocessing_pipeline()
        
        print("\n🎉 Demo completed successfully!")
        print("\n📝 Key Features Demonstrated:")
        print("   ✅ Multiple feature selection methods (filter, mRMR, wrapper, embedded)")
        print("   ✅ Automatic preprocessing pipeline")
        print("   ✅ Method comparison and evaluation")
        print("   ✅ Handling mixed data types and missing values")
        print("   ✅ Scikit-learn pipeline integration")
        
        print(f"\n💡 Next Steps:")
        print("   • Try with your own dataset")
        print("   • Experiment with different selection methods")
        print("   • Customize preprocessing options")
        print("   • Check out examples/ for more use cases")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure required packages are installed: pip install pandas scikit-learn numpy")
        return 1
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        return 1
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the project root directory.")
        return 1
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())