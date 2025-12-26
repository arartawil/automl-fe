# AutoML-FE v1.0.1 New Features Example
"""
Demonstration of new features added in version 1.0.1:
- Data Quality Assessment
- Pipeline Export/Import
- Enhanced Categorical Encoding
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Import new v1.0.1 features
from automl_fe import (
    DataQualityChecker,
    FeaturePipelineManager,
    TargetEncoder,
    FrequencyEncoder,
    ComprehensiveCategoricalEncoder,
    generate_quality_report_summary
)

def demonstrate_v101_features():
    """Demonstrate new features in AutoML-FE v1.0.1"""
    
    print("=== AutoML-FE v1.0.1 New Features Demo ===\n")
    
    # Create sample dataset with various data quality issues
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                             n_redundant=2, n_clusters_per_class=1, random_state=42)
    
    # Convert to DataFrame and add categorical features
    feature_names = [f'numeric_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    
    # Add categorical features
    df['category_high_card'] = np.random.choice([f'cat_{i}' for i in range(50)], size=len(df))
    df['category_low_card'] = np.random.choice(['A', 'B', 'C'], size=len(df))
    df['category_medium_card'] = np.random.choice([f'type_{i}' for i in range(8)], size=len(df))
    
    # Introduce missing values
    df.loc[np.random.choice(df.index, size=50), 'numeric_0'] = np.nan
    df.loc[np.random.choice(df.index, size=20), 'category_high_card'] = np.nan
    
    # Add some outliers
    df.loc[np.random.choice(df.index, size=10), 'numeric_1'] = df['numeric_1'].std() * 5
    
    target = pd.Series(y, name='target')
    
    print("1. DATA QUALITY ASSESSMENT")
    print("-" * 40)
    
    # Initialize Data Quality Checker
    quality_checker = DataQualityChecker(
        missing_threshold=0.02,
        outlier_method='iqr',
        correlation_threshold=0.9
    )
    
    # Analyze data quality
    quality_report = quality_checker.analyze(df, target)
    
    # Generate human-readable summary
    summary = generate_quality_report_summary(quality_report)
    print(summary)
    print("\n")
    
    print("2. ENHANCED CATEGORICAL ENCODING")
    print("-" * 40)
    
    # Demonstrate different categorical encoders
    
    # Target Encoding (for low-cardinality with target correlation)
    print("Target Encoding Example:")
    target_encoder = TargetEncoder(smoothing=2.0, cv_folds=3)
    target_encoded = target_encoder.fit_transform(df[['category_low_card']], target)
    print(f"Original categories: {df['category_low_card'].unique()}")
    print(f"Encoded values sample: {target_encoded['category_low_card'].head()}")
    print()
    
    # Frequency Encoding (for high-cardinality features)
    print("Frequency Encoding Example:")
    freq_encoder = FrequencyEncoder(normalize=True)
    freq_encoded = freq_encoder.fit_transform(df[['category_high_card']])
    print(f"Original unique count: {df['category_high_card'].nunique()}")
    print(f"Frequency encoded sample: {freq_encoded['category_high_card'].head()}")
    print()
    
    # Comprehensive Automatic Encoder
    print("Comprehensive Auto-Encoder:")
    comprehensive_encoder = ComprehensiveCategoricalEncoder(auto_select=True)
    auto_encoded = comprehensive_encoder.fit_transform(
        df[['category_low_card', 'category_medium_card', 'category_high_card']], 
        target
    )
    print(f"Encoding strategies selected: {comprehensive_encoder.encoding_strategy_}")
    print(f"Output shape: {auto_encoded.shape}")
    print()
    
    print("3. PIPELINE EXPORT/IMPORT")
    print("-" * 40)
    
    # Create a pipeline manager
    pipeline_manager = FeaturePipelineManager(base_dir="demo_pipelines")
    
    # Save the comprehensive encoder
    saved = pipeline_manager.save_pipeline(
        comprehensive_encoder,
        name="categorical_encoder",
        version="1.0",
        description="Comprehensive categorical encoder for demo dataset",
        tags=["categorical", "encoding", "auto"]
    )
    
    if saved:
        print("✓ Pipeline saved successfully!")
        
        # List available pipelines
        pipelines_df = pipeline_manager.list_pipelines()
        if not pipelines_df.empty:
            print("\nAvailable pipelines:")
            print(pipelines_df[['name', 'version', 'description']].to_string(index=False))
        
        # Load the pipeline
        loaded_encoder = pipeline_manager.load_pipeline("categorical_encoder", "1.0")
        if loaded_encoder is not None:
            print("\n✓ Pipeline loaded successfully!")
            
            # Test loaded pipeline on new data
            test_data = df[['category_low_card', 'category_medium_card', 'category_high_card']].head(5)
            test_encoded = loaded_encoder.transform(test_data)
            print(f"Test encoding successful. Output shape: {test_encoded.shape}")
        
    print("\n" + "="*50)
    print("Demo completed! Check out these new v1.0.1 features:")
    print("• Data Quality Assessment with detailed reports")
    print("• Advanced Categorical Encoders (Target, Frequency, Binary)")
    print("• Pipeline Export/Import for reproducibility")
    print("• Automatic encoding method selection")
    print("="*50)


if __name__ == "__main__":
    demonstrate_v101_features()