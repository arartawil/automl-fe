# Preprocessing utilities for AutoML Feature Engineering
"""
Preprocessing module for data cleaning and transformation.

This module provides standalone, reusable functions for common preprocessing
tasks including encoding, imputation, feature generation, scaling, and
outlier handling.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PolynomialFeatures,
)
from sklearn.impute import SimpleImputer
from scipy import stats
import math


def smart_encode_categorical(
    df: pd.DataFrame,
    binary_threshold: int = 2,
    low_cardinality_threshold: int = 10,
    fitted_encoders: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Smart categorical encoding based on cardinality.

    Automatically selects encoding strategy:
    - Binary columns (2 unique values) → Label encoding
    - Low cardinality (≤ threshold) → One-hot encoding
    - High cardinality (> threshold) → Label encoding

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with categorical columns.
    binary_threshold : int, default=2
        Maximum unique values to consider as binary.
    low_cardinality_threshold : int, default=10
        Maximum unique values for one-hot encoding.
    fitted_encoders : dict, optional
        Pre-fitted encoders for transform-only operation.

    Returns
    -------
    tuple
        (encoded_df, encoders_dict)

    Examples
    --------
    >>> df = pd.DataFrame({'color': ['red', 'blue', 'red'], 'size': ['S', 'M', 'L']})
    >>> encoded_df, encoders = smart_encode_categorical(df)
    """
    df_encoded = df.copy()
    encoders = fitted_encoders or {}
    fitting = fitted_encoders is None

    # Identify categorical columns
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in cat_columns:
        n_unique = df[col].nunique()

        if fitting:
            if n_unique <= binary_threshold:
                # Binary encoding
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df[col].astype(str))
                encoders[col] = {'type': 'label', 'encoder': encoder}

            elif n_unique <= low_cardinality_threshold:
                # One-hot encoding
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(df[[col]].astype(str))
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                
                # Add encoded columns
                encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
                df_encoded = pd.concat([df_encoded.drop(columns=[col]), encoded_df], axis=1)
                encoders[col] = {'type': 'onehot', 'encoder': encoder, 'feature_names': feature_names}

            else:
                # High cardinality - label encoding
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df[col].astype(str))
                encoders[col] = {'type': 'label', 'encoder': encoder}
        else:
            # Transform using fitted encoders
            if col in encoders:
                enc_info = encoders[col]
                if enc_info['type'] == 'label':
                    # Handle unseen labels
                    known_classes = set(enc_info['encoder'].classes_)
                    df_encoded[col] = df[col].astype(str).apply(
                        lambda x: x if x in known_classes else enc_info['encoder'].classes_[0]
                    )
                    df_encoded[col] = enc_info['encoder'].transform(df_encoded[col])
                else:
                    encoded = enc_info['encoder'].transform(df[[col]].astype(str))
                    encoded_df = pd.DataFrame(
                        encoded, 
                        columns=enc_info['feature_names'], 
                        index=df.index
                    )
                    df_encoded = pd.concat([df_encoded.drop(columns=[col]), encoded_df], axis=1)

    return df_encoded, encoders


def adaptive_impute(
    df: pd.DataFrame,
    missing_threshold_simple: float = 0.1,
    missing_threshold_iterative: float = 0.5,
    fitted_imputers: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Adaptive missing value imputation based on missing percentage.

    Strategy selection:
    - < 10% missing → Simple imputation (mean/mode)
    - 10-50% missing → Median imputation
    - > 50% missing → Drop column or constant imputation

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with missing values.
    missing_threshold_simple : float, default=0.1
        Threshold for simple imputation (mean/mode).
    missing_threshold_iterative : float, default=0.5
        Threshold above which column may be dropped.
    fitted_imputers : dict, optional
        Pre-fitted imputers for transform-only operation.

    Returns
    -------
    tuple
        (imputed_df, imputers_dict)

    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2, np.nan], 'b': [np.nan, 2, 3]})
    >>> imputed_df, imputers = adaptive_impute(df)
    """
    df_imputed = df.copy()
    imputers = fitted_imputers or {}
    fitting = fitted_imputers is None

    for col in df.columns:
        missing_ratio = df[col].isna().mean()

        if missing_ratio == 0:
            continue

        if fitting:
            if missing_ratio > missing_threshold_iterative:
                # High missing - use median/mode or drop
                if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    imputer = SimpleImputer(strategy='median')
                else:
                    imputer = SimpleImputer(strategy='most_frequent')
                imputers[col] = {'imputer': imputer, 'drop': False}
                
            elif missing_ratio > missing_threshold_simple:
                # Medium missing - median
                if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    imputer = SimpleImputer(strategy='median')
                else:
                    imputer = SimpleImputer(strategy='most_frequent')
                imputers[col] = {'imputer': imputer, 'drop': False}
                
            else:
                # Low missing - mean/mode
                if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    imputer = SimpleImputer(strategy='mean')
                else:
                    imputer = SimpleImputer(strategy='most_frequent')
                imputers[col] = {'imputer': imputer, 'drop': False}

            # Fit and transform
            df_imputed[col] = imputers[col]['imputer'].fit_transform(
                df[[col]]
            ).ravel()
        else:
            # Transform using fitted imputers
            if col in imputers and not imputers[col].get('drop', False):
                df_imputed[col] = imputers[col]['imputer'].transform(
                    df[[col]]
                ).ravel()

    return df_imputed, imputers


def generate_polynomial_features(
    df: pd.DataFrame,
    degree: int = 2,
    interaction_only: bool = False,
    max_features: int = 100,
    fitted_transformer: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Any]:
    """
    Generate polynomial features with explosion prevention.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with numeric features.
    degree : int, default=2
        Polynomial degree.
    interaction_only : bool, default=False
        If True, only interaction features are produced.
    max_features : int, default=100
        Maximum number of output features to prevent explosion.
    fitted_transformer : PolynomialFeatures, optional
        Pre-fitted transformer for transform-only operation.

    Returns
    -------
    tuple
        (polynomial_df, transformer)

    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> poly_df, transformer = generate_polynomial_features(df, degree=2)
    """
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        return df.copy(), None

    # Check if polynomial expansion would exceed max_features
    n_input = len(numeric_cols)
    expected_features = int(math.factorial(n_input + degree) / 
                           (math.factorial(degree) * math.factorial(n_input)))
    
    if expected_features > max_features:
        # Reduce to only most important features or use interaction_only
        warnings.warn(
            f"Polynomial expansion would create {expected_features} features. "
            f"Limiting to interaction_only to prevent feature explosion."
        )
        interaction_only = True

    if fitted_transformer is None:
        transformer = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=False
        )
        poly_features = transformer.fit_transform(df[numeric_cols])
        feature_names = transformer.get_feature_names_out(numeric_cols)
    else:
        transformer = fitted_transformer
        poly_features = transformer.transform(df[numeric_cols])
        feature_names = transformer.get_feature_names_out(numeric_cols)

    # Create DataFrame with polynomial features
    poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
    
    # Add back non-numeric columns
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
    if non_numeric_cols:
        poly_df = pd.concat([df[non_numeric_cols], poly_df], axis=1)

    return poly_df, transformer


def scale_features(
    df: pd.DataFrame,
    method: str = 'standard',
    fitted_scaler: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Any]:
    """
    Scale numeric features using specified method.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    method : str, default='standard'
        Scaling method: 'standard', 'minmax', 'robust'.
    fitted_scaler : Scaler, optional
        Pre-fitted scaler for transform-only operation.

    Returns
    -------
    tuple
        (scaled_df, scaler)

    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
    >>> scaled_df, scaler = scale_features(df, method='standard')
    """
    df_scaled = df.copy()
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        return df_scaled, None

    if fitted_scaler is None:
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        scaler = fitted_scaler
        df_scaled[numeric_cols] = scaler.transform(df[numeric_cols])

    return df_scaled, scaler


def detect_and_handle_outliers(
    df: pd.DataFrame,
    method: str = 'iqr',
    threshold: float = 1.5,
    handling: str = 'clip',
) -> pd.DataFrame:
    """
    Detect and handle outliers in numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    method : str, default='iqr'
        Detection method: 'iqr', 'zscore'.
    threshold : float, default=1.5
        Threshold for outlier detection.
        For IQR: multiplier for IQR range.
        For Z-score: number of standard deviations.
    handling : str, default='clip'
        How to handle outliers: 'clip', 'remove', 'nan'.

    Returns
    -------
    pd.DataFrame
        DataFrame with handled outliers.

    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2, 3, 100, 4, 5]})
    >>> clean_df = detect_and_handle_outliers(df, method='iqr')
    """
    df_clean = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
        elif method == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

        if handling == 'clip':
            df_clean[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        elif handling == 'nan':
            mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            df_clean.loc[mask, col] = np.nan
        elif handling == 'remove':
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df_clean = df_clean[mask]
        else:
            raise ValueError(f"Unknown outlier handling method: {handling}")

    return df_clean


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with optimized dtypes.

    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [1.0, 2.0, 3.0]})
    >>> optimized_df = optimize_dtypes(df)
    """
    df_optimized = df.copy()

    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype

        if col_type in ['int64', 'int32']:
            # Downcast integers
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
        elif col_type in ['float64', 'float32']:
            # Downcast floats
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')

    return df_optimized


def remove_low_variance_features(
    df: pd.DataFrame,
    threshold: float = 0.01,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove features with variance below threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    threshold : float, default=0.01
        Minimum variance threshold.

    Returns
    -------
    tuple
        (filtered_df, removed_columns)

    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 1, 1], 'b': [1, 2, 3]})
    >>> filtered_df, removed = remove_low_variance_features(df)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    variances = df[numeric_cols].var()
    low_var_cols = variances[variances < threshold].index.tolist()
    
    df_filtered = df.drop(columns=low_var_cols)
    
    return df_filtered, low_var_cols


def remove_highly_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.95,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove highly correlated features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    threshold : float, default=0.95
        Correlation threshold above which to remove features.

    Returns
    -------
    tuple
        (filtered_df, removed_columns)

    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 4, 6], 'c': [1, 3, 5]})
    >>> filtered_df, removed = remove_highly_correlated_features(df, threshold=0.9)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return df.copy(), []
    
    corr_matrix = df[numeric_cols].corr().abs()
    
    # Get upper triangle
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find columns with correlation above threshold
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    
    df_filtered = df.drop(columns=to_drop)
    
    return df_filtered, to_drop
