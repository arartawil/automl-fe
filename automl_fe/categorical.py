# Enhanced Categorical Encoding Module
"""
Advanced categorical encoding techniques for feature engineering.

This module provides comprehensive categorical encoding methods including
target encoding, frequency encoding, binary encoding, and other modern
techniques optimized for different use cases.
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import warnings


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoding with cross-validation and smoothing.
    
    Encodes categorical features using target statistics with proper
    cross-validation to prevent overfitting and optional smoothing.
    
    Parameters
    ----------
    smoothing : float, default=1.0
        Smoothing parameter for regularization.
    cv_folds : int, default=5
        Number of cross-validation folds.
    noise_level : float, default=0.01
        Level of noise to add to prevent overfitting.
    handle_unknown : str, default='mean'
        How to handle unknown categories: 'mean', 'ignore'.
    """
    
    def __init__(
        self,
        smoothing: float = 1.0,
        cv_folds: int = 5,
        noise_level: float = 0.01,
        handle_unknown: str = 'mean'
    ):
        self.smoothing = smoothing
        self.cv_folds = cv_folds
        self.noise_level = noise_level
        self.handle_unknown = handle_unknown
        self.encodings_ = {}
        self.global_mean_ = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'TargetEncoder':
        """
        Fit the target encoder.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : pd.Series
            Target variable.
            
        Returns
        -------
        self : TargetEncoder
            Fitted encoder.
        """
        self.global_mean_ = y.mean()
        self.encodings_ = {}
        
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            # Calculate encoding with cross-validation
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            col_encodings = []
            
            for train_idx, val_idx in kf.split(X):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                
                # Calculate smoothed target encoding
                stats = X_train.groupby(col)[y_train.name].agg(['mean', 'count'])
                stats['smoothed'] = (
                    (stats['count'] * stats['mean'] + self.smoothing * self.global_mean_) /
                    (stats['count'] + self.smoothing)
                )
                
                col_encodings.append(stats['smoothed'].to_dict())
            
            # Average encodings across folds
            self.encodings_[col] = self._average_encodings(col_encodings)
            
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical features using target encoding.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features to transform.
            
        Returns
        -------
        pd.DataFrame
            Transformed features.
        """
        X_transformed = X.copy()
        
        for col, encoding_dict in self.encodings_.items():
            if col in X_transformed.columns:
                # Apply encoding
                encoded_col = X_transformed[col].map(encoding_dict)
                
                # Handle unknown categories
                if self.handle_unknown == 'mean':
                    encoded_col = encoded_col.fillna(self.global_mean_)
                elif self.handle_unknown == 'ignore':
                    encoded_col = encoded_col.fillna(0)
                
                # Add noise to prevent overfitting
                if self.noise_level > 0:
                    noise = np.random.normal(0, self.noise_level, size=len(encoded_col))
                    encoded_col += noise
                
                X_transformed[col] = encoded_col
                
        return X_transformed
    
    def _average_encodings(self, encoding_list: List[Dict]) -> Dict:
        """Average encodings across cross-validation folds."""
        all_keys = set()
        for enc in encoding_list:
            all_keys.update(enc.keys())
            
        averaged = {}
        for key in all_keys:
            values = [enc.get(key, self.global_mean_) for enc in encoding_list]
            averaged[key] = np.mean(values)
            
        return averaged


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Frequency encoding for categorical features.
    
    Replaces categorical values with their frequency of occurrence
    in the training set.
    
    Parameters
    ----------
    handle_unknown : str, default='zero'
        How to handle unknown categories: 'zero', 'min'.
    normalize : bool, default=False
        Whether to normalize frequencies to [0, 1] range.
    """
    
    def __init__(self, handle_unknown: str = 'zero', normalize: bool = False):
        self.handle_unknown = handle_unknown
        self.normalize = normalize
        self.frequency_maps_ = {}
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FrequencyEncoder':
        """
        Fit the frequency encoder.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : pd.Series, optional
            Target variable (ignored).
            
        Returns
        -------
        self : FrequencyEncoder
            Fitted encoder.
        """
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            freq_map = X[col].value_counts().to_dict()
            
            if self.normalize:
                total_count = len(X)
                freq_map = {k: v / total_count for k, v in freq_map.items()}
                
            self.frequency_maps_[col] = freq_map
            
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical features using frequency encoding.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features to transform.
            
        Returns
        -------
        pd.DataFrame
            Transformed features.
        """
        X_transformed = X.copy()
        
        for col, freq_map in self.frequency_maps_.items():
            if col in X_transformed.columns:
                # Apply frequency encoding
                encoded_col = X_transformed[col].map(freq_map)
                
                # Handle unknown categories
                if self.handle_unknown == 'zero':
                    encoded_col = encoded_col.fillna(0)
                elif self.handle_unknown == 'min':
                    min_freq = min(freq_map.values()) if freq_map else 0
                    encoded_col = encoded_col.fillna(min_freq)
                
                X_transformed[col] = encoded_col
                
        return X_transformed


class BinaryEncoder(BaseEstimator, TransformerMixin):
    """
    Binary encoding for categorical features.
    
    Converts categorical features to binary representation,
    reducing dimensionality compared to one-hot encoding.
    
    Parameters
    ----------
    drop_first : bool, default=False
        Whether to drop the first binary column to avoid collinearity.
    """
    
    def __init__(self, drop_first: bool = False):
        self.drop_first = drop_first
        self.binary_maps_ = {}
        self.feature_names_ = []
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BinaryEncoder':
        """
        Fit the binary encoder.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : pd.Series, optional
            Target variable (ignored).
            
        Returns
        -------
        self : BinaryEncoder
            Fitted encoder.
        """
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            # Get unique categories and assign integer codes
            unique_cats = X[col].dropna().unique()
            n_categories = len(unique_cats)
            
            # Calculate number of binary digits needed
            n_bits = max(1, int(np.ceil(np.log2(n_categories))))
            
            # Create binary mapping
            binary_map = {}
            for i, cat in enumerate(unique_cats):
                binary_repr = format(i, f'0{n_bits}b')
                binary_map[cat] = [int(bit) for bit in binary_repr]
            
            self.binary_maps_[col] = {
                'mapping': binary_map,
                'n_bits': n_bits
            }
            
            # Generate feature names
            start_idx = 1 if self.drop_first else 0
            for bit_idx in range(start_idx, n_bits):
                self.feature_names_.append(f"{col}_binary_{bit_idx}")
                
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical features using binary encoding.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features to transform.
            
        Returns
        -------
        pd.DataFrame
            Transformed features with binary encoded columns.
        """
        X_transformed = X.copy()
        binary_columns = []
        
        for col, binary_info in self.binary_maps_.items():
            if col in X_transformed.columns:
                mapping = binary_info['mapping']
                n_bits = binary_info['n_bits']
                
                # Create binary columns
                start_idx = 1 if self.drop_first else 0
                for bit_idx in range(start_idx, n_bits):
                    col_name = f"{col}_binary_{bit_idx}"
                    
                    # Extract bit values
                    bit_values = X_transformed[col].map(
                        lambda x: mapping.get(x, [0] * n_bits)[bit_idx] if pd.notna(x) else 0
                    )
                    
                    binary_columns.append(pd.Series(bit_values, name=col_name, index=X_transformed.index))
                
                # Remove original categorical column
                X_transformed = X_transformed.drop(columns=[col])
        
        # Add binary columns
        if binary_columns:
            binary_df = pd.concat(binary_columns, axis=1)
            X_transformed = pd.concat([X_transformed, binary_df], axis=1)
            
        return X_transformed


class ComprehensiveCategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Comprehensive categorical encoder with automatic method selection.
    
    Automatically selects the best encoding method for each categorical
    feature based on cardinality and target correlation.
    
    Parameters
    ----------
    target_encoding_threshold : int, default=10
        Threshold for using target encoding vs frequency encoding.
    binary_encoding_threshold : int, default=20
        Threshold for using binary encoding vs one-hot encoding.
    auto_select : bool, default=True
        Whether to automatically select encoding methods.
    """
    
    def __init__(
        self,
        target_encoding_threshold: int = 10,
        binary_encoding_threshold: int = 20,
        auto_select: bool = True
    ):
        self.target_encoding_threshold = target_encoding_threshold
        self.binary_encoding_threshold = binary_encoding_threshold
        self.auto_select = auto_select
        self.encoding_strategy_ = {}
        self.encoders_ = {}
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'ComprehensiveCategoricalEncoder':
        """
        Fit the comprehensive categorical encoder.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : pd.Series, optional
            Target variable.
            
        Returns
        -------
        self : ComprehensiveCategoricalEncoder
            Fitted encoder.
        """
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            n_unique = X[col].nunique()
            
            if self.auto_select:
                # Automatically select encoding method
                if y is not None and n_unique <= self.target_encoding_threshold:
                    strategy = 'target'
                elif n_unique <= 5:
                    strategy = 'onehot'
                elif n_unique <= self.binary_encoding_threshold:
                    strategy = 'binary'
                else:
                    strategy = 'frequency'
            else:
                # Default strategy
                strategy = 'frequency'
            
            self.encoding_strategy_[col] = strategy
            
            # Fit appropriate encoder
            if strategy == 'target' and y is not None:
                encoder = TargetEncoder()
                encoder.fit(X[[col]], y)
            elif strategy == 'frequency':
                encoder = FrequencyEncoder()
                encoder.fit(X[[col]])
            elif strategy == 'binary':
                encoder = BinaryEncoder()
                encoder.fit(X[[col]])
            elif strategy == 'onehot':
                from sklearn.preprocessing import OneHotEncoder
                encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                encoder.fit(X[[col]])
            
            self.encoders_[col] = encoder
            
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical features using selected encoding methods.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features to transform.
            
        Returns
        -------
        pd.DataFrame
            Transformed features.
        """
        X_transformed = X.copy()
        
        for col, encoder in self.encoders_.items():
            if col in X_transformed.columns:
                if self.encoding_strategy_[col] == 'onehot':
                    # Handle one-hot encoding separately
                    encoded = encoder.transform(X_transformed[[col]])
                    feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                    encoded_df = pd.DataFrame(
                        encoded, 
                        columns=feature_names, 
                        index=X_transformed.index
                    )
                    X_transformed = X_transformed.drop(columns=[col])
                    X_transformed = pd.concat([X_transformed, encoded_df], axis=1)
                else:
                    # For other encoders
                    encoded_col = encoder.transform(X_transformed[[col]])
                    if isinstance(encoded_col, pd.DataFrame):
                        if col in encoded_col.columns:
                            X_transformed[col] = encoded_col[col]
                        else:
                            # For binary encoding, replace original column with new ones
                            X_transformed = X_transformed.drop(columns=[col])
                            X_transformed = pd.concat([X_transformed, encoded_col], axis=1)
                    
        return X_transformed
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Get output feature names for transformation."""
        if input_features is None:
            return []
            
        output_features = []
        
        for feature in input_features:
            if feature in self.encoding_strategy_:
                strategy = self.encoding_strategy_[feature]
                if strategy == 'onehot':
                    encoder = self.encoders_[feature]
                    for cat in encoder.categories_[0]:
                        output_features.append(f"{feature}_{cat}")
                elif strategy == 'binary':
                    encoder = self.encoders_[feature]
                    n_bits = encoder.binary_maps_[feature]['n_bits']
                    start_idx = 1 if encoder.drop_first else 0
                    for bit_idx in range(start_idx, n_bits):
                        output_features.append(f"{feature}_binary_{bit_idx}")
                else:
                    output_features.append(feature)
            else:
                output_features.append(feature)
                
        return output_features