# Data Quality Assessment Module
"""
Data quality assessment and reporting functionality for feature engineering.

This module provides comprehensive data quality analysis including missing value
patterns, outlier detection, data distribution analysis, and feature correlation
assessment.
"""

import warnings
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder


class DataQualityReport:
    """Data quality assessment report container."""
    
    def __init__(self):
        self.missing_analysis = {}
        self.outlier_analysis = {}
        self.distribution_analysis = {}
        self.correlation_analysis = {}
        self.data_types = {}
        self.recommendations = []
        
    def __str__(self):
        report = "=== Data Quality Report ===\n"
        report += f"Missing Data Issues: {len(self.missing_analysis)}\n"
        report += f"Outlier Issues: {len(self.outlier_analysis)}\n"
        report += f"Distribution Issues: {len(self.distribution_analysis)}\n"
        report += f"Correlation Issues: {len(self.correlation_analysis)}\n"
        report += f"Recommendations: {len(self.recommendations)}\n"
        return report


class DataQualityChecker:
    """
    Comprehensive data quality assessment for feature engineering.
    
    Analyzes data quality issues including missing values, outliers,
    data distributions, correlations, and provides recommendations
    for preprocessing steps.
    
    Parameters
    ----------
    missing_threshold : float, default=0.05
        Threshold for flagging high missing value columns.
    outlier_method : str, default='iqr'
        Method for outlier detection: 'iqr', 'zscore', 'isolation'.
    correlation_threshold : float, default=0.95
        Threshold for flagging highly correlated features.
    """
    
    def __init__(
        self,
        missing_threshold: float = 0.05,
        outlier_method: str = 'iqr',
        correlation_threshold: float = 0.95
    ):
        self.missing_threshold = missing_threshold
        self.outlier_method = outlier_method
        self.correlation_threshold = correlation_threshold
        
    def analyze(
        self, 
        X: pd.DataFrame, 
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> DataQualityReport:
        """
        Perform comprehensive data quality analysis.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series or np.ndarray, optional
            Target variable.
            
        Returns
        -------
        DataQualityReport
            Comprehensive quality assessment report.
        """
        report = DataQualityReport()
        
        # Basic data type analysis
        report.data_types = self._analyze_data_types(X)
        
        # Missing value analysis
        report.missing_analysis = self._analyze_missing_values(X)
        
        # Outlier analysis
        report.outlier_analysis = self._analyze_outliers(X)
        
        # Distribution analysis
        report.distribution_analysis = self._analyze_distributions(X)
        
        # Correlation analysis
        report.correlation_analysis = self._analyze_correlations(X)
        
        # Target variable analysis (if provided)
        if y is not None:
            report.target_analysis = self._analyze_target(X, y)
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report, X, y)
        
        return report
    
    def _analyze_data_types(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data types and memory usage."""
        analysis = {
            'numeric_features': list(X.select_dtypes(include=[np.number]).columns),
            'categorical_features': list(X.select_dtypes(include=['object', 'category']).columns),
            'datetime_features': list(X.select_dtypes(include=['datetime64']).columns),
            'memory_usage_mb': X.memory_usage(deep=True).sum() / 1024**2,
            'total_features': len(X.columns),
            'total_rows': len(X)
        }
        
        # Check for potential categorical features stored as numeric
        potential_categorical = []
        for col in analysis['numeric_features']:
            unique_values = X[col].nunique()
            if unique_values < 20 and unique_values < len(X) * 0.05:
                potential_categorical.append(col)
        
        analysis['potential_categorical'] = potential_categorical
        return analysis
    
    def _analyze_missing_values(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing value patterns."""
        missing_info = {}
        
        for col in X.columns:
            missing_count = X[col].isnull().sum()
            missing_percent = missing_count / len(X)
            
            if missing_percent > 0:
                missing_info[col] = {
                    'count': missing_count,
                    'percentage': missing_percent,
                    'is_problematic': missing_percent > self.missing_threshold
                }
        
        # Check for missing value patterns
        missing_patterns = X.isnull().value_counts()
        
        return {
            'column_wise': missing_info,
            'patterns': missing_patterns.head(10).to_dict(),
            'total_missing_combinations': len(missing_patterns)
        }
    
    def _analyze_outliers(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Analyze outliers in numeric features."""
        outlier_info = {}
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if X[col].isnull().all():
                continue
                
            values = X[col].dropna()
            
            if self.outlier_method == 'iqr':
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = values[(values < lower_bound) | (values > upper_bound)]
                
            elif self.outlier_method == 'zscore':
                z_scores = np.abs(stats.zscore(values))
                outliers = values[z_scores > 3]
                
            outlier_info[col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(values),
                'is_problematic': len(outliers) / len(values) > 0.05
            }
            
        return outlier_info
    
    def _analyze_distributions(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature distributions."""
        distribution_info = {}
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if X[col].isnull().all():
                continue
                
            values = X[col].dropna()
            
            # Skewness and kurtosis
            skewness = stats.skew(values)
            kurtosis = stats.kurtosis(values)
            
            # Normality test (Shapiro-Wilk for small samples, Anderson-Darling for larger)
            if len(values) <= 5000:
                _, p_value = stats.shapiro(values[:5000])
            else:
                stat, critical_values, significance_level = stats.anderson(values, dist='norm')
                p_value = 0.05 if stat > critical_values[2] else 0.10
            
            distribution_info[col] = {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'is_normal': p_value > 0.05,
                'is_highly_skewed': abs(skewness) > 2,
                'unique_values': values.nunique(),
                'zero_variance': values.var() == 0
            }
            
        return distribution_info
    
    def _analyze_correlations(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature correlations."""
        numeric_X = X.select_dtypes(include=[np.number])
        
        if len(numeric_X.columns) < 2:
            return {'highly_correlated_pairs': [], 'correlation_matrix_computed': False}
        
        # Compute correlation matrix
        corr_matrix = numeric_X.corr().abs()
        
        # Find highly correlated pairs
        highly_correlated = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > self.correlation_threshold:
                    highly_correlated.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        return {
            'highly_correlated_pairs': highly_correlated,
            'correlation_matrix_computed': True,
            'max_correlation': corr_matrix.values.max() if not corr_matrix.empty else 0
        }
    
    def _analyze_target(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Analyze target variable characteristics."""
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
            
        target_info = {
            'type': 'regression' if y.dtype in ['int64', 'float64'] and y.nunique() > 20 else 'classification',
            'unique_values': y.nunique(),
            'missing_values': y.isnull().sum(),
            'class_imbalance': None
        }
        
        if target_info['type'] == 'classification':
            class_counts = y.value_counts()
            imbalance_ratio = class_counts.min() / class_counts.max()
            target_info['class_imbalance'] = {
                'ratio': imbalance_ratio,
                'is_imbalanced': imbalance_ratio < 0.1,
                'class_distribution': class_counts.to_dict()
            }
            
        return target_info
    
    def _generate_recommendations(
        self, 
        report: DataQualityReport, 
        X: pd.DataFrame, 
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> List[str]:
        """Generate preprocessing recommendations based on analysis."""
        recommendations = []
        
        # Missing value recommendations
        for col, info in report.missing_analysis.get('column_wise', {}).items():
            if info['is_problematic']:
                if info['percentage'] > 0.5:
                    recommendations.append(f"Consider dropping feature '{col}' (>{info['percentage']:.1%} missing)")
                else:
                    recommendations.append(f"Apply imputation to feature '{col}' ({info['percentage']:.1%} missing)")
        
        # Outlier recommendations
        for col, info in report.outlier_analysis.items():
            if info['is_problematic']:
                recommendations.append(f"Consider outlier treatment for '{col}' ({info['percentage']:.1%} outliers)")
        
        # Distribution recommendations
        for col, info in report.distribution_analysis.items():
            if info['is_highly_skewed']:
                recommendations.append(f"Consider log/sqrt transformation for '{col}' (skewness: {info['skewness']:.2f})")
            if info['zero_variance']:
                recommendations.append(f"Consider dropping '{col}' (zero variance)")
        
        # Correlation recommendations
        if report.correlation_analysis['highly_correlated_pairs']:
            recommendations.append("Apply correlation-based feature selection to remove redundant features")
        
        # Data type recommendations
        if report.data_types['potential_categorical']:
            recommendations.append(f"Consider treating as categorical: {report.data_types['potential_categorical']}")
        
        return recommendations


def generate_quality_report_summary(report: DataQualityReport) -> str:
    """Generate a human-readable summary of the quality report."""
    summary = []
    summary.append("=== DATA QUALITY SUMMARY ===\n")
    
    # Basic statistics
    summary.append(f"Total Features: {report.data_types['total_features']}")
    summary.append(f"Total Rows: {report.data_types['total_rows']}")
    summary.append(f"Memory Usage: {report.data_types['memory_usage_mb']:.2f} MB\n")
    
    # Feature types
    summary.append(f"Numeric Features: {len(report.data_types['numeric_features'])}")
    summary.append(f"Categorical Features: {len(report.data_types['categorical_features'])}")
    summary.append(f"DateTime Features: {len(report.data_types['datetime_features'])}\n")
    
    # Issues found
    missing_issues = sum(1 for info in report.missing_analysis.get('column_wise', {}).values() if info['is_problematic'])
    outlier_issues = sum(1 for info in report.outlier_analysis.values() if info['is_problematic'])
    
    summary.append("ISSUES DETECTED:")
    summary.append(f"- High missing values: {missing_issues} features")
    summary.append(f"- Outlier problems: {outlier_issues} features")
    summary.append(f"- High correlations: {len(report.correlation_analysis['highly_correlated_pairs'])} pairs")
    
    # Recommendations
    if report.recommendations:
        summary.append(f"\nRECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations[:5], 1):
            summary.append(f"{i}. {rec}")
        if len(report.recommendations) > 5:
            summary.append(f"... and {len(report.recommendations) - 5} more")
    
    return "\n".join(summary)