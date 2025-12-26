# AutoML-FE: Automated Machine Learning Feature Engineering

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Automated Machine Learning Feature Engineering Library**

A comprehensive Python library for automated feature engineering with advanced feature selection methods including filter-based, wrapper-based, embedded, and mRMR (Minimum Redundancy Maximum Relevance) approaches.

## 🚀 Features

- **Multiple Selection Methods**: Filter, Wrapper, Embedded, and mRMR algorithms
- **Smart Preprocessing**: Adaptive encoding, imputation, and scaling
- **Method Comparison**: Statistical comparison of selection methods with cross-validation
- **Visualization**: Publication-ready plots for feature analysis
- **Pipeline Integration**: Compatible with scikit-learn pipelines
- **Stability Analysis**: Selection stability across cross-validation folds
- **Task Detection**: Automatic classification/regression task detection
- **Outlier Handling**: Multiple strategies for outlier detection and treatment

## 📦 Installation

### From Source
```bash
git clone https://github.com/arartawil/automl-fe.git
cd automl-fe
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/arartawil/automl-fe.git
cd automl-fe
pip install -r requirements.txt
```

## ⚡ Quick Start

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from automl_fe import FeatureEngineering

# Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Create and fit feature engineering pipeline
fe = FeatureEngineering(
    selection_method='mrmr',
    n_features=10,
    preprocessing=True,
    scaling='standard'
)

# Transform data
X_selected = fe.fit_transform(X, y)

# View selected features
print(f"Selected {len(fe.selected_features_)} features:")
print(fe.selected_features_)

# Get feature importances
print(fe.get_feature_importances())
```

## 📊 Feature Selection Methods

| Method | Type | Description | Best For |
|--------|------|-------------|----------|
| `filter` | Filter | Statistical measures (MI, chi2, F-stat) | Fast initial screening |
| `mrmr` | Filter | Minimum Redundancy Maximum Relevance | Reducing redundancy |
| `wrapper` | Wrapper | RFE, Forward/Backward selection | Small-medium datasets |
| `embedded` | Embedded | LASSO, RF, XGBoost importance | Built-in regularization |

### Filter Methods
```python
from automl_fe.selection import FilterSelector

selector = FilterSelector(
    method='mutual_info',  # 'chi2', 'f_statistic', 'variance', 'correlation', 'relief'
    n_features=10,
    task='classification'
)
X_selected = selector.fit_transform(X, y)
```

### mRMR Selection
```python
from automl_fe.selection import mRMRSelector

selector = mRMRSelector(
    n_features=10,
    alpha=1.0  # Trade-off between relevance and redundancy
)
X_selected = selector.fit_transform(X, y)
```

### Wrapper Methods
```python
from automl_fe.selection import WrapperSelector

selector = WrapperSelector(
    method='rfe',  # 'forward', 'backward', 'rfe', 'sequential'
    n_features=10,
    estimator=None  # Auto-selected based on task
)
X_selected = selector.fit_transform(X, y)
```

### Embedded Methods
```python
from automl_fe.selection import EmbeddedSelector

selector = EmbeddedSelector(
    method='random_forest',  # 'lasso', 'ridge', 'elasticnet', 'xgboost'
    n_features=10
)
X_selected = selector.fit_transform(X, y)
```

## 🔍 Advanced Usage

### Automatic Method Recommendation
```python
# Let the library choose the best method
fe = FeatureEngineering(selection_method='auto')
X_selected = fe.fit_transform(X, y)

# View recommended method
print(f"Recommended method: {fe.recommended_method_}")
```

### Method Comparison
```python
from automl_fe.evaluation import compare_selection_methods

results = compare_selection_methods(
    X, y,
    methods=['filter', 'mrmr', 'wrapper', 'embedded'],
    n_features=10,
    cv=5
)

print(results)
```

### Feature Engineering with Polynomial Features
```python
fe = FeatureEngineering(
    selection_method='mrmr',
    n_features=15,
    polynomial_features=True,
    polynomial_degree=2
)
X_transformed = fe.fit_transform(X, y)
```

### Pipeline Integration
```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('feature_engineering', FeatureEngineering(
        selection_method='mrmr',
        n_features=10,
        preprocessing=True
    )),
    ('classifier', RandomForestClassifier())
])

pipeline.fit(X, y)
predictions = pipeline.predict(X_test)
```

## 📈 Visualization

```python
from automl_fe import FeatureEngineering
from automl_fe.visualization import (
    plot_feature_importance,
    plot_selection_stability,
    plot_method_comparison
)

# Fit feature engineering
fe = FeatureEngineering(selection_method='mrmr', n_features=10)
fe.fit(X, y)

# Plot feature importance
plot_feature_importance(fe, top_n=15)

# Compare methods visually
from automl_fe.evaluation import compare_selection_methods
results = compare_selection_methods(X, y)
plot_method_comparison(results)
```

## 🛠️ Preprocessing Options

The library provides automatic preprocessing with the following options:

- **Categorical Encoding**: Label encoding, one-hot encoding, target encoding
- **Missing Value Imputation**: Mean, median, mode, KNN, iterative
- **Scaling**: Standard, MinMax, Robust, MaxAbs
- **Outlier Detection**: IQR, Z-score, Isolation Forest
- **Type Optimization**: Automatic dtype optimization for memory efficiency

```python
fe = FeatureEngineering(
    selection_method='mrmr',
    preprocessing=True,
    handle_missing=True,
    scaling='standard',
    handle_outliers=True
)
```

## 📝 Examples

Check out the `examples/` directory for more detailed examples:

- `basic_usage.py` - Simple feature selection examples
- `advanced_selection.py` - Advanced selection techniques
- `real_world_example.py` - Complete ML pipeline example
- `visualization_examples.py` - Visualization examples
- `user_dataset_example.py` - Using your own dataset

## 🧪 Running Tests

```bash
# Install dev dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=automl_fe --cov-report=html
```

## 🧪 Running the Demo

```bash
python main.py
```

This will run demonstrations of:
- Feature selection on real datasets
- Comparison of different selection methods  
- Preprocessing pipeline capabilities

## 📚 Documentation

For full documentation, visit the docs/ directory or check out the well-documented source code.

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) before submitting pull requests.

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by various AutoML and feature engineering frameworks
- Built on top of scikit-learn, pandas, and other excellent libraries
- Thanks to all contributors and users of this library

## 📧 Contact

- **Author**: Your Name
- **GitHub**: [@arartawil](https://github.com/arartawil)
- **Issues**: [GitHub Issues](https://github.com/arartawil/automl-fe/issues)

## ⭐ Star History

If you find this project helpful, please consider giving it a star on GitHub!

---

**Made with ❤️ for the ML community**
