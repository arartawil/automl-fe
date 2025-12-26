---
title: 'AutoML-FE: A Comprehensive Feature Engineering and Selection Toolkit for Machine Learning'
tags:
  - Python
  - machine learning
  - feature engineering
  - feature selection
  - automation
  - mRMR
  - preprocessing
authors:
  - name: Arar Al-Tawil
    orcid: 0000-0002-3194-1407
    affiliation: "1"
affiliations:
 - name: Department of Computer Science, Faculty of Information Technology, Applied Science Privat  University, Amman, Jordan
   index: 1

date: 26 December 2025
bibliography: paper.bib
---
# Summary

AutoML-FE is an open-source Python toolkit for automated feature engineering and selection that combines intelligent preprocessing with comprehensive feature selection algorithms. The package implements multiple selection paradigms including filter methods (mutual information, chi-square, variance threshold), wrapper approaches (recursive feature elimination, forward/backward selection), embedded techniques (LASSO, tree-based importance), and the mRMR (Minimum Redundancy Maximum Relevance) algorithm. AutoML-FE provides a unified interface for systematic comparison of feature selection methods with built-in cross-validation, statistical significance testing, and visualization capabilities. The toolkit addresses the critical bottleneck where feature engineering consumes 60-80% of machine learning development time while enabling reproducible research through standardized evaluation protocols.

# Statement of need

Feature selection remains a fundamental challenge in modern machine learning that directly impacts model performance, interpretability, and computational efficiency [@Li2023]. Current tools face significant limitations: existing implementations either provide individual algorithms without comparison frameworks (scikit-learn) [@Pedregosa2011] or offer complete automation at the expense of transparency and control (AutoGluon [@Erickson2020], H2O.ai). The research community particularly lacks tools for systematic comparison of feature selection methods on contemporary high-dimensional datasets.

Recent surveys indicate substantial performance variations across feature selection algorithms depending on dataset characteristics [@Huan2023], but reproducing these comparisons requires significant implementation effort. This implementation gap is especially pronounced for newer algorithms like mRMR and hybrid approaches that combine multiple selection paradigms [@Rajendran2022]. Practitioners often resort to simple baseline methods due to implementation barriers, limiting both research progress and practical applications.

AutoML-FE addresses these challenges by providing: (1) unified implementation of 12+ feature selection algorithms with consistent interfaces, (2) built-in comparative evaluation framework with statistical validation, (3) standardized benchmarking protocols for reproducible research, and (4) intelligent preprocessing that adapts to modern data characteristics including high-cardinality categorical features and complex missing data patterns [@Singh2023].

# Key Features

AutoML-FE implements four core functionality categories optimized for contemporary machine learning workflows:

## Smart Preprocessing

The preprocessing pipeline adapts intelligently to data characteristics. Categorical encoding strategies are automatically selected based on cardinality: binary variables use label encoding, low-cardinality features (≤10 unique values) use one-hot encoding, and high-cardinality features use label encoding to prevent dimensional explosion. Missing value imputation employs percentage-based strategy selection, using simple methods (mean/mode) for low missing rates (<10%) and sophisticated approaches (KNN imputation) for higher missing rates. The system handles mixed data types common in contemporary datasets [@Rahman2024].

## Comprehensive Feature Selection

**Filter Methods**: Mutual information [@Ross2014], chi-square test, F-statistic, variance threshold, and correlation-based filtering provide computationally efficient feature ranking independent of learning algorithms.

**Wrapper Methods**: Recursive Feature Elimination (RFE) with cross-validation, forward stepwise selection, and backward elimination enable feature subset evaluation using actual model performance with early stopping for computational efficiency.

**Embedded Methods**: LASSO regularization, Random Forest importance, and XGBoost importance integrate feature selection within learning algorithms, with automatic regularization parameter tuning.

**Advanced Algorithms**: The mRMR algorithm [@Peng2005] optimizes the trade-off between feature relevance to the target and redundancy among selected features, particularly effective for high-dimensional data. Implementation includes configurable α parameter for relevance/redundancy balance.

## Evaluation and Comparison Framework

The evaluation system applies multiple selection methods to identical datasets with stratified cross-validation, computes statistical significance tests between methods using non-parametric approaches, and analyzes feature selection stability across validation folds. The framework generates comprehensive visualizations including feature importance plots, method comparison charts, selection stability analysis, and performance versus number of features curves.

## Research and Practical Tools

Publication-ready visualizations, comprehensive result logging with parameter tracking, pipeline serialization for reproducibility, and integration with scikit-learn workflows support both research applications and production deployments [@Wang2021].

# Implementation and core method

AutoML-FE is implemented in Python with a modular architecture supporting scikit-learn compatible interfaces. The core `FeatureEngineering` class orchestrates the complete pipeline:
```python
fe = FeatureEngineering()
X_processed = fe.fit_transform(X, y, 
    preprocessing_strategy='smart',
    selection_method='mrmr', 
    k=10)
```

For feature selection comparison, the `SelectionComparator` class enables systematic evaluation:
```python
comparator = SelectionComparator()
results = comparator.compare_methods(
    methods=['mrmr', 'rfe', 'lasso'], 
    X=X, y=y, cv=5
)
```

Core algorithms utilize vectorized operations for computational efficiency. The mRMR implementation uses efficient mutual information estimation with both discrete and continuous feature support. Wrapper methods support parallel processing for large datasets, while memory-efficient processing handles multi-gigabyte datasets through chunked I/O operations.

The evaluation framework employs stratified cross-validation with multiple performance metrics and includes statistical significance testing using the Wilcoxon signed-rank test for robust method comparison. Feature selection stability is quantified using the Jaccard index across validation folds.

# Applications and scope

AutoML-FE enables several critical research and practical applications. Algorithm developers can benchmark new feature selection methods against established baselines using standardized evaluation protocols. Domain researchers can systematically identify optimal feature selection approaches for specific data characteristics, particularly valuable in bioinformatics, finance, and IoT applications where feature characteristics vary significantly [@Kumar2021].

The toolkit supports comparative studies enabling meta-analysis of algorithm performance patterns across diverse datasets. Educational applications include comprehensive feature selection tutorials with real datasets and clear algorithm implementations. Production applications benefit from automated preprocessing with transparent decision logging and integration with existing ML pipelines.

Recent applications include optimization algorithm benchmarking in academic research, where the comparative framework enabled systematic evaluation of novel metaheuristic approaches across multiple problem domains.

# Comparison with existing tools

AutoML-FE occupies a unique position in the feature engineering ecosystem. Compared to scikit-learn, it provides higher-level automation with intelligent defaults while maintaining algorithmic transparency. Unlike comprehensive AutoML platforms (AutoGluon, FLAML), it focuses specifically on feature engineering with deep algorithm coverage rather than end-to-end automation. The integrated comparison framework distinguishes it from standalone implementations by enabling systematic evaluation without substantial custom development [@Chen2022].

Table 1 summarizes key differences with commonly used alternatives:

| Feature | scikit-learn | AutoGluon/H2O | AutoML-FE |
|---------|-------------|---------------|-----------|
| Algorithm Coverage | Individual methods | Black-box selection | 12+ transparent methods |
| Comparison Framework | Manual implementation | None | Built-in with statistics |
| Preprocessing | Manual configuration | Automated but opaque | Smart with clear logic |
| Reproducibility | Manual logging required | Limited transparency | Full parameter tracking |

# Limitations

AutoML-FE focuses on tabular data and does not address feature engineering for text, images, or time series data, which require domain-specific approaches. The current implementation emphasizes feature selection over feature generation, though polynomial feature creation is supported. Very large datasets (>1GB) may require distributed processing not currently supported. The spectral unmixing implementation uses simple subtraction and may be insufficient for complex spectral overlap scenarios.

Performance optimization focuses on single-machine processing; distributed computing support for extremely large datasets is planned for future versions. Users working with specialized data types or requiring custom feature generation may need additional preprocessing tools.

# Availability

Source code, documentation, tutorials, and issue tracking are hosted at: https://github.com/arartawil/automl-fe.git. The software runs on Windows, macOS, and Linux with Python 3.8 or newer. It is released under MIT License, and contributions via pull requests are welcome. Comprehensive examples using public datasets (Titanic, Iris, Wine Quality) are provided for demonstration and testing purposes.

# References