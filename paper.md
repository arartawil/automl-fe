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

AutoML-FE is a comprehensive Python toolkit that automates feature engineering and selection by combining intelligent preprocessing with advanced selection algorithms. The package implements multiple feature selection paradigms including filter methods (mutual information, chi-square), wrapper approaches (recursive feature elimination), embedded techniques (LASSO, tree-based importance), and the mRMR (Minimum Redundancy Maximum Relevance) algorithm. AutoML-FE provides a unified interface for systematic comparison of feature selection methods with built-in cross-validation, statistical significance testing, and visualization capabilities, addressing the critical bottleneck in machine learning workflows where feature engineering remains a time-consuming manual process [@Zhang2024].

The toolkit differentiates itself from existing solutions by combining automated preprocessing with transparent, comparable feature selection methods. Unlike black-box AutoML platforms that sacrifice interpretability for automation [@Zoph2020], AutoML-FE maintains researcher control while significantly reducing implementation overhead. The package enables reproducible research through standardized evaluation protocols and supports both practical ML workflows and academic research requiring rigorous algorithm comparison.

# Statement of Need

Feature selection remains a critical challenge in modern machine learning, particularly with the exponential growth in data dimensionality and the increasing complexity of deep learning models [@Li2023]. Recent surveys indicate that feature engineering and selection continue to consume 60-80% of machine learning development time, despite advances in automated tools [@Chen2022]. The proliferation of features in contemporary datasets, especially in domains like genomics, finance, and IoT, has made effective feature selection more crucial than ever [@Kumar2021].

Current tools face several limitations that AutoML-FE addresses. Existing AutoML frameworks like AutoGluon [@Erickson2020] and FLAML [@Wang2021] focus primarily on hyperparameter optimization and model selection, with limited emphasis on feature selection methodologies. While these tools provide end-to-end automation, they often lack transparency in feature selection decisions, making it difficult for researchers to understand and validate the chosen features [@Truong2021].

The research community particularly lacks modern tools for systematic comparison of feature selection methods on contemporary datasets. Recent comparative studies [@Huan2023] highlight significant performance variations across different feature selection algorithms depending on dataset characteristics, but reproducing these comparisons requires substantial implementation effort. This implementation gap is especially pronounced for newer algorithms and hybrid approaches that combine multiple selection paradigms [@Rajendran2022].

AutoML-FE addresses these modern challenges by providing:

1. **Contemporary Algorithm Coverage**: Implementation of recent advances including hybrid selection methods and deep learning-compatible approaches
2. **Scalable Architecture**: Efficient handling of high-dimensional modern datasets
3. **Transparent Automation**: Interpretable feature selection with clear decision rationales
4. **Research Reproducibility**: Standardized benchmarking protocols aligned with current best practices

# Key Features

## Smart Preprocessing

AutoML-FE implements intelligent preprocessing adapted for modern data challenges. The preprocessing pipeline handles mixed data types common in contemporary datasets, including high-cardinality categorical features typical in recommendation systems and NLP applications [@Singh2023]. Missing value imputation strategies are optimized for different missing data mechanisms, incorporating recent advances in imputation theory for machine learning contexts [@Rahman2024].

## Comprehensive Feature Selection

The package implements both classical and contemporary feature selection approaches:

**Filter Methods**: Enhanced mutual information estimation [@Li2023], robust statistical tests for high-dimensional data, and correlation-based filtering adapted for modern datasets with complex dependency structures.

**Wrapper Methods**: Scalable recursive feature elimination with early stopping, forward/backward selection optimized for large feature spaces, and cross-validation strategies robust to modern data distribution challenges [@Kumar2021].

**Embedded Methods**: Integration with contemporary tree-based models (XGBoost, LightGBM), neural network-based feature importance, and regularization techniques adapted for high-dimensional settings [@Chen2022].

**Hybrid Approaches**: Novel combinations of filter and wrapper methods, ensemble-based feature selection [@Huan2023], and adaptive algorithms that automatically select optimal strategies based on data characteristics [@Rajendran2022].

## Modern Evaluation Framework

The evaluation system incorporates recent advances in feature selection assessment, including stability analysis methods proposed in contemporary literature [@Singh2023], statistical significance testing robust to multiple comparisons, and performance metrics aligned with current machine learning evaluation standards. The framework supports modern cross-validation strategies and handles class imbalance scenarios common in real-world applications.

# Implementation

AutoML-FE is implemented using modern Python practices with emphasis on scalability and maintainability. The architecture leverages contemporary libraries including scikit-learn 1.3+, pandas 2.0+, and NumPy with optimized array operations. Computational efficiency is achieved through vectorized operations, optional GPU acceleration for supported algorithms, and memory-efficient processing suitable for large-scale datasets typical in current applications [@Wang2021].

The software follows current best practices including type hints, comprehensive testing with pytest, continuous integration through GitHub Actions, and documentation generated with modern tools. API design prioritizes usability while maintaining flexibility, incorporating lessons learned from recent user experience studies in machine learning toolkits [@Truong2021].

# Research Applications and Impact

AutoML-FE enables several important contemporary research applications:

**Modern Algorithm Evaluation**: Systematic benchmarking of feature selection methods on current datasets including high-dimensional genomics, large-scale text, and multimodal data common in recent research [@Li2023].

**Domain-Specific Optimization**: Identification of optimal feature selection strategies for emerging application domains including IoT sensor data, financial time series, and social network analysis [@Rahman2024].

**Reproducible Research**: Support for reproducible feature selection research addressing current concerns about reproducibility in machine learning [@Singh2023], with comprehensive logging and deterministic algorithms.

**Educational Applications**: Modern pedagogical tool for teaching feature selection concepts using current datasets and contemporary evaluation methodologies [@Kumar2021].

The toolkit is designed to support current trends toward interpretable machine learning and responsible AI by providing transparent feature selection with clear justification for feature choices [@Huan2023].

# Comparison with Existing Tools

AutoML-FE addresses gaps in the current toolkit ecosystem. Compared to recent AutoML frameworks like AutoGluon [@Erickson2020] and FLAML [@Wang2021], it provides specialized depth in feature selection with maintained transparency. Unlike general-purpose tools, it incorporates recent advances in feature selection methodology while providing comprehensive evaluation capabilities not available in existing specialized tools [@Rajendran2022].

The integrated comparison framework enables systematic evaluation that addresses current challenges in feature selection research, providing capabilities that would require substantial custom implementation using existing tools [@Chen2022].

# Conclusion

AutoML-FE addresses critical gaps in modern machine learning toolkits by providing comprehensive, transparent, and efficient feature selection capabilities adapted for contemporary challenges. The combination of classical and modern algorithms, intelligent preprocessing, and rigorous evaluation frameworks makes it valuable for both current research applications and practical deployments. The package's emphasis on interpretability and reproducibility aligns with current trends toward responsible and transparent machine learning practices.

# Acknowledgements

We acknowledge the open-source community and the developers of scikit-learn, NumPy, and pandas for providing the foundational libraries that enabled this work. We thank the feature selection research community for developing the algorithms and evaluation methodologies implemented in this toolkit.

# References