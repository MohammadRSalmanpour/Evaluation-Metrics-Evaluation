# Title: Evaluation Metrics Evaluation: Investigation of Machine Learning Evaluation Metrics Across Different Programming Languages
## Background
Machine learning (ML) is foundational to numerous data-driven computational tasks. However, the consistency of ML evaluation metrics is challenged by varied implementations in Python, R, and Matlab, leading to potentially different outcomes. This study addresses the need for a unified standard and roadmap for ML evaluation metrics across these platforms.

## Method
In this work, we assess a comprehensive range of evaluation metrics across diverse ML tasks, including binary and multi-class classification, regression, clustering, correlation, statistical analysis, segmentation, and image-to-image translation (I2I). Our analysis utilizes real-world medical image datasets to compare metrics within Python libraries, R packages, and Matlab functions, aiming to evaluate metric consistency across environments.

## Results
We identify two main types of discrepancies in evaluation metrics:

Reporting Differences (RD): Variations in how metrics are reported across platforms.
Implementational Differences (ID): Discrepancies due to differences in implementation across libraries, packages, and functions.
Our study found that only specific metrics were consistent across all platforms in various ML tasks:

**Binary Classification**: Accuracy, Balanced Accuracy, Cohen’s Kappa, F-beta Score, MCC, Geometric Mean, AUC, and Log Loss.
**Multi-Class Classification**: Accuracy, Cohen’s Kappa, and F-beta Score.
**Regression**: MAE, MSE, RMSE, MAPE, Explained Variance, Median AE, MSLE, and Huber.
**Clustering**: Davies-Bouldin Index and Calinski-Harabasz Index.
**Correlation**: Pearson, Spearman, Kendall's Tau, Mutual Information, Distance Correlation, Percbend, Shepherd, and Partial Corr.
**Statistical Tests**: Paired t-test, Chi-Square Test, ANOVA, Kruskal-Wallis Test, Shapiro-Wilk Test, Welch's t-test, and Bartlett's Test.
**Segmentation (2D)**: Accuracy, Precision, and Recall.
**Segmentation (3D)**: Accuracy.
**Image-to-Image Translation (2D)**: MAE, MSE, RMSE, and R-Squared.
**Image-to-Image Translation (3D)**: MAE, MSE, and RMSE.

## Conclusion
This study emphasizes the need for standardized evaluation metrics in ML to enable accurate and comparable results across tasks and platforms. We recommend future research prioritize consistent metrics for effective comparisons.

## Table of Contents
### Introduction
### Methodology
### Evaluation Metrics
### Results
### Conclusion
### Contributors

## Introduction
In machine learning, evaluation metrics play a critical role in assessing model performance. However, discrepancies arise when these metrics vary in different programming environments. This study aims to provide a comprehensive evaluation of these metrics across Python, R, and Matlab, with the goal of establishing a standardized approach to ML performance assessment.

## Methodology
The study leverages real-world medical image datasets to compare and analyze ML evaluation metrics across Python libraries, R packages, and Matlab functions. Key metrics were analyzed for consistency across platforms, and results were categorized by type of ML task, including classification, regression, clustering, and more.

## Evaluation Metrics
A wide array of metrics was examined, categorized by ML tasks such as:

**Binary Classification
Multi-Class Classification
Regression
Clustering
Correlation
Statistical Analysis
Segmentation (2D & 3D)
Image-to-Image Translation (2D & 3D)**
Refer to the results section for details on consistent and inconsistent metrics identified for each task.

## Results
The analysis revealed two main discrepancy types:

Reporting Differences (RD): Differences in metric reporting across environments.
Implementational Differences (ID): Variations in metric implementation across libraries, packages, and functions.
## Conclusion
Standardizing ML evaluation metrics is essential for cross-platform consistency. This study suggests prioritizing specific metrics for each task to facilitate consistent and comparable results in future ML research.

## Contributors
Mohammad.R Salmanpour, Morteza Alizadeh, Ghazal Mousavi, Saba Sadeghi, Sajad Amiri, Mehrdad Oveisi, Arman Rahmim, Ilker Hacihaliloglu.
