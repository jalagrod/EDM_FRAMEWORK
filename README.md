# A Replicable and Comprehensive Framework for Machine Learning and Deep Learning in Educational Data Mining

This repository contains the code and supplementary materials for the research article "A Replicable and Comprehensive Framework for Machine Learning and Deep Learning in Educational Data Mining". The study explores the replicability of Machine Learning (ML) and Deep Learning (DL) models on educational datasets and measures the carbon footprint associated with the computational processes.

## Table of Contents

- [Datasets](#datasets)
- [Code Structure](#code-structure)
- [Preprocessing and Feature Selection](#preprocessing-and-feature-selection)
- [Deep Learning Models](#deep-learning-models)
- [Machine Learning Models](#machine-learning-models)
- [Evaluation Metrics](#evaluation-metrics)
- [Execution Instructions](#execution-instructions)

## Datasets

The study utilizes three educational datasets: SAD, EPPD, and SPD. These datasets contain various features related to student performance, demographics, and other relevant information.

### Dataset Description

- **SAD**: Data on student admissions.
- **EPPD**: Data on educational performance prediction.
- **SPD**: Student Performance Dataset from Portuguese secondary schools.

## Code Structure

The repository includes two main scripts for processing and training models on the datasets: one for DL models and another for ML models.

### Files

- `{{DATASET}}-DL.ipynb`: Contains code for training DL models.
- `{{DATASET}}-ML.ipynb`: Contains code for training ML models.

## Preprocessing and Feature Selection

### Steps

1. **Loading Dataset**: Load the dataset from a CSV file and remove spaces from column names.
2. **Transforming Target Variable**: Transform the target variable.
3. **Selecting Relevant Features**: Select the most relevant features for the analysis.
4. **Emission Tracker Configuration**: Initialize the CodeCarbon emissions tracker to measure the carbon footprint.
5. **Variance Filtering**: Use `VarianceThreshold` to filter out features with low variance.
6. **Mutual Information Filtering**: Use `mutual_info_classif` to select features with high mutual information with the target variable.
7. **Creating DataFrame with Selected Features**: Create a new DataFrame containing the selected features and the target variable.

## Deep Learning Models

The `{{DATASET}}-DL.ipynb` script trains DL models using the selected features and evaluates their performance.

### Steps

1. **Data Splitting**: Split the data into K-FOLD (K=10).
2. **Model Training**: Train DL models using `TabularPredictor` from AutoGluon.
3. **GPU Configuration**: Optionally train models using GPU.
4. **Model Evaluation**: Evaluate the models on the test set using accuracy, precision, recall, F1 score, and ROC AUC metrics.
5. **Saving Results**: Save the evaluation metrics and model information to CSV and JSON files.

## Machine Learning Models
The `{{DATASET}}-ML.ipynb` script trains various ML models using the selected features and evaluates their performance.

### Steps
1. **Data Splitting**: Split the data into K-FOLD (K=10).
2. **Model Training**: Train ML models using PyCaret's classification module.
3. **GPU Configuration**: Optionally train models using GPU.
4. **Model Evaluation**: Evaluate the models on the test set using accuracy, precision, recall, F1 score, and ROC AUC metrics.
5. **Saving Results**: Save the evaluation metrics and model information to CSV and JSON files.

## Evaluation Metrics
The following evaluation metrics are used to assess model performance:

**Accuracy**: The percentage of correctly classified instances.
**Precision**: The ratio of true positive instances to the sum of true positive and false positive instances.
**Recall**: The ratio of true positive instances to the sum of true positive and false negative instances.
**F1 Score**: The harmonic mean of precision and recall.
**ROC AUC**: The area under the receiver operating characteristic curve.

## Prerequisites
Python 3.7+
Required Python packages: pandas, scikit-learn, pycaret, autogluon