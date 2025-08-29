# Banking Domain Machine Learning Project

## Problem Statement
PredCatch Analytics' Australian banking client's profitability and reputation are being hit by fraudulent ATM transactions. They want PredCatch to help them in reducing and if possible completely eliminating such fraudulent transactions. PredCatch believes it can do the same by building a predictive model to catch such fraudulent transactions in real time and decline them. Your job as PredCatch's Data Scientist is to build this fraud detection & prevention predictive model in the first step. If successful, in the 2nd step you will have to present your solutions and explain how it works to the client. The data has been made available to you. The challenging part of the problem is that the data contains very few fraud instances in comparison to the overall population. To give more edge to the solution they have also collected data regarding location (geo_scores) of the transactions, their own proprietary index (Lambda_wts), on network turnaround times (Qset_tats) and vulnerability qualification score (instance_scores). As of now you don't need to understand what they mean. Training data contains masked variables pertaining to each transaction id. Your prediction target here is 'Target' where 1: Fraudulent transactions and 0: Clean transactions.

## Project Overview
This repository contains a complete end-to-end machine learning workflow applied to the banking domain, implemented in a Jupyter Notebook. The notebook performs data ingestion, cleaning, enrichment, exploratory data analysis, feature engineering, supervised classification, ensemble modeling, anomaly detection, and model persistence. All steps are presented sequentially and continuously without breaks in flow.

## Workflow Description
The workflow begins by importing essential libraries including numpy, pandas, matplotlib, seaborn, scikit-learn, xgboost, and joblib. Multiple CSV files are loaded such as `train.csv`, `test_share.csv`, `Geo_scores.csv`, `instance_scores.csv`, `Qset_tats.csv`, and `Lambda_wts.csv`. These represent different aspects of banking data including geographic scores, instance-level scores, question set turnaround times, and group-level lambda weights. The train and test sets are tagged and concatenated for uniform processing. Data integrity checks such as shapes, unique counts, and null value counts are performed to verify consistency across all sources. Missing values in `geo_score` and `qsets_normalized_tat` are filled with median values. Aggregation at the `id` level is performed using groupby and mean for both `geo` and `qset`. The aggregated tables are merged back into the master dataset on `id` or `Group`, enriching the training and testing sets with additional features. After enrichment, the dataset is split back into `train` and `test` subsets. Features `x` are defined by dropping identifiers and target columns (`id`, `Group`, `Target`, `data`), and the target `y` is set as the `Target` column.

Exploratory data analysis includes correlation heatmaps across numerical variables and distribution/boxplots to identify skewness and outliers. Outlier capping functions are documented, providing logic for handling extreme values using IQR and percentile-based thresholds. The training dataset is stratified split into training and testing subsets to maintain class balance using `train_test_split` with stratification. Several baseline classifiers are then trained and evaluated including Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM, KNN, and Bernoulli Naive Bayes. For each classifier, confusion matrices and classification reports are generated on both training and testing sets, along with train and test accuracy scores. To combine the strengths of multiple models, a soft voting classifier is built using Logistic Regression, Decision Tree, Random Forest, SVM with probabilities, and BernoulliNB. Accuracy of this ensemble is computed alongside the individual models. A stacking classifier is also trained using Random Forest, Gradient Boosting, and BernoulliNB as base learners, with Logistic Regression as the meta-learner. The accuracies of all models and ensembles are stored in a dataframe and visualized in a seaborn barplot for direct comparison.

In addition to supervised learning, unsupervised anomaly detection techniques are applied to the same dataset to explore their ability to identify minority risky cases. The fraction of outliers is estimated from the target distribution, and algorithms such as Isolation Forest, Local Outlier Factor, and One-Class SVM are fitted with contamination set to this fraction. Predictions of ±1 are mapped to 0/1 labels and compared against ground truth `y_test` to compute accuracy and error counts. This offers an alternative view on anomaly detection relative to supervised classification.

Finally, the best-performing Random Forest model is saved using joblib as `banking_rf_model.pkl`. Version checks for streamlit, pandas, scikit-learn, and joblib are printed to ensure environment reproducibility.

## Technologies Used
- Python (Jupyter Notebook)
- pandas and numpy for data manipulation
- seaborn and matplotlib for visualization
- scikit-learn for preprocessing, modeling, evaluation, ensemble methods, anomaly detection, and persistence
- xgboost for gradient boosting classification
- joblib for model saving
- streamlit (optional) for deployment readiness check

## Results
The notebook demonstrates that combining multiple classifiers through ensemble methods generally improves accuracy compared to individual models. Random Forest, XGBoost, and Voting ensembles achieve higher performance compared to simpler models like Logistic Regression or Naive Bayes. Anomaly detection algorithms provide additional insights into minority risky cases but are less accurate than supervised methods. The final Random Forest model is persisted for future use.

## How to Run
1. Clone the repository.
2. Place the dataset files (`train.csv`, `test_share.csv`, `Geo_scores.csv`, `instance_scores.csv`, `Qset_tats.csv`, `Lambda_wts.csv`) in the appropriate folder.
3. Open `Banking Domain.ipynb` in Jupyter Notebook.
4. Run all cells sequentially to reproduce the results.
5. The trained Random Forest model will be saved as `banking_rf_model.pkl`.

## Repository Structure
├── Banking Domain.ipynb # Main notebook containing full workflow
├── data/ # Folder for dataset CSVs
│ ├── train.csv
│ ├── test_share.csv
│ ├── Geo_scores.csv
│ ├── instance_scores.csv
│ ├── Qset_tats.csv
│ └── Lambda_wts.csv
├── models/
│ └── banking_rf_model.pkl # Saved Random Forest model
└── README.md # Project description

## Conclusion
This project provides a complete walkthrough of building, evaluating, and saving machine learning models for the banking domain, covering every step from raw data ingestion to anomaly detection and model persistence. It demonstrates the use of multiple datasets, domain-specific enrichment, careful preprocessing, supervised and unsupervised techniques, ensemble methods, and reproducibility practices. The notebook serves as both a practical guide and a foundation for further experimentation in financial data science projects.

## Live Demo
[Click here to try the Banking Fraud Detection App](https://sourav-003-banking-fraud-detection-app-9pj0kg.streamlit.app/)

