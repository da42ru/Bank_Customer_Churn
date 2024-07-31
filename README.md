# Fairway Bank Churn Reduction

This project explores a classification problem for the hypothetical Fairway Bank, in which we seek to predict customer churn. We examine a dataset of 10K customers and 15 features. Input features include credit score, country, gender, age, tenure, balance, number of products that the customer has purchased with the bank, whether or not the customer has a credit card, whether or not the customer is active, estimated salary, satisfaction score, card type, points, and whether or not the customer complained. The target variable is whether or not the customer churned. A golden feature was found in 'complain': 9 classification models were able to predict churn with 99.9% recall when this feature was included. Subsequent recall without the golden feature dropped to a median of 44.1% across 10 models. Adjusting probability thresholds and hyperparameters resulted in a best model - CatBoost Classifier - with a cross-validated train set recall of 64.7% and a test set recall of 67.4%. Through assessing the top 5 best models' feature importances along with exploratory data analysis, the following features were identified as key indicators of churn likelihood: age, number of products, active, country (specifically if the customer is based in Germany), balance, and gender. Insights herein provide Fairway with actionable solutions to reduce customer attrition and improve overall customer satisfaction and retention. Further details are included in the following files.

## Data

- bank.csv (original dataframe of 10K rows and 18 columns)
- bank_2.csv (cleaned dataframe imported for EDA with 10K rows and 15 columns)
- bank_3.csv (model-ready dataframe with dummy encoded numerical values for all columns, 10K rows by 18 columns)
- bank_4.csv (final dataframe with binned numerical variables, includes 2 umap embedding features, 10K rows by 18 colums)

## Notebooks

- 00_data_wrangling.ipynb (initial data cleanliness assessment and review of distributions)
- 01_eda.ipynb (review of correlations and pairwise scatter / count plots of between features)
- 02_preprocessing.ipynb (dummy encoding, value binning, dimensionality reduction, cluster analysis)
- 03_modeling.ipynb (discover of golden feature: 'complain')
- 04_modeling.ipynb (remove 'complain' feature, probability threshold adjustment)
- 05_modeling.ipynb (final model evaluation and comparison, feature importances)

## Hyperparameter Tuning

We searched for ideal hyperparameter settings across 10 models with Optuna. We tuned 2 versions of each model - 1 that optimized recall and 1 that optimized f1-score, for a total of 20 notebooks (1 per model per optimization metric). Ultimately we opted for all f1-score optimized model versions, given a larger overall gain in model precision as compared to the recall-optimized versions which only added a slight gain in recall.

## Reports

- best_model_metrics.pdf (CatBoost Classifier, feature importances, hyperparameters, train / test recall and f1-score)
- final_report.pdf (16-page written report of project summary and key findings)
- presentation.pdf (33-slide PowerPoint presentation of project summary and key findings)
- project_proposal.pdf (original project proposal)
