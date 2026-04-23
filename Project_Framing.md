Project Framing – Execution Goals 
Objective

The objective of this project is to build a machine learning model to predict whether a customer will default on their credit card payment. This is a binary classification problem.

Execution Questions

The project aims to answer the following key questions:

Which customer behaviors are most important in predicting default?
How can class imbalance be handled effectively?
What types of features contribute most to model performance?
How much improvement can advanced models like XGBoost provide over baseline models?
How can model predictions be explained clearly using SHAP?

Problem Understanding

Customers with high credit utilization, delayed payments, and unstable spending patterns are more likely to default. The dataset is imbalanced, as most customers do not default. Additionally, customer behavior over time plays an important role in predicting risk.

Data Strategy

We will use an 80/20 train-test split on the training dataset and evaluate model performance using ROC-AUC as the primary metric, along with Precision, Recall, and F1-score.

The GMSC dataset will be used for initial testing and debugging, while the AMEX dataset will be used for actual model training. The Kaggle test dataset will only be used on the final day.

Feature Engineering Plan

Feature engineering will focus on capturing customer financial behavior. The following types of features will be created:

Credit utilization ratio
Payment history and delays
Spending trends over time
Number of missed payments
Time-based features such as rolling averages and lag values
Execution Pipeline

The project will follow a structured workflow:

Data cleaning → Feature engineering → Model building → Evaluation → Optimization

Scope
In Scope
Data cleaning and preprocessing
Feature engineering based on domain knowledge
Building baseline models such as Logistic Regression and Decision Tree
Advanced modeling using XGBoost
Model evaluation using ROC-AUC and other metrics
Model explainability using SHAP
Out of Scope
Use of Kaggle test dataset before the final day
Deployment of the model to production
Real-time prediction systems
Integration of external datasets
Success Criteria

Success will be measured by achieving a strong ROC-AUC score, improving performance from baseline to advanced models, and providing clear explanations of model predictions.
