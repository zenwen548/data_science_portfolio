# data_science_portfolio

Welcome to my data science portfolio! This repository showcases various projects demonstrating my skills in Python, data cleaning, data analysis, and machine learning.

This repository showcases various projects that highlight my skills in Python, data cleaning, data analysis, and machine learning.

Each project folder contains:
- Python scripts for data processing and modeling.
- Datasets (where applicable).
- Detailed explanations of the project's purpose, approach, and outcomes.

## Projects
### 1. Logistic Regression Customer Churn
- **Description**: Predicts customer churn using logistic regression and random forest models based on subscription and customer data.
- **Key Features**:
  - Data cleaning to handle missing values and duplicates.
  - Calculation of derived features like tenure days and referral status.
  - Model training with logistic regression and random forest classifiers.
  - Outputs include a cleaned dataset for Tableau and model evaluation metrics.
- **Files**:
  - [`Logistic_Regression_Customer_Churn.py`](Logistic_Regression_Customer_Churn.py): Main script for data cleaning, feature engineering, and churn prediction.
## Model Performance

The Logistic Regression model achieved exceptional performance in predicting customer churn:

- **Accuracy:** 1.00
- **ROC-AUC Score:** 1.00

### Confusion Matrix
![Confusion Matrix](https://github.com/zenwen548/data_science_portfolio/blob/main/Log_Reg_Confusion_Matrix.jpg)
Interpretation:
True Positives: 2339 (Correctly identified churned customers).
True Negatives: 294 (Correctly identified non-churned customers).
False Positives: 2 (Misclassified as churned).
False Negatives: 5 (Misclassified as not churned).
ROC Curve

### ROC Curve
![ROC Curve](https://github.com/zenwen548/data_science_portfolio/blob/main/Log_Reg_ROC_Curve.jpg)

Key Insight:
The AUC (Area Under the Curve) score of 1.00 indicates perfect separation between churned and non-churned customers.

### Tableau Visualization
![Tableau_Viz](https://public.tableau.com/app/profile/lee6095/viz/Churn_Model_Viz/Dashboard1)



