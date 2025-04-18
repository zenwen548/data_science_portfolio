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
  - Created a reusable preprocessing function that handled missing values, converted categories, and extracted sale date features like year, month, and day of week.
  - Trained and tuned both a Random Forest Regressor and XGBoost model using scikit-learn, comparing baseline and optimized performance.
  - Used feature importance plots to identify the top predictors — including missingness indicators — to enhance interpretability.
  - Exported predictions and structured them for potential downstream business use (e.g. price guidance or bidding thresholds).
- **Files**:
  - [`Logistic_Regression_Customer_Churn.py`](Logistic_Regression_Customer_Churn.py): Main script for data cleaning, feature engineering, and churn prediction.
### Model Performance

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


### 2. Bulldozer_Price_Regression
- **Description**: Built a regression model to predict used bulldozer sale prices using structured auction data from Kaggle. Applied a full data science workflow including preprocessing, feature engineering, model tuning, and real-world performance evaluation.

- **Key Features**:

- Created a reusable preprocessing function that handled missing values, converted categories, and extracted sale date features like year, month, and day of week.
- Trained and tuned both a Random Forest Regressor and XGBoost model using scikit-learn, comparing baseline and optimized performance.
- Used feature importance plots to identify the top predictors — including missingness indicators — to enhance interpretability.
- Exported predictions and structured them for potential downstream business use (e.g. price guidance or bidding thresholds).

**Files**:

- [bulldozer-price-regression.ipynb](https://github.com/zenwen548/data_science_portfolio/blob/1885c46b3fbdf61bfb44258df61103076a76cbc7/Bulldozer-Price-Regression.ipynb): Full notebook including data cleaning, modeling, feature importance analysis, and predictions.

- LinkedIn_Bulldozer_Article_Improved_Final.docx: A long-form article-style write-up designed for technical storytelling and portfolio presentation.

### Model Performance
The final model demonstrated strong predictive power and generalizability:

- **R² Score (Validation Set):** 0.648

- **Mean Absolute Error (MAE):** ~$10,000

This means the model explains nearly 65% of the variation in bulldozer prices and can predict sale prices within a ten thousand dollars on average — offering actionable accuracy for real-world decisions.

### Feature Importance

**Top Predictors**:
  - Scarifier_is_missing – Surprisingly, the absence of data about scarifier equipment (used to break up tough terrain) was the strongest signal — likely correlating with
    older, less-documented, and lower-value machines.
  - Coupler_System_is_missing – Missing values in coupler system fields (which relate to attachment versatility) also ranked highly, reinforcing the impact of incomplete
    listings.
  - Coupler_System – Presence of coupler systems may signal newer, more adaptable machines, which tend to fetch higher prices.
  
  - Note: While traditional features like YearMade, ProductSize, and saleYear still matter, XGBoost revealed that missing values themselves can be highly predictive —
    offering insight into listing completeness and equipment value.
