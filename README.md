# Data Science Portfolio

Projects from my graduate coursework and independent work in machine learning, analytics, dashboards, and Python-based data workflows.

I use this repo to keep finished portfolio projects, work in progress, and supporting project files in one place.

## Project Index

| Project | Focus | Tools | Status |
| --- | --- | --- | --- |
| [Bulldozer Price Regression](projects/bulldozer-price-regression/README.md) | Price prediction, feature engineering, model evaluation, feature importance | Python, pandas, scikit-learn, XGBoost | Featured |
| Customer Churn Analytics Dashboard | Churn analysis, classification workflow, Tableau dashboard | Python, pandas, scikit-learn, Tableau | Featured |
| PySpark Clustering and Feature Engineering | Distributed data processing, unsupervised learning, production-style workflow | PySpark, KMeans, Docker/AWS | In progress |

## Bulldozer Price Regression

I built a regression workflow to predict used bulldozer auction prices from structured equipment and sales data from the Kaggle Blue Book for Bulldozers competition.

Current saved model results from the notebook:

- Validation MAE: about `$10,480`
- Validation RMSLE: `0.398`
- Validation R2: `0.648`

The strongest feature-importance signals in the tuned XGBoost model were missingness indicators, especially `Scarifier_is_missing` and `Coupler_System_is_missing`, plus `Coupler_System`. This points to listing completeness and equipment configuration as useful pricing signals.

Artifacts:

- [Case study README](projects/bulldozer-price-regression/README.md)
- [Notebook](Bulldozer-Price-Regression.ipynb)

## Customer Churn Analytics Dashboard

I built a churn analytics project with a Python modeling pipeline and Tableau dashboard. The project focuses on customer retention patterns, churn risk scoring, and dashboard-ready outputs for business review.

What the project includes:

- Customer and subscription data cleaning across multiple source tables.
- Churn feature engineering using customer activity, subscription cost, support activity, referral status, and subscription start timing.
- Logistic regression and Random Forest model comparison.
- Tableau-ready export with churn predictions and churn probabilities.
- Dashboard support for segmenting churn risk and reviewing retention patterns.

Run the churn pipeline with the included sample files:

```bash
pip install -r requirements.txt
python Logistic_Regression_Customer_Churn.py --customers data/sample_customers.csv --subscriptions data/sample_subscriptions.csv --output-dir outputs/churn
```

Pipeline outputs:

- `cleaned_data.csv`
- `final_data_for_tableau.csv`
- `model_metrics.csv`
- `confusion_matrix.csv`
- `classification_report.txt`

This pipeline was verified end-to-end against the sample files above, producing all five outputs listed.

Artifacts:

- [Churn analytics pipeline](Logistic_Regression_Customer_Churn.py)
- [Sample customer data](data/sample_customers.csv)
- [Sample subscription data](data/sample_subscriptions.csv)
- Public Tableau link coming soon.

## Upcoming PySpark / KMeans Work

The PySpark and KMeans projects are in progress and will be added as standalone projects when finished. These will show distributed processing, clustering, Docker, AWS, and production-style workflow more directly than retrofitting Spark into the current supervised learning projects.

Planned additions:

- PySpark feature engineering on larger or partitioned data
- KMeans clustering with interpretable cluster profiles
- Dockerized project environment
- Cloud workflow notes
- Case-study write-up for technical and non-technical reviewers

## Next Updates

- Add the public Tableau dashboard link and screenshots for the churn project.
- Move project assets into `projects/` folders over time.
- Add optional Docker and MLflow tracking to the bulldozer workflow.

Target structure:

```text
README.md
projects/
  bulldozer-price-regression/
    README.md
    notebooks/
    src/
    outputs/
  customer-churn/
    README.md
    src/
    outputs/
    assets/
  pyspark-clustering/
    README.md
    notebooks/
    src/
    docker/
assets/
requirements.txt
data/
  sample_customers.csv
  sample_subscriptions.csv
.gitignore
```