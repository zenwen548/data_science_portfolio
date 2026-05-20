# Data Science & ML Engineering Portfolio

Selected graduate and independent projects demonstrating Python, machine learning, feature engineering, model evaluation, dashboarding, and reproducible analysis workflows.

This portfolio is organized around two goals:

- Show clear, business-facing data science work through concise case studies.
- Highlight practical ML, analytics, and communication deliverables that map to real stakeholder decisions.

## Project Index

| Project | Focus | Tools | Status |
| --- | --- | --- | --- |
| [Bulldozer Price Regression](projects/bulldozer-price-regression/README.md) | Price prediction, feature engineering, model evaluation, feature importance | Python, pandas, scikit-learn, XGBoost | Featured |
| Customer Churn Analytics Dashboard | Churn risk analysis, classification workflow, stakeholder dashboarding | Python, pandas, scikit-learn, Tableau | Featured |
| PySpark Clustering and Feature Engineering | Distributed data processing, unsupervised learning, production-style workflow | PySpark, KMeans, Docker/AWS | In progress |

## Featured Project: Bulldozer Price Regression

Built a supervised regression workflow to predict used bulldozer auction prices from structured equipment and sales data from the Kaggle Blue Book for Bulldozers competition.

Current saved model evidence from the notebook:

- Validation MAE: about `$10,480`
- Validation RMSLE: `0.398`
- Validation R2: `0.648`

The strongest feature-importance signals in the tuned XGBoost model were missingness indicators, especially `Scarifier_is_missing` and `Coupler_System_is_missing`, plus `Coupler_System`. This suggests that listing completeness and equipment configuration may be meaningful pricing signals, not just traditional variables like model year.

Artifacts:

- [Case study README](projects/bulldozer-price-regression/README.md)
- [Notebook](Bulldozer-Price-Regression.ipynb)

## Featured Project: Customer Churn Analytics Dashboard

Built a standalone churn analytics project pairing Python-based customer feature engineering and classification with a Tableau dashboard for visualizing retention patterns and communicating customer-risk insights.

What the project demonstrates:

- Customer and subscription data cleaning across multiple source tables.
- Churn feature engineering using customer activity, subscription cost, support activity, referral status, and subscription start timing.
- Baseline logistic regression and Random Forest model comparison.
- Tableau-ready export with churn predictions and churn probabilities for dashboarding.
- Business-facing communication through visual segmentation and retention-risk reporting.

Run the churn pipeline with the included sample schema:

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

Artifacts:

- [Churn analytics pipeline](Logistic_Regression_Customer_Churn.py)
- [Sample customer data](data/sample_customers.csv)
- [Sample subscription data](data/sample_subscriptions.csv)
- Tableau dashboard: stakeholder-facing dashboard link to be added before Carrd/resume sharing.

## Upcoming PySpark / KMeans Work

The upcoming PySpark and KMeans projects will be added as standalone headline projects when complete. They are a better fit for demonstrating distributed processing, clustering, Docker, AWS, and production-style workflow than retrofitting Spark into the current supervised learning projects.

Planned portfolio framing:

- PySpark feature engineering on larger or partitioned data
- KMeans clustering / segmentation with interpretable cluster profiles
- Dockerized project environment
- Cloud-oriented workflow notes
- Clear case-study write-up for both technical and non-technical reviewers

## Repository Cleanup Roadmap

Near-term cleanup:

- Keep bulldozer and churn as the current featured portfolio projects.
- Move project assets into `projects/` folders over time.
- Add optional Docker and MLflow tracking to the bulldozer workflow.
- Add the Tableau dashboard URL and screenshots for the churn project.

Longer-term target structure:

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