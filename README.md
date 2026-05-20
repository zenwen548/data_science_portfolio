# Data Science & ML Engineering Portfolio

Selected graduate and independent projects demonstrating Python, machine learning, feature engineering, model evaluation, and reproducible analysis workflows.

This portfolio is being organized around two goals:

- Show clear, business-facing data science work through concise case studies.
- Separate polished portfolio projects from older learning exercises and work in progress.

## Project Index

| Project | Focus | Tools | Status |
| --- | --- | --- | --- |
| [Bulldozer Price Regression](projects/bulldozer-price-regression/README.md) | Price prediction, feature engineering, model evaluation, feature importance | Python, pandas, scikit-learn, XGBoost | Featured |
| PySpark Clustering and Feature Engineering | Distributed data processing, unsupervised learning, production-style workflow | PySpark, KMeans, Docker/AWS | In progress |
| Customer Churn Analytics Dashboard | Churn risk analysis, classification workflow, stakeholder dashboarding | Python, pandas, scikit-learn, Tableau | Dashboard project |

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

## Customer Churn Analytics Dashboard

Built a standalone churn analytics project pairing Python-based customer feature engineering and classification with a Tableau dashboard for visualizing retention patterns and communicating customer-risk insights.

Current positioning:

- The Tableau dashboard is the primary business-facing artifact for stakeholder communication.
- The Python workflow demonstrates customer data preparation, churn labeling, feature creation, model training, and exportable evaluation outputs.
- The old `1.00` accuracy / ROC-AUC claim is not used as final model-performance evidence until validation is tightened.

Model-validation refinements planned before featuring model scores on a resume or Carrd page:

- Add public, sample, or documented source data so the project is reproducible from the repo.
- Use a leakage-safe train/test split and remove features derived from post-outcome subscription status.
- Add a logistic regression baseline before comparing stronger models.
- Keep the Tableau dashboard as the stakeholder-facing communication deliverable.

Artifacts:

- [Churn analysis script](Logistic_Regression_Customer_Churn.py)
- Tableau dashboard: available as the stakeholder-facing deliverable; public link to be added or confirmed before resume/Carrd sharing.

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

- Keep bulldozer as the strongest current portfolio project.
- Preserve churn as a dashboard-centered analytics project while tightening the model validation story.
- Move project assets into `projects/` folders over time.
- Add `requirements.txt` and reproducibility notes.
- Add optional Docker and MLflow tracking to the bulldozer workflow.

Longer-term target structure:

```text
README.md
projects/
  bulldozer-price-regression/
    README.md
    notebooks/
    src/
    outputs/
  pyspark-clustering/
    README.md
    notebooks/
    src/
    docker/
  customer-churn/
    README.md
    src/
    outputs/
assets/
requirements.txt
.gitignore
```