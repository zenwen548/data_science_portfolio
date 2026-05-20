# Data Science & ML Engineering Portfolio

Selected graduate and independent projects demonstrating Python, machine learning, feature engineering, model evaluation, and reproducible analysis workflows.

This portfolio is being organized around two goals:

- Show clear, business-facing data science work through concise case studies.
- Separate polished portfolio projects from older learning exercises and work in progress.

## Project Index

| Project | Focus | Tools | Status |
| --- | --- | --- | --- |
| [Bulldozer Price Regression](projects/bulldozer-price-regression/README.md) | Price prediction, feature engineering, model evaluation, feature importance | Python, pandas, scikit-learn, XGBoost | Featured project |
| PySpark Clustering and Feature Engineering | Distributed data processing, unsupervised learning, production-style workflow | PySpark, KMeans, Docker/AWS | In progress |
| Customer Churn Classification | Classification workflow and Tableau export | Python, pandas, scikit-learn, Tableau | Legacy / under review |

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

## Legacy Project: Customer Churn Classification

The churn project is currently kept as a legacy learning exercise rather than a featured portfolio project.

Why it is under review:

- The current script trains a `RandomForestClassifier`, while the older README title described logistic regression.
- The original performance claim of `1.00` accuracy / ROC-AUC is not used as portfolio evidence because the feature engineering needs leakage-safe validation.
- The source CSV files are not included in this repository, so the project is not currently reproducible from the public repo alone.

Planned direction:

- Rebuild with public or sample data.
- Use a leakage-safe train/test split.
- Add a logistic regression baseline before comparing stronger models.
- Document the Tableau output as business communication rather than presenting the current model as final.

Artifact:

- [Legacy churn script](Logistic_Regression_Customer_Churn.py)

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
- Move project assets into `projects/` folders over time.
- Add `requirements.txt` and reproducibility notes.
- Add optional Docker and MLflow tracking to the bulldozer workflow.
- Demote or rebuild the churn project before using it on a resume or Carrd page.

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
