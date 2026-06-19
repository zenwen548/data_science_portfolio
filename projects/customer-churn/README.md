# Customer Churn Analytics

## Overview

This project predicts customer churn from subscription and account data, then exports a Tableau-ready dataset with churn predictions and probabilities for business review.

The pipeline covers data cleaning across two source tables, churn feature engineering, leakage-safe feature selection, a Logistic Regression and Random Forest comparison, and a dashboard export with per-customer churn risk.

## Problem

Subscription businesses lose revenue quietly when customers cancel. The goal is to score each customer's churn risk from account attributes, support activity, transaction behavior, and subscription timing, so retention work can focus on the customers most likely to leave.

## What it is and what it isn't

What it is:

- A reproducible end-to-end pipeline from raw CSVs to dashboard-ready output
- A worked example of leakage-aware feature selection in a churn setting
- The data source for the Tableau churn dashboard

What it isn't:

- A production scoring service. There is no API, scheduler, or model registry.
- A benchmark study. The included sample data is small and exists to demonstrate the pipeline, so metrics from a sample run are not meaningful performance claims.

## Data

Two input tables, joined on `customer_id`:

- `data/sample_customers.csv`: customer segment, region, and account type
- `data/sample_subscriptions.csv`: subscription dates, cost, support calls, referral code, transaction activity, and the churn label

The sample files contain 24 customers and ship with the repo so the pipeline runs out of the box. Any pair of CSVs with matching columns can be substituted via the command-line flags.

## Workflow

The pipeline:

- Standardizes column names and normalizes null values across both tables
- Deduplicates customers and subscriptions before the join
- Derives the churn label from `subscription_end_date` when no explicit label exists, and normalizes text labels when one does
- Engineers features: referral flag, subscription start year, month, and quarter, and a transactions-per-day backfill from monthly activity
- Drops leakage columns before modeling: `subscription_end_date`, `is_active`, `tenure_days`, `years_since_signup`, plus all ID and date columns
- Selects numeric features and low-cardinality categorical features automatically
- Trains Logistic Regression and Random Forest models in scikit-learn pipelines
- Evaluates on a stratified 30 percent test split using accuracy and ROC-AUC, and selects the best model by ROC-AUC
- Exports the full dataset with predictions and churn probabilities from the winning model

## Running it

From the repo root:

```bash
pip install -r requirements.txt
python projects/customer-churn/src/Logistic_Regression_Customer_Churn.py --customers projects/customer-churn/data/sample_customers.csv --subscriptions projects/customer-churn/data/sample_subscriptions.csv --output-dir outputs/churn
```

Outputs land in `outputs/churn/`:

| File | Contents |
| --- | --- |
| `cleaned_data.csv` | Merged and cleaned dataset before scoring |
| `final_data_for_tableau.csv` | Full dataset plus model name, predicted churn, and churn probability |
| `model_metrics.csv` | Accuracy and ROC-AUC for both models |
| `confusion_matrix.csv` | Confusion matrix for the selected model |
| `classification_report.txt` | Precision, recall, and F1 for the selected model |

## Dashboard

The Tableau workbook in `dashboards/Churn_Model_Viz.twbx` visualizes the pipeline output across eight views, including churn probability distribution, churn by region, customer tenure against churn probability, revenue at risk, and support-call patterns by risk group.

Public Tableau link coming after the Phase 2 republish.

![Confusion matrix](assets/Log_Reg_Confusion_Matrix.jpg)

![ROC curve](assets/Log_Reg_ROC_Curve.jpg)

## Limitations

- The sample data is small and synthetic in scale. It proves the pipeline runs end to end; it does not support performance claims.
- Classification uses the default 0.5 threshold. A real retention program would tune the threshold against the cost of missed churners versus wasted outreach.
- There is no hyperparameter search. Model settings are sensible fixed choices, not tuned values.
- When no churn label is provided, churn is inferred from the presence of a subscription end date, which treats every ended subscription as churn.
