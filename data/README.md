# Data

This folder documents local data used to reproduce portfolio projects. Large source files stay local and are not committed.

## What ships vs what rebuilds

| Data | Location | Ships with repo? | How to get it |
| --- | --- | --- | --- |
| Churn sample customers | `projects/customer-churn/data/sample_customers.csv` | Yes | Included |
| Churn sample subscriptions | `projects/customer-churn/data/sample_subscriptions.csv` | Yes | Included |
| Bulldozer training data (`TrainAndValid.csv`) | `data/TrainAndValid.csv` | No | Download from the Kaggle Blue Book for Bulldozers competition and place it here |
| Pipeline outputs | `outputs/`, `projects/*/outputs/`, or a local temp folder | No | Rebuilt by running each project pipeline |
| MLflow runs | `mlruns/` | No | Rebuilt by running the bulldozer training workflow with MLflow enabled |

## Local data notes

- The real bulldozer Kaggle file is useful for verification but should remain untracked.
- The churn sample files are intentionally small so the churn pipeline can run end to end from a clean clone.
- Generated metrics, predictions, model files, Tableau exports, and MLflow runs are local outputs unless a specific PR task card calls for a small saved sample.
