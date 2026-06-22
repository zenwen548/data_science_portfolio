# Software Requirements

Versions the projects in this repo are built and tested against.

## Runtime

| Software | Version | Notes |
| --- | --- | --- |
| Python | 3.11 or 3.12 | Docker uses `python:3.11-slim`; local verification used Python 3.12.13 |
| Docker | 24 or newer | Optional, used for the bulldozer training container |
| Git | 2.40 or newer | Used for normal branch and PR workflows |
| Tableau Desktop or Tableau Public | Local installed version | Only needed to edit or republish dashboard workbooks |

## Python packages

Install with:

```bash
pip install -r requirements.txt
```

Pinned ranges in `requirements.txt`:

```text
numpy>=2.4,<3
pandas>=2.3,<3
scikit-learn>=1.9,<2
xgboost>=3.3,<4
matplotlib>=3.11,<4
mlflow>=2.10,<3.13
```

The lower bounds match the disposable verification environment used during the Phase 1 cleanup. The upper bounds stop the next major release from changing behavior unexpectedly. The MLflow upper bound is required because MLflow 3.13 rejects `file:` tracking URIs by default, which breaks the documented local file-store workflow unless an environment override is set.

## Verifying an environment

From the repo root in a fresh virtual environment:

```bash
pip install -r requirements.txt
python -m unittest discover
python -m unittest discover -s projects/bulldozer-price-regression/tests
```

For the bulldozer real-data workflow, place Kaggle `TrainAndValid.csv` at `data/TrainAndValid.csv`, then run:

```bash
python projects/bulldozer-price-regression/src/train.py --data data/TrainAndValid.csv --output-dir outputs/bulldozer
```

For the churn sample workflow, run:

```bash
python projects/customer-churn/src/Logistic_Regression_Customer_Churn.py --customers projects/customer-churn/data/sample_customers.csv --subscriptions projects/customer-churn/data/sample_subscriptions.csv --output-dir outputs/churn
```
