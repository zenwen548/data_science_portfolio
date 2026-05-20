# Bulldozer Price Regression

## Overview

This project predicts used bulldozer auction prices using structured equipment and sales data from the Kaggle Blue Book for Bulldozers competition.

The project demonstrates a full tabular machine learning workflow: exploratory analysis, date-based feature engineering, missing-value handling, categorical encoding, model comparison, validation scoring, prediction export, and feature-importance interpretation.

## Problem

Auction buyers and sellers need a practical way to estimate equipment value from historical sale records. The modeling goal is to predict sale price from equipment attributes, sale timing, machine metadata, and listing completeness.

## Current Source Artifact

- [Notebook](../../Bulldozer-Price-Regression.ipynb)

The notebook is currently the source of truth. A future cleanup pass should split reusable logic into `src/` modules and move the notebook into this project folder.

## Data

Source: Kaggle Blue Book for Bulldozers competition.

The public repository does not include the Kaggle data files. To reproduce the notebook, download the competition data from Kaggle and place the files under a local `data/` directory.

## Modeling Workflow

The notebook currently includes:

- Loading and inspecting the training / validation data
- Parsing `saledate` as a datetime field
- Sorting records chronologically
- Creating date-derived features such as sale year, month, day, day of week, and day of year
- Encoding categorical columns as numeric values
- Adding missingness indicator columns
- Filling numeric missing values with medians
- Splitting training and validation data by sale year
- Training Random Forest and XGBoost regression models
- Evaluating with MAE, RMSLE, and R2
- Exporting test predictions
- Interpreting model feature importance

## Saved Model Evidence

Best saved tuned XGBoost validation results from the current notebook:

| Metric | Validation Result |
| --- | ---: |
| MAE | about `$10,480` |
| RMSLE | `0.398` |
| R2 | `0.648` |

The notebook uses RMSLE because it is the Kaggle competition metric, while MAE is useful for plain-language interpretation of average price error.

## Feature Importance

The tuned XGBoost model's top feature-importance signals are missingness and equipment-configuration indicators:

| Feature | Importance |
| --- | ---: |
| `Scarifier_is_missing` | `0.426` |
| `Coupler_System_is_missing` | `0.144` |
| `Coupler_System` | `0.068` |
| `ProductSize_is_missing` | `0.035` |
| `Grouser_Tracks_is_missing` | `0.035` |

Interpretation: missing listing details may carry pricing signal because incomplete records can correlate with older, less-documented, or lower-value equipment. Equipment configuration fields such as coupler system may also reflect machine capability and resale value.

## Portfolio Takeaway

This is the strongest current project in the repository because it uses a real dataset, a practical forecasting problem, time-aware validation, model comparison, and interpretable results.

It should be positioned as a classic tabular ML project, not as a PySpark or distributed-processing project. The dataset is small enough to fit in memory, so adding Spark here would read as overengineering. Docker and MLflow are better enhancement candidates because they would improve reproducibility and experiment tracking without changing the nature of the problem.

## Known Cleanup Items

Next pass improvements:

- Fix visible notebook typos and narrative polish.
- Standardize data path casing to `data/`.
- Update the notebook's final feature-importance sentence so it matches the actual `feature_importance_df.head(15)` output.
- Move notebook and outputs into this folder.
- Extract reusable preprocessing, training, and evaluation code into `src/`.
- Add `requirements.txt`.
- Optionally add Docker and MLflow tracking.
