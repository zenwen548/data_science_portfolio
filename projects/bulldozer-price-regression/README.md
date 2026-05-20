# Bulldozer Price Regression

## Overview

This project predicts used bulldozer auction prices using equipment and sales data from the Kaggle Blue Book for Bulldozers competition.

The notebook covers exploratory analysis, date-based feature engineering, missing-value handling, categorical encoding, model comparison, validation scoring, prediction export, and feature importance.

## Problem

Auction buyers and sellers need a practical way to estimate equipment value from historical sale records. The goal is to predict sale price from equipment attributes, sale timing, machine metadata, and listing details.

## Source Artifact

- [Notebook](../../Bulldozer-Price-Regression.ipynb)

The notebook is currently the main project artifact. Later cleanup can move it into this folder and split reusable logic into `src/` modules.

## Data

Source: Kaggle Blue Book for Bulldozers competition.

The Kaggle data files are not stored in this repo. To reproduce the notebook, download the competition files from Kaggle and place them in a local `data/` folder.

## Workflow

The notebook includes:

- Loading and inspecting the training and validation data
- Parsing `saledate` as a datetime field
- Sorting records chronologically
- Creating sale year, month, day, day of week, and day of year features
- Encoding categorical columns as numeric values
- Adding missingness indicator columns
- Filling numeric missing values with medians
- Splitting training and validation data by sale year
- Training Random Forest and XGBoost regression models
- Evaluating with MAE, RMSLE, and R2
- Exporting test predictions
- Reviewing model feature importance

## Saved Model Results

Best saved tuned XGBoost validation results from the current notebook:

| Metric | Validation Result |
| --- | ---: |
| MAE | about `$10,480` |
| RMSLE | `0.398` |
| R2 | `0.648` |

The notebook uses RMSLE because it is the Kaggle competition metric. MAE is included because it is easier to explain as average dollar error.

## Feature Importance

The tuned XGBoost model's top feature-importance signals are missingness and equipment-configuration fields:

| Feature | Importance |
| --- | ---: |
| `Scarifier_is_missing` | `0.426` |
| `Coupler_System_is_missing` | `0.144` |
| `Coupler_System` | `0.068` |
| `ProductSize_is_missing` | `0.035` |
| `Grouser_Tracks_is_missing` | `0.035` |

Missing listing details may carry pricing signal because incomplete records can line up with older, less-documented, or lower-value equipment. Equipment configuration fields such as coupler system may also reflect machine capability and resale value.

## Takeaway

This is a classic tabular ML project with a real dataset, a practical pricing problem, time-aware validation, model comparison, and interpretable results.

I would not add Spark to this project. The dataset fits in memory, so Spark would not add much. Docker and MLflow are better next additions because they would improve reproducibility and experiment tracking without changing the project goal.

## Next Updates

- Fix visible notebook typos.
- Standardize data path casing to `data/`.
- Update the notebook's final feature-importance sentence so it matches the `feature_importance_df.head(15)` output.
- Move the notebook and outputs into this folder.
- Extract reusable preprocessing, training, and evaluation code into `src/`.
- Optionally add Docker and MLflow tracking.
