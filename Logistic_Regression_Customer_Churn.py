"""Customer churn analytics and Tableau export pipeline.

This script cleans customer and subscription data, trains churn classifiers, and
exports a Tableau-ready dataset with churn predictions and probabilities.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_CUSTOMERS_PATH = "customers.csv"
DEFAULT_SUBSCRIPTIONS_PATH = "subscriptions.csv"
DEFAULT_OUTPUT_DIR = "."
TARGET_COLUMN = "churned"

LEAKAGE_COLUMNS = {
    "subscription_end_date",
    "is_active",
    "tenure_days",
    "years_since_signup",
}
ID_COLUMNS = {"customer_id", "subscription_id"}


# -------------------------------
# 1. Data Loading and Cleaning
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train churn models and export Tableau-ready churn data."
    )
    parser.add_argument("--customers", default=DEFAULT_CUSTOMERS_PATH)
    parser.add_argument("--subscriptions", default=DEFAULT_SUBSCRIPTIONS_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.30)
    return parser.parse_args()


def standardize_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def clean_data(df):
    df = standardize_columns(df)
    df = df.replace({"NaN": np.nan, "null": np.nan, "": np.nan, "None": np.nan})
    return df.drop_duplicates()


def load_data(customers_path, subscriptions_path):
    customers_path = Path(customers_path)
    subscriptions_path = Path(subscriptions_path)

    missing = [str(path) for path in [customers_path, subscriptions_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required input file(s): "
            + ", ".join(missing)
            + ". Provide paths with --customers and --subscriptions."
        )

    customer_data = clean_data(pd.read_csv(customers_path))
    subscription_data = clean_data(pd.read_csv(subscriptions_path))

    if "customer_id" not in customer_data.columns or "customer_id" not in subscription_data.columns:
        raise ValueError("Both input files must contain a customer_id column.")

    customer_data = customer_data.drop_duplicates(subset=["customer_id"])

    if "subscription_id" in subscription_data.columns:
        subscription_data = subscription_data.drop_duplicates(
            subset=["customer_id", "subscription_id"]
        )
    else:
        subscription_data = subscription_data.drop_duplicates(subset=["customer_id"])

    return customer_data, subscription_data


# -------------------------------
# 2. Feature Engineering
# -------------------------------
def parse_dates(df):
    df = df.copy()
    for column in ["subscription_start_date", "subscription_end_date"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def normalize_binary_target(series):
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)

    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1,
        "true": 1,
        "yes": 1,
        "y": 1,
        "churned": 1,
        "inactive": 1,
        "cancelled": 1,
        "canceled": 1,
        "0": 0,
        "false": 0,
        "no": 0,
        "n": 0,
        "active": 0,
        "retained": 0,
    }
    mapped = normalized.map(mapping)
    if mapped.isna().any():
        unknown_values = sorted(normalized[mapped.isna()].unique())
        raise ValueError(f"Unsupported churn label values: {unknown_values}")
    return mapped.astype(int)


def add_churn_features(df):
    df = parse_dates(df)

    if TARGET_COLUMN not in df.columns:
        if "subscription_end_date" not in df.columns:
            raise ValueError(
                "A churned target column or subscription_end_date column is required."
            )
        df[TARGET_COLUMN] = df["subscription_end_date"].notna().astype(int)
    else:
        df[TARGET_COLUMN] = normalize_binary_target(df[TARGET_COLUMN])

    if "referral_code" in df.columns:
        referral = df["referral_code"].astype("string").str.strip().str.lower()
        df["was_referred"] = (
            referral.notna()
            & ~referral.isin(["", "0", "none", "nan", "null"])
        ).astype(int)
    else:
        df["was_referred"] = 0

    if "subscription_start_date" in df.columns:
        df["subscription_start_year"] = df["subscription_start_date"].dt.year
        df["subscription_start_month"] = df["subscription_start_date"].dt.month
        df["subscription_start_quarter"] = df["subscription_start_date"].dt.quarter

    if (
        "avg_transactions_per_day" in df.columns
        and "avg_monthly_transactions" in df.columns
    ):
        df["avg_transactions_per_day"] = df["avg_transactions_per_day"].fillna(
            df["avg_monthly_transactions"] / 30
        )

    return df


def merge_and_prepare(customer_data, subscription_data):
    merged = pd.merge(customer_data, subscription_data, on="customer_id", how="right")
    merged = add_churn_features(merged)
    return merged


def select_model_features(df):
    excluded = set(LEAKAGE_COLUMNS) | set(ID_COLUMNS) | {TARGET_COLUMN, "referral_code"}
    excluded.update(column for column in df.columns if column.endswith("_id"))
    excluded.update(column for column in df.columns if column.endswith("_date"))

    candidate_columns = [column for column in df.columns if column not in excluded]

    numeric_features = [
        column
        for column in candidate_columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]
    categorical_features = [
        column
        for column in candidate_columns
        if not pd.api.types.is_numeric_dtype(df[column]) and df[column].nunique(dropna=True) <= 30
    ]

    if not numeric_features and not categorical_features:
        raise ValueError("No usable model features were found after preprocessing.")

    return numeric_features, categorical_features


# -------------------------------
# 3. Modeling
# -------------------------------
def build_preprocessor(numeric_features, categorical_features, scale_numeric=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers = []
    if numeric_features:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_features))
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )

    return ColumnTransformer(transformers=transformers)


def build_models(numeric_features, categorical_features):
    return {
        "Logistic Regression": Pipeline(
            steps=[
                (
                    "preprocess",
                    build_preprocessor(
                        numeric_features, categorical_features, scale_numeric=True
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                (
                    "preprocess",
                    build_preprocessor(numeric_features, categorical_features),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def score_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_prob)
        if y_test.nunique() == 2
        else np.nan,
    }
    return metrics, y_pred, y_pred_prob


def train_and_evaluate(df, output_dir, test_size=0.30):
    numeric_features, categorical_features = select_model_features(df)
    features = numeric_features + categorical_features

    X = df[features]
    y = df[TARGET_COLUMN]

    if y.nunique() != 2:
        raise ValueError("The churn target must contain exactly two classes.")

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    model_results = []
    fitted_models = {}

    for model_name, model in build_models(numeric_features, categorical_features).items():
        model.fit(X_train, y_train)
        metrics, y_pred, _ = score_model(model, X_test, y_test)
        fitted_models[model_name] = model
        model_results.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "roc_auc": metrics["roc_auc"],
            }
        )
        print(f"\n{model_name} Performance")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
        print(classification_report(y_test, y_pred, zero_division=0))

    metrics_df = pd.DataFrame(model_results).sort_values(
        by=["roc_auc", "accuracy"], ascending=False
    )
    best_model_name = metrics_df.iloc[0]["model"]
    best_model = fitted_models[best_model_name]
    best_y_pred = best_model.predict(X_test)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)
    pd.DataFrame(
        confusion_matrix(y_test, best_y_pred),
        index=["actual_retained", "actual_churned"],
        columns=["predicted_retained", "predicted_churned"],
    ).to_csv(output_dir / "confusion_matrix.csv")

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as report_file:
        report_file.write(f"Best model: {best_model_name}\n\n")
        report_file.write(classification_report(y_test, best_y_pred, zero_division=0))

    return best_model_name, best_model, features, metrics_df


# -------------------------------
# 4. Tableau Export
# -------------------------------
def export_for_tableau(df, model, features, model_name, output_dir):
    tableau_df = df.copy()
    tableau_df["model_name"] = model_name
    tableau_df["predicted_churn"] = model.predict(tableau_df[features])
    tableau_df["predicted_churn_probability"] = model.predict_proba(
        tableau_df[features]
    )[:, 1]

    cleaned_path = output_dir / "cleaned_data.csv"
    tableau_path = output_dir / "final_data_for_tableau.csv"
    df.to_csv(cleaned_path, index=False)
    tableau_df.to_csv(tableau_path, index=False)

    print(f"\nCleaned data saved to: {cleaned_path}")
    print(f"Tableau export saved to: {tableau_path}")


# -------------------------------
# 5. Run Pipeline
# -------------------------------
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    customer_data, subscription_data = load_data(args.customers, args.subscriptions)
    merged_data = merge_and_prepare(customer_data, subscription_data)

    best_model_name, best_model, features, metrics_df = train_and_evaluate(
        merged_data, output_dir, test_size=args.test_size
    )
    export_for_tableau(merged_data, best_model, features, best_model_name, output_dir)

    print("\nModel comparison:")
    print(metrics_df.to_string(index=False))
    print(f"\nSelected model for Tableau export: {best_model_name}")


if __name__ == "__main__":
    main()
