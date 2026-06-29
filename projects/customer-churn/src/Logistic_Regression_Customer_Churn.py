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
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_CUSTOMERS_PATH = "projects/customer-churn/data/sample_customers.csv"
DEFAULT_SUBSCRIPTIONS_PATH = "projects/customer-churn/data/sample_subscriptions.csv"
DEFAULT_OUTPUT_DIR = "outputs/churn"
TARGET_COLUMN = "churned"

LEAKAGE_COLUMNS = {
    "subscription_end_date",
    "is_active",
    "tenure_days",
    "years_since_signup",
    "support_calls_per_day",
    "avg_transactions_per_day",
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
    excluded.update(column for column in df.columns if column.endswith("_per_day"))
    excluded.update(column for column in df.columns if "tenure" in column)

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


def single_feature_auc_report(df, features, target_column=TARGET_COLUMN):
    y = df[target_column]
    rows = []
    for feature in features:
        series = df[feature]
        if series.nunique(dropna=True) < 2 or y.nunique() != 2:
            auc = np.nan
        elif pd.api.types.is_numeric_dtype(series):
            scored = series.fillna(series.median())
            auc = roc_auc_score(y, scored)
        else:
            encoded = pd.Categorical(series.fillna("__missing__")).codes
            auc = roc_auc_score(y, encoded)
        if not pd.isna(auc):
            auc = max(float(auc), float(1 - auc))
        rows.append({"feature": feature, "single_feature_auc": auc})
    return pd.DataFrame(rows).sort_values(
        by="single_feature_auc", ascending=False, na_position="last"
    )


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

    cv_splits = min(5, int(y.value_counts().min()))
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    for model_name, model in build_models(numeric_features, categorical_features).items():
        oof_prob = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            method="predict_proba",
        )[:, 1]
        oof_auc = roc_auc_score(y, oof_prob)
        model.fit(X_train, y_train)
        metrics, y_pred, _ = score_model(model, X_test, y_test)
        fitted_models[model_name] = model
        model_results.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "roc_auc": metrics["roc_auc"],
                "oof_roc_auc": oof_auc,
            }
        )
        print(f"\n{model_name} Performance")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
        print(f"OOF ROC-AUC: {oof_auc:.3f}")
        print(classification_report(y_test, y_pred, zero_division=0))

    metrics_df = pd.DataFrame(model_results).sort_values(
        by=["oof_roc_auc", "roc_auc", "accuracy"], ascending=False
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
def risk_tier(probability):
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


def write_qa_packet(tableau_df, leakage_report, output_dir):
    probability_summary = pd.DataFrame(
        {
            "metric": [
                "rows",
                "mean_probability",
                "median_probability",
                "min_probability",
                "max_probability",
                "extreme_probability_rate",
            ],
            "value": [
                len(tableau_df),
                tableau_df["predicted_churn_probability"].mean(),
                tableau_df["predicted_churn_probability"].median(),
                tableau_df["predicted_churn_probability"].min(),
                tableau_df["predicted_churn_probability"].max(),
                (
                    (tableau_df["predicted_churn_probability"] == 0)
                    | (tableau_df["predicted_churn_probability"] == 1)
                ).mean(),
            ],
        }
    )
    risk_counts = tableau_df["churn_risk_tier"].value_counts().rename_axis(
        "churn_risk_tier"
    ).reset_index(name="row_count")
    top_impacted = tableau_df.sort_values(
        by="revenue_at_risk", ascending=False
    ).head(10)

    probability_summary.to_csv(output_dir / "probability_summary.csv", index=False)
    risk_counts.to_csv(output_dir / "risk_tier_counts.csv", index=False)
    top_impacted.to_csv(output_dir / "top_impacted_customers.csv", index=False)

    with open(output_dir / "qa_packet.md", "w", encoding="utf-8") as qa_file:
        qa_file.write("# Customer Churn QA Packet\n\n")
        qa_file.write("## Probability Summary\n\n")
        qa_file.write(probability_summary.to_string(index=False))
        qa_file.write("\n\n## Risk Tier Counts\n\n")
        qa_file.write(risk_counts.to_string(index=False))
        qa_file.write("\n\n## Highest Single-Feature AUC\n\n")
        qa_file.write(leakage_report.head(10).to_string(index=False))
        qa_file.write("\n")


def export_for_tableau(df, model, features, model_name, output_dir):
    tableau_df = df.copy()
    X = tableau_df[features]
    y = tableau_df[TARGET_COLUMN]
    cv_splits = min(5, int(y.value_counts().min()))
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    oof_prob = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    tableau_df["model_name"] = model_name
    tableau_df["predicted_churn_probability"] = oof_prob
    tableau_df["predicted_churn"] = (oof_prob >= 0.5).astype(int)
    tableau_df["churn_risk_tier"] = tableau_df[
        "predicted_churn_probability"
    ].apply(risk_tier)
    charge_column = (
        "monthly_charges" if "monthly_charges" in tableau_df.columns else "subscription_cost"
    )
    tableau_df["revenue_at_risk"] = (
        tableau_df[charge_column] * tableau_df["predicted_churn_probability"]
    ).round(2)
    tableau_df["impacted_customer_flag"] = (
        tableau_df["predicted_churn_probability"] >= 0.70
    ).astype(int)
    tableau_df["predicted_probability"] = tableau_df[
        "predicted_churn_probability"
    ]

    # Keep the legacy Tableau workbook contract stable while the model/export logic
    # evolves. Tableau binds formulas and sorts to field names and inferred types.
    tableau_df["customer_code"] = tableau_df["customer_id"]
    tableau_df["subscription_code"] = tableau_df["subscription_id"]
    tableau_df["customer_id"] = (
        tableau_df["customer_code"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(int)
    )
    tableau_df["subscription_id"] = (
        tableau_df["subscription_code"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(int)
    )
    tableau_df["churned"] = tableau_df["churned"].astype(bool)
    tableau_df["predicted_churn"] = tableau_df["predicted_churn"].astype(bool)
    if "name" not in tableau_df.columns:
        tableau_df["name"] = tableau_df["customer_code"].map(
            lambda value: f"Customer {value}"
        )
    if "email" not in tableau_df.columns:
        tableau_df["email"] = tableau_df["customer_code"].map(
            lambda value: f"{str(value).lower()}@example.com"
        )
    if "signup_date" not in tableau_df.columns:
        tableau_df["signup_date"] = tableau_df["subscription_start_date"]
    if "years_since_signup" not in tableau_df.columns:
        tableau_df["years_since_signup"] = (tableau_df["tenure_days"] / 365.25).round(
            2
        )
    if "is_active" not in tableau_df.columns:
        tableau_df["is_active"] = (~tableau_df["churned"]).astype(int)

    leakage_report = single_feature_auc_report(tableau_df, features)
    legacy_tableau_columns = [
        "customer_id",
        "name",
        "email",
        "signup_date",
        "years_since_signup",
        "subscription_id",
        "subscription_start_date",
        "subscription_end_date",
        "subscription_cost",
        "referral_code",
        "churned",
        "tenure_days",
        "avg_monthly_transactions",
        "avg_transactions_per_day",
        "num_support_calls",
        "support_calls_per_day",
        "region",
        "state_province",
        "is_active",
        "was_referred",
        "predicted_churn",
        "predicted_probability",
    ]
    extra_columns = [
        column for column in tableau_df.columns if column not in legacy_tableau_columns
    ]
    tableau_df = tableau_df[legacy_tableau_columns + extra_columns]

    cleaned_path = output_dir / "cleaned_data.csv"
    tableau_path = output_dir / "final_data_for_tableau.csv"
    leakage_path = output_dir / "single_feature_auc.csv"
    df.to_csv(cleaned_path, index=False)
    tableau_df.to_csv(tableau_path, index=False)
    leakage_report.to_csv(leakage_path, index=False)
    write_qa_packet(tableau_df, leakage_report, output_dir)

    print(f"\nCleaned data saved to: {cleaned_path}")
    print(f"Tableau export saved to: {tableau_path}")
    print(f"Leakage report saved to: {leakage_path}")


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
