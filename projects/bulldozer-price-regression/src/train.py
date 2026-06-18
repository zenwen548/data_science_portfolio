from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TUNED_XGB_PARAMS = {
    "subsample": 0.7,
    "n_estimators": 200,
    "max_depth": 10,
    "learning_rate": 0.05,
    "colsample_bytree": 0.5,
    "n_jobs": -1,
    "random_state": 42,
}


def add_sale_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["saledate"] = pd.to_datetime(df["saledate"])
    df["saleYear"] = df["saledate"].dt.year
    df["saleMonth"] = df["saledate"].dt.month
    df["saleDay"] = df["saledate"].dt.day
    df["saleDayOfWeek"] = df["saledate"].dt.dayofweek
    df["saleDayOfYear"] = df["saledate"].dt.dayofyear
    return df.drop("saledate", axis=1)


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                df[f"{label}_is_missing"] = pd.isnull(content)
                df[label] = content.fillna(content.median())
        else:
            df[f"{label}_is_missing"] = pd.isnull(content)
            df[label] = pd.Categorical(content).codes + 1

    return df


def prepare_train_validation(
    data: pd.DataFrame,
    validation_year: int = 2012,
    target: str = "SalePrice",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    prepared = add_sale_date_features(data)
    df_train = prepared[prepared.saleYear != validation_year]
    df_valid = prepared[prepared.saleYear == validation_year]

    if df_train.empty or df_valid.empty:
        raise ValueError(
            f"Expected records before and during validation year {validation_year}."
        )

    X_train = df_train.drop(target, axis=1)
    y_train = df_train[target]
    X_valid = df_valid.drop(target, axis=1)
    y_valid = df_valid[target]

    X_train = preprocess_features(X_train)
    train_columns = X_train.columns
    X_valid = preprocess_features(X_valid)
    X_valid = X_valid.reindex(columns=train_columns, fill_value=0)

    return X_train, y_train, X_valid, y_valid


def rmsle(y_true: Any, y_pred: Any) -> float:
    y_true_array = np.asarray(y_true)
    y_pred_array = np.maximum(0, np.asarray(y_pred))
    return float(np.sqrt(np.mean((np.log1p(y_pred_array) - np.log1p(y_true_array)) ** 2)))


def regression_scores(y_true: Any, y_pred: Any) -> dict[str, float]:
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    residuals = y_true_array - y_pred_array
    total = y_true_array - y_true_array.mean()

    return {
        "mae": float(np.mean(np.abs(residuals))),
        "rmsle": rmsle(y_true_array, y_pred_array),
        "r2": float(1 - np.sum(residuals**2) / np.sum(total**2)),
    }


def evaluate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[dict[str, float], np.ndarray]:
    train_preds = model.predict(X_train)
    valid_preds = model.predict(X_valid)
    train_scores = regression_scores(y_train, train_preds)
    valid_scores = regression_scores(y_valid, valid_preds)

    return {
        "Training MAE": train_scores["mae"],
        "Valid MAE": valid_scores["mae"],
        "Training RMSLE": train_scores["rmsle"],
        "Valid RMSLE": valid_scores["rmsle"],
        "Training R2": train_scores["r2"],
        "Valid R2": valid_scores["r2"],
    }, valid_preds


def build_model(params: dict[str, Any] | None = None) -> Any:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise SystemExit(
            "xgboost is required for training. Install the repo requirements first."
        ) from exc

    model_params = dict(TUNED_XGB_PARAMS)
    if params:
        model_params.update(params)
    return XGBRegressor(**model_params)


def feature_importance_frame(model: Any, columns: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {"Feature": columns, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)


def write_outputs(
    output_dir: Path,
    model: Any,
    X_train: pd.DataFrame,
    y_valid: pd.Series,
    valid_preds: np.ndarray,
    metrics: dict[str, float],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.csv"
    feature_importance_path = output_dir / "feature_importance.csv"
    predictions_path = output_dir / "validation_predictions.csv"
    model_path = output_dir / "xgb_model.json"

    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    feature_importance_frame(model, X_train.columns).to_csv(feature_importance_path, index=False)
    pd.DataFrame(
        {"actual_sale_price": y_valid.values, "predicted_sale_price": valid_preds}
    ).to_csv(predictions_path, index=False)
    model.save_model(model_path)

    return {
        "metrics": metrics_path,
        "feature_importance": feature_importance_path,
        "predictions": predictions_path,
        "model": model_path,
    }


def log_mlflow_run(
    model: Any,
    metrics: dict[str, float],
    artifacts: dict[str, Path],
    tracking_uri: str | None,
    experiment_name: str,
    run_name: str,
) -> None:
    try:
        import mlflow
        import mlflow.xgboost
    except ImportError as exc:
        raise SystemExit(
            "mlflow is required for experiment tracking. Install the repo requirements first."
        ) from exc

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(TUNED_XGB_PARAMS)
        mlflow.log_metrics({key.replace(" ", "_").lower(): value for key, value in metrics.items()})
        mlflow.log_artifact(str(artifacts["metrics"]))
        mlflow.log_artifact(str(artifacts["feature_importance"]))
        mlflow.log_artifact(str(artifacts["predictions"]))
        mlflow.xgboost.log_model(model, artifact_path="model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the tuned XGBoost bulldozer price model with optional MLflow tracking."
    )
    parser.add_argument("--data", default="data/TrainAndValid.csv", help="Path to Kaggle train/valid CSV.")
    parser.add_argument(
        "--output-dir",
        default="projects/bulldozer-price-regression/outputs",
        help="Directory for metrics, feature importance, predictions, and model export.",
    )
    parser.add_argument("--validation-year", type=int, default=2012)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--experiment-name", default="bulldozer-price-regression")
    parser.add_argument("--run-name", default="tuned-xgb")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)

    if not data_path.exists():
        raise SystemExit(
            f"Data file not found: {data_path}. Download the Kaggle data and place it in data/."
        )

    data = pd.read_csv(data_path, low_memory=False)
    X_train, y_train, X_valid, y_valid = prepare_train_validation(
        data, validation_year=args.validation_year
    )

    model = build_model()
    model.fit(X_train, y_train)
    metrics, valid_preds = evaluate_model(model, X_train, y_train, X_valid, y_valid)
    artifacts = write_outputs(output_dir, model, X_train, y_valid, valid_preds, metrics)

    if not args.no_mlflow:
        log_mlflow_run(
            model=model,
            metrics=metrics,
            artifacts=artifacts,
            tracking_uri=args.mlflow_tracking_uri,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
        )

    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")
    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
