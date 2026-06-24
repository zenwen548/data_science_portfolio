"""Spark MLlib churn training and Tableau serving export."""

from pathlib import Path
import argparse
import json


DEFAULT_GENERATED_DATA_DIR = "outputs/churn_spark/generated"
DEFAULT_OUTPUT_DIR = "outputs/churn_spark/serving"
DEFAULT_SAMPLE_LIMIT = 1_000_000
TARGET_COLUMN = "churned"

EXPORT_CONTRACT_COLUMNS = [
    "customer_id",
    "customer_segment",
    "region",
    "account_type",
    "state_province",
    "contract_type",
    "payment_method",
    "internet_service",
    "tech_support",
    "num_addon_services",
    "senior_citizen",
    "has_partner",
    "has_dependents",
    "paperless_billing",
    "subscription_id",
    "subscription_start_date",
    "subscription_end_date",
    "subscription_cost",
    "monthly_charges",
    "num_support_calls",
    "support_calls_per_day",
    "referral_code",
    "avg_monthly_transactions",
    "avg_transactions_per_day",
    "tenure_days",
    "was_referred",
    "churned",
    "predicted_churn",
    "predicted_churn_probability",
    "churn_risk_tier",
    "revenue_at_risk",
    "impacted_customer_flag",
]

ROLLUP_GROUP_COLUMNS = ["region", "contract_type", "churn_risk_tier"]

LEAKAGE_EXACT_COLUMNS = {
    "subscription_end_date",
    "is_active",
    "tenure_days",
    "years_since_signup",
    "support_calls_per_day",
    "avg_transactions_per_day",
    "referral_code",
}

ID_COLUMNS = {"customer_id", "subscription_id"}


def require_pyspark():
    try:
        from pyspark.ml import Pipeline
        from pyspark.ml.classification import GBTClassifier, LogisticRegression
        from pyspark.ml.evaluation import BinaryClassificationEvaluator
        from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
        from pyspark.sql import SparkSession, functions as F, types as T
    except ImportError as exc:
        raise RuntimeError(
            "PySpark is required for the scale training job. "
            "Install requirements.txt or run the customer-churn Docker image."
        ) from exc
    return {
        "Pipeline": Pipeline,
        "GBTClassifier": GBTClassifier,
        "LogisticRegression": LogisticRegression,
        "BinaryClassificationEvaluator": BinaryClassificationEvaluator,
        "OneHotEncoder": OneHotEncoder,
        "StringIndexer": StringIndexer,
        "VectorAssembler": VectorAssembler,
        "SparkSession": SparkSession,
        "F": F,
        "T": T,
    }


def create_spark_session(app_name="customer-churn-spark-training"):
    spark_api = require_pyspark()
    return (
        spark_api["SparkSession"].builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "96")
        .getOrCreate()
    )


def load_generated_data(spark, generated_data_dir=DEFAULT_GENERATED_DATA_DIR):
    data_dir = Path(generated_data_dir)
    customers_path = data_dir / "customers"
    subscriptions_path = data_dir / "subscriptions"
    missing = [
        str(path)
        for path in [customers_path, subscriptions_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing Spark-generated input path(s): "
            + ", ".join(missing)
            + ". Run generate_churn_spark.py first."
        )
    customers = spark.read.parquet(str(customers_path))
    subscriptions = spark.read.parquet(str(subscriptions_path))
    return customers, subscriptions


def merge_customer_subscription_data(customers, subscriptions):
    return customers.join(subscriptions, on="customer_id", how="right")


def select_model_features(df):
    numeric_types = {"byte", "short", "int", "bigint", "float", "double", "decimal"}
    excluded = set(LEAKAGE_EXACT_COLUMNS) | set(ID_COLUMNS) | {TARGET_COLUMN}
    excluded.update(column for column in df.columns if column.endswith("_id"))
    excluded.update(column for column in df.columns if column.endswith("_date"))
    excluded.update(column for column in df.columns if column.endswith("_per_day"))
    excluded.update(column for column in df.columns if "tenure" in column)

    numeric_features = []
    categorical_features = []
    for field in df.schema.fields:
        if field.name in excluded:
            continue
        type_name = field.dataType.simpleString().split("(")[0]
        if type_name in numeric_types:
            numeric_features.append(field.name)
        else:
            categorical_features.append(field.name)

    if not numeric_features and not categorical_features:
        raise ValueError("No usable Spark model features were found.")
    return numeric_features, categorical_features


def build_pipeline(numeric_features, categorical_features, model_type="logistic"):
    spark_api = require_pyspark()
    stages = []
    encoded_columns = []

    for column in categorical_features:
        indexed = f"{column}_indexed"
        encoded = f"{column}_encoded"
        stages.append(
            spark_api["StringIndexer"](
                inputCol=column,
                outputCol=indexed,
                handleInvalid="keep",
            )
        )
        stages.append(
            spark_api["OneHotEncoder"](
                inputCols=[indexed],
                outputCols=[encoded],
                handleInvalid="keep",
            )
        )
        encoded_columns.append(encoded)

    feature_columns = numeric_features + encoded_columns
    stages.append(
        spark_api["VectorAssembler"](
            inputCols=feature_columns,
            outputCol="features",
            handleInvalid="keep",
        )
    )

    if model_type == "gbt":
        classifier = spark_api["GBTClassifier"](
            labelCol=TARGET_COLUMN,
            featuresCol="features",
            probabilityCol="probability",
            predictionCol="prediction",
            seed=42,
            maxIter=40,
            maxDepth=5,
        )
    else:
        classifier = spark_api["LogisticRegression"](
            labelCol=TARGET_COLUMN,
            featuresCol="features",
            probabilityCol="probability",
            predictionCol="prediction",
            maxIter=50,
            regParam=0.02,
            elasticNetParam=0.0,
        )
    stages.append(classifier)
    return spark_api["Pipeline"](stages=stages)


def add_tableau_contract_columns(predictions):
    spark_api = require_pyspark()
    F = spark_api["F"]
    T = spark_api["T"]

    probability_at_one = F.udf(lambda vector: float(vector[1]), T.DoubleType())
    scored = predictions.withColumn(
        "predicted_churn_probability", probability_at_one(F.col("probability"))
    )
    scored = (
        scored.withColumn("predicted_churn", F.col("prediction").cast("int"))
        .withColumn(
            "churn_risk_tier",
            F.when(F.col("predicted_churn_probability") >= 0.70, "High")
            .when(F.col("predicted_churn_probability") >= 0.40, "Medium")
            .otherwise("Low"),
        )
        .withColumn(
            "revenue_at_risk",
            F.round(F.col("monthly_charges") * F.col("predicted_churn_probability"), 2),
        )
        .withColumn(
            "impacted_customer_flag",
            (F.col("predicted_churn_probability") >= 0.70).cast("int"),
        )
    )
    return scored.select(*EXPORT_CONTRACT_COLUMNS)


def write_serving_outputs(scored, output_dir, sample_limit=DEFAULT_SAMPLE_LIMIT):
    spark_api = require_pyspark()
    F = spark_api["F"]
    output_path = Path(output_dir)

    rollups = scored.groupBy(*ROLLUP_GROUP_COLUMNS).agg(
        F.count("*").alias("customer_count"),
        F.round(F.avg("predicted_churn_probability"), 4).alias("avg_churn_probability"),
        F.round(F.sum("revenue_at_risk"), 2).alias("total_revenue_at_risk"),
        F.sum("impacted_customer_flag").alias("impacted_customer_count"),
    )
    sampled = scored.orderBy(F.desc("revenue_at_risk")).limit(sample_limit)

    rollups.write.mode("overwrite").csv(
        str(output_path / "aggregated_rollups"), header=True
    )
    sampled.write.mode("overwrite").csv(
        str(output_path / "sampled_tableau_extract"), header=True
    )
    return rollups, sampled


def train_and_export(
    spark,
    generated_data_dir=DEFAULT_GENERATED_DATA_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    sample_limit=DEFAULT_SAMPLE_LIMIT,
    model_type="logistic",
):
    spark_api = require_pyspark()
    customers, subscriptions = load_generated_data(spark, generated_data_dir)
    merged = merge_customer_subscription_data(customers, subscriptions).dropna(
        subset=[TARGET_COLUMN]
    )
    numeric_features, categorical_features = select_model_features(merged)
    train_df, test_df = merged.randomSplit([0.80, 0.20], seed=42)
    pipeline = build_pipeline(numeric_features, categorical_features, model_type=model_type)
    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)

    evaluator = spark_api["BinaryClassificationEvaluator"](
        labelCol=TARGET_COLUMN,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    test_auc = evaluator.evaluate(predictions)
    scored = add_tableau_contract_columns(predictions)
    write_serving_outputs(scored, output_dir, sample_limit=sample_limit)

    metrics = {
        "model_type": model_type,
        "test_area_under_roc": test_auc,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "serving_outputs": [
            "aggregated_rollups",
            "sampled_tableau_extract",
        ],
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "spark_metrics.json", "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Spark MLlib churn model and write Tableau serving extracts."
    )
    parser.add_argument("--generated-data-dir", default=DEFAULT_GENERATED_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-limit", type=int, default=DEFAULT_SAMPLE_LIMIT)
    parser.add_argument("--model-type", choices=["logistic", "gbt"], default="logistic")
    return parser.parse_args()


def main():
    args = parse_args()
    spark = create_spark_session()
    try:
        metrics = train_and_export(
            spark,
            generated_data_dir=args.generated_data_dir,
            output_dir=args.output_dir,
            sample_limit=args.sample_limit,
            model_type=args.model_type,
        )
        print(json.dumps(metrics, indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
