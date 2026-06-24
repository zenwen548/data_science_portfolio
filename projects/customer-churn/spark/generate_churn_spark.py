"""PySpark synthetic churn generator for large-scale portfolio demos."""

from pathlib import Path
import argparse


DEFAULT_ROW_COUNT = 10_000_000
DEFAULT_SEED = 42
DEFAULT_SAMPLE_ROWS = 1_000_000
DEFAULT_OUTPUT_DIR = "outputs/churn_spark/generated"

CUSTOMER_SCHEMA_COLUMNS = [
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
]

SUBSCRIPTION_SCHEMA_COLUMNS = [
    "customer_id",
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
]


def require_pyspark():
    try:
        from pyspark.sql import SparkSession, functions as F
    except ImportError as exc:
        raise RuntimeError(
            "PySpark is required for the scale generator. "
            "Install requirements.txt or run the customer-churn Docker image."
        ) from exc
    return SparkSession, F


def create_spark_session(app_name="customer-churn-synthetic-generator"):
    SparkSession, _ = require_pyspark()
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "96")
        .getOrCreate()
    )


def build_customer_subscription_frames(spark, row_count=DEFAULT_ROW_COUNT, seed=DEFAULT_SEED):
    _, F = require_pyspark()
    base = spark.range(1, row_count + 1).withColumnRenamed("id", "row_id")
    base = (
        base.withColumn("region_roll", F.rand(seed))
        .withColumn("segment_roll", F.rand(seed + 1))
        .withColumn("contract_roll", F.rand(seed + 2))
        .withColumn("payment_roll", F.rand(seed + 3))
        .withColumn("internet_roll", F.rand(seed + 4))
        .withColumn("tech_roll", F.rand(seed + 5))
        .withColumn("paperless_roll", F.rand(seed + 6))
        .withColumn("partner_roll", F.rand(seed + 7))
        .withColumn("dependent_roll", F.rand(seed + 8))
        .withColumn("senior_roll", F.rand(seed + 9))
        .withColumn("addon_roll", F.rand(seed + 10))
        .withColumn("charge_noise", F.rand(seed + 11))
        .withColumn("risk_noise", F.randn(seed + 12))
        .withColumn("churn_roll", F.rand(seed + 13))
        .withColumn("tenure_roll", F.rand(seed + 14))
        .withColumn("start_roll", F.rand(seed + 15))
        .withColumn("support_roll", F.rand(seed + 16))
        .withColumn("transaction_roll", F.rand(seed + 17))
        .withColumn("referral_roll", F.rand(seed + 18))
    )

    shaped = (
        base.withColumn(
            "customer_id", F.format_string("C%08d", F.col("row_id").cast("int"))
        )
        .withColumn(
            "subscription_id", F.format_string("S%08d", F.col("row_id").cast("int"))
        )
        .withColumn(
            "region",
            F.when(F.col("region_roll") < 0.25, "North")
            .when(F.col("region_roll") < 0.50, "South")
            .when(F.col("region_roll") < 0.75, "East")
            .otherwise("West"),
        )
        .withColumn(
            "customer_segment",
            F.when(F.col("segment_roll") < 0.46, "Small Business")
            .when(F.col("segment_roll") < 0.80, "Mid-Market")
            .otherwise("Enterprise"),
        )
        .withColumn(
            "contract_type",
            F.when(F.col("contract_roll") < 0.50, "Month-to-month")
            .when(F.col("contract_roll") < 0.80, "One year")
            .otherwise("Two year"),
        )
        .withColumn(
            "payment_method",
            F.when(F.col("payment_roll") < 0.34, "Electronic check")
            .when(F.col("payment_roll") < 0.62, "Credit card")
            .when(F.col("payment_roll") < 0.87, "Bank transfer")
            .otherwise("Mailed check"),
        )
        .withColumn(
            "internet_service",
            F.when(F.col("internet_roll") < 0.48, "Fiber optic")
            .when(F.col("internet_roll") < 0.90, "DSL")
            .otherwise("None"),
        )
        .withColumn(
            "tech_support", F.when(F.col("tech_roll") < 0.44, "Yes").otherwise("No")
        )
        .withColumn(
            "paperless_billing",
            F.when(F.col("paperless_roll") < 0.64, "Yes").otherwise("No"),
        )
        .withColumn(
            "has_partner", F.when(F.col("partner_roll") < 0.48, "Yes").otherwise("No")
        )
        .withColumn(
            "has_dependents",
            F.when(F.col("dependent_roll") < 0.30, "Yes").otherwise("No"),
        )
        .withColumn("senior_citizen", (F.col("senior_roll") < 0.16).cast("int"))
        .withColumn("num_addon_services", F.floor(F.col("addon_roll") * 6).cast("int"))
        .withColumn(
            "state_province",
            F.when(
                F.col("region") == "North",
                F.element_at(
                    F.array(F.lit("IL"), F.lit("MI"), F.lit("MN"), F.lit("WI")),
                    (F.pmod(F.col("row_id"), F.lit(4)) + 1).cast("int"),
                ),
            )
            .when(
                F.col("region") == "South",
                F.element_at(
                    F.array(F.lit("FL"), F.lit("GA"), F.lit("NC"), F.lit("TX")),
                    (F.pmod(F.col("row_id"), F.lit(4)) + 1).cast("int"),
                ),
            )
            .when(
                F.col("region") == "East",
                F.element_at(
                    F.array(F.lit("MA"), F.lit("NY"), F.lit("PA"), F.lit("VA")),
                    (F.pmod(F.col("row_id"), F.lit(4)) + 1).cast("int"),
                ),
            )
            .otherwise(
                F.element_at(
                    F.array(F.lit("AZ"), F.lit("CA"), F.lit("OR"), F.lit("WA")),
                    (F.pmod(F.col("row_id"), F.lit(4)) + 1).cast("int"),
                )
            ),
        )
        .withColumn(
            "monthly_charges",
            F.round(
                F.greatest(
                    F.lit(25.0),
                    F.least(
                        F.lit(150.0),
                        F.lit(68.0)
                        + (F.col("charge_noise") - F.lit(0.5)) * F.lit(32.0)
                        + F.when(F.col("internet_service") == "Fiber optic", 18).otherwise(0)
                        - F.when(F.col("internet_service") == "None", 22).otherwise(0)
                        + F.col("num_addon_services") * F.lit(4.5),
                    ),
                ),
                2,
            ),
        )
        .withColumn(
            "risk_score",
            F.lit(-0.65)
            + F.when(F.col("contract_type") == "Month-to-month", 1.25).otherwise(0)
            + F.when(F.col("contract_type") == "Two year", -0.95).otherwise(0)
            + F.when(F.col("payment_method") == "Electronic check", 0.62).otherwise(0)
            + F.when(F.col("internet_service") == "Fiber optic", 0.52).otherwise(0)
            + F.when(F.col("tech_support") == "No", 0.68).otherwise(-0.32)
            + F.when(F.col("paperless_billing") == "Yes", 0.30).otherwise(0)
            + F.col("senior_citizen") * F.lit(0.28)
            - F.when(F.col("has_partner") == "Yes", 0.26).otherwise(0)
            - F.when(F.col("has_dependents") == "Yes", 0.25).otherwise(0)
            + F.col("num_addon_services") * F.lit(0.08)
            + ((F.col("monthly_charges") - F.lit(68.0)) / F.lit(24.0)) * F.lit(0.30)
            + F.col("risk_noise") * F.lit(0.62),
        )
        .withColumn(
            "churn_probability", F.lit(1.0) / (F.lit(1.0) + F.exp(-F.col("risk_score")))
        )
        .withColumn("churned", (F.col("churn_roll") < F.col("churn_probability")).cast("int"))
        .withColumn("tenure_days", (F.floor(F.col("tenure_roll") * 1740) + 60).cast("int"))
        .withColumn(
            "subscription_start_date",
            F.date_add(
                F.to_date(F.lit("2020-01-01")),
                F.floor(F.col("start_roll") * 1500).cast("int"),
            ),
        )
        .withColumn(
            "subscription_end_date",
            F.when(
                F.col("churned") == 1,
                F.date_add(F.col("subscription_start_date"), F.col("tenure_days")),
            ),
        )
        .withColumn("subscription_cost", F.round(F.col("monthly_charges"), 0).cast("int"))
        .withColumn(
            "num_support_calls",
            F.least(
                F.lit(12),
                F.floor(
                    F.col("support_roll") * 8
                    + F.when(F.col("tech_support") == "No", 1).otherwise(0)
                    + F.when(F.col("contract_type") == "Month-to-month", 1).otherwise(0)
                ).cast("int"),
            ),
        )
        .withColumn(
            "support_calls_per_day",
            F.round(F.col("num_support_calls") / F.col("tenure_days"), 4),
        )
        .withColumn(
            "avg_monthly_transactions",
            F.greatest(
                F.lit(3),
                F.round(
                    F.lit(24)
                    + F.when(F.col("customer_segment") == "Enterprise", 18).otherwise(0)
                    + F.when(F.col("customer_segment") == "Mid-Market", 7).otherwise(0)
                    - F.col("churned") * 2
                    + (F.col("transaction_roll") - F.lit(0.5)) * 16,
                    0,
                ).cast("int"),
            ),
        )
        .withColumn(
            "avg_transactions_per_day",
            F.round(F.col("avg_monthly_transactions") / F.lit(30.0), 4),
        )
        .withColumn("was_referred", (F.col("referral_roll") < 0.24).cast("int"))
        .withColumn(
            "referral_code",
            F.when(
                F.col("was_referred") == 1,
                F.format_string(
                    "REF%03d", (F.pmod(F.col("row_id"), F.lit(900)) + 100).cast("int")
                ),
            ).otherwise(""),
        )
        .withColumn(
            "account_type",
            F.when(F.col("contract_type").isin("One year", "Two year"), "Annual").otherwise(
                "Monthly"
            ),
        )
    )

    customers = shaped.select(*CUSTOMER_SCHEMA_COLUMNS)
    subscriptions = shaped.select(*SUBSCRIPTION_SCHEMA_COLUMNS)
    return customers, subscriptions


def write_generated_data(customers, subscriptions, output_dir, partition_count=None):
    output_path = Path(output_dir)
    customer_writer = customers
    subscription_writer = subscriptions
    if partition_count:
        customer_writer = customer_writer.repartition(partition_count)
        subscription_writer = subscription_writer.repartition(partition_count)

    customer_writer.write.mode("overwrite").parquet(str(output_path / "customers"))
    subscription_writer.write.mode("overwrite").parquet(str(output_path / "subscriptions"))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Spark-scale synthetic churn data.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROW_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--partitions", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    spark = create_spark_session()
    try:
        customers, subscriptions = build_customer_subscription_frames(
            spark, row_count=args.rows, seed=args.seed
        )
        write_generated_data(
            customers,
            subscriptions,
            args.output_dir,
            partition_count=args.partitions,
        )
        print(
            f"Wrote {args.rows} Spark-generated customers and subscriptions to {args.output_dir}"
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
