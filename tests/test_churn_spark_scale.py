import importlib.util
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
CHURN_PROJECT = REPO_ROOT / "projects" / "customer-churn"
SPARK_DIR = CHURN_PROJECT / "spark"
SPARK_GENERATOR_SCRIPT = SPARK_DIR / "generate_churn_spark.py"
SPARK_TRAINING_SCRIPT = SPARK_DIR / "train_churn_spark.py"
SPARK_DOCKERFILE = CHURN_PROJECT / "docker" / "Dockerfile"
REQUIREMENTS_FILE = REPO_ROOT / "requirements.txt"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ChurnSparkScaleTests(unittest.TestCase):
    def test_spark_generator_declares_schema_parity_and_scale_defaults(self):
        generator = load_module("churn_spark_generator", SPARK_GENERATOR_SCRIPT)

        required_customer_columns = {
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
        }
        required_subscription_columns = {
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
        }

        self.assertGreaterEqual(generator.DEFAULT_ROW_COUNT, 10_000_000)
        self.assertLessEqual(generator.DEFAULT_SAMPLE_ROWS, 1_000_000)
        self.assertTrue(
            required_customer_columns.issubset(generator.CUSTOMER_SCHEMA_COLUMNS)
        )
        self.assertTrue(
            required_subscription_columns.issubset(
                generator.SUBSCRIPTION_SCHEMA_COLUMNS
            )
        )

    def test_spark_training_declares_tableau_export_contract(self):
        training = load_module("churn_spark_training", SPARK_TRAINING_SCRIPT)

        required_export_columns = {
            "customer_id",
            "region",
            "contract_type",
            "churned",
            "predicted_churn",
            "predicted_churn_probability",
            "churn_risk_tier",
            "revenue_at_risk",
            "impacted_customer_flag",
        }
        required_rollup_columns = {"region", "contract_type", "churn_risk_tier"}

        self.assertTrue(
            required_export_columns.issubset(training.EXPORT_CONTRACT_COLUMNS)
        )
        self.assertEqual(
            required_rollup_columns, set(training.ROLLUP_GROUP_COLUMNS)
        )
        self.assertLessEqual(training.DEFAULT_SAMPLE_LIMIT, 1_000_000)

    def test_spark_dockerfile_has_runtime_and_documented_entrypoint(self):
        dockerfile = SPARK_DOCKERFILE.read_text(encoding="utf-8")

        self.assertIn("FROM python:3.12-slim-bookworm", dockerfile)
        self.assertIn("openjdk-17-jre-headless", dockerfile)
        self.assertIn("pip install --no-cache-dir -r requirements.txt", dockerfile)
        self.assertIn("projects/customer-churn", dockerfile)
        self.assertIn("train_churn_spark.py", dockerfile)

    def test_requirements_include_spark_runtime_dependencies(self):
        requirements = REQUIREMENTS_FILE.read_text(encoding="utf-8")

        self.assertIn("pyspark>=3.5,<4", requirements)
        self.assertIn("setuptools>=75,<81", requirements)


if __name__ == "__main__":
    unittest.main()
