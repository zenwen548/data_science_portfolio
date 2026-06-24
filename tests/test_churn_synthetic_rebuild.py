import importlib.util
from pathlib import Path
import tempfile
import unittest

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
CHURN_SRC = REPO_ROOT / "projects" / "customer-churn" / "src"
PIPELINE_SCRIPT = CHURN_SRC / "Logistic_Regression_Customer_Churn.py"
GENERATOR_SCRIPT = CHURN_SRC / "generate_churn_data.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


churn_pipeline = load_module("churn_pipeline", PIPELINE_SCRIPT)
churn_generator = load_module("churn_generator", GENERATOR_SCRIPT)


class ChurnSyntheticRebuildTests(unittest.TestCase):
    def test_generator_is_seeded_and_preserves_dashboard_schema(self):
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

        customers_a, subscriptions_a = churn_generator.generate_churn_data(
            row_count=500, seed=2026
        )
        customers_b, subscriptions_b = churn_generator.generate_churn_data(
            row_count=500, seed=2026
        )

        pd.testing.assert_frame_equal(customers_a, customers_b)
        pd.testing.assert_frame_equal(subscriptions_a, subscriptions_b)
        self.assertTrue(required_customer_columns.issubset(customers_a.columns))
        self.assertTrue(required_subscription_columns.issubset(subscriptions_a.columns))
        self.assertEqual(len(customers_a), 500)
        self.assertEqual(len(subscriptions_a), 500)

    def test_leakage_report_flags_single_feature_separation(self):
        df = pd.DataFrame(
            {
                "safe_noise": [0.1, 0.2, 0.8, 0.7],
                "perfect_leak": [0.0, 0.1, 0.9, 1.0],
                "churned": [0, 0, 1, 1],
            }
        )

        report = churn_pipeline.single_feature_auc_report(
            df, ["safe_noise", "perfect_leak"], target_column="churned"
        )

        leak_row = report.loc[report["feature"] == "perfect_leak"].iloc[0]
        self.assertGreater(leak_row["single_feature_auc"], 0.95)

    def test_pipeline_exports_oof_dashboard_contract_and_qa_packet(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            customers_path = tmp_path / "customers.csv"
            subscriptions_path = tmp_path / "subscriptions.csv"
            output_dir = tmp_path / "outputs"

            customers, subscriptions = churn_generator.generate_churn_data(
                row_count=700, seed=42
            )
            customers.to_csv(customers_path, index=False)
            subscriptions.to_csv(subscriptions_path, index=False)

            customer_data, subscription_data = churn_pipeline.load_data(
                customers_path, subscriptions_path
            )
            merged = churn_pipeline.merge_and_prepare(customer_data, subscription_data)
            (
                best_model_name,
                best_model,
                features,
                metrics_df,
            ) = churn_pipeline.train_and_evaluate(
                merged, output_dir, test_size=0.30
            )
            churn_pipeline.export_for_tableau(
                merged, best_model, features, best_model_name, output_dir
            )

            tableau = pd.read_csv(output_dir / "final_data_for_tableau.csv")
            leakage = pd.read_csv(output_dir / "single_feature_auc.csv")
            probability_summary = pd.read_csv(output_dir / "probability_summary.csv")

            required_columns = {
                "predicted_churn",
                "predicted_churn_probability",
                "churn_risk_tier",
                "revenue_at_risk",
                "impacted_customer_flag",
            }
            extreme_probability_rate = (
                (tableau["predicted_churn_probability"] == 0)
                | (tableau["predicted_churn_probability"] == 1)
            ).mean()

            self.assertTrue(required_columns.issubset(tableau.columns))
            self.assertIn("oof_roc_auc", metrics_df.columns)
            self.assertGreaterEqual(metrics_df["oof_roc_auc"].max(), 0.75)
            self.assertLessEqual(metrics_df["oof_roc_auc"].max(), 0.88)
            self.assertTrue(tableau["predicted_churn_probability"].between(0, 1).all())
            self.assertLessEqual(extreme_probability_rate, 0.01)
            self.assertLessEqual(leakage["single_feature_auc"].max(), 0.95)
            self.assertTrue((output_dir / "qa_packet.md").exists())
            self.assertGreater(len(probability_summary), 0)


if __name__ == "__main__":
    unittest.main()
