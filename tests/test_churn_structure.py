import importlib.util
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
CHURN_SCRIPT = REPO_ROOT / "projects" / "customer-churn" / "src" / "Logistic_Regression_Customer_Churn.py"


spec = importlib.util.spec_from_file_location("churn_pipeline", CHURN_SCRIPT)
churn_pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(churn_pipeline)


class ChurnStructureTests(unittest.TestCase):
    def test_default_sample_paths_exist_after_project_move(self):
        self.assertTrue((REPO_ROOT / churn_pipeline.DEFAULT_CUSTOMERS_PATH).exists())
        self.assertTrue((REPO_ROOT / churn_pipeline.DEFAULT_SUBSCRIPTIONS_PATH).exists())
        self.assertEqual(churn_pipeline.DEFAULT_OUTPUT_DIR, "outputs/churn")

    def test_sample_files_load_with_required_join_key(self):
        customers, subscriptions = churn_pipeline.load_data(
            REPO_ROOT / churn_pipeline.DEFAULT_CUSTOMERS_PATH,
            REPO_ROOT / churn_pipeline.DEFAULT_SUBSCRIPTIONS_PATH,
        )

        self.assertIn("customer_id", customers.columns)
        self.assertIn("customer_id", subscriptions.columns)
        self.assertGreater(len(customers), 0)
        self.assertGreater(len(subscriptions), 0)


if __name__ == "__main__":
    unittest.main()
