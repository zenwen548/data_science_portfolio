from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from train import TUNED_XGB_PARAMS, prepare_train_validation, rmsle


class BulldozerTrainingTests(unittest.TestCase):
    def test_prepare_train_validation_matches_notebook_preprocessing(self):
        data = pd.DataFrame(
            {
                "SalesID": [1, 2, 3, 4],
                "SalePrice": [10000, 12000, 20000, 22000],
                "saledate": ["2011-01-03", "2011-06-15", "2012-02-10", "2012-03-12"],
                "MachineHoursCurrentMeter": [10.0, np.nan, 20.0, np.nan],
                "ProductSize": ["Small", None, "Large", "Small"],
            }
        )

        X_train, y_train, X_valid, y_valid = prepare_train_validation(data)

        self.assertEqual(list(y_train), [10000, 12000])
        self.assertEqual(list(y_valid), [20000, 22000])
        self.assertNotIn("saledate", X_train.columns)
        self.assertTrue(
            {"saleYear", "saleMonth", "saleDay", "saleDayOfWeek", "saleDayOfYear"}.issubset(
                X_train.columns
            )
        )
        self.assertIn("MachineHoursCurrentMeter_is_missing", X_train.columns)
        self.assertIn("ProductSize_is_missing", X_train.columns)
        self.assertEqual(list(X_valid.columns), list(X_train.columns))
        self.assertIn(X_train["ProductSize"].dtype.kind, {"i", "u"})

    def test_rmsle_clips_negative_predictions_before_scoring(self):
        score = rmsle([1, 10], [-5, 10])

        self.assertGreaterEqual(score, 0)

    def test_tuned_xgb_params_match_notebook_best_search_result(self):
        self.assertEqual(
            TUNED_XGB_PARAMS,
            {
                "subsample": 0.7,
                "n_estimators": 200,
                "max_depth": 10,
                "learning_rate": 0.05,
                "colsample_bytree": 0.5,
                "n_jobs": -1,
                "random_state": 42,
            },
        )


if __name__ == "__main__":
    unittest.main()
