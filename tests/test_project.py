import json
import unittest
from pathlib import Path

import pandas as pd

from src.analyze_data import benjamini_hochberg, rank_biserial


ROOT = Path(__file__).resolve().parents[1]


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(ROOT / "data/raw/heart_disease_uci.csv")
        cls.clean = pd.read_csv(ROOT / "data/processed/heart_disease_analysis_ready.csv")

    def test_expected_shape_and_sources(self):
        self.assertEqual(self.raw.shape, (920, 16))
        self.assertEqual(
            set(self.raw["dataset"].unique()),
            {"Cleveland", "Hungary", "Switzerland", "VA Long Beach"},
        )

    def test_cleaning_preserves_rows_and_target(self):
        self.assertEqual(len(self.clean), len(self.raw))
        expected = self.raw["num"].gt(0).map({True: "Yes", False: "No"})
        pd.testing.assert_series_equal(
            self.clean["disease_present"], expected, check_names=False
        )

    def test_implausible_zeros_are_flagged_and_reclassified(self):
        self.assertFalse(self.clean["chol"].eq(0).any())
        self.assertFalse(self.clean["trestbps"].eq(0).any())
        self.assertEqual(
            int(self.clean["chol_zero_reclassified"].sum()),
            int(self.raw["chol"].eq(0).sum()),
        )
        self.assertEqual(
            int(self.clean["trestbps_zero_reclassified"].sum()),
            int(self.raw["trestbps"].eq(0).sum()),
        )

    def test_generated_model_is_not_marked_for_clinical_deployment(self):
        summary = json.loads(
            (ROOT / "models/modeling_summary.json").read_text(encoding="utf-8")
        )
        self.assertFalse(summary["clinical_deployment_approved"])
        self.assertNotIn("dataset", summary["features"])

    def test_statistical_helpers(self):
        self.assertEqual(rank_biserial(4, 2, 2), 1.0)
        adjusted = benjamini_hochberg([0.01, 0.04, 0.03])
        self.assertEqual(len(adjusted), 3)
        self.assertTrue(all(0 <= value <= 1 for value in adjusted))


if __name__ == "__main__":
    unittest.main()
