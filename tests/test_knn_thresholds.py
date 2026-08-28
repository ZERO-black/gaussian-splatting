import unittest

import numpy as np

from analysis.knn import (
    apply_lower_threshold_plateau,
    distribution_quantile_label,
    distribution_thresholds,
    validate_distribution_quantiles,
)


class KNNThresholdTest(unittest.TestCase):
    def test_resolves_probabilities_against_distribution(self):
        values = np.arange(11, dtype=np.float32)

        thresholds = distribution_thresholds(values, [0.5, 0.9])

        self.assertEqual(thresholds["q0p5"], 5.0)
        self.assertEqual(thresholds["q0p9"], 9.0)

    def test_ignores_non_finite_values(self):
        values = np.array([0.0, 2.0, np.nan, np.inf])

        thresholds = distribution_thresholds(values, [0.5])

        self.assertEqual(thresholds["q0p5"], 1.0)

    def test_lower_values_share_threshold(self):
        values = np.array([0.0, 1.0, 2.0, np.nan])

        thresholded = apply_lower_threshold_plateau(values, 1.5)

        np.testing.assert_equal(thresholded[:3], [1.5, 1.5, 2.0])
        self.assertTrue(np.isnan(thresholded[3]))

    def test_quantiles_are_sorted_and_deduplicated(self):
        self.assertEqual(
            validate_distribution_quantiles([0.9, 0.5, 0.9]),
            [0.5, 0.9],
        )

    def test_rejects_percent_style_input(self):
        with self.assertRaises(ValueError):
            validate_distribution_quantiles([90.0])

    def test_label_is_filesystem_friendly(self):
        self.assertEqual(distribution_quantile_label(0.95), "q0p95")


if __name__ == "__main__":
    unittest.main()
