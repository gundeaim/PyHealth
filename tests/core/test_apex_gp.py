import math
import unittest

import numpy as np
import torch
from scipy import stats

from pyhealth.models.apex_gp import (
    arctan_mean,
    epsilon_at_bound,
    fit_learning_curve_least_squares,
    interval_coverage,
    learning_curve_mean,
    power_law_mean,
    quantized_likelihood,
    rmse,
    truncated_normal_hdi,
)

PAPER_SIZES = np.array([60.0, 94.0, 147.0, 230.0, 360.0])


def _truncnorm_01(loc: float, scale: float):
    """Normal(loc, scale) truncated to [0, 1], built directly with SciPy."""
    return stats.truncnorm((0.0 - loc) / scale, (1.0 - loc) / scale, loc, scale)


class TestRMSE(unittest.TestCase):
    """Root-mean-squared error."""

    def test_known_example(self):
        """Errors (0, -0.1, 0.3) give sqrt((0 + 0.01 + 0.09) / 3)."""
        observed = np.array([0.7, 0.8, 0.9])
        predicted = np.array([0.7, 0.9, 0.6])
        self.assertAlmostEqual(rmse(observed, predicted), math.sqrt(0.1 / 3), 12)

    def test_perfect_prediction_is_zero(self):
        self.assertEqual(rmse(np.array([0.5, 0.6]), np.array([0.5, 0.6])), 0.0)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            rmse(np.array([0.5, 0.6]), np.array([0.5]))


class TestTruncatedNormalHDI(unittest.TestCase):
    """Highest-density interval of the [0, 1]-truncated predictive."""

    def test_far_from_bounds_matches_normal_interval(self):
        """With loc = 0.5 and scale 0.01, truncation is negligible."""
        lower, upper = truncated_normal_hdi(0.5, 0.01, 0.95)
        half_width = stats.norm.ppf(0.975) * 0.01  # 1.96 * scale
        self.assertAlmostEqual(lower, 0.5 - half_width, 6)
        self.assertAlmostEqual(upper, 0.5 + half_width, 6)

    def test_loc_above_one_pins_upper_bound_and_keeps_mass(self):
        """The density peaks at 1, so the HDI ends exactly at 1."""
        loc, scale = 1.05, 0.04
        lower, upper = truncated_normal_hdi(loc, scale, 0.95)
        self.assertEqual(upper, 1.0)
        self.assertLess(lower, 1.0)
        dist = _truncnorm_01(loc, scale)
        self.assertAlmostEqual(dist.cdf(upper) - dist.cdf(lower), 0.95, 8)

    def test_invalid_level_raises(self):
        for level in (0.0, 1.0, 1.5, -0.1):
            with self.subTest(level=level), self.assertRaises(ValueError):
                truncated_normal_hdi(0.5, 0.01, level)


class TestQuantizedLikelihood(unittest.TestCase):
    """Predictive mass on (y* - delta, y* + delta)."""

    def test_matches_direct_scipy_computation(self):
        observed = np.array([0.90, 0.62])
        loc = np.array([0.92, 0.60])
        scale = np.array([0.03, 0.05])
        expected = [
            _truncnorm_01(m, s).cdf(y + 0.01) - _truncnorm_01(m, s).cdf(y - 0.01)
            for y, m, s in zip(observed, loc, scale, strict=True)
        ]
        result = quantized_likelihood(observed, loc, scale, delta=0.01)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)

    def test_untruncated_case_is_normal_mass_within_one_sd(self):
        """delta equal to the scale, centered far from bounds: ~68.27%."""
        result = quantized_likelihood([0.5], [0.5], [0.01], delta=0.01)[0]
        expected = stats.norm.cdf(1.0) - stats.norm.cdf(-1.0)
        self.assertAlmostEqual(result, expected, 9)


class TestIntervalCoverage(unittest.TestCase):
    """Fraction of observations inside their interval."""

    def test_endpoints_count_as_covered(self):
        observed = np.array([0.5, 0.6, 0.7])
        lower = np.array([0.5, 0.5, 0.5])
        upper = np.array([0.6, 0.6, 0.6])
        self.assertAlmostEqual(interval_coverage(observed, lower, upper), 2 / 3, 12)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            interval_coverage(np.array([0.5, 0.6]), np.array([0.4]), np.array([0.7]))


class TestLeastSquaresBaselines(unittest.TestCase):
    """Least-squares learning-curve fits."""

    def test_recovers_noise_free_power_law(self):
        epsilon, theta1, theta2 = 0.05, 0.9, -0.45
        aurocs = learning_curve_mean(PAPER_SIZES, "power_law", epsilon, theta1, theta2)
        fit = fit_learning_curve_least_squares(PAPER_SIZES, aurocs, "power_law")
        self.assertAlmostEqual(fit.epsilon, epsilon, 6)
        self.assertAlmostEqual(fit.theta1, theta1, 5)
        self.assertAlmostEqual(fit.theta2, theta2, 6)
        self.assertLess(fit.mse, 1e-20)
        self.assertAlmostEqual(fit.ceiling, 1.0 - epsilon, 6)
        expected = learning_curve_mean([20000.0], "power_law", epsilon, theta1, theta2)
        self.assertAlmostEqual(fit.predict([20000.0])[0], expected[0], 8)

    def test_recovers_noise_free_arctan(self):
        epsilon, theta1, theta2 = 0.05, 0.02, 1.0
        aurocs = learning_curve_mean(PAPER_SIZES, "arctan", epsilon, theta1, theta2)
        fit = fit_learning_curve_least_squares(PAPER_SIZES, aurocs, "arctan")
        self.assertAlmostEqual(fit.epsilon, epsilon, 5)
        self.assertAlmostEqual(fit.theta1, theta1, 6)
        self.assertAlmostEqual(fit.theta2, theta2, 4)

    def test_epsilon_respects_upper_bound(self):
        """A flat curve forces epsilon to 1 - max(aurocs): ceiling = best pilot."""
        aurocs = np.array([0.60, 0.61, 0.60, 0.61, 0.60])
        fit = fit_learning_curve_least_squares(PAPER_SIZES, aurocs, "power_law")
        self.assertLessEqual(fit.epsilon, 1.0 - 0.61 + 1e-12)
        self.assertGreaterEqual(fit.epsilon, 0.0)

    def test_numpy_means_match_torch_means(self):
        """The LS curves and the GP mean functions use identical formulas."""
        x = np.array([60.0, 360.0, 5000.0])
        x_t = torch.as_tensor(x, dtype=torch.float64)
        for name, torch_fn, params in (
            ("power_law", power_law_mean, (0.1, 0.8, -0.3)),
            ("arctan", arctan_mean, (0.1, 0.01, 0.5)),
        ):
            with self.subTest(mean_function=name):
                np.testing.assert_allclose(
                    learning_curve_mean(x, name, *params),
                    np.asarray(torch_fn(x_t, *params)),
                    rtol=0,
                    atol=1e-12,
                )


class TestEpsilonAtBound(unittest.TestCase):
    """Ceiling diagnostic; support is [0, 1 - 0.62] = [0, 0.38] here."""

    def test_upper(self):
        self.assertEqual(epsilon_at_bound(0.38, 0.62), "upper")
        self.assertEqual(epsilon_at_bound(0.3765, 0.62), "upper")  # within 1%

    def test_lower(self):
        self.assertEqual(epsilon_at_bound(0.0, 0.62), "lower")
        self.assertEqual(epsilon_at_bound(0.0035, 0.62), "lower")  # within 1%

    def test_none(self):
        self.assertEqual(epsilon_at_bound(0.19, 0.62), "none")
        self.assertEqual(epsilon_at_bound(0.37, 0.62), "none")  # 2.6% from bound

    def test_respects_epsilon_min(self):
        self.assertEqual(epsilon_at_bound(0.1, 0.62, epsilon_min=0.1), "lower")


if __name__ == "__main__":
    unittest.main()
