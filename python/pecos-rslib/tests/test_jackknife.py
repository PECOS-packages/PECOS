"""Tests for jackknife resampling functions.

This test suite validates the jackknife implementation:
- weighted_mean(): Weighted mean calculation
- jackknife_resamples(): Leave-one-out resample generation
- jackknife_stats(): Statistics from jackknife estimates
- jackknife_weighted(): Full weighted jackknife with bias correction

Note: These functions are accessible via the pc.stats namespace:
- pc.stats.weighted_mean()
- pc.stats.jackknife_resamples()
- pc.stats.jackknife_stats()
- pc.stats.jackknife_weighted()
"""

import pecos as pc


class TestWeightedMean:
    """Test weighted_mean function."""

    def test_weighted_mean_basic(self):
        """Basic weighted mean calculation."""
        data = [(0.98, 100.0), (0.94, 500.0), (0.96, 200.0)]
        result = pc.stats.weighted_mean(data)

        # Manual calculation: (0.98*100 + 0.94*500 + 0.96*200) / (100 + 500 + 200)
        # = (98 + 470 + 192) / 800 = 760 / 800 = 0.95
        expected = 0.95
        assert abs(result - expected) < 1e-10

    def test_weighted_mean_uniform_weights(self):
        """With uniform weights, should match unweighted mean."""
        data = [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (4.0, 1.0), (5.0, 1.0)]
        result = pc.stats.weighted_mean(data)
        assert abs(result - 3.0) < 1e-10

    def test_weighted_mean_single_value(self):
        """Single value should return that value."""
        data = [(0.95, 1000.0)]
        result = pc.stats.weighted_mean(data)
        assert abs(result - 0.95) < 1e-10

    def test_weighted_mean_empty(self):
        """Empty data should return NaN."""
        data = []
        result = pc.stats.weighted_mean(data)
        assert pc.isnan(result)

    def test_weighted_mean_zero_total_weight(self):
        """Zero total weight should return NaN."""
        data = [(0.5, 0.0), (0.7, 0.0)]
        result = pc.stats.weighted_mean(data)
        assert pc.isnan(result)

    def test_weighted_mean_heavy_weight(self):
        """One measurement with much higher weight."""
        data = [(0.5, 10.0), (0.9, 1000.0)]
        result = pc.stats.weighted_mean(data)

        # (0.5*10 + 0.9*1000) / (10 + 1000) = 905 / 1010
        expected = 905.0 / 1010.0
        assert abs(result - expected) < 1e-10


class TestJackknifeResamples:
    """Test jackknife_resamples function."""

    def test_jackknife_resamples_basic(self):
        """Basic jackknife resample generation."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        resamples = pc.stats.jackknife_resamples(data)

        # Should return 5x4 array (n × n-1)
        assert resamples.shape == (5, 4)

        # Check each resample
        assert pc.array_equal(resamples[0], pc.array([2.0, 3.0, 4.0, 5.0]))  # removed 1.0
        assert pc.array_equal(resamples[1], pc.array([1.0, 3.0, 4.0, 5.0]))  # removed 2.0
        assert pc.array_equal(resamples[2], pc.array([1.0, 2.0, 4.0, 5.0]))  # removed 3.0
        assert pc.array_equal(resamples[3], pc.array([1.0, 2.0, 3.0, 5.0]))  # removed 4.0
        assert pc.array_equal(resamples[4], pc.array([1.0, 2.0, 3.0, 4.0]))  # removed 5.0

    def test_jackknife_resamples_two_elements(self):
        """Edge case with two elements."""
        data = [10.0, 20.0]
        resamples = pc.stats.jackknife_resamples(data)

        assert resamples.shape == (2, 1)
        assert pc.array_equal(resamples[0], pc.array([20.0]))
        assert pc.array_equal(resamples[1], pc.array([10.0]))

    def test_jackknife_resamples_single_element(self):
        """Edge case with single element."""
        data = [42.0]
        resamples = pc.stats.jackknife_resamples(data)

        assert resamples.shape == (1, 0)

    def test_jackknife_resamples_negative_values(self):
        """Jackknife should work with negative values."""
        data = [-3.0, -1.0, 1.0, 3.0]
        resamples = pc.stats.jackknife_resamples(data)

        assert resamples.shape == (4, 3)
        assert pc.array_equal(resamples[0], pc.array([-1.0, 1.0, 3.0]))
        assert pc.array_equal(resamples[1], pc.array([-3.0, 1.0, 3.0]))
        assert pc.array_equal(resamples[2], pc.array([-3.0, -1.0, 3.0]))
        assert pc.array_equal(resamples[3], pc.array([-3.0, -1.0, 1.0]))


class TestJackknifeStats:
    """Test jackknife_stats function."""

    def test_jackknife_stats_basic(self):
        """Basic jackknife statistics calculation."""
        estimates = [1.5, 1.6, 1.4, 1.5, 1.7]
        jack_mean, jack_se = pc.stats.jackknife_stats(estimates)

        # Mean should be 1.54
        expected_mean = 1.54
        assert abs(jack_mean - expected_mean) < 1e-10

        # Check standard error is reasonable
        assert jack_se > 0.0
        assert jack_se < 1.0

    def test_jackknife_stats_uniform_estimates(self):
        """All estimates the same → SE should be 0."""
        estimates = [2.5, 2.5, 2.5, 2.5]
        jack_mean, jack_se = pc.stats.jackknife_stats(estimates)

        assert abs(jack_mean - 2.5) < 1e-10
        assert abs(jack_se - 0.0) < 1e-10

    def test_jackknife_stats_two_estimates(self):
        """Edge case with two estimates."""
        estimates = [1.0, 3.0]
        jack_mean, jack_se = pc.stats.jackknife_stats(estimates)

        # Mean = 2.0
        assert abs(jack_mean - 2.0) < 1e-10

        # SE should be positive
        assert jack_se > 0.0


class TestJackknifeWeighted:
    """Test jackknife_weighted function."""

    def test_jackknife_weighted_single_measurement(self):
        """Single measurement should use binomial error."""
        data = [(0.95, 1000.0)]
        estimate, error = pc.stats.jackknife_weighted(data)

        # Estimate should be the value itself
        assert abs(estimate - 0.95) < 1e-10

        # Error = sqrt(p * (1-p) / n) where p = 1 - 0.95 = 0.05
        # error = sqrt(0.05 * 0.95 / 1000)
        expected_error = pc.sqrt(0.05 * 0.95 / 1000.0)
        assert abs(error - expected_error) < 1e-10

    def test_jackknife_weighted_multiple_measurements(self):
        """Multiple measurements with different weights."""
        data = [(0.98, 100.0), (0.94, 500.0), (0.96, 200.0)]
        corrected, std_err = pc.stats.jackknife_weighted(data)

        # The corrected estimate should be close to the weighted mean
        wt_avg = pc.stats.weighted_mean(data)
        assert abs(corrected - wt_avg) < 0.1  # Loose check for bias correction

        # Standard error should be positive and reasonable
        assert std_err > 0.0
        assert std_err < 1.0

    def test_jackknife_weighted_uniform_weights(self):
        """With uniform weights, behavior should match unweighted jackknife."""
        data = [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (4.0, 1.0), (5.0, 1.0)]
        corrected, std_err = pc.stats.jackknife_weighted(data)

        # Mean is 3.0, jackknife should be close
        assert abs(corrected - 3.0) < 0.1

        # SE should be reasonable
        assert std_err > 0.0

    def test_jackknife_weighted_two_measurements(self):
        """Edge case with two measurements."""
        data = [(0.9, 100.0), (0.8, 200.0)]
        corrected, std_err = pc.stats.jackknife_weighted(data)

        # Weighted mean = (0.9*100 + 0.8*200) / 300 = 250/300
        wt_avg = pc.stats.weighted_mean(data)
        expected_wt_avg = 250.0 / 300.0
        assert abs(wt_avg - expected_wt_avg) < 1e-10

        # Corrected should be close to weighted mean
        assert abs(corrected - wt_avg) < 0.1

        # SE should be positive
        assert std_err > 0.0


class TestJackknifeIntegration:
    """Integration tests combining multiple jackknife functions."""

    def test_jackknife_resamples_and_stats_integration(self):
        """Full jackknife workflow: resample → estimate → stats."""
        data = [1.5, 1.6, 1.4, 1.5, 1.7]

        # Generate jackknife resamples
        resamples = pc.stats.jackknife_resamples(data)

        # Compute mean for each resample
        estimates = [pc.mean(resamples[i]) for i in range(len(resamples))]

        # Compute jackknife statistics
        jack_mean, jack_se = pc.stats.jackknife_stats(estimates)

        # The jackknife mean should be close to the original mean
        original_mean = pc.mean(data)
        assert abs(jack_mean - original_mean) < 1e-10

        # SE should be positive and reasonable
        assert jack_se > 0.0
        assert jack_se < 1.0

    def test_jackknife_weighted_vs_manual_calculation(self):
        """Verify jackknife_weighted matches manual calculation."""
        data = [(0.98, 100.0), (0.94, 500.0), (0.96, 200.0)]
        corrected, std_err = pc.stats.jackknife_weighted(data)

        # Manual calculation
        wt_mean = pc.stats.weighted_mean(data)

        # Leave-one-out estimates
        est_0 = pc.stats.weighted_mean([(0.94, 500.0), (0.96, 200.0)])  # removed first
        est_1 = pc.stats.weighted_mean([(0.98, 100.0), (0.96, 200.0)])  # removed second
        est_2 = pc.stats.weighted_mean([(0.98, 100.0), (0.94, 500.0)])  # removed third

        jack_estimates = [est_0, est_1, est_2]
        mean_jack = pc.mean(jack_estimates)

        # Bias = (n-1) * (mean_jack - wt_mean)
        n = len(data)
        bias = (n - 1) * (mean_jack - wt_mean)
        expected_corrected = wt_mean - bias

        # SE = sqrt((n-1) * mean((est - mean_jack)^2))
        sum_sq_diff = sum((e - mean_jack) ** 2 for e in jack_estimates)
        expected_se = pc.sqrt((n - 1) * sum_sq_diff / n)

        assert abs(corrected - expected_corrected) < 1e-10
        assert abs(std_err - expected_se) < 1e-10


class TestJackknifeQuantumComputing:
    """Test jackknife with quantum computing use cases."""

    def test_fidelity_estimation(self):
        """Typical quantum fidelity estimation scenario."""
        # Simulated fidelity measurements from repeated experiments
        data = [
            (0.982, 100),  # Run 1: 98.2% fidelity, 100 shots
            (0.975, 200),  # Run 2: 97.5% fidelity, 200 shots
            (0.988, 150),  # Run 3: 98.8% fidelity, 150 shots
            (0.979, 300),  # Run 4: 97.9% fidelity, 300 shots
        ]

        corrected, std_err = pc.stats.jackknife_weighted(data)

        # Fidelity should be between 0 and 1
        assert 0.0 <= corrected <= 1.0

        # Should be close to weighted average
        wt_avg = pc.stats.weighted_mean(data)
        assert abs(corrected - wt_avg) < 0.01

        # Error should be small (high confidence with many shots)
        assert std_err < 0.05

    def test_low_shot_count_scenario(self):
        """Scenario with very few shots (higher uncertainty)."""
        data = [(0.95, 10)]  # Single run with only 10 shots
        estimate, error = pc.stats.jackknife_weighted(data)

        # Uses binomial error formula
        assert abs(estimate - 0.95) < 1e-10

        # Error should be relatively large (low shot count)
        expected_error = pc.sqrt(0.05 * 0.95 / 10.0)
        assert abs(error - expected_error) < 1e-10
        assert error > 0.05  # Should be noticeable uncertainty
