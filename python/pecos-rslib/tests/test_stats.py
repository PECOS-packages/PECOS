"""Tests for statistical functions comparing pecos-rslib vs numpy."""

import numpy as np
import pytest

from pecos_rslib.num import mean as pecos_mean
from pecos_rslib.num import power as pecos_power
from pecos_rslib.num import sqrt as pecos_sqrt
from pecos_rslib.num import std as pecos_std


class TestMeanCorrectness:
    """Test mean() correctness against numpy."""

    def test_mean_basic(self):
        """Test basic mean calculation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        pecos_result = pecos_mean(values)
        numpy_result = np.mean(values)

        assert pecos_result == numpy_result
        assert pecos_result == 3.0

    def test_mean_tuple(self):
        """Test mean with tuple input (error model use case)."""
        values = (0.01, 0.015, 0.02)

        pecos_result = pecos_mean(values)
        numpy_result = np.mean(values)

        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - 0.015) < 1e-10

    def test_mean_single_value(self):
        """Test mean with single value."""
        values = [42.0]

        pecos_result = pecos_mean(values)
        numpy_result = np.mean(values)

        assert pecos_result == numpy_result
        assert pecos_result == 42.0

    def test_mean_two_values(self):
        """Test mean with two values."""
        values = [0.5, 0.3]

        pecos_result = pecos_mean(values)
        numpy_result = np.mean(values)

        assert pecos_result == numpy_result
        assert pecos_result == 0.4

    def test_mean_empty(self):
        """Test mean with empty sequence returns NaN."""
        values = []

        pecos_result = pecos_mean(values)

        assert np.isnan(pecos_result)

    def test_mean_negative(self):
        """Test mean with negative values."""
        values = [-1.0, -2.0, -3.0]

        pecos_result = pecos_mean(values)
        numpy_result = np.mean(values)

        assert pecos_result == numpy_result
        assert pecos_result == -2.0

    def test_mean_mixed(self):
        """Test mean with mixed positive/negative values."""
        values = [-2.0, 0.0, 2.0]

        pecos_result = pecos_mean(values)
        numpy_result = np.mean(values)

        assert pecos_result == numpy_result
        assert pecos_result == 0.0

    def test_mean_precise(self):
        """Test mean with high precision values."""
        values = [0.001, 0.002]

        pecos_result = pecos_mean(values)
        numpy_result = np.mean(values)

        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - 0.0015) < 1e-10


class TestMeanErrorModelUseCases:
    """Test mean() with patterns from actual error model usage."""

    def test_p_meas_tuple_averaging(self):
        """Test the exact pattern from error models: averaging p_meas tuple."""
        # Simulating the p_meas tuple averaging use case
        p_meas_tuple = (0.01, 0.015, 0.02)

        pecos_avg = pecos_mean(p_meas_tuple)
        numpy_avg = np.mean(p_meas_tuple)

        assert abs(pecos_avg - numpy_avg) < 1e-10
        assert abs(pecos_avg - 0.015) < 1e-10

    def test_p_meas_two_values(self):
        """Test averaging two measurement error rates."""
        p_meas = (0.001, 0.002)

        pecos_avg = pecos_mean(p_meas)
        numpy_avg = np.mean(p_meas)

        assert abs(pecos_avg - numpy_avg) < 1e-10
        assert abs(pecos_avg - 0.0015) < 1e-10

    def test_various_error_rates(self):
        """Test with various error rate combinations."""
        test_cases = [
            (0.001, 0.001),  # Same values
            (0.01, 0.02),  # Different values
            (0.0, 0.01),  # One zero
            (0.001, 0.002, 0.003),  # Three values
        ]

        for p_meas_tuple in test_cases:
            pecos_avg = pecos_mean(p_meas_tuple)
            numpy_avg = np.mean(p_meas_tuple)

            assert (
                abs(pecos_avg - numpy_avg) < 1e-10
            ), f"Mismatch for {p_meas_tuple}: pecos={pecos_avg}, numpy={numpy_avg}"


class TestMeanAxisParameter:
    """Test mean() with axis parameter for multi-dimensional arrays."""

    def test_2d_axis_0(self):
        """Test mean along axis 0 (down columns)."""
        arr = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

        pecos_result = pecos_mean(arr, axis=0)
        numpy_result = np.mean(arr, axis=0)

        assert np.allclose(pecos_result, numpy_result)
        assert np.allclose(pecos_result, [2.5, 3.5, 4.5])

    def test_2d_axis_1(self):
        """Test mean along axis 1 (across rows)."""
        arr = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

        pecos_result = pecos_mean(arr, axis=1)
        numpy_result = np.mean(arr, axis=1)

        assert np.allclose(pecos_result, numpy_result)
        assert np.allclose(pecos_result, [2.0, 5.0])

    def test_2d_axis_none(self):
        """Test mean with axis=None (mean of all elements)."""
        arr = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

        pecos_result = pecos_mean(arr, axis=None)
        numpy_result = np.mean(arr, axis=None)

        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - 3.5) < 1e-10

    def test_jackknife_pattern(self):
        """Test the exact pattern from threshold_curve.py jackknife."""
        # Simulating jackknife/bootstrap averaging across runs
        opt_list = [
            [1.5, 2.5, 3.5],  # Run 1 fit parameters
            [1.6, 2.4, 3.6],  # Run 2 fit parameters
            [1.4, 2.6, 3.4],  # Run 3 fit parameters
        ]

        pecos_result = pecos_mean(opt_list, axis=0)
        numpy_result = np.mean(opt_list, axis=0)

        assert np.allclose(pecos_result, numpy_result)
        assert np.allclose(pecos_result, [1.5, 2.5, 3.5])

    def test_3d_axis_0(self):
        """Test mean on 3D array with axis=0."""
        arr = [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]

        pecos_result = pecos_mean(arr, axis=0)
        numpy_result = np.mean(arr, axis=0)

        assert np.allclose(pecos_result, numpy_result)

    def test_numpy_array_input(self):
        """Test that numpy arrays work as input."""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        pecos_result = pecos_mean(arr, axis=0)
        numpy_result = np.mean(arr, axis=0)

        assert np.allclose(pecos_result, numpy_result)


class TestStdCorrectness:
    """Test std() correctness against numpy."""

    def test_std_population_basic(self):
        """Test basic population standard deviation (ddof=0)."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        pecos_result = pecos_std(values, ddof=0)
        numpy_result = np.std(values, ddof=0)

        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - 1.4142135623730951) < 1e-10

    def test_std_sample_basic(self):
        """Test basic sample standard deviation (ddof=1)."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        pecos_result = pecos_std(values, ddof=1)
        numpy_result = np.std(values, ddof=1)

        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - 1.5811388300841898) < 1e-10

    def test_std_single_value(self):
        """Test std with single value (should be 0)."""
        values = [42.0]

        pecos_result = pecos_std(values, ddof=0)
        numpy_result = np.std(values, ddof=0)

        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - 0.0) < 1e-10

    def test_std_empty(self):
        """Test std with empty sequence returns NaN."""
        values = []

        pecos_result = pecos_std(values, ddof=0)

        assert np.isnan(pecos_result)

    def test_std_ddof_too_large(self):
        """Test std with ddof >= n returns NaN."""
        values = [1.0, 2.0]

        # With ddof=2, corrected n would be 0
        pecos_result = pecos_std(values, ddof=2)

        assert np.isnan(pecos_result)

    def test_std_uniform_values(self):
        """Test std with all identical values (should be 0)."""
        values = [5.0, 5.0, 5.0, 5.0]

        pecos_result = pecos_std(values, ddof=0)
        numpy_result = np.std(values, ddof=0)

        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - 0.0) < 1e-10

    def test_std_negative_values(self):
        """Test std with negative values."""
        values = [-3.0, -1.0, 1.0, 3.0]

        pecos_result = pecos_std(values, ddof=0)
        numpy_result = np.std(values, ddof=0)

        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - 2.23606797749979) < 1e-10

    def test_std_two_values(self):
        """Test std with two values."""
        values = [1.0, 3.0]

        pecos_result = pecos_std(values, ddof=0)
        numpy_result = np.std(values, ddof=0)

        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - 1.0) < 1e-10


class TestStdAnalysisUseCases:
    """Test std() with patterns from actual threshold analysis usage."""

    def test_jackknife_uncertainty(self):
        """Test the pattern from threshold_curve.py: jackknife parameter uncertainty."""
        # Simulating jackknife parameter estimates
        parameter_estimates = [1.5, 1.6, 1.4, 1.5, 1.7]

        pecos_result = pecos_std(parameter_estimates, ddof=0)
        numpy_result = np.std(parameter_estimates, ddof=0)

        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - 0.10198039027185571) < 1e-10

    def test_bootstrap_pattern(self):
        """Test bootstrap parameter estimation pattern."""
        # Simulating bootstrap parameter estimates
        bootstrap_params = [2.1, 2.3, 2.0, 2.2, 2.1, 2.4]

        pecos_result = pecos_std(bootstrap_params, ddof=0)
        numpy_result = np.std(bootstrap_params, ddof=0)

        assert abs(pecos_result - numpy_result) < 1e-10

    def test_threshold_fitting_uncertainty(self):
        """Test uncertainty estimation in threshold fitting."""
        # Simulating threshold parameter fits from multiple runs
        threshold_params = [0.01, 0.012, 0.009, 0.011, 0.010]

        pecos_result = pecos_std(threshold_params, ddof=0)
        numpy_result = np.std(threshold_params, ddof=0)

        assert abs(pecos_result - numpy_result) < 1e-10


class TestStdAxisParameter:
    """Test std() with axis parameter for multi-dimensional arrays."""

    def test_2d_axis_0(self):
        """Test std along axis 0 (down columns)."""
        arr = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

        pecos_result = pecos_std(arr, axis=0, ddof=0)
        numpy_result = np.std(arr, axis=0, ddof=0)

        assert np.allclose(pecos_result, numpy_result)
        assert np.allclose(pecos_result, [1.5, 1.5, 1.5])

    def test_2d_axis_1(self):
        """Test std along axis 1 (across rows)."""
        arr = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

        pecos_result = pecos_std(arr, axis=1, ddof=0)
        numpy_result = np.std(arr, axis=1, ddof=0)

        assert np.allclose(pecos_result, numpy_result)

    def test_2d_axis_none(self):
        """Test std with axis=None (std of all elements)."""
        arr = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

        pecos_result = pecos_std(arr, axis=None, ddof=0)
        numpy_result = np.std(arr, axis=None, ddof=0)

        assert abs(pecos_result - numpy_result) < 1e-10

    def test_jackknife_multiparameter_pattern(self):
        """Test the exact pattern from threshold_curve.py: multi-parameter jackknife."""
        # Simulating jackknife/bootstrap with multiple parameters
        opt_list = [
            [1.5, 2.5, 3.5],  # Run 1 fit parameters
            [1.6, 2.4, 3.6],  # Run 2 fit parameters
            [1.4, 2.6, 3.4],  # Run 3 fit parameters
        ]

        pecos_result = pecos_std(opt_list, axis=0, ddof=0)
        numpy_result = np.std(opt_list, axis=0, ddof=0)

        assert np.allclose(pecos_result, numpy_result)

    def test_3d_axis_0(self):
        """Test std on 3D array with axis=0."""
        arr = [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]

        pecos_result = pecos_std(arr, axis=0, ddof=0)
        numpy_result = np.std(arr, axis=0, ddof=0)

        assert np.allclose(pecos_result, numpy_result)

    def test_numpy_array_input(self):
        """Test that numpy arrays work as input."""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        pecos_result = pecos_std(arr, axis=0, ddof=0)
        numpy_result = np.std(arr, axis=0, ddof=0)

        assert np.allclose(pecos_result, numpy_result)

    def test_ddof_with_axis(self):
        """Test that ddof parameter works correctly with axis parameter."""
        arr = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

        # Test with ddof=1
        pecos_result = pecos_std(arr, axis=0, ddof=1)
        numpy_result = np.std(arr, axis=0, ddof=1)

        assert np.allclose(pecos_result, numpy_result)


class TestPowerCorrectness:
    """Test power() correctness against numpy."""

    def test_power_scalar_basic(self):
        """Test basic scalar power operations."""
        assert pecos_power(2.0, 3.0) == 8.0
        assert pecos_power(3.0, 2.0) == 9.0
        assert pecos_power(10.0, 0.0) == 1.0

    def test_power_fractional_exponent(self):
        """Test fractional powers (roots)."""
        pecos_result = pecos_power(4.0, 0.5)
        numpy_result = np.power(4.0, 0.5)
        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - 2.0) < 1e-10

    def test_power_negative_exponent(self):
        """Test negative exponents."""
        pecos_result = pecos_power(2.0, -1.0)
        numpy_result = np.power(2.0, -1.0)
        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - 0.5) < 1e-10

    def test_power_array_base_scalar_exp(self):
        """Test array base with scalar exponent."""
        base = [1.0, 2.0, 3.0]
        exponent = 2.0

        pecos_result = pecos_power(base, exponent)
        numpy_result = np.power(base, exponent)

        assert np.allclose(pecos_result, numpy_result)
        assert np.allclose(pecos_result, [1.0, 4.0, 9.0])

    def test_power_scalar_base_array_exp(self):
        """Test scalar base with array exponent."""
        base = 2.0
        exponent = [1.0, 2.0, 3.0]

        pecos_result = pecos_power(base, exponent)
        numpy_result = np.power(base, exponent)

        assert np.allclose(pecos_result, numpy_result)
        assert np.allclose(pecos_result, [2.0, 4.0, 8.0])

    def test_power_broadcasting(self):
        """Test broadcasting with arrays."""
        base = [[1.0, 2.0], [3.0, 4.0]]
        exponent = 2.0

        pecos_result = pecos_power(base, exponent)
        numpy_result = np.power(base, exponent)

        assert np.allclose(pecos_result, numpy_result)
        assert np.allclose(pecos_result, [[1.0, 4.0], [9.0, 16.0]])


class TestPowerThresholdUseCases:
    """Test power() with patterns from threshold_curve.py."""

    def test_power_dist_scaling(self):
        """Test the pattern: np.power(dist, 1.0 / v0)."""
        dist = 5.0
        v0 = 2.0

        pecos_result = pecos_power(dist, 1.0 / v0)
        numpy_result = np.power(dist, 1.0 / v0)

        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - np.sqrt(5.0)) < 1e-10

    def test_power_squared(self):
        """Test the pattern: np.power(x, 2)."""
        x = 3.5

        pecos_result = pecos_power(x, 2.0)
        numpy_result = np.power(x, 2.0)

        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - 12.25) < 1e-10

    def test_power_negative_fractional(self):
        """Test the pattern: np.power(dist, -1.0 / u)."""
        dist = 5.0
        u = 2.0

        pecos_result = pecos_power(dist, -1.0 / u)
        numpy_result = np.power(dist, -1.0 / u)

        assert abs(pecos_result - numpy_result) < 1e-10

    def test_power_array_scaling(self):
        """Test power with array of distances."""
        distances = np.array([3.0, 5.0, 7.0])
        v0 = 2.0

        pecos_result = pecos_power(distances, 1.0 / v0)
        numpy_result = np.power(distances, 1.0 / v0)

        assert np.allclose(pecos_result, numpy_result)


class TestSqrtCorrectness:
    """Test sqrt() correctness against numpy."""

    def test_sqrt_perfect_squares(self):
        """Test perfect square roots."""
        assert pecos_sqrt(4.0) == 2.0
        assert pecos_sqrt(9.0) == 3.0
        assert pecos_sqrt(16.0) == 4.0
        assert pecos_sqrt(25.0) == 5.0
        assert pecos_sqrt(100.0) == 10.0

    def test_sqrt_irrational(self):
        """Test irrational square roots."""
        pecos_result = pecos_sqrt(2.0)
        numpy_result = np.sqrt(2.0)
        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - np.sqrt(2.0)) < 1e-10

    def test_sqrt_special_cases(self):
        """Test special cases."""
        assert pecos_sqrt(0.0) == 0.0
        assert pecos_sqrt(1.0) == 1.0
        assert np.isnan(pecos_sqrt(-1.0))

    def test_sqrt_array(self):
        """Test array input."""
        values = [4.0, 9.0, 16.0, 25.0]
        pecos_result = pecos_sqrt(values)
        numpy_result = np.sqrt(values)
        assert np.allclose(pecos_result, numpy_result)
        assert np.allclose(pecos_result, [2.0, 3.0, 4.0, 5.0])

    def test_sqrt_2d_array(self):
        """Test 2D array input."""
        values = [[4.0, 9.0], [16.0, 25.0]]
        pecos_result = pecos_sqrt(values)
        numpy_result = np.sqrt(values)
        assert np.allclose(pecos_result, numpy_result)


class TestSqrtVarianceUseCases:
    """Test sqrt() with variance-to-std-deviation patterns."""

    def test_sqrt_variance_to_std(self):
        """Test the pattern: np.sqrt(variance)."""
        variance = 4.0
        pecos_result = pecos_sqrt(variance)
        numpy_result = np.sqrt(variance)
        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - 2.0) < 1e-10

    def test_sqrt_variance_array(self):
        """Test variance to std deviation with arrays."""
        variances = np.array([1.0, 4.0, 9.0, 16.0])
        pecos_result = pecos_sqrt(variances)
        numpy_result = np.sqrt(variances)
        assert np.allclose(pecos_result, numpy_result)
        assert np.allclose(pecos_result, [1.0, 2.0, 3.0, 4.0])

    def test_sqrt_diag_covariance(self):
        """Test extracting std from covariance matrix diagonal."""
        # Simulate covariance matrix diagonal (variances)
        covariance_diag = np.array([0.25, 1.0, 2.25, 4.0])
        pecos_result = pecos_sqrt(covariance_diag)
        numpy_result = np.sqrt(covariance_diag)
        assert np.allclose(pecos_result, numpy_result)
        assert np.allclose(pecos_result, [0.5, 1.0, 1.5, 2.0])

    def test_sqrt_small_variances(self):
        """Test with small variance values."""
        variances = [0.01, 0.04, 0.0001]
        pecos_result = pecos_sqrt(variances)
        numpy_result = np.sqrt(variances)
        assert np.allclose(pecos_result, numpy_result)
        assert np.allclose(pecos_result, [0.1, 0.2, 0.01])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
