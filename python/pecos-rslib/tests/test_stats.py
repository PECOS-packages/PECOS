"""Tests for statistical functions comparing pecos-rslib vs numpy."""

import numpy as np
import pytest

import pecos as pc


class TestMeanCorrectness:
    """Test mean() correctness against numpy."""

    def test_mean_basic(self):
        """Test basic mean calculation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        pecos_result = pc.mean(values)
        numpy_result = np.mean(values)

        assert pecos_result == numpy_result
        assert pecos_result == 3.0

    def test_mean_tuple(self):
        """Test mean with tuple input (error model use case)."""
        values = (0.01, 0.015, 0.02)

        pecos_result = pc.mean(values)
        numpy_result = np.mean(values)

        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - 0.015) < 1e-10

    def test_mean_single_value(self):
        """Test mean with single value."""
        values = [42.0]

        pecos_result = pc.mean(values)
        numpy_result = np.mean(values)

        assert pecos_result == numpy_result
        assert pecos_result == 42.0

    def test_mean_two_values(self):
        """Test mean with two values."""
        values = [0.5, 0.3]

        pecos_result = pc.mean(values)
        numpy_result = np.mean(values)

        assert pecos_result == numpy_result
        assert pecos_result == 0.4

    def test_mean_empty(self):
        """Test mean with empty sequence returns NaN."""
        values = []

        pecos_result = pc.mean(values)

        assert np.isnan(pecos_result)

    def test_mean_negative(self):
        """Test mean with negative values."""
        values = [-1.0, -2.0, -3.0]

        pecos_result = pc.mean(values)
        numpy_result = np.mean(values)

        assert pecos_result == numpy_result
        assert pecos_result == -2.0

    def test_mean_mixed(self):
        """Test mean with mixed positive/negative values."""
        values = [-2.0, 0.0, 2.0]

        pecos_result = pc.mean(values)
        numpy_result = np.mean(values)

        assert pecos_result == numpy_result
        assert pecos_result == 0.0

    def test_mean_precise(self):
        """Test mean with high precision values."""
        values = [0.001, 0.002]

        pecos_result = pc.mean(values)
        numpy_result = np.mean(values)

        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - 0.0015) < 1e-10


class TestMeanErrorModelUseCases:
    """Test mean() with patterns from actual error model usage."""

    def test_p_meas_tuple_averaging(self):
        """Test the exact pattern from error models: averaging p_meas tuple."""
        # Simulating the p_meas tuple averaging use case
        p_meas_tuple = (0.01, 0.015, 0.02)

        pecos_avg = pc.mean(p_meas_tuple)
        numpy_avg = np.mean(p_meas_tuple)

        assert abs(pecos_avg - numpy_avg) < 1e-10
        assert abs(pecos_avg - 0.015) < 1e-10

    def test_p_meas_two_values(self):
        """Test averaging two measurement error rates."""
        p_meas = (0.001, 0.002)

        pecos_avg = pc.mean(p_meas)
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
            pecos_avg = pc.mean(p_meas_tuple)
            numpy_avg = np.mean(p_meas_tuple)

            assert (
                abs(pecos_avg - numpy_avg) < 1e-10
            ), f"Mismatch for {p_meas_tuple}: pecos={pecos_avg}, numpy={numpy_avg}"


class TestMeanAxisParameter:
    """Test mean() with axis parameter for multi-dimensional arrays."""

    def test_2d_axis_0(self):
        """Test mean along axis 0 (down columns)."""
        arr = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

        pecos_result = pc.mean(arr, axis=0)
        numpy_result = np.mean(arr, axis=0)

        assert np.allclose(pecos_result, numpy_result)
        assert np.allclose(pecos_result, [2.5, 3.5, 4.5])

    def test_2d_axis_1(self):
        """Test mean along axis 1 (across rows)."""
        arr = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

        pecos_result = pc.mean(arr, axis=1)
        numpy_result = np.mean(arr, axis=1)

        assert np.allclose(pecos_result, numpy_result)
        assert np.allclose(pecos_result, [2.0, 5.0])

    def test_2d_axis_none(self):
        """Test mean with axis=None (mean of all elements)."""
        arr = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

        pecos_result = pc.mean(arr, axis=None)
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

        pecos_result = pc.mean(opt_list, axis=0)
        numpy_result = np.mean(opt_list, axis=0)

        assert np.allclose(pecos_result, numpy_result)
        assert np.allclose(pecos_result, [1.5, 2.5, 3.5])

    def test_3d_axis_0(self):
        """Test mean on 3D array with axis=0."""
        arr = [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]

        pecos_result = pc.mean(arr, axis=0)
        numpy_result = np.mean(arr, axis=0)

        assert np.allclose(pecos_result, numpy_result)

    def test_numpy_array_input(self):
        """Test that numpy arrays work as input."""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        pecos_result = pc.mean(arr, axis=0)
        numpy_result = np.mean(arr, axis=0)

        assert np.allclose(pecos_result, numpy_result)


class TestStdCorrectness:
    """Test std() correctness against numpy."""

    def test_std_population_basic(self):
        """Test basic population standard deviation (ddof=0)."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        pecos_result = pc.std(values, ddof=0)
        numpy_result = np.std(values, ddof=0)

        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - 1.4142135623730951) < 1e-10

    def test_std_sample_basic(self):
        """Test basic sample standard deviation (ddof=1)."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        pecos_result = pc.std(values, ddof=1)
        numpy_result = np.std(values, ddof=1)

        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - 1.5811388300841898) < 1e-10

    def test_std_single_value(self):
        """Test std with single value (should be 0)."""
        values = [42.0]

        pecos_result = pc.std(values, ddof=0)
        numpy_result = np.std(values, ddof=0)

        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - 0.0) < 1e-10

    def test_std_empty(self):
        """Test std with empty sequence returns NaN."""
        values = []

        pecos_result = pc.std(values, ddof=0)

        assert np.isnan(pecos_result)

    def test_std_ddof_too_large(self):
        """Test std with ddof >= n returns NaN."""
        values = [1.0, 2.0]

        # With ddof=2, corrected n would be 0
        pecos_result = pc.std(values, ddof=2)

        assert np.isnan(pecos_result)

    def test_std_uniform_values(self):
        """Test std with all identical values (should be 0)."""
        values = [5.0, 5.0, 5.0, 5.0]

        pecos_result = pc.std(values, ddof=0)
        numpy_result = np.std(values, ddof=0)

        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - 0.0) < 1e-10

    def test_std_negative_values(self):
        """Test std with negative values."""
        values = [-3.0, -1.0, 1.0, 3.0]

        pecos_result = pc.std(values, ddof=0)
        numpy_result = np.std(values, ddof=0)

        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - 2.23606797749979) < 1e-10

    def test_std_two_values(self):
        """Test std with two values."""
        values = [1.0, 3.0]

        pecos_result = pc.std(values, ddof=0)
        numpy_result = np.std(values, ddof=0)

        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - 1.0) < 1e-10


class TestStdAnalysisUseCases:
    """Test std() with patterns from actual threshold analysis usage."""

    def test_jackknife_uncertainty(self):
        """Test the pattern from threshold_curve.py: jackknife parameter uncertainty."""
        # Simulating jackknife parameter estimates
        parameter_estimates = [1.5, 1.6, 1.4, 1.5, 1.7]

        pecos_result = pc.std(parameter_estimates, ddof=0)
        numpy_result = np.std(parameter_estimates, ddof=0)

        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - 0.10198039027185571) < 1e-10

    def test_bootstrap_pattern(self):
        """Test bootstrap parameter estimation pattern."""
        # Simulating bootstrap parameter estimates
        bootstrap_params = [2.1, 2.3, 2.0, 2.2, 2.1, 2.4]

        pecos_result = pc.std(bootstrap_params, ddof=0)
        numpy_result = np.std(bootstrap_params, ddof=0)

        assert abs(pecos_result - numpy_result) < 1e-10

    def test_threshold_fitting_uncertainty(self):
        """Test uncertainty estimation in threshold fitting."""
        # Simulating threshold parameter fits from multiple runs
        threshold_params = [0.01, 0.012, 0.009, 0.011, 0.010]

        pecos_result = pc.std(threshold_params, ddof=0)
        numpy_result = np.std(threshold_params, ddof=0)

        assert abs(pecos_result - numpy_result) < 1e-10


class TestStdAxisParameter:
    """Test std() with axis parameter for multi-dimensional arrays."""

    def test_2d_axis_0(self):
        """Test std along axis 0 (down columns)."""
        arr = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

        pecos_result = pc.std(arr, axis=0, ddof=0)
        numpy_result = np.std(arr, axis=0, ddof=0)

        assert np.allclose(pecos_result, numpy_result)
        assert np.allclose(pecos_result, [1.5, 1.5, 1.5])

    def test_2d_axis_1(self):
        """Test std along axis 1 (across rows)."""
        arr = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

        pecos_result = pc.std(arr, axis=1, ddof=0)
        numpy_result = np.std(arr, axis=1, ddof=0)

        assert np.allclose(pecos_result, numpy_result)

    def test_2d_axis_none(self):
        """Test std with axis=None (std of all elements)."""
        arr = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

        pecos_result = pc.std(arr, axis=None, ddof=0)
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

        pecos_result = pc.std(opt_list, axis=0, ddof=0)
        numpy_result = np.std(opt_list, axis=0, ddof=0)

        assert np.allclose(pecos_result, numpy_result)

    def test_3d_axis_0(self):
        """Test std on 3D array with axis=0."""
        arr = [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]

        pecos_result = pc.std(arr, axis=0, ddof=0)
        numpy_result = np.std(arr, axis=0, ddof=0)

        assert np.allclose(pecos_result, numpy_result)

    def test_numpy_array_input(self):
        """Test that numpy arrays work as input."""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        pecos_result = pc.std(arr, axis=0, ddof=0)
        numpy_result = np.std(arr, axis=0, ddof=0)

        assert np.allclose(pecos_result, numpy_result)

    def test_ddof_with_axis(self):
        """Test that ddof parameter works correctly with axis parameter."""
        arr = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

        # Test with ddof=1
        pecos_result = pc.std(arr, axis=0, ddof=1)
        numpy_result = np.std(arr, axis=0, ddof=1)

        assert np.allclose(pecos_result, numpy_result)


class TestPowerCorrectness:
    """Test power() correctness against numpy."""

    def test_power_scalar_basic(self):
        """Test basic scalar power operations."""
        assert pc.power(2.0, 3.0) == 8.0
        assert pc.power(3.0, 2.0) == 9.0
        assert pc.power(10.0, 0.0) == 1.0

    def test_power_fractional_exponent(self):
        """Test fractional powers (roots)."""
        pecos_result = pc.power(4.0, 0.5)
        numpy_result = np.power(4.0, 0.5)
        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - 2.0) < 1e-10

    def test_power_negative_exponent(self):
        """Test negative exponents."""
        pecos_result = pc.power(2.0, -1.0)
        numpy_result = np.power(2.0, -1.0)
        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - 0.5) < 1e-10

    def test_power_array_base_scalar_exp(self):
        """Test array base with scalar exponent."""
        base = [1.0, 2.0, 3.0]
        exponent = 2.0

        pecos_result = pc.power(base, exponent)
        numpy_result = np.power(base, exponent)

        assert np.allclose(pecos_result, numpy_result)
        assert np.allclose(pecos_result, [1.0, 4.0, 9.0])

    def test_power_scalar_base_array_exp(self):
        """Test scalar base with array exponent."""
        base = 2.0
        exponent = [1.0, 2.0, 3.0]

        pecos_result = pc.power(base, exponent)
        numpy_result = np.power(base, exponent)

        assert np.allclose(pecos_result, numpy_result)
        assert np.allclose(pecos_result, [2.0, 4.0, 8.0])

    def test_power_broadcasting(self):
        """Test broadcasting with arrays."""
        base = [[1.0, 2.0], [3.0, 4.0]]
        exponent = 2.0

        pecos_result = pc.power(base, exponent)
        numpy_result = np.power(base, exponent)

        assert np.allclose(pecos_result, numpy_result)
        assert np.allclose(pecos_result, [[1.0, 4.0], [9.0, 16.0]])


class TestPowerThresholdUseCases:
    """Test power() with patterns from threshold_curve.py."""

    def test_power_dist_scaling(self):
        """Test the pattern: np.power(dist, 1.0 / v0)."""
        dist = 5.0
        v0 = 2.0

        pecos_result = pc.power(dist, 1.0 / v0)
        numpy_result = np.power(dist, 1.0 / v0)

        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - np.sqrt(5.0)) < 1e-10

    def test_power_squared(self):
        """Test the pattern: np.power(x, 2)."""
        x = 3.5

        pecos_result = pc.power(x, 2.0)
        numpy_result = np.power(x, 2.0)

        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - 12.25) < 1e-10

    def test_power_negative_fractional(self):
        """Test the pattern: np.power(dist, -1.0 / u)."""
        dist = 5.0
        u = 2.0

        pecos_result = pc.power(dist, -1.0 / u)
        numpy_result = np.power(dist, -1.0 / u)

        assert abs(pecos_result - numpy_result) < 1e-10

    def test_power_array_scaling(self):
        """Test power with array of distances."""
        distances = np.array([3.0, 5.0, 7.0])
        v0 = 2.0

        pecos_result = pc.power(distances, 1.0 / v0)
        numpy_result = np.power(distances, 1.0 / v0)

        assert np.allclose(pecos_result, numpy_result)


class TestSqrtCorrectness:
    """Test sqrt() correctness against numpy."""

    def test_sqrt_perfect_squares(self):
        """Test perfect square roots."""
        assert pc.sqrt(4.0) == 2.0
        assert pc.sqrt(9.0) == 3.0
        assert pc.sqrt(16.0) == 4.0
        assert pc.sqrt(25.0) == 5.0
        assert pc.sqrt(100.0) == 10.0

    def test_sqrt_irrational(self):
        """Test irrational square roots."""
        pecos_result = pc.sqrt(2.0)
        numpy_result = np.sqrt(2.0)
        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - np.sqrt(2.0)) < 1e-10

    def test_sqrt_special_cases(self):
        """Test special cases."""
        assert pc.sqrt(0.0) == 0.0
        assert pc.sqrt(1.0) == 1.0
        assert np.isnan(pc.sqrt(-1.0))

    def test_sqrt_array(self):
        """Test array input."""
        values = [4.0, 9.0, 16.0, 25.0]
        pecos_result = pc.sqrt(values)
        numpy_result = np.sqrt(values)
        assert np.allclose(pecos_result, numpy_result)
        assert np.allclose(pecos_result, [2.0, 3.0, 4.0, 5.0])

    def test_sqrt_2d_array(self):
        """Test 2D array input."""
        values = [[4.0, 9.0], [16.0, 25.0]]
        pecos_result = pc.sqrt(values)
        numpy_result = np.sqrt(values)
        assert np.allclose(pecos_result, numpy_result)


class TestSqrtVarianceUseCases:
    """Test sqrt() with variance-to-std-deviation patterns."""

    def test_sqrt_variance_to_std(self):
        """Test the pattern: np.sqrt(variance)."""
        variance = 4.0
        pecos_result = pc.sqrt(variance)
        numpy_result = np.sqrt(variance)
        assert abs(pecos_result - numpy_result) < 1e-10
        assert abs(pecos_result - 2.0) < 1e-10

    def test_sqrt_variance_array(self):
        """Test variance to std deviation with arrays."""
        variances = np.array([1.0, 4.0, 9.0, 16.0])
        pecos_result = pc.sqrt(variances)
        numpy_result = np.sqrt(variances)
        assert np.allclose(pecos_result, numpy_result)
        assert np.allclose(pecos_result, [1.0, 2.0, 3.0, 4.0])

    def test_sqrt_diag_covariance(self):
        """Test extracting std from covariance matrix diagonal."""
        # Simulate covariance matrix diagonal (variances)
        covariance_diag = np.array([0.25, 1.0, 2.25, 4.0])
        pecos_result = pc.sqrt(covariance_diag)
        numpy_result = np.sqrt(covariance_diag)
        assert np.allclose(pecos_result, numpy_result)
        assert np.allclose(pecos_result, [0.5, 1.0, 1.5, 2.0])

    def test_sqrt_small_variances(self):
        """Test with small variance values."""
        variances = [0.01, 0.04, 0.0001]
        pecos_result = pc.sqrt(variances)
        numpy_result = np.sqrt(variances)
        assert np.allclose(pecos_result, numpy_result)
        assert np.allclose(pecos_result, [0.1, 0.2, 0.01])


class TestPolyfitCorrectness:
    """Test polyfit() correctness against numpy (without covariance)."""

    def test_polyfit_linear(self):
        """Test linear fit (degree 1)."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 3.0, 5.0, 7.0, 9.0])  # y = 2x + 1

        pecos_result = pc.polyfit(x, y, 1)
        numpy_result = np.polyfit(x, y, 1)

        assert np.allclose(pecos_result, numpy_result)
        assert np.allclose(pecos_result, [2.0, 1.0])

    def test_polyfit_quadratic(self):
        """Test quadratic fit (degree 2)."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 2.0, 5.0, 10.0, 17.0])  # y = x^2 + 1

        pecos_result = pc.polyfit(x, y, 2)
        numpy_result = np.polyfit(x, y, 2)

        assert np.allclose(pecos_result, numpy_result)
        assert np.allclose(pecos_result, [1.0, 0.0, 1.0])

    def test_polyfit_noisy_data(self):
        """Test fit with noisy data."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y = np.array([2.1, 4.9, 9.2, 15.8, 24.1, 35.9])

        pecos_result = pc.polyfit(x, y, 2)
        numpy_result = np.polyfit(x, y, 2)

        assert np.allclose(pecos_result, numpy_result)

    def test_polyfit_constant(self):
        """Test constant fit (degree 0)."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([3.1, 2.9, 3.0, 3.2, 2.8])

        pecos_result = pc.polyfit(x, y, 0)
        numpy_result = np.polyfit(x, y, 0)

        assert np.allclose(pecos_result, numpy_result)


class TestPolyfitCovariance:
    """Test polyfit() with covariance matrix (cov=True)."""

    def test_polyfit_cov_linear(self):
        """Test linear fit with covariance matrix."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 3.0, 5.0, 7.0, 9.0])

        pecos_coeffs, pecos_cov = pc.polyfit(x, y, 1, cov=True)
        numpy_coeffs, numpy_cov = np.polyfit(x, y, 1, cov=True)

        # Check coefficients match
        assert np.allclose(pecos_coeffs, numpy_coeffs)
        assert np.allclose(pecos_coeffs, [2.0, 1.0])

        # Check covariance matrices match
        assert pecos_cov.shape == (2, 2)
        assert np.allclose(pecos_cov, numpy_cov)

    def test_polyfit_cov_quadratic(self):
        """Test quadratic fit with covariance matrix."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y = np.array([2.1, 4.9, 9.2, 15.8, 24.1, 35.9])

        pecos_coeffs, pecos_cov = pc.polyfit(x, y, 2, cov=True)
        numpy_coeffs, numpy_cov = np.polyfit(x, y, 2, cov=True)

        # Check coefficients match
        assert np.allclose(pecos_coeffs, numpy_coeffs)

        # Check covariance matrices match
        assert pecos_cov.shape == (3, 3)
        assert np.allclose(pecos_cov, numpy_cov)

    def test_polyfit_cov_variances(self):
        """Test variance extraction from covariance matrix diagonal."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.1, 3.9, 6.2, 7.9, 10.1])

        pecos_coeffs, pecos_cov = pc.polyfit(x, y, 1, cov=True)
        numpy_coeffs, numpy_cov = np.polyfit(x, y, 1, cov=True)

        # Extract variances (diagonal elements)
        pecos_var = np.diag(pecos_cov)
        numpy_var = np.diag(numpy_cov)

        assert np.allclose(pecos_var, numpy_var)

        # Check standard errors
        pc.stderr = np.sqrt(pecos_var)
        numpy_stderr = np.sqrt(numpy_var)

        assert np.allclose(pc.stderr, numpy_stderr)

    def test_polyfit_cov_symmetric(self):
        """Test that covariance matrix is symmetric."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.5, 3.8, 5.2, 6.9, 8.1])

        _, pecos_cov = pc.polyfit(x, y, 2, cov=True)

        # Covariance matrix should be symmetric
        assert np.allclose(pecos_cov, pecos_cov.T)

    def test_polyfit_cov_false_explicit(self):
        """Test polyfit with cov=False returns only coefficients."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([1.0, 3.0, 5.0, 7.0])

        result = pc.polyfit(x, y, 1, cov=False)

        # Should return only coefficients, not a tuple
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert np.allclose(result, [2.0, 1.0])

    def test_polyfit_backward_compatibility(self):
        """Test that omitting cov parameter maintains backward compatibility."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([1.0, 3.0, 5.0, 7.0])

        # Without cov parameter (default behavior)
        result_default = pc.polyfit(x, y, 1)
        # With cov=False (explicit)
        result_false = pc.polyfit(x, y, 1, cov=False)

        # Both should return just coefficients
        assert isinstance(result_default, np.ndarray)
        assert isinstance(result_false, np.ndarray)
        assert np.allclose(result_default, result_false)
        assert np.allclose(result_default, [2.0, 1.0])


class TestPolyfitWithPoly1d:
    """Test polyfit() used with Poly1d for evaluation."""

    def test_polyfit_poly1d_linear(self):
        """Test using polyfit coefficients with Poly1d."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 3.0, 5.0, 7.0, 9.0])  # y = 2x + 1

        coeffs = pc.polyfit(x, y, 1)
        poly = pc.Poly1d(coeffs)

        # Evaluate at test points
        assert abs(poly.eval(0.0) - 1.0) < 1e-10
        assert abs(poly.eval(1.0) - 3.0) < 1e-10
        assert abs(poly.eval(2.0) - 5.0) < 1e-10
        assert abs(poly.eval(5.0) - 11.0) < 1e-10

    def test_polyfit_poly1d_quadratic(self):
        """Test using quadratic polyfit coefficients with Poly1d."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 2.0, 5.0, 10.0, 17.0])  # y = x^2 + 1

        coeffs = pc.polyfit(x, y, 2)
        poly = pc.Poly1d(coeffs)

        # Evaluate at test points
        assert abs(poly.eval(0.0) - 1.0) < 1e-10
        assert abs(poly.eval(1.0) - 2.0) < 1e-10
        assert abs(poly.eval(2.0) - 5.0) < 1e-10
        assert abs(poly.eval(5.0) - 26.0) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ============================================================================
# Sum Tests
# ============================================================================


class TestSumBasicTypes:
    """Test sum() with different input types."""

    def test_sum_list_float(self):
        """Test sum with list of floats."""
        from pecos import sum as pecos_sum

        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        pecos_result = pecos_sum(values)
        numpy_result = np.sum(values)

        assert pecos_result == numpy_result
        assert pecos_result == 15.0

    def test_sum_tuple_float(self):
        """Test sum with tuple of floats."""
        from pecos import sum as pecos_sum

        values = (1.0, 2.0, 3.0)

        pecos_result = pecos_sum(values)
        numpy_result = np.sum(values)

        assert pecos_result == numpy_result
        assert pecos_result == 6.0

    def test_sum_numpy_float(self):
        """Test sum with numpy array of floats."""
        from pecos import sum as pecos_sum

        values = np.array([1.0, 2.0, 3.0, 4.0])

        pecos_result = pecos_sum(values)
        numpy_result = np.sum(values)

        assert pecos_result == numpy_result
        assert pecos_result == 10.0

    def test_sum_complex_list(self):
        """Test sum with list of complex numbers."""
        from pecos import sum as pecos_sum

        values = [1 + 2j, 3 + 4j, 5 + 6j]

        pecos_result = pecos_sum(values)
        numpy_result = np.sum(values)

        assert pecos_result == numpy_result
        assert pecos_result == (9 + 12j)

    def test_sum_complex_numpy(self):
        """Test sum with numpy array of complex numbers."""
        from pecos import sum as pecos_sum

        values = np.array([1 + 2j, 3 + 4j, 5 + 6j])

        pecos_result = pecos_sum(values)
        numpy_result = np.sum(values)

        assert pecos_result == numpy_result
        assert pecos_result == (9 + 12j)


class TestSumAxisParameter:
    """Test sum() with axis parameter."""

    def test_sum_2d_axis_none(self):
        """Test sum with 2D array, axis=None (sum all elements)."""
        from pecos import sum as pecos_sum

        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        pecos_result = pecos_sum(arr, axis=None)
        numpy_result = np.sum(arr, axis=None)

        assert pecos_result == numpy_result
        assert pecos_result == 21.0

    def test_sum_2d_axis_0(self):
        """Test sum along axis 0 (down columns)."""
        from pecos import sum as pecos_sum

        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        pecos_result = pecos_sum(arr, axis=0)
        numpy_result = np.sum(arr, axis=0)

        np.testing.assert_array_equal(pecos_result, numpy_result)
        np.testing.assert_array_equal(pecos_result, [5.0, 7.0, 9.0])

    def test_sum_2d_axis_1(self):
        """Test sum along axis 1 (across rows)."""
        from pecos import sum as pecos_sum

        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        pecos_result = pecos_sum(arr, axis=1)
        numpy_result = np.sum(arr, axis=1)

        np.testing.assert_array_equal(pecos_result, numpy_result)
        np.testing.assert_array_equal(pecos_result, [6.0, 15.0])

    def test_sum_2d_axis_negative(self):
        """Test sum with negative axis."""
        from pecos import sum as pecos_sum

        arr = np.array([[1.0, 2.0], [3.0, 4.0]])

        # axis=-1 is same as axis=1 for 2D array
        pecos_result = pecos_sum(arr, axis=-1)
        numpy_result = np.sum(arr, axis=-1)

        np.testing.assert_array_equal(pecos_result, numpy_result)
        np.testing.assert_array_equal(pecos_result, [3.0, 7.0])

    def test_sum_3d_axis_0(self):
        """Test sum with 3D array along axis 0."""
        from pecos import sum as pecos_sum

        arr = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])

        pecos_result = pecos_sum(arr, axis=0)
        numpy_result = np.sum(arr, axis=0)

        np.testing.assert_array_equal(pecos_result, numpy_result)

    def test_sum_list_with_axis_0(self):
        """Test sum with list input and axis parameter."""
        from pecos import sum as pecos_sum

        values = [[1.0, 2.0], [3.0, 4.0]]

        pecos_result = pecos_sum(values, axis=0)
        numpy_result = np.sum(values, axis=0)

        np.testing.assert_array_equal(pecos_result, numpy_result)
        np.testing.assert_array_equal(pecos_result, [4.0, 6.0])


class TestSumComplexWithAxis:
    """Test sum() with complex numbers and axis parameter."""

    def test_sum_complex_2d_axis_0(self):
        """Test sum of complex 2D array along axis 0."""
        from pecos import sum as pecos_sum

        arr = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])

        pecos_result = pecos_sum(arr, axis=0)
        numpy_result = np.sum(arr, axis=0)

        np.testing.assert_array_equal(pecos_result, numpy_result)

    def test_sum_complex_2d_axis_1(self):
        """Test sum of complex 2D array along axis 1."""
        from pecos import sum as pecos_sum

        arr = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])

        pecos_result = pecos_sum(arr, axis=1)
        numpy_result = np.sum(arr, axis=1)

        np.testing.assert_array_equal(pecos_result, numpy_result)


class TestSumUseCases:
    """Test sum() in real quantum computing use cases."""

    def test_sum_probability_normalization(self):
        """Test sum for quantum state probability normalization check."""
        from pecos import abs as pecos_abs
        from pecos import sum as pecos_sum

        # Quantum state vector (normalized)
        state = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2) * 1j])

        # Calculate probability sum: sum(|psi|^2) should equal 1
        probs_np = np.abs(state) ** 2
        norm_np = np.sum(probs_np)

        # Using pecos functions
        probs_pecos = pecos_abs(state) ** 2
        norm_pecos = pecos_sum(probs_pecos)

        assert abs(norm_np - 1.0) < 1e-10
        assert abs(norm_pecos - 1.0) < 1e-10
        assert abs(norm_pecos - norm_np) < 1e-10

    def test_sum_complex_state_accumulation(self):
        """Test sum for accumulating complex quantum amplitudes."""
        from pecos import sum as pecos_sum

        # Complex amplitudes from different measurement outcomes
        amplitudes = np.array([0.5 + 0.5j, 0.3 - 0.2j, 0.1 + 0.7j])

        pecos_result = pecos_sum(amplitudes)
        numpy_result = np.sum(amplitudes)

        assert pecos_result == numpy_result
        assert abs(pecos_result - (0.9 + 1.0j)) < 1e-10

    def test_sum_threshold_analysis(self):
        """Test sum for threshold analysis (summing error rates)."""
        from pecos import sum as pecos_sum

        # Error rates across multiple qubits
        error_rates = [0.001, 0.0015, 0.002, 0.0012]

        total_error_pecos = pecos_sum(error_rates)
        total_error_numpy = np.sum(error_rates)

        assert abs(total_error_pecos - total_error_numpy) < 1e-10


class TestSumEdgeCases:
    """Test sum() edge cases."""

    def test_sum_empty_raises_error(self):
        """Test sum with empty array."""
        from pecos import sum as pecos_sum

        # NumPy returns 0.0 for empty array, we should match
        values = []

        pecos_result = pecos_sum(values)
        numpy_result = np.sum(values)

        assert pecos_result == numpy_result
        assert pecos_result == 0.0

    def test_sum_single_element(self):
        """Test sum with single element."""
        from pecos import sum as pecos_sum

        values = [42.0]

        pecos_result = pecos_sum(values)
        numpy_result = np.sum(values)

        assert pecos_result == numpy_result
        assert pecos_result == 42.0

    def test_sum_negative_values(self):
        """Test sum with negative values."""
        from pecos import sum as pecos_sum

        values = [-1.0, -2.0, -3.0, 4.0, 5.0]

        pecos_result = pecos_sum(values)
        numpy_result = np.sum(values)

        assert pecos_result == numpy_result
        assert pecos_result == 3.0

    def test_sum_mixed_sign_complex(self):
        """Test sum with mixed sign complex numbers."""
        from pecos import sum as pecos_sum

        values = np.array([1 + 2j, -3 + 4j, 5 - 6j])

        pecos_result = pecos_sum(values)
        numpy_result = np.sum(values)

        assert pecos_result == numpy_result
        assert pecos_result == (3 + 0j)

    def test_sum_axis_out_of_bounds(self):
        """Test sum with axis out of bounds raises error."""
        from pecos import sum as pecos_sum

        arr = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(ValueError, match="axis.*out of bounds"):
            pecos_sum(arr, axis=5)


class TestSumComparison:
    """Comprehensive comparison tests against NumPy."""

    def test_sum_matches_numpy_1d(self):
        """Test sum matches numpy for 1D arrays."""
        from pecos import sum as pecos_sum

        test_cases = [
            [1.0, 2.0, 3.0],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [-1.0, 0.0, 1.0],
            [100.0, 200.0, 300.0],
        ]

        for values in test_cases:
            pecos_result = pecos_sum(values)
            numpy_result = np.sum(values)
            assert abs(pecos_result - numpy_result) < 1e-10, f"Failed for {values}"

    def test_sum_matches_numpy_2d_all_axes(self):
        """Test sum matches numpy for 2D arrays with all axis values."""
        from pecos import sum as pecos_sum

        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        # Test axis=None
        pecos_result = pecos_sum(arr, axis=None)
        numpy_result = np.sum(arr, axis=None)
        assert pecos_result == numpy_result

        # Test axis=0
        pecos_result = pecos_sum(arr, axis=0)
        numpy_result = np.sum(arr, axis=0)
        np.testing.assert_array_equal(pecos_result, numpy_result)

        # Test axis=1
        pecos_result = pecos_sum(arr, axis=1)
        numpy_result = np.sum(arr, axis=1)
        np.testing.assert_array_equal(pecos_result, numpy_result)

    def test_sum_matches_numpy_complex(self):
        """Test sum matches numpy for complex arrays."""
        from pecos import sum as pecos_sum

        test_cases = [
            [1 + 1j, 2 + 2j, 3 + 3j],
            [0.5 - 0.5j, 0.5 + 0.5j],
            [1j, 2j, 3j],
        ]

        for values in test_cases:
            arr = np.array(values)
            pecos_result = pecos_sum(arr)
            numpy_result = np.sum(arr)
            assert pecos_result == numpy_result, f"Failed for {values}"
