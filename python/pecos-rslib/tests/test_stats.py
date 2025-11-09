"""Tests for statistical functions comparing pecos-rslib vs numpy."""

import numpy as np
import pytest

from pecos_rslib.num import mean as pecos_mean
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
