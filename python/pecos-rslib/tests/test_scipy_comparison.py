"""Tests comparing pecos_rslib.num with scipy.optimize."""

import pytest

# Skip entire module if scipy/numpy not available
pytest.importorskip("scipy")
pytest.importorskip("numpy")

import numpy as np

# Import both our implementation and scipy
import pecos as pc

from scipy.optimize import brentq as scipy_brentq
from scipy.optimize import newton as scipy_newton
from scipy.optimize import curve_fit as scipy_curve_fit

# Mark all tests in this module as requiring numpy
pytestmark = pytest.mark.numpy


class TestBrentqComparison:
    """Compare brentq implementations."""

    def test_sqrt2(self) -> None:
        """Find sqrt(2) by solving x^2 - 2 = 0."""

        def f(x):
            return x * x - 2.0

        pecos_root = pc.brentq(f, 0.0, 2.0)
        scipy_root = scipy_brentq(f, 0.0, 2.0)

        assert abs(pecos_root - scipy_root) < 1e-10
        assert abs(pecos_root - np.sqrt(2)) < 1e-10

    def test_cubic(self) -> None:
        """Find root of x^3 - x - 2 = 0."""

        def f(x):
            return x**3 - x - 2.0

        pecos_root = pc.brentq(f, 1.0, 2.0)
        scipy_root = scipy_brentq(f, 1.0, 2.0)

        assert abs(pecos_root - scipy_root) < 1e-10
        # Verify both found the correct root
        assert abs(f(pecos_root)) < 1e-10
        assert abs(f(scipy_root)) < 1e-10

    def test_transcendental(self) -> None:
        """Find root of cos(x) = x."""

        def f(x):
            return np.cos(x) - x

        pecos_root = pc.brentq(f, 0.0, 1.0)
        scipy_root = scipy_brentq(f, 0.0, 1.0)

        assert abs(pecos_root - scipy_root) < 1e-10

    def test_exponential(self) -> None:
        """Find root of e^x = 3."""

        def f(x):
            return np.exp(x) - 3.0

        pecos_root = pc.brentq(f, 0.0, 2.0)
        scipy_root = scipy_brentq(f, 0.0, 2.0)

        assert abs(pecos_root - scipy_root) < 1e-10
        assert abs(pecos_root - np.log(3)) < 1e-10

    def test_polynomial_near_zero(self) -> None:
        """Test with polynomial that has root near zero."""

        def f(x):
            return x**3 - 0.001

        pecos_root = pc.brentq(f, 0.0, 1.0)
        scipy_root = scipy_brentq(f, 0.0, 1.0)

        assert abs(pecos_root - scipy_root) < 1e-10

    def test_steep_function(self) -> None:
        """Test with steep function."""

        def f(x):
            return np.tanh(10 * x)

        pecos_root = pc.brentq(f, -1.0, 1.0)
        scipy_root = scipy_brentq(f, -1.0, 1.0)

        assert abs(pecos_root - scipy_root) < 1e-10

    def test_same_sign_error(self) -> None:
        """Verify both implementations raise error for same-sign endpoints."""

        def f(x):
            return x * x + 1.0  # No real roots

        with pytest.raises(ValueError, match="opposite signs"):
            pc.brentq(f, -1.0, 1.0)

        with pytest.raises(ValueError, match="sign"):
            scipy_brentq(f, -1.0, 1.0)


class TestNewtonComparison:
    """Compare newton implementations."""

    def test_sqrt2_with_derivative(self) -> None:
        """Find sqrt(2) with analytical derivative."""

        def f(x):
            return x * x - 2.0

        def fprime(x):
            return 2.0 * x

        pecos_root = pc.newton(f, 1.0, fprime=fprime)
        scipy_root = scipy_newton(f, 1.0, fprime=fprime)

        assert abs(pecos_root - scipy_root) < 1e-8
        assert abs(pecos_root - np.sqrt(2)) < 1e-8

    def test_sqrt2_numerical_derivative(self) -> None:
        """Find sqrt(2) with numerical derivative."""

        def f(x):
            return x * x - 2.0

        pecos_root = pc.newton(f, 1.0)
        scipy_root = scipy_newton(f, 1.0)

        # Numerical derivatives may differ slightly, so use larger tolerance
        assert abs(pecos_root - scipy_root) < 1e-6
        assert abs(pecos_root - np.sqrt(2)) < 1e-6

    def test_cubic_polynomial(self) -> None:
        """Find root of x^3 - x - 2 = 0."""

        def f(x):
            return x**3 - x - 2.0

        def fprime(x):
            return 3.0 * x**2 - 1.0

        pecos_root = pc.newton(f, 1.5, fprime=fprime)
        scipy_root = scipy_newton(f, 1.5, fprime=fprime)

        assert abs(pecos_root - scipy_root) < 1e-8

    def test_exponential_function(self) -> None:
        """Find root of e^x - 3 = 0."""

        def f(x):
            return np.exp(x) - 3.0

        def fprime(x):
            return np.exp(x)

        pecos_root = pc.newton(f, 1.0, fprime=fprime)
        scipy_root = scipy_newton(f, 1.0, fprime=fprime)

        assert abs(pecos_root - scipy_root) < 1e-8
        assert abs(pecos_root - np.log(3)) < 1e-8

    def test_transcendental(self) -> None:
        """Find root of cos(x) = x."""

        def f(x):
            return np.cos(x) - x

        def fprime(x):
            return -np.sin(x) - 1.0

        pecos_root = pc.newton(f, 0.5, fprime=fprime)
        scipy_root = scipy_newton(f, 0.5, fprime=fprime)

        assert abs(pecos_root - scipy_root) < 1e-8

    def test_difficult_initial_guess(self) -> None:
        """Test convergence from a non-ideal starting point."""

        def f(x):
            return x**3 - 2 * x - 5

        def fprime(x):
            return 3 * x**2 - 2

        # Start far from the root
        pecos_root = pc.newton(f, 3.0, fprime=fprime)
        scipy_root = scipy_newton(f, 3.0, fprime=fprime)

        assert abs(pecos_root - scipy_root) < 1e-8


class TestCurveFitComparison:
    """Compare curve_fit implementations."""

    def test_linear_fit(self) -> None:
        """Fit y = a*x + b."""

        def linear(x, a, b):
            return a * x + b

        xdata = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ydata = np.array([1.0, 3.0, 5.0, 7.0, 9.0])  # y = 2*x + 1
        p0 = np.array([1.0, 0.0])

        pecos_popt, pecos_pcov = pc.curve_fit(linear, xdata, ydata, p0)
        scipy_popt, scipy_pcov = scipy_curve_fit(linear, xdata, ydata, p0)

        # Parameters should match closely
        np.testing.assert_allclose(pecos_popt, scipy_popt, rtol=1e-6, atol=1e-8)
        # Covariances should match (may have small numerical differences)
        np.testing.assert_allclose(pecos_pcov, scipy_pcov, rtol=1e-4, atol=1e-8)

    def test_exponential_fit(self) -> None:
        """Fit y = a * exp(b * x)."""

        def exponential(x, a, b):
            return a * np.exp(b * x)

        xdata = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ydata = np.array([1.0, 2.718, 7.389, 20.086, 54.598])
        p0 = np.array([1.0, 1.0])

        pecos_popt, pecos_pcov = pc.curve_fit(exponential, xdata, ydata, p0)
        scipy_popt, scipy_pcov = scipy_curve_fit(exponential, xdata, ydata, p0)

        np.testing.assert_allclose(pecos_popt, scipy_popt, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(pecos_pcov, scipy_pcov, rtol=0.1, atol=1e-6)

    def test_quadratic_fit(self) -> None:
        """Fit y = a*x^2 + b*x + c."""

        def quadratic(x, a, b, c):
            return a * x**2 + b * x + c

        xdata = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ydata = np.array([3.0, 6.0, 11.0, 18.0, 27.0])  # y = x^2 + 2*x + 3
        p0 = np.array([1.0, 1.0, 1.0])

        pecos_popt, pecos_pcov = pc.curve_fit(quadratic, xdata, ydata, p0)
        scipy_popt, scipy_pcov = scipy_curve_fit(quadratic, xdata, ydata, p0)

        np.testing.assert_allclose(pecos_popt, scipy_popt, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(pecos_pcov, scipy_pcov, rtol=1e-4, atol=1e-8)

    def test_gaussian_fit(self) -> None:
        """Fit Gaussian function."""

        def gaussian(x, amp, mu, sigma):
            return amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

        xdata = np.linspace(-5, 5, 50)
        ydata = gaussian(xdata, 2.0, 1.0, 1.5) + 0.01 * np.random.randn(50)
        p0 = np.array([1.0, 0.0, 1.0])

        # Set random seed for reproducibility
        np.random.seed(42)
        ydata = gaussian(xdata, 2.0, 1.0, 1.5) + 0.01 * np.random.randn(50)

        pecos_popt, pecos_pcov = pc.curve_fit(gaussian, xdata, ydata, p0, maxfev=5000)
        scipy_popt, scipy_pcov = scipy_curve_fit(gaussian, xdata, ydata, p0, maxfev=5000)

        # Parameters should be close (some variation due to optimization differences)
        np.testing.assert_allclose(pecos_popt, scipy_popt, rtol=0.1, atol=0.1)

    def test_tuple_xdata_quantum_pecos_pattern(self) -> None:
        """Test curve_fit with tuple xdata (quantum-pecos pattern)."""

        def func(x, a, b, c):
            p, d = x
            return a * p ** (b / d + c)

        p = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
        d = np.array([3, 3, 3, 3, 3])  # Integer array!
        plog = np.array([0.01, 0.015, 0.02, 0.025, 0.03])
        p0 = np.array([1.0, 1.0, 1.0])

        pecos_popt, pecos_pcov = pc.curve_fit(func, (p, d), plog, p0, maxfev=5000)
        scipy_popt, scipy_pcov = scipy_curve_fit(func, (p, d), plog, p0, maxfev=5000)

        # This is a difficult optimization problem - different optimizers may converge
        # to different local minima. Instead of comparing parameters, verify both
        # implementations produce good fits to the data.
        pecos_residuals = []
        scipy_residuals = []
        for i in range(len(p)):
            pecos_pred = func((p[i], d[i]), *pecos_popt)
            scipy_pred = func((p[i], d[i]), *scipy_popt)
            pecos_residuals.append((pecos_pred - plog[i]) ** 2)
            scipy_residuals.append((scipy_pred - plog[i]) ** 2)

        pecos_rmse = np.sqrt(np.mean(pecos_residuals))
        scipy_rmse = np.sqrt(np.mean(scipy_residuals))

        # Both should produce reasonable fits
        assert pecos_rmse < 0.01, f"PECOS fit too poor: RMSE={pecos_rmse}"
        assert scipy_rmse < 0.01, f"Scipy fit too poor: RMSE={scipy_rmse}"
        # And similar fit quality
        assert abs(pecos_rmse - scipy_rmse) < 0.005, "Fit quality differs too much"

    def test_sine_fit(self) -> None:
        """Fit sine wave."""

        def sine(x, amp, freq, phase):
            return amp * np.sin(2 * np.pi * freq * x + phase)

        xdata = np.linspace(0, 2, 100)
        np.random.seed(42)
        ydata = sine(xdata, 1.5, 2.0, 0.5) + 0.05 * np.random.randn(100)
        p0 = np.array([1.0, 2.0, 0.0])

        pecos_popt, pecos_pcov = pc.curve_fit(sine, xdata, ydata, p0, maxfev=5000)
        scipy_popt, scipy_pcov = scipy_curve_fit(sine, xdata, ydata, p0, maxfev=5000)

        # Parameters should be similar
        np.testing.assert_allclose(pecos_popt, scipy_popt, rtol=0.1, atol=0.1)

    def test_power_law_fit(self) -> None:
        """Fit power law y = a * x^b."""

        def power_law(x, a, b):
            return a * x**b

        xdata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ydata = np.array([2.0, 5.66, 10.39, 16.0, 22.36])  # y ≈ 2*x^1.5
        p0 = np.array([1.0, 1.0])

        pecos_popt, pecos_pcov = pc.curve_fit(power_law, xdata, ydata, p0)
        scipy_popt, scipy_pcov = scipy_curve_fit(power_law, xdata, ydata, p0)

        np.testing.assert_allclose(pecos_popt, scipy_popt, rtol=1e-3, atol=1e-4)

    def test_noisy_data(self) -> None:
        """Test with noisy data."""

        def linear(x, a, b):
            return a * x + b

        np.random.seed(123)
        xdata = np.linspace(0, 10, 50)
        ydata = 2.5 * xdata + 1.3 + np.random.normal(0, 0.5, 50)
        p0 = np.array([1.0, 0.0])

        pecos_popt, pecos_pcov = pc.curve_fit(linear, xdata, ydata, p0)
        scipy_popt, scipy_pcov = scipy_curve_fit(linear, xdata, ydata, p0)

        # Should converge to similar values
        np.testing.assert_allclose(pecos_popt, scipy_popt, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(pecos_pcov, scipy_pcov, rtol=0.01, atol=1e-8)

    def test_p0_accepts_sequence_types(self) -> None:
        """Test that p0 accepts tuple, list, and array (scipy compatibility)."""

        def quadratic(x, a, b, c):
            return a * x**2 + b * x + c

        xdata = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ydata = np.array([1.0, 2.0, 5.0, 10.0, 17.0])  # y = x^2 + 1

        # Test with tuple (quantum-pecos usage pattern)
        p0_tuple = (1.0, 0.0, 0.0)
        popt_tuple, _ = pc.curve_fit(quadratic, xdata, ydata, p0_tuple)

        # Test with list
        p0_list = [1.0, 0.0, 0.0]
        popt_list, _ = pc.curve_fit(quadratic, xdata, ydata, p0_list)

        # Test with array
        p0_array = np.array([1.0, 0.0, 0.0])
        popt_array, _ = pc.curve_fit(quadratic, xdata, ydata, p0_array)

        # All should produce the same result
        np.testing.assert_allclose(popt_tuple, popt_array, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(popt_list, popt_array, rtol=1e-10, atol=1e-12)

        # Should match expected values
        np.testing.assert_allclose(popt_array, [1.0, 0.0, 1.0], rtol=1e-6, atol=1e-8)


class TestPolyfitComparison:
    """Compare polyfit implementations."""

    def test_linear_fit(self) -> None:
        """Fit degree 1 polynomial (line)."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 3.0, 5.0, 7.0, 9.0])  # y = 2*x + 1

        pecos_coeffs = pc.polyfit(x, y, 1)
        scipy_coeffs = np.polyfit(x, y, 1)

        np.testing.assert_allclose(pecos_coeffs, scipy_coeffs, rtol=1e-10, atol=1e-12)

    def test_quadratic_fit(self) -> None:
        """Fit degree 2 polynomial."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([3.0, 6.0, 11.0, 18.0, 27.0])  # y = x^2 + 2*x + 3

        pecos_coeffs = pc.polyfit(x, y, 2)
        scipy_coeffs = np.polyfit(x, y, 2)

        np.testing.assert_allclose(pecos_coeffs, scipy_coeffs, rtol=1e-10, atol=1e-12)

    def test_cubic_fit(self) -> None:
        """Fit degree 3 polynomial."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 3.0, 17.0, 55.0, 129.0, 251.0])  # y = x^3 + 2*x^2 + 3*x + 1

        pecos_coeffs = pc.polyfit(x, y, 3)
        scipy_coeffs = np.polyfit(x, y, 3)

        np.testing.assert_allclose(pecos_coeffs, scipy_coeffs, rtol=1e-9, atol=1e-10)

    def test_high_degree(self) -> None:
        """Test higher degree polynomial."""
        np.random.seed(42)
        x = np.linspace(0, 1, 20)
        # Generate y from a degree 5 polynomial
        true_coeffs = np.array([1.0, -2.0, 3.0, -1.0, 2.0, 1.0])
        y = np.polyval(true_coeffs, x)

        pecos_coeffs = pc.polyfit(x, y, 5)
        scipy_coeffs = np.polyfit(x, y, 5)

        np.testing.assert_allclose(pecos_coeffs, scipy_coeffs, rtol=1e-8, atol=1e-10)
        # Verify we recovered the original coefficients
        np.testing.assert_allclose(pecos_coeffs, true_coeffs, rtol=1e-8, atol=1e-10)

    def test_noisy_data(self) -> None:
        """Test polyfit with noisy data."""
        np.random.seed(456)
        x = np.linspace(0, 5, 30)
        y = 2 * x**2 - 3 * x + 1 + np.random.normal(0, 0.5, 30)

        pecos_coeffs = pc.polyfit(x, y, 2)
        scipy_coeffs = np.polyfit(x, y, 2)

        # Should get similar coefficients
        np.testing.assert_allclose(pecos_coeffs, scipy_coeffs, rtol=1e-8, atol=1e-10)

    def test_overdetermined_system(self) -> None:
        """Test with many more data points than parameters."""
        np.random.seed(789)
        x = np.linspace(-2, 2, 100)
        y = 1.5 * x + 2.0 + np.random.normal(0, 0.1, 100)

        pecos_coeffs = pc.polyfit(x, y, 1)
        scipy_coeffs = np.polyfit(x, y, 1)

        np.testing.assert_allclose(pecos_coeffs, scipy_coeffs, rtol=1e-8, atol=1e-10)


class TestPoly1dComparison:
    """Compare Poly1d implementations."""

    def test_evaluation(self) -> None:
        """Test polynomial evaluation."""
        coeffs = np.array([2.0, 3.0, 1.0])  # 2*x^2 + 3*x + 1

        pecos_poly = pc.Poly1d(coeffs)
        scipy_poly = np.poly1d(coeffs)

        test_points = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.5]
        for x in test_points:
            pecos_val = pecos_poly.eval(x)
            scipy_val = scipy_poly(x)
            assert abs(pecos_val - scipy_val) < 1e-12

    def test_degree(self) -> None:
        """Test degree calculation."""
        coeffs = np.array([1.0, 2.0, 3.0, 4.0])  # degree 3

        pecos_poly = pc.Poly1d(coeffs)
        scipy_poly = np.poly1d(coeffs)

        assert pecos_poly.degree() == len(coeffs) - 1
        assert pecos_poly.degree() == scipy_poly.order

    def test_fit_and_evaluate(self) -> None:
        """Test fitting then evaluating."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 3.0, 5.0, 7.0, 9.0])

        pecos_coeffs = pc.polyfit(x, y, 1)
        scipy_coeffs = np.polyfit(x, y, 1)

        pecos_poly = pc.Poly1d(pecos_coeffs)
        scipy_poly = np.poly1d(scipy_coeffs)

        # Evaluate at original points
        for xi, yi in zip(x, y, strict=False):
            pecos_val = pecos_poly.eval(xi)
            scipy_val = scipy_poly(xi)
            assert abs(pecos_val - scipy_val) < 1e-10
            assert abs(pecos_val - yi) < 1e-10

    def test_complex_polynomial(self) -> None:
        """Test with complex polynomial."""
        coeffs = np.array([1.0, -2.5, 3.7, -1.2, 0.5])

        pecos_poly = pc.Poly1d(coeffs)
        scipy_poly = np.poly1d(coeffs)

        test_points = np.linspace(-3, 3, 20)
        for x in test_points:
            pecos_val = pecos_poly.eval(x)
            scipy_val = scipy_poly(x)
            assert abs(pecos_val - scipy_val) < 1e-10


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_brentq_narrow_interval(self) -> None:
        """Test brentq with very narrow interval."""

        def f(x):
            return x - 0.5

        pecos_root = pc.brentq(f, 0.4999, 0.5001)
        scipy_root = scipy_brentq(f, 0.4999, 0.5001)

        assert abs(pecos_root - scipy_root) < 1e-10

    def test_newton_near_zero_derivative(self) -> None:
        """Test newton with function that has small derivative.

        Note: This is a pathological case where x^3 has a triple root at 0.
        Newton's method may struggle to converge precisely due to the flat derivative.
        """

        def f(x):
            return x**3

        def fprime(x):
            return 3 * x**2

        # Both should converge to something close to 0
        pecos_root = pc.newton(f, 0.1, fprime=fprime)
        scipy_root = scipy_newton(f, 0.1, fprime=fprime)

        # Verify both find a root (may not be exactly 0 due to numerical issues)
        assert abs(f(pecos_root)) < 1e-6, f"PECOS didn't find root: f({pecos_root})={f(pecos_root)}"
        assert abs(f(scipy_root)) < 1e-6, f"Scipy didn't find root: f({scipy_root})={f(scipy_root)}"

        # Both should be close to 0 (allow larger tolerance due to pathological case)
        assert abs(pecos_root) < 0.01, f"PECOS root too far from 0: {pecos_root}"
        assert abs(scipy_root) < 0.01, f"Scipy root too far from 0: {scipy_root}"

    def test_curve_fit_exact_fit(self) -> None:
        """Test curve_fit with data that fits model exactly."""

        def linear(x, a, b):
            return a * x + b

        xdata = np.array([0.0, 1.0, 2.0])
        ydata = np.array([1.0, 3.0, 5.0])  # Exactly y = 2*x + 1
        p0 = np.array([1.0, 0.0])

        pecos_popt, _ = pc.curve_fit(linear, xdata, ydata, p0)
        scipy_popt, _ = scipy_curve_fit(linear, xdata, ydata, p0)

        # Should get exact solution
        np.testing.assert_allclose(pecos_popt, [2.0, 1.0], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(scipy_popt, [2.0, 1.0], rtol=1e-10, atol=1e-12)

    def test_polyfit_exact_degree(self) -> None:
        """Test polyfit when data is exact polynomial."""
        # Generate data from exact polynomial
        x = np.array([0.0, 1.0, 2.0, 3.0])
        coeffs_true = np.array([2.0, -1.0, 3.0])  # 2*x^2 - x + 3
        y = np.polyval(coeffs_true, x)

        pecos_coeffs = pc.polyfit(x, y, 2)
        scipy_coeffs = np.polyfit(x, y, 2)

        # Should recover exact coefficients
        np.testing.assert_allclose(pecos_coeffs, coeffs_true, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(scipy_coeffs, coeffs_true, rtol=1e-12, atol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
