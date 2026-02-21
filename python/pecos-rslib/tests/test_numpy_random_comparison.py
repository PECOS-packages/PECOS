"""
Comparison tests between pecos_rslib.num.random and numpy.random.

This module tests that our Rust implementations of numpy.random functions
produce statistically equivalent results to numpy's implementations.
"""

import time
import pytest

# Skip entire module if scipy/numpy not available
pytest.importorskip("scipy")
pytest.importorskip("numpy")

import numpy as np
from scipy import stats

import pecos as pc

# Mark all tests in this module as requiring numpy
pytestmark = pytest.mark.numpy


class TestRandomComparison:
    """Test random() function against numpy.random.random()."""

    def test_random_output_shape(self):
        """Test that output shapes match numpy."""
        for size in [1, 10, 100, 1000]:
            pecos_vals = pc.random.random(size)
            numpy_vals = np.random.random(size)

            assert pecos_vals.shape == numpy_vals.shape
            assert len(pecos_vals) == size

    def test_random_output_type(self):
        """Test that output type matches numpy."""
        pecos_vals = pc.random.random(100)
        numpy_vals = np.random.random(100)

        assert isinstance(pecos_vals, pc.Array)
        assert pecos_vals.dtype == numpy_vals.dtype

    def test_random_range(self):
        """Test that all values are in [0, 1) like numpy."""
        vals = pc.random.random(10000)
        assert np.all(vals >= 0.0)
        assert np.all(vals < 1.0)

    def test_random_statistical_mean(self):
        """Test that mean is approximately 0.5 (uniform distribution)."""
        vals = pc.random.random(10000)
        mean = np.mean(vals)

        # For uniform [0, 1), theoretical mean = 0.5
        # With n=10000, standard error ~ 0.003
        assert abs(mean - 0.5) < 0.02, f"Mean {mean} too far from expected 0.5"

    def test_random_statistical_variance(self):
        """Test that variance matches uniform [0, 1) distribution."""
        vals = pc.random.random(10000)
        variance = np.var(vals)

        # For uniform [0, 1), theoretical variance = 1/12 ≈ 0.0833
        expected_variance = 1.0 / 12.0
        assert (
            abs(variance - expected_variance) < 0.01
        ), f"Variance {variance} too far from expected {expected_variance}"

    def test_random_uniformity_ks_test(self):
        """Test uniformity using Kolmogorov-Smirnov test."""
        # Use a fixed seed for deterministic test behavior
        pc.random.seed(42)
        vals = pc.random.random(1000)

        # KS test against uniform [0, 1) distribution
        ks_statistic, p_value = stats.kstest(vals, "uniform")

        # p-value > 0.01 means we can't reject the null hypothesis
        # (i.e., data is consistent with uniform distribution)
        assert p_value > 0.01, f"KS test failed: p={p_value}, statistic={ks_statistic}"

    def test_random_chi_square_uniformity(self):
        """Test uniformity using chi-square goodness-of-fit test."""
        # Use a fixed seed for deterministic test behavior
        pc.random.seed(123)
        vals = pc.random.random(10000)

        # Divide [0, 1) into 10 bins
        num_bins = 10
        observed, _ = np.histogram(vals, bins=num_bins, range=(0, 1))
        expected = np.full(num_bins, len(vals) / num_bins)

        # Chi-square test
        chi2_statistic, p_value = stats.chisquare(observed, expected)

        # p-value > 0.01 means distribution is consistent with uniform
        assert p_value > 0.01, f"Chi-square test failed: p={p_value}, statistic={chi2_statistic}"


class TestRandintComparison:
    """Test randint() function against numpy.random.randint()."""

    def test_randint_array_shape(self):
        """Test that output shapes match numpy."""
        for size in [1, 10, 100]:
            pecos_vals = pc.random.randint(0, 10, size)
            numpy_vals = np.random.randint(0, 10, size)

            assert pecos_vals.shape == numpy_vals.shape
            assert len(pecos_vals) == size

    def test_randint_array_type(self):
        """Test that output type matches numpy."""
        pecos_vals = pc.random.randint(0, 10, 100)
        numpy_vals = np.random.randint(0, 10, 100)

        assert isinstance(pecos_vals, pc.Array)
        assert pecos_vals.dtype == numpy_vals.dtype

    def test_randint_scalar_type(self):
        """Test that scalar output is Python int like numpy."""
        pecos_val = pc.random.randint(0, 10)
        numpy_val = np.random.randint(0, 10)

        assert isinstance(pecos_val, int)
        assert isinstance(numpy_val, (int, np.integer))

    def test_randint_range(self):
        """Test that values are in correct range [low, high)."""
        vals = pc.random.randint(5, 15, 1000)
        assert np.all(vals >= 5)
        assert np.all(vals < 15)

    def test_randint_negative_range(self):
        """Test that negative ranges work like numpy."""
        vals = pc.random.randint(-10, 10, 1000)
        assert np.all(vals >= -10)
        assert np.all(vals < 10)

    def test_randint_uniformity(self):
        """Test that randint produces uniform distribution."""
        low, high = 0, 10
        vals = pc.random.randint(low, high, 10000)

        # Count occurrences of each value
        unique, counts = np.unique(vals, return_counts=True)
        expected_count = len(vals) / (high - low)

        # Chi-square test for uniformity
        chi2_statistic, p_value = stats.chisquare(counts, np.full(len(unique), expected_count))

        assert p_value > 0.01, f"Chi-square test failed: p={p_value}, statistic={chi2_statistic}"

    def test_randint_default_low(self):
        """Test [0, n) behavior when only one argument provided."""
        # NumPy: np.random.randint(10) gives [0, 10)
        # Our API: randint(10, None) gives [0, 10)
        vals = pc.random.randint(10, None, 100)
        assert np.all(vals >= 0)
        assert np.all(vals < 10)


class TestChoiceComparison:
    """Test choice() function against numpy.random.choice()."""

    def test_choice_scalar_type(self):
        """Test that scalar choice returns correct type."""
        items = ["X", "Y", "Z"]
        sample = pc.random.choice(items)

        assert isinstance(sample, str)
        assert sample in items

    def test_choice_array_length(self):
        """Test that array choice returns correct length."""
        items = [1, 2, 3, 4, 5]
        for size in [1, 5, 10, 100]:
            samples = pc.random.choice(items, size)
            assert len(samples) == size

    def test_choice_all_valid(self):
        """Test that all samples are from the original array."""
        items = ["A", "B", "C"]
        samples = pc.random.choice(items, 1000)

        for sample in samples:
            assert sample in items

    def test_choice_with_replacement_allows_duplicates(self):
        """Test that choice with replacement can produce duplicates."""
        items = ["X", "Y", "Z"]
        samples = pc.random.choice(items, 100, replace=True)

        # With replacement and 100 samples from 3 items, we SHOULD see duplicates
        unique_count = len(set(samples))
        assert unique_count <= len(items)

    def test_choice_without_replacement_no_duplicates(self):
        """Test that choice without replacement produces no duplicates."""
        items = [1, 2, 3, 4, 5]
        samples = pc.random.choice(items, 5, replace=False)

        # Without replacement, all samples should be unique
        assert len(set(samples)) == 5
        assert set(samples) == set(items)

    def test_choice_without_replacement_error(self):
        """Test that choice without replacement fails if size > len(array)."""
        items = [1, 2, 3]

        with pytest.raises(ValueError, match="Cannot take larger sample"):
            pc.random.choice(items, 5, replace=False)

    def test_choice_empty_array_error(self):
        """Test that choice from empty array raises error."""
        with pytest.raises(ValueError, match="Cannot sample from empty"):
            pc.random.choice([], 5)

    def test_choice_uniformity(self):
        """Test that choice samples uniformly from array."""
        items = [0, 1, 2, 3, 4]
        samples = pc.random.choice(items, 10000)

        # Count occurrences
        unique, counts = np.unique(samples, return_counts=True)
        expected_count = len(samples) / len(items)

        # Chi-square test for uniformity
        chi2_statistic, p_value = stats.chisquare(counts, np.full(len(unique), expected_count))

        assert p_value > 0.01, f"Chi-square test failed: p={p_value}, statistic={chi2_statistic}"

    def test_choice_with_numpy_array(self):
        """Test that choice works with numpy arrays like numpy.random.choice."""
        items = np.array([10, 20, 30, 40, 50])
        samples = pc.random.choice(items, 100)

        for sample in samples:
            assert sample in items


class TestPerformanceComparison:
    """Basic performance comparison tests."""

    @pytest.mark.performance
    def test_random_performance(self):
        """Compare performance of random() vs numpy.random.random()."""
        size = 100000

        # Time our implementation
        start = time.perf_counter()
        for _ in range(10):
            pc.random.random(size)
        pecos_time = time.perf_counter() - start

        # Time numpy
        start = time.perf_counter()
        for _ in range(10):
            np.random.random(size)
        numpy_time = time.perf_counter() - start

        speedup = numpy_time / pecos_time
        print(f"\nrandom({size}) speedup: {speedup:.2f}x")

        # We expect 1.2-2x speedup, but don't fail if slower
        # (depends on numpy version, CPU, etc.)
        assert speedup > 0.5, f"Implementation is too slow: {speedup:.2f}x"

    @pytest.mark.performance
    def test_randint_performance(self):
        """Compare performance of randint() vs numpy.random.randint()."""
        size = 100000

        # Time our implementation
        start = time.perf_counter()
        for _ in range(10):
            pc.random.randint(0, 100, size)
        pecos_time = time.perf_counter() - start

        # Time numpy
        start = time.perf_counter()
        for _ in range(10):
            np.random.randint(0, 100, size)
        numpy_time = time.perf_counter() - start

        speedup = numpy_time / pecos_time
        print(f"\nrandint(0, 100, {size}) speedup: {speedup:.2f}x")

        # We expect 1.2-1.5x speedup
        assert speedup > 0.5, f"Implementation is too slow: {speedup:.2f}x"

    @pytest.mark.performance
    def test_choice_performance(self):
        """Compare performance of choice() vs numpy.random.choice()."""
        items = list(range(100))
        size = 10000

        # Time our implementation
        start = time.perf_counter()
        for _ in range(10):
            pc.random.choice(items, size)
        pecos_time = time.perf_counter() - start

        # Time numpy
        start = time.perf_counter()
        for _ in range(10):
            np.random.choice(items, size)
        numpy_time = time.perf_counter() - start

        speedup = numpy_time / pecos_time
        print(f"\nchoice(100 items, {size}) speedup: {speedup:.2f}x")

        # We expect 1.3-2x speedup
        assert speedup > 0.5, f"Implementation is too slow: {speedup:.2f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
