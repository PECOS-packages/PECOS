"""Tests for fused RNG operations comparing against numpy unfused versions.

This test suite validates correctness and performance of fused operations:
- compare_any(): Fused random generation + any() reduction
- compare_indices(): Fused random generation + filtering
"""

import time

import numpy as np
import pytest

import pecos as pc


class TestCompareAnyCorrectness:
    """Test compare_any correctness against numpy."""

    def test_compare_any_always_true(self):
        """With threshold=1.0, should always be True."""
        # Numpy version
        pc.random.seed(42)
        result = pc.random.compare_any(100, 1.0)
        assert result is True

    def test_compare_any_always_false(self):
        """With threshold=0.0, should always be False."""
        pc.random.seed(42)
        result = pc.random.compare_any(100, 0.0)
        assert result is False

    def test_compare_any_reproducibility(self):
        """Same seed should produce same result."""
        pc.random.seed(12345)
        result1 = pc.random.compare_any(1000, 0.05)

        pc.random.seed(12345)
        result2 = pc.random.compare_any(1000, 0.05)

        assert result1 == result2

    def test_compare_any_vs_unfused(self):
        """Verify compare_any matches unfused pecos behavior."""
        seed_val = 999
        n = 1000
        threshold = 0.01

        # Fused pecos version
        pc.random.seed(seed_val)
        pecos_result = pc.random.compare_any(n, threshold)

        # Unfused pecos version
        pc.random.seed(seed_val)
        unfused_result = any(pc.random.random(1)[0] < threshold for _ in range(n))

        # Results should match with same seed
        assert pecos_result == unfused_result

    def test_compare_any_statistical_properties(self):
        """Test statistical properties match expected probabilities."""
        # For p=0.5, n=1000, P(at least one) ≈ 1.0
        pc.random.seed(777)
        assert pc.random.compare_any(1000, 0.5) is True

        # For p=0.001, n=10, P(at least one) = 1 - (1-0.001)^10 ≈ 0.01
        # Run 1000 trials, expect ~10 hits
        pc.random.seed(666)
        hits = sum(pc.random.compare_any(10, 0.001) for _ in range(1000))
        # Allow wide tolerance for low probability events
        assert 0 <= hits <= 30, f"Expected ~10 hits, got {hits}"


class TestCompareIndicesCorrectness:
    """Test compare_indices correctness against numpy."""

    def test_compare_indices_all(self):
        """With threshold=1.0, should return all indices."""
        pc.random.seed(42)
        result = pc.random.compare_indices(10, 1.0)
        assert result == list(range(10))

    def test_compare_indices_none(self):
        """With threshold=0.0, should return empty."""
        pc.random.seed(42)
        result = pc.random.compare_indices(10, 0.0)
        assert result == []

    def test_compare_indices_reproducibility(self):
        """Same seed should produce same indices."""
        pc.random.seed(54321)
        result1 = pc.random.compare_indices(100, 0.1)

        pc.random.seed(54321)
        result2 = pc.random.compare_indices(100, 0.1)

        assert result1 == result2

    def test_compare_indices_vs_unfused(self):
        """Verify compare_indices matches unfused pecos behavior."""
        seed_val = 888
        n = 100
        threshold = 0.1

        # Fused pecos version
        pc.random.seed(seed_val)
        pecos_result = pc.random.compare_indices(n, threshold)

        # Unfused pecos version
        pc.random.seed(seed_val)
        unfused_result = [i for i in range(n) if pc.random.random(1)[0] < threshold]

        # Results should match with same seed
        assert pecos_result == unfused_result

    def test_compare_indices_statistical_properties(self):
        """Test statistical properties match expected probabilities."""
        # For p=0.5, n=10000, expect ~5000 indices
        pc.random.seed(555)
        result = pc.random.compare_indices(10000, 0.5)
        count = len(result)
        expected = 5000
        tolerance = 200  # ±200 for statistical variation

        assert (
            expected - tolerance < count < expected + tolerance
        ), f"Expected ~{expected} indices (±{tolerance}), got {count}"

        # Verify all indices are valid and in ascending order
        assert all(0 <= idx < 10000 for idx in result)
        assert result == sorted(result)


class TestCompareConsistency:
    """Test consistency between compare_any and compare_indices."""

    def test_consistency_with_seed(self):
        """If compare_indices returns non-empty, compare_any should be True."""
        for seed_val in [111, 222, 333, 444, 555]:
            pc.random.seed(seed_val)
            indices = pc.random.compare_indices(100, 0.1)

            pc.random.seed(seed_val)
            has_any = pc.random.compare_any(100, 0.1)

            if len(indices) > 0:
                assert (
                    has_any
                ), f"Seed {seed_val}: indices non-empty but compare_any is False"
            else:
                assert (
                    not has_any
                ), f"Seed {seed_val}: indices empty but compare_any is True"


class TestComparePerformance:
    """Benchmark fused operations against numpy unfused versions."""

    @pytest.mark.performance
    def test_compare_any_performance(self):
        """Benchmark compare_any vs numpy unfused version."""
        n = 100000
        threshold = 0.01
        iterations = 1000

        # Warmup
        for _ in range(10):
            pc.random.seed(42)
            pc.random.compare_any(n, threshold)
            np.random.seed(42)
            np.any(np.random.random(n) < threshold)

        # Benchmark fused version
        pc.random.seed(123)
        start = time.perf_counter()
        for _ in range(iterations):
            pc.random.compare_any(n, threshold)
        pecos_time = time.perf_counter() - start

        # Benchmark unfused numpy version
        np.random.seed(123)
        start = time.perf_counter()
        for _ in range(iterations):
            np.any(np.random.random(n) < threshold)
        numpy_time = time.perf_counter() - start

        speedup = numpy_time / pecos_time
        print(f"\ncompare_any speedup: {speedup:.2f}x")
        print(
            f"  Fused:   {pecos_time*1000:.2f}ms ({pecos_time/iterations*1000:.3f}ms/iter)"
        )
        print(
            f"  Unfused: {numpy_time*1000:.2f}ms ({numpy_time/iterations*1000:.3f}ms/iter)"
        )

        # Should be at least 1.5x faster (conservative target, expect 2-3x)
        assert speedup > 1.5, f"Expected >1.5x speedup, got {speedup:.2f}x"

    @pytest.mark.performance
    def test_compare_indices_performance(self):
        """Benchmark compare_indices vs numpy unfused version."""
        n = 100000
        threshold = 0.01
        iterations = 100  # Fewer iterations since this generates more data

        # Warmup
        for _ in range(5):
            pc.random.seed(42)
            pc.random.compare_indices(n, threshold)
            np.random.seed(42)
            rand_nums = np.random.random(n) < threshold
            [i for i, r in enumerate(rand_nums) if r]

        # Benchmark fused version
        pc.random.seed(456)
        start = time.perf_counter()
        for _ in range(iterations):
            pc.random.compare_indices(n, threshold)
        pecos_time = time.perf_counter() - start

        # Benchmark unfused numpy version
        np.random.seed(456)
        start = time.perf_counter()
        for _ in range(iterations):
            rand_nums = np.random.random(n) < threshold
            [i for i, r in enumerate(rand_nums) if r]
        numpy_time = time.perf_counter() - start

        speedup = numpy_time / pecos_time
        print(f"\ncompare_indices speedup: {speedup:.2f}x")
        print(
            f"  Fused:   {pecos_time*1000:.2f}ms ({pecos_time/iterations*1000:.3f}ms/iter)"
        )
        print(
            f"  Unfused: {numpy_time*1000:.2f}ms ({numpy_time/iterations*1000:.3f}ms/iter)"
        )

        # Should be at least 1.3x faster (conservative target, expect 1.5-2x)
        assert speedup > 1.3, f"Expected >1.3x speedup, got {speedup:.2f}x"


class TestErrorModelUsage:
    """Test realistic error model usage patterns."""

    def test_error_model_pattern_compare_any(self):
        """Test pattern: if compare_any(n, p) then generate full error mask."""
        n_qubits = 1000
        error_rate = 0.01
        n_trials = 1000

        pc.random.seed(777)

        # Count trials with errors using fused operation
        trials_with_errors = 0
        for _ in range(n_trials):
            if pc.random.compare_any(n_qubits, error_rate):
                trials_with_errors += 1

        # Expected probability: P(at least one error) = 1 - (1-p)^n
        expected_prob = 1 - (1 - error_rate) ** n_qubits
        expected_count = n_trials * expected_prob
        tolerance = 3 * np.sqrt(
            n_trials * expected_prob * (1 - expected_prob)
        )  # 3-sigma

        assert (
            abs(trials_with_errors - expected_count) < tolerance
        ), f"Expected ~{expected_count:.0f} trials with errors (±{tolerance:.0f}), got {trials_with_errors}"

    def test_error_model_pattern_compare_indices(self):
        """Test pattern: get error indices and apply errors."""
        n_qubits = 1000
        error_rate = 0.01

        pc.random.seed(888)
        error_indices = pc.random.compare_indices(n_qubits, error_rate)

        # All indices should be valid
        assert all(0 <= idx < n_qubits for idx in error_indices)

        # Expected number of errors: n * p
        expected_count = n_qubits * error_rate
        tolerance = 3 * np.sqrt(n_qubits * error_rate * (1 - error_rate))

        assert (
            abs(len(error_indices) - expected_count) < tolerance
        ), f"Expected ~{expected_count:.0f} errors (±{tolerance:.0f}), got {len(error_indices)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
