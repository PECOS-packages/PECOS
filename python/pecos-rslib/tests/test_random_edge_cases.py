"""
Additional edge case tests for pecos_rslib.num.random.

Tests for seeding, reproducibility, edge cases, and integration patterns.
"""

import numpy as np
import pytest

import pecos as pc


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_random_size_zero(self):
        """Test that size=0 returns empty array."""
        result = pc.random.random(0)
        assert len(result) == 0
        assert isinstance(result, pc.Array)

    def test_random_size_one(self):
        """Test that size=1 returns single element array."""
        result = pc.random.random(1)
        assert len(result) == 1
        assert isinstance(result, pc.Array)
        assert 0.0 <= result[0] < 1.0

    def test_random_large_array(self):
        """Test that large arrays work correctly."""
        size = 1_000_000
        result = pc.random.random(size)
        assert len(result) == size
        # Statistical test on large sample
        mean = np.mean(result)
        assert abs(mean - 0.5) < 0.005  # Tighter bound for large sample

    def test_randint_size_zero(self):
        """Test that randint with size=0 returns empty array."""
        result = pc.random.randint(0, 10, 0)
        assert len(result) == 0
        assert isinstance(result, pc.Array)

    def test_randint_single_value_range(self):
        """Test randint with high=low+1 (only one possible value)."""
        result = pc.random.randint(5, 6, 100)
        assert np.all(result == 5)

    def test_randint_large_range(self):
        """Test randint with very large range."""
        result = pc.random.randint(-1_000_000, 1_000_000, 1000)
        assert len(result) == 1000
        assert np.all(result >= -1_000_000)
        assert np.all(result < 1_000_000)

    def test_choice_size_zero(self):
        """Test that choice with size=0 returns empty list."""
        items = [1, 2, 3, 4, 5]
        result = pc.random.choice(items, 0)
        assert len(result) == 0

    def test_choice_single_element_array(self):
        """Test choice from single-element array."""
        items = [42]
        result = pc.random.choice(items, 10)
        assert len(result) == 10
        assert all(x == 42 for x in result)

    def test_choice_all_elements_no_replacement(self):
        """Test sampling all elements without replacement."""
        items = [1, 2, 3, 4, 5]
        result = pc.random.choice(items, 5, replace=False)
        assert len(result) == 5
        assert set(result) == set(items)


class TestMultiThreading:
    """Test thread safety of random number generation."""

    def test_concurrent_random_calls(self):
        """Test that concurrent calls don't interfere."""
        import concurrent.futures

        def generate_random(n):
            return pc.random.random(n)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(generate_random, 1000) for _ in range(10)]
            results = [f.result() for f in futures]

        # Each result should be valid
        for result in results:
            assert len(result) == 1000
            assert np.all(result >= 0.0)
            assert np.all(result < 1.0)

    def test_concurrent_randint_calls(self):
        """Test that concurrent randint calls work correctly."""
        import concurrent.futures

        def generate_randint(n):
            return pc.random.randint(0, 100, n)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(generate_randint, 1000) for _ in range(10)]
            results = [f.result() for f in futures]

        # Each result should be valid
        for result in results:
            assert len(result) == 1000
            assert np.all(result >= 0)
            assert np.all(result < 100)


class TestQuantumPecosPatterns:
    """Test common patterns used in quantum-pecos."""

    def test_error_generation_pattern(self):
        """Test typical error generation pattern from quantum-pecos."""
        # Simulate: errors = np.random.random(n_qubits) < error_rate
        n_qubits = 1000
        error_rate = 0.01

        random_vals = pc.random.random(n_qubits)
        errors = random_vals < error_rate

        # Should have approximately error_rate fraction of True values
        error_count = np.sum(errors)
        expected = n_qubits * error_rate
        # Allow 3-sigma deviation: sqrt(n*p*(1-p))
        sigma = np.sqrt(n_qubits * error_rate * (1 - error_rate))
        assert abs(error_count - expected) < 3 * sigma

    def test_qubit_selection_pattern(self):
        """Test random qubit selection pattern."""
        # Simulate: selected_qubits = np.random.choice(qubit_indices, n_select)
        all_qubits = list(range(100))
        n_select = 10

        selected = pc.random.choice(all_qubits, n_select, replace=False)

        assert len(selected) == n_select
        assert len(set(selected)) == n_select  # All unique
        assert all(q in all_qubits for q in selected)

    def test_measurement_outcome_pattern(self):
        """Test random measurement outcome generation."""
        # Simulate: outcomes = np.random.randint(0, 2, n_measurements)
        n_measurements = 1000

        outcomes = pc.random.randint(0, 2, n_measurements)

        assert len(outcomes) == n_measurements
        # Convert to numpy for logical operations
        outcomes_np = np.asarray(outcomes)
        assert np.all((outcomes_np == 0) | (outcomes_np == 1))

        # Should be approximately 50/50
        ones_count = np.sum(outcomes)
        assert 400 < ones_count < 600  # Loose bound for randomness

    def test_syndrome_generation_pattern(self):
        """Test syndrome generation with multiple random calls."""
        # Simulate complex pattern with multiple RNG calls
        n_qubits = 100
        n_rounds = 10

        # Generate errors for each round
        for _ in range(n_rounds):
            error_mask = pc.random.random(n_qubits) < 0.01
            assert len(error_mask) == n_qubits
            # PECOS comparison returns numeric (0/1) while NumPy returns bool
            # Both are valid - just check the values are binary
            error_mask_np = np.asarray(error_mask)
            assert np.all((error_mask_np == 0) | (error_mask_np == 1))

    def test_batch_random_integers(self):
        """Test generating batches of random integers (common in sampling)."""
        # Pattern: multiple independent random integer samples
        batch_size = 50
        n_samples = 100

        results = []
        for _ in range(batch_size):
            sample = pc.random.randint(0, 1000, n_samples)
            results.append(sample)

        # Verify all batches are valid
        for batch in results:
            assert len(batch) == n_samples
            assert np.all(batch >= 0)
            assert np.all(batch < 1000)


class TestNumpyCompatibilityExtended:
    """Extended numpy compatibility tests."""

    def test_random_dtype_compatibility(self):
        """Verify dtype matches numpy exactly."""
        pecos_result = pc.random.random(100)
        numpy_result = np.random.random(100)

        assert pecos_result.dtype == numpy_result.dtype
        assert pecos_result.dtype == np.float64

    def test_randint_dtype_compatibility(self):
        """Verify randint dtype matches numpy."""
        pecos_result = pc.random.randint(0, 100, 100)
        numpy_result = np.random.randint(0, 100, 100)

        assert pecos_result.dtype == numpy_result.dtype

    def test_random_array_flags(self):
        """Verify array flags match numpy."""
        result = pc.random.random(100)

        # Convert to numpy to check flags
        result_np = np.asarray(result)

        # Should be C-contiguous like numpy
        assert result_np.flags["C_CONTIGUOUS"]
        # Note: OWNDATA will be True for the numpy view, WRITEABLE should also be True
        assert result_np.flags["WRITEABLE"]

    def test_choice_preserves_types(self):
        """Test that choice preserves element types."""
        # String elements
        string_items = ["a", "b", "c", "d"]
        string_result = pc.random.choice(string_items, 10)
        assert all(isinstance(x, str) for x in string_result)

        # Integer elements
        int_items = [1, 2, 3, 4, 5]
        int_result = pc.random.choice(int_items, 10)
        assert all(isinstance(x, int) for x in int_result)

        # Float elements
        float_items = [1.5, 2.5, 3.5, 4.5]
        float_result = pc.random.choice(float_items, 10)
        assert all(isinstance(x, (float, np.floating)) for x in float_result)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
