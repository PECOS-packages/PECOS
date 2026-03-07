"""
Tests for random number seeding and reproducibility.

Ensures that pecos_rslib.num.random.seed() provides reproducibility
compatible with numpy.random.seed().
"""

import numpy as np
import pytest

from pecos_rslib import array_equal, random as pecos_random


class TestSeedReproducibility:
    """Test that seeding produces reproducible sequences."""

    def test_seed_random_reproducibility(self) -> None:
        """Test that same seed produces same random() sequence."""
        pecos_random.seed(42)
        values1 = pecos_random.random(10)

        pecos_random.seed(42)
        values2 = pecos_random.random(10)

        np.testing.assert_array_equal(values1, values2)

    def test_seed_randint_reproducibility(self) -> None:
        """Test that same seed produces same randint() sequence."""
        pecos_random.seed(123)
        values1 = pecos_random.randint(0, 100, 10)

        pecos_random.seed(123)
        values2 = pecos_random.randint(0, 100, 10)

        np.testing.assert_array_equal(values1, values2)

    def test_seed_choice_reproducibility(self) -> None:
        """Test that same seed produces same choice() sequence."""
        items = [1, 2, 3, 4, 5]

        pecos_random.seed(456)
        samples1 = pecos_random.choice(items, 10)

        pecos_random.seed(456)
        samples2 = pecos_random.choice(items, 10)

        assert samples1 == samples2

    def test_different_seeds_different_sequences(self) -> None:
        """Test that different seeds produce different sequences."""
        pecos_random.seed(42)
        values1 = pecos_random.random(100)

        pecos_random.seed(43)
        values2 = pecos_random.random(100)

        # With 100 random floats, sequences should be different
        assert not array_equal(values1, values2)

    def test_seed_advances_state(self) -> None:
        """Test that RNG state advances between calls."""
        pecos_random.seed(789)
        val1 = pecos_random.random(1)
        val2 = pecos_random.random(1)

        # These should be different (state advances)
        assert val1[0] != val2[0]

        # Re-seed and verify we get the same first value
        pecos_random.seed(789)
        val3 = pecos_random.random(1)
        np.testing.assert_array_equal(val1, val3)


class TestSeedIntegration:
    """Test seeding with multiple functions."""

    def test_seed_affects_all_functions(self) -> None:
        """Test that seed() affects random(), randint(), and choice()."""
        # Set seed and generate values
        pecos_random.seed(999)
        r1 = pecos_random.random(5)
        i1 = pecos_random.randint(0, 10, 5)
        c1 = pecos_random.choice([1, 2, 3], 5)

        # Re-seed and generate again
        pecos_random.seed(999)
        r2 = pecos_random.random(5)
        i2 = pecos_random.randint(0, 10, 5)
        c2 = pecos_random.choice([1, 2, 3], 5)

        # All should be identical
        np.testing.assert_array_equal(r1, r2)
        np.testing.assert_array_equal(i1, i2)
        assert c1 == c2

    def test_seed_sequence_order_matters(self) -> None:
        """Test that the order of operations affects the sequence."""
        # Sequence 1: random then randint
        pecos_random.seed(111)
        r1 = pecos_random.random(3)
        i1 = pecos_random.randint(0, 10, 3)

        # Sequence 2: randint then random
        pecos_random.seed(111)
        i2 = pecos_random.randint(0, 10, 3)
        r2 = pecos_random.random(3)

        # r1 should match r2 position-wise, i1 should match i2
        # This confirms state advances properly
        assert not array_equal(r1, r2)  # Different because order changed
        assert not array_equal(i1, i2)


class TestSeedLargeScale:
    """Test seeding with large datasets."""

    def test_seed_large_array_reproducibility(self) -> None:
        """Test reproducibility with large arrays."""
        size = 100_000

        pecos_random.seed(777)
        large1 = pecos_random.random(size)

        pecos_random.seed(777)
        large2 = pecos_random.random(size)

        np.testing.assert_array_equal(large1, large2)

    def test_seed_multiple_large_generations(self) -> None:
        """Test that state persists correctly across multiple large generations."""
        pecos_random.seed(888)

        # Generate multiple arrays
        arrays1 = [pecos_random.random(1000) for _ in range(10)]

        pecos_random.seed(888)
        arrays2 = [pecos_random.random(1000) for _ in range(10)]

        # All should match
        for a1, a2 in zip(arrays1, arrays2, strict=False):
            np.testing.assert_array_equal(a1, a2)


class TestSeedNumericRange:
    """Test seeding with different seed values."""

    def test_seed_zero(self) -> None:
        """Test that seed(0) works."""
        pecos_random.seed(0)
        values1 = pecos_random.random(10)

        pecos_random.seed(0)
        values2 = pecos_random.random(10)

        np.testing.assert_array_equal(values1, values2)

    def test_seed_large_value(self) -> None:
        """Test that large seed values work."""
        large_seed = 2**63 - 1  # Max u64

        pecos_random.seed(large_seed)
        values1 = pecos_random.random(10)

        pecos_random.seed(large_seed)
        values2 = pecos_random.random(10)

        np.testing.assert_array_equal(values1, values2)

    def test_different_small_seeds(self) -> None:
        """Test that consecutive small seeds produce different sequences."""
        pecos_random.seed(1)
        seq1 = pecos_random.random(10)

        pecos_random.seed(2)
        seq2 = pecos_random.random(10)

        assert not array_equal(seq1, seq2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
