"""Testing module for the RNG Model."""

import random
import unittest

from pecos.engines.cvm.rng_model import RNGModel


class TestRNG(unittest.TestCase):
    """This class is responsible for testing RNG platform calls."""

    def test_set_seed(self) -> None:
        """Verifies that a seed is set properly for our RNG model."""
        rng = RNGModel(shot_id=0)
        seed = 42
        rng.set_seed(seed)
        assert rng.seed == seed

    def test_random_number(self) -> None:
        """Verifies that the random number generated is an int type."""
        rng = RNGModel(shot_id=0)
        random = rng.rng_random()
        assert isinstance(random, int)

    def test_bounded_random(self) -> None:
        """Verifies that a single generated random number is within bounds."""
        rng = RNGModel(shot_id=0)
        rng.set_seed(42)
        bound = 16
        rng.set_bound(bound)
        assert rng.current_bound == bound

        random_number = rng.rng_random()
        assert 0 <= random_number < bound

    def test_set_idx(self) -> None:
        """Verifies that the idx is set properly for our model."""
        rng = RNGModel(shot_id=0)
        rng.set_seed(42)
        idx = 4
        rng.set_index(idx)
        assert rng.count == idx

    def test_multiple_bounded_rand(self) -> None:
        """For several randomly generated number, with a random bound, verifies that its appropriate."""
        rng = RNGModel(shot_id=0)
        rng.set_seed(42)

        for _ in range(100):
            random_bound = random.randint(1, 2**32 - 1)  # noqa: S311
            rng.set_bound(random_bound)
            random_number = rng.rng_random()
            assert 0 <= random_number < random_bound


if __name__ == "__main__":
    unittest.main()
