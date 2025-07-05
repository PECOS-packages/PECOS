import unittest
import sys
from pecos.engines.cvm.rng_model import RNGModel

import random

class TestRNG(unittest.TestCase):

    def test_set_seed(self):
        rng = RNGModel(shot_id = 0)
        seed = 42
        rng.set_seed(seed)
        self.assertEqual(rng.seed, seed)

    def test_random_number(self):
        rng = RNGModel(shot_id = 0)
        random = rng.rng_random()
        self.assertTrue(isinstance(random, int))

    def test_bounded_random(self):
        rng = RNGModel(shot_id = 0)
        rng.set_seed(42)
        bound = 16
        rng.set_bound(bound)
        self.assertEqual(rng.current_bound, bound)

        random_number = rng.rng_random()
        self.assertTrue(random_number < bound)
    
    def test_set_idx(self):
        rng = RNGModel(shot_id = 0)
        rng.set_seed(42)
        idx = 4
        rng.set_index(idx)
        self.assertEqual(rng.count, idx)

    def test_multiple_bounded_rand(self):
        rng = RNGModel(shot_id = 0)
        rng.set_seed(42)

        for _ in range(100):
            random_bound = random.randint(1, 2**32-1)
            rng.set_bound(random_bound)
            random_number = rng.rng_random()
            self.assertTrue(0 <= random_number < random_bound)

   
if __name__ == '__main__':
    unittest.main()


