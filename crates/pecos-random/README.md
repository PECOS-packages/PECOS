# pecos-random

Random number generation for PECOS.

## Purpose

Provides deterministic, reproducible random number generation with support for parallel execution (each thread gets independent streams from the same seed).

## Key Types

- `PecosRng` - Fast RNG for general use (rapidhash-based)
- `PecosQualityRng` - High-quality RNG (SIMD Xoshiro256++)
- `PCG64Fast` - PCG-based RNG
- `RngManageable` - Trait for RNG-equipped simulators

## Features

- Deterministic from seed
- Parallel-safe stream generation
- SIMD-optimized bulk generation (4x parallel Xoshiro256++)
- Multiple algorithm options for different use cases

## Acknowledgements

This crate implements algorithms designed by:
- [Xoshiro256++](https://prng.di.unimi.it/) by David Blackman and Sebastiano Vigna
- [PCG](https://www.pcg-random.org/) by Melissa O'Neill

**Papers:**
- Blackman, D. & Vigna, S. (2021). "Scrambled Linear Pseudorandom Number Generators." ACM Transactions on Mathematical Software, 47(4), 1-32. [arXiv:1805.01407](https://arxiv.org/abs/1805.01407)
- O'Neill, M. E. (2014). "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation." [HMC-CS-2014-0905](https://www.cs.hmc.edu/tr/hmc-cs-2014-0905.pdf)
