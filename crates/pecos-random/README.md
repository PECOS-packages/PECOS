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
- Bulk generation from 4 parallel streams per generator
- Multiple algorithm options for different use cases

## Acknowledgements

This crate implements algorithms designed by:
- [Xoshiro256++](https://prng.di.unimi.it/) by David Blackman and Sebastiano Vigna
- [PCG](https://www.pcg-random.org/) by Melissa O'Neill
- `RapidRng` (behind `PecosRng`): the wyrand step from [wyhash](https://github.com/wangyi-fudan/wyhash) by Wang Yi, with constants from [rapidhash](https://github.com/Nicoshev/rapidhash) by Nicolas De Carli, ported from [rapidrand](https://github.com/hoxxep/rapidrand) by Liam Gray (used under Apache-2.0)
- SplitMix64 (seed scrambling) by Guy L. Steele Jr., Doug Lea, and Christine H. Flood

**Papers:**
- Blackman, D. & Vigna, S. (2021). "Scrambled Linear Pseudorandom Number Generators." ACM Transactions on Mathematical Software, 47(4), 1-32. [arXiv:1805.01407](https://arxiv.org/abs/1805.01407)
- Steele, G. L., Lea, D. & Flood, C. H. (2014). "Fast Splittable Pseudorandom Number Generators." OOPSLA 2014. [doi:10.1145/2660193.2660195](https://doi.org/10.1145/2660193.2660195)
- O'Neill, M. E. (2014). "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation." [HMC-CS-2014-0905](https://www.cs.hmc.edu/tr/hmc-cs-2014-0905.pdf)
