// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! RNG benchmarks covering typical use cases in quantum simulation.
//!
//! This module benchmarks RNG performance for patterns that commonly appear in:
//! - Measurement sampling (bool generation, probability checks)
//! - Noise models (probability checks, Pauli selection)
//! - State vector simulation (Born rule probability checks)
//! - Stabilizer simulation (random measurement outcomes)
//!
//! Benchmark groups:
//! - **Scalar Operations**: Individual scalar generation (u64, f64, bool, range)
//! - **Bulk Operations**: Filling arrays efficiently
//! - **Real-World Patterns**: Composite patterns as they appear in actual code
//!
//! All benchmarks compare the same 7 RNGs:
//! - `PecosRng` (SIMD Xoshiro256++ with buffering)
//! - `SmallRng` (rand's default)
//! - Xoshiro256++, Xoroshiro128++, Xoshiro512++ (xoshiro family)
//! - `RapidRng` (successor to wyhash)
//! - `PCG64Fast` (fast PCG variant)

use criterion::{BenchmarkId, Criterion, Throughput, measurement::Measurement};
use pecos::prelude::{PCG64Fast, PecosQualityRng, PecosRng};
use pecos_rng::{PecosScalarRng, Rng, RngCore, SeedableRng};
use rand::rngs::SmallRng;
use rand_xoshiro::{Xoroshiro128PlusPlus, Xoshiro256PlusPlus, Xoshiro512PlusPlus};
use rapidhash::rng::RapidRng;
use std::hint::black_box;
use wide::u64x4;

// Helper to convert u64 to f64 in [0, 1) - same as rand's method
#[allow(clippy::cast_precision_loss)]
#[inline]
fn u64_to_f64(x: u64) -> f64 {
    (x >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

// Helper to convert u64 to bool using high bit
#[inline]
fn u64_to_bool(x: u64) -> bool {
    (x >> 63) != 0
}

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    bench_scalar_operations(c);
    bench_bulk_operations(c);
    bench_real_world_patterns(c);
    bench_fused_noise_sampling(c);
    bench_batched_probability(c);
    bench_probability_optimizations(c);
}

// ============================================================================
// Scalar Operations - Individual value generation
// ============================================================================

const NUM_SCALAR_CALLS: usize = 10_000;

/// Benchmark scalar operations: the building blocks of RNG usage.
fn bench_scalar_operations<M: Measurement>(c: &mut Criterion<M>) {
    bench_scalar_u64(c);
    bench_scalar_f64(c);
    bench_scalar_bool(c);
    bench_scalar_range(c);
}

fn bench_scalar_u64<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("RNG Scalar/u64");
    group.throughput(Throughput::Elements(NUM_SCALAR_CALLS as u64));

    group.bench_function("PecosQualityRng", |b| {
        let mut rng = PecosQualityRng::seed_from_u64(42);
        b.iter(|| {
            let mut sum = 0u64;
            for _ in 0..NUM_SCALAR_CALLS {
                sum = sum.wrapping_add(rng.next_u64());
            }
            black_box(sum)
        });
    });

    group.bench_function("PecosRng", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| {
            let mut sum = 0u64;
            for _ in 0..NUM_SCALAR_CALLS {
                sum = sum.wrapping_add(rng.next_u64());
            }
            black_box(sum)
        });
    });

    group.bench_function("PecosScalarRng", |b| {
        let mut rng = PecosScalarRng::seed_from_u64(42);
        b.iter(|| {
            let mut sum = 0u64;
            for _ in 0..NUM_SCALAR_CALLS {
                sum = sum.wrapping_add(rng.next_u64());
            }
            black_box(sum)
        });
    });

    group.bench_function("SmallRng", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        b.iter(|| {
            let mut sum = 0u64;
            for _ in 0..NUM_SCALAR_CALLS {
                sum = sum.wrapping_add(rng.next_u64());
            }
            black_box(sum)
        });
    });

    group.bench_function("Xoshiro256++", |b| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        b.iter(|| {
            let mut sum = 0u64;
            for _ in 0..NUM_SCALAR_CALLS {
                sum = sum.wrapping_add(rng.next_u64());
            }
            black_box(sum)
        });
    });

    group.bench_function("Xoroshiro128++", |b| {
        let mut rng = Xoroshiro128PlusPlus::seed_from_u64(42);
        b.iter(|| {
            let mut sum = 0u64;
            for _ in 0..NUM_SCALAR_CALLS {
                sum = sum.wrapping_add(rng.next_u64());
            }
            black_box(sum)
        });
    });

    group.bench_function("Xoshiro512++", |b| {
        let mut rng = Xoshiro512PlusPlus::seed_from_u64(42);
        b.iter(|| {
            let mut sum = 0u64;
            for _ in 0..NUM_SCALAR_CALLS {
                sum = sum.wrapping_add(rng.next_u64());
            }
            black_box(sum)
        });
    });

    group.bench_function("RapidRng", |b| {
        let mut rng = RapidRng::new(42);
        b.iter(|| {
            let mut sum = 0u64;
            for _ in 0..NUM_SCALAR_CALLS {
                sum = sum.wrapping_add(rng.next_u64());
            }
            black_box(sum)
        });
    });

    group.bench_function("PCG64Fast", |b| {
        let mut rng = PCG64Fast::seed_from_u64(42);
        b.iter(|| {
            let mut sum = 0u64;
            for _ in 0..NUM_SCALAR_CALLS {
                sum = sum.wrapping_add(rng.next_u64());
            }
            black_box(sum)
        });
    });

    group.finish();
}

fn bench_scalar_f64<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("RNG Scalar/f64");
    group.throughput(Throughput::Elements(NUM_SCALAR_CALLS as u64));

    group.bench_function("PecosQualityRng", |b| {
        let mut rng = PecosQualityRng::seed_from_u64(42);
        b.iter(|| {
            let mut sum = 0.0f64;
            for _ in 0..NUM_SCALAR_CALLS {
                sum += rng.random::<f64>();
            }
            black_box(sum)
        });
    });

    group.bench_function("PecosRng", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| {
            let mut sum = 0.0f64;
            for _ in 0..NUM_SCALAR_CALLS {
                sum += rng.next_f64();
            }
            black_box(sum)
        });
    });

    group.bench_function("PecosScalarRng", |b| {
        let mut rng = PecosScalarRng::seed_from_u64(42);
        b.iter(|| {
            let mut sum = 0.0f64;
            for _ in 0..NUM_SCALAR_CALLS {
                sum += rng.next_f64();
            }
            black_box(sum)
        });
    });

    group.bench_function("SmallRng", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        b.iter(|| {
            let mut sum = 0.0f64;
            for _ in 0..NUM_SCALAR_CALLS {
                sum += rng.random::<f64>();
            }
            black_box(sum)
        });
    });

    group.bench_function("Xoshiro256++", |b| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        b.iter(|| {
            let mut sum = 0.0f64;
            for _ in 0..NUM_SCALAR_CALLS {
                sum += rng.random::<f64>();
            }
            black_box(sum)
        });
    });

    group.bench_function("Xoroshiro128++", |b| {
        let mut rng = Xoroshiro128PlusPlus::seed_from_u64(42);
        b.iter(|| {
            let mut sum = 0.0f64;
            for _ in 0..NUM_SCALAR_CALLS {
                sum += rng.random::<f64>();
            }
            black_box(sum)
        });
    });

    group.bench_function("Xoshiro512++", |b| {
        let mut rng = Xoshiro512PlusPlus::seed_from_u64(42);
        b.iter(|| {
            let mut sum = 0.0f64;
            for _ in 0..NUM_SCALAR_CALLS {
                sum += rng.random::<f64>();
            }
            black_box(sum)
        });
    });

    group.bench_function("RapidRng", |b| {
        let mut rng = RapidRng::new(42);
        b.iter(|| {
            let mut sum = 0.0f64;
            for _ in 0..NUM_SCALAR_CALLS {
                sum += u64_to_f64(rng.next_u64());
            }
            black_box(sum)
        });
    });

    group.bench_function("PCG64Fast", |b| {
        let mut rng = PCG64Fast::seed_from_u64(42);
        b.iter(|| {
            let mut sum = 0.0f64;
            for _ in 0..NUM_SCALAR_CALLS {
                sum += u64_to_f64(rng.next_u64());
            }
            black_box(sum)
        });
    });

    group.finish();
}

#[allow(clippy::too_many_lines)]
fn bench_scalar_bool<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("RNG Scalar/bool");
    group.throughput(Throughput::Elements(NUM_SCALAR_CALLS as u64));

    group.bench_function("PecosQualityRng", |b| {
        let mut rng = PecosQualityRng::seed_from_u64(42);
        b.iter(|| {
            let mut count = 0u32;
            for _ in 0..NUM_SCALAR_CALLS {
                if rng.random::<bool>() {
                    count += 1;
                }
            }
            black_box(count)
        });
    });

    // Optimized: bit-packed bool extraction (64 bools per u64)
    group.bench_function("PecosQualityRng (fast)", |b| {
        let mut rng = PecosQualityRng::seed_from_u64(42);
        b.iter(|| {
            let mut count = 0u32;
            for _ in 0..NUM_SCALAR_CALLS {
                if rng.next_bool_fast() {
                    count += 1;
                }
            }
            black_box(count)
        });
    });

    group.bench_function("PecosRng", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| {
            let mut count = 0u32;
            for _ in 0..NUM_SCALAR_CALLS {
                if rng.random::<bool>() {
                    count += 1;
                }
            }
            black_box(count)
        });
    });

    group.bench_function("PecosRng (fast)", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| {
            let mut count = 0u32;
            for _ in 0..NUM_SCALAR_CALLS {
                if rng.next_bool_fast() {
                    count += 1;
                }
            }
            black_box(count)
        });
    });

    group.bench_function("SmallRng", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        b.iter(|| {
            let mut count = 0u32;
            for _ in 0..NUM_SCALAR_CALLS {
                if rng.random::<bool>() {
                    count += 1;
                }
            }
            black_box(count)
        });
    });

    group.bench_function("Xoshiro256++", |b| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        b.iter(|| {
            let mut count = 0u32;
            for _ in 0..NUM_SCALAR_CALLS {
                if rng.random::<bool>() {
                    count += 1;
                }
            }
            black_box(count)
        });
    });

    group.bench_function("Xoroshiro128++", |b| {
        let mut rng = Xoroshiro128PlusPlus::seed_from_u64(42);
        b.iter(|| {
            let mut count = 0u32;
            for _ in 0..NUM_SCALAR_CALLS {
                if rng.random::<bool>() {
                    count += 1;
                }
            }
            black_box(count)
        });
    });

    group.bench_function("Xoshiro512++", |b| {
        let mut rng = Xoshiro512PlusPlus::seed_from_u64(42);
        b.iter(|| {
            let mut count = 0u32;
            for _ in 0..NUM_SCALAR_CALLS {
                if rng.random::<bool>() {
                    count += 1;
                }
            }
            black_box(count)
        });
    });

    group.bench_function("RapidRng", |b| {
        let mut rng = RapidRng::new(42);
        b.iter(|| {
            let mut count = 0u32;
            for _ in 0..NUM_SCALAR_CALLS {
                if (rng.next_u64() >> 63) != 0 {
                    count += 1;
                }
            }
            black_box(count)
        });
    });

    group.bench_function("PCG64Fast", |b| {
        let mut rng = PCG64Fast::seed_from_u64(42);
        b.iter(|| {
            let mut count = 0u32;
            for _ in 0..NUM_SCALAR_CALLS {
                if (rng.next_u64() >> 63) != 0 {
                    count += 1;
                }
            }
            black_box(count)
        });
    });

    group.finish();
}

fn bench_scalar_range<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("RNG Scalar/range_100");
    group.throughput(Throughput::Elements(NUM_SCALAR_CALLS as u64));

    group.bench_function("PecosQualityRng", |b| {
        let mut rng = PecosQualityRng::seed_from_u64(42);
        b.iter(|| {
            let mut sum = 0usize;
            for _ in 0..NUM_SCALAR_CALLS {
                sum = sum.wrapping_add(rng.random_range(0usize..100));
            }
            black_box(sum)
        });
    });

    group.bench_function("PecosRng", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| {
            let mut sum = 0usize;
            for _ in 0..NUM_SCALAR_CALLS {
                sum = sum.wrapping_add(rng.random_range(0usize..100));
            }
            black_box(sum)
        });
    });

    group.bench_function("SmallRng", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        b.iter(|| {
            let mut sum = 0usize;
            for _ in 0..NUM_SCALAR_CALLS {
                sum = sum.wrapping_add(rng.random_range(0usize..100));
            }
            black_box(sum)
        });
    });

    group.bench_function("Xoshiro256++", |b| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        b.iter(|| {
            let mut sum = 0usize;
            for _ in 0..NUM_SCALAR_CALLS {
                sum = sum.wrapping_add(rng.random_range(0usize..100));
            }
            black_box(sum)
        });
    });

    group.bench_function("Xoroshiro128++", |b| {
        let mut rng = Xoroshiro128PlusPlus::seed_from_u64(42);
        b.iter(|| {
            let mut sum = 0usize;
            for _ in 0..NUM_SCALAR_CALLS {
                sum = sum.wrapping_add(rng.random_range(0usize..100));
            }
            black_box(sum)
        });
    });

    group.bench_function("Xoshiro512++", |b| {
        let mut rng = Xoshiro512PlusPlus::seed_from_u64(42);
        b.iter(|| {
            let mut sum = 0usize;
            for _ in 0..NUM_SCALAR_CALLS {
                sum = sum.wrapping_add(rng.random_range(0usize..100));
            }
            black_box(sum)
        });
    });

    group.bench_function("RapidRng", |b| {
        let mut rng = RapidRng::new(42);
        b.iter(|| {
            let mut sum = 0usize;
            for _ in 0..NUM_SCALAR_CALLS {
                sum = sum.wrapping_add((rng.next_u64() % 100) as usize);
            }
            black_box(sum)
        });
    });

    group.bench_function("PCG64Fast", |b| {
        let mut rng = PCG64Fast::seed_from_u64(42);
        b.iter(|| {
            let mut sum = 0usize;
            for _ in 0..NUM_SCALAR_CALLS {
                sum = sum.wrapping_add((rng.next_u64() % 100) as usize);
            }
            black_box(sum)
        });
    });

    group.finish();
}

// ============================================================================
// Bulk Operations - Filling arrays efficiently
// ============================================================================

/// Benchmark bulk fill operations used in columnar sampling.
#[allow(clippy::too_many_lines)]
fn bench_bulk_operations<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("RNG Bulk");

    // Test different sizes relevant to quantum simulation
    for &shots in &[1_000usize, 10_000, 100_000, 1_000_000] {
        let num_words = shots.div_ceil(64);
        let num_simd = num_words.div_ceil(4);

        group.throughput(Throughput::Elements(shots as u64));

        // --- fill_u64 benchmarks ---

        // PecosRng: compare loop vs bulk fill_u64
        group.bench_with_input(
            BenchmarkId::new("PecosRng/loop", shots),
            &num_words,
            |b, &n| {
                let mut rng = PecosRng::seed_from_u64(42);
                let mut data = vec![0u64; n];
                b.iter(|| {
                    for val in &mut data {
                        *val = rng.next_u64();
                    }
                    black_box(&data);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("PecosRng/fill_u64", shots),
            &num_words,
            |b, &n| {
                let mut rng = PecosRng::seed_from_u64(42);
                let mut data = vec![0u64; n];
                b.iter(|| {
                    rng.fill_u64(&mut data);
                    black_box(&data);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("SmallRng/fill_u64", shots),
            &num_words,
            |b, &n| {
                let mut rng = SmallRng::seed_from_u64(42);
                let mut data = vec![0u64; n];
                b.iter(|| {
                    for val in &mut data {
                        *val = rng.next_u64();
                    }
                    black_box(&data);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Xoshiro256++/fill_u64", shots),
            &num_words,
            |b, &n| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
                let mut data = vec![0u64; n];
                b.iter(|| {
                    for val in &mut data {
                        *val = rng.next_u64();
                    }
                    black_box(&data);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Xoroshiro128++/fill_u64", shots),
            &num_words,
            |b, &n| {
                let mut rng = Xoroshiro128PlusPlus::seed_from_u64(42);
                let mut data = vec![0u64; n];
                b.iter(|| {
                    for val in &mut data {
                        *val = rng.next_u64();
                    }
                    black_box(&data);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Xoshiro512++/fill_u64", shots),
            &num_words,
            |b, &n| {
                let mut rng = Xoshiro512PlusPlus::seed_from_u64(42);
                let mut data = vec![0u64; n];
                b.iter(|| {
                    for val in &mut data {
                        *val = rng.next_u64();
                    }
                    black_box(&data);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("RapidRng/fill_u64", shots),
            &num_words,
            |b, &n| {
                let mut rng = RapidRng::new(42);
                let mut data = vec![0u64; n];
                b.iter(|| {
                    for val in &mut data {
                        *val = rng.next_u64();
                    }
                    black_box(&data);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("PCG64Fast/fill_u64", shots),
            &num_words,
            |b, &n| {
                let mut rng = PCG64Fast::seed_from_u64(42);
                let mut data = vec![0u64; n];
                b.iter(|| {
                    for val in &mut data {
                        *val = rng.next_u64();
                    }
                    black_box(&data);
                });
            },
        );

        // --- simd_column benchmarks ---

        group.bench_with_input(
            BenchmarkId::new("PecosRng/simd_column", shots),
            &num_simd,
            |b, &n| {
                let mut rng = PecosRng::seed_from_u64(42);
                b.iter(|| {
                    let mut column = Vec::with_capacity(n);
                    for _ in 0..n {
                        column.push(rng.next_u64x4());
                    }
                    black_box(column)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("SmallRng/simd_column", shots),
            &num_simd,
            |b, &n| {
                let mut rng = SmallRng::seed_from_u64(42);
                b.iter(|| {
                    let mut column = Vec::with_capacity(n);
                    for _ in 0..n {
                        column.push(u64x4::new([
                            rng.next_u64(),
                            rng.next_u64(),
                            rng.next_u64(),
                            rng.next_u64(),
                        ]));
                    }
                    black_box(column)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Xoshiro256++/simd_column", shots),
            &num_simd,
            |b, &n| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
                b.iter(|| {
                    let mut column = Vec::with_capacity(n);
                    for _ in 0..n {
                        column.push(u64x4::new([
                            rng.next_u64(),
                            rng.next_u64(),
                            rng.next_u64(),
                            rng.next_u64(),
                        ]));
                    }
                    black_box(column)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Xoroshiro128++/simd_column", shots),
            &num_simd,
            |b, &n| {
                let mut rng = Xoroshiro128PlusPlus::seed_from_u64(42);
                b.iter(|| {
                    let mut column = Vec::with_capacity(n);
                    for _ in 0..n {
                        column.push(u64x4::new([
                            rng.next_u64(),
                            rng.next_u64(),
                            rng.next_u64(),
                            rng.next_u64(),
                        ]));
                    }
                    black_box(column)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Xoshiro512++/simd_column", shots),
            &num_simd,
            |b, &n| {
                let mut rng = Xoshiro512PlusPlus::seed_from_u64(42);
                b.iter(|| {
                    let mut column = Vec::with_capacity(n);
                    for _ in 0..n {
                        column.push(u64x4::new([
                            rng.next_u64(),
                            rng.next_u64(),
                            rng.next_u64(),
                            rng.next_u64(),
                        ]));
                    }
                    black_box(column)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("RapidRng/simd_column", shots),
            &num_simd,
            |b, &n| {
                let mut rng = RapidRng::new(42);
                b.iter(|| {
                    let mut column = Vec::with_capacity(n);
                    for _ in 0..n {
                        column.push(u64x4::new([
                            rng.next_u64(),
                            rng.next_u64(),
                            rng.next_u64(),
                            rng.next_u64(),
                        ]));
                    }
                    black_box(column)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("PCG64Fast/simd_column", shots),
            &num_simd,
            |b, &n| {
                let mut rng = PCG64Fast::seed_from_u64(42);
                b.iter(|| {
                    let mut column = Vec::with_capacity(n);
                    for _ in 0..n {
                        column.push(u64x4::new([
                            rng.next_u64(),
                            rng.next_u64(),
                            rng.next_u64(),
                            rng.next_u64(),
                        ]));
                    }
                    black_box(column)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Real-World Patterns - Composite patterns from actual code
// ============================================================================

/// Benchmark patterns that appear in real quantum simulation code.
fn bench_real_world_patterns<M: Measurement>(c: &mut Criterion<M>) {
    bench_measurement_pattern(c);
    bench_noise_model_pattern(c);
    bench_state_vec_pattern(c);
    bench_stabilizer_pattern(c);
}

/// Pattern: Measurement sampling (from `measurement_sampler.rs`)
#[allow(clippy::cast_precision_loss)] // Bounded values fit in f64 mantissa
fn bench_measurement_pattern<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("RNG Pattern: Measurement Sampling");

    let num_measurements: usize = 600;
    let shots: usize = 100_000;
    let random_fraction = 0.2;

    group.throughput(Throughput::Elements((num_measurements * shots) as u64));

    let is_random: Vec<bool> = (0..num_measurements)
        .map(|i| (i as f64 / num_measurements as f64) < random_fraction)
        .collect();

    macro_rules! bench_measurement {
        ($name:expr, $rng:expr, $random_fn:expr) => {
            group.bench_function($name, |b| {
                let mut rng = $rng;
                b.iter(|| {
                    let mut results = vec![false; num_measurements];
                    for _ in 0..shots {
                        for (m, &needs_random) in is_random.iter().enumerate() {
                            results[m] = if needs_random {
                                $random_fn(&mut rng)
                            } else if m > 0 {
                                results[m - 1]
                            } else {
                                false
                            };
                        }
                    }
                    black_box(results)
                });
            });
        };
    }

    bench_measurement!(
        "PecosQualityRng",
        PecosQualityRng::seed_from_u64(42),
        |r: &mut PecosQualityRng| r.random::<bool>()
    );
    bench_measurement!(
        "PecosQualityRng (fast)",
        PecosQualityRng::seed_from_u64(42),
        |r: &mut PecosQualityRng| r.next_bool_fast()
    );
    bench_measurement!(
        "PecosRng",
        PecosRng::seed_from_u64(42),
        |r: &mut PecosRng| r.random::<bool>()
    );
    bench_measurement!(
        "PecosRng (fast)",
        PecosRng::seed_from_u64(42),
        |r: &mut PecosRng| r.next_bool_fast()
    );
    bench_measurement!(
        "SmallRng",
        SmallRng::seed_from_u64(42),
        |r: &mut SmallRng| r.random::<bool>()
    );
    bench_measurement!(
        "Xoshiro256++",
        Xoshiro256PlusPlus::seed_from_u64(42),
        |r: &mut Xoshiro256PlusPlus| r.random::<bool>()
    );
    bench_measurement!(
        "Xoroshiro128++",
        Xoroshiro128PlusPlus::seed_from_u64(42),
        |r: &mut Xoroshiro128PlusPlus| r.random::<bool>()
    );
    bench_measurement!(
        "Xoshiro512++",
        Xoshiro512PlusPlus::seed_from_u64(42),
        |r: &mut Xoshiro512PlusPlus| r.random::<bool>()
    );
    bench_measurement!("RapidRng", RapidRng::new(42), |r: &mut RapidRng| {
        u64_to_bool(r.next_u64())
    });
    bench_measurement!(
        "PCG64Fast",
        PCG64Fast::seed_from_u64(42),
        |r: &mut PCG64Fast| u64_to_bool(r.next_u64())
    );

    group.finish();
}

/// Pattern: Noise model (from noise/depolarizing.rs, noise/general.rs)
#[allow(clippy::too_many_lines)]
fn bench_noise_model_pattern<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("RNG Pattern: Noise Model");

    let num_gates: usize = 10_000;
    let error_rate = 0.001f64;

    group.throughput(Throughput::Elements(num_gates as u64));

    macro_rules! bench_noise {
        ($name:expr, $rng:expr, $f64_fn:expr, $range_fn:expr) => {
            group.bench_function($name, |b| {
                let mut rng = $rng;
                b.iter(|| {
                    let mut error_count = 0u32;
                    let mut pauli_sum = 0u32;
                    for _ in 0..num_gates {
                        if $f64_fn(&mut rng) < error_rate {
                            error_count += 1;
                            pauli_sum += $range_fn(&mut rng);
                        }
                    }
                    black_box((error_count, pauli_sum))
                });
            });
        };
    }

    bench_noise!(
        "PecosQualityRng",
        PecosQualityRng::seed_from_u64(42),
        |r: &mut PecosQualityRng| r.random::<f64>(),
        |r: &mut PecosQualityRng| r.random_range(0u32..3)
    );

    // Optimized: fixed-point probability check (no f64 conversion)
    group.bench_function("PecosQualityRng (fixed-point)", |b| {
        let mut rng = PecosQualityRng::seed_from_u64(42);
        let threshold = PecosQualityRng::probability_threshold(error_rate);
        b.iter(|| {
            let mut error_count = 0u32;
            let mut pauli_sum = 0u32;
            for _ in 0..num_gates {
                if rng.check_probability(threshold) {
                    error_count += 1;
                    pauli_sum += (rng.next_u64() % 3) as u32;
                }
            }
            black_box((error_count, pauli_sum))
        });
    });

    bench_noise!(
        "PecosRng",
        PecosRng::seed_from_u64(42),
        |r: &mut PecosRng| r.next_f64(),
        |r: &mut PecosRng| r.random_range(0u32..3)
    );

    group.bench_function("PecosRng (fixed-point)", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        let threshold = PecosRng::probability_threshold(error_rate);
        b.iter(|| {
            let mut error_count = 0u32;
            let mut pauli_sum = 0u32;
            for _ in 0..num_gates {
                if rng.check_probability(threshold) {
                    error_count += 1;
                    pauli_sum += (rng.next_u64() % 3) as u32;
                }
            }
            black_box((error_count, pauli_sum))
        });
    });

    bench_noise!(
        "SmallRng",
        SmallRng::seed_from_u64(42),
        |r: &mut SmallRng| r.random::<f64>(),
        |r: &mut SmallRng| r.random_range(0u32..3)
    );
    bench_noise!(
        "Xoshiro256++",
        Xoshiro256PlusPlus::seed_from_u64(42),
        |r: &mut Xoshiro256PlusPlus| r.random::<f64>(),
        |r: &mut Xoshiro256PlusPlus| r.random_range(0u32..3)
    );
    bench_noise!(
        "Xoroshiro128++",
        Xoroshiro128PlusPlus::seed_from_u64(42),
        |r: &mut Xoroshiro128PlusPlus| r.random::<f64>(),
        |r: &mut Xoroshiro128PlusPlus| r.random_range(0u32..3)
    );
    bench_noise!(
        "Xoshiro512++",
        Xoshiro512PlusPlus::seed_from_u64(42),
        |r: &mut Xoshiro512PlusPlus| r.random::<f64>(),
        |r: &mut Xoshiro512PlusPlus| r.random_range(0u32..3)
    );
    bench_noise!(
        "RapidRng",
        RapidRng::new(42),
        |r: &mut RapidRng| u64_to_f64(r.next_u64()),
        |r: &mut RapidRng| (r.next_u64() % 3) as u32
    );
    bench_noise!(
        "PCG64Fast",
        PCG64Fast::seed_from_u64(42),
        |r: &mut PCG64Fast| u64_to_f64(r.next_u64()),
        |r: &mut PCG64Fast| (r.next_u64() % 3) as u32
    );

    group.finish();
}

/// Pattern: State vector Born rule (from `state_vec.rs`)
#[allow(clippy::cast_precision_loss)] // Bounded values fit in f64 mantissa
fn bench_state_vec_pattern<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("RNG Pattern: StateVec Born Rule");

    let num_measurements: usize = 10_000;

    group.throughput(Throughput::Elements(num_measurements as u64));

    let probs: Vec<f64> = (0..num_measurements)
        .map(|i| 0.3 + 0.4 * (i as f64 / num_measurements as f64))
        .collect();

    macro_rules! bench_statevec {
        ($name:expr, $rng:expr, $f64_fn:expr) => {
            group.bench_function($name, |b| {
                let mut rng = $rng;
                b.iter(|| {
                    let mut ones = 0u32;
                    for &prob in &probs {
                        if $f64_fn(&mut rng) < prob {
                            ones += 1;
                        }
                    }
                    black_box(ones)
                });
            });
        };
    }

    bench_statevec!(
        "PecosQualityRng",
        PecosQualityRng::seed_from_u64(42),
        |r: &mut PecosQualityRng| r.random::<f64>()
    );
    bench_statevec!(
        "PecosRng",
        PecosRng::seed_from_u64(42),
        |r: &mut PecosRng| r.next_f64()
    );
    bench_statevec!(
        "SmallRng",
        SmallRng::seed_from_u64(42),
        |r: &mut SmallRng| r.random::<f64>()
    );
    bench_statevec!(
        "Xoshiro256++",
        Xoshiro256PlusPlus::seed_from_u64(42),
        |r: &mut Xoshiro256PlusPlus| r.random::<f64>()
    );
    bench_statevec!(
        "Xoroshiro128++",
        Xoroshiro128PlusPlus::seed_from_u64(42),
        |r: &mut Xoroshiro128PlusPlus| r.random::<f64>()
    );
    bench_statevec!(
        "Xoshiro512++",
        Xoshiro512PlusPlus::seed_from_u64(42),
        |r: &mut Xoshiro512PlusPlus| r.random::<f64>()
    );
    bench_statevec!(
        "RapidRng",
        RapidRng::new(42),
        |r: &mut RapidRng| u64_to_f64(r.next_u64())
    );
    bench_statevec!(
        "PCG64Fast",
        PCG64Fast::seed_from_u64(42),
        |r: &mut PCG64Fast| u64_to_f64(r.next_u64())
    );

    group.finish();
}

/// Pattern: Stabilizer simulation (from `sparse_stab.rs`)
fn bench_stabilizer_pattern<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("RNG Pattern: Stabilizer Measurement");

    let num_measurements: usize = 10_000;

    group.throughput(Throughput::Elements(num_measurements as u64));

    macro_rules! bench_stabilizer {
        ($name:expr, $rng:expr, $bool_fn:expr) => {
            group.bench_function($name, |b| {
                let mut rng = $rng;
                b.iter(|| {
                    let mut ones = 0u32;
                    for _ in 0..num_measurements {
                        if $bool_fn(&mut rng) {
                            ones += 1;
                        }
                    }
                    black_box(ones)
                });
            });
        };
    }

    bench_stabilizer!(
        "PecosQualityRng",
        PecosQualityRng::seed_from_u64(42),
        |r: &mut PecosQualityRng| r.random_bool(0.5)
    );
    bench_stabilizer!(
        "PecosRng",
        PecosRng::seed_from_u64(42),
        |r: &mut PecosRng| r.random_bool(0.5)
    );
    bench_stabilizer!(
        "SmallRng",
        SmallRng::seed_from_u64(42),
        |r: &mut SmallRng| r.random_bool(0.5)
    );
    bench_stabilizer!(
        "Xoshiro256++",
        Xoshiro256PlusPlus::seed_from_u64(42),
        |r: &mut Xoshiro256PlusPlus| r.random_bool(0.5)
    );
    bench_stabilizer!(
        "Xoroshiro128++",
        Xoroshiro128PlusPlus::seed_from_u64(42),
        |r: &mut Xoroshiro128PlusPlus| r.random_bool(0.5)
    );
    bench_stabilizer!(
        "Xoshiro512++",
        Xoshiro512PlusPlus::seed_from_u64(42),
        |r: &mut Xoshiro512PlusPlus| r.random_bool(0.5)
    );
    bench_stabilizer!(
        "RapidRng",
        RapidRng::new(42),
        |r: &mut RapidRng| u64_to_f64(r.next_u64()) < 0.5
    );
    bench_stabilizer!(
        "PCG64Fast",
        PCG64Fast::seed_from_u64(42),
        |r: &mut PCG64Fast| u64_to_f64(r.next_u64()) < 0.5
    );

    group.finish();
}

// ============================================================================
// Optimization Comparison - Bulk vs Scalar probability checking
// ============================================================================

/// Benchmark comparing fused vs separate noise sampling.
///
/// Compares:
/// - Separate: `check_probability` + `random_index_3` (original pattern)
/// - Fused: `noise_sample_1q` (new combined method)
#[allow(clippy::cast_possible_truncation)]
fn bench_fused_noise_sampling<M: Measurement>(c: &mut Criterion<M>) {
    use pecos_rng::rng_ext::RngProbabilityExt;

    let mut group = c.benchmark_group("RNG Optimization: Fused Noise Sampling");

    let num_gates = 10_000usize;
    let error_rate = 0.001f64;

    group.throughput(Throughput::Elements(num_gates as u64));

    // Baseline: separate check + Pauli selection (original pattern)
    group.bench_function("1q separate (PecosRng)", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        let threshold = rng.probability_threshold(error_rate);
        b.iter(|| {
            let mut error_count = 0u32;
            let mut pauli_sum = 0u32;
            for _ in 0..num_gates {
                if rng.check_probability(threshold) {
                    error_count += 1;
                    pauli_sum += u32::from(rng.random_index_3());
                }
            }
            black_box((error_count, pauli_sum))
        });
    });

    // Fused: noise_sample_1q
    group.bench_function("1q fused (PecosRng)", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        let threshold = rng.probability_threshold(error_rate);
        b.iter(|| {
            let mut error_count = 0u32;
            let mut pauli_sum = 0u32;
            for _ in 0..num_gates {
                if let Some(pauli) = rng.noise_sample_1q(threshold) {
                    error_count += 1;
                    pauli_sum += u32::from(pauli);
                }
            }
            black_box((error_count, pauli_sum))
        });
    });

    // Baseline: separate for 2q noise
    group.bench_function("2q separate (PecosRng)", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        let threshold = rng.probability_threshold(error_rate);
        b.iter(|| {
            let mut error_count = 0u32;
            let mut pauli_sum = 0u32;
            for _ in 0..num_gates {
                if rng.check_probability(threshold) {
                    error_count += 1;
                    pauli_sum += u32::from(rng.random_index_15());
                }
            }
            black_box((error_count, pauli_sum))
        });
    });

    // Fused: noise_sample_2q
    group.bench_function("2q fused (PecosRng)", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        let threshold = rng.probability_threshold(error_rate);
        b.iter(|| {
            let mut error_count = 0u32;
            let mut pauli_sum = 0u32;
            for _ in 0..num_gates {
                if let Some(pauli) = rng.noise_sample_2q(threshold) {
                    error_count += 1;
                    pauli_sum += u32::from(pauli);
                }
            }
            black_box((error_count, pauli_sum))
        });
    });

    // Compare with SmallRng (standard Rust RNG)
    group.bench_function("1q separate (SmallRng)", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        let threshold = rng.probability_threshold(error_rate);
        b.iter(|| {
            let mut error_count = 0u32;
            let mut pauli_sum = 0u32;
            for _ in 0..num_gates {
                if rng.check_probability(threshold) {
                    error_count += 1;
                    pauli_sum += u32::from(rng.random_index_3());
                }
            }
            black_box((error_count, pauli_sum))
        });
    });

    group.bench_function("1q fused (SmallRng)", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        let threshold = rng.probability_threshold(error_rate);
        b.iter(|| {
            let mut error_count = 0u32;
            let mut pauli_sum = 0u32;
            for _ in 0..num_gates {
                if let Some(pauli) = rng.noise_sample_1q(threshold) {
                    error_count += 1;
                    pauli_sum += u32::from(pauli);
                }
            }
            black_box((error_count, pauli_sum))
        });
    });

    group.finish();
}

/// Benchmark comparing batched vs scalar probability checking.
///
/// This simulates the noise model pattern: check many gates, get sparse error indices.
#[allow(clippy::cast_possible_truncation)]
fn bench_batched_probability<M: Measurement>(c: &mut Criterion<M>) {
    use pecos_rng::rng_ext::RngProbabilityExt;

    let mut group = c.benchmark_group("RNG Optimization: Batched Probability");

    let num_gates = 10_000usize;
    let error_rate = 0.001f64;

    group.throughput(Throughput::Elements(num_gates as u64));

    // Scalar loop: check each gate individually
    group.bench_function("scalar loop (PecosRng)", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        let threshold = rng.probability_threshold(error_rate);
        b.iter(|| {
            let mut indices = Vec::with_capacity(20);
            for i in 0..num_gates {
                if rng.check_probability(threshold) {
                    indices.push(i);
                }
            }
            black_box(indices)
        });
    });

    // Batched: get all indices at once
    group.bench_function("batched (PecosRng)", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        let threshold = rng.probability_threshold(error_rate);
        b.iter(|| {
            let indices = rng.check_probability_indices(threshold, num_gates);
            black_box(indices)
        });
    });

    // Batched with PecosScalarRng (scalar-optimized design)
    group.bench_function("batched (PecosScalarRng)", |b| {
        let mut rng = PecosScalarRng::seed_from_u64(42);
        let threshold = PecosScalarRng::probability_threshold(error_rate);
        b.iter(|| {
            let indices = rng.check_probability_indices(threshold, num_gates);
            black_box(indices)
        });
    });

    // SmallRng baseline (trait default implementation)
    group.bench_function("batched (SmallRng)", |b| {
        let mut rng = SmallRng::seed_from_u64(42);
        let threshold = rng.probability_threshold(error_rate);
        b.iter(|| {
            let indices = rng.check_probability_indices(threshold, num_gates);
            black_box(indices)
        });
    });

    group.finish();
}

/// Benchmark comparing different probability checking methods.
///
/// This compares:
/// - Scalar: `check_probability()` in a loop
/// - Bulk: `check_probability_bulk()` filling a slice
/// - x4: `check_probability_x4()` returning 4 results
/// - Count: `count_occurrences()` for just counting
#[allow(clippy::cast_possible_truncation)] // Count is bounded by 4-element array
pub fn bench_probability_optimizations<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("RNG Optimization: Probability Check");

    let num_checks = 10_000usize;
    let error_rate = 0.001f64;
    let threshold = PecosRng::probability_threshold(error_rate);

    group.throughput(Throughput::Elements(num_checks as u64));

    // PecosRng: scalar check_probability in a loop
    group.bench_function("PecosRng scalar", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| {
            let mut count = 0u32;
            for _ in 0..num_checks {
                if rng.check_probability(threshold) {
                    count += 1;
                }
            }
            black_box(count)
        });
    });

    // PecosRng: x4 check
    group.bench_function("PecosRng x4", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| {
            let mut count = 0u32;
            for _ in 0..(num_checks / 4) {
                let results = rng.check_probability_x4(threshold);
                count += results.iter().filter(|&&x| x).count() as u32;
            }
            for _ in 0..(num_checks % 4) {
                if rng.check_probability(threshold) {
                    count += 1;
                }
            }
            black_box(count)
        });
    });

    // PecosRng: count_occurrences
    group.bench_function("PecosRng count", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| {
            let count = rng.count_occurrences(threshold, num_checks);
            black_box(count)
        });
    });

    // PecosScalarRng: scalar check_probability (no buffer, direct RapidRng)
    group.bench_function("PecosScalarRng scalar", |b| {
        let mut rng = PecosScalarRng::seed_from_u64(42);
        b.iter(|| {
            let mut count = 0u32;
            for _ in 0..num_checks {
                if rng.check_probability(threshold) {
                    count += 1;
                }
            }
            black_box(count)
        });
    });

    // PecosScalarRng: count_occurrences
    group.bench_function("PecosScalarRng count", |b| {
        let mut rng = PecosScalarRng::seed_from_u64(42);
        b.iter(|| {
            let count = rng.count_occurrences(threshold, num_checks);
            black_box(count)
        });
    });

    // Comparison: RapidRng scalar (fastest scalar RNG)
    group.bench_function("RapidRng scalar", |b| {
        let mut rng = RapidRng::new(42);
        b.iter(|| {
            let mut count = 0u32;
            for _ in 0..num_checks {
                if rng.next_u64() < threshold {
                    count += 1;
                }
            }
            black_box(count)
        });
    });

    // Trait default: SmallRng using RngProbabilityExt trait
    group.bench_function("SmallRng (trait)", |b| {
        use pecos_rng::RngProbabilityExt;
        let mut rng = SmallRng::seed_from_u64(42);
        b.iter(|| {
            let count = rng.count_occurrences(threshold, num_checks);
            black_box(count)
        });
    });

    group.finish();
}
