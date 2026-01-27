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

//! Benchmarks comparing `Angle<T>` trig functions against raw `f64` trig.
//!
//! Three baselines are tested:
//! - **angle**: Our octant-based implementation on `Angle64`
//! - **f64 raw**: `f64::sin/cos/sin_cos` on pre-computed radians (best possible)
//! - **angle via radians**: `angle.to_radians().sin()` (old implementation path)

use criterion::{BenchmarkId, Criterion, Throughput, measurement::Measurement};
use pecos_core::Angle64;
use std::hint::black_box;

const BATCH: usize = 10_000;

/// Pre-generated test data: paired `Angle64` values and their radian equivalents.
struct TrigData {
    angles: Vec<Angle64>,
    radians: Vec<f64>,
}

impl TrigData {
    fn new(n: usize) -> Self {
        // Use a fixed-step pattern to cover all octants uniformly.
        // Avoids RNG overhead and ensures reproducible benchmarks.
        let step = u64::MAX / n as u64;
        let angles: Vec<Angle64> = (0..n)
            .map(|i| Angle64::new(step.wrapping_mul(i as u64)))
            .collect();
        let radians: Vec<f64> = angles.iter().map(Angle64::to_radians).collect();
        Self { angles, radians }
    }
}

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    bench_sin_cos(c);
    bench_sin(c);
    bench_cos(c);
    bench_tan(c);
}

fn bench_sin_cos<M: Measurement>(c: &mut Criterion<M>) {
    let data = TrigData::new(BATCH);
    let mut group = c.benchmark_group("Trig/sin_cos");
    group.throughput(Throughput::Elements(BATCH as u64));

    group.bench_function(BenchmarkId::new("angle", BATCH), |b| {
        b.iter(|| {
            for a in &data.angles {
                black_box(a.sin_cos());
            }
        });
    });

    group.bench_function(BenchmarkId::new("f64_raw", BATCH), |b| {
        b.iter(|| {
            for &r in &data.radians {
                black_box(r.sin_cos());
            }
        });
    });

    group.bench_function(BenchmarkId::new("angle_via_radians", BATCH), |b| {
        b.iter(|| {
            for a in &data.angles {
                let r = a.to_radians();
                black_box(r.sin_cos());
            }
        });
    });

    group.finish();
}

fn bench_sin<M: Measurement>(c: &mut Criterion<M>) {
    let data = TrigData::new(BATCH);
    let mut group = c.benchmark_group("Trig/sin");
    group.throughput(Throughput::Elements(BATCH as u64));

    group.bench_function(BenchmarkId::new("angle", BATCH), |b| {
        b.iter(|| {
            for a in &data.angles {
                black_box(a.sin());
            }
        });
    });

    group.bench_function(BenchmarkId::new("f64_raw", BATCH), |b| {
        b.iter(|| {
            for &r in &data.radians {
                black_box(r.sin());
            }
        });
    });

    group.bench_function(BenchmarkId::new("angle_via_radians", BATCH), |b| {
        b.iter(|| {
            for a in &data.angles {
                black_box(a.to_radians().sin());
            }
        });
    });

    group.finish();
}

fn bench_cos<M: Measurement>(c: &mut Criterion<M>) {
    let data = TrigData::new(BATCH);
    let mut group = c.benchmark_group("Trig/cos");
    group.throughput(Throughput::Elements(BATCH as u64));

    group.bench_function(BenchmarkId::new("angle", BATCH), |b| {
        b.iter(|| {
            for a in &data.angles {
                black_box(a.cos());
            }
        });
    });

    group.bench_function(BenchmarkId::new("f64_raw", BATCH), |b| {
        b.iter(|| {
            for &r in &data.radians {
                black_box(r.cos());
            }
        });
    });

    group.bench_function(BenchmarkId::new("angle_via_radians", BATCH), |b| {
        b.iter(|| {
            for a in &data.angles {
                black_box(a.to_radians().cos());
            }
        });
    });

    group.finish();
}

fn bench_tan<M: Measurement>(c: &mut Criterion<M>) {
    let data = TrigData::new(BATCH);
    let mut group = c.benchmark_group("Trig/tan");
    group.throughput(Throughput::Elements(BATCH as u64));

    group.bench_function(BenchmarkId::new("angle", BATCH), |b| {
        b.iter(|| {
            for a in &data.angles {
                black_box(a.tan());
            }
        });
    });

    group.bench_function(BenchmarkId::new("f64_raw", BATCH), |b| {
        b.iter(|| {
            for &r in &data.radians {
                black_box(r.tan());
            }
        });
    });

    group.bench_function(BenchmarkId::new("angle_via_radians", BATCH), |b| {
        b.iter(|| {
            for a in &data.angles {
                black_box(a.to_radians().tan());
            }
        });
    });

    group.finish();
}
