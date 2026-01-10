// Copyright 2024 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

use criterion::{BenchmarkGroup, Criterion, measurement::Measurement};
use pecos::prelude::*;
use std::hint::black_box;

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Set Operations");
    bench_set_operations(&mut group);
    bench_vecset_operations(&mut group);
    group.finish();
}

fn bench_set_operations<M: Measurement>(group: &mut BenchmarkGroup<M>) {
    group.bench_function("set_operations_usize", |b| {
        b.iter(|| {
            let mut set = VecSet::<usize>::new();
            for i in 0..100_usize {
                set.insert(i);
            }
            for i in 0..100_usize {
                black_box(set.contains(&i));
            }
            for i in 0..100_usize {
                set.remove(&i);
            }
        });
    });
}

fn bench_vecset_operations<M: Measurement>(group: &mut BenchmarkGroup<M>) {
    // Benchmark insert
    group.bench_function("VecSet<usize>/insert", |b| {
        b.iter(|| {
            let mut set = VecSet::<usize>::new();
            for i in 0..100_usize {
                set.insert(i);
            }
        });
    });

    // Benchmark contains
    group.bench_function("VecSet<usize>/contains", |b| {
        let set: VecSet<usize> = (0..100_usize).collect();
        b.iter(|| {
            for i in 0..100_usize {
                black_box(set.contains(&i));
            }
        });
    });

    // Benchmark remove
    group.bench_function("VecSet<usize>/remove", |b| {
        b.iter(|| {
            let mut set: VecSet<usize> = (0..100_usize).collect();
            for i in 0..100_usize {
                set.remove(&i);
            }
        });
    });

    // Benchmark union
    group.bench_function("VecSet<usize>/union", |b| {
        let set1: VecSet<usize> = (0..50_usize).collect();
        let set2: VecSet<usize> = (25..75_usize).collect();
        b.iter(|| {
            let mut result = VecSet::<usize>::new();
            for &item in set1.union(&set2) {
                result.insert(item);
            }
            black_box(result);
        });
    });

    // Benchmark intersection
    group.bench_function("VecSet<usize>/intersection", |b| {
        let set1: VecSet<usize> = (0..50_usize).collect();
        let set2: VecSet<usize> = (25..75_usize).collect();
        b.iter(|| {
            let mut result = VecSet::<usize>::new();
            for &item in set1.intersection(&set2) {
                result.insert(item);
            }
            black_box(result);
        });
    });
}
