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

use criterion::{black_box, measurement::Measurement, BenchmarkGroup, Criterion};
use pecos_core::{IndexableElement, Set, VecSet};

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Set Operations");
    bench_set_operations::<usize, M>(&mut group, "usize");
    bench_set_operations::<u16, M>(&mut group, "u16");
    bench_vecset_operations::<usize, M>(&mut group, "usize");
    bench_vecset_operations::<u16, M>(&mut group, "u16");
    group.finish();
}

fn bench_set_operations<E: IndexableElement + From<u16>, M: Measurement>(
    group: &mut BenchmarkGroup<M>,
    type_name: &str,
) {
    group.bench_function(format!("set_operations_{type_name}"), |b| {
        b.iter(|| {
            let mut set = VecSet::<E>::new();
            for i in 0..100_u16 {
                set.insert(E::from(i));
            }
            for i in 0..100_u16 {
                black_box(set.contains(&E::from(i)));
            }
            for i in 0..100_u16 {
                set.remove(&E::from(i));
            }
        });
    });
}

fn bench_vecset_operations<E: IndexableElement + From<u8> + Copy, M: Measurement>(
    group: &mut BenchmarkGroup<M>,
    type_name: &str,
) {
    // Benchmark insert
    group.bench_function(format!("VecSet<{type_name}>/insert"), |b| {
        b.iter(|| {
            let mut set = VecSet::<E>::new();
            for i in 0..100_u8 {
                set.insert(E::from(i));
            }
        });
    });

    // Benchmark contains
    group.bench_function(format!("VecSet<{type_name}>/contains"), |b| {
        let set: VecSet<E> = (0..100_u8).map(E::from).collect();
        b.iter(|| {
            for i in 0..100_u8 {
                black_box(set.contains(&E::from(i)));
            }
        });
    });

    // Benchmark remove
    group.bench_function(format!("VecSet<{type_name}>/remove"), |b| {
        b.iter(|| {
            let mut set: VecSet<E> = (0..100_u8).map(E::from).collect();
            for i in 0..100_u8 {
                set.remove(&E::from(i));
            }
        });
    });

    // Benchmark union
    group.bench_function(format!("VecSet<{type_name}>/union"), |b| {
        let set1: VecSet<E> = (0..50_u8).map(E::from).collect();
        let set2: VecSet<E> = (25..75_u8).map(E::from).collect();
        b.iter(|| {
            let mut result = VecSet::<E>::new();
            for &item in set1.union(&set2) {
                result.insert(item);
            }
            black_box(result);
        });
    });

    // Benchmark intersection
    group.bench_function(format!("VecSet<{type_name}>/intersection"), |b| {
        let set1: VecSet<E> = (0..50_u8).map(E::from).collect();
        let set2: VecSet<E> = (25..75_u8).map(E::from).collect();
        b.iter(|| {
            let mut result = VecSet::<E>::new();
            for &item in set1.intersection(&set2) {
                result.insert(item);
            }
            black_box(result);
        });
    });
}
