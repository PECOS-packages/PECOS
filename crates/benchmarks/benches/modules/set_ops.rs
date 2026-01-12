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
use pecos_core::BitSet;
use std::hint::black_box;

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Set Operations");
    bench_set_operations(&mut group);
    bench_vecset_operations(&mut group);
    group.finish();

    let mut comparison_group = c.benchmark_group("BitSet vs VecSet");
    bench_bitset_vs_vecset(&mut comparison_group);
    comparison_group.finish();
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

/// Benchmark `BitSet` vs `VecSet` for stabilizer simulation operations.
///
/// Key operations tested:
/// - Single element toggle (used in CX gate's inner loop)
/// - XOR / `symmetric_difference_update` (used in many gates)
/// - Iteration (used for applying updates)
#[allow(clippy::too_many_lines)]
fn bench_bitset_vs_vecset<M: Measurement>(group: &mut BenchmarkGroup<M>) {
    // Parameters matching surface code simulation
    // Small sets: stabilizer weights of 2-4
    // Element range: 0 to num_qubits (e.g., 241 for d11)

    // === Single element toggle (most critical for CX gate) ===

    // VecSet: toggle single element in small set (weight 4)
    group.bench_function("toggle_single/VecSet/size4", |b| {
        let mut set: VecSet<usize> = [10, 50, 100, 200].into_iter().collect();
        b.iter(|| {
            // Toggle element 50 (present) then toggle it back
            set.symmetric_difference_item_update(&50);
            set.symmetric_difference_item_update(&50);
            black_box(&set);
        });
    });

    // BitSet: toggle single element in small set (weight 4)
    group.bench_function("toggle_single/BitSet/size4", |b| {
        let mut set: BitSet = [10, 50, 100, 200].into_iter().collect();
        let toggle = BitSet::single(50); // Pre-create
        b.iter(|| {
            // Toggle element 50 (present) then toggle it back
            set.symmetric_difference_update(&toggle);
            set.symmetric_difference_update(&toggle);
            black_box(&set);
        });
    });

    // === XOR with another set (used in H, SZ, Y gates) ===

    // VecSet: XOR two small sets
    group.bench_function("xor_sets/VecSet/size4x4", |b| {
        let set2: VecSet<usize> = [10, 50, 150, 220].into_iter().collect();
        b.iter(|| {
            let mut set1: VecSet<usize> = [10, 50, 100, 200].into_iter().collect();
            set1.symmetric_difference_update(&set2);
            black_box(set1);
        });
    });

    // BitSet: XOR two small sets
    group.bench_function("xor_sets/BitSet/size4x4", |b| {
        let set2: BitSet = [10, 50, 150, 220].into_iter().collect();
        b.iter(|| {
            let mut set1: BitSet = [10, 50, 100, 200].into_iter().collect();
            set1.symmetric_difference_update(&set2);
            black_box(set1);
        });
    });

    // === CX-like loop: toggle many single elements ===
    // This simulates the CX inner loop: for each generator, toggle a qubit

    group.bench_function("cx_loop/VecSet/100_toggles", |b| {
        b.iter(|| {
            let mut sets: Vec<VecSet<usize>> = (0..100)
                .map(|i| [i, i + 100, i + 200, i + 300].into_iter().collect())
                .collect();
            let target_qubit = 50_usize;
            for set in &mut sets {
                set.symmetric_difference_item_update(&target_qubit);
            }
            black_box(&sets);
        });
    });

    group.bench_function("cx_loop/BitSet/100_toggles", |b| {
        let target_set = BitSet::single(50); // Pre-create outside loop
        b.iter(|| {
            let mut sets: Vec<BitSet> = (0..100)
                .map(|i| [i, i + 100, i + 200, i + 300].into_iter().collect())
                .collect();
            for set in &mut sets {
                set.symmetric_difference_update(&target_set);
            }
            black_box(&sets);
        });
    });

    // === CX-like loop with pre-existing sets (realistic) ===
    // Sets created once, then modified many times

    group.bench_function("cx_realistic/VecSet/100_toggles", |b| {
        let mut sets: Vec<VecSet<usize>> = (0..100)
            .map(|i| [i, i + 100, i + 200, i + 300].into_iter().collect())
            .collect();
        let target_qubit = 50_usize;
        b.iter(|| {
            for set in &mut sets {
                set.symmetric_difference_item_update(&target_qubit);
            }
            black_box(&sets);
        });
    });

    group.bench_function("cx_realistic/BitSet/100_toggles", |b| {
        let mut sets: Vec<BitSet> = (0..100)
            .map(|i| [i, i + 100, i + 200, i + 300].into_iter().collect())
            .collect();
        let target_set = BitSet::single(50);
        b.iter(|| {
            for set in &mut sets {
                set.symmetric_difference_update(&target_set);
            }
            black_box(&sets);
        });
    });

    // === Iteration (needed to apply row updates) ===

    group.bench_function("iterate/VecSet/size4", |b| {
        let set: VecSet<usize> = [10, 50, 100, 200].into_iter().collect();
        b.iter(|| {
            let sum: usize = set.iter().copied().sum();
            black_box(sum);
        });
    });

    group.bench_function("iterate/BitSet/size4", |b| {
        let set: BitSet = [10, 50, 100, 200].into_iter().collect();
        b.iter(|| {
            let sum: usize = set.iter().sum();
            black_box(sum);
        });
    });

    // === Larger sets (for bigger circuits) ===

    group.bench_function("xor_sets/VecSet/size20x20", |b| {
        let set2: VecSet<usize> = (0..20).map(|i| i * 25).collect();
        b.iter(|| {
            let mut set1: VecSet<usize> = (0..20).map(|i| i * 25 + 10).collect();
            set1.symmetric_difference_update(&set2);
            black_box(set1);
        });
    });

    group.bench_function("xor_sets/BitSet/size20x20", |b| {
        let set2: BitSet = (0..20).map(|i| i * 25).collect();
        b.iter(|| {
            let mut set1: BitSet = (0..20).map(|i| i * 25 + 10).collect();
            set1.symmetric_difference_update(&set2);
            black_box(set1);
        });
    });
}
