// Copyright 2026 The PECOS Developers
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
use pecos_core::{PauliBitmask, PauliBitmaskSmall, PauliBitmaskVec};
use std::hint::black_box;

macro_rules! define_operand_builders {
    ($dense:ident, $sparse:ident, $pauli:ty) => {
        fn $dense(num_qubits: usize, offset: usize) -> $pauli {
            let mut result = <$pauli>::identity();
            for qubit in 0..num_qubits {
                let factor = match (qubit + offset) % 3 {
                    0 => <$pauli>::x(qubit),
                    1 => <$pauli>::y(qubit),
                    _ => <$pauli>::z(qubit),
                };
                result = result.multiply(&factor);
            }
            result
        }

        fn $sparse(sites: &[usize], register_width: usize, offset: usize) -> $pauli {
            let mut result = <$pauli>::identity();
            for &qubit in sites {
                let factor = match (qubit + offset) % 3 {
                    0 => <$pauli>::x(qubit),
                    1 => <$pauli>::y(qubit),
                    _ => <$pauli>::z(qubit),
                };
                result = result.multiply(&factor);
            }

            // Preserve a wide backing store while leaving the top qubit as identity.
            let padding = <$pauli>::y(register_width - 1);
            result = result.multiply(&padding);
            result.multiply(&padding)
        }
    };
}

define_operand_builders!(dense_u128, sparse_u128, PauliBitmask);
define_operand_builders!(dense_vec, sparse_vec, PauliBitmaskVec);
define_operand_builders!(dense_small, sparse_small, PauliBitmaskSmall);

macro_rules! bench_multiply_with_phase {
    ($group:expr, $id:expr, $left:expr, $right:expr) => {{
        let left = $left;
        let right = $right;
        $group.bench_function($id, move |b| {
            b.iter(|| black_box(black_box(&left).multiply_with_phase(black_box(&right))));
        });
    }};
}

macro_rules! bench_multiply {
    ($group:expr, $id:expr, $left:expr, $right:expr) => {{
        let left = $left;
        let right = $right;
        $group.bench_function($id, move |b| {
            b.iter(|| black_box(black_box(&left).multiply(black_box(&right))));
        });
    }};
}

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Pauli Operations");
    bench_u128(&mut group);
    bench_vec(&mut group);
    bench_small(&mut group);
    group.finish();
}

fn bench_u128<M: Measurement>(group: &mut BenchmarkGroup<M>) {
    bench_multiply_with_phase!(
        group,
        "multiply_with_phase/dense/u128/64q",
        dense_u128(64, 0),
        dense_u128(64, 1)
    );
    bench_multiply_with_phase!(
        group,
        "multiply_with_phase/dense/u128/128q",
        dense_u128(128, 0),
        dense_u128(128, 1)
    );
    bench_multiply_with_phase!(
        group,
        "multiply_with_phase/sparse_low/u128/128q",
        sparse_u128(&[0, 1, 2, 3], 128, 0),
        sparse_u128(&[0, 1, 2, 3], 128, 1)
    );
    bench_multiply!(
        group,
        "multiply/dense/u128/128q",
        dense_u128(128, 0),
        dense_u128(128, 1)
    );
}

fn bench_vec<M: Measurement>(group: &mut BenchmarkGroup<M>) {
    for (size, id) in [
        (64, "multiply_with_phase/dense/vec_u64/64q"),
        (128, "multiply_with_phase/dense/vec_u64/128q"),
        (512, "multiply_with_phase/dense/vec_u64/512q"),
        (1024, "multiply_with_phase/dense/vec_u64/1024q"),
    ] {
        bench_multiply_with_phase!(group, id, dense_vec(size, 0), dense_vec(size, 1));
    }
    bench_multiply_with_phase!(
        group,
        "multiply_with_phase/sparse_low/vec_u64/1024q",
        sparse_vec(&[0, 1, 2, 3], 1024, 0),
        sparse_vec(&[0, 1, 2, 3], 1024, 1)
    );
    bench_multiply_with_phase!(
        group,
        "multiply_with_phase/sparse_high/vec_u64/1024q",
        sparse_vec(&[1000, 1007, 1016, 1023], 1024, 0),
        sparse_vec(&[1000, 1007, 1016, 1023], 1024, 1)
    );

    let identity = PauliBitmaskVec::identity();
    let dense = dense_vec(1024, 0);
    bench_multiply_with_phase!(
        group,
        "multiply_with_phase/empty_times_long/vec_u64/identity_left/1024q",
        identity.clone(),
        dense.clone()
    );
    bench_multiply_with_phase!(
        group,
        "multiply_with_phase/empty_times_long/vec_u64/identity_right/1024q",
        dense,
        identity
    );
    bench_multiply!(
        group,
        "multiply/dense/vec_u64/128q",
        dense_vec(128, 0),
        dense_vec(128, 1)
    );
}

fn bench_small<M: Measurement>(group: &mut BenchmarkGroup<M>) {
    for (size, id) in [
        (64, "multiply_with_phase/dense/smallvec_u64/64q"),
        (128, "multiply_with_phase/dense/smallvec_u64/128q"),
        (512, "multiply_with_phase/dense/smallvec_u64/512q"),
        (1024, "multiply_with_phase/dense/smallvec_u64/1024q"),
    ] {
        bench_multiply_with_phase!(group, id, dense_small(size, 0), dense_small(size, 1));
    }
    bench_multiply_with_phase!(
        group,
        "multiply_with_phase/sparse_low/smallvec_u64/1024q",
        sparse_small(&[0, 1, 2, 3], 1024, 0),
        sparse_small(&[0, 1, 2, 3], 1024, 1)
    );
    bench_multiply_with_phase!(
        group,
        "multiply_with_phase/sparse_high/smallvec_u64/1024q",
        sparse_small(&[1000, 1007, 1016, 1023], 1024, 0),
        sparse_small(&[1000, 1007, 1016, 1023], 1024, 1)
    );

    let identity = PauliBitmaskSmall::identity();
    let dense = dense_small(1024, 0);
    bench_multiply_with_phase!(
        group,
        "multiply_with_phase/empty_times_long/smallvec_u64/identity_left/1024q",
        identity.clone(),
        dense.clone()
    );
    bench_multiply_with_phase!(
        group,
        "multiply_with_phase/empty_times_long/smallvec_u64/identity_right/1024q",
        dense,
        identity
    );
    bench_multiply!(
        group,
        "multiply/dense/smallvec_u64/128q",
        dense_small(128, 0),
        dense_small(128, 1)
    );
}
