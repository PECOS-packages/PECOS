// Copyright 2026 The PECOS Developers
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

//! Exhaustive code-distance search benchmarks.

use criterion::{Criterion, measurement::Measurement};
use pecos_core::{Xs, Zs};
use pecos_qec::{
    DistanceSearchConfig, StabilizerCode, StabilizerCodeSpec, calculate_distance,
    find_shortest_logicals,
};
use std::hint::black_box;

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    eprintln!(
        "code-distance benchmark available parallelism: {}",
        std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
    );

    let five_qubit = standard_code_spec(&StabilizerCode::five_qubit());
    let steane = standard_code_spec(&StabilizerCode::steane());
    let toric_3 = standard_code_spec(&StabilizerCode::toric(3));
    let color_17 = color_code_17();
    let config = DistanceSearchConfig::default();

    let mut group = c.benchmark_group("code_distance/calculate_distance");
    group.sample_size(10);
    group.bench_function("five_qubit_5_1_3", |b| {
        b.iter(|| calculate_distance(black_box(&five_qubit), black_box(&config)));
    });
    group.bench_function("steane_7_1_3", |b| {
        b.iter(|| calculate_distance(black_box(&steane), black_box(&config)));
    });
    group.bench_function("toric_3_18_2_3", |b| {
        b.iter(|| calculate_distance(black_box(&toric_3), black_box(&config)));
    });
    group.bench_function("color_17_1_5", |b| {
        b.iter(|| calculate_distance(black_box(&color_17), black_box(&config)));
    });
    group.finish();

    let mut group = c.benchmark_group("code_distance/find_shortest_logicals");
    group.sample_size(10);
    group.bench_function("color_17_1_5_delta_1", |b| {
        b.iter(|| find_shortest_logicals(black_box(&color_17), black_box(&config), 1));
    });
    group.finish();
}

fn standard_code_spec(code: &StabilizerCode) -> StabilizerCodeSpec {
    StabilizerCodeSpec::from_stabilizer_code(code)
        .expect("standard stabilizer code should have discoverable logicals")
}

fn color_code_17() -> StabilizerCodeSpec {
    const SUPPORTS: [&[usize]; 8] = [
        &[0, 9, 12, 15],
        &[1, 9, 12, 16],
        &[2, 11, 13, 14],
        &[3, 8, 9, 12],
        &[4, 8, 10, 12, 13, 14, 15, 16],
        &[5, 10, 11, 13],
        &[6, 8, 9, 10, 13, 14, 15, 16],
        &[7, 10, 11, 14],
    ];

    let mut builder = StabilizerCodeSpec::builder(17);
    for support in SUPPORTS {
        builder = builder.check(Xs(support));
    }
    for support in SUPPORTS {
        builder = builder.check(Zs(support));
    }

    builder
        .logical_x(Xs(0..17))
        .logical_z(Zs(0..17))
        .build()
        .expect("[[17,1,5]] color code should be valid")
}
