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

//! DEM construction and decomposition benchmarks.
//!
//! These benchmarks exercise `DetectorErrorModel::to_string()` and the two
//! decomposed-output paths (`to_string_decomposed`, `to_string_decomposed_maximally`)
//! on surface-code-like DEMs of increasing distance. Decomposition exercises the
//! hot maps inside `dem_builder/types.rs` (`candidates_by_detector`, the memoization
//! caches inside `find_hyperedge_decomposition` / `find_singleton_decomposition`,
//! and the render-side `rendered_targets_cache`). These are the structures whose
//! HashMap/BTreeMap choice has been debated; this bench gives the numbers to
//! settle it.

use criterion::{BenchmarkId, Criterion, Throughput, measurement::Measurement};
use pecos_qec::fault_tolerance::dem_builder::{DemBuilder, DetectorErrorModel};
use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;
use pecos_quantum::DagCircuit;
use std::fmt::Write;
use std::hint::black_box;

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    bench_dem_build(c);
    bench_dem_render(c);
    bench_dem_render_decomposed(c);
    bench_dem_render_decomposed_maximally(c);
}

/// Build a surface-code-like DAG and its fault influence map.
///
/// Mirrors the simplified construction in `dem_sampler.rs`. Not a true surface
/// code, but produces correlated multi-detector errors that exercise the DEM
/// decomposition paths.
fn build_surface_code_dem(distance: usize, rounds: usize) -> DetectorErrorModel {
    let num_data = distance * distance;
    let num_ancilla = num_data - 1;

    let mut dag = DagCircuit::new();

    for q in 0..num_data {
        dag.pz(&[q]);
        dag.h(&[q]);
    }

    for _round in 0..rounds {
        for a in 0..num_ancilla {
            dag.pz(&[num_data + a]);
        }
        for a in 0..num_ancilla {
            let ancilla = num_data + a;
            let d1 = a % num_data;
            let d2 = (a + 1) % num_data;
            dag.cx(&[(ancilla, d1)]);
            dag.cx(&[(ancilla, d2)]);
        }
        for a in 0..num_ancilla {
            dag.mz(&[num_data + a]);
        }
    }

    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence_map = analyzer.build_influence_map();

    // Detectors: each ancilla measurement after the first round XORed with the
    // previous round's measurement of the same ancilla. This produces typical
    // 2-detector graphlike mechanisms plus some higher-weight correlated errors.
    let mut detectors = String::from("[");
    let mut first = true;
    let mut det_id: u32 = 0;
    for round in 1..rounds {
        for a in 0..num_ancilla {
            #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
            let curr = -((round * num_ancilla - a) as i32);
            #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
            let prev = -(((round + 1) * num_ancilla - a) as i32);
            if !first {
                detectors.push(',');
            }
            write!(detectors, r#"{{"id":{det_id},"records":[{curr},{prev}]}}"#)
                .expect("writing to String cannot fail");
            det_id += 1;
            first = false;
        }
    }
    detectors.push(']');

    DemBuilder::new(&influence_map)
        .with_noise(0.001, 0.001, 0.001, 0.001)
        .with_detectors_json(&detectors)
        .expect("detectors json should parse")
        .with_observables_json("[]")
        .expect("observables json should parse")
        .build()
}

fn bench_dem_build<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("dem_builder/build");
    for &(distance, rounds) in &[(3usize, 3usize), (5, 5), (7, 5)] {
        let num_contribs_hint = distance * distance * rounds;
        #[allow(clippy::cast_possible_truncation)]
        group.throughput(Throughput::Elements(num_contribs_hint as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("d{distance}_r{rounds}")),
            &(distance, rounds),
            |b, &(d, r)| {
                b.iter(|| black_box(build_surface_code_dem(black_box(d), black_box(r))));
            },
        );
    }
    group.finish();
}

fn bench_dem_render<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("dem_builder/render_plain");
    for &(distance, rounds) in &[(3usize, 3usize), (5, 5), (7, 5)] {
        let dem = build_surface_code_dem(distance, rounds);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("d{distance}_r{rounds}")),
            &dem,
            |b, dem| {
                b.iter(|| black_box(dem.to_string()));
            },
        );
    }
    group.finish();
}

fn bench_dem_render_decomposed<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("dem_builder/render_decomposed");
    for &(distance, rounds) in &[(3usize, 3usize), (5, 5), (7, 5)] {
        let dem = build_surface_code_dem(distance, rounds);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("d{distance}_r{rounds}")),
            &dem,
            |b, dem| {
                b.iter(|| black_box(dem.to_string_decomposed()));
            },
        );
    }
    group.finish();
}

fn bench_dem_render_decomposed_maximally<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("dem_builder/render_decomposed_maximally");
    for &(distance, rounds) in &[(3usize, 3usize), (5, 5), (7, 5)] {
        let dem = build_surface_code_dem(distance, rounds);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("d{distance}_r{rounds}")),
            &dem,
            |b, dem| {
                b.iter(|| black_box(dem.to_string_decomposed_maximally()));
            },
        );
    }
    group.finish();
}
