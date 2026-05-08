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

//! DEM Sampler benchmarks for threshold estimation.

use criterion::{BenchmarkId, Criterion, Throughput, measurement::Measurement};
use pecos_qec::fault_tolerance::dem_builder::DemSamplerBuilder;
use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;
use pecos_quantum::DagCircuit;
use pecos_random::PecosRng;
use std::hint::black_box;

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    bench_sampler(c);
    bench_statistics(c);
}

/// Create a realistic DEM sampler from a surface-code-like circuit.
fn create_surface_code_sampler(
    distance: usize,
    rounds: usize,
) -> pecos_qec::fault_tolerance::dem_builder::DemSampler {
    // Create a simplified surface code circuit
    let num_data = distance * distance;
    let num_ancilla = num_data - 1;
    let total_qubits = num_data + num_ancilla;

    let mut dag = DagCircuit::new();

    // Prep all qubits
    let all_qubits: Vec<usize> = (0..total_qubits).collect();
    dag.pz(&all_qubits);

    // Syndrome extraction rounds
    let ancilla_qubits: Vec<usize> = (num_data..total_qubits).collect();
    for _ in 0..rounds {
        dag.h(&ancilla_qubits);
        for a in 0..num_ancilla {
            let data_q = a.min(num_data - 1);
            dag.cx(&[(data_q, num_data + a)]);
        }
        dag.h(&ancilla_qubits);
        dag.mz(&ancilla_qubits);
        dag.pz(&ancilla_qubits);
    }

    // Measure data qubits
    let data_qubits: Vec<usize> = (0..num_data).collect();
    dag.mz(&data_qubits);

    // Build influence map
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence_map = analyzer.build_influence_map();

    // Build DEM sampler with simple detectors
    let num_measurements = num_ancilla * rounds + num_data;
    let mut detector_records = Vec::new();
    for i in 0..num_measurements.min(50) {
        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
        detector_records.push(vec![-(i as i32 + 1)]);
    }

    DemSamplerBuilder::new(&influence_map)
        .with_noise(0.001, 0.001, 0.001, 0.001)
        .with_detector_records(detector_records)
        .with_observable_records(vec![])
        .build()
        .expect("Failed to build DEM sampler")
}

/// Benchmark sampling with different circuit sizes.
fn bench_sampler<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("DEM Sampler - Batch");

    for (distance, rounds) in [(3, 2), (5, 3)] {
        let sampler = create_surface_code_sampler(distance, rounds);
        let num_mechanisms = sampler.num_mechanisms();
        let num_detectors = sampler.num_detectors();

        for shots in [1_000, 10_000, 100_000] {
            let label = format!("d{distance}_r{rounds}_{shots}");
            group.throughput(Throughput::Elements((num_mechanisms * shots) as u64));

            group.bench_with_input(BenchmarkId::new("sample_batch", &label), &(), |b, ()| {
                let mut rng = PecosRng::seed_from_u64(42);
                b.iter(|| {
                    let result = sampler.sample_batch(shots, &mut rng);
                    black_box(result)
                });
            });
        }

        println!(
            "  d={distance} r={rounds}: {num_mechanisms} mechanisms, {num_detectors} detectors"
        );
    }

    group.finish();
}

/// Benchmark statistics methods.
fn bench_statistics<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("DEM Sampler - Statistics");

    let sampler = create_surface_code_sampler(5, 3);
    let num_mechanisms = sampler.num_mechanisms();

    for shots in [10_000, 100_000] {
        let label = format!("{shots}shots");
        group.throughput(Throughput::Elements((num_mechanisms * shots) as u64));

        group.bench_with_input(
            BenchmarkId::new("sample_statistics", &label),
            &shots,
            |b, &shots| {
                let mut rng = PecosRng::seed_from_u64(42);
                b.iter(|| {
                    let result = sampler.sample_statistics_with_rng(shots, &mut rng);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}
