// Copyright 2025 The PECOS Developers
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

//! GPU Influence Sampler benchmarks.
//!
//! Tests realistic QEC workloads:
//! - Surface codes with distance d and 2*d syndrome extraction rounds
//! - Varying shot counts
//!
//! Run with: cargo bench --features gpu-sims -p benchmarks -- `gpu_influence`

use criterion::{BenchmarkId, Criterion, Throughput, measurement::Measurement};
use pecos_gpu_sims::{GpuInfluenceMapData, GpuInfluenceSampler};
use pecos_qec::fault_tolerance::dem_builder::DemSampler;
use pecos_qec::fault_tolerance::{DagFaultInfluenceMap, InfluenceBuilder};
use pecos_quantum::DagCircuit;
use std::hint::black_box;

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    bench_cpu_vs_gpu_surface_codes(c);
    bench_gpu_sampler_shot_scaling(c);
}

/// Build a 2D surface code plaquette grid.
fn build_surface_code_grid(distance: usize, num_rounds: usize) -> DagCircuit {
    let mut dag = DagCircuit::new();

    let num_data = distance * distance;
    let num_x_ancillas = (distance - 1) * (distance - 1);
    let num_z_ancillas = (distance - 1) * (distance - 1);

    let x_ancilla_start = num_data;
    let z_ancilla_start = num_data + num_x_ancillas;

    for _round in 0..num_rounds {
        // Prepare X ancillas in |+>
        for a in 0..num_x_ancillas {
            dag.pz(&[x_ancilla_start + a]);
            dag.h(&[x_ancilla_start + a]);
        }

        // Prepare Z ancillas in |0>
        for a in 0..num_z_ancillas {
            dag.pz(&[z_ancilla_start + a]);
        }

        // X plaquette measurements (CNOT from ancilla to data)
        for row in 0..(distance - 1) {
            for col in 0..(distance - 1) {
                let ancilla = x_ancilla_start + row * (distance - 1) + col;
                let d0 = row * distance + col;
                let d1 = row * distance + col + 1;
                let d2 = (row + 1) * distance + col;
                let d3 = (row + 1) * distance + col + 1;

                dag.cx(&[(ancilla, d0)]);
                dag.cx(&[(ancilla, d1)]);
                dag.cx(&[(ancilla, d2)]);
                dag.cx(&[(ancilla, d3)]);
            }
        }

        // Z plaquette measurements (CNOT from data to ancilla)
        for row in 0..(distance - 1) {
            for col in 0..(distance - 1) {
                let ancilla = z_ancilla_start + row * (distance - 1) + col;
                let d0 = row * distance + col;
                let d1 = row * distance + col + 1;
                let d2 = (row + 1) * distance + col;
                let d3 = (row + 1) * distance + col + 1;

                dag.cx(&[(d0, ancilla)]);
                dag.cx(&[(d1, ancilla)]);
                dag.cx(&[(d2, ancilla)]);
                dag.cx(&[(d3, ancilla)]);
            }
        }

        // Measure X ancillas (H then MZ)
        for a in 0..num_x_ancillas {
            dag.h(&[x_ancilla_start + a]);
            dag.mz(&[x_ancilla_start + a]);
        }

        // Measure Z ancillas
        for a in 0..num_z_ancillas {
            dag.mz(&[z_ancilla_start + a]);
        }
    }

    dag
}

/// Helper to build influence maps for both CPU and GPU.
fn build_influence_maps(
    circuit: &DagCircuit,
    num_data: usize,
) -> (DagFaultInfluenceMap, GpuInfluenceMapData) {
    let tracked_pauli_qubits: Vec<usize> = (0..num_data).collect();
    let builder = InfluenceBuilder::new(circuit).with_z(&tracked_pauli_qubits);
    let influence_map = builder.build();

    let (
        num_loc,
        num_det,
        num_dem_outputs,
        det_off_x,
        det_data_x,
        det_off_y,
        det_data_y,
        det_off_z,
        det_data_z,
        dem_output_offsets_x,
        dem_output_data_x,
        dem_output_offsets_y,
        dem_output_data_y,
        dem_output_offsets_z,
        dem_output_data_z,
    ) = influence_map.export_csr();

    let gpu_map = GpuInfluenceMapData::from_csr(
        num_loc,
        num_det,
        num_dem_outputs,
        det_off_x,
        det_data_x,
        det_off_y,
        det_data_y,
        det_off_z,
        det_data_z,
        dem_output_offsets_x,
        dem_output_data_x,
        dem_output_offsets_y,
        dem_output_data_y,
        dem_output_offsets_z,
        dem_output_data_z,
    );

    (influence_map, gpu_map)
}

/// Benchmark CPU vs GPU for surface codes with 2*d rounds.
fn bench_cpu_vs_gpu_surface_codes<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Influence Sampler - CPU vs GPU");

    let p_error = 0.001;
    let seed = 42u64;
    let num_shots = 100_000u32;

    // Surface codes with d=3,5,7,9,11 and rounds=2*d
    for distance in [3, 5, 7, 9, 11] {
        let num_rounds = 2 * distance;
        let circuit = build_surface_code_grid(distance, num_rounds);
        let num_data = distance * distance;
        let (cpu_map, gpu_map) = build_influence_maps(&circuit, num_data);

        let label = format!("d{distance}_r{num_rounds}");

        group.throughput(Throughput::Elements(u64::from(num_shots)));

        // CPU benchmark
        let probs = vec![p_error; cpu_map.locations.len()];
        let cpu_sampler = DemSampler::from_influence_map(&cpu_map, &probs);

        group.bench_with_input(BenchmarkId::new("CPU", &label), &(), |b, ()| {
            b.iter(|| black_box(cpu_sampler.sample_statistics(num_shots as usize, seed)));
        });

        // GPU benchmark
        let mut gpu_sampler =
            GpuInfluenceSampler::new(&gpu_map, seed).expect("Failed to create GPU sampler");
        // Warm up
        let _ = gpu_sampler.sample_uniform(1000, p_error);

        group.bench_with_input(BenchmarkId::new("GPU", &label), &(), |b, ()| {
            b.iter(|| black_box(gpu_sampler.sample_uniform(num_shots, p_error)));
        });
    }

    group.finish();
}

/// Benchmark how CPU and GPU scale with shot count.
fn bench_gpu_sampler_shot_scaling<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Influence Sampler - Shot Scaling");

    let p_error = 0.001;
    let seed = 42u64;

    // Use d=5, 10 rounds as representative workload
    let distance = 5;
    let num_rounds = 10;
    let circuit = build_surface_code_grid(distance, num_rounds);
    let num_data = distance * distance;
    let (cpu_map, gpu_map) = build_influence_maps(&circuit, num_data);

    for num_shots in [10_000u32, 50_000, 100_000, 500_000, 1_000_000] {
        let label = format!("{num_shots}shots");

        group.throughput(Throughput::Elements(u64::from(num_shots)));

        // CPU benchmark
        let probs = vec![p_error; cpu_map.locations.len()];
        let cpu_sampler = DemSampler::from_influence_map(&cpu_map, &probs);

        group.bench_with_input(BenchmarkId::new("CPU", &label), &num_shots, |b, &shots| {
            b.iter(|| black_box(cpu_sampler.sample_statistics(shots as usize, seed)));
        });

        // GPU benchmark
        let mut gpu_sampler =
            GpuInfluenceSampler::new(&gpu_map, seed).expect("Failed to create GPU sampler");
        let _ = gpu_sampler.sample_uniform(1000, p_error);

        group.bench_with_input(BenchmarkId::new("GPU", &label), &num_shots, |b, &shots| {
            b.iter(|| black_box(gpu_sampler.sample_uniform(shots, p_error)));
        });
    }

    group.finish();
}
