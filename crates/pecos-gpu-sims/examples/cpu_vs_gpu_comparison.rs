//! Quick CPU vs GPU comparison for surface codes with 2*d rounds
//!
//! Run with: cargo run --example `cpu_vs_gpu_comparison` --release -p pecos-gpu-sims

use pecos_gpu_sims::{GpuInfluenceMapData, GpuInfluenceSampler};
use pecos_qec::fault_tolerance::InfluenceBuilder;
use pecos_qec::fault_tolerance::dem_builder::DemSampler;
use pecos_quantum::DagCircuit;
use std::time::Instant;

fn build_surface_code_grid(distance: usize, num_rounds: usize) -> DagCircuit {
    let mut dag = DagCircuit::new();

    let num_data = distance * distance;
    let num_x_ancillas = (distance - 1) * (distance - 1);
    let num_z_ancillas = (distance - 1) * (distance - 1);

    let x_ancilla_start = num_data;
    let z_ancilla_start = num_data + num_x_ancillas;

    for _round in 0..num_rounds {
        for a in 0..num_x_ancillas {
            dag.pz(&[x_ancilla_start + a]);
            dag.h(&[x_ancilla_start + a]);
        }
        for a in 0..num_z_ancillas {
            dag.pz(&[z_ancilla_start + a]);
        }

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

        for a in 0..num_x_ancillas {
            dag.h(&[x_ancilla_start + a]);
            dag.mz(&[x_ancilla_start + a]);
        }
        for a in 0..num_z_ancillas {
            dag.mz(&[z_ancilla_start + a]);
        }
    }

    dag
}

fn main() {
    println!("CPU vs GPU Influence Sampler Comparison");
    println!("Surface codes with distance d, rounds = 2*d");
    println!("100,000 shots per measurement");
    println!("=========================================\n");

    let p_error = 0.001;
    let seed = 42u64;
    let num_shots = 100_000u32;

    println!(
        "{:>4} {:>6} {:>8} {:>12} {:>12} {:>10}",
        "d", "Rounds", "Locs", "CPU (ms)", "GPU (ms)", "Speedup"
    );
    println!("{:-<60}", "");

    for distance in [3, 5, 7, 9, 11] {
        let num_rounds = 2 * distance;
        let circuit = build_surface_code_grid(distance, num_rounds);
        let num_data = distance * distance;

        // Build influence map
        let tracked_pauli_qubits: Vec<usize> = (0..num_data).collect();
        let builder = InfluenceBuilder::new(&circuit).with_z(&tracked_pauli_qubits);
        let influence_map = builder.build();

        let num_locations = influence_map.locations.len();

        // Export for GPU
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

        // CPU benchmark
        let num_loc = influence_map.locations.len();
        let probs = vec![p_error; num_loc];
        let cpu_sampler = DemSampler::from_influence_map(&influence_map, &probs);

        let cpu_start = Instant::now();
        let _ = cpu_sampler.sample_statistics(num_shots as usize, seed);
        let cpu_time = cpu_start.elapsed();

        // GPU benchmark (with warmup)
        let mut gpu_sampler =
            GpuInfluenceSampler::new(&gpu_map, seed).expect("Failed to create GPU sampler");
        let _ = gpu_sampler.sample_uniform(1000, p_error); // Warmup

        let gpu_start = Instant::now();
        let _ = gpu_sampler.sample_uniform(num_shots, p_error);
        let gpu_time = gpu_start.elapsed();

        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

        println!(
            "{:>4} {:>6} {:>8} {:>12.1} {:>12.1} {:>9.1}x",
            distance,
            num_rounds,
            num_locations,
            cpu_time.as_secs_f64() * 1000.0,
            gpu_time.as_secs_f64() * 1000.0,
            speedup
        );
    }

    println!("{:-<60}", "");
    println!("\nThroughput (M shots/sec):");
    println!("{:-<60}", "");
    println!(
        "{:>4} {:>6} {:>15} {:>15} {:>10}",
        "d", "Rounds", "CPU", "GPU", "Speedup"
    );
    println!("{:-<60}", "");

    for distance in [3, 5, 7, 9, 11] {
        let num_rounds = 2 * distance;
        let circuit = build_surface_code_grid(distance, num_rounds);
        let num_data = distance * distance;

        let tracked_pauli_qubits: Vec<usize> = (0..num_data).collect();
        let builder = InfluenceBuilder::new(&circuit).with_z(&tracked_pauli_qubits);
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

        // CPU
        let probs2 = vec![p_error; influence_map.locations.len()];
        let cpu_sampler = DemSampler::from_influence_map(&influence_map, &probs2);
        let cpu_start = Instant::now();
        let _ = cpu_sampler.sample_statistics(num_shots as usize, seed);
        let cpu_time = cpu_start.elapsed();

        // GPU
        let mut gpu_sampler = GpuInfluenceSampler::new(&gpu_map, seed).unwrap();
        let _ = gpu_sampler.sample_uniform(1000, p_error);
        let gpu_start = Instant::now();
        let _ = gpu_sampler.sample_uniform(num_shots, p_error);
        let gpu_time = gpu_start.elapsed();

        let cpu_throughput = f64::from(num_shots) / cpu_time.as_secs_f64() / 1_000_000.0;
        let gpu_throughput = f64::from(num_shots) / gpu_time.as_secs_f64() / 1_000_000.0;
        let speedup = gpu_throughput / cpu_throughput;

        println!(
            "{distance:>4} {num_rounds:>6} {cpu_throughput:>15.3} {gpu_throughput:>15.3} {speedup:>9.1}x"
        );
    }

    println!("\nComparison complete!");
}
