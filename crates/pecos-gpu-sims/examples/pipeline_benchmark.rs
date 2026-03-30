//! Benchmark comparing CPU vs GPU sampling pipelines
//!
//! This benchmark tests both pipelines across various circuit complexities:
//! - Different code sizes (number of data qubits)
//! - Different round counts
//! - Different shot counts
//!
//! Run with: cargo run --example `pipeline_benchmark` --release

use pecos_gpu_sims::{GpuInfluenceMapData, GpuInfluenceSampler};
use pecos_qec::fault_tolerance::InfluenceBuilder;
use pecos_qec::fault_tolerance::noisy_sampler::{NoisySampler, UniformNoiseModel};
use pecos_quantum::DagCircuit;
use std::time::{Duration, Instant};

/// Build a repetition code syndrome extraction circuit.
///
/// Data qubits: `0..num_data`
/// Ancilla qubits: `num_data..(num_data` + `num_data` - 1)
fn build_repetition_code(num_data: usize, num_rounds: usize) -> DagCircuit {
    let mut dag = DagCircuit::new();
    let num_ancillas = num_data - 1;

    for _round in 0..num_rounds {
        // Prepare ancillas
        for a in 0..num_ancillas {
            dag.pz(&[num_data + a]);
        }

        // Parity checks: Z_i * Z_{i+1} for each adjacent pair
        for a in 0..num_ancillas {
            dag.cx(&[(a, num_data + a)]);
            dag.cx(&[(a + 1, num_data + a)]);
        }

        // Measure ancillas
        for a in 0..num_ancillas {
            dag.mz(&[num_data + a]);
        }
    }

    dag
}

/// Build a 2D surface code plaquette grid.
///
/// This creates a simplified surface code with X and Z plaquettes.
/// Data qubits form a grid, ancillas measure 4-body operators.
fn build_surface_code_grid(distance: usize, num_rounds: usize) -> DagCircuit {
    let mut dag = DagCircuit::new();

    // For a distance-d surface code:
    // - d^2 data qubits arranged in a grid
    // - (d-1)^2 X plaquettes + (d-1)^2 Z plaquettes (simplified)
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
                // Four data qubits around this plaquette
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

struct BenchmarkResult {
    name: String,
    num_locations: usize,
    num_detectors: usize,
    num_shots: usize,
    build_time: Duration,
    cpu_time: Duration,
    gpu_time: Duration,
    _cpu_logical_errors: usize,
    _gpu_logical_errors: usize,
}

impl BenchmarkResult {
    fn speedup(&self) -> f64 {
        self.cpu_time.as_secs_f64() / self.gpu_time.as_secs_f64()
    }

    fn cpu_throughput(&self) -> f64 {
        self.num_shots as f64 / self.cpu_time.as_secs_f64() / 1_000_000.0
    }

    fn gpu_throughput(&self) -> f64 {
        self.num_shots as f64 / self.gpu_time.as_secs_f64() / 1_000_000.0
    }
}

fn benchmark_circuit(
    name: &str,
    circuit: &DagCircuit,
    logical_qubits: Vec<usize>,
    num_shots: u32,
    p_error: f64,
    seed: u64,
) -> BenchmarkResult {
    // Build influence map (common to both pipelines)
    let build_start = Instant::now();
    let builder = InfluenceBuilder::new(circuit).with_logical_z(logical_qubits);
    let influence_map = builder.build();
    let build_time = build_start.elapsed();

    let num_locations = influence_map.locations.len();
    let num_detectors = influence_map.detectors.len();

    // CPU sampling
    let noise = UniformNoiseModel::depolarizing(p_error);
    let mut cpu_sampler = NoisySampler::new(&influence_map, noise, seed);

    let cpu_start = Instant::now();
    let cpu_results = cpu_sampler.sample(num_shots as usize);
    let cpu_time = cpu_start.elapsed();

    let cpu_logical_errors = cpu_results.iter().filter(|r| r.has_logical_error()).count();

    // GPU sampling
    let (
        num_loc,
        num_det,
        num_log,
        det_off_x,
        det_data_x,
        det_off_y,
        det_data_y,
        det_off_z,
        det_data_z,
        log_off_x,
        log_data_x,
        log_off_y,
        log_data_y,
        log_off_z,
        log_data_z,
    ) = influence_map.export_csr();

    let gpu_map = GpuInfluenceMapData::from_csr(
        num_loc, num_det, num_log, det_off_x, det_data_x, det_off_y, det_data_y, det_off_z,
        det_data_z, log_off_x, log_data_x, log_off_y, log_data_y, log_off_z, log_data_z,
    );

    let mut gpu_sampler =
        GpuInfluenceSampler::new(&gpu_map, seed).expect("Failed to create GPU sampler");

    // Warm up GPU
    let _ = gpu_sampler.sample_uniform(100, p_error);

    let gpu_start = Instant::now();
    let gpu_result = gpu_sampler.sample_uniform(num_shots, p_error);
    let gpu_time = gpu_start.elapsed();

    let gpu_logical_errors = gpu_result.count_logical_errors();

    BenchmarkResult {
        name: name.to_string(),
        num_locations,
        num_detectors,
        num_shots: num_shots as usize,
        build_time,
        cpu_time,
        gpu_time,
        _cpu_logical_errors: cpu_logical_errors,
        _gpu_logical_errors: gpu_logical_errors,
    }
}

fn print_results(results: &[BenchmarkResult]) {
    println!("\n{:=<100}", "");
    println!(
        "{:<25} {:>8} {:>8} {:>10} {:>12} {:>12} {:>10}",
        "Circuit", "Locs", "Dets", "Shots", "CPU (ms)", "GPU (ms)", "Speedup"
    );
    println!("{:-<100}", "");

    for r in results {
        println!(
            "{:<25} {:>8} {:>8} {:>10} {:>12.2} {:>12.2} {:>9.1}x",
            r.name,
            r.num_locations,
            r.num_detectors,
            r.num_shots,
            r.cpu_time.as_secs_f64() * 1000.0,
            r.gpu_time.as_secs_f64() * 1000.0,
            r.speedup()
        );
    }
    println!("{:=<100}", "");
}

fn print_throughput(results: &[BenchmarkResult]) {
    println!("\nThroughput (M shots/sec):");
    println!("{:-<70}", "");
    println!(
        "{:<25} {:>15} {:>15} {:>12}",
        "Circuit", "CPU", "GPU", "Speedup"
    );
    println!("{:-<70}", "");

    for r in results {
        println!(
            "{:<25} {:>15.2} {:>15.2} {:>11.1}x",
            r.name,
            r.cpu_throughput(),
            r.gpu_throughput(),
            r.speedup()
        );
    }
}

fn main() {
    println!("Pipeline Benchmark: CPU vs GPU Sampling");
    println!("========================================\n");

    let p_error = 0.001;
    let seed = 42u64;

    // =========================================================================
    // Test 1: Varying circuit size (repetition code)
    // =========================================================================
    println!("Test 1: Repetition Code - Varying Size (fixed 100k shots)\n");

    let mut results = Vec::new();
    let num_shots = 100_000u32;

    for (num_data, num_rounds) in [(3, 2), (5, 3), (7, 4), (9, 5), (11, 6), (15, 8)] {
        let circuit = build_repetition_code(num_data, num_rounds);
        let logical_qubits: Vec<usize> = (0..num_data).collect();
        let name = format!("rep_d{num_data}r{num_rounds}");

        let result = benchmark_circuit(&name, &circuit, logical_qubits, num_shots, p_error, seed);
        results.push(result);
    }

    print_results(&results);

    // =========================================================================
    // Test 2: Varying shot count (fixed circuit)
    // =========================================================================
    println!("\nTest 2: Fixed Circuit (rep_d7r4) - Varying Shots\n");

    let circuit = build_repetition_code(7, 4);
    let logical_qubits: Vec<usize> = (0..7).collect();

    let mut shot_results = Vec::new();

    for num_shots in [1_000u32, 10_000, 50_000, 100_000, 500_000, 1_000_000] {
        let name = format!("{}k shots", num_shots / 1000);
        let result = benchmark_circuit(
            &name,
            &circuit,
            logical_qubits.clone(),
            num_shots,
            p_error,
            seed,
        );
        shot_results.push(result);
    }

    print_results(&shot_results);

    // =========================================================================
    // Test 3: Surface code (higher complexity)
    // =========================================================================
    println!("\nTest 3: Surface Code Grid - Varying Distance (fixed 50k shots)\n");

    let mut surface_results = Vec::new();
    let num_shots = 50_000u32;

    for (distance, rounds) in [(3, 2), (4, 2), (5, 3), (6, 3), (7, 4)] {
        let circuit = build_surface_code_grid(distance, rounds);
        let num_data = distance * distance;
        let logical_qubits: Vec<usize> = (0..num_data).collect();
        let name = format!("surf_d{distance}r{rounds}");

        let result = benchmark_circuit(&name, &circuit, logical_qubits, num_shots, p_error, seed);
        surface_results.push(result);
    }

    print_results(&surface_results);

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\nThroughput Summary:");
    println!("{:=<70}", "");

    println!("\nRepetition Codes:");
    print_throughput(&results);

    println!("\nSurface Codes:");
    print_throughput(&surface_results);

    // =========================================================================
    // Build time analysis
    // =========================================================================
    println!("\n\nBuild Time Analysis (influence map construction):");
    println!("{:-<70}", "");
    println!(
        "{:<25} {:>10} {:>12} {:>15}",
        "Circuit", "Locs", "Build (ms)", "Locs/ms"
    );
    println!("{:-<70}", "");

    for r in results.iter().chain(surface_results.iter()) {
        println!(
            "{:<25} {:>10} {:>12.2} {:>15.1}",
            r.name,
            r.num_locations,
            r.build_time.as_secs_f64() * 1000.0,
            r.num_locations as f64 / (r.build_time.as_secs_f64() * 1000.0)
        );
    }

    println!("\n{:=<70}", "");
    println!("Benchmark complete!");
}
