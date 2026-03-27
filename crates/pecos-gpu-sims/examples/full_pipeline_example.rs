//! Full pipeline example: Circuit -> Influence Map -> GPU Sampling
//!
//! This example demonstrates the complete workflow for building fault influence
//! maps from quantum error correction circuits and sampling with the GPU.
//!
//! Run with: cargo run --example `full_pipeline_example` --release
//!
//! Pipeline steps:
//! 1. Build a syndrome extraction circuit using `DagCircuit`
//! 2. Use `InfluenceBuilder` to extract detectors and build influence map
//! 3. Convert to GPU format using `export_csr()`
//! 4. Use `GpuInfluenceSampler` for fast noisy sampling

use pecos_gpu_sims::{GpuInfluenceMapData, GpuInfluenceSampler};
use pecos_qec::fault_tolerance::InfluenceBuilder;
use pecos_quantum::DagCircuit;

/// Build a simple repetition code syndrome extraction circuit.
///
/// Data qubits: 0, 1, 2 (Z-stabilizer = Z0 Z1 Z2)
/// Ancilla qubits: 3, 4 (measure Z0*Z1 and Z1*Z2)
fn build_repetition_code_circuit(num_rounds: usize) -> DagCircuit {
    let mut dag = DagCircuit::new();

    for _round in 0..num_rounds {
        // Prepare ancillas in |0>
        dag.pz(&[3]);
        dag.pz(&[4]);

        // First parity check: Z0 * Z1
        // CNOT from data to ancilla to copy Z parities
        dag.cx(&[(0, 3)]);
        dag.cx(&[(1, 3)]);

        // Second parity check: Z1 * Z2
        dag.cx(&[(1, 4)]);
        dag.cx(&[(2, 4)]);

        // Measure ancillas
        dag.mz(&[3]);
        dag.mz(&[4]);
    }

    dag
}

/// Build a surface code plaquette extraction circuit (simplified).
///
/// This is a single X-type plaquette measuring X0 X1 X2 X3.
/// Data qubits: 0, 1, 2, 3
/// Ancilla qubit: 4
fn build_surface_code_plaquette(num_rounds: usize) -> DagCircuit {
    let mut dag = DagCircuit::new();

    for _round in 0..num_rounds {
        // Prepare ancilla in |+> (H applied to |0>)
        dag.pz(&[4]);
        dag.h(&[4]);

        // CNOT from ancilla to each data qubit (X-basis measurement)
        dag.cx(&[(4, 0)]);
        dag.cx(&[(4, 1)]);
        dag.cx(&[(4, 2)]);
        dag.cx(&[(4, 3)]);

        // H then measure (X-basis measurement on ancilla)
        dag.h(&[4]);
        dag.mz(&[4]);
    }

    dag
}

fn main() {
    println!("Full Pipeline Example: Circuit -> Influence Map -> GPU Sampling\n");
    println!("{:=<70}", "");

    // =========================================================================
    // Example 1: Repetition Code
    // =========================================================================
    println!("\n1. Repetition Code (3 data qubits, 2 rounds)\n");

    let circuit = build_repetition_code_circuit(2);
    println!("   Circuit built: {} gates", circuit.gate_count());

    // Build influence map with logical Z (sensitive to X errors)
    let builder = InfluenceBuilder::new(&circuit).with_logical_z(vec![0, 1, 2]);

    let influence_map = builder.build();
    println!("   Locations: {}", influence_map.locations.len());
    println!("   Detectors: {}", influence_map.detectors.len());
    println!("   Measurements: {}", influence_map.measurements.len());

    // Export to GPU format
    let (
        num_locations,
        num_detectors,
        num_logicals,
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

    println!(
        "   Exported CSR: {num_locations} locations, {num_detectors} detectors, {num_logicals} logicals"
    );

    let gpu_map = GpuInfluenceMapData::from_csr(
        num_locations,
        num_detectors,
        num_logicals,
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
    );

    // Sample with GPU
    let mut sampler = GpuInfluenceSampler::new(&gpu_map, 42).expect("Failed to create GPU sampler");

    let num_shots = 10_000;
    let p_error = 0.001; // 0.1% error rate

    let result = sampler.sample_uniform(num_shots, p_error);
    let logical_errors = result.count_logical_errors();
    let error_rate = logical_errors as f64 / f64::from(num_shots);

    println!(
        "   GPU Sampling: {} shots, p={}, logical error rate: {:.4}%",
        num_shots,
        p_error,
        error_rate * 100.0
    );

    // =========================================================================
    // Example 2: Surface Code Plaquette
    // =========================================================================
    println!("\n2. Surface Code X-Plaquette (4 data qubits, 3 rounds)\n");

    let circuit = build_surface_code_plaquette(3);
    println!("   Circuit built: {} gates", circuit.gate_count());

    // Build influence map with logical X (sensitive to Z errors on this plaquette)
    let builder = InfluenceBuilder::new(&circuit).with_logical_x(vec![0, 1, 2, 3]);

    let influence_map = builder.build();
    println!("   Locations: {}", influence_map.locations.len());
    println!("   Detectors: {}", influence_map.detectors.len());

    // Export to GPU format
    let (
        num_locations,
        num_detectors,
        num_logicals,
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
        num_locations,
        num_detectors,
        num_logicals,
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
    );

    let mut sampler = GpuInfluenceSampler::new(&gpu_map, 42).expect("Failed to create GPU sampler");

    let result = sampler.sample_uniform(num_shots, p_error);
    let logical_errors = result.count_logical_errors();
    let error_rate = logical_errors as f64 / f64::from(num_shots);

    println!(
        "   GPU Sampling: {} shots, p={}, logical error rate: {:.4}%",
        num_shots,
        p_error,
        error_rate * 100.0
    );

    // =========================================================================
    // Example 3: Scaling test
    // =========================================================================
    println!("\n3. Scaling Test (repetition code, varying rounds)\n");

    for num_rounds in [1, 2, 4, 8] {
        let circuit = build_repetition_code_circuit(num_rounds);
        let builder = InfluenceBuilder::new(&circuit).with_logical_z(vec![0, 1, 2]);
        let influence_map = builder.build();

        let (
            num_locations,
            num_detectors,
            num_logicals,
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
            num_locations,
            num_detectors,
            num_logicals,
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
        );

        let mut sampler =
            GpuInfluenceSampler::new(&gpu_map, 42).expect("Failed to create GPU sampler");

        let start = std::time::Instant::now();
        let result = sampler.sample_uniform(100_000, 0.001);
        let elapsed = start.elapsed();

        let logical_errors = result.count_logical_errors();

        println!(
            "   {} rounds: {} locations, {} detectors, {} logical errors, {:.2}ms",
            num_rounds,
            num_locations,
            num_detectors,
            logical_errors,
            elapsed.as_secs_f64() * 1000.0
        );
    }

    println!("\n{:=<70}", "");
    println!("\nPipeline complete!");
}
