//! Example showing the full CPU pipeline for noisy QEC sampling.
//!
//! Pipeline steps:
//! 1. Build a syndrome extraction circuit using `DagCircuit`
//! 2. Use `InfluenceBuilder` to extract detectors and build influence map
//! 3. Use `DemSampler` for fast CPU-based noisy sampling
//!
//! Run with: cargo run --example `influence_builder_example` --release -p pecos-qec

use pecos_qec::fault_tolerance::InfluenceBuilder;
use pecos_qec::fault_tolerance::dem_builder::DemSampler;
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

fn main() {
    println!("CPU Pipeline Example: Circuit -> Influence Map -> DemSampler\n");
    println!("{:=<70}", "");

    // =========================================================================
    // Build circuit
    // =========================================================================
    let num_rounds = 3;
    let circuit = build_repetition_code_circuit(num_rounds);
    println!("\n1. Circuit built:");
    println!("   Rounds: {num_rounds}");
    println!("   Gates: {}", circuit.gate_count());

    // =========================================================================
    // Build influence map with InfluenceBuilder
    // =========================================================================
    let builder = InfluenceBuilder::new(&circuit).with_z(&[0, 1, 2]); // Z logical on all data qubits

    let influence_map = builder.build();

    println!("\n2. Influence map built:");
    println!("   Fault locations: {}", influence_map.locations.len());
    println!("   Detectors: {}", influence_map.detectors.len());
    println!("   Measurements: {}", influence_map.measurements.len());

    // Show detector definitions
    println!("\n   Detector definitions:");
    for (i, detector) in influence_map.detectors.iter().enumerate() {
        let meas_str: Vec<String> = detector
            .measurements
            .iter()
            .map(|m| format!("m[t={},q={}]", m.tick, m.qubit))
            .collect();
        println!("     D{}: XOR({})", i, meas_str.join(", "));
    }

    // =========================================================================
    // Sample with DemSampler
    // =========================================================================
    let p_error = 0.001; // 0.1% error rate per location
    let seed = 42u64;
    let num_locations = influence_map.locations.len();
    let per_location_probs = vec![p_error; num_locations];

    let sampler = DemSampler::from_influence_map(&influence_map, &per_location_probs);

    println!("\n3. Sampling with DemSampler:");
    println!("   Error rate: {p_error}");
    println!("   Mechanisms: {}", sampler.num_mechanisms());

    let num_shots = 100_000;
    let start = std::time::Instant::now();
    let stats = sampler.sample_statistics(num_shots, seed);
    let elapsed = start.elapsed();

    println!("   Shots: {num_shots}");
    println!("   Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    #[allow(clippy::cast_precision_loss)]
    let throughput = num_shots as f64 / elapsed.as_secs_f64() / 1_000_000.0;
    println!("   Throughput: {throughput:.2}M shots/sec");

    println!("\n4. Results:");
    println!(
        "   Logical error rate: {:.4}%",
        stats.logical_error_rate() * 100.0
    );
    println!("   Syndrome rate: {:.4}%", stats.syndrome_rate() * 100.0);
    println!(
        "   Undetectable error rate: {:.6}%",
        stats.undetectable_rate() * 100.0
    );

    // =========================================================================
    // Compare with different error rates
    // =========================================================================
    println!("\n5. Scaling with error rate:\n");
    println!(
        "   {:>8} {:>15} {:>15}",
        "p_error", "Logical Error%", "Syndrome%"
    );
    println!("   {:->8} {:->15} {:->15}", "", "", "");

    for p in [0.0001, 0.0005, 0.001, 0.002, 0.005] {
        let probs = vec![p; num_locations];
        let sampler = DemSampler::from_influence_map(&influence_map, &probs);
        let stats = sampler.sample_statistics(50_000, seed);

        println!(
            "   {:>8.4} {:>14.4}% {:>14.4}%",
            p,
            stats.logical_error_rate() * 100.0,
            stats.syndrome_rate() * 100.0
        );
    }

    println!("\n{:=<70}", "");
    println!("\nCPU pipeline complete!");
}
