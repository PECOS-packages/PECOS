// Benchmark: Gate fusion in realistic circuit patterns
// Compares StateVecSoA with fusion enabled vs disabled

use pecos_core::QubitId;
use pecos_qsim::{CliffordGateable, QuantumSimulator, StateVecSoA};
use std::time::Instant;

fn qid(q: usize) -> [QubitId; 1] {
    [QubitId(q)]
}

fn qid2(q1: usize, q2: usize) -> [QubitId; 2] {
    [QubitId(q1), QubitId(q2)]
}

/// Pattern 1: Random-ish Clifford circuit with frequent CX gates
/// This is typical of many quantum algorithms
fn circuit_frequent_cx(sim: &mut StateVecSoA, num_qubits: usize, depth: usize) {
    for layer in 0..depth {
        // Single-qubit layer
        for q in 0..num_qubits {
            match (layer + q) % 3 {
                0 => { sim.h(&qid(q)); }
                1 => { sim.sz(&qid(q)); }
                _ => { sim.x(&qid(q)); }
            }
        }
        // Two-qubit layer (causes flush)
        for q in (0..num_qubits - 1).step_by(2) {
            sim.cx(&qid2(q, q + 1));
        }
    }
}

/// Pattern 2: Multiple single-qubit gates per qubit before CX
/// This is where fusion should help
fn circuit_multi_single_then_cx(sim: &mut StateVecSoA, num_qubits: usize, depth: usize) {
    for _ in 0..depth {
        // Multiple single-qubit gates per qubit (fusion opportunity)
        for q in 0..num_qubits {
            sim.h(&qid(q));
            sim.sz(&qid(q));
            sim.h(&qid(q));
        }
        // Two-qubit layer
        for q in (0..num_qubits - 1).step_by(2) {
            sim.cx(&qid2(q, q + 1));
        }
    }
}

/// Pattern 3: State preparation with many rotations per qubit
/// Strong fusion opportunity
fn circuit_state_prep(sim: &mut StateVecSoA, num_qubits: usize, rotations_per_qubit: usize) {
    for q in 0..num_qubits {
        for _ in 0..rotations_per_qubit {
            sim.h(&qid(q));
            sim.sz(&qid(q));
        }
    }
}

/// Pattern 4: Alternating qubits with non-canceling gates
/// Gates spread across different qubits, using SZ which accumulates
fn circuit_alternating(sim: &mut StateVecSoA, num_qubits: usize, num_gates: usize) {
    for i in 0..num_gates {
        let q = i % num_qubits;
        sim.sz(&qid(q)); // SZ^4 = I, but SZ^1,2,3 don't cancel
    }
}

/// Pattern 5: True worst case - one gate per qubit, then flush
fn circuit_single_gate_per_qubit(sim: &mut StateVecSoA, num_qubits: usize, repetitions: usize) {
    for _ in 0..repetitions {
        for q in 0..num_qubits {
            sim.h(&qid(q));
        }
        // Force flush by accessing state
        sim.flush();
    }
}

fn benchmark_circuit<F>(name: &str, num_qubits: usize, iterations: usize, circuit_fn: F)
where
    F: Fn(&mut StateVecSoA),
{
    // Warmup
    for _ in 0..3 {
        let mut sim: StateVecSoA = StateVecSoA::new(num_qubits);
        sim.set_fusion(true);
        circuit_fn(&mut sim);

        let mut sim: StateVecSoA = StateVecSoA::new(num_qubits);
        sim.set_fusion(false);
        circuit_fn(&mut sim);
    }

    // Benchmark with fusion ON
    let start = Instant::now();
    for _ in 0..iterations {
        let mut sim: StateVecSoA = StateVecSoA::new(num_qubits);
        sim.set_fusion(true);
        circuit_fn(&mut sim);
        sim.flush(); // Ensure all gates applied
    }
    let fusion_on_time = start.elapsed();

    // Benchmark with fusion OFF
    let start = Instant::now();
    for _ in 0..iterations {
        let mut sim: StateVecSoA = StateVecSoA::new(num_qubits);
        sim.set_fusion(false);
        circuit_fn(&mut sim);
    }
    let fusion_off_time = start.elapsed();

    let speedup = fusion_off_time.as_nanos() as f64 / fusion_on_time.as_nanos() as f64;

    println!(
        "{:40} | ON: {:>8.2}ms | OFF: {:>8.2}ms | Speedup: {:.2}x",
        name,
        fusion_on_time.as_secs_f64() * 1000.0,
        fusion_off_time.as_secs_f64() * 1000.0,
        speedup
    );
}

fn verify_correctness() {
    println!("Verifying fusion correctness...");

    let num_qubits = 4;

    // Test each circuit pattern produces identical results
    let patterns: Vec<(&str, Box<dyn Fn(&mut StateVecSoA)>)> = vec![
        ("Frequent CX", Box::new(|sim| circuit_frequent_cx(sim, 4, 5))),
        ("Multi single-qubit", Box::new(|sim| circuit_multi_single_then_cx(sim, 4, 5))),
        ("State prep", Box::new(|sim| circuit_state_prep(sim, 4, 4))),
        ("Alternating SZ", Box::new(|sim| circuit_alternating(sim, 4, 20))),
    ];

    for (name, circuit_fn) in patterns {
        let mut sim_on: StateVecSoA = StateVecSoA::new(num_qubits);
        sim_on.set_fusion(true);
        circuit_fn(&mut sim_on);
        sim_on.flush();

        let mut sim_off: StateVecSoA = StateVecSoA::new(num_qubits);
        sim_off.set_fusion(false);
        circuit_fn(&mut sim_off);

        // Compare states
        let state_on = sim_on.to_complex_vec();
        let state_off = sim_off.to_complex_vec();

        let mut max_diff = 0.0f64;
        for (a, b) in state_on.iter().zip(state_off.iter()) {
            let diff = (a - b).norm();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        if max_diff < 1e-10 {
            println!("  [OK] {} - states match (max diff: {:.2e})", name, max_diff);
        } else {
            println!("  [FAIL] {} - states differ (max diff: {:.2e})", name, max_diff);
        }
    }
    println!();
}

fn main() {
    println!("Gate Fusion Benchmark - Realistic Circuit Patterns");
    println!("===================================================\n");

    verify_correctness();

    let num_qubits = 16;
    let iterations = 100;

    println!("Configuration: {} qubits, {} iterations per benchmark\n", num_qubits, iterations);
    println!("{:40} | {:>12} | {:>13} | {}", "Circuit Pattern", "Fusion ON", "Fusion OFF", "Speedup");
    println!("{}", "-".repeat(90));

    // Pattern 1: Frequent CX (typical)
    benchmark_circuit(
        "Frequent CX (typical algorithm)",
        num_qubits,
        iterations,
        |sim| circuit_frequent_cx(sim, num_qubits, 20),
    );

    // Pattern 2: Multiple single-qubit gates before CX
    benchmark_circuit(
        "3 single-qubit gates before CX",
        num_qubits,
        iterations,
        |sim| circuit_multi_single_then_cx(sim, num_qubits, 20),
    );

    // Pattern 3: State preparation (many rotations per qubit)
    benchmark_circuit(
        "State prep (8 gates/qubit, no CX)",
        num_qubits,
        iterations,
        |sim| circuit_state_prep(sim, num_qubits, 4),
    );

    // Pattern 4: Alternating qubits with SZ (accumulates)
    benchmark_circuit(
        "Alternating SZ gates (accumulates)",
        num_qubits,
        iterations,
        |sim| circuit_alternating(sim, num_qubits, 200),
    );

    // Pattern 5: True worst case - one gate per qubit then flush
    benchmark_circuit(
        "One H per qubit, then flush (worst)",
        num_qubits,
        iterations,
        |sim| circuit_single_gate_per_qubit(sim, num_qubits, 20),
    );

    println!("\n{}", "=".repeat(90));
    println!("\nAnalysis:");
    println!("- Speedup > 1.0 means fusion helps");
    println!("- Speedup < 1.0 means fusion overhead hurts");
    println!("- Speedup ~ 1.0 means no significant difference");

    println!("\nKey findings:");
    println!("- Fusion helps most when multiple gates target the same qubit");
    println!("- Fusion has overhead that may hurt when gates are spread out");
    println!("- Frequent CX gates flush the queue, reducing fusion benefit");
}
