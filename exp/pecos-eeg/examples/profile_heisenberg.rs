//! Profile the Heisenberg DEM build.
//! Usage: cargo run -p pecos-eeg --example `profile_heisenberg` --profile profiling
//! Perf:  perf record -g -F 4999 -- `target/profiling/examples/profile_heisenberg`

use pecos_core::gate_type::GateType;
use pecos_core::pauli::pauli_bitmask::BitmaskStorage;
use pecos_core::{Gate, GateAngles, GateParams, QubitId};
use pecos_eeg::Bm;
use std::time::Instant;

fn gate(gt: GateType, qubits: &[usize]) -> Gate {
    Gate {
        gate_type: gt,
        qubits: qubits.iter().map(|&q| QubitId(q)).collect(),
        angles: GateAngles::new(),
        params: GateParams::new(),
        meas_ids: pecos_core::GateMeasIds::new(),
        channel: None,
    }
}

/// Build a single weight-4 X-check plaquette with 4 data + 1 ancilla, N rounds.
/// This is the hotspot structure in surface codes.
fn build_weight4_circuit(num_rounds: usize) -> Vec<Gate> {
    // Data: 0,1,2,3. Ancilla: 4.
    let mut gates = Vec::new();
    for q in 0..5 {
        gates.push(gate(GateType::PZ, &[q]));
    }
    for round in 0..num_rounds {
        gates.push(gate(GateType::H, &[4]));
        gates.push(gate(GateType::CX, &[4, 0]));
        gates.push(gate(GateType::CX, &[4, 1]));
        gates.push(gate(GateType::CX, &[4, 2]));
        gates.push(gate(GateType::CX, &[4, 3]));
        gates.push(gate(GateType::H, &[4]));
        gates.push(gate(GateType::MZ, &[4]));
        if round < num_rounds - 1 {
            gates.push(gate(GateType::PZ, &[4]));
        }
    }
    for q in 0..4 {
        gates.push(gate(GateType::MZ, &[q]));
    }
    gates
}

fn main() {
    let num_rounds = 8;
    let gates = build_weight4_circuit(num_rounds);
    let noise = pecos_eeg::noise::UniformNoise::coherent_only(0.05);
    let expanded = pecos_eeg::expand::expand_circuit(&gates);
    let na = 1; // one ancilla

    let mut detectors = Vec::new();
    for round in 0..(num_rounds - 1) {
        let m1 = round * na;
        let m2 = (round + 1) * na;
        let mut det = Bm::default();
        det.z_bits.set_bit(expanded.measurement_qubit[m1]);
        det.z_bits.set_bit(expanded.measurement_qubit[m2]);
        detectors.push(det);
    }

    let init_gates: Vec<Gate> = (0..5)
        .map(|q| pecos_eeg::expand::make_gate(GateType::PZ, &[q]))
        .collect();
    let stab =
        pecos_eeg::stabilizer::StabilizerGroup::from_circuit(&init_gates, expanded.num_qubits);

    eprintln!(
        "Weight-4 X-check, {num_rounds} rounds, {} expanded qubits, {} detectors",
        expanded.num_qubits,
        detectors.len()
    );

    // Run 20 iterations to get enough samples for perf
    let iters = 20;
    let t = Instant::now();
    for _ in 0..iters {
        for det in &detectors {
            let _p = pecos_eeg::heisenberg::heisenberg_detection_probability(
                &expanded.gates,
                det,
                &noise,
                &stab,
                0.0,
            );
        }
    }
    let total = t.elapsed();
    let calls = u32::try_from(detectors.len() * iters).expect("profile call count fits in u32");
    let per_det = total.as_secs_f64() * 1000.0 / f64::from(calls);
    eprintln!(
        "{iters} iterations x {} dets = {} calls in {:.2}s ({:.2}ms/det)",
        detectors.len(),
        detectors.len() * iters,
        total.as_secs_f64(),
        per_det
    );
}
