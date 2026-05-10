// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! Ground-truth comparison: EEG analytical DEM vs `StateVec` simulation.
//!
//! Uses Bell-state parity circuit where idle RZ noise creates detectable
//! parity violations. Without noise, MZ parity is always even.
//! With RZ(θ) noise after CX, P(odd parity) = sin²(θ).
//! EEG predicts ≈ θ² at leading order.

use pecos_core::gate_type::GateType;
use pecos_core::pauli::pauli_bitmask::BitmaskStorage;
use pecos_core::{Angle64, Gate, GateAngles, GateParams, QubitId};
use pecos_eeg::Bm;
use pecos_eeg::circuit::{NoiseModel, analyze_expanded};
use pecos_eeg::dem_mapping::{Detector, build_dem_with_stabilizers};
use pecos_eeg::expand;
use pecos_eeg::noise::UniformNoise;
use pecos_eeg::stabilizer::StabilizerGroup;
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StateVec};

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

fn qid(q: usize) -> QubitId {
    QubitId(q)
}

fn u64_to_f64(value: u64) -> f64 {
    f64::from(u32::try_from(value).expect("test sample count fits in u32"))
}

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("test dimension fits in u32"))
}

fn bit_to_f64(value: usize) -> f64 {
    f64::from(u8::try_from(value).expect("bit value fits in u8"))
}

// ============================================================
// Shared helpers for EEG analysis and StateVec simulation
// ============================================================

/// Run EEG on a gate list with a parity detector over all measurements.
fn eeg_detection_prob(gates: &[Gate], theta: f64) -> f64 {
    let expanded = expand::expand_circuit(gates);
    let noise = NoiseModel::coherent_only(theta);
    let result = analyze_expanded(&expanded.gates, &noise);

    // Parity detector: Z on all auxiliary qubits
    let mut det_stab = Bm::default();
    for &aux in &expanded.measurement_qubit {
        det_stab.z_bits.set_bit(aux);
    }
    let det = Detector {
        id: 0,
        stabilizer: det_stab,
    };

    // Stabilizer group from expanded circuit (strip trailing deferred MZ)
    let exp_pre: Vec<_> = {
        let last = expanded
            .gates
            .iter()
            .rposition(|g| g.gate_type != GateType::MZ)
            .unwrap();
        expanded.gates[..=last].to_vec()
    };
    let stab_group = StabilizerGroup::from_circuit(&exp_pre, expanded.num_qubits);

    let entries = build_dem_with_stabilizers(&result.generators, &[det], &[], Some(&stab_group));

    entries.iter().map(|e| e.probability).sum()
}

/// Run EEG with per-round detectors, return per-detector probabilities.
fn eeg_per_round_probs(gates: &[Gate], theta: f64, num_rounds: usize) -> Vec<f64> {
    let expanded = expand::expand_circuit(gates);
    let noise = NoiseModel::coherent_only(theta);
    let result = analyze_expanded(&expanded.gates, &noise);

    // One detector per round (Z on that round's aux qubit)
    let dets: Vec<Detector> = (0..num_rounds)
        .map(|r| {
            let aux = expanded.measurement_qubit[r];
            Detector {
                id: r,
                stabilizer: Bm::z(aux),
            }
        })
        .collect();

    // Stabilizer group from expanded circuit (strip trailing deferred MZ)
    let exp_pre: Vec<_> = {
        let last = expanded
            .gates
            .iter()
            .rposition(|g| g.gate_type != GateType::MZ)
            .unwrap();
        expanded.gates[..=last].to_vec()
    };
    let stab_group = StabilizerGroup::from_circuit(&exp_pre, expanded.num_qubits);

    let entries = build_dem_with_stabilizers(&result.generators, &dets, &[], Some(&stab_group));

    let mut probs = vec![0.0; num_rounds];
    for e in &entries {
        for &d in &e.event.detectors {
            probs[d] += e.probability;
        }
    }
    probs
}

/// Bell-state parity circuit:
///   PZ(0,1), H(0), CX(0,1), [idle RZ], H(0), H(1), MZ(0), MZ(1)
///
/// Without noise: |Φ+> → H⊗H → |Φ+> → MZ parity always even.
/// With RZ(θ) on both qubits: P(odd parity) = sin²(θ) ≈ θ² at leading order.
#[test]
fn test_eeg_vs_statevec_bell_parity() {
    let theta = 0.05;
    let num_shots = 100_000;

    // --- EEG analytical path ---
    let gates = vec![
        gate(GateType::PZ, &[0]),
        gate(GateType::PZ, &[1]),
        gate(GateType::H, &[0]),
        gate(GateType::CX, &[0, 1]),
        // idle RZ(θ) on both qubits is implicit in noise model
        gate(GateType::H, &[0]),
        gate(GateType::H, &[1]),
        gate(GateType::MZ, &[0]),
        gate(GateType::MZ, &[1]),
    ];

    let expanded = expand::expand_circuit(&gates);
    let noise = NoiseModel::coherent_only(theta);
    let result = analyze_expanded(&expanded.gates, &noise);

    // Parity detector: Z_aux0 * Z_aux1
    assert_eq!(expanded.measurement_qubit.len(), 2);
    let aux0 = expanded.measurement_qubit[0];
    let aux1 = expanded.measurement_qubit[1];
    let mut det_stab = Bm::default();
    det_stab.z_bits.set_bit(aux0);
    det_stab.z_bits.set_bit(aux1);
    let det = Detector {
        id: 0,
        stabilizer: det_stab,
    };

    // Pre-readout stabilizer group (exclude final MZ gates)
    // Stabilizer group from expanded circuit (strip trailing deferred MZ)
    let exp_pre: Vec<_> = {
        let last = expanded
            .gates
            .iter()
            .rposition(|g| g.gate_type != GateType::MZ)
            .unwrap();
        expanded.gates[..=last].to_vec()
    };
    let stab_group = StabilizerGroup::from_circuit(&exp_pre, expanded.num_qubits);

    let entries = build_dem_with_stabilizers(&result.generators, &[det], &[], Some(&stab_group));

    let eeg_prob: f64 = entries.iter().map(|e| e.probability).sum();

    // --- StateVec simulation path ---
    let mut odd_parity_count = 0u64;
    let mut sim = StateVec::new(2);

    for _ in 0..num_shots {
        sim.pz(&[qid(0), qid(1)]);
        sim.h(&[qid(0)]);
        sim.cx(&[(qid(0), qid(1))]);

        // Idle RZ noise (same as EEG coherent_only model)
        sim.rz(Angle64::from_radians(theta), &[qid(0)]);
        sim.rz(Angle64::from_radians(theta), &[qid(1)]);

        sim.h(&[qid(0)]);
        sim.h(&[qid(1)]);

        let r = sim.mz(&[qid(0), qid(1)]);
        if r[0].outcome != r[1].outcome {
            odd_parity_count += 1;
        }
    }

    let sv_rate = u64_to_f64(odd_parity_count) / f64::from(num_shots);
    let sv_stderr = (sv_rate * (1.0 - sv_rate) / f64::from(num_shots)).sqrt();
    let exact = theta.sin().powi(2);

    eprintln!("theta = {theta}");
    eprintln!("EEG:     {eeg_prob:.6}");
    eprintln!("StateVec: {sv_rate:.6} +/- {sv_stderr:.6}");
    eprintln!("Exact:   {exact:.6}");

    // EEG should match StateVec within statistical noise + perturbative error.
    // At θ=0.05: exact=sin²(0.05)≈0.002499, EEG leading-order≈θ²≈0.0025
    // Perturbative error is O(θ⁴) ≈ 6.25e-6, well within statistical noise.
    let diff = (eeg_prob - sv_rate).abs();
    let tolerance = 5.0 * sv_stderr + theta.powi(4); // 5σ + perturbative bound
    assert!(
        diff < tolerance,
        "EEG ({eeg_prob:.6}) vs StateVec ({sv_rate:.6}): diff={diff:.6} > tol={tolerance:.6}"
    );
}

/// Same comparison at larger angle to verify scaling.
#[test]
fn test_eeg_vs_statevec_larger_angle() {
    let theta = 0.1;
    let num_shots = 100_000;

    let gates = vec![
        gate(GateType::PZ, &[0]),
        gate(GateType::PZ, &[1]),
        gate(GateType::H, &[0]),
        gate(GateType::CX, &[0, 1]),
        gate(GateType::H, &[0]),
        gate(GateType::H, &[1]),
        gate(GateType::MZ, &[0]),
        gate(GateType::MZ, &[1]),
    ];

    let expanded = expand::expand_circuit(&gates);
    let noise = NoiseModel::coherent_only(theta);
    let result = analyze_expanded(&expanded.gates, &noise);

    let aux0 = expanded.measurement_qubit[0];
    let aux1 = expanded.measurement_qubit[1];
    let mut det_stab = Bm::default();
    det_stab.z_bits.set_bit(aux0);
    det_stab.z_bits.set_bit(aux1);
    let det = Detector {
        id: 0,
        stabilizer: det_stab,
    };

    // Stabilizer group from expanded circuit (strip trailing deferred MZ)
    let exp_pre: Vec<_> = {
        let last = expanded
            .gates
            .iter()
            .rposition(|g| g.gate_type != GateType::MZ)
            .unwrap();
        expanded.gates[..=last].to_vec()
    };
    let stab_group = StabilizerGroup::from_circuit(&exp_pre, expanded.num_qubits);

    let entries = build_dem_with_stabilizers(&result.generators, &[det], &[], Some(&stab_group));

    let eeg_prob: f64 = entries.iter().map(|e| e.probability).sum();

    let mut odd_parity_count = 0u64;
    let mut sim = StateVec::new(2);

    for _ in 0..num_shots {
        sim.pz(&[qid(0), qid(1)]);
        sim.h(&[qid(0)]);
        sim.cx(&[(qid(0), qid(1))]);
        sim.rz(Angle64::from_radians(theta), &[qid(0)]);
        sim.rz(Angle64::from_radians(theta), &[qid(1)]);
        sim.h(&[qid(0)]);
        sim.h(&[qid(1)]);
        let r = sim.mz(&[qid(0), qid(1)]);
        if r[0].outcome != r[1].outcome {
            odd_parity_count += 1;
        }
    }

    let sv_rate = u64_to_f64(odd_parity_count) / f64::from(num_shots);
    let sv_stderr = (sv_rate * (1.0 - sv_rate) / f64::from(num_shots)).sqrt();
    let exact = theta.sin().powi(2);

    eprintln!("theta = {theta}");
    eprintln!("EEG:      {eeg_prob:.6}");
    eprintln!("StateVec: {sv_rate:.6} +/- {sv_stderr:.6}");
    eprintln!("Exact:    {exact:.6}");

    // At θ=0.1: exact≈0.00998, EEG≈θ²≈0.01, perturbative error≈O(θ⁴)≈0.0001
    // Allow larger tolerance for bigger angle
    let diff = (eeg_prob - sv_rate).abs();
    let tolerance = 5.0 * sv_stderr + 2.0 * theta.powi(4);
    assert!(
        diff < tolerance,
        "EEG ({eeg_prob:.6}) vs StateVec ({sv_rate:.6}): diff={diff:.6} > tol={tolerance:.6}"
    );
}

// ============================================================
// Benchmark sweeps (run with: cargo test -p pecos-eeg --test statevec_comparison -- --ignored --nocapture)
// ============================================================

/// Sweep theta for the Bell parity circuit.
/// Exact answer: sin²(θ). EEG leading-order: θ².
#[test]
#[ignore = "benchmark sweep; run manually with --ignored --nocapture"]
fn bench_bell_parity_theta_sweep() {
    let num_shots = 200_000;

    let gates = vec![
        gate(GateType::PZ, &[0]),
        gate(GateType::PZ, &[1]),
        gate(GateType::H, &[0]),
        gate(GateType::CX, &[0, 1]),
        gate(GateType::H, &[0]),
        gate(GateType::H, &[1]),
        gate(GateType::MZ, &[0]),
        gate(GateType::MZ, &[1]),
    ];

    eprintln!("\n=== Bell parity: EEG vs StateVec vs Exact ===");
    eprintln!(
        "{:>8} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "theta", "EEG", "StateVec", "SV_stderr", "Exact", "EEG/Exact"
    );

    for &theta in &[0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5] {
        let eeg_prob = eeg_detection_prob(&gates, theta);

        let mut odd = 0u64;
        let mut sim = StateVec::new(2);
        for _ in 0..num_shots {
            sim.pz(&[qid(0), qid(1)]);
            sim.h(&[qid(0)]);
            sim.cx(&[(qid(0), qid(1))]);
            sim.rz(Angle64::from_radians(theta), &[qid(0)]);
            sim.rz(Angle64::from_radians(theta), &[qid(1)]);
            sim.h(&[qid(0)]);
            sim.h(&[qid(1)]);
            let r = sim.mz(&[qid(0), qid(1)]);
            if r[0].outcome != r[1].outcome {
                odd += 1;
            }
        }

        let sv = u64_to_f64(odd) / f64::from(num_shots);
        let se = (sv * (1.0 - sv) / f64::from(num_shots)).sqrt();
        let exact = theta.sin().powi(2);
        let ratio = if exact > 1e-10 {
            eeg_prob / exact
        } else {
            f64::NAN
        };

        eprintln!(
            "{theta:>8.3} {eeg_prob:>10.6} {sv:>10.6} {se:>10.6} {exact:>10.6} {ratio:>10.4}"
        );
    }
}

/// Multi-round X-check: 2 data qubits, 1 ancilla, N rounds of X-check with reset.
/// Data prepared in |++>, ancilla measures X0*X1 each round.
#[test]
#[ignore = "benchmark sweep; run manually with --ignored --nocapture"]
fn bench_x_check_multi_round() {
    let num_shots = 200_000;

    eprintln!("\n=== Multi-round X-check (2 data + 1 ancilla) ===");

    for &num_rounds in &[1, 2, 3, 4] {
        // Build circuit: PZ(0,1,2), H(0), H(1), then N rounds of X-check
        let mut gates = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::PZ, &[1]),
            gate(GateType::PZ, &[2]),
            gate(GateType::H, &[0]),
            gate(GateType::H, &[1]),
        ];

        for _ in 0..num_rounds {
            gates.push(gate(GateType::H, &[2]));
            gates.push(gate(GateType::CX, &[2, 0]));
            gates.push(gate(GateType::CX, &[2, 1]));
            gates.push(gate(GateType::H, &[2]));
            gates.push(gate(GateType::MZ, &[2]));
            gates.push(gate(GateType::PZ, &[2]));
        }
        // Remove trailing PZ (no reset after last round)
        gates.pop();

        for &theta in &[0.01, 0.05, 0.1] {
            let eeg_probs = eeg_per_round_probs(&gates, theta, num_rounds);

            // StateVec: run circuit with idle RZ after each CX
            let mut round_detections = vec![0u64; num_rounds];
            let mut sim = StateVec::new(3);

            for _ in 0..num_shots {
                sim.pz(&[qid(0), qid(1), qid(2)]);
                sim.h(&[qid(0)]);
                sim.h(&[qid(1)]);

                for (round, round_detection) in
                    round_detections.iter_mut().enumerate().take(num_rounds)
                {
                    sim.h(&[qid(2)]);
                    sim.cx(&[(qid(2), qid(0))]);
                    sim.rz(Angle64::from_radians(theta), &[qid(0)]);
                    sim.rz(Angle64::from_radians(theta), &[qid(2)]);
                    sim.cx(&[(qid(2), qid(1))]);
                    sim.rz(Angle64::from_radians(theta), &[qid(1)]);
                    sim.rz(Angle64::from_radians(theta), &[qid(2)]);
                    sim.h(&[qid(2)]);
                    let r = sim.mz(&[qid(2)]);
                    if r[0].outcome {
                        *round_detection += 1;
                    }
                    if round < num_rounds - 1 {
                        sim.pz(&[qid(2)]);
                    }
                }
            }

            let sv_rates: Vec<f64> = round_detections
                .iter()
                .map(|&d| u64_to_f64(d) / f64::from(num_shots))
                .collect();

            eprintln!("\nrounds={num_rounds}, theta={theta}:");
            for r in 0..num_rounds {
                let se = (sv_rates[r] * (1.0 - sv_rates[r]) / f64::from(num_shots)).sqrt();
                let ratio = if sv_rates[r] > 1e-10 {
                    eeg_probs[r] / sv_rates[r]
                } else {
                    f64::NAN
                };
                eprintln!(
                    "  D{r}: EEG={:.6} SV={:.6}+/-{:.6} ratio={:.4}",
                    eeg_probs[r], sv_rates[r], se, ratio
                );
            }
        }
    }
}

/// Z-basis: data in |00>, Z-check measures Z0*Z1. Coherent RZ noise.
/// Z errors commute with Z measurements, so the X-propagated components matter.
#[test]
#[ignore = "benchmark sweep; run manually with --ignored --nocapture"]
fn bench_z_basis_check() {
    let num_shots = 200_000;

    eprintln!("\n=== Z-basis parity check (CX syndrome extraction) ===");
    eprintln!(
        "{:>8} {:>10} {:>10} {:>10} {:>10}",
        "theta", "EEG", "StateVec", "SV_stderr", "EEG/SV"
    );

    // Z-check: CX(0,2), CX(1,2), MZ(2). Ancilla 2 measures Z0*Z1 parity.
    // For |00>: Z0Z1|00> = +|00>, deterministic 0.
    // RZ noise creates Z errors which don't flip Z-checks directly,
    // but the CX propagation can create cross-terms.
    let gates = vec![
        gate(GateType::PZ, &[0]),
        gate(GateType::PZ, &[1]),
        gate(GateType::PZ, &[2]),
        gate(GateType::CX, &[0, 2]),
        gate(GateType::CX, &[1, 2]),
        gate(GateType::MZ, &[2]),
    ];

    for &theta in &[0.01, 0.05, 0.1, 0.2, 0.3] {
        let expanded = expand::expand_circuit(&gates);
        let noise = NoiseModel::coherent_only(theta);
        let result = analyze_expanded(&expanded.gates, &noise);

        let aux = expanded.measurement_qubit[0];
        let det = Detector {
            id: 0,
            stabilizer: Bm::z(aux),
        };
        let gates_pre = &gates[..gates.len() - 1];
        let stab_group = StabilizerGroup::from_circuit(gates_pre, expanded.num_original_qubits);
        let entries =
            build_dem_with_stabilizers(&result.generators, &[det], &[], Some(&stab_group));
        let eeg_prob: f64 = entries.iter().map(|e| e.probability).sum();

        // StateVec
        let mut det_count = 0u64;
        let mut sim = StateVec::new(3);
        for _ in 0..num_shots {
            sim.pz(&[qid(0), qid(1), qid(2)]);
            sim.cx(&[(qid(0), qid(2))]);
            sim.rz(Angle64::from_radians(theta), &[qid(0)]);
            sim.rz(Angle64::from_radians(theta), &[qid(2)]);
            sim.cx(&[(qid(1), qid(2))]);
            sim.rz(Angle64::from_radians(theta), &[qid(1)]);
            sim.rz(Angle64::from_radians(theta), &[qid(2)]);
            let r = sim.mz(&[qid(2)]);
            if r[0].outcome {
                det_count += 1;
            }
        }

        let sv = u64_to_f64(det_count) / f64::from(num_shots);
        let se = (sv * (1.0 - sv) / f64::from(num_shots)).sqrt();
        let ratio = if sv > 1e-10 { eeg_prob / sv } else { f64::NAN };

        eprintln!("{theta:>8.3} {eeg_prob:>10.6} {sv:>10.6} {se:>10.6} {ratio:>10.4}");
    }
}

// ============================================================
// Repetition code comparison: EEG vs Heisenberg vs StateVec
// ============================================================

/// Build an X-check repetition code circuit.
///
/// d data qubits, d-1 ancillas measuring `X_i` * X_{i+1} using
/// H-CX-CX-H on ancilla (sensitive to Z errors from coherent RZ noise).
/// `num_rounds` syndrome extraction rounds with reset.
/// Returns (gates, `num_qubits`, ancilla indices).
fn build_repetition_code(d: usize, num_rounds: usize) -> (Vec<Gate>, usize, Vec<usize>) {
    let num_data = d;
    let num_ancilla = d - 1;
    let num_qubits = num_data + num_ancilla;

    // Ancilla i checks X_{i} * X_{i+1}, located at qubit index d + i
    let ancilla_start = num_data;

    let mut gates = Vec::new();

    // Initialize all qubits
    for q in 0..num_qubits {
        gates.push(gate(GateType::PZ, &[q]));
    }

    for round in 0..num_rounds {
        // X-check: H(anc), CX(anc, data_i), CX(anc, data_{i+1}), H(anc), MZ(anc)
        for i in 0..num_ancilla {
            gates.push(gate(GateType::H, &[ancilla_start + i]));
        }
        for i in 0..num_ancilla {
            let anc = ancilla_start + i;
            gates.push(gate(GateType::CX, &[anc, i]));
        }
        for i in 0..num_ancilla {
            let anc = ancilla_start + i;
            gates.push(gate(GateType::CX, &[anc, i + 1]));
        }
        for i in 0..num_ancilla {
            gates.push(gate(GateType::H, &[ancilla_start + i]));
        }

        // Measure ancillas
        for i in 0..num_ancilla {
            gates.push(gate(GateType::MZ, &[ancilla_start + i]));
        }

        // Reset ancillas (except last round)
        if round < num_rounds - 1 {
            for i in 0..num_ancilla {
                gates.push(gate(GateType::PZ, &[ancilla_start + i]));
            }
        }
    }

    // Final data readout
    for q in 0..num_data {
        gates.push(gate(GateType::MZ, &[q]));
    }

    let ancillas: Vec<usize> = (0..num_ancilla).map(|i| ancilla_start + i).collect();
    (gates, num_qubits, ancillas)
}

/// Repetition code: compare EEG (forward), Heisenberg (backward), and `StateVec`.
#[test]
#[ignore = "benchmark sweep; run manually with --ignored --nocapture"]
fn bench_repetition_code_comparison() {
    use pecos_eeg::dem_mapping::EegConfig;
    use pecos_eeg::heisenberg::heisenberg_detection_probability_from_circuit;

    let num_shots = 500_000;
    let theta = 0.05;

    eprintln!("\n=== Repetition code: EEG vs Heisenberg vs StateVec ===");
    eprintln!("theta = {theta}, shots = {num_shots}");

    for &d in &[3, 5] {
        for &num_rounds in &[2, 3] {
            let (gates, num_qubits, ancillas) = build_repetition_code(d, num_rounds);
            let num_ancilla = ancillas.len();

            // Measurement record layout: round 0 ancillas, round 1 ancillas, ..., data readout
            // Round comparison detectors: meas[round*num_ancilla + i] XOR meas[(round+1)*num_ancilla + i]
            let num_detectors = num_ancilla * (num_rounds - 1);

            eprintln!(
                "\n  d={d}, rounds={num_rounds}, qubits={num_qubits}, detectors={num_detectors}"
            );

            // --- EEG forward ---
            let expanded = expand::expand_circuit(&gates);
            let noise_model = NoiseModel::coherent_only(theta);
            let noise_spec = UniformNoise::coherent_only(theta);
            let result = analyze_expanded(&expanded.gates, &noise_model);

            let mut dets = Vec::new();
            for round in 0..(num_rounds - 1) {
                for i in 0..num_ancilla {
                    let m1 = round * num_ancilla + i;
                    let m2 = (round + 1) * num_ancilla + i;
                    let aux1 = expanded.measurement_qubit[m1];
                    let aux2 = expanded.measurement_qubit[m2];
                    let mut stab = Bm::default();
                    stab.z_bits.set_bit(aux1);
                    stab.z_bits.set_bit(aux2);
                    dets.push(Detector {
                        id: dets.len(),
                        stabilizer: stab,
                    });
                }
            }

            let exp_pre: Vec<_> = {
                let last = expanded
                    .gates
                    .iter()
                    .rposition(|g| g.gate_type != GateType::MZ)
                    .unwrap();
                expanded.gates[..=last].to_vec()
            };
            let stab_group = StabilizerGroup::from_circuit(&exp_pre, expanded.num_qubits);

            let entries = pecos_eeg::dem_mapping::build_dem_configured(
                &result.generators,
                &dets,
                &[],
                Some(&stab_group),
                &EegConfig::default(),
            );

            let mut eeg_probs = vec![0.0; num_detectors];
            for e in &entries {
                for &det_id in &e.event.detectors {
                    if det_id < num_detectors {
                        eeg_probs[det_id] += e.probability;
                    }
                }
            }

            // --- Heisenberg backward ---
            let mut heis_probs = vec![0.0; num_detectors];
            for round in 0..(num_rounds - 1) {
                for i in 0..num_ancilla {
                    let det_idx = round * num_ancilla + i;
                    let m1 = round * num_ancilla + i;
                    let m2 = (round + 1) * num_ancilla + i;
                    heis_probs[det_idx] = heisenberg_detection_probability_from_circuit(
                        &gates,
                        &[m1, m2],
                        &noise_spec,
                        num_qubits,
                        1e-12,
                    );
                }
            }

            // --- StateVec simulation ---
            let mut sv_counts = vec![0u64; num_detectors];
            let mut sim = StateVec::new(num_qubits);

            for _ in 0..num_shots {
                // Initialize
                let all_qubits: Vec<_> = (0..num_qubits).map(qid).collect();
                sim.pz(&all_qubits);

                let mut meas_outcomes = Vec::new();

                for round in 0..num_rounds {
                    // X-check: H(anc), CX(anc, data_i), CX(anc, data_{i+1}), H(anc)
                    let anc_qubits: Vec<_> = ancillas.iter().map(|&a| qid(a)).collect();
                    sim.h(&anc_qubits);

                    for (i, &anc) in ancillas.iter().enumerate().take(num_ancilla) {
                        sim.cx(&[(qid(anc), qid(i))]);
                        sim.rz(Angle64::from_radians(theta), &[qid(anc)]);
                        sim.rz(Angle64::from_radians(theta), &[qid(i)]);
                    }
                    for (i, &anc) in ancillas.iter().enumerate().take(num_ancilla) {
                        sim.cx(&[(qid(anc), qid(i + 1))]);
                        sim.rz(Angle64::from_radians(theta), &[qid(anc)]);
                        sim.rz(Angle64::from_radians(theta), &[qid(i + 1)]);
                    }

                    sim.h(&anc_qubits);

                    // Measure ancillas
                    for &anc in ancillas.iter().take(num_ancilla) {
                        let r = sim.mz(&[qid(anc)]);
                        meas_outcomes.push(r[0].outcome);
                    }

                    // Reset (except last round)
                    if round < num_rounds - 1 {
                        sim.pz(&anc_qubits);
                    }
                }

                // Count detector firings (round comparison)
                for round in 0..(num_rounds - 1) {
                    for i in 0..num_ancilla {
                        let m1 = round * num_ancilla + i;
                        let m2 = (round + 1) * num_ancilla + i;
                        if meas_outcomes[m1] != meas_outcomes[m2] {
                            sv_counts[round * num_ancilla + i] += 1;
                        }
                    }
                }
            }

            // Print comparison
            eprintln!(
                "  {:>6} {:>10} {:>10} {:>10} {:>10} {:>10}",
                "Det", "EEG", "Heisen", "StateVec", "SV_err", "H/SV"
            );
            for det_idx in 0..num_detectors {
                let sv_rate = u64_to_f64(sv_counts[det_idx]) / f64::from(num_shots);
                let sv_err = (sv_rate * (1.0 - sv_rate) / f64::from(num_shots)).sqrt();
                let ratio = if sv_rate > 1e-10 {
                    heis_probs[det_idx] / sv_rate
                } else {
                    f64::NAN
                };
                let round = det_idx / num_ancilla;
                let anc = det_idx % num_ancilla;
                eprintln!(
                    "  R{round}A{anc} {:>10.6} {:>10.6} {:>10.6} {:>10.6} {:>10.4}",
                    eeg_probs[det_idx], heis_probs[det_idx], sv_rate, sv_err, ratio
                );
            }
        }
    }
}

/// KEY DIAGNOSTIC: Compare original-circuit `StateVec`, expanded-circuit `StateVec`,
/// and Heisenberg for the simplest failing case (weight-2, 2 rounds, 3 qubits).
///
/// If expanded SV matches original SV: expansion is correct, Heisenberg has a bug.
/// If expanded SV differs: expansion is wrong.
#[test]
#[ignore = "benchmark sweep; run manually with --ignored --nocapture"]
fn bench_expansion_equivalence() {
    use pecos_eeg::heisenberg::heisenberg_detection_probability_from_circuit;

    let num_shots = 1_000_000;
    let theta = 0.05;

    eprintln!("\n=== Expansion equivalence: 3 qubits, 2 rounds ===");

    // Original circuit with mid-circuit measurement
    let gates_orig = vec![
        gate(GateType::PZ, &[0]),
        gate(GateType::PZ, &[1]),
        gate(GateType::PZ, &[2]),
        gate(GateType::H, &[2]),
        gate(GateType::CX, &[2, 0]),
        gate(GateType::CX, &[2, 1]),
        gate(GateType::H, &[2]),
        gate(GateType::MZ, &[2]),
        gate(GateType::PZ, &[2]),
        gate(GateType::H, &[2]),
        gate(GateType::CX, &[2, 0]),
        gate(GateType::CX, &[2, 1]),
        gate(GateType::H, &[2]),
        gate(GateType::MZ, &[2]),
    ];

    // Expand the circuit
    let expanded = expand::expand_circuit(&gates_orig);
    eprintln!(
        "Expanded: {} gates, {} qubits",
        expanded.gates.len(),
        expanded.num_qubits
    );
    eprintln!("Measurement map: {:?}", expanded.measurement_qubit);
    for (i, g) in expanded.gates.iter().enumerate() {
        let qs: Vec<usize> = g.qubits.iter().map(pecos_core::QubitId::index).collect();
        eprintln!("  [{i:2}] {:?}({qs:?})", g.gate_type);
    }

    // --- Heisenberg on expanded circuit ---
    let noise = UniformNoise::coherent_only(theta);
    let h_p = heisenberg_detection_probability_from_circuit(&gates_orig, &[0, 1], &noise, 3, 0.0);

    // --- StateVec on ORIGINAL circuit (with mid-circuit measurements) ---
    let mut orig_det = 0u64;
    {
        let mut sim = StateVec::new(3);
        for _ in 0..num_shots {
            sim.pz(&[qid(0), qid(1), qid(2)]);
            let mut outs = [false; 2];
            for (r, out) in outs.iter_mut().enumerate() {
                sim.h(&[qid(2)]);
                sim.cx(&[(qid(2), qid(0))]);
                sim.rz(Angle64::from_radians(theta), &[qid(2)]);
                sim.rz(Angle64::from_radians(theta), &[qid(0)]);
                sim.cx(&[(qid(2), qid(1))]);
                sim.rz(Angle64::from_radians(theta), &[qid(1)]);
                sim.rz(Angle64::from_radians(theta), &[qid(2)]);
                sim.h(&[qid(2)]);
                *out = sim.mz(&[qid(2)])[0].outcome;
                if r == 0 {
                    sim.pz(&[qid(2)]);
                }
            }
            if outs[0] != outs[1] {
                orig_det += 1;
            }
        }
    }
    let sv_orig = u64_to_f64(orig_det) / f64::from(num_shots);

    // --- StateVec on EXPANDED circuit (no mid-circuit measurements) ---
    let mut exp_det = 0u64;
    {
        let num_exp_q = expanded.num_qubits;
        let mut sim = StateVec::new(num_exp_q);
        for _ in 0..num_shots {
            // Execute the expanded circuit gate by gate
            let all_q: Vec<_> = (0..num_exp_q).map(qid).collect();
            sim.pz(&all_q);

            for (i, g) in expanded.gates.iter().enumerate() {
                let qs: Vec<usize> = g.qubits.iter().map(pecos_core::QubitId::index).collect();

                // Skip expansion gates for noise (same logic as Heisenberg)
                let is_exp_gate = {
                    let is_qalloc = g.gate_type == pecos_core::gate_type::GateType::QAlloc;
                    let is_exp_cx = i > 0
                        && g.gate_type == pecos_core::gate_type::GateType::CX
                        && expanded.gates[i - 1].gate_type
                            == pecos_core::gate_type::GateType::QAlloc
                        && expanded.gates[i - 1].qubits[0].index()
                            == qs.get(1).copied().unwrap_or(999);
                    let is_exp_pz = i > 1
                        && g.gate_type == pecos_core::gate_type::GateType::PZ
                        && expanded.gates[i - 1].gate_type == pecos_core::gate_type::GateType::CX
                        && expanded.gates[i - 2].gate_type
                            == pecos_core::gate_type::GateType::QAlloc;
                    is_qalloc || is_exp_cx || is_exp_pz
                };

                match g.gate_type {
                    pecos_core::gate_type::GateType::PZ
                    | pecos_core::gate_type::GateType::QAlloc => {
                        for &q in &qs {
                            sim.pz(&[qid(q)]);
                        }
                    }
                    pecos_core::gate_type::GateType::H => {
                        for &q in &qs {
                            sim.h(&[qid(q)]);
                        }
                    }
                    pecos_core::gate_type::GateType::CX if qs.len() >= 2 => {
                        sim.cx(&[(qid(qs[0]), qid(qs[1]))]);
                    }
                    _ => {}
                }

                // Add noise after non-expansion CX gates
                if !is_exp_gate
                    && (g.gate_type == pecos_core::gate_type::GateType::CX)
                    && qs.len() >= 2
                {
                    sim.rz(Angle64::from_radians(theta), &[qid(qs[0])]);
                    sim.rz(Angle64::from_radians(theta), &[qid(qs[1])]);
                }
            }

            // Measure the two aux qubits
            let aux0 = expanded.measurement_qubit[0];
            let aux1 = expanded.measurement_qubit[1];
            let r0 = sim.mz(&[qid(aux0)])[0].outcome;
            let r1 = sim.mz(&[qid(aux1)])[0].outcome;
            if r0 != r1 {
                exp_det += 1;
            }
        }
    }
    let sv_exp = u64_to_f64(exp_det) / f64::from(num_shots);

    // Exact analytical
    let exact = (2.0 - (6.0 * theta).cos() - (2.0 * theta).cos()) / 4.0;

    let se_orig = (sv_orig * (1.0 - sv_orig) / f64::from(num_shots)).sqrt();
    let se_exp = (sv_exp * (1.0 - sv_exp) / f64::from(num_shots)).sqrt();

    eprintln!("\nResults:");
    eprintln!("  Exact analytical:    {exact:.6}");
    eprintln!("  SV original circuit: {sv_orig:.6} +/- {se_orig:.6}");
    eprintln!("  SV expanded circuit: {sv_exp:.6} +/- {se_exp:.6}");
    eprintln!("  Heisenberg:          {h_p:.6}");
    eprintln!("  H/Exact = {:.4}", h_p / exact);
    eprintln!("  SVexp/SVorig = {:.4}", sv_exp / sv_orig);
}

/// Ground truth: compute the backward Heisenberg via DIRECT MATRIX MULTIPLICATION
/// on the expanded circuit. This bypasses the Pauli-tracking backward walk entirely.
///
/// The detection probability is:
///   p = (1 - <0...0| `O_backward` |0...0>) / 2
/// where `O_backward` = `E_1`† ... `E_n†(D)`
///
/// We compute `O_backward` as a 2^n × 2^n matrix by multiplying the adjoint
/// of each gate/noise channel, then evaluate the diagonal element.
#[test]
#[ignore = "benchmark sweep; run manually with --ignored --nocapture"]
fn bench_matrix_heisenberg() {
    use pecos_eeg::heisenberg::heisenberg_detection_probability_from_circuit;

    let theta = 0.05;
    let n = 5; // qubits: q0,q1 data, q2 ancilla, q3 aux R1, q4 aux R2
    let dim = 1 << n; // 32

    // The expanded circuit gates (from bench_expansion_equivalence)
    let gates_orig = vec![
        gate(GateType::PZ, &[0]),
        gate(GateType::PZ, &[1]),
        gate(GateType::PZ, &[2]),
        gate(GateType::H, &[2]),
        gate(GateType::CX, &[2, 0]),
        gate(GateType::CX, &[2, 1]),
        gate(GateType::H, &[2]),
        gate(GateType::MZ, &[2]),
        gate(GateType::PZ, &[2]),
        gate(GateType::H, &[2]),
        gate(GateType::CX, &[2, 0]),
        gate(GateType::CX, &[2, 1]),
        gate(GateType::H, &[2]),
        gate(GateType::MZ, &[2]),
    ];
    let expanded = expand::expand_circuit(&gates_orig);

    // Build the detector matrix: Z_3 * Z_4
    // Z_q has eigenvalue +1 for |0> and -1 for |1>
    let mut obs = vec![0.0f64; dim * dim]; // real part (obs is Hermitian diagonal for Pauli Z)
    for i in 0..dim {
        let bit3 = (i >> 3) & 1;
        let bit4 = (i >> 4) & 1;
        let z3 = if bit3 == 0 { 1.0 } else { -1.0 };
        let z4 = if bit4 == 0 { 1.0 } else { -1.0 };
        obs[i * dim + i] = z3 * z4;
    }

    // Now apply the adjoint of each gate/noise channel in REVERSE order.
    // For unitary U: O → U† O U
    // For PZ_q (reset): O → <0_q| O |0_q> tensored with I_q (extract q=0 block)
    // For noise RZ(θ): O → RZ†(θ) O RZ(θ) (unitary conjugation)
    // For MZ_q: O → project to Z eigenstates (diagonal on q)

    // Helper: build a gate matrix for the full n-qubit space
    // We work with real+imaginary pairs: obs_re[i*dim+j], obs_im[i*dim+j]
    let mut obs_re = vec![0.0f64; dim * dim];
    let mut obs_im = vec![0.0f64; dim * dim];
    for i in 0..dim {
        let bit3 = (i >> 3) & 1;
        let bit4 = (i >> 4) & 1;
        obs_re[i * dim + i] = if bit3 == bit4 { 1.0 } else { -1.0 };
    }

    // Process gates in reverse order
    let exp_gates_set = {
        let mut s = std::collections::HashSet::new();
        // Detect expansion gates (same logic as heisenberg.rs)
        for i in 1..expanded.gates.len() {
            if expanded.gates[i].gate_type == pecos_core::gate_type::GateType::QAlloc {
                s.insert(i);
            }
            if expanded.gates[i].gate_type == pecos_core::gate_type::GateType::CX
                && expanded.gates[i - 1].gate_type == pecos_core::gate_type::GateType::QAlloc
            {
                let aq = expanded.gates[i - 1].qubits[0].index();
                if expanded.gates[i].qubits.len() >= 2 && expanded.gates[i].qubits[1].index() == aq
                {
                    s.insert(i);
                    if i + 1 < expanded.gates.len()
                        && expanded.gates[i + 1].gate_type == pecos_core::gate_type::GateType::PZ
                        && expanded.gates[i + 1].qubits[0].index()
                            == expanded.gates[i].qubits[0].index()
                    {
                        s.insert(i + 1);
                    }
                }
            }
        }
        s
    };

    for idx in (0..expanded.gates.len()).rev() {
        let g = &expanded.gates[idx];
        let qs: Vec<usize> = g.qubits.iter().map(pecos_core::QubitId::index).collect();

        // Apply noise adjoint (if not expansion gate)
        if !exp_gates_set.contains(&idx) && g.gate_type == pecos_core::gate_type::GateType::CX {
            // idle_rz on both qubits
            for &q in &qs {
                apply_rz_adjoint(&mut obs_re, &mut obs_im, q, theta, n);
            }
        }

        // Apply gate adjoint
        match g.gate_type {
            pecos_core::gate_type::GateType::PZ | pecos_core::gate_type::GateType::QAlloc => {
                // PZ†(O) = <0_q| O |0_q> ⊗ I_q
                // This zeros all matrix elements where q is in state |1>
                // and copies q=0 block to the full matrix
                apply_pz_adjoint(&mut obs_re, &mut obs_im, qs[0], n);
            }
            pecos_core::gate_type::GateType::MZ => {
                // MZ†(O) = Σ_m |m><m| O |m><m| (decohere in Z basis)
                apply_mz_adjoint(&mut obs_re, &mut obs_im, qs[0], n);
            }
            pecos_core::gate_type::GateType::H => {
                apply_h_adjoint(&mut obs_re, &mut obs_im, qs[0], n);
            }
            pecos_core::gate_type::GateType::CX => {
                apply_cx_adjoint(&mut obs_re, &mut obs_im, qs[0], qs[1], n);
            }
            _ => {}
        }
    }

    // Evaluate <0...0| O_backward |0...0>
    let expectation = obs_re[0]; // |0...0> is index 0
    let matrix_detection = 0.5 * (1.0 - expectation);

    // Step-by-step matrix trace: print <0|O|0> after each gate
    // Re-run with printing
    {
        let mut tr_re = vec![0.0f64; dim * dim];
        let mut tr_im = vec![0.0f64; dim * dim];
        for i in 0..dim {
            let bit3 = (i >> 3) & 1;
            let bit4 = (i >> 4) & 1;
            tr_re[i * dim + i] = if bit3 == bit4 { 1.0 } else { -1.0 };
        }

        eprintln!("\n  Step-by-step <0|O|0> comparison (matrix vs walk):");
        eprintln!(
            "  {:>4} {:>20} {:>12}",
            "Gate", "Description", "Matrix<0|O|0>"
        );

        for idx in (0..expanded.gates.len()).rev() {
            let g = &expanded.gates[idx];
            let qs: Vec<usize> = g.qubits.iter().map(pecos_core::QubitId::index).collect();
            let is_exp = exp_gates_set.contains(&idx);

            if !is_exp && g.gate_type == pecos_core::gate_type::GateType::CX && qs.len() >= 2 {
                for &q in &qs {
                    apply_rz_adjoint(&mut tr_re, &mut tr_im, q, theta, n);
                }
            }

            match g.gate_type {
                pecos_core::gate_type::GateType::PZ | pecos_core::gate_type::GateType::QAlloc => {
                    apply_pz_adjoint(&mut tr_re, &mut tr_im, qs[0], n);
                }
                pecos_core::gate_type::GateType::MZ => {
                    apply_mz_adjoint(&mut tr_re, &mut tr_im, qs[0], n);
                }
                pecos_core::gate_type::GateType::H => {
                    apply_h_adjoint(&mut tr_re, &mut tr_im, qs[0], n);
                }
                pecos_core::gate_type::GateType::CX => {
                    apply_cx_adjoint(&mut tr_re, &mut tr_im, qs[0], qs[1], n);
                }
                _ => {}
            }

            let e = tr_re[0];
            let tag = if is_exp { " [EXP]" } else { "" };
            eprintln!("  [{idx:>2}] {:?}({qs:?}){tag}: {e:.10}", g.gate_type);
        }
    }

    // Heisenberg backward walk
    let noise = UniformNoise::coherent_only(theta);
    let h_p = heisenberg_detection_probability_from_circuit(&gates_orig, &[0, 1], &noise, 3, 0.0);

    // Exact analytical
    let exact = (2.0 - (6.0 * theta).cos() - (2.0 * theta).cos()) / 4.0;

    eprintln!("\n=== Matrix Heisenberg ground truth ===");
    eprintln!("  Exact analytical:     {exact:.10}");
    eprintln!("  Matrix Heisenberg:    {matrix_detection:.10}");
    eprintln!("  Backward walk:        {h_p:.10}");
    eprintln!("  Matrix/Exact:         {:.6}", matrix_detection / exact);
    eprintln!("  Walk/Matrix:          {:.6}", h_p / matrix_detection);
}

// Matrix helpers for n-qubit system
fn apply_rz_adjoint(re: &mut [f64], im: &mut [f64], q: usize, theta: f64, n: usize) {
    let dim = 1 << n;
    // For each matrix element O[i,j]:
    // New O[i,j] = e^{i(b_i - b_j)θ/2} · O[i,j]
    // where b_i = bit q of i (0 → phase -θ/2, 1 → phase +θ/2)
    for i in 0..dim {
        let bi = bit_to_f64((i >> q) & 1); // 0 or 1
        for j in 0..dim {
            let bj = bit_to_f64((j >> q) & 1);
            let phase = (bi - bj) * theta; // phase angle
            if phase.abs() < 1e-20 {
                continue;
            }
            let cp = phase.cos();
            let sp = phase.sin();
            let idx = i * dim + j;
            let r = re[idx];
            let m = im[idx];
            re[idx] = cp * r - sp * m;
            im[idx] = sp * r + cp * m;
        }
    }
}

fn apply_pz_adjoint(re: &mut [f64], im: &mut [f64], q: usize, n: usize) {
    // PZ†(O) = Σ_m |m⟩⟨0| O |0⟩⟨m| where K_m = |0⟩⟨m|
    // Matrix elements: [PZ†(O)]_{ij} = δ(i_q, j_q) · O_{(i with q=0), (j with q=0)}
    // Off-diagonal elements (where qubit q differs) are ZERO.
    let dim = 1 << n;
    let mask = 1 << q;
    for i in 0..dim {
        let iq = (i >> q) & 1;
        for j in 0..dim {
            let jq = (j >> q) & 1;
            let idx = i * dim + j;
            if iq == jq {
                let i0 = i & !mask;
                let j0 = j & !mask;
                let idx0 = i0 * dim + j0;
                re[idx] = re[idx0];
                im[idx] = im[idx0];
            } else {
                re[idx] = 0.0;
                im[idx] = 0.0;
            }
        }
    }
}

fn apply_mz_adjoint(re: &mut [f64], im: &mut [f64], q: usize, n: usize) {
    let dim = 1 << n;
    for i in 0..dim {
        let bi = (i >> q) & 1;
        for j in 0..dim {
            let bj = (j >> q) & 1;
            if bi != bj {
                let idx = i * dim + j;
                re[idx] = 0.0;
                im[idx] = 0.0;
            }
        }
    }
}

fn apply_h_adjoint(re: &mut [f64], im: &mut [f64], q: usize, n: usize) {
    // H† O H = H O H (H is self-adjoint)
    // H|0> = (|0>+|1>)/√2, H|1> = (|0>-|1>)/√2
    let dim = 1 << n;
    let mask = 1 << q;
    // For each pair (i, i^mask), apply the 2x2 Hadamard conjugation
    // O' = H ⊗ I · O · H ⊗ I
    // This swaps/combines rows and columns corresponding to bit q
    let mut new_re = vec![0.0; dim * dim];
    let mut new_im = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            // new O[i,j] = Σ_{a,b} H[i_q,a] O[i_with_a, j_with_b] H[b, j_q]
            let i0 = i & !mask;
            let i1 = i | mask;
            let j0 = j & !mask;
            let j1 = j | mask;
            let iq = (i >> q) & 1;
            let jq = (j >> q) & 1;
            // H[0,0]=1/√2, H[0,1]=1/√2, H[1,0]=1/√2, H[1,1]=-1/√2
            // H[x,y] = (1/√2)(-1)^{xy}
            // H[iq,a]*H[b,jq] = (1/2)(-1)^{iq*a+b*jq}
            let mut sum_r = 0.0;
            let mut sum_i = 0.0;
            for a in 0..2usize {
                for b in 0..2usize {
                    let ia = if a == 0 { i0 } else { i1 };
                    let jb = if b == 0 { j0 } else { j1 };
                    let idx = ia * dim + jb;
                    let sign = if (iq * a + b * jq).is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    };
                    let c = 0.5 * sign;
                    sum_r += c * re[idx];
                    sum_i += c * im[idx];
                }
            }
            new_re[i * dim + j] = sum_r;
            new_im[i * dim + j] = sum_i;
        }
    }
    re.copy_from_slice(&new_re);
    im.copy_from_slice(&new_im);
}

fn apply_cx_adjoint(re: &mut [f64], im: &mut [f64], control: usize, target: usize, n: usize) {
    // CX† O CX = CX O CX (CX is self-adjoint)
    // CX flips target when control=1: CX|c,t> = |c, c⊕t>
    let dim = 1 << n;
    let cmask = 1 << control;
    let tmask = 1 << target;
    // CX permutation: state index i maps to i ^ (tmask if control bit set)
    let cx_perm = |i: usize| -> usize { if (i & cmask) != 0 { i ^ tmask } else { i } };
    // O' = CX · O · CX: new O[i,j] = O[CX(i), CX(j)]
    let mut new_re = vec![0.0; dim * dim];
    let mut new_im = vec![0.0; dim * dim];
    for i in 0..dim {
        let ci = cx_perm(i);
        for j in 0..dim {
            let cj = cx_perm(j);
            new_re[i * dim + j] = re[ci * dim + cj];
            new_im[i * dim + j] = im[ci * dim + cj];
        }
    }
    re.copy_from_slice(&new_re);
    im.copy_from_slice(&new_im);
}

/// Per-noise-source attribution: which noise sources does the backward walk miss?
///
/// Enable one noise source at a time and compare matrix vs backward walk.
#[test]
#[ignore = "benchmark sweep; run manually with --ignored --nocapture"]
fn bench_per_noise_attribution() {
    use pecos_eeg::heisenberg::heisenberg_detection_probability_from_circuit;

    let theta = 0.05;
    let n = 5;
    let dim = 1 << n;

    let gates_orig = vec![
        gate(GateType::PZ, &[0]),
        gate(GateType::PZ, &[1]),
        gate(GateType::PZ, &[2]),
        gate(GateType::H, &[2]),
        gate(GateType::CX, &[2, 0]),
        gate(GateType::CX, &[2, 1]),
        gate(GateType::H, &[2]),
        gate(GateType::MZ, &[2]),
        gate(GateType::PZ, &[2]),
        gate(GateType::H, &[2]),
        gate(GateType::CX, &[2, 0]),
        gate(GateType::CX, &[2, 1]),
        gate(GateType::H, &[2]),
        gate(GateType::MZ, &[2]),
    ];
    let expanded = expand::expand_circuit(&gates_orig);

    // Noise source locations in expanded circuit:
    // Gate 4: CX(2,0) R1 → noise on q2 and q0
    // Gate 5: CX(2,1) R1 → noise on q2 and q1
    // Gate 12: CX(2,0) R2 → noise on q2 and q0
    // Gate 13: CX(2,1) R2 → noise on q2 and q1
    let noise_sources = [
        (4, 2, "R1 CX(2,0) Z2"),
        (4, 0, "R1 CX(2,0) Z0"),
        (5, 2, "R1 CX(2,1) Z2"),
        (5, 1, "R1 CX(2,1) Z1"),
        (12, 2, "R2 CX(2,0) Z2"),
        (12, 0, "R2 CX(2,0) Z0"),
        (13, 2, "R2 CX(2,1) Z2"),
        (13, 1, "R2 CX(2,1) Z1"),
    ];

    eprintln!("\n=== Per-noise-source attribution ===");
    eprintln!(
        "{:>25} {:>12} {:>12} {:>8}",
        "Source", "Matrix", "Walk", "Ratio"
    );

    for &(gate_idx, qubit, label) in &noise_sources {
        // Matrix computation with only this one noise source
        let mut obs_re = vec![0.0f64; dim * dim];
        let mut obs_im = vec![0.0f64; dim * dim];
        for i in 0..dim {
            let bit3 = (i >> 3) & 1;
            let bit4 = (i >> 4) & 1;
            obs_re[i * dim + i] = if bit3 == bit4 { 1.0 } else { -1.0 };
        }

        // Process gates in reverse, only applying noise for the specified source
        for idx in (0..expanded.gates.len()).rev() {
            let g = &expanded.gates[idx];
            let qs: Vec<usize> = g.qubits.iter().map(pecos_core::QubitId::index).collect();

            // Noise: only the specified source
            if idx == gate_idx {
                apply_rz_adjoint(&mut obs_re, &mut obs_im, qubit, theta, n);
            }

            // Gate adjoint (always)
            match g.gate_type {
                pecos_core::gate_type::GateType::PZ | pecos_core::gate_type::GateType::QAlloc => {
                    apply_pz_adjoint(&mut obs_re, &mut obs_im, qs[0], n);
                }
                pecos_core::gate_type::GateType::MZ => {
                    apply_mz_adjoint(&mut obs_re, &mut obs_im, qs[0], n);
                }
                pecos_core::gate_type::GateType::H => {
                    apply_h_adjoint(&mut obs_re, &mut obs_im, qs[0], n);
                }
                pecos_core::gate_type::GateType::CX => {
                    apply_cx_adjoint(&mut obs_re, &mut obs_im, qs[0], qs[1], n);
                }
                _ => {}
            }
        }

        let matrix_p = 0.5 * (1.0 - obs_re[0]);

        // Backward walk: use a custom noise spec that only injects at the specified source
        // (We can't easily do this with the public API, so just report the matrix result.)
        eprintln!("{label:>25} {matrix_p:>12.8} {:>12} {:>8}", "-", "-");
    }

    // Also show all-noise results
    let mut obs_re = vec![0.0f64; dim * dim];
    let mut obs_im = vec![0.0f64; dim * dim];
    for i in 0..dim {
        let bit3 = (i >> 3) & 1;
        let bit4 = (i >> 4) & 1;
        obs_re[i * dim + i] = if bit3 == bit4 { 1.0 } else { -1.0 };
    }
    for idx in (0..expanded.gates.len()).rev() {
        let g = &expanded.gates[idx];
        let qs: Vec<usize> = g.qubits.iter().map(pecos_core::QubitId::index).collect();
        if g.gate_type == pecos_core::gate_type::GateType::CX && qs.len() >= 2 {
            // Check if expansion gate
            let is_exp = idx > 0
                && expanded.gates[idx - 1].gate_type == pecos_core::gate_type::GateType::QAlloc
                && expanded.gates[idx - 1].qubits[0].index() == qs[1];
            if !is_exp {
                apply_rz_adjoint(&mut obs_re, &mut obs_im, qs[0], theta, n);
                apply_rz_adjoint(&mut obs_re, &mut obs_im, qs[1], theta, n);
            }
        }
        match g.gate_type {
            pecos_core::gate_type::GateType::PZ | pecos_core::gate_type::GateType::QAlloc => {
                apply_pz_adjoint(&mut obs_re, &mut obs_im, qs[0], n);
            }
            pecos_core::gate_type::GateType::MZ => {
                apply_mz_adjoint(&mut obs_re, &mut obs_im, qs[0], n);
            }
            pecos_core::gate_type::GateType::H => {
                apply_h_adjoint(&mut obs_re, &mut obs_im, qs[0], n);
            }
            pecos_core::gate_type::GateType::CX => {
                apply_cx_adjoint(&mut obs_re, &mut obs_im, qs[0], qs[1], n);
            }
            _ => {}
        }
    }
    let matrix_all = 0.5 * (1.0 - obs_re[0]);
    let noise = UniformNoise::coherent_only(theta);
    let walk_all =
        heisenberg_detection_probability_from_circuit(&gates_orig, &[0, 1], &noise, 3, 0.0);
    eprintln!(
        "{:>25} {:>12.8} {:>12.8} {:>8.4}",
        "ALL",
        matrix_all,
        walk_all,
        walk_all / matrix_all
    );
}

/// Isolate weight-2 vs weight-4 X-check, single vs multi-round.
#[test]
#[ignore = "benchmark sweep; run manually with --ignored --nocapture"]
fn bench_weight_isolation() {
    use pecos_eeg::heisenberg::heisenberg_detection_probability_from_circuit;

    let num_shots = 500_000;
    let theta = 0.05;

    eprintln!("\n=== Weight isolation: single ancilla, no shared qubits ===");
    eprintln!("theta = {theta}, shots = {num_shots}\n");

    // ---- Weight-2, 1 round: H(2), CX(2,0), CX(2,1), H(2), MZ(2) ----
    {
        let gates = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::PZ, &[1]),
            gate(GateType::PZ, &[2]),
            gate(GateType::H, &[2]),
            gate(GateType::CX, &[2, 0]),
            gate(GateType::CX, &[2, 1]),
            gate(GateType::H, &[2]),
            gate(GateType::MZ, &[2]),
        ];
        let noise = UniformNoise::coherent_only(theta);
        let h_p = heisenberg_detection_probability_from_circuit(&gates, &[0], &noise, 3, 0.0);

        let mut det = 0u64;
        let mut sim = StateVec::new(3);
        for _ in 0..num_shots {
            sim.pz(&[qid(0), qid(1), qid(2)]);
            sim.h(&[qid(2)]);
            sim.cx(&[(qid(2), qid(0))]);
            sim.rz(Angle64::from_radians(theta), &[qid(2)]);
            sim.rz(Angle64::from_radians(theta), &[qid(0)]);
            sim.cx(&[(qid(2), qid(1))]);
            sim.rz(Angle64::from_radians(theta), &[qid(1)]);
            sim.rz(Angle64::from_radians(theta), &[qid(2)]);
            sim.h(&[qid(2)]);
            if sim.mz(&[qid(2)])[0].outcome {
                det += 1;
            }
        }
        let sv = u64_to_f64(det) / f64::from(num_shots);
        let se = (sv * (1.0 - sv) / f64::from(num_shots)).sqrt();
        eprintln!(
            "Wt-2 1rnd:  H={h_p:.6}  SV={sv:.6}+/-{se:.6}  H/SV={:.4}",
            if sv > 1e-10 { h_p / sv } else { f64::NAN }
        );
    }

    // ---- Weight-2, 2 rounds: round comparison ----
    {
        let gates = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::PZ, &[1]),
            gate(GateType::PZ, &[2]),
            gate(GateType::H, &[2]),
            gate(GateType::CX, &[2, 0]),
            gate(GateType::CX, &[2, 1]),
            gate(GateType::H, &[2]),
            gate(GateType::MZ, &[2]),
            gate(GateType::PZ, &[2]),
            gate(GateType::H, &[2]),
            gate(GateType::CX, &[2, 0]),
            gate(GateType::CX, &[2, 1]),
            gate(GateType::H, &[2]),
            gate(GateType::MZ, &[2]),
        ];
        let noise = UniformNoise::coherent_only(theta);
        let h_p = heisenberg_detection_probability_from_circuit(&gates, &[0, 1], &noise, 3, 0.0);

        let mut det = 0u64;
        let mut sim = StateVec::new(3);
        for _ in 0..num_shots {
            sim.pz(&[qid(0), qid(1), qid(2)]);
            let mut outs = [false; 2];
            for (r, out) in outs.iter_mut().enumerate() {
                sim.h(&[qid(2)]);
                sim.cx(&[(qid(2), qid(0))]);
                sim.rz(Angle64::from_radians(theta), &[qid(2)]);
                sim.rz(Angle64::from_radians(theta), &[qid(0)]);
                sim.cx(&[(qid(2), qid(1))]);
                sim.rz(Angle64::from_radians(theta), &[qid(1)]);
                sim.rz(Angle64::from_radians(theta), &[qid(2)]);
                sim.h(&[qid(2)]);
                *out = sim.mz(&[qid(2)])[0].outcome;
                if r == 0 {
                    sim.pz(&[qid(2)]);
                }
            }
            if outs[0] != outs[1] {
                det += 1;
            }
        }
        let sv = u64_to_f64(det) / f64::from(num_shots);
        let se = (sv * (1.0 - sv) / f64::from(num_shots)).sqrt();
        eprintln!(
            "Wt-2 2rnd:  H={h_p:.6}  SV={sv:.6}+/-{se:.6}  H/SV={:.4}",
            if sv > 1e-10 { h_p / sv } else { f64::NAN }
        );
    }

    // ---- Weight-4, 1 round ----
    {
        let gates = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::PZ, &[1]),
            gate(GateType::PZ, &[2]),
            gate(GateType::PZ, &[3]),
            gate(GateType::PZ, &[4]),
            gate(GateType::H, &[4]),
            gate(GateType::CX, &[4, 0]),
            gate(GateType::CX, &[4, 1]),
            gate(GateType::CX, &[4, 2]),
            gate(GateType::CX, &[4, 3]),
            gate(GateType::H, &[4]),
            gate(GateType::MZ, &[4]),
        ];
        let noise = UniformNoise::coherent_only(theta);
        let h_p = heisenberg_detection_probability_from_circuit(&gates, &[0], &noise, 5, 0.0);

        let mut det = 0u64;
        let mut sim = StateVec::new(5);
        for _ in 0..num_shots {
            sim.pz(&[qid(0), qid(1), qid(2), qid(3), qid(4)]);
            sim.h(&[qid(4)]);
            for &d in &[0usize, 1, 2, 3] {
                sim.cx(&[(qid(4), qid(d))]);
                sim.rz(Angle64::from_radians(theta), &[qid(4)]);
                sim.rz(Angle64::from_radians(theta), &[qid(d)]);
            }
            sim.h(&[qid(4)]);
            if sim.mz(&[qid(4)])[0].outcome {
                det += 1;
            }
        }
        let sv = u64_to_f64(det) / f64::from(num_shots);
        let se = (sv * (1.0 - sv) / f64::from(num_shots)).sqrt();
        eprintln!(
            "Wt-4 1rnd:  H={h_p:.6}  SV={sv:.6}+/-{se:.6}  H/SV={:.4}",
            if sv > 1e-10 { h_p / sv } else { f64::NAN }
        );
    }

    // ---- Weight-4, 2 rounds ----
    {
        let gates = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::PZ, &[1]),
            gate(GateType::PZ, &[2]),
            gate(GateType::PZ, &[3]),
            gate(GateType::PZ, &[4]),
            gate(GateType::H, &[4]),
            gate(GateType::CX, &[4, 0]),
            gate(GateType::CX, &[4, 1]),
            gate(GateType::CX, &[4, 2]),
            gate(GateType::CX, &[4, 3]),
            gate(GateType::H, &[4]),
            gate(GateType::MZ, &[4]),
            gate(GateType::PZ, &[4]),
            gate(GateType::H, &[4]),
            gate(GateType::CX, &[4, 0]),
            gate(GateType::CX, &[4, 1]),
            gate(GateType::CX, &[4, 2]),
            gate(GateType::CX, &[4, 3]),
            gate(GateType::H, &[4]),
            gate(GateType::MZ, &[4]),
        ];
        let noise = UniformNoise::coherent_only(theta);
        let h_p = heisenberg_detection_probability_from_circuit(&gates, &[0, 1], &noise, 5, 0.0);

        let mut det = 0u64;
        let mut sim = StateVec::new(5);
        for _ in 0..num_shots {
            sim.pz(&[qid(0), qid(1), qid(2), qid(3), qid(4)]);
            let mut outs = [false; 2];
            for (r, out) in outs.iter_mut().enumerate() {
                sim.h(&[qid(4)]);
                for &d in &[0usize, 1, 2, 3] {
                    sim.cx(&[(qid(4), qid(d))]);
                    sim.rz(Angle64::from_radians(theta), &[qid(4)]);
                    sim.rz(Angle64::from_radians(theta), &[qid(d)]);
                }
                sim.h(&[qid(4)]);
                *out = sim.mz(&[qid(4)])[0].outcome;
                if r == 0 {
                    sim.pz(&[qid(4)]);
                }
            }
            if outs[0] != outs[1] {
                det += 1;
            }
        }
        let sv = u64_to_f64(det) / f64::from(num_shots);
        let se = (sv * (1.0 - sv) / f64::from(num_shots)).sqrt();
        eprintln!(
            "Wt-4 2rnd:  H={h_p:.6}  SV={sv:.6}+/-{se:.6}  H/SV={:.4}",
            if sv > 1e-10 { h_p / sv } else { f64::NAN }
        );
    }

    eprintln!();
    // ---- 2 weight-2 ancillas sharing a data qubit, 2 rounds ----
    {
        let gates = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::PZ, &[1]),
            gate(GateType::PZ, &[2]),
            gate(GateType::PZ, &[3]),
            gate(GateType::PZ, &[4]),
            // Round 1
            gate(GateType::H, &[3]),
            gate(GateType::H, &[4]),
            gate(GateType::CX, &[3, 0]),
            gate(GateType::CX, &[4, 1]),
            gate(GateType::CX, &[3, 1]),
            gate(GateType::CX, &[4, 2]),
            gate(GateType::H, &[3]),
            gate(GateType::H, &[4]),
            gate(GateType::MZ, &[3]),
            gate(GateType::MZ, &[4]),
            gate(GateType::PZ, &[3]),
            gate(GateType::PZ, &[4]),
            // Round 2
            gate(GateType::H, &[3]),
            gate(GateType::H, &[4]),
            gate(GateType::CX, &[3, 0]),
            gate(GateType::CX, &[4, 1]),
            gate(GateType::CX, &[3, 1]),
            gate(GateType::CX, &[4, 2]),
            gate(GateType::H, &[3]),
            gate(GateType::H, &[4]),
            gate(GateType::MZ, &[3]),
            gate(GateType::MZ, &[4]),
        ];
        let noise = UniformNoise::coherent_only(theta);
        let h_a0 = heisenberg_detection_probability_from_circuit(&gates, &[0, 2], &noise, 5, 0.0);
        let h_a1 = heisenberg_detection_probability_from_circuit(&gates, &[1, 3], &noise, 5, 0.0);

        let mut a0 = 0u64;
        let mut a1 = 0u64;
        let mut sim = StateVec::new(5);
        for _ in 0..num_shots {
            sim.pz(&[qid(0), qid(1), qid(2), qid(3), qid(4)]);
            let mut outs = [false; 4]; // [r0a0, r0a1, r1a0, r1a1]
            for (r, out_pair) in outs.chunks_mut(2).enumerate() {
                sim.h(&[qid(3), qid(4)]);
                sim.cx(&[(qid(3), qid(0))]);
                sim.rz(Angle64::from_radians(theta), &[qid(3)]);
                sim.rz(Angle64::from_radians(theta), &[qid(0)]);
                sim.cx(&[(qid(4), qid(1))]);
                sim.rz(Angle64::from_radians(theta), &[qid(4)]);
                sim.rz(Angle64::from_radians(theta), &[qid(1)]);
                sim.cx(&[(qid(3), qid(1))]);
                sim.rz(Angle64::from_radians(theta), &[qid(3)]);
                sim.rz(Angle64::from_radians(theta), &[qid(1)]);
                sim.cx(&[(qid(4), qid(2))]);
                sim.rz(Angle64::from_radians(theta), &[qid(4)]);
                sim.rz(Angle64::from_radians(theta), &[qid(2)]);
                sim.h(&[qid(3), qid(4)]);
                out_pair[0] = sim.mz(&[qid(3)])[0].outcome;
                out_pair[1] = sim.mz(&[qid(4)])[0].outcome;
                if r == 0 {
                    sim.pz(&[qid(3), qid(4)]);
                }
            }
            if outs[0] != outs[2] {
                a0 += 1;
            }
            if outs[1] != outs[3] {
                a1 += 1;
            }
        }
        let sv0 = u64_to_f64(a0) / f64::from(num_shots);
        let sv1 = u64_to_f64(a1) / f64::from(num_shots);
        let se0 = (sv0 * (1.0 - sv0) / f64::from(num_shots)).sqrt();
        let se1 = (sv1 * (1.0 - sv1) / f64::from(num_shots)).sqrt();
        eprintln!(
            "Shared A0:  H={h_a0:.6}  SV={sv0:.6}+/-{se0:.6}  H/SV={:.4}",
            if sv0 > 1e-10 { h_a0 / sv0 } else { f64::NAN }
        );
        eprintln!(
            "Shared A1:  H={h_a1:.6}  SV={sv1:.6}+/-{se1:.6}  H/SV={:.4}",
            if sv1 > 1e-10 { h_a1 / sv1 } else { f64::NAN }
        );
    }
}

/// Measure how Heisenberg backward walk cost scales with surface code distance and rounds.
///
/// The backward walk creates 2^m terms where m is the number of anticommuting
/// noise sources per detector. For larger codes, m grows, making the walk
/// exponentially more expensive.
///
/// We measure ALL round-comparison detectors and report per-detector timing,
/// since boundary detectors (ancilla 0/last) only couple to 2 CX gates while
/// bulk detectors (middle ancillas) couple to 4, seeing more noise sources.
#[test]
#[ignore = "benchmark sweep; run manually with --ignored --nocapture"]
fn bench_heisenberg_scaling() {
    use pecos_eeg::heisenberg::heisenberg_detection_probability_from_circuit;
    use std::time::Instant;

    let theta = 0.05;

    // --- Part 1: Vary distance at fixed rounds=2 ---
    eprintln!("\n=== Heisenberg scaling: distance sweep (rounds=2, all detectors) ===");
    eprintln!(
        "{:>4} {:>10} {:>14} {:>6} {:>18} {:>12} {:>12} {:>12}",
        "d", "num_qubits", "expanded_q", "n_det", "max_prob", "max_ms", "total_ms", "per_det_ms"
    );

    for &d in &[3, 5, 7, 9] {
        let num_rounds = 2;
        let (gates, num_qubits, _ancillas) = build_repetition_code(d, num_rounds);
        let num_ancilla = d - 1;
        let num_detectors = num_ancilla; // rounds-1 == 1 comparison per ancilla

        let expanded = expand::expand_circuit(&gates);
        let noise = UniformNoise::coherent_only(theta);

        let mut max_prob = 0.0f64;
        let mut max_ms = 0.0f64;
        let mut total_ms = 0.0f64;

        eprintln!("  d={d} per-detector detail:");
        eprintln!("    {:>6} {:>18} {:>12}", "det", "prob", "time_ms");

        for i in 0..num_detectors {
            // Round comparison: meas record i (round 0) vs i + num_ancilla (round 1)
            let m1 = i;
            let m2 = i + num_ancilla;

            let start = Instant::now();
            let prob = heisenberg_detection_probability_from_circuit(
                &gates,
                &[m1, m2],
                &noise,
                num_qubits,
                0.0,
            );
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

            eprintln!("    {i:>6} {prob:>18.10} {elapsed_ms:>12.2}");

            max_prob = max_prob.max(prob);
            max_ms = max_ms.max(elapsed_ms);
            total_ms += elapsed_ms;
        }

        let per_det = total_ms / usize_to_f64(num_detectors);
        eprintln!(
            "{d:>4} {num_qubits:>10} {:>14} {num_detectors:>6} {max_prob:>18.10} {max_ms:>12.2} {total_ms:>12.2} {per_det:>12.2}",
            expanded.num_qubits
        );
    }

    // --- Part 2: Vary rounds at fixed d=3 ---
    eprintln!("\n=== Heisenberg scaling: rounds sweep (d=3, all detectors) ===");
    eprintln!(
        "{:>6} {:>10} {:>14} {:>6} {:>18} {:>12} {:>12} {:>12}",
        "rounds",
        "num_qubits",
        "expanded_q",
        "n_det",
        "max_prob",
        "max_ms",
        "total_ms",
        "per_det_ms"
    );

    for &num_rounds in &[2, 3, 4, 5] {
        let d = 3;
        let (gates, num_qubits, _ancillas) = build_repetition_code(d, num_rounds);
        let num_ancilla = d - 1;
        // Round comparison detectors: (num_rounds - 1) comparisons per ancilla
        let num_detectors = num_ancilla * (num_rounds - 1);

        let expanded = expand::expand_circuit(&gates);
        let noise = UniformNoise::coherent_only(theta);

        let mut max_prob = 0.0f64;
        let mut max_ms = 0.0f64;
        let mut total_ms = 0.0f64;

        eprintln!("  rounds={num_rounds} per-detector detail:");
        eprintln!("    {:>6} {:>18} {:>12}", "det", "prob", "time_ms");

        for round in 0..(num_rounds - 1) {
            for i in 0..num_ancilla {
                let m1 = round * num_ancilla + i;
                let m2 = (round + 1) * num_ancilla + i;

                let start = Instant::now();
                let prob = heisenberg_detection_probability_from_circuit(
                    &gates,
                    &[m1, m2],
                    &noise,
                    num_qubits,
                    0.0,
                );
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

                eprintln!("    R{round}A{i} {prob:>18.10} {elapsed_ms:>12.2}");

                max_prob = max_prob.max(prob);
                max_ms = max_ms.max(elapsed_ms);
                total_ms += elapsed_ms;
            }
        }

        let per_det = total_ms / usize_to_f64(num_detectors);
        eprintln!(
            "{num_rounds:>6} {num_qubits:>10} {:>14} {num_detectors:>6} {max_prob:>18.10} {max_ms:>12.2} {total_ms:>12.2} {per_det:>12.2}",
            expanded.num_qubits
        );
    }
}

/// Combined coherent + stochastic noise on a Wt-2 2-round X-check circuit.
///
/// Uses `idle_rz=0.05` (coherent RZ after each CX) plus `p_meas=0.003`
/// (measurement bit-flip). The Heisenberg walk handles both H-type and
/// S-type generators in a single backward pass.
///
/// `StateVec` applies identical noise: RZ(theta) on both qubits after each
/// CX, and flips the MZ outcome with probability `p_meas`.
///
/// Detector: round-comparison (meas[0] XOR meas[1]).
#[test]
#[ignore = "benchmark sweep; run manually with --ignored --nocapture"]
fn bench_combined_noise() {
    use pecos_eeg::heisenberg::heisenberg_detection_probability_from_circuit;
    use pecos_random::PecosRng;
    use pecos_random::rng_ext::RngProbabilityExt;

    let idle_rz = 0.05;
    let p_meas = 0.003;
    let num_shots = 500_000;

    eprintln!("\n=== Combined coherent + stochastic noise: Wt-2 2-round X-check ===");
    eprintln!("idle_rz = {idle_rz}, p_meas = {p_meas}, shots = {num_shots}\n");

    // ---- Build the circuit: 2 data (q0,q1) + 1 ancilla (q2), 2 rounds ----
    let gates = vec![
        gate(GateType::PZ, &[0]),
        gate(GateType::PZ, &[1]),
        gate(GateType::PZ, &[2]),
        // Round 1
        gate(GateType::H, &[2]),
        gate(GateType::CX, &[2, 0]),
        gate(GateType::CX, &[2, 1]),
        gate(GateType::H, &[2]),
        gate(GateType::MZ, &[2]),
        gate(GateType::PZ, &[2]),
        // Round 2
        gate(GateType::H, &[2]),
        gate(GateType::CX, &[2, 0]),
        gate(GateType::CX, &[2, 1]),
        gate(GateType::H, &[2]),
        gate(GateType::MZ, &[2]),
    ];

    // ---- Heisenberg walk with combined noise ----
    let noise = UniformNoise {
        idle_rz,
        p1: 0.0,
        p2: 0.0,
        p_meas,
        p_prep: 0.0,
    };
    // Detector = Z on meas[0] * Z on meas[1] (round comparison)
    let h_p = heisenberg_detection_probability_from_circuit(&gates, &[0, 1], &noise, 3, 0.0);

    // ---- Also compute coherent-only and meas-only for decomposition ----
    let noise_coh = UniformNoise::coherent_only(idle_rz);
    let h_coh = heisenberg_detection_probability_from_circuit(&gates, &[0, 1], &noise_coh, 3, 0.0);

    let noise_meas = UniformNoise {
        idle_rz: 0.0,
        p1: 0.0,
        p2: 0.0,
        p_meas,
        p_prep: 0.0,
    };
    let h_meas =
        heisenberg_detection_probability_from_circuit(&gates, &[0, 1], &noise_meas, 3, 0.0);

    // ---- StateVec simulation with matching noise ----
    let mut rng = PecosRng::seed_from_u64(12345);
    let meas_threshold = rng.probability_threshold(p_meas);

    let mut det = 0u64;
    let mut sim = StateVec::new(3);
    for _ in 0..num_shots {
        sim.pz(&[qid(0), qid(1), qid(2)]);
        let mut outs = [false; 2];
        for (r, out_slot) in outs.iter_mut().enumerate() {
            sim.h(&[qid(2)]);
            // CX(2,0) + idle RZ noise
            sim.cx(&[(qid(2), qid(0))]);
            sim.rz(Angle64::from_radians(idle_rz), &[qid(2)]);
            sim.rz(Angle64::from_radians(idle_rz), &[qid(0)]);
            // CX(2,1) + idle RZ noise
            sim.cx(&[(qid(2), qid(1))]);
            sim.rz(Angle64::from_radians(idle_rz), &[qid(1)]);
            sim.rz(Angle64::from_radians(idle_rz), &[qid(2)]);
            sim.h(&[qid(2)]);
            // MZ with measurement error
            let mut outcome = sim.mz(&[qid(2)])[0].outcome;
            if rng.check_probability(meas_threshold) {
                outcome = !outcome;
            }
            *out_slot = outcome;
            if r == 0 {
                sim.pz(&[qid(2)]);
            }
        }
        if outs[0] != outs[1] {
            det += 1;
        }
    }
    let sv = u64_to_f64(det) / f64::from(num_shots);
    let se = (sv * (1.0 - sv) / f64::from(num_shots)).sqrt();
    let ratio = if sv > 1e-10 { h_p / sv } else { f64::NAN };

    eprintln!("Heisenberg (combined): {h_p:.6}");
    eprintln!("Heisenberg (coh only): {h_coh:.6}");
    eprintln!("Heisenberg (meas only):{h_meas:.6}");
    eprintln!("StateVec:              {sv:.6} +/- {se:.6}");
    eprintln!("H/SV ratio:            {ratio:.4}");
    eprintln!();

    // ---- Sweep p_meas to see how combined noise scales ----
    eprintln!(
        "{:>8} {:>10} {:>10} {:>10} {:>10}",
        "p_meas", "H_comb", "SV", "SV_stderr", "H/SV"
    );

    for &pm in &[0.0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.05] {
        let n = UniformNoise {
            idle_rz,
            p1: 0.0,
            p2: 0.0,
            p_meas: pm,
            p_prep: 0.0,
        };
        let hp = heisenberg_detection_probability_from_circuit(&gates, &[0, 1], &n, 3, 0.0);

        let pm_threshold = rng.probability_threshold(pm);
        let mut d = 0u64;
        for _ in 0..num_shots {
            sim.pz(&[qid(0), qid(1), qid(2)]);
            let mut os = [false; 2];
            for (r, out_slot) in os.iter_mut().enumerate() {
                sim.h(&[qid(2)]);
                sim.cx(&[(qid(2), qid(0))]);
                sim.rz(Angle64::from_radians(idle_rz), &[qid(2)]);
                sim.rz(Angle64::from_radians(idle_rz), &[qid(0)]);
                sim.cx(&[(qid(2), qid(1))]);
                sim.rz(Angle64::from_radians(idle_rz), &[qid(1)]);
                sim.rz(Angle64::from_radians(idle_rz), &[qid(2)]);
                sim.h(&[qid(2)]);
                let mut out = sim.mz(&[qid(2)])[0].outcome;
                if rng.check_probability(pm_threshold) {
                    out = !out;
                }
                *out_slot = out;
                if r == 0 {
                    sim.pz(&[qid(2)]);
                }
            }
            if os[0] != os[1] {
                d += 1;
            }
        }
        let s = u64_to_f64(d) / f64::from(num_shots);
        let e = (s * (1.0 - s) / f64::from(num_shots)).sqrt();
        let r = if s > 1e-10 { hp / s } else { f64::NAN };
        eprintln!("{pm:>8.4} {hp:>10.6} {s:>10.6} {e:>10.6} {r:>10.4}");
    }
}
