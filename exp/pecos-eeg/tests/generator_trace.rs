// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! Trace generator propagation for d=2 Z-basis surface code.
//! Diagnose why D1 and D2 have identical EEG probabilities
//! when `StateVec` shows they should differ.

use std::collections::BTreeMap;

use pecos_core::gate_type::GateType;
use pecos_core::{Gate, GateAngles, GateParams, QubitId};
use pecos_eeg::Bm;
use pecos_eeg::circuit::{NoiseModel, analyze_expanded};
use pecos_eeg::dem_mapping::*;
use pecos_eeg::eeg::EegType;
use pecos_eeg::expand;
use pecos_eeg::stabilizer::StabilizerGroup;

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

/// Build the d=2 Z-basis surface code circuit (2 rounds).
/// Matches what `LogicalCircuitBuilder` produces.
fn build_d2_zbasis() -> Vec<Gate> {
    vec![
        // Init
        gate(GateType::PZ, &[0]),
        gate(GateType::PZ, &[1]),
        gate(GateType::PZ, &[2]),
        gate(GateType::PZ, &[3]),
        gate(GateType::PZ, &[4]),
        gate(GateType::PZ, &[5]),
        gate(GateType::PZ, &[6]),
        // Round 1
        gate(GateType::H, &[4]),
        gate(GateType::H, &[5]),
        gate(GateType::CX, &[1, 6]),
        gate(GateType::CX, &[5, 3]), // tick 3
        gate(GateType::CX, &[3, 6]),
        gate(GateType::CX, &[5, 2]), // tick 4
        gate(GateType::CX, &[4, 1]),
        gate(GateType::CX, &[0, 6]), // tick 5
        gate(GateType::CX, &[4, 0]),
        gate(GateType::CX, &[2, 6]), // tick 6
        gate(GateType::H, &[4]),
        gate(GateType::H, &[5]),
        gate(GateType::MZ, &[4]),
        gate(GateType::MZ, &[5]),
        gate(GateType::MZ, &[6]),
        // Reset
        gate(GateType::PZ, &[4]),
        gate(GateType::PZ, &[5]),
        gate(GateType::PZ, &[6]),
        // Round 2
        gate(GateType::H, &[4]),
        gate(GateType::H, &[5]),
        gate(GateType::CX, &[1, 6]),
        gate(GateType::CX, &[5, 3]), // tick 11
        gate(GateType::CX, &[3, 6]),
        gate(GateType::CX, &[5, 2]), // tick 12
        gate(GateType::CX, &[4, 1]),
        gate(GateType::CX, &[0, 6]), // tick 13
        gate(GateType::CX, &[4, 0]),
        gate(GateType::CX, &[2, 6]), // tick 14
        gate(GateType::H, &[4]),
        gate(GateType::H, &[5]),
        gate(GateType::MZ, &[4]),
        gate(GateType::MZ, &[5]),
        gate(GateType::MZ, &[6]),
        // Final data readout
        gate(GateType::MZ, &[0]),
        gate(GateType::MZ, &[1]),
        gate(GateType::MZ, &[2]),
        gate(GateType::MZ, &[3]),
    ]
}

#[test]
fn trace_d2_zbasis_generators() {
    let gates = build_d2_zbasis();
    let expanded = expand::expand_circuit(&gates);
    let noise = NoiseModel::coherent_only(0.01);
    let result = analyze_expanded(&expanded.gates, &noise);

    eprintln!(
        "Expanded: {} qubits ({} orig + {} aux), {} measurements",
        expanded.num_qubits,
        expanded.num_original_qubits,
        expanded.num_qubits - expanded.num_original_qubits,
        expanded.measurement_qubit.len()
    );

    eprintln!("\nMeasurement mapping:");
    for (i, (&aux, &orig)) in expanded
        .measurement_qubit
        .iter()
        .zip(expanded.original_measured_qubit.iter())
        .enumerate()
    {
        eprintln!("  meas[{i}]: aux=q{aux}, orig=q{orig}");
    }

    // Detectors: D1 = Z_{aux_meas0} * Z_{aux_meas3} (ancilla 4, rounds 1&2)
    //            D2 = Z_{aux_meas1} * Z_{aux_meas4} (ancilla 5, rounds 1&2)
    let aux_m0 = expanded.measurement_qubit[0]; // q4 round 1
    let aux_m1 = expanded.measurement_qubit[1]; // q5 round 1
    let aux_m3 = expanded.measurement_qubit[3]; // q4 round 2
    let aux_m4 = expanded.measurement_qubit[4]; // q5 round 2

    let d1_stab = Bm::z(aux_m0).multiply(&Bm::z(aux_m3));
    let d2_stab = Bm::z(aux_m1).multiply(&Bm::z(aux_m4));

    eprintln!("\nD1 stabilizer: Z on aux q{aux_m0} and q{aux_m3} (ancilla 4 rounds 1&2)");
    eprintln!("D2 stabilizer: Z on aux q{aux_m1} and q{aux_m4} (ancilla 5 rounds 1&2)");

    let _dets = [
        Detector {
            id: 1,
            stabilizer: d1_stab.clone(),
        },
        Detector {
            id: 2,
            stabilizer: d2_stab.clone(),
        },
    ];

    // Classify each H generator
    let h_gens: Vec<_> = result
        .generators
        .iter()
        .filter(|g| g.eeg_type == EegType::H)
        .collect();

    eprintln!("\n{} H generators. Classification:", h_gens.len());

    let mut d1_gens = Vec::new();
    let mut d2_gens = Vec::new();

    for g in &h_gens {
        let flips_d1 = !g.label.commutes_with(&d1_stab);
        let flips_d2 = !g.label.commutes_with(&d2_stab);

        if flips_d1 || flips_d2 {
            let orig = expanded.map_to_original_frame(&g.label);
            eprintln!(
                "  {:?} coeff={:.6} -> orig={:?} flips: D1={} D2={}",
                g.label, g.coeff, orig, flips_d1, flips_d2
            );

            if flips_d1 {
                d1_gens.push((g.label.clone(), g.coeff));
            }
            if flips_d2 {
                d2_gens.push((g.label.clone(), g.coeff));
            }
        }
    }

    eprintln!("\nD1 generators: {} (ancilla 4)", d1_gens.len());
    for (label, coeff) in &d1_gens {
        eprintln!("  {label:?} coeff={coeff:.6}");
    }

    eprintln!("\nD2 generators: {} (ancilla 5)", d2_gens.len());
    for (label, coeff) in &d2_gens {
        eprintln!("  {label:?} coeff={coeff:.6}");
    }

    // After BCH combination (same label → sum coefficients)
    let mut d1_bch: BTreeMap<Bm, f64> = BTreeMap::new();
    let mut d2_bch: BTreeMap<Bm, f64> = BTreeMap::new();
    for (l, c) in &d1_gens {
        *d1_bch.entry(l.clone()).or_default() += c;
    }
    for (l, c) in &d2_gens {
        *d2_bch.entry(l.clone()).or_default() += c;
    }

    eprintln!("\nD1 after BCH: {} distinct labels", d1_bch.len());
    for (l, c) in &d1_bch {
        eprintln!("  {l:?} rate={c:.6}");
    }

    eprintln!("\nD2 after BCH: {} distinct labels", d2_bch.len());
    for (l, c) in &d2_bch {
        eprintln!("  {l:?} rate={c:.6}");
    }

    // Verify asymmetry in generator counts
    assert_ne!(
        d1_bch.len(),
        d2_bch.len(),
        "D1 and D2 should have different numbers of BCH-combined generators"
    );

    // Compute probabilities manually to trace the beta function
    let gates_pre = crate::exclude_final_readout(&gates);
    let stab_group = StabilizerGroup::from_circuit(&gates_pre, expanded.num_original_qubits);

    // Check what the stabilizer group contains
    eprintln!("\nStabilizer group membership checks:");
    let test_paulis = vec![
        ("Z0", Bm::z(0)),
        ("Z1", Bm::z(1)),
        ("Z2", Bm::z(2)),
        ("Z3", Bm::z(3)),
        ("Z0Z1", Bm::z(0).multiply(&Bm::z(1))),
        ("Z0Z3", Bm::z(0).multiply(&Bm::z(3))),
        ("X0X1", Bm::x(0).multiply(&Bm::x(1))),
    ];
    for (name, p) in &test_paulis {
        let result = stab_group.is_stabilizer(p);
        eprintln!("  {name}: {result:?}");
    }

    eprintln!("\nPre-readout gates count: {}", gates_pre.len());
    eprintln!("Original gates count: {}", gates.len());

    // Dump raw SparseStab generators
    {
        use pecos_simulators::{CliffordGateable, SparseStab};
        let mut sim = SparseStab::with_seed(7, 0);
        for g in &gates_pre {
            let qs: Vec<QubitId> = g.qubits.iter().copied().collect();
            if qs.is_empty() {
                continue;
            }
            match g.gate_type {
                GateType::PZ => {
                    for &q in &qs {
                        sim.pz(&[q]);
                    }
                }
                GateType::H => {
                    sim.h(&qs);
                }
                GateType::CX if qs.len() >= 2 => {
                    sim.cx(&[(qs[0], qs[1])]);
                }
                GateType::MZ => {
                    let _ = sim.mz(&qs);
                }
                _ => {}
            }
        }
        let stabs = sim.stabs();
        let n_gens = stabs.num_generators();
        eprintln!("\nSparseStab generators ({n_gens}):");
        for i in 0..n_gens {
            let ps = stabs.generator(i);
            let phase = stabs.generator_phase(i);
            eprintln!("  [{i}] phase={phase:?} {ps}");
        }

        // Direct test: can find_pauli_sign find Z0?
        let result = stabs.find_pauli_sign(
            sim.destabs(),
            std::iter::empty::<usize>(),
            std::iter::once(0usize),
            0,
        );
        eprintln!("\nfind_pauli_sign(Z0) WITH MZ = {result:?}");

        // Try without MZ: skip measurements in stabilizer computation
        let mut sim2 = SparseStab::with_seed(7, 0);
        for g in &gates_pre {
            let qs: Vec<QubitId> = g.qubits.iter().copied().collect();
            if qs.is_empty() {
                continue;
            }
            match g.gate_type {
                GateType::PZ => {
                    for &q in &qs {
                        sim2.pz(&[q]);
                    }
                }
                GateType::H => {
                    sim2.h(&qs);
                }
                GateType::CX if qs.len() >= 2 => {
                    sim2.cx(&[(qs[0], qs[1])]);
                }
                _ => {}
            }
        }
        let stabs2 = sim2.stabs();
        let result2 = stabs2.find_pauli_sign(
            sim2.destabs(),
            std::iter::empty::<usize>(),
            std::iter::once(0usize),
            0,
        );
        eprintln!("find_pauli_sign(Z0) WITHOUT MZ = {result2:?}");

        // Check X0X1 without MZ
        let result3 =
            stabs2.find_pauli_sign(sim2.destabs(), [0usize, 1], std::iter::empty::<usize>(), 0);
        eprintln!("find_pauli_sign(X0X1) WITHOUT MZ = {result3:?}");
    }

    // Check: how many qubits in the stabilizer group? And test Z0Z1Z2Z3
    let z_all = Bm::z(0)
        .multiply(&Bm::z(1))
        .multiply(&Bm::z(2))
        .multiply(&Bm::z(3));
    eprintln!("Z0Z1Z2Z3: {:?}", stab_group.is_stabilizer(&z_all));
    let _z01 = Bm::z(0).multiply(&Bm::z(1));
    let z23 = Bm::z(2).multiply(&Bm::z(3));
    eprintln!("Z2Z3: {:?}", stab_group.is_stabilizer(&z23));
    eprintln!(
        "Z0Z1Z2: {:?}",
        stab_group.is_stabilizer(&Bm::z(0).multiply(&Bm::z(1)).multiply(&Bm::z(2)))
    );
    eprintln!(
        "Z1Z2: {:?}",
        stab_group.is_stabilizer(&Bm::z(1).multiply(&Bm::z(2)))
    );

    for (det_name, bch) in [("D1", &d1_bch), ("D2", &d2_bch)] {
        let labels: Vec<Bm> = bch.keys().cloned().collect();
        let coeffs: Vec<f64> = bch.values().copied().collect();
        let n = labels.len();

        let mut diag = 0.0;
        let mut offdiag = 0.0;
        let mut offdiag_zero = 0;
        let mut offdiag_plus = 0;
        let mut offdiag_minus = 0;
        let mut offdiag_anticommute = 0;

        for j in 0..n {
            diag += coeffs[j] * coeffs[j];
            for k in (j + 1)..n {
                if !labels[j].commutes_with(&labels[k]) {
                    offdiag_anticommute += 1;
                    continue;
                }
                let product = labels[j].multiply(&labels[k]);
                let orig_product = expanded.map_to_original_frame(&product);

                if orig_product.is_identity() {
                    offdiag += 2.0 * coeffs[j] * coeffs[k];
                    offdiag_plus += 1;
                    continue;
                }
                match stab_group.is_stabilizer(&orig_product) {
                    Some(true) => {
                        offdiag += 2.0 * coeffs[j] * coeffs[k];
                        offdiag_plus += 1;
                    }
                    Some(false) => {
                        offdiag -= 2.0 * coeffs[j] * coeffs[k];
                        offdiag_minus += 1;
                    }
                    None => {
                        offdiag_zero += 1;
                        eprintln!(
                            "    beta=0: {:?} * {:?} = {:?} (orig: {:?})",
                            labels[j], labels[k], product, orig_product
                        );
                    }
                }
            }
        }

        let total = diag + offdiag;
        eprintln!("\n{det_name} probability breakdown:");
        eprintln!("  Diagonal:     {diag:.8}");
        eprintln!(
            "  Off-diagonal: {offdiag:.8} (+{offdiag_plus} pairs, -{offdiag_minus} pairs, 0:{offdiag_zero} pairs, anticommute:{offdiag_anticommute})"
        );
        eprintln!("  Total:        {total:.8}");
    }
}

fn exclude_final_readout(gates: &[Gate]) -> Vec<Gate> {
    use pecos_core::gate_type::GateType;
    let mut ancilla_qubits = std::collections::HashSet::new();
    let mut past_init = false;
    for g in gates {
        if past_init && (g.gate_type == GateType::PZ || g.gate_type == GateType::QAlloc) {
            for q in &g.qubits {
                ancilla_qubits.insert(q.index());
            }
        }
        if g.gate_type != GateType::PZ && g.gate_type != GateType::QAlloc {
            past_init = true;
        }
    }
    let mut end = gates.len();
    for g in gates.iter().rev() {
        if g.gate_type != GateType::MZ {
            break;
        }
        if g.qubits
            .iter()
            .all(|q| !ancilla_qubits.contains(&q.index()))
        {
            end -= 1;
        } else {
            break;
        }
    }
    gates[..end].to_vec()
}
