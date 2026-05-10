// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! Investigation: why off-diagonal beta terms don't fire for Z-basis.

use pecos_core::gate_type::GateType;
use pecos_core::{Gate, GateAngles, GateParams, QubitId};
use pecos_eeg::Bm;
use pecos_eeg::circuit::{NoiseModel, PropagatedEeg, analyze_expanded};
use pecos_eeg::eeg::EegType;
use pecos_eeg::expand;
use pecos_eeg::stabilizer::StabilizerGroup;
use pecos_simulators::{CliffordGateable, SparseStab};

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

/// Build a minimal Z-basis circuit: 2 data + 1 X-ancilla, 2 rounds.
fn build_minimal_zbasis() -> Vec<Gate> {
    vec![
        // Init
        gate(GateType::PZ, &[0]),
        gate(GateType::PZ, &[1]),
        gate(GateType::PZ, &[2]), // X-ancilla
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
        // Final data readout
        gate(GateType::MZ, &[0]),
        gate(GateType::MZ, &[1]),
    ]
}

#[test]
fn test_zbasis_generator_labels() {
    let gates = build_minimal_zbasis();
    let expanded = expand::expand_circuit(&gates);

    eprintln!(
        "Expanded circuit: {} qubits ({} original + {} aux)",
        expanded.num_qubits,
        expanded.num_original_qubits,
        expanded.num_qubits - expanded.num_original_qubits
    );
    eprintln!("Measurement mapping:");
    for (i, (&aux, &orig)) in expanded
        .measurement_qubit
        .iter()
        .zip(expanded.original_measured_qubit.iter())
        .enumerate()
    {
        eprintln!("  meas {i}: aux={aux} orig={orig}");
    }

    let noise = NoiseModel::coherent_only(0.001);
    let result = analyze_expanded(&expanded.gates, &noise);

    let h_gens: Vec<&PropagatedEeg> = result
        .generators
        .iter()
        .filter(|g| g.eeg_type == EegType::H)
        .collect();

    eprintln!("\nH generators ({}):", h_gens.len());
    for (i, g) in h_gens.iter().enumerate() {
        let orig = expanded.map_to_original_frame(&g.label);
        eprintln!(
            "  [{i}] expanded={:?} coeff={:.6} original_frame={:?}",
            g.label, g.coeff, orig
        );
    }

    // Check products of all pairs
    eprintln!("\nPairwise products:");
    let stab_group = StabilizerGroup::from_circuit(&gates, expanded.num_original_qubits);

    for j in 0..h_gens.len() {
        for k in (j + 1)..h_gens.len() {
            let qj = &h_gens[j].label;
            let qk = &h_gens[k].label;
            if !qj.commutes_with(qk) {
                continue; // Skip anticommuting pairs
            }
            let product = qj.multiply(qk);
            let orig_product = expanded.map_to_original_frame(&product);
            let is_stab = stab_group.is_stabilizer(&orig_product);

            if is_stab.is_some() || !orig_product.is_identity() {
                eprintln!(
                    "  [{j},{k}] commute=true product_orig={orig_product:?} is_stab={is_stab:?}"
                );
            }
        }
    }
}

#[test]
fn test_zbasis_stabilizer_group() {
    let gates = build_minimal_zbasis();
    // Exclude final MZ readout — keep syndrome MZ
    let last_non_mz = gates
        .iter()
        .rposition(|g| g.gate_type != GateType::MZ)
        .unwrap();
    let gates_pre = &gates[..=last_non_mz];
    let stab_group = StabilizerGroup::from_circuit(gates_pre, 3);

    // Dump actual generators
    eprintln!("Stabilizer generators:");
    // Run SparseStab manually to see generators
    let mut sim = SparseStab::with_seed(3, 0);
    for g in gates_pre {
        let qs: Vec<QubitId> = g.qubits.iter().copied().collect();
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
                let _r = sim.mz(&qs);
                eprintln!("  MZ({qs:?})");
            }
            _ => {}
        }
    }
    let stab_gens = sim.stabs().generators();
    for (i, g) in stab_gens.iter().enumerate() {
        eprintln!("  stab[{i}] = {g:?}");
    }

    // Check what's in the stabilizer group
    eprintln!("\nStabilizer group membership checks:");
    let test_paulis = vec![
        ("X0", Bm::x(0)),
        ("X1", Bm::x(1)),
        ("X0X1", Bm::x(0).multiply(&Bm::x(1))),
        ("Z0", Bm::z(0)),
        ("Z1", Bm::z(1)),
        ("Z0Z1", Bm::z(0).multiply(&Bm::z(1))),
        ("Z2", Bm::z(2)),
    ];

    for (name, p) in &test_paulis {
        let result = stab_group.is_stabilizer(p);
        eprintln!("  {name}: {result:?}");
    }

    // X0*X1 should be a stabilizer (from X-type syndrome extraction)
    assert_eq!(
        stab_group.is_stabilizer(&Bm::x(0).multiply(&Bm::x(1))),
        Some(true),
        "X0*X1 should be stabilizer after 2 rounds of X-stabilizer measurement"
    );
}
