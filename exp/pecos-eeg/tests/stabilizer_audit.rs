// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! Audit: does the `StabilizerGroup` correctly identify all stabilizers?
//! Test by generating all 2^n products of n generators and checking
//! that `is_stabilizer` returns Some for each.

use pecos_core::gate_type::GateType;
use pecos_core::{Gate, GateAngles, GateParams, QubitId};
use pecos_eeg::Bm;
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

/// Extract generators as Bm from `SparseStab`.
fn extract_generators(sim: &SparseStab) -> Vec<Bm> {
    let stabs = sim.stabs();
    let n = stabs.num_generators();
    let mut gens = Vec::with_capacity(n);
    for i in 0..n {
        let ps = stabs.generator(i);
        gens.push(pecos_eeg::dem_mapping::pauli_string_to_bitmask(&ps));
    }
    gens
}

/// Check that `StabilizerGroup.is_stabilizer` returns Some for ALL products
/// of the `SparseStab` generators (which are by definition in the group).
fn audit_stabilizer_group(label: &str, gates: &[Gate], num_qubits: usize) {
    let stab_group = StabilizerGroup::from_circuit(gates, num_qubits);

    // Also build a raw SparseStab to extract generators
    let mut sim = SparseStab::with_seed(num_qubits, 0);
    for g in gates {
        let qs: Vec<QubitId> = g.qubits.iter().copied().collect();
        if qs.is_empty() {
            continue;
        }
        match g.gate_type {
            GateType::PZ | GateType::QAlloc => {
                for &q in &qs {
                    sim.pz(&[q]);
                }
            }
            GateType::H => {
                sim.h(&qs);
            }
            GateType::SZ => {
                sim.sz(&qs);
            }
            GateType::SZdg => {
                sim.szdg(&qs);
            }
            GateType::X => {
                sim.x(&qs);
            }
            GateType::Y => {
                sim.y(&qs);
            }
            GateType::Z => {
                sim.z(&qs);
            }
            GateType::CX if qs.len() >= 2 => {
                sim.cx(&[(qs[0], qs[1])]);
            }
            GateType::CZ if qs.len() >= 2 => {
                sim.cz(&[(qs[0], qs[1])]);
            }
            GateType::MZ => {
                sim.mz(&qs);
            }
            _ => {}
        }
    }

    let generators = extract_generators(&sim);
    let n = generators.len();

    eprintln!("\n{label}: {n} generators on {num_qubits} qubits");
    for (i, g) in generators.iter().enumerate() {
        eprintln!("  gen[{i}] = {g:?}");
    }

    // Test all 2^n products (for small n)
    let max_subsets = if n <= 10 { 1 << n } else { 1024 }; // cap at 1024 for large n
    let mut failures = Vec::new();

    for mask in 0..max_subsets {
        let mut product = Bm::default();
        for (i, generator) in generators.iter().enumerate().take(n) {
            if mask & (1 << i) != 0 {
                product = product.multiply(generator);
            }
        }

        let result = stab_group.is_stabilizer(&product);
        if result.is_none() && !product.is_identity() {
            failures.push((mask, product));
        }
    }

    if failures.is_empty() {
        eprintln!("  OK: all {max_subsets} products correctly identified");
    } else {
        eprintln!("  FAILURES: {} products not found:", failures.len());
        for (mask, product) in &failures {
            let gens_used: Vec<usize> = (0..n).filter(|&i| mask & (1 << i) != 0).collect();
            eprintln!("    mask={mask:#06b} gens={gens_used:?} product={product:?}");
        }
    }

    assert!(
        failures.is_empty(),
        "{label}: {}/{max_subsets} stabilizer products not found by is_stabilizer",
        failures.len()
    );
}

#[test]
fn audit_simple_states() {
    // |0>: stabilizer = Z
    audit_stabilizer_group("|0>", &[], 1);

    // |+>: stabilizer = X
    audit_stabilizer_group("|+>", &[gate(GateType::H, &[0])], 1);

    // Bell state
    audit_stabilizer_group(
        "|Phi+>",
        &[gate(GateType::H, &[0]), gate(GateType::CX, &[0, 1])],
        2,
    );
}

#[test]
fn audit_syndrome_extraction() {
    // Simple 2-qubit Z-check with ancilla: PZ(0,1,2), CX(0,2), CX(1,2), MZ(2)
    audit_stabilizer_group(
        "Z-check 2q",
        &[
            gate(GateType::PZ, &[0]),
            gate(GateType::PZ, &[1]),
            gate(GateType::PZ, &[2]),
            gate(GateType::CX, &[0, 2]),
            gate(GateType::CX, &[1, 2]),
            gate(GateType::MZ, &[2]),
        ],
        3,
    );

    // X-check: H(2), CX(2,0), CX(2,1), H(2), MZ(2)
    audit_stabilizer_group(
        "X-check 2q",
        &[
            gate(GateType::PZ, &[0]),
            gate(GateType::PZ, &[1]),
            gate(GateType::PZ, &[2]),
            gate(GateType::H, &[2]),
            gate(GateType::CX, &[2, 0]),
            gate(GateType::CX, &[2, 1]),
            gate(GateType::H, &[2]),
            gate(GateType::MZ, &[2]),
        ],
        3,
    );
}

#[test]
fn audit_d2_zbasis_pre_readout() {
    // d=2 Z-basis surface code, 2 rounds, pre-readout circuit
    let gates = vec![
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
        gate(GateType::CX, &[5, 3]),
        gate(GateType::CX, &[3, 6]),
        gate(GateType::CX, &[5, 2]),
        gate(GateType::CX, &[4, 1]),
        gate(GateType::CX, &[0, 6]),
        gate(GateType::CX, &[4, 0]),
        gate(GateType::CX, &[2, 6]),
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
        gate(GateType::CX, &[5, 3]),
        gate(GateType::CX, &[3, 6]),
        gate(GateType::CX, &[5, 2]),
        gate(GateType::CX, &[4, 1]),
        gate(GateType::CX, &[0, 6]),
        gate(GateType::CX, &[4, 0]),
        gate(GateType::CX, &[2, 6]),
        gate(GateType::H, &[4]),
        gate(GateType::H, &[5]),
        gate(GateType::MZ, &[4]),
        gate(GateType::MZ, &[5]),
        gate(GateType::MZ, &[6]),
    ];
    audit_stabilizer_group("d=2 Z-basis pre-readout", &gates, 7);
}
