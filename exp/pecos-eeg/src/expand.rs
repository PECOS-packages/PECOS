// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! Deferred measurement: expand a circuit by replacing mid-circuit
//! MZ+PZ with CX to auxiliary qubits, deferring all measurements to the end.
//!
//! After expansion, the circuit is purely Clifford (no mid-circuit
//! measurements), and error generators can be propagated straight through.

use crate::Bm;
use pecos_core::gate_type::GateType;
use pecos_core::pauli::pauli_bitmask::BitmaskStorage;
use pecos_core::{Gate, GateAngles, GateParams, QubitId};

/// Result of circuit expansion.
pub struct ExpandedCircuit {
    /// The expanded gate sequence (purely Clifford, no mid-circuit measurements).
    pub gates: Vec<Gate>,
    /// Total number of qubits (original + auxiliary).
    pub num_qubits: usize,
    /// Number of original qubits.
    pub num_original_qubits: usize,
    /// Mapping: measurement record index → auxiliary qubit index.
    /// measurement_qubit[k] = the auxiliary qubit whose Z-measurement at
    /// the end gives the k-th measurement record.
    pub measurement_qubit: Vec<usize>,
    /// Mapping: measurement record index → original qubit that was measured.
    /// original_measured_qubit[k] = the qubit in the original circuit that
    /// the k-th MZ gate acted on.
    pub original_measured_qubit: Vec<usize>,
}

/// Expand a circuit by deferring mid-circuit measurements.
///
/// For each MZ(q) followed by PZ(q), replaces with:
/// 1. CX(q, aux) — copy q's state to a fresh auxiliary qubit
/// 2. PZ(q) — reset q to |0> (kept as-is, since PZ after CX is valid)
///
/// All auxiliary qubits are measured at the end via MZ.
/// Final data measurements (MZ not followed by PZ) are also deferred
/// to auxiliary qubits for uniformity.
pub fn expand_circuit(gates: &[Gate]) -> ExpandedCircuit {
    // First pass: find the max qubit index to know where auxiliaries start
    let max_qubit = gates
        .iter()
        .flat_map(|g| g.qubits.iter())
        .map(pecos_core::QubitId::index)
        .max()
        .unwrap_or(0);
    let num_original = max_qubit + 1;
    let mut next_aux = num_original;

    let mut expanded = Vec::with_capacity(gates.len() * 2);
    let mut measurement_qubit = Vec::new();
    let mut original_measured_qubit = Vec::new();

    // Identify which MZ gates are mid-circuit (followed by PZ on same qubit)
    // vs final (not followed by any operation on that qubit, or followed by
    // a different operation).
    //
    // Track ancilla qubits: those with a PZ after the first non-PZ gate.
    // Only ancilla MZ gets a post-expansion PZ (measurement projection).
    let mut ancilla_qubits = std::collections::HashSet::new();
    {
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
    }

    // Strategy: walk gates, when we see MZ(q):
    //   - Replace with CX(q, aux_new) where aux_new is a fresh auxiliary
    //   - For ancilla qubits: add PZ(q) to model measurement projection
    //   - Record that measurement k maps to aux_new
    //   - The auxiliary qubit is measured at the end

    let mut i = 0;
    while i < gates.len() {
        let gate = &gates[i];

        match gate.gate_type {
            GateType::MZ => {
                // For each qubit in this MZ gate, create CX to auxiliary
                for q in &gate.qubits {
                    let q_idx = q.index();
                    let aux = next_aux;
                    next_aux += 1;

                    // Initialize auxiliary: QAlloc(aux)
                    // Use QAlloc (not PZ) so the noise model can distinguish
                    // auxiliary initialization from original circuit resets.
                    expanded.push(make_gate(GateType::QAlloc, &[aux]));

                    // CX(q, aux) — copy measurement info to auxiliary
                    expanded.push(make_gate(GateType::CX, &[q_idx, aux]));

                    // For ancilla qubits: add PZ to model measurement projection.
                    // MZ projects to a Z eigenstate, destroying X/Y coherences.
                    // Without this, the last round's syndrome generators retain
                    // X on the measured ancilla, creating spurious correlations.
                    // For intermediate rounds this is redundant (circuit PZ follows).
                    //
                    // Data readout MZ does NOT get PZ: data qubit Z components
                    // must persist for correct generator labels (Z errors are
                    // invisible to Z-basis readout and must not be cleared).
                    if ancilla_qubits.contains(&q_idx) {
                        expanded.push(make_gate(GateType::PZ, &[q_idx]));
                    }

                    // Record: this measurement maps to the auxiliary qubit
                    measurement_qubit.push(aux);
                    original_measured_qubit.push(q_idx);
                }
            }
            GateType::PZ | GateType::QAlloc => {
                // Keep resets — they re-initialize the qubit for the next round
                expanded.push(gate.clone());
            }
            _ => {
                // All other gates pass through unchanged
                expanded.push(gate.clone());
            }
        }

        i += 1;
    }

    // Add final measurements of all auxiliary qubits at the end
    for &aux in &measurement_qubit {
        expanded.push(make_gate(GateType::MZ, &[aux]));
    }

    ExpandedCircuit {
        gates: expanded,
        num_qubits: next_aux,
        num_original_qubits: num_original,
        measurement_qubit,
        original_measured_qubit,
    }
}

impl ExpandedCircuit {
    /// Map an expanded-circuit Pauli back to the original circuit frame.
    ///
    /// X on auxiliary qubit `aux_k` → X on `original_measured_qubit[k]`
    /// (because `CX(q, aux)` copies X from control to target: X on aux
    /// in the expanded circuit corresponds to X on q in the original).
    ///
    /// Z on auxiliary qubits is dropped (doesn't correspond to original).
    /// Components on original qubits pass through unchanged.
    #[must_use]
    pub fn map_to_original_frame(&self, p: &Bm) -> Bm {
        let mut result = Bm::default();

        // Copy components on original qubits directly
        for q in 0..self.num_original_qubits {
            if p.has_x(q) {
                result.x_bits.set_bit(q);
            }
            if p.has_z(q) {
                result.z_bits.set_bit(q);
            }
        }

        // Map X on auxiliary qubits to X on original measured qubits
        for (meas_idx, &aux_q) in self.measurement_qubit.iter().enumerate() {
            if p.has_x(aux_q) {
                let orig_q = self.original_measured_qubit[meas_idx];
                result.x_bits.xor_bit(orig_q); // XOR because same qubit may be measured multiple times
            }
            // Z on aux is ignored (measurement projection absorbs Z)
        }

        result
    }
}

/// Precomputed qubit-to-gate index for sparse backward traversal.
///
/// For each qubit, stores the gate indices (in the flat gate list) that
/// touch it, sorted in ascending order. This enables the backward walk
/// to visit only gates on active qubits instead of scanning all gates.
pub struct GateIndex {
    /// qubit_gates[q] = sorted Vec of gate indices touching qubit q.
    qubit_gates: Vec<Vec<u32>>,
    /// Which gates are expansion gates (no physical noise).
    pub expansion_gates: Vec<bool>,
}

impl GateIndex {
    /// Build the index from a gate list (typically the expanded circuit).
    #[must_use]
    pub fn build(gates: &[Gate], num_qubits: usize) -> Self {
        let mut qubit_gates = vec![Vec::new(); num_qubits];

        for (i, gate) in gates.iter().enumerate() {
            for q in &gate.qubits {
                qubit_gates[q.index()].push(i as u32);
            }
        }

        // Identify expansion gates (QAlloc + subsequent CX + PZ)
        let mut expansion = vec![false; gates.len()];
        for i in 0..gates.len() {
            if gates[i].gate_type == GateType::QAlloc {
                expansion[i] = true;
            }
        }
        for i in 1..gates.len() {
            if gates[i].gate_type == GateType::CX && gates[i - 1].gate_type == GateType::QAlloc {
                let alloc_q = gates[i - 1].qubits[0].index();
                if gates[i].qubits.len() >= 2 && gates[i].qubits[1].index() == alloc_q {
                    expansion[i] = true;
                    if i + 1 < gates.len()
                        && gates[i + 1].gate_type == GateType::PZ
                        && gates[i + 1].qubits[0].index() == gates[i].qubits[0].index()
                    {
                        expansion[i + 1] = true;
                    }
                }
            }
        }

        Self {
            qubit_gates,
            expansion_gates: expansion,
        }
    }

    /// Gate indices touching qubit `q` in reverse order (for backward walk).
    pub fn gates_on_qubit_rev(&self, q: usize) -> impl Iterator<Item = u32> + '_ {
        self.qubit_gates
            .get(q)
            .into_iter()
            .flat_map(|v| v.iter().copied().rev())
    }

    /// Is this gate an expansion gate (no physical noise)?
    #[inline]
    #[must_use]
    pub fn is_expansion(&self, gate_idx: usize) -> bool {
        self.expansion_gates.get(gate_idx).copied().unwrap_or(false)
    }
}

#[must_use]
pub fn make_gate(gt: GateType, qubits: &[usize]) -> Gate {
    Gate {
        gate_type: gt,
        qubits: qubits.iter().map(|&q| QubitId(q)).collect(),
        angles: GateAngles::new(),
        params: GateParams::new(),
        meas_ids: pecos_core::GateMeasIds::new(),
        channel: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate(gt: GateType, qubits: &[usize]) -> Gate {
        make_gate(gt, qubits)
    }

    #[test]
    fn test_expand_simple_mcm() {
        // PZ(0), H(0), MZ(0), PZ(0), H(0), MZ(0)
        // Two rounds: measure, reset, measure again
        let gates = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::H, &[0]),
            gate(GateType::MZ, &[0]), // → CX(0, aux0)
            gate(GateType::PZ, &[0]), // reset
            gate(GateType::H, &[0]),
            gate(GateType::MZ, &[0]), // → CX(0, aux1)
        ];

        let expanded = expand_circuit(&gates);

        // Should have 3 qubits: original 0, aux 1, aux 2
        assert_eq!(expanded.num_original_qubits, 1);
        assert_eq!(expanded.num_qubits, 3);
        assert_eq!(expanded.measurement_qubit.len(), 2);

        // No MZ in the middle — only at the end
        let mid_mz = expanded.gates[..expanded.gates.len() - 2]
            .iter()
            .filter(|g| g.gate_type == GateType::MZ)
            .count();
        assert_eq!(mid_mz, 0, "No mid-circuit MZ in expanded circuit");

        // Two MZ at the end (one per auxiliary)
        let end_mz = expanded
            .gates
            .iter()
            .rev()
            .take_while(|g| g.gate_type == GateType::MZ)
            .count();
        assert_eq!(end_mz, 2);
    }

    #[test]
    fn test_expand_preserves_cliffords() {
        let gates = vec![
            gate(GateType::PZ, &[0, 1]),
            gate(GateType::H, &[0]),
            gate(GateType::CX, &[0, 1]),
        ];

        let expanded = expand_circuit(&gates);

        // No measurements → no expansion needed
        assert_eq!(expanded.num_qubits, 2);
        assert_eq!(expanded.measurement_qubit.len(), 0);
        assert_eq!(expanded.gates.len(), 3); // same gates
    }

    #[test]
    fn test_measurement_qubit_mapping() {
        // 2 qubits, measure both
        let gates = vec![
            gate(GateType::PZ, &[0, 1]),
            gate(GateType::H, &[0]),
            gate(GateType::CX, &[0, 1]),
            gate(GateType::MZ, &[0]), // meas record 0 → aux 2
            gate(GateType::MZ, &[1]), // meas record 1 → aux 3
        ];

        let expanded = expand_circuit(&gates);

        assert_eq!(expanded.measurement_qubit, vec![2, 3]);
        assert_eq!(expanded.num_qubits, 4);
    }

    #[test]
    fn test_map_to_original_frame_x_on_aux() {
        // X on auxiliary → X on original measured qubit
        let gates = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::MZ, &[0]), // meas 0 → aux 1
        ];
        let expanded = expand_circuit(&gates);

        // X on aux 1 maps to X on original qubit 0
        let p = Bm::x(1); // aux qubit
        let mapped = expanded.map_to_original_frame(&p);
        assert_eq!(mapped, Bm::x(0));
    }

    #[test]
    fn test_map_to_original_frame_z_on_aux_dropped() {
        // Z on auxiliary is dropped (measurement projection absorbs it)
        let gates = vec![gate(GateType::PZ, &[0]), gate(GateType::MZ, &[0])];
        let expanded = expand_circuit(&gates);

        let p = Bm::z(1); // Z on aux
        let mapped = expanded.map_to_original_frame(&p);
        assert!(mapped.is_identity(), "Z on aux should be dropped");
    }

    #[test]
    fn test_map_to_original_frame_original_passthrough() {
        // Components on original qubits pass through unchanged
        let gates = vec![gate(GateType::PZ, &[0, 1]), gate(GateType::MZ, &[0])];
        let expanded = expand_circuit(&gates);

        let p = Bm::x(0).multiply(&Bm::z(1)); // X0 Z1
        let mapped = expanded.map_to_original_frame(&p);
        assert_eq!(mapped, Bm::x(0).multiply(&Bm::z(1)));
    }

    #[test]
    fn test_expand_final_only_mz() {
        // Circuit with only final MZ (no mid-circuit measurement)
        let gates = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::H, &[0]),
            gate(GateType::MZ, &[0]),
        ];
        let expanded = expand_circuit(&gates);

        // Still creates one aux qubit for the final MZ
        assert_eq!(expanded.num_qubits, 2);
        assert_eq!(expanded.measurement_qubit.len(), 1);
    }

    #[test]
    fn test_expansion_pz_for_ancilla_mz() {
        // Circuit: PZ(0,1), H(1), CX(1,0), MZ(1), PZ(1), H(1), CX(1,0), MZ(1), MZ(0)
        // Qubit 1 is ancilla (has mid-circuit PZ). Qubit 0 is data.
        // Last-round MZ(1) should get expansion PZ(1).
        // Final MZ(0) should NOT get expansion PZ.
        let gates = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::PZ, &[1]),
            gate(GateType::H, &[1]),
            gate(GateType::CX, &[1, 0]),
            gate(GateType::MZ, &[1]), // round 1 syndrome
            gate(GateType::PZ, &[1]), // reset
            gate(GateType::H, &[1]),
            gate(GateType::CX, &[1, 0]),
            gate(GateType::MZ, &[1]), // round 2 syndrome (last round, no PZ after)
            gate(GateType::MZ, &[0]), // data readout
        ];

        let expanded = expand_circuit(&gates);

        // Count PZ/QAlloc gates on qubit 1 in the expanded circuit
        let resets_on_1: Vec<_> = expanded
            .gates
            .iter()
            .filter(|g| {
                (g.gate_type == GateType::PZ || g.gate_type == GateType::QAlloc)
                    && g.qubits.iter().any(|q| q.index() == 1)
            })
            .collect();

        // Should have: original PZ(1) init + expansion PZ(1) round 1 + circuit PZ(1) reset
        //            + expansion PZ(1) round 2 = 4 reset gates on qubit 1
        eprintln!("Resets on qubit 1: {} gates", resets_on_1.len());
        assert!(
            resets_on_1.len() >= 4,
            "Should have expansion PZ for last-round MZ(1): got {} on q1",
            resets_on_1.len()
        );

        // Count resets on qubit 0 in expanded circuit
        let resets_on_0: Vec<_> = expanded
            .gates
            .iter()
            .filter(|g| {
                (g.gate_type == GateType::PZ || g.gate_type == GateType::QAlloc)
                    && g.qubits.iter().any(|q| q.index() == 0)
            })
            .collect();
        // Should have only: original PZ(0) init = 1
        eprintln!("Resets on qubit 0: {} gates", resets_on_0.len());
        assert_eq!(
            resets_on_0.len(),
            1,
            "Data qubit should NOT get expansion PZ"
        );
    }

    #[test]
    fn test_expand_multi_round_tracks_original_qubits() {
        // Two rounds measuring qubit 0: both aux should map back to qubit 0
        let gates = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::MZ, &[0]),
            gate(GateType::PZ, &[0]),
            gate(GateType::MZ, &[0]),
        ];
        let expanded = expand_circuit(&gates);

        assert_eq!(expanded.measurement_qubit.len(), 2);
        assert_eq!(expanded.original_measured_qubit, vec![0, 0]);
    }
}
