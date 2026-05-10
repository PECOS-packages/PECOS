// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! Track the stabilizer group of the noiseless output state using SparseStab.

use crate::Bm;
use pecos_core::gate_type::GateType;
use pecos_core::pauli::pauli_bitmask::BitmaskStorage;
use pecos_core::{Gate, QuarterPhase, QubitId};
use pecos_simulators::{CliffordGateable, SparseStab};

/// The stabilizer group, tracked via SparseStab.
///
/// Keeps the full SparseStab (stabilizers + destabilizers) so we can
/// determine the exact sign (+1 or -1) of stabilizer group elements.
pub struct StabilizerGroup {
    sim: SparseStab,
}

impl StabilizerGroup {
    /// Run the noiseless circuit on SparseStab.
    #[must_use]
    pub fn from_circuit(gates: &[Gate], num_qubits: usize) -> Self {
        let mut sim = SparseStab::with_seed(num_qubits, 0);

        for gate in gates {
            let qubits: Vec<QubitId> = gate.qubits.iter().copied().collect();
            if qubits.is_empty() {
                continue;
            }

            match gate.gate_type {
                GateType::PZ | GateType::QAlloc => {
                    for &q in &qubits {
                        sim.pz(&[q]);
                    }
                }
                GateType::H => {
                    sim.h(&qubits);
                }
                GateType::SZ => {
                    sim.sz(&qubits);
                }
                GateType::SZdg => {
                    sim.szdg(&qubits);
                }
                GateType::SX => {
                    sim.sx(&qubits);
                }
                GateType::SXdg => {
                    sim.sxdg(&qubits);
                }
                GateType::SY => {
                    sim.sy(&qubits);
                }
                GateType::SYdg => {
                    sim.sydg(&qubits);
                }
                GateType::X => {
                    sim.x(&qubits);
                }
                GateType::Y => {
                    sim.y(&qubits);
                }
                GateType::Z => {
                    sim.z(&qubits);
                }
                GateType::CX if qubits.len() >= 2 => {
                    sim.cx(&[(qubits[0], qubits[1])]);
                }
                GateType::CY if qubits.len() >= 2 => {
                    sim.cy(&[(qubits[0], qubits[1])]);
                }
                GateType::CZ if qubits.len() >= 2 => {
                    sim.cz(&[(qubits[0], qubits[1])]);
                }
                GateType::SWAP if qubits.len() >= 2 => {
                    sim.swap(&[(qubits[0], qubits[1])]);
                }
                GateType::MZ => {
                    sim.mz(&qubits);
                }
                _ => {}
            }
        }

        Self { sim }
    }

    /// Check if Pauli P is in the stabilizer group and return its sign.
    ///
    /// Returns:
    /// - `Some(true)` if P is a +1 stabilizer
    /// - `Some(false)` if P is a -1 stabilizer (anti-stabilizer)
    /// - `None` if P is not in the stabilizer group
    #[must_use]
    pub fn is_stabilizer(&self, p: &Bm) -> Option<bool> {
        if p.is_identity() {
            return Some(true);
        }

        let mut x_positions = Vec::new();
        let mut z_positions = Vec::new();
        let mut num_ys = 0usize;

        let max_q = match (p.x_bits.highest_set_bit(), p.z_bits.highest_set_bit()) {
            (None, None) => return Some(true),
            (Some(a), None) | (None, Some(a)) => a + 1,
            (Some(a), Some(b)) => a.max(b) + 1,
        };

        for q in 0..max_q {
            let has_x = p.has_x(q);
            let has_z = p.has_z(q);
            if has_x {
                x_positions.push(q);
            }
            if has_z {
                z_positions.push(q);
            }
            if has_x && has_z {
                num_ys += 1;
            }
        }

        let stabs = self.sim.stabs();
        let destabs = self.sim.destabs();
        let phase = stabs.find_pauli_sign(destabs, x_positions, z_positions, num_ys)?;

        match phase {
            QuarterPhase::PlusOne => Some(true),
            QuarterPhase::MinusOne => Some(false),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::{GateAngles, GateParams};

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

    #[test]
    fn test_identity_circuit() {
        let stabs = StabilizerGroup::from_circuit(&[], 1);
        assert_eq!(stabs.is_stabilizer(&Bm::z(0)), Some(true));
        assert_eq!(stabs.is_stabilizer(&Bm::x(0)), None);
    }

    #[test]
    fn test_h_circuit() {
        let gates = vec![gate(GateType::H, &[0])];
        let stabs = StabilizerGroup::from_circuit(&gates, 1);
        assert_eq!(stabs.is_stabilizer(&Bm::x(0)), Some(true));
        assert_eq!(stabs.is_stabilizer(&Bm::z(0)), None);
    }

    #[test]
    fn test_bell_state() {
        let gates = vec![gate(GateType::H, &[0]), gate(GateType::CX, &[0, 1])];
        let stabs = StabilizerGroup::from_circuit(&gates, 2);
        assert_eq!(
            stabs.is_stabilizer(&Bm::x(0).multiply(&Bm::x(1))),
            Some(true)
        );
        assert_eq!(
            stabs.is_stabilizer(&Bm::z(0).multiply(&Bm::z(1))),
            Some(true)
        );
    }

    #[test]
    fn test_x0x1_after_syndrome_extraction() {
        let gates = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::PZ, &[1]),
            gate(GateType::PZ, &[4]),
            gate(GateType::H, &[4]),
            gate(GateType::CX, &[4, 1]),
            gate(GateType::CX, &[4, 0]),
            gate(GateType::H, &[4]),
            gate(GateType::MZ, &[4]),
            gate(GateType::PZ, &[4]),
        ];
        let stabs = StabilizerGroup::from_circuit(&gates, 5);

        let x0x1 = Bm::x(0).multiply(&Bm::x(1));
        assert_eq!(
            stabs.is_stabilizer(&x0x1),
            Some(true),
            "X0*X1 should be stabilizer after syndrome extraction with MZ projection"
        );
    }

    #[test]
    fn test_anti_stabilizer() {
        // After PZ, Z is +1 stabilizer. -Z should be anti-stabilizer.
        // But -Z isn't a Pauli label in our system (no phase in Bm).
        // Instead, apply X to flip the state to |1>, then Z has eigenvalue -1.
        let gates = vec![gate(GateType::X, &[0])];
        let stabs = StabilizerGroup::from_circuit(&gates, 1);
        // Initial state is |0>, X takes it to |1>.
        // Z|1> = -|1>, so Z is a -1 stabilizer.
        assert_eq!(
            stabs.is_stabilizer(&Bm::z(0)),
            Some(false),
            "Z should be -1 stabilizer for |1> state"
        );
    }

    #[test]
    fn test_sign_bell_minus() {
        // |Phi-> = (|00> - |11>)/sqrt(2) = CX H X |00>
        // X flips to |10>, H gives |−0>, CX gives |Phi->
        // Stabilizers: -XX, +ZZ
        let gates = vec![
            gate(GateType::X, &[0]),
            gate(GateType::H, &[0]),
            gate(GateType::CX, &[0, 1]),
        ];
        let stabs = StabilizerGroup::from_circuit(&gates, 2);
        // XX should be -1 stabilizer (the minus Bell state)
        assert_eq!(
            stabs.is_stabilizer(&Bm::x(0).multiply(&Bm::x(1))),
            Some(false),
            "XX should be -1 stabilizer for |Phi->"
        );
        // ZZ should be +1 stabilizer
        assert_eq!(
            stabs.is_stabilizer(&Bm::z(0).multiply(&Bm::z(1))),
            Some(true),
            "ZZ should be +1 stabilizer for |Phi->"
        );
    }
}
