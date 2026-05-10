// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! EEG circuit analysis on the expanded (measurement-deferred) circuit.
//!
//! After expansion, the circuit is purely Clifford. Error generators are
//! propagated straight to the end via Clifford conjugation (no measurement
//! absorption). At the end, each Pauli P flips measurement k iff P has
//! X on measurement_qubit[k].

use crate::Bm;
use crate::eeg::EegType;
use pecos_core::Gate;
use pecos_core::gate_type::GateType;
use pecos_core::pauli::pauli_bitmask::{
    BitmaskStorage, Conjugated, conjugate_cx, conjugate_cy, conjugate_cz, conjugate_h,
    conjugate_swap, conjugate_sx, conjugate_sxdg, conjugate_sy, conjugate_sydg, conjugate_sz,
    conjugate_szdg, conjugate_x, conjugate_y, conjugate_z,
};

/// Noise model parameters.
#[derive(Clone, Debug)]
pub struct NoiseModel {
    /// Coherent RZ angle (radians) on both qubits after each 2-qubit gate.
    pub idle_rz: f64,
    /// Single-qubit depolarizing probability.
    pub p1: f64,
    /// Two-qubit depolarizing probability.
    pub p2: f64,
    /// Measurement bit-flip probability.
    pub p_meas: f64,
    /// Preparation error probability.
    pub p_prep: f64,
}

impl NoiseModel {
    #[must_use]
    pub fn coherent_only(idle_rz: f64) -> Self {
        Self {
            idle_rz,
            p1: 0.0,
            p2: 0.0,
            p_meas: 0.0,
            p_prep: 0.0,
        }
    }

    #[must_use]
    pub fn depolarizing(p: f64) -> Self {
        Self {
            idle_rz: 0.0,
            p1: p,
            p2: p,
            p_meas: p,
            p_prep: p,
        }
    }

    #[must_use]
    pub fn with_idle_rz(mut self, angle: f64) -> Self {
        self.idle_rz = angle;
        self
    }
}

/// Identifies the physical noise source that produced a generator.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NoiseSource {
    /// Index of the gate in the expanded circuit that the noise follows.
    pub gate_index: usize,
    /// Qubit the noise acts on (for per-qubit noise like idle RZ).
    pub qubit: usize,
}

/// A propagated EEG generator at the end of the expanded circuit.
#[derive(Clone, Debug)]
pub struct PropagatedEeg {
    /// EEG type (H, S, C, or A).
    pub eeg_type: EegType,
    /// Primary Pauli label at end of circuit.
    pub label: Bm,
    /// Second Pauli label for C and A types (None for H and S).
    pub label2: Option<Bm>,
    /// Coefficient (rate). For H: includes sign from conjugation. For S: always negative.
    pub coeff: f64,
    /// Physical noise source that produced this generator (for sensitivity analysis).
    pub source: Option<NoiseSource>,
}

/// Result of EEG analysis on the expanded circuit.
#[derive(Clone, Debug)]
pub struct EegAnalysisResult {
    /// All propagated generators at end of circuit.
    pub generators: Vec<PropagatedEeg>,
    /// Number of measurement records.
    pub num_measurements: usize,
}

impl EegAnalysisResult {
    /// Generator fidelity ε_gen = Σ h_P² + Σ |s_P| (Hines Eq. 2/Eq. 6).
    ///
    /// Measures the total "size" of the error. The DEM prediction error
    /// (TVD) scales as ε_gen^{1.5}.
    #[must_use]
    pub fn generator_fidelity(&self) -> f64 {
        let mut eps = 0.0;
        for g in &self.generators {
            match g.eeg_type {
                EegType::H => eps += g.coeff * g.coeff,
                EegType::S => eps += g.coeff.abs(),
                _ => {}
            }
        }
        eps
    }
}

/// Analyze the expanded circuit with a flexible noise specification.
///
/// For each gate, calls `noise.noise_after_gate()` to get generators,
/// then propagates them to the end via Clifford conjugation.
///
/// Expansion gates (QAlloc, expansion CX, expansion PZ) are skipped
/// for noise injection.
pub fn analyze_with_noise(
    gates: &[Gate],
    noise: &dyn crate::noise::NoiseSpec,
) -> EegAnalysisResult {
    let mut generators = Vec::new();
    let mut num_measurements = 0;

    // Build sets of expansion gate indices
    let mut expansion_cx_indices = std::collections::HashSet::new();
    let mut expansion_pz_indices = std::collections::HashSet::new();
    for i in 1..gates.len() {
        if gates[i].gate_type == GateType::CX && gates[i - 1].gate_type == GateType::QAlloc {
            let alloc_q = gates[i - 1].qubits[0].index();
            if gates[i].qubits.len() >= 2 && gates[i].qubits[1].index() == alloc_q {
                expansion_cx_indices.insert(i);
                if i + 1 < gates.len() && gates[i + 1].gate_type == GateType::PZ {
                    let cx_control = gates[i].qubits[0].index();
                    if gates[i + 1].qubits.len() == 1
                        && gates[i + 1].qubits[0].index() == cx_control
                    {
                        expansion_pz_indices.insert(i + 1);
                    }
                }
            }
        }
    }

    for (i, gate) in gates.iter().enumerate() {
        let remaining = &gates[i + 1..];
        let qubits: Vec<usize> = gate.qubits.iter().map(pecos_core::QubitId::index).collect();

        // Skip expansion gates (virtual, not physical)
        let is_expansion = expansion_cx_indices.contains(&i)
            || expansion_pz_indices.contains(&i)
            || gate.gate_type == GateType::QAlloc;

        if !is_expansion {
            // Get noise generators from the noise specification
            let injections = noise.noise_after_gate(i, gate.gate_type, &qubits);

            for inj in injections {
                match inj.eeg_type {
                    EegType::H => {
                        let (pl, coeff) = propagate_h(inj.label, inj.rate, remaining);
                        generators.push(PropagatedEeg {
                            eeg_type: EegType::H,
                            label: pl,
                            label2: None,
                            coeff,
                            source: Some(NoiseSource {
                                gate_index: i,
                                qubit: qubits.first().copied().unwrap_or(0),
                            }),
                        });
                    }
                    EegType::S => {
                        let (pl, _) = propagate_s(inj.label, remaining);
                        generators.push(PropagatedEeg {
                            eeg_type: EegType::S,
                            label: pl,
                            label2: None,
                            coeff: inj.rate,
                            source: None,
                        });
                    }
                    EegType::C | EegType::A => {
                        if let Some(label2) = inj.label2 {
                            let (l1, l2, coeff) =
                                propagate_ca(inj.label, label2, inj.rate, remaining);
                            generators.push(PropagatedEeg {
                                eeg_type: inj.eeg_type,
                                label: l1,
                                label2: Some(l2),
                                coeff,
                                source: None,
                            });
                        }
                    }
                }
            }
        }

        // Handle explicit RZ gates (from the circuit, not noise model)
        if gate.gate_type == GateType::RZ
            && let Some(&angle) = gate.angles.first()
        {
            for &q in &qubits {
                let label = Bm::z(q);
                let (pl, coeff) = propagate_h(label, angle.to_radians() / 2.0, remaining);
                generators.push(PropagatedEeg {
                    eeg_type: EegType::H,
                    label: pl,
                    label2: None,
                    coeff,
                    source: Some(NoiseSource {
                        gate_index: i,
                        qubit: q,
                    }),
                });
            }
        }

        // Count measurements
        if gate.gate_type == GateType::MZ {
            num_measurements += qubits.len();
        }
    }

    EegAnalysisResult {
        generators,
        num_measurements,
    }
}

/// Analyze the expanded circuit with the legacy NoiseModel.
///
/// Delegates to `analyze_with_noise` using a `UniformNoise` specification.
#[must_use]
pub fn analyze_expanded(gates: &[Gate], noise: &NoiseModel) -> EegAnalysisResult {
    let uniform = crate::noise::UniformNoise {
        idle_rz: noise.idle_rz,
        p1: noise.p1,
        p2: noise.p2,
        p_meas: noise.p_meas,
        p_prep: noise.p_prep,
    };
    analyze_with_noise(gates, &uniform)
}

/// Propagate H_P forward: sign changes under Clifford conjugation.
/// PZ/QAlloc clears all Pauli components on the reset qubit.
fn propagate_h(mut label: Bm, mut coeff: f64, remaining: &[Gate]) -> (Bm, f64) {
    for gate in remaining {
        match gate.gate_type {
            GateType::PZ | GateType::QAlloc => {
                // Reset removes any error on this qubit
                for q in &gate.qubits {
                    label.x_bits.clear_bit(q.index());
                    label.z_bits.clear_bit(q.index());
                }
            }
            _ => {
                if let Some(r) = conjugate_by_gate(&label, gate) {
                    label = r.label;
                    if r.sign_negative {
                        coeff = -coeff;
                    }
                }
            }
        }
    }
    (label, coeff)
}

/// Propagate C_{P,Q} or A_{P,Q} forward: both labels conjugate, signs multiply.
/// gamma(C_{P,Q}, U) = s_{U,P} * s_{U,Q}. Same for A.
/// PZ/QAlloc clears components on both labels.
fn propagate_ca(
    mut label1: Bm,
    mut label2: Bm,
    mut coeff: f64,
    remaining: &[Gate],
) -> (Bm, Bm, f64) {
    for gate in remaining {
        match gate.gate_type {
            GateType::PZ | GateType::QAlloc => {
                for q in &gate.qubits {
                    label1.x_bits.clear_bit(q.index());
                    label1.z_bits.clear_bit(q.index());
                    label2.x_bits.clear_bit(q.index());
                    label2.z_bits.clear_bit(q.index());
                }
            }
            _ => {
                let mut sign = false;
                if let Some(r) = conjugate_by_gate(&label1, gate) {
                    label1 = r.label;
                    if r.sign_negative {
                        sign = !sign;
                    }
                }
                if let Some(r) = conjugate_by_gate(&label2, gate) {
                    label2 = r.label;
                    if r.sign_negative {
                        sign = !sign;
                    }
                }
                if sign {
                    coeff = -coeff;
                }
            }
        }
    }
    (label1, label2, coeff)
}

/// Propagate S_P forward: no sign change (gamma(S_P, U) = 1 always).
/// PZ/QAlloc clears all Pauli components on the reset qubit.
fn propagate_s(mut label: Bm, remaining: &[Gate]) -> (Bm, f64) {
    for gate in remaining {
        match gate.gate_type {
            GateType::PZ | GateType::QAlloc => {
                for q in &gate.qubits {
                    label.x_bits.clear_bit(q.index());
                    label.z_bits.clear_bit(q.index());
                }
            }
            _ => {
                if let Some(r) = conjugate_by_gate(&label, gate) {
                    label = r.label;
                }
            }
        }
    }
    (label, 0.0)
}

fn conjugate_by_gate(label: &Bm, gate: &Gate) -> Option<Conjugated<smallvec::SmallVec<[u64; 8]>>> {
    if gate.qubits.is_empty() {
        return None;
    }
    let q0 = || gate.qubits[0].index();
    let q1 = || gate.qubits[1].index();
    match gate.gate_type {
        GateType::H => Some(conjugate_h(label, q0())),
        GateType::SZ => Some(conjugate_sz(label, q0())),
        GateType::SZdg => Some(conjugate_szdg(label, q0())),
        GateType::SX => Some(conjugate_sx(label, q0())),
        GateType::SXdg => Some(conjugate_sxdg(label, q0())),
        GateType::SY => Some(conjugate_sy(label, q0())),
        GateType::SYdg => Some(conjugate_sydg(label, q0())),
        GateType::X => Some(conjugate_x(label, q0())),
        GateType::Y => Some(conjugate_y(label, q0())),
        GateType::Z => Some(conjugate_z(label, q0())),
        GateType::CX => Some(conjugate_cx(label, q0(), q1())),
        GateType::CY => Some(conjugate_cy(label, q0(), q1())),
        GateType::CZ => Some(conjugate_cz(label, q0(), q1())),
        GateType::SWAP => Some(conjugate_swap(label, q0(), q1())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::{GateAngles, GateParams, QubitId};

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
    fn test_rz_rate_is_half_theta() {
        // Idle RZ(0.1) after CX: rate should be 0.05 (theta/2)
        // because RZ(theta) = exp(-i*theta*Z/2) → H_Z with rate theta/2
        let gates = vec![gate(GateType::CX, &[0, 1])];
        let noise = NoiseModel::coherent_only(0.1);
        let result = analyze_expanded(&gates, &noise);

        let h_gens: Vec<_> = result
            .generators
            .iter()
            .filter(|g| g.eeg_type == EegType::H)
            .collect();
        assert_eq!(h_gens.len(), 2);
        for g in &h_gens {
            assert!(
                (g.coeff.abs() - 0.05).abs() < 1e-10,
                "Rate should be 0.05 (theta/2), got {}",
                g.coeff
            );
        }
    }

    #[test]
    fn test_h_propagation_through_hadamard() {
        // H_Z after H gate: Z → X, sign positive
        let gates = vec![gate(GateType::CX, &[0, 1]), gate(GateType::H, &[0])];
        let noise = NoiseModel::coherent_only(0.1);
        let result = analyze_expanded(&gates, &noise);

        let q0_gen = result
            .generators
            .iter()
            .find(|g| g.eeg_type == EegType::H && g.label.has_x(0))
            .expect("Should have H_X on qubit 0");
        assert!(
            (q0_gen.coeff - 0.05).abs() < 1e-10,
            "H: Z→X, rate=theta/2=0.05"
        );
    }

    #[test]
    fn test_sx_propagation() {
        // SX on qubit 1 after CX: Z1 → -Y1 (sign flip)
        let gates = vec![gate(GateType::CX, &[0, 1]), gate(GateType::SX, &[1])];
        let noise = NoiseModel::coherent_only(0.1);
        let result = analyze_expanded(&gates, &noise);

        // H_Z(1) propagated through SX(1): Z→-Y, coeff flips sign
        let q1_gen = result
            .generators
            .iter()
            .find(|g| g.eeg_type == EegType::H && g.label.has_x(1) && g.label.has_z(1))
            .expect("Should have H_Y on qubit 1 after SX");
        assert!(
            (q1_gen.coeff + 0.05).abs() < 1e-10,
            "SX: Z→-Y, sign flips: expected -0.05, got {}",
            q1_gen.coeff
        );

        // H_Z(0) should be unaffected by SX on qubit 1
        let q0_gen = result
            .generators
            .iter()
            .find(|g| g.eeg_type == EegType::H && g.label == Bm::z(0))
            .expect("Should still have H_Z on qubit 0");
        assert!((q0_gen.coeff - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_cy_propagation() {
        // CY after CX: Z on target propagates like CX (Z_t → Z_c Z_t)
        let gates = vec![gate(GateType::CX, &[0, 1]), gate(GateType::CY, &[0, 1])];
        let noise = NoiseModel::coherent_only(0.1);
        let result = analyze_expanded(&gates, &noise);

        // H_Z(1) from CX, propagated through CY: Z_t → Z_c Z_t
        let zz_gen = result
            .generators
            .iter()
            .find(|g| g.eeg_type == EegType::H && g.label == Bm::z(0).multiply(&Bm::z(1)))
            .expect("Should have Z0Z1 after CY propagation of Z1");
        assert!((zz_gen.coeff.abs() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_sy_propagation() {
        // SY: X→-Z, Z→X. So H_Z through SY gives H_X with no sign flip
        let gates = vec![gate(GateType::CX, &[0, 1]), gate(GateType::SY, &[1])];
        let noise = NoiseModel::coherent_only(0.1);
        let result = analyze_expanded(&gates, &noise);

        // H_Z(1) through SY(1): Z→X, no sign flip
        let q1_gen = result
            .generators
            .iter()
            .find(|g| g.eeg_type == EegType::H && g.label == Bm::x(1))
            .expect("Should have H_X on qubit 1 after SY");
        assert!(
            (q1_gen.coeff - 0.05).abs() < 1e-10,
            "SY: Z→X, no sign: expected 0.05, got {}",
            q1_gen.coeff
        );
    }

    #[test]
    fn test_pz_clears_propagated_errors() {
        // Error injected before PZ should be cleared
        let gates = vec![
            gate(GateType::CX, &[0, 1]),
            gate(GateType::PZ, &[1]), // Reset qubit 1
        ];
        let noise = NoiseModel::coherent_only(0.1);
        let result = analyze_expanded(&gates, &noise);

        // H_Z(1) should be cleared by PZ(1)
        let q1_gens: Vec<_> = result
            .generators
            .iter()
            .filter(|g| g.eeg_type == EegType::H && (g.label.has_x(1) || g.label.has_z(1)))
            .collect();
        assert!(
            q1_gens.is_empty(),
            "PZ should clear all error components on qubit 1"
        );

        // H_Z(0) should survive (PZ on qubit 1 doesn't touch qubit 0)
        let q0_gen = result
            .generators
            .iter()
            .find(|g| g.eeg_type == EegType::H && g.label == Bm::z(0));
        assert!(q0_gen.is_some(), "H_Z(0) should survive PZ(1)");
    }

    #[test]
    fn test_no_noise_no_generators() {
        let gates = vec![gate(GateType::CX, &[0, 1]), gate(GateType::H, &[0])];
        let noise = NoiseModel::coherent_only(0.0);
        let result = analyze_expanded(&gates, &noise);
        assert!(result.generators.is_empty());
    }

    #[test]
    fn test_depol_1q_injects_three_paulis() {
        // Single-qubit depolarizing on H gate produces S_X, S_Y, S_Z
        let gates = vec![gate(GateType::H, &[0])];
        let noise = NoiseModel {
            idle_rz: 0.0,
            p1: 0.03,
            p2: 0.0,
            p_meas: 0.0,
            p_prep: 0.0,
        };
        let result = analyze_expanded(&gates, &noise);

        let s_gens: Vec<_> = result
            .generators
            .iter()
            .filter(|g| g.eeg_type == EegType::S)
            .collect();
        assert_eq!(
            s_gens.len(),
            3,
            "1q depolarizing should inject 3 S generators"
        );
        for g in &s_gens {
            assert!(
                (g.coeff + 0.01).abs() < 1e-10,
                "Rate should be -p/3 = -0.01"
            );
        }
    }

    #[test]
    fn test_depol_2q_injects_fifteen_paulis() {
        // Two-qubit depolarizing on CX: 15 S generators (3 single + 3 single + 9 tensor)
        let gates = vec![gate(GateType::CX, &[0, 1])];
        let noise = NoiseModel {
            idle_rz: 0.0,
            p1: 0.0,
            p2: 0.15,
            p_meas: 0.0,
            p_prep: 0.0,
        };
        let result = analyze_expanded(&gates, &noise);

        let s_gens: Vec<_> = result
            .generators
            .iter()
            .filter(|g| g.eeg_type == EegType::S)
            .collect();
        assert_eq!(
            s_gens.len(),
            15,
            "2q depolarizing should inject 15 S generators"
        );
        for g in &s_gens {
            assert!(
                (g.coeff + 0.01).abs() < 1e-10,
                "Rate should be -p/15 = -0.01"
            );
        }
    }

    #[test]
    fn test_meas_noise_injects_sx() {
        // Measurement error produces S_X on the measured qubit
        let gates = vec![gate(GateType::MZ, &[0])];
        let noise = NoiseModel {
            idle_rz: 0.0,
            p1: 0.0,
            p2: 0.0,
            p_meas: 0.05,
            p_prep: 0.0,
        };
        let result = analyze_expanded(&gates, &noise);

        let s_gens: Vec<_> = result
            .generators
            .iter()
            .filter(|g| g.eeg_type == EegType::S)
            .collect();
        assert_eq!(s_gens.len(), 1);
        assert_eq!(s_gens[0].label, Bm::x(0));
        assert!((s_gens[0].coeff + 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_prep_noise_injects_sx() {
        // Preparation error: S_X after PZ
        let gates = vec![gate(GateType::PZ, &[0])];
        let noise = NoiseModel {
            idle_rz: 0.0,
            p1: 0.0,
            p2: 0.0,
            p_meas: 0.0,
            p_prep: 0.03,
        };
        let result = analyze_expanded(&gates, &noise);

        let s_gens: Vec<_> = result
            .generators
            .iter()
            .filter(|g| g.eeg_type == EegType::S)
            .collect();
        assert_eq!(s_gens.len(), 1);
        assert!((s_gens[0].coeff + 0.03).abs() < 1e-10);
    }

    #[test]
    fn test_expansion_pz_gets_no_prep_noise() {
        // Expansion PZ (measurement projection) should NOT inject prep noise.
        // Circuit: PZ(0,1), H(1), CX(1,0), MZ(1), PZ(1), H(1), CX(1,0), MZ(1), MZ(0)
        // With p_prep > 0: only the original PZ gates should inject noise.
        let original_gates = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::PZ, &[1]),
            gate(GateType::H, &[1]),
            gate(GateType::CX, &[1, 0]),
            gate(GateType::MZ, &[1]),
            gate(GateType::PZ, &[1]), // original reset
            gate(GateType::H, &[1]),
            gate(GateType::CX, &[1, 0]),
            gate(GateType::MZ, &[1]), // last round
            gate(GateType::MZ, &[0]),
        ];
        let expanded = crate::expand::expand_circuit(&original_gates);

        // Count PZ gates in expanded circuit (originals + expansion projections)
        let all_pz: Vec<_> = expanded
            .gates
            .iter()
            .filter(|g| g.gate_type == GateType::PZ)
            .collect();

        // With prep noise: count S generators from prep
        let noise = NoiseModel {
            idle_rz: 0.0,
            p1: 0.0,
            p2: 0.0,
            p_meas: 0.0,
            p_prep: 0.1,
        };
        let result = analyze_expanded(&expanded.gates, &noise);

        let prep_gens: Vec<_> = result
            .generators
            .iter()
            .filter(|g| g.eeg_type == EegType::S)
            .collect();

        // Original PZ gates: PZ(0) init, PZ(1) init, PZ(1) reset = 3 PZ with noise
        // Expansion PZ: should NOT inject noise
        // Each original PZ injects 1 S generator (S_X)
        assert_eq!(
            prep_gens.len(),
            3,
            "Only original PZ should inject prep noise, not expansion PZ. \
             Got {} S generators, total PZ gates in expanded: {}",
            prep_gens.len(),
            all_pz.len()
        );
    }

    #[test]
    fn test_expansion_cx_gets_no_noise() {
        // The CX gates added by expansion (deferred measurement) should not get noise.
        // Circuit: PZ(0), CX(0,1), MZ(0), MZ(1)
        // Expanded: PZ(0), CX(0,1), QAlloc(2), CX(0,2), QAlloc(3), CX(1,3), MZ(2), MZ(3)
        // Only CX(0,1) should get noise, not CX(0,2) or CX(1,3).
        let gates = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::PZ, &[1]),
            gate(GateType::CX, &[0, 1]),
            gate(GateType::MZ, &[0]),
            gate(GateType::MZ, &[1]),
        ];
        let expanded = crate::expand::expand_circuit(&gates);
        let noise = NoiseModel::coherent_only(0.1);
        let result = analyze_expanded(&expanded.gates, &noise);

        let h_gens: Vec<_> = result
            .generators
            .iter()
            .filter(|g| g.eeg_type == EegType::H)
            .collect();
        // Only 2 H generators from the original CX (one per qubit)
        assert_eq!(
            h_gens.len(),
            2,
            "Only original CX should get noise, not expansion CX gates"
        );
    }
}
