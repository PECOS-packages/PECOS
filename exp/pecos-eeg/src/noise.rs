// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! Noise model specification for EEG analysis.
//!
//! Defines the [`NoiseSpec`] trait for specifying how noise generators are
//! injected at each gate in the circuit. The built-in [`UniformNoise`]
//! applies the same rates to all gates of each type (matching the original
//! `NoiseModel`). Users can implement custom noise for per-gate control.

use crate::Bm;
use crate::eeg::EegType;
use pecos_core::gate_type::GateType;

/// A noise generator to inject at a specific point in the circuit.
#[derive(Clone, Debug)]
pub struct NoiseInjection {
    /// EEG type of the generator.
    pub eeg_type: EegType,
    /// Primary Pauli label.
    pub label: Bm,
    /// Second label for C/A types.
    pub label2: Option<Bm>,
    /// Rate (coefficient).
    pub rate: f64,
}

/// Trait for noise models that produce EEG generators at each gate.
///
/// Implement this to specify arbitrary per-gate noise. The EEG analysis
/// calls `noise_after_gate` for each gate in the expanded circuit,
/// propagates the returned generators to the end, and accumulates them.
///
/// The built-in [`UniformNoise`] applies the same rates to all gates of
/// each type. For per-gate or per-qubit noise, implement this trait
/// with a custom struct.
pub trait NoiseSpec: Send + Sync {
    /// Return noise generators to inject after the gate at `gate_index`.
    ///
    /// The `qubits` are the qubit indices of the gate. For 2-qubit gates,
    /// idle coherent noise is typically injected on both qubits.
    ///
    /// Return an empty vec for no noise at this gate.
    fn noise_after_gate(
        &self,
        gate_index: usize,
        gate_type: GateType,
        qubits: &[usize],
    ) -> Vec<NoiseInjection>;
}

/// Uniform noise model: same rates for all gates of each type.
///
/// This is the original `NoiseModel` wrapped as a `NoiseSpec`.
#[derive(Clone, Debug)]
pub struct UniformNoise {
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

impl UniformNoise {
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

impl NoiseSpec for UniformNoise {
    fn noise_after_gate(
        &self,
        _gate_index: usize,
        gate_type: GateType,
        qubits: &[usize],
    ) -> Vec<NoiseInjection> {
        let mut injections = Vec::new();

        match gate_type {
            // Two-qubit gates: idle RZ + depolarizing
            GateType::CX
            | GateType::CZ
            | GateType::CY
            | GateType::SWAP
            | GateType::SZZ
            | GateType::SZZdg
            | GateType::SXX
            | GateType::SXXdg
            | GateType::SYY
            | GateType::SYYdg => {
                if self.idle_rz.abs() > 0.0 && qubits.len() >= 2 {
                    for &q in &qubits[..2] {
                        injections.push(NoiseInjection {
                            eeg_type: EegType::H,
                            label: Bm::z(q),
                            label2: None,
                            rate: self.idle_rz / 2.0,
                        });
                    }
                }
                if self.p2 > 0.0 && qubits.len() >= 2 {
                    inject_depol_2q(qubits[0], qubits[1], self.p2, &mut injections);
                }
            }

            // Single-qubit Clifford: depolarizing
            GateType::H
            | GateType::SZ
            | GateType::SZdg
            | GateType::SX
            | GateType::SXdg
            | GateType::SY
            | GateType::SYdg
            | GateType::X
            | GateType::Y
            | GateType::Z
                if self.p1 > 0.0 && !qubits.is_empty() =>
            {
                inject_depol_1q(qubits[0], self.p1, &mut injections);
            }

            // Measurement error
            GateType::MZ if self.p_meas > 0.0 => {
                for &q in qubits {
                    injections.push(NoiseInjection {
                        eeg_type: EegType::S,
                        label: Bm::x(q),
                        label2: None,
                        rate: -self.p_meas,
                    });
                }
            }

            // Preparation error
            GateType::PZ if self.p_prep > 0.0 => {
                for &q in qubits {
                    injections.push(NoiseInjection {
                        eeg_type: EegType::S,
                        label: Bm::x(q),
                        label2: None,
                        rate: -self.p_prep,
                    });
                }
            }

            _ => {}
        }

        injections
    }
}

fn inject_depol_1q(q: usize, prob: f64, out: &mut Vec<NoiseInjection>) {
    let rate = -prob / 3.0;
    for pf in [Bm::x, Bm::y, Bm::z] {
        out.push(NoiseInjection {
            eeg_type: EegType::S,
            label: pf(q),
            label2: None,
            rate,
        });
    }
}

fn inject_depol_2q(qa: usize, qb: usize, prob: f64, out: &mut Vec<NoiseInjection>) {
    let rate = -prob / 15.0;
    let pfs = [Bm::x, Bm::y, Bm::z];
    for &pa in &pfs {
        out.push(NoiseInjection {
            eeg_type: EegType::S,
            label: pa(qa),
            label2: None,
            rate,
        });
        out.push(NoiseInjection {
            eeg_type: EegType::S,
            label: pa(qb),
            label2: None,
            rate,
        });
        for &pb in &pfs {
            out.push(NoiseInjection {
                eeg_type: EegType::S,
                label: pa(qa).multiply(&pb(qb)),
                label2: None,
                rate,
            });
        }
    }
}
