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

/// One categorical depolarizing channel: choose at most one nonidentity Pauli
/// at this location, with total probability `probability`.
#[derive(Clone, Debug, PartialEq)]
pub enum DepolarizingChannel {
    /// Uniform choice among X, Y and Z on one qubit.
    OneQubit { qubit: usize, probability: f64 },
    /// Uniform choice among the fifteen nonidentity two-qubit Paulis.
    TwoQubit {
        qubits: [usize; 2],
        probability: f64,
    },
}

impl DepolarizingChannel {
    pub(crate) fn qubits(&self) -> &[usize] {
        match self {
            Self::OneQubit { qubit, .. } => std::slice::from_ref(qubit),
            Self::TwoQubit { qubits, .. } => qubits,
        }
    }

    /// Adjoint eigenvalue on any Pauli with nonidentity channel support.
    pub(crate) fn eigenvalue(&self) -> f64 {
        match self {
            Self::OneQubit { probability, .. } => 1.0 - 4.0 * probability / 3.0,
            Self::TwoQubit { probability, .. } => 1.0 - 16.0 * probability / 15.0,
        }
    }
}

/// Physical noise for a Heisenberg walk, distinct from a forward EEG's
/// approximate generator representation of categorical channels.
/// In forward time, injections act in list order, then categorical channels
/// act in list order. Backward walks apply the adjoints in reverse order.
#[derive(Clone, Debug, Default)]
pub struct GateNoise {
    /// Injections in forward-time application order. Each S injection remains
    /// an independent Pauli flip with probability `-rate`.
    pub injections: Vec<NoiseInjection>,
    /// Categorical channels in forward-time application order, after all
    /// injections. Channels at separate locations are independent; Pauli choices
    /// within one channel are exclusive.
    pub depolarizing: Vec<DepolarizingChannel>,
}

impl GateNoise {
    /// Every qubit an injection label or channel acts on, possibly repeated.
    pub(crate) fn qubits(&self) -> impl Iterator<Item = usize> + '_ {
        self.injections
            .iter()
            .flat_map(|inj| std::iter::once(&inj.label).chain(inj.label2.as_ref()))
            .flat_map(label_qubits)
            .chain(
                self.depolarizing
                    .iter()
                    .flat_map(|channel| channel.qubits().iter().copied()),
            )
    }
}

/// The qubits a Pauli label acts on, in ascending order. Walks set bits word by
/// word, so the cost follows the support, not the highest qubit index.
pub(crate) fn label_qubits(label: &Bm) -> impl Iterator<Item = usize> + '_ {
    let word = |bits: &[u64], w: usize| bits.get(w).copied().unwrap_or(0);
    let words = label.x_bits.len().max(label.z_bits.len());
    (0..words).flat_map(move |w| {
        let mut bits = word(&label.x_bits, w) | word(&label.z_bits, w);
        std::iter::from_fn(move || {
            (bits != 0).then(|| {
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                w * 64 + bit
            })
        })
    })
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
    /// Return the generator view used by forward EEG and mechanism extraction.
    /// The returned list is in forward-time application order.
    /// `UniformNoise` represents categorical depolarizing to first order with
    /// S coefficients `-p/3` or `-p/15`; these are not an exact channel composition.
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

    /// Return physical channels for the backward Heisenberg walk.
    /// The returned lists are in forward-time application order: injections, then channels.
    ///
    /// The default preserves custom models: each returned S injection is an
    /// independent Pauli flip at probability `-rate`. Equal rates or a count of
    /// three/fifteen injections never imply a categorical channel. Override
    /// this method to provide explicit categorical depolarizing channels.
    /// Injections and channels may act on qubits outside `qubits`, for example
    /// crosstalk onto a neighbour; the walks decide relevance by the noise's own
    /// support.
    fn exact_noise_after_gate(
        &self,
        gate_index: usize,
        gate_type: GateType,
        qubits: &[usize],
    ) -> GateNoise {
        GateNoise {
            injections: self.noise_after_gate(gate_index, gate_type, qubits),
            depolarizing: Vec::new(),
        }
    }
}

/// Uniform noise model: same rates for all gates of each type.
///
/// This is the original `NoiseModel` wrapped as a `NoiseSpec`.
#[derive(Clone, Debug)]
pub struct UniformNoise {
    /// Coherent RZ angle (radians) on both qubits after each 2-qubit gate.
    pub idle_rz: f64,
    /// Total categorical single-qubit depolarizing probability.
    pub p1: f64,
    /// Total categorical two-qubit depolarizing probability.
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

#[derive(Clone, Copy)]
enum DepolarizingRepresentation {
    Generators,
    Channels,
}

impl UniformNoise {
    fn gate_noise(
        &self,
        gate_type: GateType,
        qubits: &[usize],
        representation: DepolarizingRepresentation,
    ) -> GateNoise {
        let mut injections = Vec::new();
        let mut depolarizing = Vec::new();

        match gate_type {
            GateType::CS | GateType::CSdg | GateType::CCZ => {
                panic!("EEG noise does not support {gate_type:?}")
            }
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
                for qubits in qubits.as_chunks::<2>().0 {
                    if self.idle_rz.abs() > 0.0 {
                        for &q in qubits {
                            injections.push(NoiseInjection {
                                eeg_type: EegType::H,
                                label: Bm::z(q),
                                label2: None,
                                rate: self.idle_rz / 2.0,
                            });
                        }
                    }
                    if self.p2 > 0.0 {
                        match representation {
                            DepolarizingRepresentation::Generators => {
                                inject_depol_2q(qubits[0], qubits[1], self.p2, &mut injections);
                            }
                            DepolarizingRepresentation::Channels => {
                                depolarizing.push(DepolarizingChannel::TwoQubit {
                                    qubits: *qubits,
                                    probability: self.p2,
                                });
                            }
                        }
                    }
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
                for &qubit in qubits {
                    match representation {
                        DepolarizingRepresentation::Generators => {
                            inject_depol_1q(qubit, self.p1, &mut injections);
                        }
                        DepolarizingRepresentation::Channels => {
                            depolarizing.push(DepolarizingChannel::OneQubit {
                                qubit,
                                probability: self.p1,
                            });
                        }
                    }
                }
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

            // Preparation error, for resets and allocations alike
            GateType::PZ | GateType::QAlloc if self.p_prep > 0.0 => {
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

        GateNoise {
            injections,
            depolarizing,
        }
    }
}

impl NoiseSpec for UniformNoise {
    fn noise_after_gate(
        &self,
        _gate_index: usize,
        gate_type: GateType,
        qubits: &[usize],
    ) -> Vec<NoiseInjection> {
        self.gate_noise(gate_type, qubits, DepolarizingRepresentation::Generators)
            .injections
    }

    fn exact_noise_after_gate(
        &self,
        _gate_index: usize,
        gate_type: GateType,
        qubits: &[usize],
    ) -> GateNoise {
        self.gate_noise(gate_type, qubits, DepolarizingRepresentation::Channels)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_gates_are_rejected_by_noise() {
        for gt in [GateType::CS, GateType::CSdg, GateType::CCZ] {
            for representation in [
                DepolarizingRepresentation::Generators,
                DepolarizingRepresentation::Channels,
            ] {
                assert!(
                    std::panic::catch_unwind(|| UniformNoise::coherent_only(0.0).gate_noise(
                        gt,
                        &(0..gt.quantum_arity()).collect::<Vec<_>>(),
                        representation
                    ))
                    .is_err()
                );
            }
        }
    }

    #[test]
    fn label_qubits_spans_storage_words() {
        // X, Z and Y parts on both sides of the 64- and 128-qubit word edges.
        let label = [Bm::x(0), Bm::z(63), Bm::y(64), Bm::x(127), Bm::z(130)]
            .iter()
            .fold(Bm::default(), |acc, p| acc.multiply(p));
        assert_eq!(
            label_qubits(&label).collect::<Vec<_>>(),
            [0, 63, 64, 127, 130]
        );
        assert_eq!(label_qubits(&Bm::default()).count(), 0);
    }

    #[test]
    fn batched_single_qubit_generators_cover_every_operand() {
        let noise = UniformNoise::depolarizing(0.3);
        let batched = noise.noise_after_gate(0, GateType::H, &[0, 1, 2]);
        let separate: Vec<_> = (0..3)
            .flat_map(|q| noise.noise_after_gate(0, GateType::H, &[q]))
            .collect();
        assert_eq!(batched.len(), 9);
        for (actual, expected) in batched.iter().zip(&separate) {
            assert_eq!(actual.eeg_type, expected.eeg_type);
            assert_eq!(actual.label, expected.label);
            assert_eq!(actual.rate.to_bits(), expected.rate.to_bits());
        }
        let exact = noise.exact_noise_after_gate(0, GateType::H, &[0, 1, 2]);
        assert!(exact.injections.is_empty());
        assert_eq!(exact.depolarizing.len(), 3);
        for (qubit, channel) in exact.depolarizing.into_iter().enumerate() {
            assert_eq!(
                channel,
                DepolarizingChannel::OneQubit {
                    qubit,
                    probability: 0.3
                }
            );
        }
    }

    #[test]
    fn test_batched_two_qubit_injections() {
        let noise = UniformNoise::depolarizing(1.0).with_idle_rz(0.1);
        let batched = noise.noise_after_gate(0, GateType::CX, &[0, 1, 2, 3]);
        let separate: Vec<_> = [[0, 1], [2, 3]]
            .iter()
            .flat_map(|pair| noise.noise_after_gate(0, GateType::CX, pair))
            .collect();
        assert_eq!(batched.len(), 34);
        assert_eq!(batched.len(), separate.len());
        for (actual, expected) in batched.iter().zip(&separate) {
            assert_eq!(actual.eeg_type, expected.eeg_type);
            assert_eq!(actual.label, expected.label);
            assert_eq!(actual.label2, expected.label2);
            assert_eq!(actual.rate.to_bits(), expected.rate.to_bits());
        }
    }
}
