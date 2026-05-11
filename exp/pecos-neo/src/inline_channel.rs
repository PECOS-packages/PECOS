// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Execution helpers for `TickCircuit`s containing inline channel gates.
//!
//! These routines consume the concrete channel operations inserted into a
//! `TickCircuit`, rather than adding a separate event-driven noise model at
//! execution time.

use pecos_core::gate_type::GateType;
use pecos_core::{Angle64, ChannelExpr, Gate, Pauli, PauliOperator, PauliString, QubitId};
use pecos_quantum::TickCircuit;
use pecos_random::{PecosRng, RngExt};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, DensityMatrix, SparseStab};
use thiserror::Error;

const PROBABILITY_SUM_TOLERANCE: f64 = 1e-9;

/// Error returned while executing a circuit with inline channel gates.
#[derive(Debug, Error)]
pub enum InlineChannelError {
    /// A `Channel` gate had no channel payload.
    #[error("Channel gate missing channel payload")]
    MissingChannelPayload,

    /// A two-qubit gate was given an odd number of qubits.
    #[error("{gate_type:?} requires an even number of qubits, got {qubit_count}")]
    InvalidPairArity {
        gate_type: GateType,
        qubit_count: usize,
    },

    /// A parameterized gate was missing one of its angle parameters.
    #[error("{gate_type:?} is missing angle parameter {index}")]
    MissingAngle { gate_type: GateType, index: usize },

    /// Density-matrix execution does not support this gate on the inline path.
    #[error("DensityMatrix inline-channel path does not support gate {gate_type:?}")]
    UnsupportedDensityMatrixGate { gate_type: GateType },

    /// Stabilizer execution does not support this gate on the inline path.
    #[error("stabilizer inline-channel path does not support non-Clifford gate {gate_type:?}")]
    UnsupportedStabilizerGate { gate_type: GateType },

    /// The stabilizer backend was asked to sample a non-Pauli channel.
    #[error("stabilizer backend can only sample inline Pauli channels")]
    NonPauliChannel,

    /// The stabilizer backend was asked to sample a channel that is not a Pauli mixed-unitary.
    #[error("stabilizer backend can only sample inline Pauli mixed-unitary channels")]
    NonPauliMixedUnitaryChannel,

    /// A mixed-unitary channel had invalid probabilities.
    #[error("channel probabilities must be non-negative and sum to 1")]
    InvalidProbabilities,

    /// Probability validation passed, but random sampling still missed every term.
    #[error("Pauli channel sampling failed after probability validation")]
    SamplingMiss,

    /// `DensityMatrix` rejected a channel expression.
    #[error("channel application failed: {0}")]
    ChannelApplication(String),
}

/// Return the number of qubits needed to simulate the circuit.
#[must_use]
pub fn tick_circuit_num_qubits(circuit: &TickCircuit) -> usize {
    circuit
        .all_qubits()
        .into_iter()
        .map(|q| q.index() + 1)
        .max()
        .unwrap_or(0)
}

/// Execute an inline-channel `TickCircuit` using a density matrix simulator.
///
/// This path supports arbitrary channel expressions accepted by
/// [`DensityMatrix::apply_channel_expr`], and also supports the standard
/// unitary gates implemented by the density matrix simulator.
///
/// # Errors
///
/// Returns [`InlineChannelError`] when a channel payload is malformed, when a
/// gate has invalid arity or missing parameters, or when the density matrix
/// backend does not support a gate/channel.
pub fn run_inline_channels_density_matrix(
    circuit: &TickCircuit,
    shots: usize,
    seed: u64,
) -> Result<Vec<Vec<u8>>, InlineChannelError> {
    let num_qubits = tick_circuit_num_qubits(circuit);
    let mut rows = Vec::with_capacity(shots);

    for shot in 0..shots {
        let shot_seed = seed.wrapping_add(shot as u64);
        let mut sim = DensityMatrix::with_seed(num_qubits, shot_seed);
        let mut row = Vec::new();

        for tick in circuit.ticks() {
            for gate in tick.iter_gate_batches() {
                row.extend(apply_gate_to_density_matrix(&mut sim, gate.as_gate())?);
            }
        }

        rows.push(row);
    }

    Ok(rows)
}

/// Execute an inline-channel `TickCircuit` using a stabilizer simulator.
///
/// This path supports Clifford gates plus channel gates whose payloads are
/// Pauli or Pauli mixed-unitary channels. Non-Pauli channels are rejected
/// explicitly.
///
/// # Errors
///
/// Returns [`InlineChannelError`] for unsupported gates, malformed channel
/// probabilities, non-Pauli channels, invalid arity, or missing gate
/// parameters.
pub fn run_inline_pauli_channels_stabilizer(
    circuit: &TickCircuit,
    shots: usize,
    seed: u64,
) -> Result<Vec<Vec<u8>>, InlineChannelError> {
    let num_qubits = tick_circuit_num_qubits(circuit);
    let mut rows = Vec::with_capacity(shots);

    for shot in 0..shots {
        let shot_seed = seed.wrapping_add(shot as u64);
        let mut sim = SparseStab::with_seed(num_qubits, shot_seed);
        let mut rng = PecosRng::seed_from_u64(shot_seed ^ 0x5eed_5eed_5eed_5eed);
        let mut row = Vec::new();

        for tick in circuit.ticks() {
            for gate in tick.iter_gate_batches() {
                row.extend(apply_gate_to_stabilizer_with_pauli_channels(
                    &mut sim,
                    gate.as_gate(),
                    &mut rng,
                )?);
            }
        }

        rows.push(row);
    }

    Ok(rows)
}

fn qubit_pairs(
    qubits: &[QubitId],
    gate_type: GateType,
) -> Result<Vec<(QubitId, QubitId)>, InlineChannelError> {
    if !qubits.len().is_multiple_of(2) {
        return Err(InlineChannelError::InvalidPairArity {
            gate_type,
            qubit_count: qubits.len(),
        });
    }
    Ok(qubits
        .chunks_exact(2)
        .map(|pair| (pair[0], pair[1]))
        .collect())
}

fn gate_angle(gate: &Gate, index: usize) -> Result<Angle64, InlineChannelError> {
    gate.angles
        .get(index)
        .copied()
        .ok_or(InlineChannelError::MissingAngle {
            gate_type: gate.gate_type,
            index,
        })
}

fn apply_gate_to_density_matrix(
    sim: &mut DensityMatrix,
    gate: &Gate,
) -> Result<Vec<u8>, InlineChannelError> {
    let qubits = gate.qubits.as_slice();
    match gate.gate_type {
        GateType::Channel => {
            let channel = gate
                .channel_expr()
                .ok_or(InlineChannelError::MissingChannelPayload)?;
            sim.apply_channel_expr(channel)
                .map_err(|e| InlineChannelError::ChannelApplication(e.to_string()))?;
            Ok(Vec::new())
        }
        GateType::PZ | GateType::QAlloc => {
            sim.pz(qubits);
            Ok(Vec::new())
        }
        GateType::MZ | GateType::MeasureFree | GateType::MeasureLeaked => Ok(sim
            .mz(qubits)
            .into_iter()
            .map(|r| u8::from(r.outcome))
            .collect()),
        GateType::I | GateType::Idle => Ok(Vec::new()),
        GateType::X => {
            sim.x(qubits);
            Ok(Vec::new())
        }
        GateType::Y => {
            sim.y(qubits);
            Ok(Vec::new())
        }
        GateType::Z => {
            sim.z(qubits);
            Ok(Vec::new())
        }
        GateType::H => {
            sim.h(qubits);
            Ok(Vec::new())
        }
        GateType::F => {
            sim.f(qubits);
            Ok(Vec::new())
        }
        GateType::Fdg => {
            sim.fdg(qubits);
            Ok(Vec::new())
        }
        GateType::SX => {
            sim.sx(qubits);
            Ok(Vec::new())
        }
        GateType::SXdg => {
            sim.sxdg(qubits);
            Ok(Vec::new())
        }
        GateType::SY => {
            sim.sy(qubits);
            Ok(Vec::new())
        }
        GateType::SYdg => {
            sim.sydg(qubits);
            Ok(Vec::new())
        }
        GateType::SZ => {
            sim.sz(qubits);
            Ok(Vec::new())
        }
        GateType::SZdg => {
            sim.szdg(qubits);
            Ok(Vec::new())
        }
        GateType::CX => {
            sim.cx(&qubit_pairs(qubits, gate.gate_type)?);
            Ok(Vec::new())
        }
        GateType::CY => {
            sim.cy(&qubit_pairs(qubits, gate.gate_type)?);
            Ok(Vec::new())
        }
        GateType::CZ => {
            sim.cz(&qubit_pairs(qubits, gate.gate_type)?);
            Ok(Vec::new())
        }
        GateType::SXX => {
            sim.sxx(&qubit_pairs(qubits, gate.gate_type)?);
            Ok(Vec::new())
        }
        GateType::SXXdg => {
            sim.sxxdg(&qubit_pairs(qubits, gate.gate_type)?);
            Ok(Vec::new())
        }
        GateType::SYY => {
            sim.syy(&qubit_pairs(qubits, gate.gate_type)?);
            Ok(Vec::new())
        }
        GateType::SYYdg => {
            sim.syydg(&qubit_pairs(qubits, gate.gate_type)?);
            Ok(Vec::new())
        }
        GateType::SZZ => {
            sim.szz(&qubit_pairs(qubits, gate.gate_type)?);
            Ok(Vec::new())
        }
        GateType::SZZdg => {
            sim.szzdg(&qubit_pairs(qubits, gate.gate_type)?);
            Ok(Vec::new())
        }
        GateType::SWAP => {
            sim.swap(&qubit_pairs(qubits, gate.gate_type)?);
            Ok(Vec::new())
        }
        GateType::T => {
            sim.t(qubits);
            Ok(Vec::new())
        }
        GateType::Tdg => {
            sim.tdg(qubits);
            Ok(Vec::new())
        }
        GateType::RX => {
            sim.rx(gate_angle(gate, 0)?, qubits);
            Ok(Vec::new())
        }
        GateType::RY => {
            sim.ry(gate_angle(gate, 0)?, qubits);
            Ok(Vec::new())
        }
        GateType::RZ => {
            sim.rz(gate_angle(gate, 0)?, qubits);
            Ok(Vec::new())
        }
        GateType::U => {
            sim.u(
                gate_angle(gate, 0)?,
                gate_angle(gate, 1)?,
                gate_angle(gate, 2)?,
                qubits,
            );
            Ok(Vec::new())
        }
        GateType::R1XY => {
            sim.r1xy(gate_angle(gate, 0)?, gate_angle(gate, 1)?, qubits);
            Ok(Vec::new())
        }
        GateType::RXX => {
            sim.rxx(gate_angle(gate, 0)?, &qubit_pairs(qubits, gate.gate_type)?);
            Ok(Vec::new())
        }
        GateType::RYY => {
            sim.ryy(gate_angle(gate, 0)?, &qubit_pairs(qubits, gate.gate_type)?);
            Ok(Vec::new())
        }
        GateType::RZZ => {
            sim.rzz(gate_angle(gate, 0)?, &qubit_pairs(qubits, gate.gate_type)?);
            Ok(Vec::new())
        }
        gate_type => Err(InlineChannelError::UnsupportedDensityMatrixGate { gate_type }),
    }
}

fn apply_pauli_string_to_stabilizer(sim: &mut SparseStab, pauli: &PauliString) {
    for (p, q) in pauli.paulis() {
        match p {
            Pauli::I => {}
            Pauli::X => {
                sim.x(&[*q]);
            }
            Pauli::Y => {
                sim.y(&[*q]);
            }
            Pauli::Z => {
                sim.z(&[*q]);
            }
        }
    }
}

fn sample_pauli_channel(
    channel: &ChannelExpr,
    rng: &mut PecosRng,
) -> Result<Option<PauliString>, InlineChannelError> {
    let ops = match channel {
        ChannelExpr::Unitary(unitary) => {
            let pauli = unitary
                .clone()
                .try_to_pauli_string()
                .ok_or(InlineChannelError::NonPauliChannel)?;
            return Ok((pauli.weight() > 0).then_some(pauli));
        }
        ChannelExpr::MixedUnitary(ops) => ops,
        _ => return Err(InlineChannelError::NonPauliMixedUnitaryChannel),
    };

    let total: f64 = ops.iter().map(|(p, _)| *p).sum();
    if (total - 1.0).abs() > PROBABILITY_SUM_TOLERANCE || ops.iter().any(|(p, _)| *p < 0.0) {
        return Err(InlineChannelError::InvalidProbabilities);
    }

    let mut threshold = rng.random::<f64>() * total;
    for (prob, unitary) in ops {
        if threshold < *prob {
            let pauli = unitary
                .clone()
                .try_to_pauli_string()
                .ok_or(InlineChannelError::NonPauliChannel)?;
            return Ok((pauli.weight() > 0).then_some(pauli));
        }
        threshold -= *prob;
    }

    Err(InlineChannelError::SamplingMiss)
}

fn apply_gate_to_stabilizer_with_pauli_channels(
    sim: &mut SparseStab,
    gate: &Gate,
    rng: &mut PecosRng,
) -> Result<Vec<u8>, InlineChannelError> {
    let qubits = gate.qubits.as_slice();
    match gate.gate_type {
        GateType::Channel => {
            let channel = gate
                .channel_expr()
                .ok_or(InlineChannelError::MissingChannelPayload)?;
            if let Some(pauli) = sample_pauli_channel(channel, rng)? {
                apply_pauli_string_to_stabilizer(sim, &pauli);
            }
            Ok(Vec::new())
        }
        GateType::PZ | GateType::QAlloc => {
            sim.pz(qubits);
            Ok(Vec::new())
        }
        GateType::MZ | GateType::MeasureFree | GateType::MeasureLeaked => Ok(sim
            .mz(qubits)
            .into_iter()
            .map(|r| u8::from(r.outcome))
            .collect()),
        GateType::I | GateType::Idle => Ok(Vec::new()),
        GateType::X => {
            sim.x(qubits);
            Ok(Vec::new())
        }
        GateType::Y => {
            sim.y(qubits);
            Ok(Vec::new())
        }
        GateType::Z => {
            sim.z(qubits);
            Ok(Vec::new())
        }
        GateType::H => {
            sim.h(qubits);
            Ok(Vec::new())
        }
        GateType::F => {
            sim.f(qubits);
            Ok(Vec::new())
        }
        GateType::Fdg => {
            sim.fdg(qubits);
            Ok(Vec::new())
        }
        GateType::SX => {
            sim.sx(qubits);
            Ok(Vec::new())
        }
        GateType::SXdg => {
            sim.sxdg(qubits);
            Ok(Vec::new())
        }
        GateType::SY => {
            sim.sy(qubits);
            Ok(Vec::new())
        }
        GateType::SYdg => {
            sim.sydg(qubits);
            Ok(Vec::new())
        }
        GateType::SZ => {
            sim.sz(qubits);
            Ok(Vec::new())
        }
        GateType::SZdg => {
            sim.szdg(qubits);
            Ok(Vec::new())
        }
        GateType::CX => {
            sim.cx(&qubit_pairs(qubits, gate.gate_type)?);
            Ok(Vec::new())
        }
        GateType::CY => {
            sim.cy(&qubit_pairs(qubits, gate.gate_type)?);
            Ok(Vec::new())
        }
        GateType::CZ => {
            sim.cz(&qubit_pairs(qubits, gate.gate_type)?);
            Ok(Vec::new())
        }
        GateType::SXX => {
            sim.sxx(&qubit_pairs(qubits, gate.gate_type)?);
            Ok(Vec::new())
        }
        GateType::SXXdg => {
            sim.sxxdg(&qubit_pairs(qubits, gate.gate_type)?);
            Ok(Vec::new())
        }
        GateType::SYY => {
            sim.syy(&qubit_pairs(qubits, gate.gate_type)?);
            Ok(Vec::new())
        }
        GateType::SYYdg => {
            sim.syydg(&qubit_pairs(qubits, gate.gate_type)?);
            Ok(Vec::new())
        }
        GateType::SZZ => {
            sim.szz(&qubit_pairs(qubits, gate.gate_type)?);
            Ok(Vec::new())
        }
        GateType::SZZdg => {
            sim.szzdg(&qubit_pairs(qubits, gate.gate_type)?);
            Ok(Vec::new())
        }
        GateType::SWAP => {
            sim.swap(&qubit_pairs(qubits, gate.gate_type)?);
            Ok(Vec::new())
        }
        gate_type => Err(InlineChannelError::UnsupportedStabilizerGate { gate_type }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn density_matrix_inline_bit_flip_channel_flips_measurement() {
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0]);
        circuit.tick().channel(pecos_core::channel::BitFlip(1.0, 0));
        circuit.tick().mz(&[0]);

        let rows = run_inline_channels_density_matrix(&circuit, 3, 123).unwrap();

        assert_eq!(rows, vec![vec![1], vec![1], vec![1]]);
    }

    #[test]
    fn stabilizer_inline_pauli_channel_flips_measurement() {
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0]);
        circuit.tick().channel(pecos_core::channel::BitFlip(1.0, 0));
        circuit.tick().mz(&[0]);

        let rows = run_inline_pauli_channels_stabilizer(&circuit, 3, 123).unwrap();

        assert_eq!(rows, vec![vec![1], vec![1], vec![1]]);
    }

    #[test]
    fn stabilizer_inline_channel_rejects_non_clifford_gate() {
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0]);
        circuit.tick().t(&[0]);
        circuit.tick().channel(pecos_core::channel::BitFlip(0.5, 0));
        circuit.tick().mz(&[0]);

        let err = run_inline_pauli_channels_stabilizer(&circuit, 1, 123).unwrap_err();

        assert!(matches!(
            err,
            InlineChannelError::UnsupportedStabilizerGate {
                gate_type: GateType::T
            }
        ));
    }

    #[test]
    fn stabilizer_inline_channel_rejects_non_pauli_channel() {
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0]);
        circuit
            .tick()
            .channel(pecos_core::channel::AmplitudeDamping(0.5, 0));
        circuit.tick().mz(&[0]);

        let err = run_inline_pauli_channels_stabilizer(&circuit, 1, 123).unwrap_err();

        assert!(matches!(
            err,
            InlineChannelError::NonPauliMixedUnitaryChannel
        ));
    }
}
