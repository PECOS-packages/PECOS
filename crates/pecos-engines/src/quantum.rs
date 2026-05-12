use crate::Engine;
use crate::byte_message::ByteMessage;
use crate::byte_message::GateType;
use dyn_clone::DynClone;
use log::debug;
use pecos_core::Angle64;
use pecos_core::ChannelExpr;
use pecos_core::QubitId;
use pecos_core::RngManageable;
use pecos_core::errors::PecosError;
use pecos_random::{PecosRng, SeedableRng};
use pecos_simulators::clifford_rotation::CliffordRotation;
use pecos_simulators::{
    ArbitraryRotationGateable, CliffordGateable, CoinToss, DensityMatrix, QuantumSimulator,
    SparseStab, StabVec, Stabilizer, StateVec, StateVecAoS, StateVecSoA,
};
use std::any::Any;
use std::fmt::Debug;

/// Helper function to create quantum engine errors
fn quantum_error<S: Into<String>>(msg: S) -> PecosError {
    PecosError::Processing(msg.into())
}

/// Apply a closure to a flat qubit slice `[c0, t0, c1, t1, ...]` and return its result.
///
/// Most commands contain a single pair, so avoid heap allocation in that case
/// and reuse a scratch buffer for the rarer batched-pair path. The closure's
/// return value is forwarded so fallible gates (e.g. `try_rzz`) can bubble up
/// a `Result` without a separate borrow-and-stash dance at the call site.
fn with_flat_pairs<F, R>(
    qubits: &[QubitId],
    pair_scratch: &mut Vec<(QubitId, QubitId)>,
    mut f: F,
) -> R
where
    F: FnMut(&[(QubitId, QubitId)]) -> R,
{
    debug_assert_eq!(qubits.len() % 2, 0);

    if qubits.len() == 2 {
        let pair = [(qubits[0], qubits[1])];
        return f(&pair);
    }

    pair_scratch.clear();
    pair_scratch.extend(qubits.chunks_exact(2).map(|pair| (pair[0], pair[1])));
    f(pair_scratch)
}

/// Convert a flat qubit slice `[c0, t0, c1, t1, ...]` to a vec of pairs.
fn flat_to_pairs(qubits: &[QubitId]) -> Vec<(QubitId, QubitId)> {
    let mut pairs = Vec::with_capacity(qubits.len() / 2);
    pairs.extend(qubits.chunks_exact(2).map(|pair| (pair[0], pair[1])));
    pairs
}

trait ChannelDispatch {
    fn apply_channel_expr(&mut self, channel: &ChannelExpr) -> Result<(), PecosError>;
}

impl ChannelDispatch for StabVec {
    fn apply_channel_expr(&mut self, _channel: &ChannelExpr) -> Result<(), PecosError> {
        Err(quantum_error(
            "Channel gate requires a channel-aware simulator path",
        ))
    }
}

impl ChannelDispatch for DensityMatrix {
    fn apply_channel_expr(&mut self, channel: &ChannelExpr) -> Result<(), PecosError> {
        DensityMatrix::apply_channel_expr(self, channel)
            .map(|_| ())
            .map_err(|err| quantum_error(format!("invalid channel gate: {err}")))
    }
}

/// Process a `ByteMessage` against any Clifford-capable simulator.
///
/// Shared gate dispatch for `SparseStabEngine`, `StabilizerEngine`, etc.
/// Supports Clifford gates, preparations, measurements, and Clifford rotations
/// (non-Clifford angles produce an error via `CliffordRotation::try_*`).
fn process_clifford_message<S: CliffordGateable + CliffordRotation + QuantumSimulator>(
    sim: &mut S,
    message: &ByteMessage,
) -> Result<ByteMessage, PecosError> {
    let batch = message.quantum_ops()?;
    let mut measurements: Vec<usize> = Vec::new();
    let mut pair_scratch: Vec<(QubitId, QubitId)> = Vec::new();
    let mut mz_qubits: Vec<QubitId> = Vec::new();

    let mut cmd_idx = 0;
    while cmd_idx < batch.len() {
        let cmd = &batch[cmd_idx];
        match cmd.gate_type {
            // Single-qubit Clifford gates
            GateType::X => {
                sim.x(&cmd.qubits);
            }
            GateType::Y => {
                sim.y(&cmd.qubits);
            }
            GateType::Z => {
                sim.z(&cmd.qubits);
            }
            GateType::H => {
                sim.h(&cmd.qubits);
            }
            GateType::SZ => {
                sim.sz(&cmd.qubits);
            }
            GateType::SZdg => {
                sim.szdg(&cmd.qubits);
            }
            GateType::SX => {
                sim.sx(&cmd.qubits);
            }
            GateType::SXdg => {
                sim.sxdg(&cmd.qubits);
            }
            GateType::SY => {
                sim.sy(&cmd.qubits);
            }
            GateType::SYdg => {
                sim.sydg(&cmd.qubits);
            }

            // Two-qubit Clifford gates
            GateType::CX => {
                with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                    sim.cx(pairs);
                });
            }
            GateType::CY => {
                with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                    sim.cy(pairs);
                });
            }
            GateType::CZ => {
                with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                    sim.cz(pairs);
                });
            }
            GateType::SWAP => {
                with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                    sim.swap(pairs);
                });
            }
            GateType::SZZ => {
                with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                    sim.szz(pairs);
                });
            }
            GateType::SZZdg => {
                with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                    sim.szzdg(pairs);
                });
            }
            GateType::SXX => {
                with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                    sim.sxx(pairs);
                });
            }
            GateType::SXXdg => {
                with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                    sim.sxxdg(pairs);
                });
            }
            GateType::SYY => {
                with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                    sim.syy(pairs);
                });
            }
            GateType::SYYdg => {
                with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                    sim.syydg(pairs);
                });
            }

            // Batch consecutive MZ commands
            GateType::MZ | GateType::MeasureLeaked => {
                mz_qubits.clear();
                mz_qubits.extend_from_slice(&cmd.qubits);
                while cmd_idx + 1 < batch.len()
                    && matches!(
                        batch[cmd_idx + 1].gate_type,
                        GateType::MZ | GateType::MeasureLeaked
                    )
                {
                    cmd_idx += 1;
                    mz_qubits.extend_from_slice(&batch[cmd_idx].qubits);
                }
                let meas_ids = sim.mz(&mz_qubits);
                for meas_id in meas_ids {
                    measurements.push(usize::from(meas_id.outcome));
                }
            }

            GateType::PZ => {
                sim.pz(&cmd.qubits);
            }
            GateType::Idle | GateType::I => {}

            // Rotation gates via CliffordRotation (errors on non-Clifford angles)
            GateType::RZ
            | GateType::RX
            | GateType::RY
            | GateType::RZZ
            | GateType::RXX
            | GateType::RYY => {
                if !cmd.angles.is_empty() {
                    let angle = cmd.angles[0];
                    let result: Result<(), String> = match cmd.gate_type {
                        GateType::RZ => sim.try_rz(angle, &cmd.qubits).map(|_| ()),
                        GateType::RX => sim.try_rx(angle, &cmd.qubits).map(|_| ()),
                        GateType::RY => sim.try_ry(angle, &cmd.qubits).map(|_| ()),
                        GateType::RZZ => with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                            sim.try_rzz(angle, pairs).map(|_| ())
                        }),
                        GateType::RXX => with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                            sim.try_rxx(angle, pairs).map(|_| ())
                        }),
                        GateType::RYY => with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                            sim.try_ryy(angle, pairs).map(|_| ())
                        }),
                        _ => unreachable!(),
                    };
                    result.map_err(PecosError::Processing)?;
                }
            }
            GateType::R1XY => {
                if cmd.angles.len() >= 2 {
                    sim.try_r1xy(cmd.angles[0], cmd.angles[1], &cmd.qubits)
                        .map_err(PecosError::Processing)?;
                }
            }
            GateType::CRZ => {
                if !cmd.angles.is_empty() {
                    with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                        sim.try_crz(cmd.angles[0], pairs).map(|_| ())
                    })
                    .map_err(PecosError::Processing)?;
                }
            }
            GateType::U => {
                if cmd.angles.len() >= 3 {
                    sim.try_u(cmd.angles[0], cmd.angles[1], cmd.angles[2], &cmd.qubits)
                        .map_err(PecosError::Processing)?;
                }
            }
            GateType::RXXRYYRZZ => {
                if cmd.angles.len() >= 3 {
                    with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                        sim.try_rxxryyrzz(cmd.angles[0], cmd.angles[1], cmd.angles[2], pairs)
                            .map(|_| ())
                    })
                    .map_err(PecosError::Processing)?;
                }
            }
            GateType::U2q => {
                if cmd.angles.len() >= 15 {
                    let before = [
                        [cmd.angles[0], cmd.angles[1], cmd.angles[2]],
                        [cmd.angles[3], cmd.angles[4], cmd.angles[5]],
                    ];
                    let interaction = [cmd.angles[6], cmd.angles[7], cmd.angles[8]];
                    let after = [
                        [cmd.angles[9], cmd.angles[10], cmd.angles[11]],
                        [cmd.angles[12], cmd.angles[13], cmd.angles[14]],
                    ];
                    with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                        sim.try_u2q(before, interaction, after, pairs).map(|_| ())
                    })
                    .map_err(PecosError::Processing)?;
                }
            }

            _ => {
                return Err(PecosError::Processing(format!(
                    "Gate {:?} is not supported by the stabilizer simulator.",
                    cmd.gate_type
                )));
            }
        }
        cmd_idx += 1;
    }

    let mut builder = ByteMessage::outcomes_builder();
    builder.add_outcomes(&measurements);
    Ok(builder.build())
}

/// Process a `ByteMessage` against any simulator supporting full gate set.
///
/// Shared gate dispatch for `StabVecEngine`, `DensityMatrixEngine`, etc.
/// Supports all Clifford gates, arbitrary rotations, composite gates (CH, CCX, CRZ),
/// preparations, and measurements with MZ batching.
fn process_general_message<
    S: CliffordGateable + ArbitraryRotationGateable + QuantumSimulator + ChannelDispatch,
>(
    sim: &mut S,
    message: &ByteMessage,
) -> Result<ByteMessage, PecosError> {
    let batch = message.quantum_ops()?;
    let mut measurements: Vec<usize> = Vec::new();
    let mut pair_scratch: Vec<(QubitId, QubitId)> = Vec::new();
    let mut mz_qubits: Vec<QubitId> = Vec::new();

    let mut cmd_idx = 0;
    while cmd_idx < batch.len() {
        let cmd = &batch[cmd_idx];
        match cmd.gate_type {
            // Single-qubit Clifford gates
            GateType::X => {
                sim.x(&cmd.qubits);
            }
            GateType::Y => {
                sim.y(&cmd.qubits);
            }
            GateType::Z => {
                sim.z(&cmd.qubits);
            }
            GateType::H => {
                sim.h(&cmd.qubits);
            }
            GateType::SZ => {
                sim.sz(&cmd.qubits);
            }
            GateType::SZdg => {
                sim.szdg(&cmd.qubits);
            }
            GateType::SX => {
                sim.sx(&cmd.qubits);
            }
            GateType::SXdg => {
                sim.sxdg(&cmd.qubits);
            }
            GateType::SY => {
                sim.sy(&cmd.qubits);
            }
            GateType::SYdg => {
                sim.sydg(&cmd.qubits);
            }
            GateType::F => {
                sim.f(&cmd.qubits);
            }
            GateType::Fdg => {
                sim.fdg(&cmd.qubits);
            }

            // T gates
            GateType::T => {
                sim.t(&cmd.qubits);
            }
            GateType::Tdg => {
                sim.tdg(&cmd.qubits);
            }

            // Two-qubit Clifford gates
            GateType::CX => {
                with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                    sim.cx(pairs);
                });
            }
            GateType::CY => {
                with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                    sim.cy(pairs);
                });
            }
            GateType::CZ => {
                with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                    sim.cz(pairs);
                });
            }
            GateType::SZZ => {
                with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                    sim.szz(pairs);
                });
            }
            GateType::SZZdg => {
                with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                    sim.szzdg(pairs);
                });
            }
            GateType::SXX => {
                with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                    sim.sxx(pairs);
                });
            }
            GateType::SXXdg => {
                with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                    sim.sxxdg(pairs);
                });
            }
            GateType::SYY => {
                with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                    sim.syy(pairs);
                });
            }
            GateType::SYYdg => {
                with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                    sim.syydg(pairs);
                });
            }
            GateType::SWAP => {
                with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                    sim.swap(pairs);
                });
            }

            // Composite gates (decomposed into primitives)
            GateType::CH => {
                for qubits in cmd.qubits.chunks_exact(2) {
                    let target_slice = &[qubits[1]];
                    sim.ry(
                        Angle64::from_radians(std::f64::consts::FRAC_PI_4),
                        target_slice,
                    );
                    sim.cx(&[(qubits[0], qubits[1])]);
                    sim.ry(
                        Angle64::from_radians(-std::f64::consts::FRAC_PI_4),
                        target_slice,
                    );
                }
            }
            GateType::CCX => {
                for qubits in cmd.qubits.chunks_exact(3) {
                    let c0 = qubits[0];
                    let c1 = qubits[1];
                    let target = qubits[2];
                    sim.h(&[target]);
                    sim.cx(&[(c1, target)]);
                    sim.tdg(&[target]);
                    sim.cx(&[(c0, target)]);
                    sim.t(&[target]);
                    sim.cx(&[(c1, target)]);
                    sim.tdg(&[target]);
                    sim.cx(&[(c0, target)]);
                    sim.t(&[c1]);
                    sim.t(&[target]);
                    sim.cx(&[(c0, c1)]);
                    sim.h(&[target]);
                    sim.t(&[c0]);
                    sim.tdg(&[c1]);
                    sim.cx(&[(c0, c1)]);
                }
            }

            // Rotation gates
            GateType::RZ => {
                if !cmd.angles.is_empty() {
                    sim.rz(cmd.angles[0], &cmd.qubits);
                }
            }
            GateType::RX => {
                if !cmd.angles.is_empty() {
                    sim.rx(cmd.angles[0], &cmd.qubits);
                }
            }
            GateType::RY => {
                if !cmd.angles.is_empty() {
                    sim.ry(cmd.angles[0], &cmd.qubits);
                }
            }
            GateType::RZZ => {
                if !cmd.angles.is_empty() {
                    with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                        sim.rzz(cmd.angles[0], pairs);
                    });
                }
            }
            GateType::RXX => {
                if !cmd.angles.is_empty() {
                    with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                        sim.rxx(cmd.angles[0], pairs);
                    });
                }
            }
            GateType::RYY => {
                if !cmd.angles.is_empty() {
                    with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                        sim.ryy(cmd.angles[0], pairs);
                    });
                }
            }
            GateType::CRZ => {
                if !cmd.angles.is_empty() {
                    let angle = cmd.angles[0];
                    let half_angle = angle / 2u64;
                    for qubits in cmd.qubits.chunks_exact(2) {
                        sim.rz(half_angle, &[qubits[1]]);
                        sim.cx(&[(qubits[0], qubits[1])]);
                        sim.rz(-half_angle, &[qubits[1]]);
                        sim.cx(&[(qubits[0], qubits[1])]);
                    }
                }
            }
            GateType::R1XY => {
                if cmd.angles.len() >= 2 {
                    sim.r1xy(cmd.angles[0], cmd.angles[1], &cmd.qubits);
                }
            }
            GateType::U => {
                if cmd.angles.len() >= 3 {
                    sim.u(cmd.angles[0], cmd.angles[1], cmd.angles[2], &cmd.qubits);
                }
            }
            GateType::RXXRYYRZZ => {
                if cmd.angles.len() >= 3 {
                    with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                        sim.rxxryyrzz(cmd.angles[0], cmd.angles[1], cmd.angles[2], pairs);
                    });
                }
            }
            GateType::U2q => {
                if cmd.angles.len() >= 15 {
                    let before = [
                        [cmd.angles[0], cmd.angles[1], cmd.angles[2]],
                        [cmd.angles[3], cmd.angles[4], cmd.angles[5]],
                    ];
                    let interaction = [cmd.angles[6], cmd.angles[7], cmd.angles[8]];
                    let after = [
                        [cmd.angles[9], cmd.angles[10], cmd.angles[11]],
                        [cmd.angles[12], cmd.angles[13], cmd.angles[14]],
                    ];
                    with_flat_pairs(&cmd.qubits, &mut pair_scratch, |pairs| {
                        sim.u2q(before, interaction, after, pairs);
                    });
                }
            }

            // Batch consecutive MZ commands into one simulator call
            GateType::MZ | GateType::MeasureLeaked => {
                mz_qubits.clear();
                mz_qubits.extend_from_slice(&cmd.qubits);
                while cmd_idx + 1 < batch.len()
                    && matches!(
                        batch[cmd_idx + 1].gate_type,
                        GateType::MZ | GateType::MeasureLeaked
                    )
                {
                    cmd_idx += 1;
                    mz_qubits.extend_from_slice(&batch[cmd_idx].qubits);
                }
                let meas_ids = sim.mz(&mz_qubits);
                for meas_id in meas_ids {
                    measurements.push(usize::from(meas_id.outcome));
                }
            }
            GateType::MeasureFree => {
                let meas_ids = sim.mz(&cmd.qubits);
                for meas_id in meas_ids {
                    measurements.push(usize::from(meas_id.outcome));
                }
            }

            // State preparation
            GateType::PZ | GateType::QAlloc => {
                sim.pz(&cmd.qubits);
            }
            GateType::Channel => {
                let channel = cmd
                    .channel_expr()
                    .ok_or_else(|| quantum_error("Channel gate is missing its channel payload"))?;
                sim.apply_channel_expr(channel)?;
            }

            // No-ops
            GateType::I
            | GateType::Idle
            | GateType::MeasCrosstalkLocalPayload
            | GateType::MeasCrosstalkGlobalPayload
            | GateType::QFree
            | GateType::Custom
            | GateType::TrackedPauliMeta => {}
        }
        cmd_idx += 1;
    }

    let mut builder = ByteMessage::outcomes_builder();
    builder.add_outcomes(&measurements);
    Ok(builder.build())
}

/// Trait for quantum engines that can process quantum operations
pub trait QuantumEngine:
    Engine<Input = ByteMessage, Output = ByteMessage> + DynClone + Debug
{
    /// Sets a specific seed for the quantum engine
    ///
    /// # Arguments
    /// * `seed` - Seed value for the random number generator
    fn set_seed(&mut self, seed: u64);

    /// Returns a reference to this object as Any, for downcasting
    fn as_any(&self) -> &dyn Any;

    /// Returns a mutable reference to this object as Any, for downcasting
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

dyn_clone::clone_trait_object!(QuantumEngine);

// Implement Engine for Box<dyn QuantumEngine> to allow using it directly
// as a controlled engine in EngineSystem
impl Engine for Box<dyn QuantumEngine> {
    type Input = ByteMessage;
    type Output = ByteMessage;

    fn process(&mut self, input: Self::Input) -> Result<Self::Output, PecosError> {
        (**self).process(input)
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        (**self).reset()
    }
}

/// Trait for simulators that can be used with `StateVectorEngine`.
///
/// This trait combines all the requirements for a state vector simulator.
pub trait StateVectorSimulator:
    QuantumSimulator
    + CliffordGateable
    + ArbitraryRotationGateable
    + RngManageable
    + Clone
    + Debug
    + Send
    + Sync
where
    <Self as RngManageable>::Rng: Clone,
{
    /// Create a new simulator with the specified number of qubits.
    fn create(num_qubits: usize) -> Self;

    /// Create a new simulator with a specific seed.
    fn create_with_seed(num_qubits: usize, seed: u64) -> Self;

    /// Create a new simulator with a custom RNG.
    fn create_with_rng(num_qubits: usize, rng: <Self as RngManageable>::Rng) -> Self;
}

impl StateVectorSimulator for StateVec {
    fn create(num_qubits: usize) -> Self {
        StateVec::new(num_qubits)
    }

    fn create_with_seed(num_qubits: usize, seed: u64) -> Self {
        StateVec::with_seed(num_qubits, seed)
    }

    fn create_with_rng(num_qubits: usize, rng: PecosRng) -> Self {
        StateVec::with_rng(num_qubits, rng)
    }
}

impl StateVectorSimulator for StateVecAoS {
    fn create(num_qubits: usize) -> Self {
        StateVecAoS::new(num_qubits)
    }

    fn create_with_seed(num_qubits: usize, seed: u64) -> Self {
        StateVecAoS::with_seed(num_qubits, seed)
    }

    fn create_with_rng(num_qubits: usize, rng: PecosRng) -> Self {
        StateVecAoS::with_rng(num_qubits, rng)
    }
}

impl StateVectorSimulator for StateVecSoA {
    fn create(num_qubits: usize) -> Self {
        StateVecSoA::new(num_qubits)
    }

    fn create_with_seed(num_qubits: usize, seed: u64) -> Self {
        StateVecSoA::with_seed(num_qubits, seed)
    }

    fn create_with_rng(num_qubits: usize, rng: PecosRng) -> Self {
        StateVecSoA::with_rng(num_qubits, rng)
    }
}

/// A generic quantum engine that uses any state vector simulator.
///
/// This engine works with any simulator implementing `StateVectorSimulator`.
#[derive(Debug, Clone)]
pub struct StateVectorEngine<S: StateVectorSimulator>
where
    <S as RngManageable>::Rng: Clone,
{
    simulator: S,
}

impl<S: StateVectorSimulator> StateVectorEngine<S>
where
    <S as RngManageable>::Rng: Clone,
{
    /// Create a new state vector engine with the specified number of qubits
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            simulator: S::create(num_qubits),
        }
    }

    /// Create a new state vector engine with a specific seed
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self {
            simulator: S::create_with_seed(num_qubits, seed),
        }
    }

    /// Ensure the simulator has the correct number of qubits, recreating if necessary
    pub fn ensure_qubit_count(&mut self, required_qubits: usize) {
        if self.simulator.num_qubits() < required_qubits {
            debug!(
                "StateVectorEngine: Expanding simulator (was {} qubits, now {} qubits)",
                self.simulator.num_qubits(),
                required_qubits
            );
            let rng = self.simulator.rng().clone();
            self.simulator = S::create_with_rng(required_qubits, rng);
        }
    }
}

/// Type alias for state vector engine using the default `StateVec` simulator
/// (sparse `SoA`, optimized for QEC workloads).
pub type StateVecEngine = StateVectorEngine<StateVec>;

/// Type alias for state vector engine using the dense `StateVecSoA` simulator.
///
/// `DenseStateVecEngine` uses:
/// - `SoA` (Structure of Arrays) layout for better SIMD performance
/// - Strided iteration for cache-efficient access patterns
/// - Fused gate primitives for reduced memory bandwidth
/// - Optional parallel execution for large state vectors
pub type DenseStateVecEngine = StateVectorEngine<StateVecSoA>;

impl DenseStateVecEngine {
    /// Create a new dense state vector engine with parallel execution enabled/disabled
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits in the system
    /// * `parallel` - Whether to enable parallel execution for large states
    /// * `num_threads` - Number of threads for parallel execution (None = use Rayon's default)
    #[must_use]
    pub fn with_parallel(num_qubits: usize, parallel: bool, num_threads: Option<usize>) -> Self {
        let mut simulator = StateVecSoA::new(num_qubits);
        simulator.set_parallel(parallel);
        simulator.set_num_threads(num_threads);
        Self { simulator }
    }
}

impl<S: StateVectorSimulator + 'static> Engine for StateVectorEngine<S>
where
    <S as RngManageable>::Rng: Clone,
{
    type Input = ByteMessage;
    type Output = ByteMessage;

    #[allow(clippy::too_many_lines)]
    fn process(&mut self, message: Self::Input) -> Result<Self::Output, PecosError> {
        // Parse commands from the message
        let batch = message.quantum_ops()?;

        // Calculate required number of qubits from operations and ensure simulator has correct size
        if !batch.is_empty() {
            let max_qubit_index = batch
                .iter()
                .flat_map(|cmd| cmd.qubits.iter())
                .map(|q| usize::from(*q))
                .max()
                .unwrap_or(0);
            let required_qubits = max_qubit_index + 1;
            self.ensure_qubit_count(required_qubits);
        }

        let mut measurements: Vec<usize> = Vec::new();

        // Use indexed iteration so we can batch consecutive MZ commands into
        // one simulator call, enabling joint-sampling optimizations.
        let mut cmd_idx = 0;
        while cmd_idx < batch.len() {
            let cmd = &batch[cmd_idx];
            match cmd.gate_type {
                GateType::X => {
                    debug!("Processing X gate on qubits {:?}", cmd.qubits);
                    self.simulator.x(&cmd.qubits);
                }
                GateType::Y => {
                    debug!("Processing Y gate on qubits {:?}", cmd.qubits);
                    self.simulator.y(&cmd.qubits);
                }
                GateType::Z => {
                    debug!("Processing Z gate on qubits {:?}", cmd.qubits);
                    self.simulator.z(&cmd.qubits);
                }
                GateType::H => {
                    debug!("Processing H gate on qubits {:?}", cmd.qubits);
                    self.simulator.h(&cmd.qubits);
                }
                GateType::SZ => {
                    debug!("Processing SZ gate on qubits {:?}", cmd.qubits);
                    self.simulator.sz(&cmd.qubits);
                }
                GateType::SZdg => {
                    debug!("Processing SZdg gate on qubits {:?}", cmd.qubits);
                    self.simulator.szdg(&cmd.qubits);
                }
                GateType::SX => {
                    debug!("Processing SX gate on qubits {:?}", cmd.qubits);
                    self.simulator.sx(&cmd.qubits);
                }
                GateType::SXdg => {
                    debug!("Processing SXdg gate on qubits {:?}", cmd.qubits);
                    self.simulator.sxdg(&cmd.qubits);
                }
                GateType::T => {
                    debug!("Processing T gate on qubits {:?}", cmd.qubits);
                    self.simulator.t(&cmd.qubits);
                }
                GateType::Tdg => {
                    debug!("Processing Tdg gate on qubits {:?}", cmd.qubits);
                    self.simulator.tdg(&cmd.qubits);
                }
                GateType::CX => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "CX gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    debug!("Processing CX gate on qubits {:?}", cmd.qubits);
                    let pairs = flat_to_pairs(&cmd.qubits);
                    self.simulator.cx(&pairs);
                }
                GateType::CY => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "CY gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    debug!("Processing CY gate on qubits {:?}", cmd.qubits);
                    let pairs = flat_to_pairs(&cmd.qubits);
                    self.simulator.cy(&pairs);
                }
                GateType::CZ => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "CZ gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    debug!("Processing CZ gate on qubits {:?}", cmd.qubits);
                    let pairs = flat_to_pairs(&cmd.qubits);
                    self.simulator.cz(&pairs);
                }
                // CH = Ry(π/4)_target, CX(control, target), Ry(-π/4)_target
                GateType::CH => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "CH gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    for qubits in cmd.qubits.chunks_exact(2) {
                        debug!(
                            "Processing CH gate with control {:?} and target {:?}",
                            qubits[0], qubits[1]
                        );
                        let target_slice = &[qubits[1]];
                        self.simulator.ry(
                            Angle64::from_radians(std::f64::consts::FRAC_PI_4),
                            target_slice,
                        );
                        self.simulator.cx(&[(qubits[0], qubits[1])]);
                        self.simulator.ry(
                            Angle64::from_radians(-std::f64::consts::FRAC_PI_4),
                            target_slice,
                        );
                    }
                }
                GateType::RZZ => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "RZZ gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    if cmd.angles.is_empty() {
                        return Err(quantum_error("RZZ gate requires at least one angle"));
                    }
                    let angle = cmd.angles[0];
                    debug!("Processing RZZ gate on qubits {:?}", cmd.qubits);
                    let pairs = flat_to_pairs(&cmd.qubits);
                    self.simulator.rzz(angle, &pairs);
                }
                GateType::SZZ => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "SZZ gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    debug!("Processing SZZ gate on qubits {:?}", cmd.qubits);
                    let pairs = flat_to_pairs(&cmd.qubits);
                    self.simulator.szz(&pairs);
                }
                GateType::SZZdg => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "SZZdg gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    debug!("Processing SZZdg gate on qubits {:?}", cmd.qubits);
                    let pairs = flat_to_pairs(&cmd.qubits);
                    self.simulator.szzdg(&pairs);
                }
                GateType::F => {
                    debug!("Processing F gate on qubits {:?}", cmd.qubits);
                    self.simulator.f(&cmd.qubits);
                }
                GateType::Fdg => {
                    debug!("Processing Fdg gate on qubits {:?}", cmd.qubits);
                    self.simulator.fdg(&cmd.qubits);
                }
                GateType::SY => {
                    debug!("Processing SY gate on qubits {:?}", cmd.qubits);
                    self.simulator.sy(&cmd.qubits);
                }
                GateType::SYdg => {
                    debug!("Processing SYdg gate on qubits {:?}", cmd.qubits);
                    self.simulator.sydg(&cmd.qubits);
                }
                GateType::SXX => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "SXX gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    debug!("Processing SXX gate on qubits {:?}", cmd.qubits);
                    let pairs = flat_to_pairs(&cmd.qubits);
                    self.simulator.sxx(&pairs);
                }
                GateType::SXXdg => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "SXXdg gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    debug!("Processing SXXdg gate on qubits {:?}", cmd.qubits);
                    let pairs = flat_to_pairs(&cmd.qubits);
                    self.simulator.sxxdg(&pairs);
                }
                GateType::SYY => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "SYY gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    debug!("Processing SYY gate on qubits {:?}", cmd.qubits);
                    let pairs = flat_to_pairs(&cmd.qubits);
                    self.simulator.syy(&pairs);
                }
                GateType::SYYdg => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "SYYdg gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    debug!("Processing SYYdg gate on qubits {:?}", cmd.qubits);
                    let pairs = flat_to_pairs(&cmd.qubits);
                    self.simulator.syydg(&pairs);
                }
                GateType::SWAP => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "SWAP gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    debug!("Processing SWAP gate on qubits {:?}", cmd.qubits);
                    let pairs = flat_to_pairs(&cmd.qubits);
                    self.simulator.swap(&pairs);
                }
                GateType::CRZ => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "CRZ gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    if cmd.angles.is_empty() {
                        return Err(quantum_error("CRZ gate requires at least one angle"));
                    }
                    let angle = cmd.angles[0];
                    let half_angle = angle / 2u64;
                    // CRZ(θ) = Rz(θ/2) on target, CX, Rz(-θ/2) on target, CX
                    for qubits in cmd.qubits.chunks_exact(2) {
                        debug!(
                            "Processing CRZ gate on qubits {:?} and {:?} with angle {:?}",
                            qubits[0], qubits[1], angle
                        );
                        self.simulator.rz(half_angle, &[qubits[1]]);
                        self.simulator.cx(&[(qubits[0], qubits[1])]);
                        self.simulator.rz(-half_angle, &[qubits[1]]);
                        self.simulator.cx(&[(qubits[0], qubits[1])]);
                    }
                }
                GateType::CCX => {
                    if cmd.qubits.len() % 3 != 0 {
                        return Err(quantum_error(format!(
                            "CCX gate requires a multiple of 3 qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    for qubits in cmd.qubits.chunks_exact(3) {
                        debug!(
                            "Processing CCX gate with controls {:?}, {:?} and target {:?}",
                            qubits[0], qubits[1], qubits[2]
                        );
                        // Toffoli decomposition into Clifford+T gates
                        let c0 = qubits[0];
                        let c1 = qubits[1];
                        let target = qubits[2];
                        // Standard decomposition (15 gates)
                        self.simulator.h(&[target]);
                        self.simulator.cx(&[(c1, target)]);
                        self.simulator.tdg(&[target]);
                        self.simulator.cx(&[(c0, target)]);
                        self.simulator.t(&[target]);
                        self.simulator.cx(&[(c1, target)]);
                        self.simulator.tdg(&[target]);
                        self.simulator.cx(&[(c0, target)]);
                        self.simulator.t(&[c1]);
                        self.simulator.t(&[target]);
                        self.simulator.cx(&[(c0, c1)]);
                        self.simulator.h(&[target]);
                        self.simulator.t(&[c0]);
                        self.simulator.tdg(&[c1]);
                        self.simulator.cx(&[(c0, c1)]);
                    }
                }
                GateType::RX => {
                    if !cmd.angles.is_empty() {
                        let angle = cmd.angles[0];
                        debug!(
                            "Processing RX gate with angle {angle:?} on qubits {:?}",
                            cmd.qubits
                        );
                        self.simulator.rx(angle, &cmd.qubits);
                    }
                }
                GateType::RY => {
                    if !cmd.angles.is_empty() {
                        let angle = cmd.angles[0];
                        debug!(
                            "Processing RY gate with angle {angle:?} on qubits {:?}",
                            cmd.qubits
                        );
                        self.simulator.ry(angle, &cmd.qubits);
                    }
                }
                GateType::RZ => {
                    if !cmd.angles.is_empty() {
                        let angle = cmd.angles[0];
                        debug!(
                            "Processing RZ gate with angle {angle:?} on qubits {:?}",
                            cmd.qubits
                        );
                        self.simulator.rz(angle, &cmd.qubits);
                    }
                }
                GateType::R1XY => {
                    if cmd.angles.len() >= 2 {
                        let theta = cmd.angles[0];
                        let phi = cmd.angles[1];
                        debug!(
                            "Processing R1XY gate with angles theta={theta:?}, phi={phi:?} on qubits {:?}",
                            cmd.qubits
                        );
                        self.simulator.r1xy(theta, phi, &cmd.qubits);
                    }
                }

                // Batch consecutive MZ commands into one simulator call.
                // This enables joint-sampling optimizations (fewer state vector passes).
                GateType::MZ | GateType::MeasureLeaked => {
                    // Collect qubits from consecutive MZ/MeasureLeaked commands
                    let mut mz_qubits: Vec<QubitId> = cmd.qubits.to_vec();
                    while cmd_idx + 1 < batch.len()
                        && matches!(
                            batch[cmd_idx + 1].gate_type,
                            GateType::MZ | GateType::MeasureLeaked
                        )
                    {
                        cmd_idx += 1;
                        mz_qubits.extend_from_slice(&batch[cmd_idx].qubits);
                    }

                    debug!(
                        "Processing batched measurement on {} qubits",
                        mz_qubits.len()
                    );
                    let meas_ids = self.simulator.mz(&mz_qubits);
                    for meas_id in meas_ids {
                        measurements.push(usize::from(meas_id.outcome));
                    }
                }
                GateType::PZ => {
                    debug!("Processing Prep gate on qubits {:?}", cmd.qubits);
                    self.simulator.pz(&cmd.qubits);
                }
                GateType::Channel => {
                    return Err(quantum_error(
                        "Channel gate requires a channel-aware simulator path",
                    ));
                }
                GateType::I
                | GateType::Idle
                | GateType::MeasCrosstalkLocalPayload
                | GateType::MeasCrosstalkGlobalPayload
                | GateType::QFree
                | GateType::Custom
                | GateType::TrackedPauliMeta => {
                    // Just let the system naturally evolve for the specified duration
                    // No active operation needed in the simulator
                    // QFree is a no-op for state vector simulation (qubit tracking is handled elsewhere)
                    // Custom is a no-op placeholder (actual gate name is in metadata)
                }
                GateType::RXX => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "RXX gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    if cmd.angles.is_empty() {
                        return Err(quantum_error("RXX gate requires at least one angle"));
                    }
                    let angle = cmd.angles[0];
                    debug!("Processing RXX gate on qubits {:?}", cmd.qubits);
                    let pairs = flat_to_pairs(&cmd.qubits);
                    self.simulator.rxx(angle, &pairs);
                }
                GateType::RYY => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "RYY gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    if cmd.angles.is_empty() {
                        return Err(quantum_error("RYY gate requires at least one angle"));
                    }
                    let angle = cmd.angles[0];
                    debug!("Processing RYY gate on qubits {:?}", cmd.qubits);
                    let pairs = flat_to_pairs(&cmd.qubits);
                    self.simulator.ryy(angle, &pairs);
                }
                GateType::QAlloc => {
                    // Allocate qubits in |0⟩ state - for state vector sim, same as Prep
                    debug!("Processing QAlloc gate on qubits {:?}", cmd.qubits);
                    self.simulator.pz(&cmd.qubits);
                }
                GateType::MeasureFree => {
                    // Measure and deallocate - measure first, then the qubit is implicitly freed
                    debug!("Processing MeasureFree gate on qubits {:?}", cmd.qubits);
                    let meas_ids = self.simulator.mz(&cmd.qubits);
                    for meas_id in meas_ids {
                        measurements.push(usize::from(meas_id.outcome));
                    }
                }
                GateType::U => {
                    if cmd.angles.len() >= 3 {
                        let theta = cmd.angles[0];
                        let phi = cmd.angles[1];
                        let lambda = cmd.angles[2];
                        debug!(
                            "Processing U gate with angles theta={theta:?}, phi={phi:?}, lambda={lambda:?} on qubits {:?}",
                            cmd.qubits
                        );
                        self.simulator.u(theta, phi, lambda, &cmd.qubits);
                    }
                }
                GateType::RXXRYYRZZ => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "RXXRYYRZZ gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    if cmd.angles.len() < 3 {
                        return Err(quantum_error(
                            "RXXRYYRZZ gate requires 3 angles (alpha, beta, gamma)",
                        ));
                    }
                    let alpha = cmd.angles[0];
                    let beta = cmd.angles[1];
                    let gamma = cmd.angles[2];
                    debug!("Processing RXXRYYRZZ gate on qubits {:?}", cmd.qubits);
                    let pairs = flat_to_pairs(&cmd.qubits);
                    self.simulator.rxxryyrzz(alpha, beta, gamma, &pairs);
                }
                GateType::U2q => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "U2q gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    if cmd.angles.len() < 15 {
                        return Err(quantum_error("U2q gate requires 15 angles"));
                    }
                    let before = [
                        [cmd.angles[0], cmd.angles[1], cmd.angles[2]],
                        [cmd.angles[3], cmd.angles[4], cmd.angles[5]],
                    ];
                    let interaction = [cmd.angles[6], cmd.angles[7], cmd.angles[8]];
                    let after = [
                        [cmd.angles[9], cmd.angles[10], cmd.angles[11]],
                        [cmd.angles[12], cmd.angles[13], cmd.angles[14]],
                    ];
                    debug!("Processing U2q gate on qubits {:?}", cmd.qubits);
                    let pairs = flat_to_pairs(&cmd.qubits);
                    self.simulator.u2q(before, interaction, after, &pairs);
                }
            }
            cmd_idx += 1;
        }

        // Create a message with the measurement results
        let mut builder = ByteMessage::outcomes_builder();
        builder.add_outcomes(&measurements);

        Ok(builder.build())
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        debug!("StateVecEngine: reset() called");
        self.simulator.reset();
        Ok(())
    }
}

impl<S: StateVectorSimulator> RngManageable for StateVectorEngine<S>
where
    <S as RngManageable>::Rng: Clone,
{
    type Rng = <S as RngManageable>::Rng;

    fn set_rng(&mut self, rng: Self::Rng) {
        self.simulator.set_rng(rng);
    }

    fn rng(&self) -> &Self::Rng {
        self.simulator.rng()
    }

    fn rng_mut(&mut self) -> &mut Self::Rng {
        self.simulator.rng_mut()
    }
}

impl<S: StateVectorSimulator + 'static> QuantumEngine for StateVectorEngine<S>
where
    <S as RngManageable>::Rng: Clone,
{
    fn set_seed(&mut self, seed: u64) {
        let rng = <S as RngManageable>::Rng::seed_from_u64(seed);
        self.simulator.set_rng(rng);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// A quantum engine that uses a stabilizer simulator
#[derive(Debug, Clone)]
pub struct SparseStabEngine {
    simulator: SparseStab,
}

impl SparseStabEngine {
    /// Create a new stabilizer engine with the specified number of qubits
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            simulator: SparseStab::new(num_qubits),
        }
    }

    /// Create a new stabilizer engine with a specific seed
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits in the system
    /// * `seed` - Seed value for the random number generator
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self {
            simulator: SparseStab::with_seed(num_qubits, seed),
        }
    }
}

impl Engine for SparseStabEngine {
    type Input = ByteMessage;
    type Output = ByteMessage;

    fn process(&mut self, message: Self::Input) -> Result<Self::Output, PecosError> {
        process_clifford_message(&mut self.simulator, &message)
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        self.simulator.reset();
        Ok(())
    }
}

impl RngManageable for SparseStabEngine {
    type Rng = <SparseStab as RngManageable>::Rng;

    fn set_rng(&mut self, rng: Self::Rng) {
        self.simulator.set_rng(rng);
    }

    /// Get a read-only reference to the internal random number generator
    ///
    /// This method delegates to the underlying simulator's RNG
    ///
    /// # Returns
    /// A reference to the internal RNG
    fn rng(&self) -> &Self::Rng {
        self.simulator.rng()
    }

    /// Get a mutable reference to the internal random number generator
    ///
    /// This method delegates to the underlying simulator's RNG
    ///
    /// # Returns
    /// A mutable reference to the internal RNG
    fn rng_mut(&mut self) -> &mut Self::Rng {
        self.simulator.rng_mut()
    }
}

impl QuantumEngine for SparseStabEngine {
    fn set_seed(&mut self, seed: u64) {
        // Create a new RNG with the given seed
        let rng = <SparseStab as RngManageable>::Rng::seed_from_u64(seed);

        // Set the simulator's RNG
        self.simulator.set_rng(rng);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ============================================================================
// Stabilizer Engine (wraps the recommended Stabilizer type)
// ============================================================================

/// A quantum engine that uses the recommended [`Stabilizer`] simulator.
///
/// This tracks whatever implementation `Stabilizer` selects internally.
/// Prefer this over `SparseStabEngine` for new code.
#[derive(Debug, Clone)]
pub struct StabilizerEngine {
    simulator: Stabilizer,
}

impl StabilizerEngine {
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            simulator: Stabilizer::new(num_qubits),
        }
    }

    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self {
            simulator: Stabilizer::with_seed(num_qubits, seed),
        }
    }
}

impl Engine for StabilizerEngine {
    type Input = ByteMessage;
    type Output = ByteMessage;

    fn process(&mut self, message: Self::Input) -> Result<Self::Output, PecosError> {
        process_clifford_message(&mut self.simulator, &message)
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        self.simulator.reset();
        Ok(())
    }
}

impl RngManageable for StabilizerEngine {
    type Rng = <Stabilizer as RngManageable>::Rng;

    fn set_rng(&mut self, rng: Self::Rng) {
        self.simulator.set_rng(rng);
    }

    fn rng(&self) -> &Self::Rng {
        self.simulator.rng()
    }

    fn rng_mut(&mut self) -> &mut Self::Rng {
        self.simulator.rng_mut()
    }
}

impl QuantumEngine for StabilizerEngine {
    fn set_seed(&mut self, seed: u64) {
        let rng = <Stabilizer as RngManageable>::Rng::seed_from_u64(seed);
        self.simulator.set_rng(rng);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ============================================================================
// StabVec Engine
// ============================================================================

/// A quantum engine that uses the `StabVec` simulator.
///
/// Supports all Clifford gates plus arbitrary rotation gates (RZ, RX, RY, RZZ, etc.)
/// via sum-over-Cliffords decomposition. More efficient than state vector for
/// circuits with many qubits and few non-Clifford gates.
#[derive(Debug, Clone)]
pub struct StabVecEngine {
    simulator: StabVec,
}

impl StabVecEngine {
    /// Create a new `StabVec` engine with the specified number of qubits.
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            simulator: StabVec::new(num_qubits),
        }
    }

    /// Create with a specific seed.
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self {
            simulator: StabVec::new_with_seed(num_qubits, seed),
        }
    }
}

impl Engine for StabVecEngine {
    type Input = ByteMessage;
    type Output = ByteMessage;

    fn process(&mut self, message: Self::Input) -> Result<Self::Output, PecosError> {
        process_general_message(&mut self.simulator, &message)
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        self.simulator.reset();
        Ok(())
    }
}

impl RngManageable for StabVecEngine {
    type Rng = PecosRng;

    fn set_rng(&mut self, rng: Self::Rng) {
        self.simulator.set_rng(rng);
    }

    fn rng(&self) -> &Self::Rng {
        self.simulator.rng()
    }

    fn rng_mut(&mut self) -> &mut Self::Rng {
        self.simulator.rng_mut()
    }
}

impl QuantumEngine for StabVecEngine {
    fn set_seed(&mut self, seed: u64) {
        let rng = PecosRng::seed_from_u64(seed);
        self.simulator.set_rng(rng);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ============================================================================
// Density Matrix Engine
// ============================================================================

/// A quantum engine that uses the density matrix simulator.
///
/// Supports all Clifford gates plus arbitrary rotation gates. Uses the
/// Choi-Jamiolkowski isomorphism to represent an N-qubit density matrix
/// as a 2N-qubit state vector, enabling simulation of mixed states and noise.
#[derive(Debug, Clone)]
pub struct DensityMatrixEngine {
    simulator: DensityMatrix,
}

impl DensityMatrixEngine {
    /// Create a new density matrix engine with the specified number of qubits.
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            simulator: DensityMatrix::new(num_qubits),
        }
    }

    /// Create with a specific seed.
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self {
            simulator: DensityMatrix::with_seed(num_qubits, seed),
        }
    }
}

impl Engine for DensityMatrixEngine {
    type Input = ByteMessage;
    type Output = ByteMessage;

    fn process(&mut self, message: Self::Input) -> Result<Self::Output, PecosError> {
        process_general_message(&mut self.simulator, &message)
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        self.simulator.reset();
        Ok(())
    }
}

impl RngManageable for DensityMatrixEngine {
    type Rng = PecosRng;

    fn set_rng(&mut self, rng: Self::Rng) {
        self.simulator.set_rng(rng);
    }

    fn rng(&self) -> &Self::Rng {
        self.simulator.rng()
    }

    fn rng_mut(&mut self) -> &mut Self::Rng {
        self.simulator.rng_mut()
    }
}

impl QuantumEngine for DensityMatrixEngine {
    fn set_seed(&mut self, seed: u64) {
        let rng = PecosRng::seed_from_u64(seed);
        self.simulator.set_rng(rng);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// A quantum engine that uses a coin toss simulator
///
/// This engine ignores all quantum gates and returns random measurement results.
/// Useful for testing classical control logic without quantum overhead.
#[derive(Debug, Clone)]
pub struct CoinTossEngine {
    simulator: CoinToss,
}

impl CoinTossEngine {
    /// Create a new coin toss engine with the specified number of qubits
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            simulator: CoinToss::new(num_qubits),
        }
    }

    /// Create a new coin toss engine with a specific seed
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self {
            simulator: CoinToss::with_seed(num_qubits, Some(seed)),
        }
    }

    /// Create a new coin toss engine with custom probability
    #[must_use]
    pub fn with_prob(num_qubits: usize, prob: f64) -> Self {
        Self {
            simulator: CoinToss::with_prob(num_qubits, prob),
        }
    }

    /// Create a new coin toss engine with custom probability and seed
    #[must_use]
    pub fn with_prob_and_seed(num_qubits: usize, prob: f64, seed: u64) -> Self {
        Self {
            simulator: CoinToss::with_prob_and_seed(num_qubits, prob, Some(seed)),
        }
    }
}

impl Engine for CoinTossEngine {
    type Input = ByteMessage;
    type Output = ByteMessage;

    fn process(&mut self, message: Self::Input) -> Result<Self::Output, PecosError> {
        let batch = message.quantum_ops()?;
        let mut measurements = Vec::new();

        for cmd in &batch {
            match cmd.gate_type {
                // All gates are no-ops for CoinToss - only measurements matter
                GateType::MZ | GateType::MeasureLeaked | GateType::MeasureFree => {
                    for q in &cmd.qubits {
                        debug!("CoinToss: Processing measurement on qubit {q:?}");
                        let meas_ids = self.simulator.mz(&[*q]);
                        for meas_id in &meas_ids {
                            let outcome = u32::from(meas_id.outcome);
                            measurements.push(outcome);
                        }
                    }
                }
                // All other gates are ignored
                _ => {}
            }
        }

        // Create a message with the measurement results
        let mut builder = ByteMessage::outcomes_builder();
        let outcomes: Vec<usize> = measurements.iter().map(|&m| m as usize).collect();
        builder.add_outcomes(&outcomes);

        Ok(builder.build())
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        self.simulator.reset();
        Ok(())
    }
}

impl RngManageable for CoinTossEngine {
    type Rng = <CoinToss as RngManageable>::Rng;

    fn set_rng(&mut self, rng: Self::Rng) {
        self.simulator.set_rng(rng);
    }

    fn rng(&self) -> &Self::Rng {
        self.simulator.rng()
    }

    fn rng_mut(&mut self) -> &mut Self::Rng {
        self.simulator.rng_mut()
    }
}

impl QuantumEngine for CoinTossEngine {
    fn set_seed(&mut self, seed: u64) {
        self.simulator.set_seed(seed);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Creates a new quantum engine that supports both Clifford gates and arbitrary rotation gates
///
/// This factory function creates a new `StateVecEngine` and returns it as a boxed `QuantumEngine`
/// trait object, allowing for polymorphic usage.
///
/// # Parameters
/// * `simulator` - A state vector simulator
///
/// # Returns
/// A boxed `QuantumEngine` trait object
#[must_use]
pub fn new_quantum_engine_arbitrary_qgate(simulator: StateVec) -> Box<dyn QuantumEngine> {
    Box::new(StateVecEngine { simulator })
}

/// Creates a new quantum engine with a specific seed
///
/// This factory function creates a new `StateVecEngine` with a specific seed
/// and returns it as a boxed `QuantumEngine` trait object.
///
/// # Parameters
/// * `num_qubits` - Number of qubits in the system
/// * `seed` - Seed value for the random number generator
///
/// # Returns
/// A boxed `QuantumEngine` trait object
#[must_use]
pub fn new_quantum_engine_with_seed(num_qubits: usize, seed: u64) -> Box<dyn QuantumEngine> {
    Box::new(StateVecEngine::with_seed(num_qubits, seed))
}

/// Creates a new stabilizer quantum engine
///
/// This factory function creates a new `SparseStabEngine` and returns it as a boxed `QuantumEngine`
/// trait object, allowing for polymorphic usage.
///
/// # Parameters
/// * `num_qubits` - Number of qubits in the system
///
/// # Returns
/// A boxed `QuantumEngine` trait object
#[must_use]
pub fn new_stabilizer_engine(num_qubits: usize) -> Box<dyn QuantumEngine> {
    Box::new(SparseStabEngine::new(num_qubits))
}

/// Creates a new stabilizer quantum engine with a specific seed
///
/// This factory function creates a new `SparseStabEngine` with a specific seed
/// and returns it as a boxed `QuantumEngine` trait object.
///
/// # Parameters
/// * `num_qubits` - Number of qubits in the system
/// * `seed` - Seed value for the random number generator
///
/// # Returns
/// A boxed `QuantumEngine` trait object
#[must_use]
pub fn new_stabilizer_engine_with_seed(num_qubits: usize, seed: u64) -> Box<dyn QuantumEngine> {
    Box::new(SparseStabEngine::with_seed(num_qubits, seed))
}
