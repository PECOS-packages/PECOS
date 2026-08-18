//! Generic adapter for PECOS simulators as Selene plugins.
//!
//! Each PECOS simulator plugin (stabilizer, statevec, clifford-rz) implements
//! [`SeleneSimBehavior`] for its simulator-specific parts. The generic
//! [`SeleneAdapter`] handles the common boilerplate.
//!
//! This reduces three ~300-400 line plugins to three ~50 line plugins plus
//! this shared crate.

use anyhow::{Result, anyhow};
use pecos_core::QubitId;
use pecos_simulators::CliffordGateable;
use selene_core::error_model::BatchResult;
use selene_core::operation::{BatchOperation, Operation};
use selene_core::simulator::SimulatorInterface;
use selene_core::utils::MetricValue;
use std::path::Path;

/// Simulator-specific behavior that each plugin provides.
///
/// The generic adapter handles bounds checking, gate dispatch, measure, exit,
/// and shot lifecycle. The behavior trait handles simulator-specific parts.
#[allow(clippy::missing_errors_doc)]
pub trait SeleneSimBehavior: Send {
    /// The underlying PECOS simulator type.
    type Sim: CliffordGateable;

    /// Create a fresh simulator for a new shot.
    fn create_sim(&self, num_qubits: usize, seed: u64) -> Self::Sim;

    /// Get a mutable reference to the simulator.
    fn sim_mut(&mut self) -> &mut Self::Sim;

    /// Apply `RXY(theta, phi)` to a qubit. Angles in radians.
    fn apply_rxy(&mut self, qubit: QubitId, theta: f64, phi: f64) -> Result<()>;

    /// Apply `RZ(theta)` to a qubit. Angle in radians.
    fn apply_rz(&mut self, qubit: QubitId, theta: f64) -> Result<()>;

    /// Apply `RZZ(theta)` to a pair of qubits. Angle in radians.
    fn apply_rzz(&mut self, q1: QubitId, q2: QubitId, theta: f64) -> Result<()>;

    /// Reset a single qubit to `|0>`.
    fn reset_qubit(&mut self, qubit: QubitId) -> Result<()>;

    /// Postselect a qubit to a target value. Default: error (unsupported).
    fn postselect(&mut self, _qubit: QubitId, _target: bool) -> Result<()> {
        Err(anyhow!("Postselection is not supported by this simulator"))
    }

    /// Get the nth metric. Default: no metrics.
    fn get_metric(&mut self, _nth: u8) -> Result<Option<(String, MetricValue)>> {
        Ok(None)
    }

    /// Dump the simulator state. Default: not supported.
    fn dump_state(&mut self, _file: &Path, _qubits: &[u64]) -> Result<()> {
        Err(anyhow!("State dumping is not supported by this simulator"))
    }

    /// Called at `shot_start` after creating the simulator. Override for per-shot init.
    fn on_shot_start(&mut self) {}
}

/// Generic Selene adapter wrapping any PECOS simulator.
pub struct SeleneAdapter<B: SeleneSimBehavior> {
    pub behavior: B,
    pub num_qubits: u64,
}

/// Convert u64 qubit index to usize. All qubit indices are bounds-checked
/// before this is called, so truncation is safe.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
#[inline]
pub const fn to_usize(value: u64) -> usize {
    value as usize
}

impl<B: SeleneSimBehavior> SeleneAdapter<B> {
    fn check_qubit(&self, qubit: u64, op: &str) -> Result<()> {
        if qubit >= self.num_qubits {
            return Err(anyhow!(
                "{op}(qubit={qubit}) is out of bounds. \
                 qubit must be less than the number of qubits ({}).",
                self.num_qubits
            ));
        }
        Ok(())
    }

    fn check_pair(&self, q1: u64, q2: u64, op: &str) -> Result<()> {
        if q1 >= self.num_qubits || q2 >= self.num_qubits {
            return Err(anyhow!(
                "{op}(qubit1={q1}, qubit2={q2}) is out of bounds. \
                 qubits must be less than the number of qubits ({}).",
                self.num_qubits
            ));
        }
        Ok(())
    }
}

impl<B: SeleneSimBehavior> SimulatorInterface for SeleneAdapter<B> {
    fn exit(&mut self) -> Result<()> {
        Ok(())
    }

    fn shot_start(&mut self, _shot_id: u64, seed: u64) -> Result<()> {
        let sim = self.behavior.create_sim(to_usize(self.num_qubits), seed);
        *self.behavior.sim_mut() = sim;
        self.behavior.on_shot_start();
        Ok(())
    }

    fn shot_end(&mut self) -> Result<()> {
        Ok(())
    }

    fn handle_operations(&mut self, operations: BatchOperation) -> Result<BatchResult> {
        let mut results = BatchResult::default();
        for operation in operations {
            match operation {
                Operation::RXYGate {
                    qubit_id,
                    theta,
                    phi,
                } => {
                    self.check_qubit(qubit_id, "RXY")?;
                    self.behavior
                        .apply_rxy(QubitId(to_usize(qubit_id)), theta, phi)?;
                }
                Operation::RZGate { qubit_id, theta } => {
                    self.check_qubit(qubit_id, "RZ")?;
                    self.behavior.apply_rz(QubitId(to_usize(qubit_id)), theta)?;
                }
                Operation::RZZGate {
                    qubit_id_1,
                    qubit_id_2,
                    theta,
                } => {
                    self.check_pair(qubit_id_1, qubit_id_2, "RZZ")?;
                    self.behavior.apply_rzz(
                        QubitId(to_usize(qubit_id_1)),
                        QubitId(to_usize(qubit_id_2)),
                        theta,
                    )?;
                }
                Operation::Measure {
                    qubit_id,
                    result_id,
                } => {
                    self.check_qubit(qubit_id, "Measure")?;
                    let outcome =
                        self.behavior.sim_mut().mz(&[QubitId(to_usize(qubit_id))])[0].outcome;
                    results.set_bool_result(result_id, outcome);
                }
                Operation::MeasureLeaked {
                    qubit_id,
                    result_id,
                } => {
                    self.check_qubit(qubit_id, "MeasureLeaked")?;
                    let outcome =
                        self.behavior.sim_mut().mz(&[QubitId(to_usize(qubit_id))])[0].outcome;
                    results.set_u64_result(result_id, u64::from(outcome));
                }
                Operation::Reset { qubit_id } => {
                    self.check_qubit(qubit_id, "Reset")?;
                    self.behavior.reset_qubit(QubitId(to_usize(qubit_id)))?;
                }
                Operation::RPPGate { .. } => {
                    return Err(anyhow!("RPP gates are not supported by this simulator"));
                }
                Operation::Custom { .. } => {}
                _ => return Err(anyhow!("Unsupported Selene operation")),
            }
        }
        Ok(results)
    }

    fn postselect(&mut self, qubit: u64, target_value: bool) -> Result<()> {
        self.check_qubit(qubit, "Postselect")?;
        self.behavior
            .postselect(QubitId(to_usize(qubit)), target_value)
    }

    fn get_metric(&mut self, nth_metric: u8) -> Result<Option<(String, MetricValue)>> {
        self.behavior.get_metric(nth_metric)
    }

    fn dump_state(&mut self, file: &Path, qubits: &[u64]) -> Result<()> {
        self.behavior.dump_state(file, qubits)
    }
}
