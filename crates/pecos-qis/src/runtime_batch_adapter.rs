//! Runtime-independent normalization before simulation and operation tracing.
//!
//! The adapter owns interpretation of opaque events. PECOS owns timing and
//! measurement routing. No private runtime protocol is built into this boundary.

use crate::runtime::{Result, RuntimeError};
use crate::scheduled::{MAX_PAYLOAD_BYTES, RuntimeScheduledOp};

/// Maximum source or normalized operations admitted by a runtime batch adapter.
pub const MAX_ADAPTER_OPERATIONS: usize = 4096;

/// Original native output, in emission order, before idle insertion.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RuntimeBatch {
    pub shot_id: u64,
    pub batch_index: usize,
    pub num_qubits: usize,
    pub start_time_nanos: u64,
    pub duration_nanos: u64,
    pub operations: Vec<RuntimeScheduledOp>,
}

/// Deterministic, shot-local interpretation of a runtime's custom semantics.
///
/// Implementations must validate every custom event and reject unknown schemas
/// or effects they cannot represent. Return ordinary native operations only;
/// understood metadata may produce no operation. Physical effects must be
/// represented, not acknowledged as metadata. Native measurements must retain
/// their order, kind, targets and result IDs. The adapter cannot retime a batch
/// or insert its own idle gaps; PECOS derives those from the original schedule.
///
/// The factory creates a fresh instance for every shot. No outcome, RNG or
/// simulator state is provided. This interface cannot implement stochastic
/// channels or effects that depend on measurement outcomes during a batch.
pub trait RuntimeBatchAdapter: Send + Sync {
    /// Normalize a complete batch, or reject unsupported semantics.
    /// # Errors
    /// Any failure poisons the runtime until reset.
    fn normalize(&mut self, batch: &RuntimeBatch) -> Result<Vec<RuntimeScheduledOp>>;
}

fn invalid(message: &str) -> RuntimeError {
    RuntimeError::ExecutionError(format!("runtime batch adapter: {message}"))
}

impl RuntimeBatch {
    pub(crate) fn validate(&self) -> Result<()> {
        if self
            .start_time_nanos
            .checked_add(self.duration_nanos)
            .is_none()
        {
            return Err(invalid("batch end time overflow"));
        }
        validate_operations(&self.operations, self.num_qubits, true)
    }

    pub(crate) fn validate_output(&self, output: &[RuntimeScheduledOp]) -> Result<()> {
        validate_operations(output, self.num_qubits, false)?;
        let measured = |op: &&RuntimeScheduledOp| {
            matches!(
                op,
                RuntimeScheduledOp::Measure { .. } | RuntimeScheduledOp::MeasureLeaked { .. }
            )
        };
        if !self
            .operations
            .iter()
            .filter(measured)
            .eq(output.iter().filter(measured))
        {
            return Err(invalid(
                "changed measurement order, kind, target or result ID",
            ));
        }
        Ok(())
    }
}

fn validate_operations(
    operations: &[RuntimeScheduledOp],
    capacity: usize,
    source: bool,
) -> Result<()> {
    if operations.len() > MAX_ADAPTER_OPERATIONS {
        return Err(invalid("operation expansion limit exceeded"));
    }
    let mut payload = 0usize;
    for op in operations {
        let (qubits, angles): (&[u64], &[f64]) = match op {
            RuntimeScheduledOp::Rxy {
                qubit_id,
                theta,
                phi,
            } => (&[*qubit_id], &[*theta, *phi]),
            RuntimeScheduledOp::Rz { qubit_id, theta } => (&[*qubit_id], &[*theta]),
            RuntimeScheduledOp::Rzz {
                qubit_id_1,
                qubit_id_2,
                theta,
            } => (&[*qubit_id_1, *qubit_id_2], &[*theta]),
            RuntimeScheduledOp::Rpp {
                qubit_id_1,
                qubit_id_2,
                theta,
                phi,
            } => (&[*qubit_id_1, *qubit_id_2], &[*theta, *phi]),
            RuntimeScheduledOp::Reset { qubit_id }
            | RuntimeScheduledOp::Measure { qubit_id, .. }
            | RuntimeScheduledOp::MeasureLeaked { qubit_id, .. } => (&[*qubit_id], &[]),
            RuntimeScheduledOp::Custom { data, .. } => {
                if !source {
                    return Err(invalid("normalization left an unhandled custom event"));
                }
                payload = payload
                    .checked_add(data.len())
                    .ok_or_else(|| invalid("payload overflow"))?;
                if payload > MAX_PAYLOAD_BYTES {
                    return Err(invalid("payload limit exceeded"));
                }
                (&[], &[])
            }
        };
        if qubits
            .iter()
            .any(|q| usize::try_from(*q).map_or(true, |q| q >= capacity))
            || angles.iter().any(|a| !a.is_finite())
            || (qubits.len() == 2 && qubits[0] == qubits[1])
        {
            return Err(invalid("invalid qubit or nonfinite angle"));
        }
    }
    Ok(())
}
