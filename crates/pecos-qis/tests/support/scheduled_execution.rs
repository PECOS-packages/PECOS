//! Experimental example/test harness for original native schedules.
//! Not a supported library API.
//!
//! This diagnostic Rust path uses the existing `QuantumSystem` and state-vector
//! engine. It admits an entire extraction result before quantum mutation. It is
//! not connected to Python or `QisEngine`. Only the optional idle-Z profile below
//! consumes timing; arbitrary noise configurations and all custom events reject.

use pecos_core::{Angle64, errors::PecosError};
use pecos_engines::runtime_frame::ShotContext;
use pecos_engines::{ByteMessage, Engine, quantum::StateVecEngine, quantum_system::QuantumSystem};
use pecos_qis::runtime::{QisRuntime, Shot};
use pecos_qis::scheduled::{RuntimeScheduledOp, ScheduledBatch};
use pecos_qis_ffi_types::Operation;
use std::collections::{BTreeMap, BTreeSet};

fn error(message: impl Into<String>) -> PecosError {
    PecosError::Input(message.into())
}
fn runtime_error(error: &pecos_qis::runtime::RuntimeError) -> PecosError {
    PecosError::Processing(error.to_string())
}

/// Original input boundaries and resulting measurements from one execution call.
#[derive(Debug)]
pub struct ScheduledExecutionOutput {
    /// Explicit caller-assigned host identity; never inferred from native shot IDs.
    pub context: ShotContext,
    /// Original native batches, with timing and opaque data unmodified.
    pub batches: Vec<ScheduledBatch>,
    /// Processed measurements keyed by source-program result ID.
    pub measurements: BTreeMap<usize, u32>,
}

/// Narrow timing consumer: Z faults only, no leakage, gate or readout faults.
/// Linear rate is per second; sine and coherent rates are radians per second.
#[derive(Clone, Copy, Debug, Default)]
pub struct IdleZNoise {
    pub linear: f64,
    pub sine: f64,
    pub coherent: f64,
}
impl IdleZNoise {
    fn validate(self, seconds: f64) -> Result<(), PecosError> {
        if [self.linear, self.sine, self.coherent]
            .into_iter()
            .any(|rate| !rate.is_finite() || rate < 0.0 || !(rate * seconds).is_finite())
        {
            return Err(error("invalid idle-Z rate or duration product"));
        }
        Ok(())
    }
}

struct AdmittedSchedule {
    commands: ByteMessage,
    ids: Vec<usize>,
    next_batch: usize,
    end_times: Vec<u64>,
}

/// Experimental ideal or idle-Z execution for testing native scheduling and feedback.
///
/// Owns runtime and quantum state together; no live cloning or mutable component
/// access is offered. A failed submission poisons the owner until both components
/// reset successfully. State-vector capacity is fixed at construction (1–16 qubits).
/// This consumer rejects all custom events and cannot be used as a device model.
pub struct ScheduledExecutor {
    runtime: Box<dyn QisRuntime>,
    quantum: QuantumSystem,
    capacity: usize,
    context: Option<ShotContext>,
    runtime_shot_id: u64,
    next_batch: usize,
    end_times: Vec<u64>,
    poisoned: bool,
    runtime_ready: bool,
    idle_noise: Option<IdleZNoise>,
}

impl ScheduledExecutor {
    /// Construct the fixed-capacity state-vector consumer.
    ///
    /// # Errors
    /// Rejects zero or more than 16 qubits, bounding state-vector storage.
    pub fn new(mut runtime: Box<dyn QisRuntime>, qubits: usize) -> Result<Self, PecosError> {
        if !(1..=16).contains(&qubits) {
            return Err(error("scheduled state-vector capacity must be 1..=16"));
        }
        runtime.set_num_qubits(qubits);
        Ok(Self {
            runtime,
            quantum: QuantumSystem::new_without_noise(Box::new(StateVecEngine::new(qubits))),
            capacity: qubits,
            context: None,
            runtime_shot_id: 0,
            next_batch: 0,
            end_times: vec![0; qubits],
            poisoned: false,
            runtime_ready: false,
            idle_noise: None,
        })
    }

    /// Use existing `GeneralNoiseModel` idle channels with native batch lifecycles.
    /// All other noise channels remain at their zero defaults.
    ///
    /// # Errors
    /// Rejects invalid rates, model configuration, or qubit capacity.
    pub fn with_idle_z(
        runtime: Box<dyn QisRuntime>,
        qubits: usize,
        noise: IdleZNoise,
    ) -> Result<Self, PecosError> {
        noise.validate(1.0)?;
        let mut executor = Self::new(runtime, qubits)?;
        let z = BTreeMap::from([("Z".to_owned(), 1.0)]);
        let rz = BTreeMap::from([("RZ".to_owned(), 1.0)]);
        let builder = pecos_engines::noise::GeneralNoiseModelBuilder::new()
            .with_p_idle_linear(noise.linear, &z)
            .with_p_idle_sin_squared(noise.sine, &z)
            .with_p_idle_coherent(noise.coherent, &rz);
        builder.validate_configuration().map_err(error)?;
        executor.quantum = QuantumSystem::new(
            Box::new(builder.build()),
            Box::new(StateVecEngine::new(qubits)),
        );
        executor.idle_noise = Some(noise);
        Ok(executor)
    }

    /// Reset both components, abandoning a failed or unfinished shot.
    ///
    /// # Errors
    /// Any component failure keeps this owner poisoned, including caught panics.
    pub fn reset(&mut self) -> Result<(), PecosError> {
        self.poisoned = true;
        self.context = None;
        self.runtime
            .reset()
            .map_err(|error| runtime_error(&error))?;
        self.quantum.reset()?;
        self.next_batch = 0;
        self.end_times.fill(0);
        self.runtime_ready = true;
        self.poisoned = false;
        Ok(())
    }

    /// Begin a shot with explicit host and native identities. After clean completion,
    /// reuse the native instance and reset only quantum state and host bookkeeping.
    /// Callers must assign distinct host contexts to independent shots/workers.
    /// The seed is supplied to the runtime and to the existing quantum seed setup.
    ///
    /// # Errors
    /// An active shot must finish or be explicitly reset. Failed owners require reset.
    pub fn start_shot(
        &mut self,
        context: ShotContext,
        runtime_shot_id: u64,
        seed: u64,
    ) -> Result<(), PecosError> {
        if self.poisoned || self.context.is_some() {
            return Err(error("finish or reset the current scheduled shot"));
        }
        if !self.runtime_ready {
            self.reset()?;
        }
        self.poisoned = true;
        self.quantum.reset()?;
        self.next_batch = 0;
        self.end_times.fill(0);
        self.quantum.set_seed(seed);
        self.quantum.begin_shot(context)?;
        self.runtime
            .shot_start(runtime_shot_id, Some(seed))
            .map_err(|error| runtime_error(&error))?;
        self.runtime_shot_id = runtime_shot_id;
        self.context = Some(context);
        self.poisoned = false;
        Ok(())
    }

    fn require_active(&self) -> Result<ShotContext, PecosError> {
        if self.poisoned {
            return Err(error(
                "scheduled executor failed; successful reset required",
            ));
        }
        self.context
            .ok_or_else(|| error("scheduled executor requires start_shot"))
    }

    /// Submit native operations and execute all returned batches. Schedulers may
    /// retain work until a barrier; empty output is not completion.
    ///
    /// # Errors
    /// Rejects unsupported events, invalid timing/identity or malformed measurement
    /// mappings before quantum execution. Extraction already mutated the runtime,
    /// so any failure poisons the whole owner. Feedback failures also poison it.
    pub fn submit(
        &mut self,
        operations: &[Operation],
    ) -> Result<ScheduledExecutionOutput, PecosError> {
        let context = self.require_active()?;
        self.poisoned = true;
        let batches = self
            .runtime
            .lower_scheduled_operations(operations)
            .map_err(|error| runtime_error(&error))?;
        let output = self.execute(context, batches)?;
        self.poisoned = false;
        Ok(output)
    }

    /// Force terminal release, execute it, deliver results and end the runtime shot.
    ///
    /// # Errors
    /// Any drain, execution, feedback or shot-end failure poisons the owner.
    pub fn finish_shot(&mut self) -> Result<(ScheduledExecutionOutput, Shot), PecosError> {
        let context = self.require_active()?;
        self.poisoned = true;
        let batches = self
            .runtime
            .drain_pending_scheduled_operations()
            .map_err(|error| runtime_error(&error))?;
        let output = self.execute(context, batches)?;
        let shot = self
            .runtime
            .shot_end()
            .map_err(|error| runtime_error(&error))?;
        self.context = None;
        self.poisoned = false;
        Ok((output, shot))
    }

    fn execute(
        &mut self,
        context: ShotContext,
        batches: Vec<ScheduledBatch>,
    ) -> Result<ScheduledExecutionOutput, PecosError> {
        // Prepare the entire extraction result first. No simulator/noise call is
        // made until every batch and measurement mapping has been admitted.
        let AdmittedSchedule {
            next_batch,
            end_times,
            ..
        } = self.admit(&batches)?;
        // Keep the whole-result pass for cross-batch measurement-ID uniqueness.
        // Both profiles then use the same original native batch boundaries.
        let mut inputs = Vec::new();
        let mut cursor = self.next_batch;
        let mut ends = self.end_times.clone();
        for batch in &batches {
            let admitted = self.admit_at(std::slice::from_ref(batch), cursor, ends)?;
            cursor = admitted.next_batch;
            ends = admitted.end_times;
            if !admitted.commands.is_empty()? {
                inputs.push((admitted.commands, admitted.ids));
            }
        }
        let mut measurements = BTreeMap::new();
        for (commands, ids) in inputs {
            let output = self.quantum.process(commands)?;
            let outcomes = output.outcomes().map_err(|e| error(e.to_string()))?;
            if outcomes.len() != ids.len() {
                return Err(error("scheduled measurement count mismatch"));
            }
            let feedback = ids.into_iter().zip(outcomes).collect::<BTreeMap<_, _>>();
            self.runtime
                .provide_measurement_outcomes(feedback.clone())
                .map_err(|error| runtime_error(&error))?;
            measurements.extend(feedback);
        }
        self.next_batch = next_batch;
        self.end_times = end_times;
        Ok(ScheduledExecutionOutput {
            context,
            batches,
            measurements,
        })
    }

    fn admit(&self, batches: &[ScheduledBatch]) -> Result<AdmittedSchedule, PecosError> {
        self.admit_at(batches, self.next_batch, self.end_times.clone())
    }

    fn admit_at(
        &self,
        batches: &[ScheduledBatch],
        mut next_batch: usize,
        mut end_times: Vec<u64>,
    ) -> Result<AdmittedSchedule, PecosError> {
        let mut builder = ByteMessage::quantum_operations_builder();
        let mut ids = Vec::new();
        let mut program_ids = BTreeSet::new();
        let mut native_ids = BTreeSet::new();
        for batch in batches {
            if batch.runtime_shot_id != self.runtime_shot_id || batch.batch_index != next_batch {
                return Err(error("scheduled shot/batch identity mismatch"));
            }
            next_batch = next_batch
                .checked_add(1)
                .ok_or_else(|| error("scheduled ordinal overflow"))?;
            let end = batch
                .start_time_nanos
                .checked_add(batch.duration_nanos)
                .ok_or_else(|| error("scheduled time overflow"))?;
            let mut mappings = BTreeMap::new();
            for m in &batch.measurements {
                if m.operation_index >= batch.operations.len()
                    || mappings.insert(m.operation_index, m).is_some()
                {
                    return Err(error("invalid scheduled measurement position"));
                }
            }
            // Multiple operations in one native batch may touch a qubit (e.g.
            // reset plus pulse); check against earlier batches, then advance once.
            let mut touched = BTreeSet::new();
            for (index, op) in batch.operations.iter().enumerate() {
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
                    RuntimeScheduledOp::Custom { .. } => {
                        return Err(error("scheduled consumer does not admit custom events"));
                    }
                };
                if angles.iter().any(|a| !a.is_finite())
                    || (qubits.len() == 2 && qubits[0] == qubits[1])
                {
                    return Err(error("invalid scheduled gate"));
                }
                let mut targets = Vec::new();
                for q in qubits {
                    let q = usize::try_from(*q).map_err(|_| error("scheduled target overflow"))?;
                    if q >= self.capacity || batch.start_time_nanos < end_times[q] {
                        return Err(error("scheduled target capacity or overlapping timing"));
                    }
                    // A repeated target within one indivisible native batch
                    // has one preceding idle interval, not one per operation.
                    if touched.insert(q)
                        && let Some(noise) = self.idle_noise
                    {
                        let gap = batch.start_time_nanos - end_times[q];
                        let seconds = std::time::Duration::from_nanos(gap).as_secs_f64();
                        noise.validate(seconds)?;
                        if gap > 0 {
                            builder.idle(seconds, &[q]);
                        }
                    }
                    targets.push(q);
                }
                match op {
                    RuntimeScheduledOp::Rxy { theta, phi, .. } => {
                        builder.rxy1q(
                            Angle64::from_radians(*theta),
                            Angle64::from_radians(*phi),
                            &targets,
                        );
                    }
                    RuntimeScheduledOp::Rz { theta, .. } => {
                        builder.rz(Angle64::from_radians(*theta), &targets);
                    }
                    RuntimeScheduledOp::Rzz { theta, .. } => {
                        builder.rzz(Angle64::from_radians(*theta), &[(targets[0], targets[1])]);
                    }
                    RuntimeScheduledOp::Rpp { theta, phi, .. } => {
                        builder.rxyxy2q(
                            Angle64::from_radians(*theta),
                            Angle64::from_radians(*phi),
                            &[(targets[0], targets[1])],
                        );
                    }
                    RuntimeScheduledOp::Reset { .. } => {
                        builder.pz(&targets);
                    }
                    RuntimeScheduledOp::Measure { result_id, .. }
                    | RuntimeScheduledOp::MeasureLeaked { result_id, .. } => {
                        let m = mappings
                            .remove(&index)
                            .ok_or_else(|| error("missing scheduled measurement mapping"))?;
                        if m.runtime_result != *result_id
                            || !program_ids.insert(m.program_result)
                            || !native_ids.insert(*result_id)
                            || (matches!(op, RuntimeScheduledOp::MeasureLeaked { .. })
                                && !m.leakage_aware)
                        {
                            return Err(error(
                                "invalid or duplicate scheduled measurement identity",
                            ));
                        }
                        ids.push(m.program_result);
                        if m.leakage_aware {
                            builder.measure_leakages(&targets);
                        } else {
                            builder.mz(&targets);
                        }
                    }
                    RuntimeScheduledOp::Custom { .. } => unreachable!("rejected before encoding"),
                }
            }
            if !mappings.is_empty() {
                return Err(error("measurement mapping attached to a non-measurement"));
            }
            for q in touched {
                end_times[q] = end;
            }
        }
        Ok(AdmittedSchedule {
            commands: builder.build(),
            ids,
            next_batch,
            end_times,
        })
    }
}

#[cfg(test)]
#[path = "scheduled_execution_tests.rs"]
mod tests;
