//! QIS Control Engine - with trait-based interfaces
//!
//! This module implements a `QisEngine` that works with both
//! trait-based interfaces and runtimes, mediating between them.
//!
//! # Dynamic Circuit Support
//!
//! For programs with conditionals that depend on measurement results (dynamic circuits),
//! the engine runs LLVM execution on a worker thread. When a measurement result is needed:
//! 1. The worker thread pauses and sends pending operations to the main thread
//! 2. The main thread returns operations via `ControlEngine::start()` / `continue_processing()`
//! 3. `continue_processing()` receives measurements and signals the worker to continue
//! 4. The worker resumes with the measurement results available
//!
//! Reset requests cooperative cancellation and reclaims the worker's interface
//! before resetting runtime/scheduler state. Its ten-second deadline bounds only
//! the result-channel wait, not host synchronization or a runtime plugin's exit.
//! Native loops and blocking callbacks that never reach a QIS checkpoint cannot
//! be cancelled; reset fails and stays latched until a later reset reclaims the
//! worker. Python `run` and `reset` share a simulation mutex, so Python reset
//! cannot interrupt a running call. Drop requests cancellation, closes the work
//! channel, and waits up to ten seconds for the worker to finish before joining
//! it. If the worker misses that deadline, Drop warns and detaches it; an abort
//! failure is also reported. Only the thread wait is bounded, not the abort's
//! sync lock. Recovering the original worker through `Clone` remains unsupported.
//! Cancellation also transfers through in-process Selene QIS plugin frames that
//! call PECOS entry points, so plugins must not hold locks or owned resources
//! across any `selene_*` or QIS entry call.

use crate::program::QisInterfaceBuilder;
use crate::qis_interface::{BoxedInterface, DynamicSyncHandle, InterfaceError, ProgramFormat};
use crate::runtime::{QisRuntime, for_each_quantum_qubit};
use crate::scheduled_transport::ScheduledTransport;
use log::{debug, warn};
use pecos_core::Angle64;
use pecos_core::prelude::PecosError;
use pecos_engines::shot_results::{Data, Shot};
use pecos_engines::{
    ByteMessage, ByteMessageBuilder, ClassicalEngine, ControlEngine, Engine, EngineStage,
};
use pecos_qis_ffi_types::{
    LoweredQuantumOp, NamedResultTrace, Operation, OperationCollector as OperationList, QuantumOp,
    TraceMetadata,
};
use pecos_random::PecosRng;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

#[cfg(test)]
use tests::drop_tests;

const RESET_WORKER_TIMEOUT: Duration = Duration::from_secs(10);

static TRACE_ENGINE_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// One lowered quantum gate in a traced batch.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LoweredQuantumGateTrace {
    pub gate_type: String,
    pub angles: Vec<f64>,
    pub params: Vec<f64>,
    pub qubits: Vec<usize>,
    pub measurement_result_ids: Vec<usize>,
    pub metadata: TraceMetadata,
}

/// One traced batch of QIS operations and their lowered simulator commands.
#[derive(Debug, Clone, serde::Serialize)]
pub struct OperationTraceChunk {
    pub format: &'static str,
    pub engine_trace_id: u64,
    pub shot_index: usize,
    pub chunk_index: usize,
    pub stage: String,
    pub waiting_for_result_id: Option<u64>,
    pub current_shot_seed: Option<u64>,
    pub simulated_op_count: usize,
    pub num_operations: usize,
    pub operations: Vec<Operation>,
    pub lowered_quantum_ops: Vec<LoweredQuantumGateTrace>,
    pub lowered_quantum_ops_complete: bool,
    pub named_result_traces: Vec<NamedResultTrace>,
    /// Physical measurement outcomes keyed by stable QIS result id.
    ///
    /// This is populated only on the terminal ``trace_complete`` chunk. It
    /// lets consumers certify aggregate named-result provenance without
    /// relying on when the compiled program happened to read each future.
    pub measurement_results: BTreeMap<usize, u32>,
}

/// Shared in-memory store for traced QIS operation batches.
pub type OperationTraceStore = Arc<Mutex<Vec<OperationTraceChunk>>>;

/// Result from worker thread - returns both the operations and the interface.
///
/// The error arm also carries the interface back when the worker still holds
/// it, so one failed shot does not permanently strip the engine of its
/// interface. `None` means the interface was genuinely lost (worker died).
type WorkerResult =
    Result<(OperationList, BoxedInterface), (WorkerFailure, Option<BoxedInterface>)>;

/// Preserve execution and teardown failures independently across the worker channel.
#[derive(Debug)]
struct WorkerFailure {
    execution: Option<InterfaceError>,
    teardown: Option<InterfaceError>,
}

impl WorkerFailure {
    fn cancelled(&self) -> bool {
        matches!(
            self.execution,
            Some(InterfaceError::ProgramError(
                pecos_qis_ffi_types::ProgramError::Cancelled
            ))
        ) && self.teardown.is_none()
    }
}

impl From<String> for WorkerFailure {
    fn from(message: String) -> Self {
        Self {
            execution: Some(InterfaceError::Other(message)),
            teardown: None,
        }
    }
}

impl std::fmt::Display for WorkerFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(error) = &self.execution {
            write!(f, "{error}")?;
        }
        if let Some(error) = &self.teardown {
            write!(
                f,
                "; worker teardown (disable_dynamic_mode) failed: {error}"
            )?;
        }
        Ok(())
    }
}

/// Simulator commands plus one metadata record per lowered quantum gate.
struct LoweredCommandBatch {
    commands: ByteMessage,
    gate_metadata: Vec<TraceMetadata>,
    /// Whether lowering emitted no command batches (including scheduled formats).
    is_empty: bool,
    /// Imported measurement credits consumed by this command batch.
    measurement_credits_consumed: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum QubitPrepState {
    Pending,
    Prepared,
    Released,
}

/// Certification belongs to a shot attempt, including failures before submission.
#[derive(Debug, PartialEq, Eq)]
enum ShotLifecycle {
    Idle,
    Running,
    Finalized,
    Failed(String),
}

/// State for dynamic circuit execution
///
/// The LLVM program runs in a worker thread. When it needs a measurement result,
/// it blocks in `___read_future_bool`. The main thread simulates operations,
/// provides the result, and signals the worker to continue.
///
/// State for dynamic execution, tracking whether the worker is complete and
/// providing synchronization primitives.
struct DynamicExecutionState {
    /// Whether execution has completed
    execution_complete: bool,
    /// Terminal shot failure (worker error, failed drain verification).
    /// Sticky: completion paths must keep erroring out instead of certifying
    /// a partial trace as a complete shot, even across retries.
    terminal_error: Option<String>,
    /// Whether this shot has already been finalized and certified. A caller
    /// polling `continue_processing` again after `Complete` must not re-run
    /// the drain / `shot_end` gates or emit a second terminal marker.
    finalized: bool,
    /// The operation-lowering route gets one final barrier batch before the
    /// fail-loud drain check. Later emissions must still fail certification.
    terminal_lowering_flushed: bool,
    /// Sync handle for main thread FFI calls
    /// Uses the same library instance (singleton) as the worker thread,
    /// ensuring TLS consistency across platforms
    sync_handle: Option<Box<dyn DynamicSyncHandle>>,
}

/// Work item sent to the persistent dynamic worker thread
struct DynamicWorkItem {
    /// The interface to execute
    interface: BoxedInterface,
}

/// Persistent worker thread for dynamic execution
///
/// This worker thread stays alive across multiple shots, avoiding the overhead
/// and TLS allocation issues that come from spawning a new thread per shot.
/// The thread waits for work items via a channel, executes `collect_operations()`,
/// and sends results back via another channel.
struct PersistentDynamicWorker {
    /// Channel to send work items to the worker
    work_tx: Option<Sender<DynamicWorkItem>>,
    /// Channel to receive results from the worker (wrapped in Mutex for Sync)
    result_rx: Mutex<Receiver<WorkerResult>>,
    /// Joined on drop after a bounded wait; detached with a warning on timeout.
    handle: Option<JoinHandle<()>>,
    /// Deadline for `Drop`'s thread wait.
    drop_timeout: Duration,
    /// Retain cancellation failure for the detach diagnostic.
    abort_error: Option<InterfaceError>,
    #[cfg(test)]
    worker_counts: Arc<drop_tests::WorkerCounts>,
}

impl PersistentDynamicWorker {
    /// Create a new persistent dynamic worker thread
    fn new(#[cfg(test)] worker_counts: Arc<drop_tests::WorkerCounts>) -> Self {
        let (work_tx, work_rx) = mpsc::channel::<DynamicWorkItem>();
        let (result_tx, result_rx) = mpsc::channel::<WorkerResult>();

        #[cfg(test)]
        let exit_guard = drop_tests::WorkerExitGuard::new(Arc::clone(&worker_counts));
        #[cfg(test)]
        let thread_counts = Arc::clone(&worker_counts);
        let handle = std::thread::Builder::new()
            .name("pecos-dynamic-worker".to_string())
            .spawn(move || {
                // Declare first so this witness drops after all worker locals,
                // including the channels, on normal exit and unwinding.
                #[cfg(test)]
                let _exit_guard = exit_guard;
                #[cfg(test)]
                let thread_counts = thread_counts;
                let (work_rx, result_tx) = (work_rx, result_tx);
                debug!("Persistent dynamic worker started");
                while let Ok(work_item) = work_rx.recv() {
                    debug!("Persistent worker: received work item, starting collect_operations");
                    let mut interface = work_item.interface;
                    let result = interface.collect_operations();
                    debug!("Persistent worker: collect_operations returned");

                    // Disable dynamic mode before returning; a teardown
                    // failure poisons the shot like any other worker error.
                    let teardown = interface.disable_dynamic_mode();

                    // Send result back to main thread; the interface goes
                    // back with BOTH outcomes so the engine stays usable.
                    let send_result = match (result, teardown) {
                        (Ok(collector), Ok(())) => Ok((collector, interface)),
                        (execution, teardown) => Err((
                            WorkerFailure {
                                execution: execution.err(),
                                teardown: teardown.err(),
                            },
                            Some(interface),
                        )),
                    };

                    if result_tx.send(send_result).is_err() {
                        // Main thread dropped receiver, exit
                        debug!("Persistent worker: result channel closed, exiting");
                        break;
                    }
                    #[cfg(test)]
                    thread_counts.returned.fetch_add(1, Ordering::SeqCst);
                }
                debug!("Persistent dynamic worker exiting");
            })
            .expect("Failed to spawn persistent dynamic worker thread");

        Self {
            work_tx: Some(work_tx),
            result_rx: Mutex::new(result_rx),
            handle: Some(handle),
            drop_timeout: RESET_WORKER_TIMEOUT,
            abort_error: None,
            #[cfg(test)]
            worker_counts,
        }
    }

    /// Send a work item to the persistent worker
    fn execute(&self, interface: BoxedInterface) -> Result<(), PecosError> {
        debug!(
            "Submitting shot to dynamic worker {:?}",
            self.handle.as_ref().map(|handle| handle.thread().id())
        );
        self.work_tx
            .as_ref()
            .ok_or_else(|| PecosError::Generic("Persistent worker channel closed".to_string()))?
            .send(DynamicWorkItem { interface })
            .map_err(|_| PecosError::Generic("Persistent worker thread died".to_string()))
    }

    /// Bound only the channel wait, using a monotonic deadline and no helper thread.
    fn recv_result_until(&self, deadline: Instant) -> Result<WorkerResult, PecosError> {
        let receiver = self
            .result_rx
            .lock()
            .map_err(|_| PecosError::Generic("dynamic worker result lock poisoned".to_string()))?;
        receiver
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .map_err(|error| {
                PecosError::Generic(match error {
                    mpsc::RecvTimeoutError::Timeout => {
                        "timed out reclaiming dynamic worker interface".to_string()
                    }
                    mpsc::RecvTimeoutError::Disconnected => {
                        "dynamic worker result channel disconnected".to_string()
                    }
                })
            })
    }

    /// Try to receive a result without blocking.
    ///
    /// Worker death (channel disconnect) and a poisoned lock are surfaced as
    /// worker errors rather than `None`: `None` must only ever mean "still
    /// running", or the engine would poll an already-dead worker forever.
    fn try_recv_result(&self) -> Option<WorkerResult> {
        let Ok(rx) = self.result_rx.lock() else {
            return Some(Err((
                "dynamic worker result lock poisoned".to_string().into(),
                None,
            )));
        };
        match rx.try_recv() {
            Ok(result) => Some(result),
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => Some(Err((
                "dynamic worker thread died before returning a result"
                    .to_string()
                    .into(),
                None,
            ))),
        }
    }
}

impl Drop for PersistentDynamicWorker {
    fn drop(&mut self) {
        // Field destruction happens after this method. Close work explicitly,
        // but keep result_rx alive for the worker's final interface handoff.
        drop(self.work_tx.take());
        let Some(handle) = self.handle.take() else {
            return;
        };
        let started = Instant::now();
        while !handle.is_finished() {
            let remaining = self.drop_timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                if let Some(error) = &self.abort_error {
                    warn!(
                        "Dynamic worker did not finish within {:?}; detaching; abort failed: {error}",
                        self.drop_timeout
                    );
                } else {
                    warn!(
                        "Dynamic worker did not finish within {:?}; detaching",
                        self.drop_timeout
                    );
                }
                return;
            }
            std::thread::sleep(remaining.min(Duration::from_millis(1)));
        }
        if let Err(panic) = handle.join() {
            if let Some(message) = panic.downcast_ref::<String>() {
                warn!("Dynamic worker panicked: {message}");
            } else if let Some(message) = panic.downcast_ref::<&str>() {
                warn!("Dynamic worker panicked: {message}");
            } else {
                warn!("Dynamic worker panicked with a non-string payload");
            }
        }
        #[cfg(test)]
        self.worker_counts.joined.fetch_add(1, Ordering::SeqCst);
    }
}

/// Imported measurements remain outstanding even while the runtime holds them.
#[derive(Default)]
struct PendingMeasurements {
    outstanding: usize,
    /// Imported measurements not yet present in a returned command batch.
    unemitted: usize,
}

/// QIS Control Engine that mediates between interface and runtime
///
/// This engine contains:
/// - A `QisInterface` implementation (JIT, Helios, etc.) for executing programs
/// - A `QisRuntime` implementation (Native, Selene, etc.) for managing control flow
///
/// # Dynamic Circuit Support
///
/// The engine always runs LLVM on a worker thread and coordinates via channels.
/// This allows conditionals that depend on measurement results to work correctly.
pub struct QisEngine {
    /// The QIS interface (program executor)
    interface: Option<BoxedInterface>,

    /// The QIS runtime (classical interpreter)
    runtime: Box<dyn QisRuntime>,

    /// Current operations collected from the interface
    current_operations: Option<OperationList>,

    /// High-water mark of physical simulator slots allocated across the current shot.
    ///
    /// Equals `max(slot_index) + 1` over every slot ever activated by
    /// `allocate_qubit_slot`. Because `allocate_qubit_slot` refills freed slots
    /// before extending the range, this is also the minimum number of simulator
    /// slots that must exist to execute the program. Not the count of program
    /// qubit handles — use `active_qubit_slots.len()` for that.
    num_physical_slots: usize,

    /// Optional device-size hint supplied by the top-level simulation builder.
    num_qubits_hint: Option<usize>,

    /// Mapping from program-level qubit handles to physical simulator slots.
    active_qubit_slots: BTreeMap<usize, usize>,

    /// Reusable physical simulator slots freed by `ReleaseQubit`.
    free_qubit_slots: BTreeSet<usize>,

    /// Program-level qubit handles seen during the current shot.
    ///
    /// Some QIS interfaces model initial/static qubits via `allocated_qubits`
    /// metadata instead of explicit `AllocateQubit` operations. We accept a
    /// first use of such a handle and lazily materialize a simulator slot, but
    /// still reject a later use-after-release unless a new `AllocateQubit`
    /// arrives.
    seen_program_qubits: BTreeSet<usize>,

    /// Prep state for source handles, shared by all lowering routes within a shot.
    qubit_prep_states: BTreeMap<usize, QubitPrepState>,

    /// Whether we've started processing
    started: bool,

    /// Tracking measurement result IDs for the current batch
    measurement_mapping: Vec<usize>,

    /// Current measurement outcomes; a later undelivered measurement blocks reuse.
    measurement_results: BTreeMap<usize, u32>,

    /// Per-slot imported measurements awaiting delivery.
    pending_measurements: BTreeMap<usize, PendingMeasurements>,

    /// A failed `Engine::reset`, held until a reset succeeds. Reset drops the
    /// per-shot terminal error with the worker state before resetting the
    /// runtime, so without this a failed runtime reset would let `get_results`
    /// return the previous, failed shot.
    reset_failure: Option<String>,
    shot_lifecycle: ShotLifecycle,
    /// Terminal scheduled drains require feedback before releasing more work.
    scheduled_drain_round: usize,
    scheduled_drain_feedback: bool,

    /// RNG for generating per-shot seeds
    rng: PecosRng,

    /// Current shot seed (stored for quantum engine seeding)
    current_shot_seed: Option<u64>,

    /// Dynamic execution state (when dynamic mode is active)
    dynamic_state: Option<DynamicExecutionState>,

    /// Pending operations from dynamic execution (for current batch)
    pending_dynamic_ops: Vec<Operation>,

    /// Number of operations already simulated (for dynamic mode)
    simulated_op_count: usize,

    /// Program bytes for re-execution in dynamic mode
    program_bytes: Option<Vec<u8>>,

    /// Program format for re-execution
    program_format: Option<ProgramFormat>,

    /// Interface builder for recreating interfaces during clone (dynamic mode)
    interface_builder: Option<Box<dyn QisInterfaceBuilder>>,

    /// Persistent worker thread for dynamic execution (stays alive across shots)
    /// This avoids spawning a new thread per shot, which causes TLS allocation issues.
    persistent_worker: Option<PersistentDynamicWorker>,

    /// Counts only this engine's workers, including clones used by `MonteCarlo`.
    #[cfg(test)]
    worker_counts: Arc<drop_tests::WorkerCounts>,

    /// Directory where operation trace chunks are dumped as JSON.
    operation_trace_dir: Option<PathBuf>,

    /// Optional in-memory collector for traced chunks.
    operation_trace_collector: Option<OperationTraceStore>,

    /// Unique trace id for this engine instance.
    trace_engine_id: u64,

    /// 1-based shot index for operation traces.
    trace_shot_index: usize,

    /// 0-based chunk index within the current shot.
    trace_chunk_index: usize,

    /// Scratch builder reused when materializing command batches.
    command_builder: ByteMessageBuilder,
    pub(crate) scheduled_transport: ScheduledTransport,
}

impl QisEngine {
    fn parse_measurement_outcomes(message: &ByteMessage) -> Result<Vec<u32>, PecosError> {
        message
            .outcomes()
            .map(|outcomes| outcomes.into_iter().collect())
            .map_err(|e| PecosError::Generic(format!("Failed to parse measurements: {e}")))
    }

    /// Consume exactly one outcome for each measurement emitted in this batch.
    /// A mismatch must fail before storing any outcomes or certifying the shot.
    fn map_measurements(&mut self, measurements: &[u32]) -> Result<Vec<(usize, u32)>, PecosError> {
        if let Some(error) = self.terminal_failure_error() {
            return Err(error);
        }
        if self.measurement_mapping.len() != measurements.len() {
            return Err(self.latch_terminal_error(format!(
                "QIS measurement count mismatch: {} queued measurements, {} outcomes",
                self.measurement_mapping.len(),
                measurements.len()
            )));
        }
        Ok(std::mem::take(&mut self.measurement_mapping)
            .into_iter()
            .zip(measurements.iter().copied())
            .collect())
    }

    /// Complete each measurement in order, exposing only a slot's last delivery.
    fn store_measurement_updates(
        &mut self,
        updates: &[(usize, u32)],
    ) -> Result<Vec<(usize, u32)>, PecosError> {
        let mut current = Vec::new();
        for &(result_id, value) in updates {
            let Some(pending) = self.pending_measurements.get_mut(&result_id) else {
                return Err(self.latch_terminal_error(format!(
                    "outcome for result {result_id} has no outstanding measurement"
                )));
            };
            pending.outstanding -= 1;
            if pending.outstanding == 0 {
                self.pending_measurements.remove(&result_id);
                self.measurement_results.insert(result_id, value);
                current.push((result_id, value));
                debug!("QisEngine: Stored current result_id={result_id}, value={value}");
            }
        }
        Ok(current)
    }

    fn provide_measurement_updates_to_runtime(
        &mut self,
        updates: &[(usize, u32)],
    ) -> Result<(), PecosError> {
        if updates.is_empty() {
            return Ok(());
        }
        self.runtime
            .provide_measurement_outcomes(updates.to_vec())
            .map_err(|e| PecosError::Generic(format!("Failed to provide measurements: {e}")))?;
        if self.scheduled_transport.enabled() {
            self.scheduled_drain_feedback = true;
        }
        Ok(())
    }

    /// Create a new engine with the given interface and runtime
    ///
    /// Dynamic execution is always enabled - all LLVM runs on a worker thread.
    #[must_use]
    pub fn new(interface: BoxedInterface, runtime: Box<dyn QisRuntime>) -> Self {
        debug!("Creating QisEngine with dynamic execution");

        Self {
            interface: Some(interface),
            runtime,
            current_operations: None,
            num_physical_slots: 0,
            num_qubits_hint: None,
            active_qubit_slots: BTreeMap::new(),
            free_qubit_slots: BTreeSet::new(),
            seen_program_qubits: BTreeSet::new(),
            qubit_prep_states: BTreeMap::new(),
            started: false,
            measurement_mapping: Vec::new(),
            measurement_results: BTreeMap::new(),
            pending_measurements: BTreeMap::new(),
            reset_failure: None,
            shot_lifecycle: ShotLifecycle::Idle,
            scheduled_drain_round: 0,
            scheduled_drain_feedback: false,
            rng: PecosRng::seed_from_u64(0), // Will be properly seeded via set_seed()
            current_shot_seed: None,
            dynamic_state: None,
            pending_dynamic_ops: Vec::new(),
            simulated_op_count: 0,
            program_bytes: None,
            program_format: None,
            interface_builder: None,
            persistent_worker: None,
            #[cfg(test)]
            worker_counts: Arc::default(),
            operation_trace_dir: None,
            operation_trace_collector: None,
            trace_engine_id: TRACE_ENGINE_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            trace_shot_index: 0,
            trace_chunk_index: 0,
            command_builder: ByteMessageBuilder::new(),
            scheduled_transport: ScheduledTransport::Off,
        }
    }

    /// Get the current shot seed for quantum engine seeding
    /// This should be called after `start()` to get the seed generated for the current shot
    #[must_use]
    pub fn current_shot_seed(&self) -> Option<u64> {
        self.current_shot_seed
    }

    /// Check if the engine has an interface
    #[must_use]
    pub fn has_interface(&self) -> bool {
        self.interface.is_some()
    }

    /// Set the interface builder and program source for dynamic mode cloning
    ///
    /// This stores the information needed to recreate the interface when the engine is cloned.
    /// Required for dynamic execution in `MonteCarloEngine` where the engine is cloned for each worker.
    pub fn set_dynamic_config(
        &mut self,
        builder: Box<dyn QisInterfaceBuilder>,
        program_source: &str,
    ) {
        self.interface_builder = Some(builder);
        self.program_bytes = Some(program_source.as_bytes().to_vec());
        self.program_format = Some(ProgramFormat::LlvmIrText);
    }

    /// Configure a directory where Helios-collected operation chunks are written as JSON.
    pub fn set_operation_trace_dir(&mut self, trace_dir: impl Into<PathBuf>) {
        self.operation_trace_dir = Some(trace_dir.into());
    }

    /// Configure an in-memory collector that receives traced operation chunks.
    pub fn set_operation_trace_collector(&mut self, collector: OperationTraceStore) {
        self.operation_trace_collector = Some(collector);
    }

    /// Initialize the engine for dynamic execution
    ///
    /// This verifies the interface supports dynamic execution and defers
    /// actual execution to `start()`.
    ///
    /// # Errors
    /// Returns an error if no interface is available or it doesn't support dynamic execution.
    pub fn initialize_from_interface(&mut self) -> Result<(), PecosError> {
        if let Some(ref interface) = self.interface {
            if !interface.supports_dynamic() {
                return Err(PecosError::Generic(
                    "QisEngine requires a dynamic-capable interface (e.g., QisHeliosInterface)"
                        .to_string(),
                ));
            }
            // Dynamic mode: defer execution to start()
            debug!("Dynamic mode: deferring operation collection to start()");
            Ok(())
        } else {
            Err(PecosError::Generic("No interface available".to_string()))
        }
    }

    /// Create with just a runtime (interface will be set later)
    #[must_use]
    pub fn with_runtime(runtime: Box<dyn QisRuntime>) -> Self {
        Self {
            interface: None,
            runtime,
            current_operations: None,
            num_physical_slots: 0,
            num_qubits_hint: None,
            active_qubit_slots: BTreeMap::new(),
            free_qubit_slots: BTreeSet::new(),
            seen_program_qubits: BTreeSet::new(),
            qubit_prep_states: BTreeMap::new(),
            started: false,
            measurement_mapping: Vec::new(),
            measurement_results: BTreeMap::new(),
            pending_measurements: BTreeMap::new(),
            reset_failure: None,
            shot_lifecycle: ShotLifecycle::Idle,
            scheduled_drain_round: 0,
            scheduled_drain_feedback: false,
            rng: PecosRng::seed_from_u64(0), // Will be properly seeded via set_seed()
            current_shot_seed: None,
            dynamic_state: None,
            pending_dynamic_ops: Vec::new(),
            simulated_op_count: 0,
            program_bytes: None,
            program_format: None,
            interface_builder: None,
            persistent_worker: None,
            #[cfg(test)]
            worker_counts: Arc::default(),
            operation_trace_dir: None,
            operation_trace_collector: None,
            trace_engine_id: TRACE_ENGINE_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            trace_shot_index: 0,
            trace_chunk_index: 0,
            command_builder: ByteMessageBuilder::new(),
            scheduled_transport: ScheduledTransport::Off,
        }
    }

    /// Set the interface
    pub fn set_interface(&mut self, interface: BoxedInterface) {
        self.interface = Some(interface);
    }

    /// Load a program into the interface
    ///
    /// The program is loaded but not executed yet. Execution happens on the
    /// worker thread during `start()`.
    ///
    /// # Errors
    /// Returns an error if no interface is set, program loading fails, or the
    /// interface doesn't support dynamic execution.
    pub fn load_program(
        &mut self,
        program_bytes: &[u8],
        format: ProgramFormat,
    ) -> Result<(), PecosError> {
        debug!("Loading program into QisEngine");

        // Store program for potential re-execution
        self.program_bytes = Some(program_bytes.to_vec());
        self.program_format = Some(format);

        // Load into the interface
        if let Some(ref mut interface) = self.interface {
            interface
                .load_program(program_bytes, format)
                .map_err(crate::interface_impl::interface_error_to_pecos)?;

            if !interface.supports_dynamic() {
                return Err(PecosError::Generic(
                    "QisEngine requires a dynamic-capable interface (e.g., QisHeliosInterface)"
                        .to_string(),
                ));
            }

            debug!("Program loaded, deferring execution to start()");
            Ok(())
        } else {
            Err(PecosError::Generic("No interface set".to_string()))
        }
    }

    fn reset_qubit_slots(&mut self) {
        self.active_qubit_slots.clear();
        self.free_qubit_slots.clear();
        self.seen_program_qubits.clear();
        self.qubit_prep_states.clear();
        self.num_physical_slots = 0;
    }

    fn allocate_qubit_slot(&mut self, program_id: usize) -> Result<usize, PecosError> {
        if self.active_qubit_slots.contains_key(&program_id) {
            return Err(PecosError::Generic(format!(
                "QIS program qubit {program_id} is already allocated; release live handles before reallocation"
            )));
        }

        let slot = if let Some(slot) = self.free_qubit_slots.pop_first() {
            slot
        } else {
            if let Some(limit) = self.num_qubits_hint
                && self.num_physical_slots >= limit
            {
                return Err(PecosError::Generic(format!(
                    "QIS program requires more than the configured {limit} physical qubit slots while allocating program qubit {program_id}"
                )));
            }
            self.num_physical_slots
        };
        self.num_physical_slots = self.num_physical_slots.max(slot + 1);
        self.active_qubit_slots.insert(program_id, slot);
        self.seen_program_qubits.insert(program_id);
        Ok(slot)
    }

    fn release_qubit_slot(&mut self, program_id: usize) {
        if let Some(slot) = self.active_qubit_slots.remove(&program_id) {
            self.free_qubit_slots.insert(slot);
        }
    }

    fn mapped_qubit(&mut self, program_id: usize, op: &QuantumOp) -> Result<usize, PecosError> {
        if let Some(&slot) = self.active_qubit_slots.get(&program_id) {
            return Ok(slot);
        }

        if self.seen_program_qubits.contains(&program_id) {
            return Err(PecosError::Generic(format!(
                "QIS runtime emitted {op:?} for program qubit {program_id}, but that handle is not currently active; it was likely released without a matching re-allocation"
            )));
        }

        self.allocate_qubit_slot(program_id)
    }

    /// Convert dynamic QIS operations into a `ByteMessage` for the quantum engine.
    ///
    /// Guppy and the LLVM/QIS path allocate fresh qubit handles over time, even when
    /// the source program is reusing ancillas logically. The quantum simulators used
    /// by `sim()` operate on a fixed physical qubit pool, so we must honor
    /// `AllocateQubit`/`ReleaseQubit` and remap program handles back onto reusable
    /// physical slots before sending the quantum ops downstream.
    fn push_gate_metadata(
        gate_metadata: &mut Vec<TraceMetadata>,
        pending_metadata: &mut TraceMetadata,
    ) {
        gate_metadata.push(std::mem::take(pending_metadata));
    }

    fn operations_to_lowered_commands(
        &mut self,
        ops: &[Operation],
    ) -> Result<LoweredCommandBatch, PecosError> {
        let mut builder = std::mem::take(&mut self.command_builder);
        builder.reset();
        self.measurement_mapping.clear();
        let mut gate_metadata = Vec::new();
        let mut pending_metadata = TraceMetadata::new();

        let result = (|| -> Result<(), PecosError> {
            for op in ops {
                match op {
                    Operation::TraceMetadata { metadata, .. } => {
                        pending_metadata.extend(metadata.clone());
                    }
                    Operation::AllocateQubit { id } => {
                        self.allocate_qubit_slot(*id)?;
                    }
                    Operation::ReleaseQubit { id } => {
                        self.release_qubit_slot(*id);
                    }
                    Operation::AllocateResult { .. }
                    | Operation::RecordOutput { .. }
                    | Operation::Barrier => {}
                    Operation::Quantum(qop) => match qop {
                        QuantumOp::H(qubit) => {
                            builder.h(&[self.mapped_qubit(*qubit, qop)?]);
                            Self::push_gate_metadata(&mut gate_metadata, &mut pending_metadata);
                        }
                        QuantumOp::X(qubit) => {
                            builder.x(&[self.mapped_qubit(*qubit, qop)?]);
                            Self::push_gate_metadata(&mut gate_metadata, &mut pending_metadata);
                        }
                        QuantumOp::Y(qubit) => {
                            builder.y(&[self.mapped_qubit(*qubit, qop)?]);
                            Self::push_gate_metadata(&mut gate_metadata, &mut pending_metadata);
                        }
                        QuantumOp::Z(qubit) => {
                            builder.z(&[self.mapped_qubit(*qubit, qop)?]);
                            Self::push_gate_metadata(&mut gate_metadata, &mut pending_metadata);
                        }
                        QuantumOp::S(qubit) => {
                            builder.sz(&[self.mapped_qubit(*qubit, qop)?]);
                            Self::push_gate_metadata(&mut gate_metadata, &mut pending_metadata);
                        }
                        QuantumOp::Sdg(qubit) => {
                            builder.szdg(&[self.mapped_qubit(*qubit, qop)?]);
                            Self::push_gate_metadata(&mut gate_metadata, &mut pending_metadata);
                        }
                        QuantumOp::T(qubit) => {
                            builder.t(&[self.mapped_qubit(*qubit, qop)?]);
                            Self::push_gate_metadata(&mut gate_metadata, &mut pending_metadata);
                        }
                        QuantumOp::Tdg(qubit) => {
                            builder.tdg(&[self.mapped_qubit(*qubit, qop)?]);
                            Self::push_gate_metadata(&mut gate_metadata, &mut pending_metadata);
                        }
                        QuantumOp::RX(angle, qubit) => {
                            builder.rx(
                                Angle64::from_radians(*angle),
                                &[self.mapped_qubit(*qubit, qop)?],
                            );
                            Self::push_gate_metadata(&mut gate_metadata, &mut pending_metadata);
                        }
                        QuantumOp::RY(angle, qubit) => {
                            builder.ry(
                                Angle64::from_radians(*angle),
                                &[self.mapped_qubit(*qubit, qop)?],
                            );
                            Self::push_gate_metadata(&mut gate_metadata, &mut pending_metadata);
                        }
                        QuantumOp::RZ(angle, qubit) => {
                            builder.rz(
                                Angle64::from_radians(*angle),
                                &[self.mapped_qubit(*qubit, qop)?],
                            );
                            Self::push_gate_metadata(&mut gate_metadata, &mut pending_metadata);
                        }
                        QuantumOp::RXY(theta, phi, qubit) => {
                            builder.rxy1q(
                                Angle64::from_radians(*theta),
                                Angle64::from_radians(*phi),
                                &[self.mapped_qubit(*qubit, qop)?],
                            );
                            Self::push_gate_metadata(&mut gate_metadata, &mut pending_metadata);
                        }
                        QuantumOp::Idle(duration, qubit) => {
                            builder.idle(*duration, &[self.mapped_qubit(*qubit, qop)?]);
                            Self::push_gate_metadata(&mut gate_metadata, &mut pending_metadata);
                        }
                        QuantumOp::CX(control, target) => {
                            builder.cx(&[(
                                self.mapped_qubit(*control, qop)?,
                                self.mapped_qubit(*target, qop)?,
                            )]);
                            Self::push_gate_metadata(&mut gate_metadata, &mut pending_metadata);
                        }
                        QuantumOp::Measure(qubit, result_id) => {
                            self.measurement_mapping.push(*result_id);
                            builder.mz(&[self.mapped_qubit(*qubit, qop)?]);
                            Self::push_gate_metadata(&mut gate_metadata, &mut pending_metadata);
                        }
                        QuantumOp::MeasureLeaked(qubit, result_id) => {
                            self.measurement_mapping.push(*result_id);
                            builder.measure_leakages(&[self.mapped_qubit(*qubit, qop)?]);
                            Self::push_gate_metadata(&mut gate_metadata, &mut pending_metadata);
                        }
                        QuantumOp::ZZ(qubit1, qubit2) => {
                            builder.szz(&[(
                                self.mapped_qubit(*qubit1, qop)?,
                                self.mapped_qubit(*qubit2, qop)?,
                            )]);
                            Self::push_gate_metadata(&mut gate_metadata, &mut pending_metadata);
                        }
                        QuantumOp::RXYXY2Q(theta, phi, qubit1, qubit2) => {
                            builder.rxyxy2q(
                                Angle64::from_radians(*theta),
                                Angle64::from_radians(*phi),
                                &[(
                                    self.mapped_qubit(*qubit1, qop)?,
                                    self.mapped_qubit(*qubit2, qop)?,
                                )],
                            );
                            Self::push_gate_metadata(&mut gate_metadata, &mut pending_metadata);
                        }
                        QuantumOp::RZZ(angle, qubit1, qubit2) => {
                            builder.rzz(
                                Angle64::from_radians(*angle),
                                &[(
                                    self.mapped_qubit(*qubit1, qop)?,
                                    self.mapped_qubit(*qubit2, qop)?,
                                )],
                            );
                            Self::push_gate_metadata(&mut gate_metadata, &mut pending_metadata);
                        }
                        QuantumOp::CRZ(angle, control, target) => {
                            let control = self.mapped_qubit(*control, qop)?;
                            let target = self.mapped_qubit(*target, qop)?;
                            let gates = pecos_core::controlled_rotations::lower_crz(
                                *angle,
                                control.into(),
                                target.into(),
                            );
                            builder.add_gate_commands(&gates);
                            let metadata = std::mem::take(&mut pending_metadata);
                            gate_metadata.extend(std::iter::repeat_n(metadata, gates.len()));
                        }
                        QuantumOp::Reset(qubit) => {
                            builder.pz(&[self.mapped_qubit(*qubit, qop)?]);
                            Self::push_gate_metadata(&mut gate_metadata, &mut pending_metadata);
                        }
                        _ => {
                            return Err(PecosError::Generic(format!(
                                "Unsupported operation: {qop:?}"
                            )));
                        }
                    },
                }
            }

            if !pending_metadata.is_empty() {
                warn!(
                    "QIS operation trace metadata was not followed by a lowerable quantum operation"
                );
            }
            Ok(())
        })();

        let message = result.map(|()| LoweredCommandBatch {
            commands: builder.build(),
            is_empty: gate_metadata.is_empty(),
            gate_metadata,
            measurement_credits_consumed: 0,
        });
        self.command_builder = builder;
        message
    }

    /// Defer each lifetime's prep until its first quantum operation.
    fn normalize_qubit_preps(&mut self, ops: &[Operation]) -> Vec<Operation> {
        let mut normalized = Vec::with_capacity(ops.len());
        let mut pending_trace_metadata = Vec::new();
        for op in ops {
            match op {
                Operation::AllocateQubit { id } => {
                    self.qubit_prep_states.insert(*id, QubitPrepState::Pending);
                }
                Operation::ReleaseQubit { id } => {
                    if let Some(state) = self.qubit_prep_states.get_mut(id)
                        && *state != QubitPrepState::Released
                    {
                        *state = QubitPrepState::Released;
                    }
                }
                Operation::TraceMetadata { .. } => {
                    pending_trace_metadata.push(op.clone());
                    continue;
                }
                Operation::Quantum(qop) => {
                    let mut qubits = Vec::new();
                    for_each_quantum_qubit(qop, |qubit| {
                        qubits.push(qubit);
                        let state = self
                            .qubit_prep_states
                            .entry(qubit)
                            .or_insert(QubitPrepState::Pending);
                        if *state == QubitPrepState::Pending {
                            if !matches!(qop, QuantumOp::Reset(_)) {
                                normalized.push(QuantumOp::Reset(qubit).into());
                            }
                            *state = QubitPrepState::Prepared;
                        }
                    });
                    // Hold even nonadjacent scoped metadata until its source op.
                    // Emitting it after the preps keeps them free of source labels.
                    pending_trace_metadata.retain(|metadata| {
                        if let Operation::TraceMetadata { qubit, .. } = metadata
                            && qubit.is_none_or(|q| qubits.contains(&q))
                        {
                            normalized.push(metadata.clone());
                            return false;
                        }
                        true
                    });
                }
                Operation::AllocateResult { .. }
                | Operation::RecordOutput { .. }
                | Operation::Barrier => {}
            }
            normalized.push(op.clone());
        }
        // Leave dangling metadata visible to this chunk's downstream validation.
        normalized.extend(pending_trace_metadata);
        normalized
    }

    /// Convert freshly collected dynamic operations into a `ByteMessage`.
    ///
    /// Selene runtime plugins can opt in to lowering so their scheduler sees
    /// the same operation stream that Selene would receive. Other runtimes keep
    /// using PECOS's direct QIS lowering path.
    fn lower_operations_to_commands(
        &mut self,
        ops: &[Operation],
    ) -> Result<LoweredCommandBatch, PecosError> {
        self.register_imported_measurements(ops);
        let normalized = self.normalize_qubit_preps(ops);
        let ops = normalized.as_slice();
        let mut lowered = if self.scheduled_transport.enabled() {
            let batches = self
                .runtime
                .lower_scheduled_operations(ops)
                .map_err(|e| PecosError::Generic(format!("scheduled extraction failed: {e}")))?;
            let shot = u64::try_from(self.trace_shot_index)
                .map_err(|_| PecosError::Generic("shot index exceeds u64".into()))?;
            let is_empty = batches.is_empty();
            let (commands, ids) =
                crate::scheduled_transport::encode_mode(batches, shot, self.scheduled_transport)?;
            self.measurement_mapping = ids;
            LoweredCommandBatch {
                commands,
                gate_metadata: Vec::new(),
                is_empty,
                measurement_credits_consumed: 0,
            }
        } else if self.runtime.supports_operation_lowering() {
            let lowered_ops = self
                .runtime
                .lower_operations_with_metadata(ops)
                .map_err(|e| PecosError::Generic(format!("Runtime lowering error: {e}")))?;
            self.quantum_ops_to_lowered_commands(lowered_ops)?
        } else {
            self.operations_to_lowered_commands(ops)?
        };
        lowered.measurement_credits_consumed = self.register_emitted_measurements()?;
        Ok(lowered)
    }

    /// Convert already-materialized quantum ops into a `ByteMessage`.
    ///
    /// This path is used by runtimes that already present qubit ids in the fixed
    /// simulator space, so no allocate/release remapping is needed.
    fn quantum_ops_to_lowered_commands(
        &mut self,
        ops: Vec<LoweredQuantumOp>,
    ) -> Result<LoweredCommandBatch, PecosError> {
        let mut builder = std::mem::take(&mut self.command_builder);
        builder.reset();
        self.measurement_mapping.clear();
        let mut gate_metadata = Vec::new();

        let result = (|| -> Result<(), PecosError> {
            for LoweredQuantumOp { op, metadata } in ops {
                match op {
                    QuantumOp::H(qubit) => {
                        builder.h(&[qubit]);
                        gate_metadata.push(metadata);
                    }
                    QuantumOp::X(qubit) => {
                        builder.x(&[qubit]);
                        gate_metadata.push(metadata);
                    }
                    QuantumOp::Y(qubit) => {
                        builder.y(&[qubit]);
                        gate_metadata.push(metadata);
                    }
                    QuantumOp::Z(qubit) => {
                        builder.z(&[qubit]);
                        gate_metadata.push(metadata);
                    }
                    QuantumOp::S(qubit) => {
                        builder.sz(&[qubit]);
                        gate_metadata.push(metadata);
                    }
                    QuantumOp::Sdg(qubit) => {
                        builder.szdg(&[qubit]);
                        gate_metadata.push(metadata);
                    }
                    QuantumOp::T(qubit) => {
                        builder.t(&[qubit]);
                        gate_metadata.push(metadata);
                    }
                    QuantumOp::Tdg(qubit) => {
                        builder.tdg(&[qubit]);
                        gate_metadata.push(metadata);
                    }
                    QuantumOp::RX(angle, qubit) => {
                        builder.rx(Angle64::from_radians(angle), &[qubit]);
                        gate_metadata.push(metadata);
                    }
                    QuantumOp::RY(angle, qubit) => {
                        builder.ry(Angle64::from_radians(angle), &[qubit]);
                        gate_metadata.push(metadata);
                    }
                    QuantumOp::RZ(angle, qubit) => {
                        builder.rz(Angle64::from_radians(angle), &[qubit]);
                        gate_metadata.push(metadata);
                    }
                    QuantumOp::RXY(theta, phi, qubit) => {
                        builder.rxy1q(
                            Angle64::from_radians(theta),
                            Angle64::from_radians(phi),
                            &[qubit],
                        );
                        gate_metadata.push(metadata);
                    }
                    QuantumOp::Idle(duration, qubit) => {
                        builder.idle(duration, &[qubit]);
                        gate_metadata.push(metadata);
                    }
                    QuantumOp::CX(control, target) => {
                        builder.cx(&[(control, target)]);
                        gate_metadata.push(metadata);
                    }
                    QuantumOp::Measure(qubit, result_id) => {
                        self.measurement_mapping.push(result_id);
                        builder.mz(&[qubit]);
                        gate_metadata.push(metadata);
                    }
                    QuantumOp::MeasureLeaked(qubit, result_id) => {
                        self.measurement_mapping.push(result_id);
                        builder.measure_leakages(&[qubit]);
                        gate_metadata.push(metadata);
                    }
                    QuantumOp::ZZ(qubit1, qubit2) => {
                        builder.szz(&[(qubit1, qubit2)]);
                        gate_metadata.push(metadata);
                    }
                    QuantumOp::RXYXY2Q(theta, phi, qubit1, qubit2) => {
                        builder.rxyxy2q(
                            Angle64::from_radians(theta),
                            Angle64::from_radians(phi),
                            &[(qubit1, qubit2)],
                        );
                        gate_metadata.push(metadata);
                    }
                    QuantumOp::RZZ(angle, qubit1, qubit2) => {
                        builder.rzz(Angle64::from_radians(angle), &[(qubit1, qubit2)]);
                        gate_metadata.push(metadata);
                    }
                    QuantumOp::CRZ(angle, control, target) => {
                        let gates = pecos_core::controlled_rotations::lower_crz(
                            angle,
                            control.into(),
                            target.into(),
                        );
                        builder.add_gate_commands(&gates);
                        gate_metadata.extend(std::iter::repeat_n(metadata, gates.len()));
                    }
                    QuantumOp::Reset(qubit) => {
                        builder.pz(&[qubit]);
                        gate_metadata.push(metadata);
                    }
                    _ => {
                        return Err(PecosError::Generic(format!(
                            "Unsupported operation: {op:?}"
                        )));
                    }
                }
            }

            Ok(())
        })();

        let message = result.map(|()| LoweredCommandBatch {
            commands: builder.build(),
            is_empty: gate_metadata.is_empty(),
            gate_metadata,
            measurement_credits_consumed: 0,
        });
        self.command_builder = builder;
        message
    }
}

impl Clone for QisEngine {
    fn clone(&self) -> Self {
        // Recreate the interface from stored program bytes
        let interface = if let (Some(builder), Some(program_bytes)) =
            (&self.interface_builder, &self.program_bytes)
        {
            // Recreate the interface for this clone
            let program_str = String::from_utf8_lossy(program_bytes).into_owned();
            let qis_prog = pecos_programs::Qis::from_string(program_str);
            match builder.create_dynamic_interface_from_qis(qis_prog) {
                Ok(interface) => {
                    debug!("QisEngine::clone() - recreated interface");
                    Some(interface)
                }
                Err(e) => {
                    log::error!("QisEngine::clone() - failed to recreate interface: {e}");
                    None
                }
            }
        } else {
            debug!("QisEngine::clone() - missing builder or program bytes");
            None
        };

        Self {
            interface,
            runtime: dyn_clone::clone_box(&*self.runtime),
            current_operations: self.current_operations.clone(),
            num_physical_slots: self.num_physical_slots,
            num_qubits_hint: self.num_qubits_hint,
            active_qubit_slots: self.active_qubit_slots.clone(),
            free_qubit_slots: self.free_qubit_slots.clone(),
            seen_program_qubits: self.seen_program_qubits.clone(),
            qubit_prep_states: self.qubit_prep_states.clone(),
            started: false,                       // Reset started flag for the clone
            measurement_mapping: Vec::new(),      // Clear for new shot
            measurement_results: BTreeMap::new(), // Clear for new shot
            pending_measurements: BTreeMap::new(),
            reset_failure: self.reset_failure.clone(), // Keep a failed reset latched
            shot_lifecycle: ShotLifecycle::Idle,
            scheduled_drain_round: 0,
            scheduled_drain_feedback: false,
            rng: self.rng.clone(),
            current_shot_seed: None,         // Will be set on next start()
            dynamic_state: None,             // Can't clone thread state
            pending_dynamic_ops: Vec::new(), // Clear for new shot
            simulated_op_count: 0,           // Reset for new shot
            program_bytes: self.program_bytes.clone(),
            program_format: self.program_format,
            interface_builder: self
                .interface_builder
                .as_ref()
                .map(|b| dyn_clone::clone_box(&**b)),
            // Create a new persistent worker for this clone (can't share threads across clones)
            persistent_worker: None,
            #[cfg(test)]
            worker_counts: Arc::clone(&self.worker_counts),
            operation_trace_dir: self.operation_trace_dir.clone(),
            operation_trace_collector: self.operation_trace_collector.clone(),
            trace_engine_id: TRACE_ENGINE_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            trace_shot_index: 0,
            trace_chunk_index: 0,
            command_builder: ByteMessageBuilder::new(),
            scheduled_transport: self.scheduled_transport,
        }
    }
}

// Helper methods for dynamic execution
impl QisEngine {
    fn begin_trace_shot(&mut self) {
        self.trace_shot_index = self
            .trace_shot_index
            .checked_add(1)
            .expect("trace_shot_index overflow: too many shots for a single trace engine");
        self.trace_chunk_index = 0;
    }

    fn lowered_quantum_ops_trace(
        commands: &ByteMessage,
        measurement_mapping: &[usize],
        gate_metadata: &[TraceMetadata],
    ) -> Result<Vec<LoweredQuantumGateTrace>, String> {
        commands
            .quantum_ops()
            .map_err(|err| format!("failed to parse lowered quantum ops: {err}"))
            .and_then(|gates| {
                let mut measurement_cursor = 0usize;
                let mut traces = Vec::with_capacity(gates.len());
                if gate_metadata.len() != gates.len() {
                    return Err(format!(
                        "lowered operation trace has {} metadata record(s) for {} gate(s)",
                        gate_metadata.len(),
                        gates.len()
                    ));
                }
                for (gate_index, gate) in gates.iter().enumerate() {
                    let gate_type = gate.gate_type.to_string();
                    let qubits = gate
                        .qubits
                        .iter()
                        .map(|q| usize::from(*q))
                        .collect::<Vec<_>>();
                    let measurement_result_ids = if matches!(gate_type.as_str(), "MZ" | "MeasureLeaked") {
                        let end = measurement_cursor + qubits.len();
                        if end > measurement_mapping.len() {
                            return Err(
                                "lowered operation trace has more measured qubits than result-id mappings"
                                    .to_string(),
                            );
                        }
                        let ids = measurement_mapping[measurement_cursor..end].to_vec();
                        measurement_cursor = end;
                        ids
                    } else {
                        Vec::new()
                    };
                    traces.push(LoweredQuantumGateTrace {
                        gate_type,
                        angles: gate
                            .angles
                            .iter()
                            .map(Angle64::to_radians)
                            .collect::<Vec<_>>(),
                        params: gate.params.iter().copied().collect::<Vec<_>>(),
                        qubits,
                        measurement_result_ids,
                        metadata: gate_metadata.get(gate_index).cloned().unwrap_or_default(),
                    });
                }
                if measurement_cursor != measurement_mapping.len() {
                    return Err(format!(
                        "lowered operation trace consumed {} measurement mapping(s), but {} were present",
                        measurement_cursor,
                        measurement_mapping.len()
                    ));
                }
                Ok(traces)
            })
    }

    fn trace_operations_chunk(
        &mut self,
        stage: &str,
        ops: &[Operation],
        waiting_for_result_id: Option<u64>,
        lowered_quantum_ops: Option<&LoweredCommandBatch>,
    ) {
        if self.operation_trace_dir.is_none() && self.operation_trace_collector.is_none() {
            return;
        }

        let (lowered_trace, lowered_quantum_ops_complete) = lowered_quantum_ops.map_or_else(
            || (Vec::new(), false),
            |lowered| match Self::lowered_quantum_ops_trace(
                &lowered.commands,
                &self.measurement_mapping,
                &lowered.gate_metadata,
            ) {
                Ok(trace) => (trace, true),
                Err(err) => {
                    warn!("Failed to certify lowered operation trace: {err}");
                    (Vec::new(), false)
                }
            },
        );
        let file_name = format!(
            "engine_{:04}_shot_{:06}_chunk_{:04}_{}.json",
            self.trace_engine_id, self.trace_shot_index, self.trace_chunk_index, stage
        );
        let chunk_index = self.trace_chunk_index;
        self.trace_chunk_index = self
            .trace_chunk_index
            .checked_add(1)
            .expect("trace_chunk_index overflow: too many chunks for a single trace shot");
        let chunk = OperationTraceChunk {
            format: "pecos_qis_operation_trace_v1",
            engine_trace_id: self.trace_engine_id,
            shot_index: self.trace_shot_index,
            chunk_index,
            stage: stage.to_string(),
            waiting_for_result_id,
            current_shot_seed: self.current_shot_seed,
            simulated_op_count: self.simulated_op_count,
            num_operations: ops.len(),
            operations: ops.to_vec(),
            lowered_quantum_ops: lowered_trace,
            lowered_quantum_ops_complete,
            named_result_traces: Vec::new(),
            measurement_results: if stage == "trace_complete" {
                self.measurement_results.clone()
            } else {
                BTreeMap::new()
            },
        };

        if let Some(ref collector) = self.operation_trace_collector {
            match collector.lock() {
                Ok(mut guard) => guard.push(chunk.clone()),
                Err(err) => warn!("Failed to store operation trace chunk in memory: {err}"),
            }
        }

        if let Some(ref trace_dir) = self.operation_trace_dir {
            if let Err(err) = fs::create_dir_all(trace_dir) {
                warn!(
                    "Failed to create operation trace directory {}: {err}",
                    trace_dir.display()
                );
                return;
            }

            let trace_path = trace_dir.join(file_name);
            let serialized = match serde_json::to_string_pretty(&chunk) {
                Ok(serialized) => serialized,
                Err(err) => {
                    warn!(
                        "Failed to serialize operation trace chunk for {}: {err}",
                        trace_path.display()
                    );
                    return;
                }
            };

            if let Err(err) = fs::write(&trace_path, serialized) {
                warn!(
                    "Failed to write operation trace chunk {}: {err}",
                    trace_path.display()
                );
            }
        }
    }

    fn trace_named_result_traces_chunk(&mut self, named_result_traces: &[NamedResultTrace]) {
        if named_result_traces.is_empty()
            || (self.operation_trace_dir.is_none() && self.operation_trace_collector.is_none())
        {
            return;
        }

        let stage = "named_results";
        let file_name = format!(
            "engine_{:04}_shot_{:06}_chunk_{:04}_{}.json",
            self.trace_engine_id, self.trace_shot_index, self.trace_chunk_index, stage
        );
        let chunk_index = self.trace_chunk_index;
        self.trace_chunk_index = self
            .trace_chunk_index
            .checked_add(1)
            .expect("trace_chunk_index overflow: too many chunks for a single trace shot");
        let chunk = OperationTraceChunk {
            format: "pecos_qis_operation_trace_v1",
            engine_trace_id: self.trace_engine_id,
            shot_index: self.trace_shot_index,
            chunk_index,
            stage: stage.to_string(),
            waiting_for_result_id: None,
            current_shot_seed: self.current_shot_seed,
            simulated_op_count: self.simulated_op_count,
            num_operations: 0,
            operations: Vec::new(),
            lowered_quantum_ops: Vec::new(),
            lowered_quantum_ops_complete: true,
            named_result_traces: named_result_traces.to_vec(),
            measurement_results: BTreeMap::new(),
        };

        if let Some(ref collector) = self.operation_trace_collector {
            match collector.lock() {
                Ok(mut guard) => guard.push(chunk.clone()),
                Err(err) => warn!("Failed to store named result trace chunk in memory: {err}"),
            }
        }

        if let Some(ref trace_dir) = self.operation_trace_dir {
            if let Err(err) = fs::create_dir_all(trace_dir) {
                warn!(
                    "Failed to create operation trace directory {}: {err}",
                    trace_dir.display()
                );
                return;
            }

            let trace_path = trace_dir.join(file_name);
            let serialized = match serde_json::to_string_pretty(&chunk) {
                Ok(serialized) => serialized,
                Err(err) => {
                    warn!(
                        "Failed to serialize named result trace chunk for {}: {err}",
                        trace_path.display()
                    );
                    return;
                }
            };

            if let Err(err) = fs::write(&trace_path, serialized) {
                warn!(
                    "Failed to write named result trace chunk {}: {err}",
                    trace_path.display()
                );
            }
        }
    }

    fn trace_complete_chunk(&mut self) {
        if self.operation_trace_dir.is_none() && self.operation_trace_collector.is_none() {
            return;
        }
        self.trace_operations_chunk(
            "trace_complete",
            &[],
            None,
            Some(&LoweredCommandBatch {
                commands: ByteMessage::builder().build(),
                gate_metadata: Vec::new(),
                is_empty: true,
                measurement_credits_consumed: 0,
            }),
        );
    }

    fn trace_named_result_traces_from_dynamic_handle(&mut self) {
        let named_result_traces = if let Some(state) = &self.dynamic_state
            && let Some(handle) = &state.sync_handle
        {
            match handle.get_named_result_traces() {
                Ok(named_result_traces) => named_result_traces,
                Err(e) => {
                    debug!("QisEngine: Failed to get named result traces: {e}");
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };
        self.trace_named_result_traces_chunk(&named_result_traces);
    }

    /// Start the LLVM program execution in a worker thread
    ///
    /// Uses a persistent worker thread to avoid TLS allocation issues from
    /// spawning a new thread per shot.
    fn start_dynamic_worker(&mut self) -> Result<(), PecosError> {
        debug!("Starting dynamic execution");

        // Get reference to interface for setup
        let interface = self.interface.as_mut().ok_or_else(|| {
            PecosError::Generic("No interface available for dynamic execution".to_string())
        })?;

        // Verify interface supports dynamic execution
        if !interface.supports_dynamic() {
            return Err(PecosError::Generic(
                "Interface does not support dynamic execution".to_string(),
            ));
        }

        // Enable dynamic mode on the interface
        interface
            .enable_dynamic_mode()
            .map_err(|e| PecosError::Generic(format!("Failed to enable dynamic mode: {e}")))?;

        // Get the sync handle BEFORE moving the interface
        // This handle uses the same singleton library as the worker, ensuring TLS consistency
        let sync_handle = interface.get_sync_handle();
        debug!("Got sync handle for main thread: {}", sync_handle.is_some());

        // Take the interface for the worker thread
        let interface = self.interface.take().ok_or_else(|| {
            PecosError::Generic("No interface available for dynamic execution".to_string())
        })?;

        // Create persistent worker if it doesn't exist
        if self.persistent_worker.is_none() {
            debug!("Creating new persistent dynamic worker thread");
            self.persistent_worker = Some(PersistentDynamicWorker::new(
                #[cfg(test)]
                Arc::clone(&self.worker_counts),
            ));
        }

        // Send work to persistent worker
        self.persistent_worker
            .as_ref()
            .expect("persistent worker was just created")
            .execute(interface)?;

        // Initialize dynamic state
        self.dynamic_state = Some(DynamicExecutionState {
            sync_handle,
            execution_complete: false,
            terminal_error: None,
            finalized: false,
            terminal_lowering_flushed: false,
        });

        Ok(())
    }

    /// Wait for the worker to need a result
    ///
    /// Returns `Some(result_id)` if worker needs a result, None if complete or timeout
    fn wait_for_result_needed(&mut self, timeout_ms: u64) -> Option<u64> {
        let state = self.dynamic_state.as_ref()?;
        let handle = state.sync_handle.as_ref()?;
        handle.wait_for_need_result(timeout_ms)
    }

    /// Set a measurement result for the running program
    fn set_dynamic_result(&mut self, result_id: u64, value: u32) -> Result<(), PecosError> {
        let state = self
            .dynamic_state
            .as_ref()
            .ok_or_else(|| PecosError::Generic("No dynamic execution in progress".to_string()))?;
        let handle = state
            .sync_handle
            .as_ref()
            .ok_or_else(|| PecosError::Generic("No sync handle available".to_string()))?;

        handle
            .set_measurement_outcome(result_id, u64::from(value))
            .map_err(|e| PecosError::Generic(format!("Failed to set measurement result: {e}")))?;
        debug!("Set dynamic result: {result_id} = {value}");
        Ok(())
    }

    /// Signal that the measurement result is ready
    fn signal_dynamic_result_ready(&mut self) -> Result<(), PecosError> {
        let state = self
            .dynamic_state
            .as_ref()
            .ok_or_else(|| PecosError::Generic("No dynamic execution in progress".to_string()))?;
        let handle = state
            .sync_handle
            .as_ref()
            .ok_or_else(|| PecosError::Generic("No sync handle available".to_string()))?;

        handle
            .signal_result_ready()
            .map_err(|e| PecosError::Generic(format!("Failed to signal result ready: {e}")))?;
        debug!("Signaled result ready");
        Ok(())
    }

    /// Get pending operations from the dynamic execution.
    ///
    /// The worker exports only newly generated operations before each wait, so
    /// this handoff stays proportional to fresh work instead of full history.
    fn get_dynamic_operations(&mut self) -> Result<Vec<Operation>, PecosError> {
        let imported = self
            .dynamic_state
            .as_ref()
            .and_then(|state| state.sync_handle.as_ref())
            .ok_or_else(|| InterfaceError::ExecutionError("No dynamic sync handle".into()))
            .and_then(|handle| handle.get_pending_operations());
        imported.map_err(|error| {
            self.latch_terminal_error(format!("failed to import pending operations: {error}"))
        })
    }

    /// Advance a requested read using exported work, cached outcomes, or the
    /// runtime's existing drain. A request without any producer must fail.
    fn process_result_request(
        &mut self,
        result_id: u64,
        stage: &str,
    ) -> Result<Option<ByteMessage>, PecosError> {
        // Import before consulting the cache: new measurements own their slots.
        let ops = self.get_dynamic_operations()?;
        if !ops.is_empty() {
            self.simulated_op_count += ops.len();
            let lowered = self.lower_operations_terminal(&ops)?;
            self.trace_operations_chunk(stage, &ops, Some(result_id), Some(&lowered));
            if !lowered.is_empty {
                return Ok(Some(lowered.commands));
            }
        }
        let result_key = usize::try_from(result_id)
            .map_err(|_| self.latch_terminal_error("result ID exceeds usize".into()))?;
        if let Some(&value) = self.measurement_results.get(&result_key) {
            self.set_dynamic_result(result_id, value)?;
            self.signal_dynamic_result_ready()?;
            return Ok(None);
        }
        // While the worker is blocked, imported credits are finite. Each
        // progressing drain consumes at least one; feedback may release the
        // requested measurement, but padding cannot keep a request alive.
        if let Some((commands, consumed)) = self.drain_commands(false)?
            && consumed > 0
        {
            return Ok(Some(commands));
        }
        Err(self.latch_terminal_error(format!(
            "runtime released no measurement for worker-requested result {result_id}"
        )))
    }

    /// Import all producers before lowering, including those a runtime defers.
    fn register_imported_measurements(&mut self, operations: &[Operation]) {
        for op in operations {
            if let Operation::Quantum(
                QuantumOp::Measure(_, result_id) | QuantumOp::MeasureLeaked(_, result_id),
            ) = op
            {
                let pending = self.pending_measurements.entry(*result_id).or_default();
                pending.outstanding += 1;
                pending.unemitted += 1;
                self.measurement_results.remove(result_id);
            }
        }
    }

    /// Every emission must consume an imported credit. The returned count
    /// bounds the number of progressing drains while the worker is blocked.
    fn register_emitted_measurements(&mut self) -> Result<usize, PecosError> {
        let mut consumed = 0;
        for index in 0..self.measurement_mapping.len() {
            let result_id = self.measurement_mapping[index];
            let Some(pending) = self
                .pending_measurements
                .get_mut(&result_id)
                .filter(|pending| pending.unemitted > 0)
            else {
                return Err(self.latch_terminal_error(format!(
                    "runtime emitted a measurement for result {result_id} that was never imported"
                )));
            };
            pending.unemitted -= 1;
            consumed += 1;
            self.measurement_results.remove(&result_id);
        }
        Ok(consumed)
    }

    /// Check if dynamic execution is complete
    fn check_worker_complete(&mut self) -> bool {
        // First check if already complete
        if let Some(ref state) = self.dynamic_state
            && state.execution_complete
        {
            return true;
        }

        // Check if persistent worker has a result ready
        let result: Option<WorkerResult> = self
            .persistent_worker
            .as_ref()
            .and_then(PersistentDynamicWorker::try_recv_result);

        // Process result if we got one
        if let Some(result) = result {
            match result {
                Ok((collector, interface)) => {
                    let operations = collector.operations;
                    let remaining_ops = operations.len();
                    debug!(
                        "Worker completed with {} remaining operations after {} already simulated",
                        remaining_ops, self.simulated_op_count
                    );
                    self.pending_dynamic_ops = operations;
                    self.interface = Some(interface);
                    if let Some(ref mut state) = self.dynamic_state {
                        state.execution_complete = true;
                    }
                    return true;
                }
                Err((e, interface)) => {
                    log::error!("Worker failed: {e}");
                    if let Some(interface) = interface {
                        // Keep the interface so the engine survives the
                        // failed shot; only this shot is poisoned.
                        self.interface = Some(interface);
                    }
                    if let Some(ref mut state) = self.dynamic_state {
                        state.execution_complete = true;
                    }
                    self.latch_terminal_error(format!("dynamic QIS worker failed: {e}"));
                    return true;
                }
            }
        }

        false
    }

    /// Sticky error if this shot already failed terminally (worker failure,
    /// failed drain verification): completion paths call this before emitting
    /// the terminal trace marker or returning shot results, so a poisoned
    /// shot can never certify a partial trace as complete — including on
    /// retried `continue_processing` calls after the failure was reported.
    fn terminal_failure_error(&self) -> Option<PecosError> {
        self.reset_failure
            .as_ref()
            .or_else(|| {
                self.dynamic_state
                    .as_ref()
                    .and_then(|state| state.terminal_error.as_ref())
            })
            .or(match &self.shot_lifecycle {
                ShotLifecycle::Failed(error) => Some(error),
                _ => None,
            })
            .map(|err| PecosError::Generic(err.clone()))
    }

    /// Latch a terminal failure for this shot and return it as an error.
    fn latch_terminal_error(&mut self, message: String) -> PecosError {
        self.shot_lifecycle = ShotLifecycle::Failed(message.clone());
        if let Some(ref mut state) = self.dynamic_state {
            state.terminal_error = Some(message.clone());
        }
        PecosError::Generic(message)
    }

    fn drain_terminal_commands(&mut self) -> Result<Option<ByteMessage>, PecosError> {
        self.drain_commands(true)
            .map(|drained| drained.map(|(commands, _)| commands))
    }

    /// Use the same drain for a stalled read and the final tail. Only the final
    /// flush is one-shot; reads may need several flushes during a shot.
    fn drain_commands(
        &mut self,
        terminal: bool,
    ) -> Result<Option<(ByteMessage, usize)>, PecosError> {
        if self.scheduled_transport.enabled() {
            return self.drain_scheduled_commands(terminal);
        }
        if !self.runtime.supports_operation_lowering() {
            return Ok(None);
        }
        let Some(state) = self.dynamic_state.as_mut() else {
            return Ok(None);
        };
        if state.finalized || (terminal && state.terminal_lowering_flushed) {
            return Ok(None);
        }
        if terminal {
            state.terminal_lowering_flushed = true;
        }

        // A drain barrier is ordinary lowering: use the same conversion,
        // metadata matching, measurement mapping and sticky error path as source
        // operations. The subsequent certification drain remains only a guard.
        let ops = [Operation::Barrier];
        let lowered = self.lower_operations_terminal(&ops)?;
        if lowered.commands.is_empty()? {
            return Ok(None);
        }
        let stage = if terminal {
            "terminal_flush"
        } else {
            "result_flush"
        };
        self.trace_operations_chunk(stage, &ops, None, Some(&lowered));
        Ok(Some((
            lowered.commands,
            lowered.measurement_credits_consumed,
        )))
    }

    fn drain_scheduled_commands(
        &mut self,
        terminal: bool,
    ) -> Result<Option<(ByteMessage, usize)>, PecosError> {
        if !self.scheduled_transport.enabled()
            || self.dynamic_state.as_ref().is_some_and(|s| s.finalized)
        {
            return Ok(None);
        }
        let permitted =
            !terminal || self.scheduled_drain_round == 0 || self.scheduled_drain_feedback;
        if terminal {
            self.scheduled_drain_round += 1;
            self.scheduled_drain_feedback = false;
        }
        let result = (|| {
            let batches = self
                .runtime
                .drain_pending_scheduled_operations()
                .map_err(|e| PecosError::Generic(e.to_string()))?;
            if batches.is_empty() {
                return Ok(None);
            }
            if !permitted {
                return Err(PecosError::Generic(format!(
                    "scheduled runtime {} returned work without measurement feedback in terminal drain round {}",
                    self.runtime.name(),
                    self.scheduled_drain_round
                )));
            }
            let shot = u64::try_from(self.trace_shot_index)
                .map_err(|_| PecosError::Generic("shot index exceeds u64".into()))?;
            let (commands, ids) =
                crate::scheduled_transport::encode_mode(batches, shot, self.scheduled_transport)?;
            self.measurement_mapping = ids;
            let consumed = self.register_emitted_measurements()?;
            Ok(Some((commands, consumed)))
        })();
        result.map_err(|e: PecosError| {
            self.latch_terminal_error(format!("scheduled drain failed: {e}"))
        })
    }

    /// Refuse to certify a complete trace while the runtime scheduler still
    /// holds operations. Per-batch lowering drains the runtime until it stops
    /// producing, but a scheduling runtime may defer operations past the final
    /// batch; those would otherwise be dropped silently after the terminal
    /// marker. Late operations cannot be simulated at this point, so this is
    /// fail-loud rather than a flush — and the failure is latched sticky,
    /// because the verification itself consumes the late operations (a retry
    /// would otherwise find an innocently empty scheduler and certify).
    fn verify_runtime_drained(&mut self) -> Result<(), PecosError> {
        if self.scheduled_transport.enabled() {
            // Each completion branch just observed an empty scheduled drain.
            return Ok(());
        }
        if !self.runtime.supports_operation_lowering() {
            return Ok(());
        }
        match self.runtime.drain_pending_operations() {
            Ok(late_ops) if late_ops.is_empty() => Ok(()),
            Ok(late_ops) => Err(self.latch_terminal_error(format!(
                "runtime scheduler emitted {} operation(s) after the final lowered batch; \
                 refusing to certify a complete trace",
                late_ops.len()
            ))),
            Err(e) => Err(self.latch_terminal_error(format!("Runtime drain check failed: {e}"))),
        }
    }

    /// Lower operations for the running shot, latching any failure sticky.
    ///
    /// By the time lowering runs, the operations have already been consumed
    /// from the pending queue and a scheduling runtime may have consumed a
    /// prefix of them; retrying after a lowering failure could therefore
    /// only ever certify a trace with those operations deleted. The failure
    /// must poison the shot, not merely propagate once.
    fn lower_operations_terminal(
        &mut self,
        ops: &[Operation],
    ) -> Result<LoweredCommandBatch, PecosError> {
        match self.lower_operations_to_commands(ops) {
            Ok(lowered) => Ok(lowered),
            Err(e) => {
                Err(self
                    .latch_terminal_error(format!("failed to lower operations for this shot: {e}")))
            }
        }
    }

    /// Deliver measurement outcomes to the runtime, latching any failure
    /// sticky: if the scheduler did not receive an outcome it may schedule
    /// from stale state, and no later gate can certify that trace.
    fn provide_measurements_terminal(
        &mut self,
        updates: &[(usize, u32)],
    ) -> Result<(), PecosError> {
        match self.provide_measurement_updates_to_runtime(updates) {
            Ok(()) => Ok(()),
            Err(e) => Err(self.latch_terminal_error(format!(
                "failed to deliver measurement results to the runtime: {e}"
            ))),
        }
    }

    /// Run the shot-completion gates in order: sticky terminal check, drain
    /// verification, the runtime's own `shot_end` finalization hook, then the
    /// named-result traces and terminal trace marker. A runtime that only
    /// detects an invalid final schedule at shot end therefore fails the shot
    /// instead of receiving a certified trace. Failures latch sticky and
    /// success is one-shot.
    fn finalize_shot_for_certification(&mut self) -> Result<(), PecosError> {
        if let Some(err) = self.terminal_failure_error() {
            return Err(err);
        }
        // One-shot: a redundant continue_processing poll after Complete must
        // not re-run the gates (a second shot_end on an ended shot is a
        // legitimate runtime error) or emit a second terminal marker.
        if self
            .dynamic_state
            .as_ref()
            .is_some_and(|state| state.finalized)
        {
            return Ok(());
        }
        self.verify_runtime_drained()?;
        if !self.pending_measurements.is_empty() {
            let slots: Vec<_> = self.pending_measurements.keys().copied().collect();
            return Err(self.latch_terminal_error(format!(
                "runtime dropped imported measurements for result slots {slots:?}"
            )));
        }
        if let Err(e) = self.runtime.shot_end() {
            return Err(self.latch_terminal_error(format!("runtime shot_end failed: {e}")));
        }
        self.trace_named_result_traces_from_dynamic_handle();
        self.trace_complete_chunk();
        if let Some(ref mut state) = self.dynamic_state {
            state.finalized = true;
        }
        self.shot_lifecycle = ShotLifecycle::Finalized;
        Ok(())
    }

    /// The one complete reset behind `Engine::reset`, `ClassicalEngine::reset`
    /// and `ControlEngine::reset`: stop the worker, reset the runtime and the
    /// interface, and clear per-shot state. A failure is latched in
    /// `reset_failure` until a later reset succeeds.
    fn reset_all(&mut self) -> Result<(), PecosError> {
        self.reset_all_with_timeout(RESET_WORKER_TIMEOUT)
    }

    fn reset_all_with_timeout(&mut self, timeout: Duration) -> Result<(), PecosError> {
        debug!("QisEngine: reset() called");
        let reset = self
            .abort_dynamic_execution(timeout)
            .and_then(|()| {
                self.runtime
                    .reset()
                    .map_err(|e| PecosError::Generic(format!("Failed to reset runtime: {e}")))
            })
            .and_then(|()| match self.interface {
                Some(ref mut interface) => interface
                    .reset()
                    .map_err(crate::interface_impl::interface_error_to_pecos),
                None => Ok(()),
            });
        if let Err(error) = reset {
            self.shot_lifecycle = ShotLifecycle::Failed(error.to_string());
            self.reset_failure = Some(error.to_string());
            return Err(error);
        }
        self.reset_failure = None;
        self.shot_lifecycle = ShotLifecycle::Idle;
        self.scheduled_drain_round = 0;
        self.scheduled_drain_feedback = false;
        self.current_operations = None;
        self.started = false;
        self.measurement_mapping.clear();
        self.measurement_results.clear();
        self.pending_measurements.clear();
        self.current_shot_seed = None;
        debug!("QisEngine: reset() completed, cleared measurement_results");
        Ok(())
    }

    /// Reclaim ownership before discarding any shot state. Pure native loops
    /// and blocking callbacks cannot be interrupted until their next checkpoint.
    /// The deadline bounds the receive, not host waits or runtime/plugin reset.
    fn abort_dynamic_execution(&mut self, timeout: Duration) -> Result<(), PecosError> {
        if let Some(state) = &self.dynamic_state
            && !state.execution_complete
        {
            let worker = self.persistent_worker.as_ref().ok_or_else(|| {
                PecosError::Generic("No dynamic worker available to reclaim interface".to_string())
            })?;
            // A late result recovers even if cancellation itself is unavailable.
            let result = if let Some(result) = worker.try_recv_result() {
                result
            } else {
                let handle = state.sync_handle.as_ref().ok_or_else(|| {
                    PecosError::Generic("No sync handle available to cancel worker".to_string())
                })?;
                handle
                    .abort_execution()
                    .map_err(crate::interface_impl::interface_error_to_pecos)?;
                worker.recv_result_until(Instant::now() + timeout)?
            };
            let failure = match result {
                Ok((_, interface)) => {
                    self.interface = Some(interface);
                    None
                }
                Err((failure, interface)) => {
                    if let Some(interface) = interface {
                        self.interface = Some(interface);
                    }
                    Some(failure)
                }
            };
            // This receive is one-shot, even when the result reports a failure.
            if let Some(state) = self.dynamic_state.as_mut() {
                state.execution_complete = true;
            }
            if let Some(failure) = failure
                && !failure.cancelled()
            {
                if failure.teardown.is_some() || self.interface.is_none() {
                    return Err(PecosError::Generic(format!(
                        "dynamic QIS worker failed during reset: {failure}"
                    )));
                }
                warn!("Discarding abandoned QIS shot execution failure during reset: {failure}");
            }
        }
        // A shot's interface must come back before its state is discarded; an
        // engine that never ran a dynamic shot has nothing to reclaim.
        if self.dynamic_state.is_some() && self.interface.is_none() {
            return Err(PecosError::Generic(
                "Reset could not reclaim worker interface".to_string(),
            ));
        }
        self.dynamic_state = None;
        self.pending_dynamic_ops.clear();
        Ok(())
    }
}

impl Drop for QisEngine {
    fn drop(&mut self) {
        if let Some(state) = &self.dynamic_state
            && !state.execution_complete
            && let Some(handle) = &state.sync_handle
            && let Err(error) = handle.abort_execution()
        {
            warn!("Failed to abort dynamic execution during engine drop: {error}");
            if let Some(worker) = &mut self.persistent_worker {
                worker.abort_error = Some(error);
            }
        }
    }
}

impl Engine for QisEngine {
    type Input = ();
    type Output = Shot;

    fn process(&mut self, _input: Self::Input) -> Result<Self::Output, PecosError> {
        debug!("QisEngine::process called");

        // Use the ControlEngine implementation for processing
        let mut stage = self.start(())?;

        loop {
            match stage {
                EngineStage::NeedsProcessing(_) => {
                    // In standalone mode, we can't actually execute quantum ops
                    // Just return empty measurements
                    let empty_msg = ByteMessage::builder().build();
                    stage = self.continue_processing(empty_msg)?;
                }
                EngineStage::Complete(shot) => {
                    return Ok(shot);
                }
            }
        }
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        self.reset_all()
    }
}

impl ClassicalEngine for QisEngine {
    fn num_qubits(&self) -> usize {
        // The trait contract asks for the number of simulator slots required,
        // not the count of live program handles: freed handles shrink
        // `active_qubit_slots.len()` but never shrink the simulator, so we
        // return the physical-slot high-water mark instead. The runtime can
        // report its own baseline (e.g. from `allocated_qubits` metadata) and
        // we take the larger of the two.
        let num_qubits = if let Some(hint) = self.num_qubits_hint {
            hint
        } else {
            self.runtime.num_qubits().max(self.num_physical_slots)
        };
        debug!("QisEngine: num_qubits() returning {num_qubits}");
        num_qubits
    }

    fn set_num_qubits_hint(&mut self, num_qubits: usize) {
        self.num_qubits_hint = Some(num_qubits);
        self.runtime.set_num_qubits(num_qubits);
    }

    fn set_seed(&mut self, seed: u64) {
        // Seed the RNG for generating per-shot seeds
        self.rng = PecosRng::seed_from_u64(seed);
        debug!("QisEngine: Set master seed to {seed}");
    }

    // HybridEngine resets its classical engine through this method.
    fn reset(&mut self) -> Result<(), PecosError> {
        self.reset_all()
    }

    fn generate_commands(&mut self) -> Result<ByteMessage, PecosError> {
        Err(PecosError::Generic(
            "QisEngine must be driven through ControlEngine::start/continue_processing, which apply the qubit lifetime and prep rules".into(),
        ))
    }

    fn get_results(&self) -> Result<Shot, PecosError> {
        debug!("QisEngine::get_results called");
        debug!(
            "QisEngine: get_results() called, stored results: {:?}",
            self.measurement_results
        );

        // Convert stored measurement results to PECOS shot format
        let mut shot = Shot::default();

        if let Some(error) = self.terminal_failure_error() {
            return Err(error);
        }

        match &self.shot_lifecycle {
            ShotLifecycle::Idle => return Ok(shot),
            ShotLifecycle::Running => {
                return Err(PecosError::Generic("shot not finished (Running)".into()));
            }
            ShotLifecycle::Failed(error) => return Err(PecosError::Generic(error.clone())),
            ShotLifecycle::Finalized => {}
        }

        // Named outputs preserve the scalar-for-one, vector-otherwise rule.
        let mut has_named_results = false;
        if let Some(state) = &self.dynamic_state
            && let Some(handle) = &state.sync_handle
        {
            let named_results = handle.get_named_results().map_err(|error| {
                PecosError::Generic(format!("Failed to get named results: {error}"))
            })?;
            has_named_results = !named_results.is_empty();
            for (name, values) in named_results {
                use pecos_qis_ffi_types::NamedResult;
                let mut data: Vec<Data> = match values {
                    NamedResult::Bool(values) => values
                        .into_iter()
                        .map(|b| Data::U32(u32::from(b)))
                        .collect(),
                    NamedResult::I64(values) => values.into_iter().map(Data::I64).collect(),
                    NamedResult::U64(values) => values.into_iter().map(Data::U64).collect(),
                    NamedResult::F64(values) => values.into_iter().map(Data::F64).collect(),
                };
                let value = if data.len() == 1 {
                    data.remove(0)
                } else {
                    Data::Vec(data)
                };
                shot.data.insert(name, value);
            }
        }

        // Only add raw measurements if there are no named results.
        // This handles circuits with variable loop iterations where each shot
        // may produce a different number of raw measurements, but the named
        // results (from result() calls) are consistent.
        if !has_named_results {
            for (result_id, value) in &self.measurement_results {
                shot.data
                    .insert(format!("measurement_{result_id}"), Data::U32(*value));
                debug!("QisEngine: Added to shot: measurement_{result_id} = {value}");
            }
        }

        debug!("QisEngine: Final shot data: {:?}", shot.data);
        debug!(
            "Returning shot with {} measurement results (has_named_results={})",
            self.measurement_results.len(),
            has_named_results
        );
        Ok(shot)
    }

    fn handle_measurements(&mut self, message: ByteMessage) -> Result<(), PecosError> {
        debug!("QisEngine::handle_measurements called");

        // Extract measurements from ByteMessage
        let measurements = Self::parse_measurement_outcomes(&message)?;

        debug!(
            "QisEngine: Received {} measurements: {:?}",
            measurements.len(),
            measurements
        );
        debug!(
            "QisEngine: Mapping size: {}, mapping: {:?}",
            self.measurement_mapping.len(),
            self.measurement_mapping
        );

        let updates = self.map_measurements(&measurements)?;
        let _ = self.store_measurement_updates(&updates)?;

        debug!(
            "QisEngine: Final measurement_results: {:?}",
            self.measurement_results
        );

        self.provide_measurements_terminal(&updates)
    }

    fn compile(&self) -> Result<(), PecosError> {
        // The QIS program is compiled/loaded when the interface is created
        // This method just confirms the engine is ready for execution
        log::info!("QIS program compilation verified - engine ready for execution");
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl ControlEngine for QisEngine {
    type Input = ();
    type Output = Shot;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    fn start(
        &mut self,
        _input: Self::Input,
    ) -> Result<EngineStage<Self::EngineInput, Self::Output>, PecosError> {
        debug!("QisEngine::start called");

        // A latched reset failure is cleared only by a complete reset, which
        // also resets the interface; starting a shot must not bypass it.
        if let Some(failure) = &self.reset_failure {
            return Err(PecosError::Generic(failure.clone()));
        }

        // Verify we have a dynamic-capable interface
        if !self
            .interface
            .as_ref()
            .is_some_and(|i| i.supports_dynamic())
        {
            return Err(PecosError::Generic(
                "QisEngine requires a dynamic-capable interface (e.g., QisHeliosInterface)"
                    .to_string(),
            ));
        }

        if self.scheduled_transport.enabled()
            && (self.operation_trace_dir.is_some() || self.operation_trace_collector.is_some())
        {
            return Err(PecosError::Input(
                "scheduled transport does not yet support operation tracing".into(),
            ));
        }
        self.shot_lifecycle = ShotLifecycle::Running;
        self.scheduled_drain_round = 0;
        self.scheduled_drain_feedback = false;
        let attempt = (|| {
            // Clear previous shot's measurement state
            self.measurement_results.clear();
            self.pending_measurements.clear();
            self.measurement_mapping.clear();
            self.pending_dynamic_ops.clear();
            self.simulated_op_count = 0;
            self.reset_qubit_slots();
            debug!("QisEngine: Cleared previous measurement results for new shot");

            // Generate a per-shot seed from our RNG
            let shot_seed = self.rng.next_u64();
            debug!("QisEngine: Generated shot seed {shot_seed}");

            // Store the shot seed for quantum engine access
            self.current_shot_seed = Some(shot_seed);
            self.begin_trace_shot();

            // Reset the runtime to ensure clean state for new shot. A failure is
            // latched like a failed `reset_all`, which alone clears the latch.
            if let Err(e) = self.runtime.reset() {
                let error = PecosError::Generic(format!("Failed to reset runtime: {e}"));
                self.shot_lifecycle = ShotLifecycle::Failed(error.to_string());
                self.reset_failure = Some(error.to_string());
                return Err(error);
            }

            // Start a new shot with the generated seed and a real, monotonically
            // increasing shot id (a plugin keying state or telemetry on the shot
            // id must not see every shot as shot 0).
            let shot_id = u64::try_from(self.trace_shot_index)
                .map_err(|_| PecosError::Generic("shot index exceeds u64".to_string()))?;
            self.runtime
                .shot_start(shot_id, Some(shot_seed))
                .map_err(|e| PecosError::Generic(format!("Failed to start shot: {e}")))?;

            self.started = true;

            // Start LLVM program in worker thread
            self.start_dynamic_worker()?;

            // Wait for the worker to either need a result or complete
            // Use long timeout as safety net - condvar will wake immediately on signal
            if let Some(result_id) = self.wait_for_result_needed(30_000) {
                debug!("Worker needs result for id={result_id}");
                if let Some(commands) = self.process_result_request(result_id, "pending_start")? {
                    return Ok(EngineStage::NeedsProcessing(commands));
                }
            }

            // Check if worker completed without needing any results
            if self.check_worker_complete() {
                if let Some(err) = self.terminal_failure_error() {
                    return Err(err);
                }
                // Worker completed but we still need to process any pending operations
                // through the quantum engine (e.g., programs without measurement-dependent conditionals)
                if !self.pending_dynamic_ops.is_empty() {
                    let final_ops = std::mem::take(&mut self.pending_dynamic_ops);
                    if !final_ops.is_empty() {
                        let lowered = self.lower_operations_terminal(&final_ops)?;
                        self.trace_operations_chunk(
                            "pending_final",
                            &final_ops,
                            None,
                            Some(&lowered),
                        );
                        return Ok(EngineStage::NeedsProcessing(lowered.commands));
                    }
                }
                if let Some(commands) = self.drain_terminal_commands()? {
                    return Ok(EngineStage::NeedsProcessing(commands));
                }
                self.finalize_shot_for_certification()?;
                let shot = self.get_results()?;
                return Ok(EngineStage::Complete(shot));
            }

            // Return empty commands while we wait
            Ok(EngineStage::NeedsProcessing(ByteMessage::builder().build()))
        })();
        attempt.inspect_err(|error: &PecosError| {
            // Preserve an error already latched during this attempt. Startup
            // can also fail before there is a dynamic state to hold the latch.
            if !matches!(self.shot_lifecycle, ShotLifecycle::Failed(_)) {
                self.latch_terminal_error(error.to_string());
            }
        })
    }

    fn continue_processing(
        &mut self,
        input: Self::EngineOutput,
    ) -> Result<EngineStage<Self::EngineInput, Self::Output>, PecosError> {
        debug!("QisEngine::continue_processing called");
        if let Some(failure) = &self.reset_failure {
            return Err(PecosError::Generic(format!(
                "Cannot continue after failed reset: {failure}"
            )));
        }

        // Verify dynamic state exists (set by start())
        if self.dynamic_state.is_none() {
            return Err(PecosError::Generic(
                "continue_processing called without dynamic state - was start() called?"
                    .to_string(),
            ));
        }

        let measurements = Self::parse_measurement_outcomes(&input)?;
        let measurement_updates = self.map_measurements(&measurements)?;
        let current_updates = self.store_measurement_updates(&measurement_updates)?;
        if !measurement_updates.is_empty() {
            self.provide_measurements_terminal(&measurement_updates)?;
        }

        // First, check if worker already completed (before processing anything else)
        // This avoids unnecessary work if the worker finished
        if self.check_worker_complete() {
            if let Some(err) = self.terminal_failure_error() {
                return Err(err);
            }
            debug!("Worker already complete, finishing shot");
            // Process any final operations
            if !self.pending_dynamic_ops.is_empty() {
                let final_ops = std::mem::take(&mut self.pending_dynamic_ops);
                if !final_ops.is_empty() {
                    let lowered = self.lower_operations_terminal(&final_ops)?;
                    self.trace_operations_chunk("pending_final", &final_ops, None, Some(&lowered));
                    return Ok(EngineStage::NeedsProcessing(lowered.commands));
                }
            }
            if let Some(commands) = self.drain_terminal_commands()? {
                return Ok(EngineStage::NeedsProcessing(commands));
            }
            self.finalize_shot_for_certification()?;
            let shot = self.get_results()?;
            return Ok(EngineStage::Complete(shot));
        }

        // Provide new measurement values to the dynamic worker thread.
        for &(result_id, value) in &current_updates {
            debug!("Stored and providing measurement: result_id={result_id} value={value}");
            self.set_dynamic_result(result_id as u64, value)?;
        }

        // Only the outstanding read's freshly delivered outcome permits ready.
        // Other measurements leave need_result set so the same request can
        // process any work the runtime still holds.
        if !current_updates.is_empty()
            && self.wait_for_result_needed(0).is_some_and(|requested| {
                current_updates
                    .iter()
                    .any(|&(result_id, _)| result_id as u64 == requested)
            })
        {
            self.signal_dynamic_result_ready()?;
        }

        // Clear measurement mapping for next batch
        self.measurement_mapping.clear();

        // Wait for worker to need more results or complete
        // Condvar wakes immediately on signal; timeout is just a safety net
        if let Some(result_id) = self.wait_for_result_needed(30_000) {
            debug!("Worker needs result for id={result_id}");

            if let Some(commands) = self.process_result_request(result_id, "pending_continue")? {
                return Ok(EngineStage::NeedsProcessing(commands));
            }
        }

        // Check if worker completed after the wait
        if self.check_worker_complete() {
            if let Some(err) = self.terminal_failure_error() {
                return Err(err);
            }
            debug!("Worker completed after wait");
            // Process any final operations
            if !self.pending_dynamic_ops.is_empty() {
                let final_ops = std::mem::take(&mut self.pending_dynamic_ops);
                if !final_ops.is_empty() {
                    let lowered = self.lower_operations_terminal(&final_ops)?;
                    self.trace_operations_chunk("pending_final", &final_ops, None, Some(&lowered));
                    return Ok(EngineStage::NeedsProcessing(lowered.commands));
                }
            }
            if let Some(commands) = self.drain_terminal_commands()? {
                return Ok(EngineStage::NeedsProcessing(commands));
            }
            self.finalize_shot_for_certification()?;
            let shot = self.get_results()?;
            return Ok(EngineStage::Complete(shot));
        }

        // Return empty commands while we wait
        Ok(EngineStage::NeedsProcessing(ByteMessage::builder().build()))
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        self.reset_all()
    }
}

// Tests for QisEngine are in integration tests since they require
// actual interface and runtime implementations.

#[cfg(test)]
mod tests {
    pub(super) mod drop_tests {
        include!("ccengine_drop_tests.rs");
    }
    mod reset_tests {
        include!("ccengine_reset_tests.rs");
    }
    use super::*;
    use crate::runtime::{ClassicalState, Result as RuntimeResult};
    use tempfile::TempDir;

    #[derive(Clone, Default)]
    struct DummyRuntime {
        state: ClassicalState,
        delivered: Arc<Mutex<Vec<(usize, bool)>>>,
    }

    #[test]
    fn boolean_feedback_preserves_order_duplicates_and_rejects_leakage() {
        let mut runtime = DummyRuntime::default();
        runtime
            .provide_measurement_outcomes(vec![(7, 0), (2, 1), (7, 1)])
            .unwrap();
        assert_eq!(
            *runtime.delivered.lock().unwrap(),
            [(7, false), (2, true), (7, true)]
        );
        runtime.delivered.lock().unwrap().clear();
        assert!(
            runtime
                .provide_measurement_outcomes(vec![(7, 1), (7, 2)])
                .is_err()
        );
        assert_eq!(*runtime.delivered.lock().unwrap(), []);
    }

    #[test]
    fn host_feedback_preserves_repeated_program_slots() {
        let runtime = DummyRuntime::default();
        let delivered = Arc::clone(&runtime.delivered);
        let mut engine = QisEngine::with_runtime(Box::new(runtime));
        engine
            .provide_measurement_updates_to_runtime(&[(7, 0), (7, 1)])
            .unwrap();
        assert_eq!(*delivered.lock().unwrap(), [(7, false), (7, true)]);
    }

    impl QisRuntime for DummyRuntime {
        fn load_interface(&mut self, _interface: OperationList) -> RuntimeResult<()> {
            Ok(())
        }

        fn execute_until_quantum(&mut self) -> RuntimeResult<Option<Vec<QuantumOp>>> {
            Ok(None)
        }

        fn provide_measurements(
            &mut self,
            measurements: BTreeMap<usize, bool>,
        ) -> RuntimeResult<()> {
            self.delivered.lock().unwrap().extend(measurements);
            Ok(())
        }

        fn get_classical_state(&self) -> &ClassicalState {
            &self.state
        }

        fn get_classical_state_mut(&mut self) -> &mut ClassicalState {
            &mut self.state
        }

        fn is_complete(&self) -> bool {
            true
        }

        fn num_qubits(&self) -> usize {
            1
        }
    }

    struct PoisonedImport {
        imports: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl crate::qis_interface::DynamicSyncHandle for PoisonedImport {
        fn wait_for_need_result(&self, _: u64) -> Option<u64> {
            Some(7)
        }
        fn set_measurement_result(&self, _: u64, _: bool) -> Result<(), InterfaceError> {
            panic!("an import failure cannot supply a result")
        }
        fn signal_result_ready(&self) -> Result<(), InterfaceError> {
            panic!("an import failure cannot signal readiness")
        }
        fn get_pending_operations(&self) -> Result<Vec<Operation>, InterfaceError> {
            self.imports
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Err(InterfaceError::ExecutionError(
                "poisoned pending-operations lock".into(),
            ))
        }
        fn abort_execution(&self) -> Result<(), InterfaceError> {
            Ok(())
        }
        fn get_named_results(
            &self,
        ) -> Result<BTreeMap<String, pecos_qis_ffi_types::NamedResult>, InterfaceError> {
            Ok(BTreeMap::new())
        }
        fn get_named_result_traces(&self) -> Result<Vec<NamedResultTrace>, InterfaceError> {
            Ok(vec![])
        }
    }

    #[test]
    fn pending_operations_import_error_is_terminal_and_sticky() {
        let imports = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut engine = QisEngine::with_runtime(Box::new(DummyRuntime::default()));
        engine.dynamic_state = Some(DynamicExecutionState {
            sync_handle: Some(Box::new(PoisonedImport {
                imports: imports.clone(),
            })),
            execution_complete: false,
            terminal_error: None,
            finalized: false,
            terminal_lowering_flushed: false,
        });
        let error = engine
            .continue_processing(ByteMessage::outcomes_builder().build())
            .err()
            .expect("import must fail");
        assert!(
            error
                .to_string()
                .contains("failed to import pending operations"),
            "{error}"
        );
        assert!(
            error
                .to_string()
                .contains("poisoned pending-operations lock"),
            "{error}"
        );
        let retry = engine
            .continue_processing(ByteMessage::outcomes_builder().build())
            .err()
            .expect("import must fail");
        assert_eq!(retry.to_string(), error.to_string());
        assert_eq!(imports.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[test]
    fn measurement_count_mismatches_fail_before_storing_outcomes() {
        for dynamic in [false, true] {
            for (mapping, outcomes) in [
                (vec![], vec![1]),
                (vec![8], vec![0, 1]),
                (vec![8, 9], vec![1]),
                (vec![8], vec![]),
            ] {
                let mut engine = QisEngine::with_runtime(Box::new(DummyRuntime::default()));
                engine.measurement_mapping.clone_from(&mapping);
                engine.dynamic_state = Some(DynamicExecutionState {
                    sync_handle: None,
                    execution_complete: true,
                    terminal_error: None,
                    finalized: false,
                    terminal_lowering_flushed: false,
                });
                let input = ByteMessage::builder().add_outcomes(&outcomes).build();
                let error = if dynamic {
                    engine
                        .continue_processing(input)
                        .err()
                        .expect("count mismatch")
                } else {
                    engine
                        .handle_measurements(input)
                        .expect_err("count mismatch")
                };
                assert!(error.to_string().contains(&format!(
                    "QIS measurement count mismatch: {} queued measurements, {} outcomes",
                    mapping.len(),
                    outcomes.len()
                )));
                assert!(engine.measurement_results.is_empty());
                assert_eq!(engine.measurement_mapping, mapping);
                assert!(engine.terminal_failure_error().is_some());
                assert!(engine.get_results().is_err());
            }
        }
    }

    #[test]
    fn generate_commands_requires_control_engine() {
        let mut engine: Box<dyn ClassicalEngine> =
            Box::new(QisEngine::with_runtime(Box::new(DummyRuntime::default())));
        let error = engine
            .generate_commands()
            .err()
            .expect("unsupported entry point")
            .to_string();
        assert!(error.contains("QisEngine"), "{error}");
        assert!(
            error.contains("ControlEngine::start/continue_processing"),
            "{error}"
        );
        assert!(error.contains("lifetime and prep rules"), "{error}");
    }

    #[cfg(feature = "selene-runtimes")]
    fn delayed_metadata_engine() -> QisEngine {
        let mut engine = QisEngine::with_runtime(Box::new(
            crate::selene_runtimes::selene_soft_rz_runtime().unwrap(),
        ));
        engine.set_num_qubits_hint(2);
        engine.runtime.shot_start(0, None).unwrap();
        engine
    }

    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn delayed_inserted_prep_does_not_take_later_reset_metadata() {
        use pecos_core::gate_type::GateType::{MZ, PZ};
        let mut engine = delayed_metadata_engine();
        let first = engine
            .lower_operations_to_commands(&[
                QuantumOp::Measure(0, 0).into(),
                QuantumOp::RZ(0.5, 1).into(),
            ])
            .unwrap();
        assert_eq!(
            first
                .commands
                .quantum_ops()
                .unwrap()
                .iter()
                .map(|gate| gate.gate_type)
                .collect::<Vec<_>>(),
            [PZ, MZ]
        );
        let metadata = TraceMetadata::from([("source_label".into(), "program reset".into())]);
        let second = engine
            .lower_operations_to_commands(&[
                Operation::TraceMetadata {
                    metadata: metadata.clone(),
                    qubit: Some(1),
                },
                QuantumOp::Reset(1).into(),
                QuantumOp::Measure(1, 1).into(),
            ])
            .unwrap();
        assert_eq!(
            second
                .commands
                .quantum_ops()
                .unwrap()
                .iter()
                .map(|gate| gate.gate_type)
                .collect::<Vec<_>>(),
            [PZ, PZ, MZ]
        );
        assert_eq!(
            second.gate_metadata,
            [TraceMetadata::new(), metadata, TraceMetadata::new()]
        );
    }

    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn delayed_labelled_operation_keeps_metadata_across_chunks() {
        for op in [QuantumOp::Reset(1), QuantumOp::RXY(0.25, 0.0, 1)] {
            let mut engine = delayed_metadata_engine();
            let metadata = TraceMetadata::from([
                ("source_label".into(), "delayed operation".into()),
                ("source_lowering_required".into(), "true".into()),
            ]);
            let expected_gate = if matches!(op, QuantumOp::Reset(_)) {
                "PZ"
            } else {
                "RXY1Q"
            };
            let first = engine
                .lower_operations_to_commands(&[
                    QuantumOp::Measure(0, 0).into(),
                    QuantumOp::RZ(0.5, 1).into(),
                    Operation::TraceMetadata {
                        metadata: metadata.clone(),
                        qubit: Some(1),
                    },
                    op.into(),
                ])
                .unwrap();
            // #1027's label-start barrier emits the queued inserted prep in A.
            // The labelled operation itself remains queued until B.
            assert_eq!(first.gate_metadata, vec![TraceMetadata::new(); 3]);
            assert_eq!(
                first
                    .commands
                    .quantum_ops()
                    .unwrap()
                    .iter()
                    .map(|gate| gate.gate_type.to_string())
                    .collect::<Vec<_>>(),
                ["PZ", "MZ", "PZ"]
            );
            let second = engine
                .lower_operations_to_commands(&[QuantumOp::Measure(1, 1).into()])
                .unwrap();
            assert_eq!(
                second
                    .commands
                    .quantum_ops()
                    .unwrap()
                    .iter()
                    .map(|gate| gate.gate_type.to_string())
                    .collect::<Vec<_>>(),
                [expected_gate, "MZ"]
            );
            assert_eq!(second.gate_metadata, [metadata, TraceMetadata::new()]);
        }
    }

    fn prep_test_engine(selene: bool) -> QisEngine {
        let mut engine = if selene {
            #[cfg(feature = "selene-runtimes")]
            {
                QisEngine::with_runtime(Box::new(
                    crate::selene_runtimes::selene_simple_runtime().unwrap(),
                ))
            }
            #[cfg(not(feature = "selene-runtimes"))]
            panic!("Selene tests require selene-runtimes");
        } else {
            QisEngine::with_runtime(Box::new(DummyRuntime::default()))
        };
        engine.set_num_qubits_hint(3);
        engine.runtime.shot_start(0, None).unwrap();
        engine
    }

    fn prep_test_gates(
        engine: &mut QisEngine,
        ops: &[Operation],
    ) -> Vec<pecos_core::gate_type::GateType> {
        engine
            .lower_operations_to_commands(ops)
            .unwrap()
            .commands
            .quantum_ops()
            .unwrap()
            .iter()
            .map(|gate| gate.gate_type)
            .collect()
    }

    // Red/green: the direct allocation used to add a second prep.
    #[test]
    fn prep_direct_first_reset_is_the_only_prep() {
        use pecos_core::gate_type::GateType::{H, PZ};
        let mut engine = prep_test_engine(false);
        let lowered = engine
            .lower_operations_to_commands(&[
                Operation::AllocateQubit { id: 71 },
                QuantumOp::Reset(71).into(),
                QuantumOp::H(71).into(),
            ])
            .unwrap();
        let gates = lowered.commands.quantum_ops().unwrap();
        assert_eq!(
            gates.iter().map(|gate| gate.gate_type).collect::<Vec<_>>(),
            [PZ, H]
        );
        assert!(
            gates
                .iter()
                .all(|gate| gate.qubits.as_slice() == [0.into()])
        );
    }

    // Compatibility: direct allocation followed by H already had one prep.
    #[test]
    fn prep_direct_allocation_before_h() {
        use pecos_core::gate_type::GateType::{H, PZ};
        assert_eq!(
            prep_test_gates(
                &mut prep_test_engine(false),
                &[Operation::AllocateQubit { id: 0 }, QuantumOp::H(0).into(),]
            ),
            [PZ, H]
        );
    }

    // Red/green: Selene previously received no prep for this allocation.
    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn prep_selene_allocation_before_h() {
        use pecos_core::gate_type::GateType::{PZ, RXY1Q, RZ};
        assert_eq!(
            prep_test_gates(
                &mut prep_test_engine(true),
                &[Operation::AllocateQubit { id: 0 }, QuantumOp::H(0).into(),]
            ),
            [PZ, RXY1Q, RZ]
        );
    }

    // Red/green: legacy first use lacked a prep; later uses must not repeat it.
    #[test]
    fn prep_legacy_first_use_and_shot_boundary() {
        use pecos_core::gate_type::GateType::{H, PZ, X};
        let mut engine = prep_test_engine(false);
        assert_eq!(
            prep_test_gates(&mut engine, &[QuantumOp::H(7).into()]),
            [PZ, H]
        );
        assert_eq!(prep_test_gates(&mut engine, &[QuantumOp::X(7).into()]), [X]);
        engine.reset_qubit_slots();
        assert_eq!(
            prep_test_gates(&mut engine, &[QuantumOp::H(7).into()]),
            [PZ, H]
        );
    }

    // Red/green: the prep belongs to the first-use chunk, including after feedback.
    #[test]
    fn prep_deferred_across_feedback() {
        use pecos_core::gate_type::GateType::{H, MZ, PZ, X};
        let mut engine = prep_test_engine(false);
        assert_eq!(
            prep_test_gates(&mut engine, &[Operation::AllocateQubit { id: 71 }]),
            vec![]
        );
        assert_eq!(
            prep_test_gates(
                &mut engine,
                &[QuantumOp::H(71).into(), QuantumOp::Measure(71, 9).into(),]
            ),
            [PZ, H, MZ]
        );
        engine
            .handle_measurements(ByteMessage::builder().add_outcomes(&[0]).build())
            .unwrap();
        assert_eq!(
            prep_test_gates(&mut engine, &[QuantumOp::X(71).into()]),
            [X]
        );
    }

    // Red/green: a re-allocation defers its own prep until the next use.
    #[test]
    fn prep_reallocation_starts_new_lifetime() {
        use pecos_core::gate_type::GateType::{H, PZ, X};
        let mut engine = prep_test_engine(false);
        assert_eq!(
            prep_test_gates(
                &mut engine,
                &[
                    Operation::AllocateQubit { id: 7 },
                    QuantumOp::X(7).into(),
                    Operation::ReleaseQubit { id: 7 },
                    Operation::AllocateQubit { id: 7 },
                ]
            ),
            [PZ, X]
        );
        assert_eq!(
            prep_test_gates(&mut engine, &[QuantumOp::H(7).into()]),
            [PZ, H]
        );
    }

    // Red/green: unused lifetimes must not produce a prep.
    #[test]
    fn prep_unused_allocation_emits_nothing() {
        assert_eq!(
            prep_test_gates(
                &mut prep_test_engine(false),
                &[
                    Operation::AllocateQubit { id: 7 },
                    Operation::ReleaseQubit { id: 7 },
                ]
            ),
            vec![]
        );
    }

    // Red/green: every fresh target needs a prep, in source operand order.
    #[test]
    fn prep_multi_qubit_first_use() {
        use pecos_core::gate_type::GateType::{CX, PZ};
        let mut engine = prep_test_engine(false);
        let lowered = engine
            .lower_operations_to_commands(&[QuantumOp::CX(7, 3).into()])
            .unwrap();
        let gates = lowered.commands.quantum_ops().unwrap();
        assert_eq!(
            gates.iter().map(|gate| gate.gate_type).collect::<Vec<_>>(),
            [PZ, PZ, CX]
        );
        assert_eq!(gates[0].qubits.as_slice(), [0.into()]);
        assert_eq!(gates[1].qubits.as_slice(), [1.into()]);
        assert_eq!(gates[2].qubits.as_slice(), [0.into(), 1.into()]);
    }

    // Compatibility: program-written preps remain normal PZ gates throughout a lifetime.
    #[test]
    fn prep_program_mid_circuit_reset_is_preserved() {
        use pecos_core::gate_type::GateType::{H, PZ};
        let mut engine = prep_test_engine(false);
        assert_eq!(
            prep_test_gates(
                &mut engine,
                &[
                    QuantumOp::Reset(0).into(),
                    QuantumOp::H(0).into(),
                    QuantumOp::Reset(0).into(),
                ]
            ),
            [PZ, H, PZ]
        );
    }

    // Compatibility: normalization must leave the original use-after-release error intact.
    #[test]
    fn prep_released_handle_is_not_a_legacy_first_use() {
        let mut engine = prep_test_engine(false);
        let error = engine
            .lower_operations_to_commands(&[
                Operation::AllocateQubit { id: 7 },
                QuantumOp::Reset(7).into(),
                Operation::ReleaseQubit { id: 7 },
                QuantumOp::H(7).into(),
            ])
            .err()
            .unwrap();
        assert!(error.to_string().contains("emitted H(7)"));
        assert!(error.to_string().contains("not currently active"));
    }

    fn check_prep_metadata(selene: bool) {
        let mut engine = prep_test_engine(selene);
        let global = TraceMetadata::from([("global".into(), "h".into())]);
        let scoped = TraceMetadata::from([("scoped".into(), "x".into())]);
        let ops = [
            Operation::TraceMetadata {
                metadata: scoped.clone(),
                qubit: Some(7),
            },
            Operation::TraceMetadata {
                metadata: global.clone(),
                qubit: None,
            },
            Operation::AllocateQubit { id: 3 },
            QuantumOp::H(3).into(),
            Operation::Barrier,
            QuantumOp::X(7).into(),
        ];
        let lowered = engine.lower_operations_to_commands(&ops).unwrap();
        assert_eq!(
            lowered.gate_metadata,
            if selene {
                vec![
                    TraceMetadata::new(),
                    global,
                    TraceMetadata::new(),
                    TraceMetadata::new(),
                    scoped,
                ]
            } else {
                vec![TraceMetadata::new(), global, TraceMetadata::new(), scoped]
            }
        );
        let gates = lowered.commands.quantum_ops().unwrap();
        assert_eq!(
            gates
                .iter()
                .map(|gate| gate.gate_type.to_string())
                .collect::<Vec<_>>(),
            if selene {
                vec!["PZ", "RXY1Q", "RZ", "PZ", "RXY1Q"]
            } else {
                vec!["PZ", "H", "PZ", "X"]
            }
        );
        let store = Arc::new(Mutex::new(Vec::new()));
        engine.set_operation_trace_collector(store.clone());
        engine.simulated_op_count = ops.len();
        engine.trace_operations_chunk("prep_test", &ops, None, Some(&lowered));
        let traces = store.lock().unwrap();
        assert_eq!(traces[0].operations, ops);
        assert_eq!(traces[0].num_operations, ops.len());
        assert_eq!(traces[0].simulated_op_count, ops.len());
        assert!(traces[0].lowered_quantum_ops_complete);
        assert_eq!(
            traces[0].lowered_quantum_ops.len(),
            if selene { 5 } else { 4 }
        );
    }

    // Red/green: global and scoped labels must stay on source ops, across allocations/barriers.
    #[test]
    fn prep_metadata_direct() {
        check_prep_metadata(false);
    }

    // Red/green: exercise Selene's source-to-lowered metadata matching with inserted preps.
    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn prep_metadata_selene() {
        check_prep_metadata(true);
    }

    // Compatibility: the direct route drops dangling metadata at the chunk boundary.
    #[test]
    fn prep_metadata_across_chunks() {
        for qubit in [None, Some(7)] {
            let mut engine = prep_test_engine(false);
            engine
                .lower_operations_to_commands(&[
                    QuantumOp::Reset(7).into(),
                    Operation::TraceMetadata {
                        metadata: TraceMetadata::from([("source_label".into(), "h".into())]),
                        qubit,
                    },
                ])
                .unwrap();
            let lowered = engine
                .lower_operations_to_commands(&[QuantumOp::H(7).into()])
                .unwrap();
            assert_eq!(lowered.gate_metadata, [TraceMetadata::new()]);
        }
    }

    // Compatibility: Selene rejects dangling global and qubit-scoped metadata in its chunk.
    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn prep_metadata_dangling_selene() {
        for (qubit, message) in [
            (
                None,
                "trace metadata was not followed by a quantum operation",
            ),
            (
                Some(7),
                "qubit-scoped trace metadata was not followed by a compatible quantum operation",
            ),
        ] {
            let mut engine = prep_test_engine(true);
            let error = engine
                .lower_operations_to_commands(&[
                    QuantumOp::Reset(7).into(),
                    Operation::TraceMetadata {
                        metadata: TraceMetadata::from([("source_label".into(), "h".into())]),
                        qubit,
                    },
                ])
                .err()
                .expect("dangling metadata must fail in its own chunk");
            assert!(error.to_string().contains(message), "{error}");
        }
    }

    fn check_prep_nonadjacent_metadata(selene: bool) {
        let mut engine = prep_test_engine(selene);
        let metadata = TraceMetadata::from([("source_label".into(), "x".into())]);
        let lowered = engine
            .lower_operations_to_commands(&[
                Operation::TraceMetadata {
                    metadata: metadata.clone(),
                    qubit: Some(1),
                },
                QuantumOp::H(0).into(),
                QuantumOp::X(1).into(),
            ])
            .unwrap();
        let gates = lowered.commands.quantum_ops().unwrap();
        assert_eq!(
            gates
                .iter()
                .map(|gate| gate.gate_type.to_string())
                .collect::<Vec<_>>(),
            if selene {
                vec!["PZ", "RXY1Q", "RZ", "PZ", "RXY1Q"]
            } else {
                vec!["PZ", "H", "PZ", "X"]
            }
        );
        assert_eq!(
            lowered.gate_metadata,
            if selene {
                vec![
                    TraceMetadata::new(),
                    TraceMetadata::new(),
                    TraceMetadata::new(),
                    TraceMetadata::new(),
                    metadata,
                ]
            } else {
                vec![
                    TraceMetadata::new(),
                    TraceMetadata::new(),
                    TraceMetadata::new(),
                    metadata,
                ]
            }
        );
    }

    // Red/green: both fresh legacy handles get preps without consuming the scoped label.
    #[test]
    fn prep_metadata_nonadjacent_direct() {
        check_prep_nonadjacent_metadata(false);
    }

    // Red/green: Selene keeps the scoped label on X after both inserted preps.
    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn prep_metadata_nonadjacent_selene() {
        check_prep_nonadjacent_metadata(true);
    }

    // Red/green: unlabelled native occurrences must consume their own lowered gates.
    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn prep_native_metadata_does_not_label_inserted_reset() {
        for qubit in [None, Some(0)] {
            let mut engine = prep_test_engine(true);
            let metadata = TraceMetadata::from([("source_label".into(), "program prep".into())]);
            let lowered = engine
                .lower_operations_to_commands(&[
                    QuantumOp::RZ(0.5, 0).into(),
                    Operation::TraceMetadata {
                        metadata: metadata.clone(),
                        qubit,
                    },
                    QuantumOp::Reset(0).into(),
                ])
                .unwrap();
            assert_eq!(
                lowered
                    .commands
                    .quantum_ops()
                    .unwrap()
                    .iter()
                    .map(|gate| gate.gate_type.to_string())
                    .collect::<Vec<_>>(),
                ["PZ", "RZ", "PZ"]
            );
            assert_eq!(
                lowered.gate_metadata,
                [TraceMetadata::new(), TraceMetadata::new(), metadata]
            );
        }
    }

    // Red/green: repeated program-native ops match metadata by occurrence too.
    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn prep_native_metadata_repeated_program_ops() {
        for op in [
            QuantumOp::Reset(0),
            QuantumOp::RZ(0.5, 0),
            QuantumOp::RXY(0.5, 0.25, 0),
            QuantumOp::RZZ(0.5, 0, 1),
        ] {
            for qubit in [None, Some(0)] {
                let mut engine = prep_test_engine(true);
                let metadata =
                    TraceMetadata::from([("source_label".into(), "second occurrence".into())]);
                let lowered = engine
                    .lower_operations_to_commands(&[
                        op.clone().into(),
                        Operation::TraceMetadata {
                            metadata: metadata.clone(),
                            qubit,
                        },
                        op.clone().into(),
                    ])
                    .unwrap();
                assert_eq!(lowered.gate_metadata.last(), Some(&metadata), "{op:?}");
                assert!(
                    lowered.gate_metadata[..lowered.gate_metadata.len() - 1]
                        .iter()
                        .all(TraceMetadata::is_empty),
                    "{op:?}"
                );
            }
        }
    }

    fn check_prep_unknown_release(selene: bool) {
        let mut engine = prep_test_engine(selene);
        engine.set_num_qubits_hint(1);
        let lowered = engine
            .lower_operations_to_commands(&[
                Operation::AllocateQubit { id: 0 },
                QuantumOp::X(0).into(),
                Operation::ReleaseQubit { id: 0 },
                Operation::ReleaseQubit { id: 9 },
                QuantumOp::Measure(9, 0).into(),
            ])
            .unwrap();
        let gates = lowered.commands.quantum_ops().unwrap();
        assert_eq!(
            gates
                .iter()
                .map(|gate| gate.gate_type.to_string())
                .collect::<Vec<_>>(),
            if selene {
                ["PZ", "RXY1Q", "PZ", "MZ"]
            } else {
                ["PZ", "X", "PZ", "MZ"]
            }
        );
        assert!(
            gates
                .iter()
                .all(|gate| gate.qubits.as_slice() == [0.into()])
        );
    }

    // Red/green: releasing an unknown handle must not suppress its first-use prep.
    #[test]
    fn prep_unknown_release_direct() {
        check_prep_unknown_release(false);
    }

    // Red/green: native slot reuse after an unknown release still receives a prep.
    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn prep_unknown_release_selene() {
        check_prep_unknown_release(true);
    }

    // Red/green: a second explicit allocation cannot restart a live lifetime.
    #[test]
    fn prep_duplicate_live_allocation_direct() {
        for prefix in [
            vec![Operation::AllocateQubit { id: 7 }, QuantumOp::H(7).into()],
            vec![QuantumOp::Measure(7, 0).into()],
            vec![Operation::AllocateQubit { id: 7 }],
        ] {
            for split in [false, true] {
                let mut engine = prep_test_engine(false);
                let mut ops = if split {
                    engine.lower_operations_to_commands(&prefix).unwrap();
                    Vec::new()
                } else {
                    prefix.clone()
                };
                ops.extend([Operation::AllocateQubit { id: 7 }, QuantumOp::X(7).into()]);
                let error = engine
                    .lower_operations_to_commands(&ops)
                    .err()
                    .expect("duplicate live allocation must fail");
                assert!(
                    error
                        .to_string()
                        .contains("program qubit 7 is already allocated"),
                    "{error}"
                );
            }
        }
    }

    // Red/green: the scheduled route also receives the deferred prep exactly once.
    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn prep_scheduled_first_use() {
        use pecos_core::gate_type::GateType::{PZ, RZ};
        use pecos_engines::scheduled_events::{ScheduledEventOp, decode_event_batches};
        let mut engine = prep_test_engine(true);
        engine.scheduled_transport = ScheduledTransport::V4;
        let lowered = engine
            .lower_operations_to_commands(&[
                Operation::AllocateQubit { id: 7 },
                QuantumOp::RZ(0.5, 7).into(),
            ])
            .unwrap();
        let gates = decode_event_batches(&lowered.commands)
            .unwrap()
            .into_iter()
            .flat_map(|batch| batch.operations)
            .map(|op| match op {
                ScheduledEventOp::Gate(gate) => gate.gate_type,
                ScheduledEventOp::Custom { .. } => panic!("expected gate"),
            })
            .collect::<Vec<_>>();
        assert_eq!(gates, [PZ, RZ]);
    }

    #[test]
    fn test_operation_trace_chunk_writes_json() {
        let temp_dir = TempDir::new().expect("tempdir");
        let mut engine = QisEngine::with_runtime(Box::new(DummyRuntime::default()));
        engine.set_operation_trace_dir(temp_dir.path());
        let collector: OperationTraceStore = Arc::new(Mutex::new(Vec::new()));
        engine.set_operation_trace_collector(collector.clone());
        engine.current_shot_seed = Some(123);
        engine.begin_trace_shot();

        let ops = vec![
            Operation::AllocateQubit { id: 0 },
            QuantumOp::H(0).into(),
            QuantumOp::Idle(20e-9, 0).into(),
            QuantumOp::Measure(0, 7).into(),
        ];
        let lowered = engine
            .lower_operations_to_commands(&ops)
            .expect("convert ops to lowered commands");
        engine.trace_operations_chunk("unit_test", &ops, Some(7), Some(&lowered));

        let mut trace_files = std::fs::read_dir(temp_dir.path())
            .expect("read trace dir")
            .map(|entry| entry.expect("dir entry").path())
            .collect::<Vec<_>>();
        trace_files.sort();
        assert_eq!(trace_files.len(), 1);

        let trace_json = std::fs::read_to_string(&trace_files[0]).expect("read trace json");
        let value: serde_json::Value = serde_json::from_str(&trace_json).expect("parse trace json");

        assert_eq!(value["format"], "pecos_qis_operation_trace_v1");
        assert_eq!(value["stage"], "unit_test");
        assert_eq!(value["shot_index"], 1);
        assert_eq!(value["waiting_for_result_id"], 7);
        assert_eq!(value["current_shot_seed"], 123);
        assert_eq!(value["num_operations"], 4);
        assert_eq!(value["lowered_quantum_ops_complete"], true);
        assert_eq!(value["operations"][0]["AllocateQubit"]["id"], 0);
        assert_eq!(value["operations"][1]["Quantum"]["H"], 0);
        assert_eq!(value["operations"][2]["Quantum"]["Idle"][0], 20e-9);
        assert_eq!(value["lowered_quantum_ops"][0]["gate_type"], "PZ");
        assert_eq!(
            value["lowered_quantum_ops"][0]["metadata"],
            serde_json::json!({})
        );
        assert_eq!(value["lowered_quantum_ops"][1]["gate_type"], "H");
        assert_eq!(value["lowered_quantum_ops"][2]["gate_type"], "Idle");
        assert_eq!(value["lowered_quantum_ops"][2]["params"][0], 20e-9);
        assert_eq!(value["lowered_quantum_ops"][3]["gate_type"], "MZ");
        assert_eq!(
            value["lowered_quantum_ops"][3]["measurement_result_ids"],
            serde_json::json!([7])
        );

        let in_memory = collector.lock().expect("collector lock");
        assert_eq!(in_memory.len(), 1);
        assert_eq!(in_memory[0].stage, "unit_test");
        assert_eq!(in_memory[0].lowered_quantum_ops[0].gate_type, "PZ");
        assert_eq!(in_memory[0].lowered_quantum_ops[2].gate_type, "Idle");
        assert_eq!(in_memory[0].lowered_quantum_ops[2].params, vec![20e-9]);
        assert_eq!(
            in_memory[0].lowered_quantum_ops[3].measurement_result_ids,
            vec![7]
        );
        assert!(in_memory[0].measurement_results.is_empty());
        drop(in_memory);

        engine.measurement_results.insert(7, 1);
        engine.trace_complete_chunk();
        let in_memory = collector.lock().expect("collector lock");
        assert_eq!(in_memory[1].stage, "trace_complete");
        assert_eq!(in_memory[1].measurement_results, BTreeMap::from([(7, 1)]));
    }

    #[test]
    fn qis_crz_carrier_lowers_before_angle_storage() {
        let mut engine = QisEngine::with_runtime(Box::new(DummyRuntime::default()));
        let lowered = engine
            .quantum_ops_to_lowered_commands(vec![
                QuantumOp::CRZ(std::f64::consts::TAU, 0, 1).into(),
            ])
            .expect("lower QIS CRZ");
        let gates = lowered.commands.quantum_ops().unwrap();
        assert_eq!(gates.len(), 3);
        assert_eq!(gates[0].gate_type, pecos_core::gate_type::GateType::Z);
        assert_eq!(gates[0].qubits.as_slice(), [0.into()]);
        assert_eq!(gates[1].gate_type, pecos_core::gate_type::GateType::RZZ);
        assert_eq!(gates[2].gate_type, pecos_core::gate_type::GateType::RZ);
        assert_eq!(gates[2].qubits.as_slice(), [1.into()]);
        assert_eq!(lowered.gate_metadata.len(), 3);
    }

    #[test]
    fn corrected_crz_copies_metadata_to_every_emitted_gate() {
        let mut engine = QisEngine::with_runtime(Box::new(DummyRuntime::default()));
        let metadata = TraceMetadata::from([("source_label".to_string(), "crz".to_string())]);
        let ops = vec![
            Operation::AllocateQubit { id: 0 },
            Operation::AllocateQubit { id: 1 },
            Operation::TraceMetadata {
                metadata: metadata.clone(),
                qubit: None,
            },
            QuantumOp::CRZ(std::f64::consts::TAU, 0, 1).into(),
            QuantumOp::H(1).into(),
        ];
        let lowered = engine
            .lower_operations_to_commands(&ops)
            .expect("lower annotated CRZ");
        let gates = lowered.commands.quantum_ops().unwrap();
        // First use emits two PZ commands before the three CRZ legs and H.
        assert_eq!(gates.len(), 6);
        assert_eq!(gates[2].gate_type, pecos_core::gate_type::GateType::Z);
        assert!(
            lowered.gate_metadata[..2]
                .iter()
                .all(TraceMetadata::is_empty)
        );
        assert_eq!(
            &lowered.gate_metadata[2..5],
            &[metadata.clone(), metadata.clone(), metadata]
        );
        assert!(lowered.gate_metadata[5].is_empty());
    }

    fn qis_crz_state(theta: f64, basis: usize) -> Vec<(f64, f64)> {
        use pecos_engines::Engine;
        use pecos_engines::quantum::StateVecEngine;

        let mut ops = vec![
            Operation::AllocateQubit { id: 0 },
            Operation::AllocateQubit { id: 1 },
        ];
        if basis & 1 != 0 {
            ops.push(QuantumOp::X(0).into());
        }
        if basis & 2 != 0 {
            ops.push(QuantumOp::X(1).into());
        }
        ops.push(QuantumOp::CRZ(theta, 1, 0).into());

        let mut engine = QisEngine::with_runtime(Box::new(DummyRuntime::default()));
        let lowered = engine
            .lower_operations_to_commands(&ops)
            .expect("lower QIS operations");
        let mut simulator = StateVecEngine::new(2);
        simulator
            .process(lowered.commands)
            .expect("execute lowered QIS operations");
        simulator
            .simulator_mut()
            .state()
            .iter()
            .map(|amplitude| (amplitude.re, amplitude.im))
            .collect()
    }

    #[test]
    fn qis_crz_carrier_preserves_full_matrix() {
        for theta in [
            -std::f64::consts::PI,
            std::f64::consts::PI / 3.0,
            std::f64::consts::PI,
            std::f64::consts::TAU,
            3.0 * std::f64::consts::PI,
        ] {
            let mut actual = vec![vec![(0.0, 0.0); 4]; 4];
            for column in 0..4 {
                for (row_values, amplitude) in actual.iter_mut().zip(qis_crz_state(theta, column)) {
                    row_values[column] = amplitude;
                }
            }
            let half = theta / 2.0;
            let reference = [
                (1.0, 0.0),
                (1.0, 0.0),
                (half.cos(), -half.sin()),
                (half.cos(), half.sin()),
            ];
            let phase = actual[0][0];
            let phase_norm = phase.0 * phase.0 + phase.1 * phase.1;
            assert!((phase_norm - 1.0).abs() < 1e-12);
            assert!((phase.0 - 1.0).abs() < 1e-12 && phase.1.abs() < 1e-12);
            for (row, row_values) in actual.iter().enumerate() {
                for (column, &value) in row_values.iter().enumerate() {
                    let normalized = (
                        (value.0 * phase.0 + value.1 * phase.1) / phase_norm,
                        (value.1 * phase.0 - value.0 * phase.1) / phase_norm,
                    );
                    let expected = if row == column {
                        reference[row]
                    } else {
                        (0.0, 0.0)
                    };
                    assert!(
                        (normalized.0 - expected.0).abs() < 1e-12
                            && (normalized.1 - expected.1).abs() < 1e-12,
                        "theta={theta}, entry=({row}, {column}), actual={normalized:?}, expected={expected:?}, phase={phase:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_rxyxy2q_lowering_preserves_gate_and_metadata() {
        let mut engine = QisEngine::with_runtime(Box::new(DummyRuntime::default()));
        let metadata = TraceMetadata::from([("source_label".to_string(), "xyxy".to_string())]);
        let source = QuantumOp::RXYXY2Q(-0.73, 0.41, 2, 0);
        let direct = engine
            .operations_to_lowered_commands(&[
                Operation::TraceMetadata {
                    metadata: metadata.clone(),
                    qubit: None,
                },
                source.clone().into(),
            ])
            .unwrap();
        let scheduled = engine
            .quantum_ops_to_lowered_commands(vec![LoweredQuantumOp::new(source, metadata.clone())])
            .unwrap();
        // Direct operations use program handles, which are assigned slots
        // in encounter order. Scheduled operations already use physical IDs.
        for (batch, qubits) in [(direct, [0, 1]), (scheduled, [2, 0])] {
            let gates = batch.commands.quantum_ops().unwrap();
            assert_eq!(gates.len(), 1);
            assert_eq!(gates[0].gate_type, pecos_core::gate_type::GateType::RXYXY2Q);
            assert_eq!(gates[0].qubits.as_slice(), &qubits.map(pecos_core::QubitId));
            assert_eq!(
                gates[0].angles.as_slice(),
                &[Angle64::from_radians(-0.73), Angle64::from_radians(0.41)]
            );
            assert_eq!(batch.gate_metadata, vec![metadata.clone()]);
        }
    }

    #[test]
    fn test_direct_lowering_preserves_leakage_measurement() {
        let mut engine = QisEngine::with_runtime(Box::new(DummyRuntime::default()));
        let ops = vec![
            Operation::AllocateQubit { id: 0 },
            QuantumOp::MeasureLeaked(0, 8).into(),
        ];

        let lowered = engine
            .lower_operations_to_commands(&ops)
            .expect("lower leakage-aware measurement");
        let quantum_ops = lowered
            .commands
            .quantum_ops()
            .expect("parse quantum operations");

        assert_eq!(quantum_ops.len(), 2);
        assert_eq!(
            quantum_ops[1].gate_type,
            pecos_core::gate_type::GateType::MeasureLeaked
        );
        assert_eq!(engine.measurement_mapping, vec![8]);
    }

    #[test]
    fn test_general_noise_returns_two_for_lowered_leakage_measurement() {
        use pecos_engines::QuantumSystem;
        use pecos_engines::noise::general::GeneralNoiseModel;
        use pecos_engines::quantum::StateVecEngine;

        let mut emission_model = BTreeMap::new();
        emission_model.insert("L".to_string(), 1.0);
        let noise = GeneralNoiseModel::builder()
            .with_p1(1.0)
            .with_p1_emission_ratio(1.0)
            .with_p1_emission_model(&emission_model)
            .build();
        let mut system = QuantumSystem::new(Box::new(noise), Box::new(StateVecEngine::new(1)));
        let mut builder = ByteMessage::quantum_operations_builder();
        builder.pz(&[0]);
        builder.rxy1q(
            Angle64::from_radians(std::f64::consts::FRAC_PI_2),
            Angle64::from_radians(3.0 * std::f64::consts::FRAC_PI_2),
            &[0],
        );
        builder.rz(Angle64::HALF_TURN, &[0]);
        builder.measure_leakages(&[0]);

        let result = system.process(builder.build()).expect("simulate leakage");

        assert_eq!(result.outcomes().expect("parse outcome"), vec![2]);
    }

    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn selene_native_commands_use_only_runtime_gates() {
        use pecos_core::gate_type::GateType;
        let runtime = crate::selene_runtimes::selene_simple_runtime().unwrap();
        let mut engine = QisEngine::with_runtime(Box::new(runtime));
        let lowered = engine
            .lower_operations_to_commands(&[
                Operation::AllocateQubit { id: 0 },
                Operation::AllocateQubit { id: 1 },
                Operation::AllocateQubit { id: 2 },
                QuantumOp::Reset(0).into(),
                QuantumOp::H(0).into(),
                QuantumOp::CX(0, 1).into(),
                QuantumOp::CCX(0, 1, 2).into(),
                QuantumOp::Measure(2, 0).into(),
            ])
            .unwrap();
        let gates = lowered.commands.quantum_ops().unwrap();
        assert!(gates.iter().any(|gate| gate.gate_type == GateType::RXY1Q));
        assert!(gates.iter().any(|gate| gate.gate_type == GateType::RZZ));
        assert!(gates.iter().all(|gate| matches!(
            gate.gate_type,
            GateType::RXY1Q | GateType::RZ | GateType::RZZ | GateType::PZ | GateType::MZ
        )));
    }

    #[test]
    fn test_direct_lowering_attaches_trace_metadata_to_next_gate() {
        let mut engine = QisEngine::with_runtime(Box::new(DummyRuntime::default()));
        let collector: OperationTraceStore = Arc::new(Mutex::new(Vec::new()));
        engine.set_operation_trace_collector(collector.clone());
        engine.begin_trace_shot();

        let mut metadata = TraceMetadata::new();
        metadata.insert(
            "source_label".to_string(),
            "szz_physical_prefix:H:X0:q0".to_string(),
        );
        metadata.insert("source_kind".to_string(), "szz_prefix".to_string());
        let ops = vec![
            Operation::TraceMetadata {
                metadata,
                qubit: None,
            },
            QuantumOp::H(0).into(),
            QuantumOp::Measure(0, 7).into(),
        ];

        let lowered = engine
            .operations_to_lowered_commands(&ops)
            .expect("convert ops to lowered commands");
        engine.trace_operations_chunk("unit_test", &ops, None, Some(&lowered));

        let in_memory = collector.lock().expect("collector lock");
        assert_eq!(in_memory.len(), 1);
        assert_eq!(in_memory[0].lowered_quantum_ops[0].gate_type, "H");
        assert_eq!(
            in_memory[0].lowered_quantum_ops[0]
                .metadata
                .get("source_label"),
            Some(&"szz_physical_prefix:H:X0:q0".to_string())
        );
        assert_eq!(in_memory[0].lowered_quantum_ops[1].gate_type, "MZ");
        assert!(in_memory[0].lowered_quantum_ops[1].metadata.is_empty());
    }

    #[derive(Clone, Default)]
    struct IdleLoweringRuntime {
        state: ClassicalState,
    }

    impl QisRuntime for IdleLoweringRuntime {
        fn load_interface(&mut self, _interface: OperationList) -> RuntimeResult<()> {
            Ok(())
        }

        fn execute_until_quantum(&mut self) -> RuntimeResult<Option<Vec<QuantumOp>>> {
            Ok(None)
        }

        fn provide_measurements(
            &mut self,
            _measurements: BTreeMap<usize, bool>,
        ) -> RuntimeResult<()> {
            Ok(())
        }

        fn get_classical_state(&self) -> &ClassicalState {
            &self.state
        }

        fn get_classical_state_mut(&mut self) -> &mut ClassicalState {
            &mut self.state
        }

        fn is_complete(&self) -> bool {
            true
        }

        fn num_qubits(&self) -> usize {
            1
        }

        fn supports_operation_lowering(&self) -> bool {
            true
        }

        fn lower_operations(&mut self, _operations: &[Operation]) -> RuntimeResult<Vec<QuantumOp>> {
            Ok(vec![
                QuantumOp::Idle(20e-9, 0),
                QuantumOp::H(0),
                QuantumOp::Measure(0, 17),
            ])
        }

        fn lower_operations_with_metadata(
            &mut self,
            _operations: &[Operation],
        ) -> RuntimeResult<Vec<LoweredQuantumOp>> {
            let mut idle_metadata = TraceMetadata::new();
            idle_metadata.insert("runtime_stage".to_string(), "scheduled_idle".to_string());
            Ok(vec![
                LoweredQuantumOp::new(QuantumOp::Idle(20e-9, 0), idle_metadata),
                QuantumOp::H(0).into(),
                QuantumOp::Measure(0, 17).into(),
            ])
        }

        fn drain_pending_operations(&mut self) -> RuntimeResult<Vec<QuantumOp>> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn test_operation_trace_chunk_includes_runtime_lowered_idles() {
        let mut engine = QisEngine::with_runtime(Box::new(IdleLoweringRuntime::default()));
        let collector: OperationTraceStore = Arc::new(Mutex::new(Vec::new()));
        engine.set_operation_trace_collector(collector.clone());
        engine.begin_trace_shot();

        let ops = vec![QuantumOp::H(0).into(), QuantumOp::Measure(0, 17).into()];
        let lowered = engine
            .lower_operations_to_commands(&ops)
            .expect("runtime lower ops to commands");
        engine.trace_operations_chunk("unit_test", &ops, None, Some(&lowered));

        let in_memory = collector.lock().expect("collector lock");
        assert_eq!(in_memory.len(), 1);
        assert_eq!(in_memory[0].lowered_quantum_ops[0].gate_type, "Idle");
        assert_eq!(in_memory[0].lowered_quantum_ops[0].params, vec![20e-9]);
        assert_eq!(in_memory[0].lowered_quantum_ops[0].qubits, vec![0]);
        assert_eq!(
            in_memory[0].lowered_quantum_ops[0]
                .metadata
                .get("runtime_stage"),
            Some(&"scheduled_idle".to_string())
        );
        assert_eq!(in_memory[0].lowered_quantum_ops[1].gate_type, "H");
        assert_eq!(in_memory[0].lowered_quantum_ops[2].gate_type, "MZ");
        assert_eq!(
            in_memory[0].lowered_quantum_ops[2].measurement_result_ids,
            vec![17]
        );
    }

    #[test]
    fn test_operations_to_bytemessage_accepts_implicit_static_qubit_handles() {
        let mut engine = QisEngine::with_runtime(Box::new(DummyRuntime::default()));
        let ops = vec![QuantumOp::H(0).into(), QuantumOp::Measure(0, 7).into()];

        let lowered_commands = engine
            .operations_to_lowered_commands(&ops)
            .expect("convert ops with implicit static handles");

        let lowered = lowered_commands
            .commands
            .quantum_ops()
            .expect("parse lowered commands");
        assert_eq!(lowered.len(), 2);
        assert_eq!(lowered[0].gate_type.to_string(), "H");
        assert_eq!(lowered[0].qubits.as_slice(), &[pecos_core::QubitId(0)]);
        assert_eq!(lowered[1].gate_type.to_string(), "MZ");
        assert_eq!(lowered[1].qubits.as_slice(), &[pecos_core::QubitId(0)]);
    }

    #[test]
    fn test_operations_to_bytemessage_rejects_use_after_release_without_reallocate() {
        let mut engine = QisEngine::with_runtime(Box::new(DummyRuntime::default()));
        let ops = vec![
            Operation::AllocateQubit { id: 0 },
            QuantumOp::H(0).into(),
            Operation::ReleaseQubit { id: 0 },
            QuantumOp::X(0).into(),
        ];

        let Err(err) = engine.operations_to_lowered_commands(&ops) else {
            panic!("released qubit reuse should error");
        };

        assert!(
            err.to_string().contains("not currently active"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_num_qubits_hint_is_physical_capacity_for_sparse_handles() {
        let mut engine = QisEngine::with_runtime(Box::new(DummyRuntime::default()));
        engine.set_num_qubits_hint(98);
        let ops = vec![
            Operation::AllocateQubit { id: 81 },
            Operation::AllocateQubit { id: 105 },
            QuantumOp::CX(81, 105).into(),
        ];

        let lowered_commands = engine
            .lower_operations_to_commands(&ops)
            .expect("sparse handles should map onto live physical slots");

        let lowered = lowered_commands
            .commands
            .quantum_ops()
            .expect("parse lowered commands");
        assert_eq!(lowered[0].qubits.as_slice(), &[pecos_core::QubitId(0)]);
        assert_eq!(lowered[1].qubits.as_slice(), &[pecos_core::QubitId(1)]);
        assert_eq!(
            lowered[2].qubits.as_slice(),
            &[pecos_core::QubitId(0), pecos_core::QubitId(1)]
        );
        assert_eq!(engine.num_physical_slots, 2);
        assert_eq!(engine.num_qubits(), 98);
    }

    #[test]
    fn test_worker_failure_does_not_certify_a_complete_trace() {
        let temp_dir = TempDir::new().expect("tempdir");
        let mut engine = QisEngine::with_runtime(Box::new(DummyRuntime::default()));
        engine.set_operation_trace_dir(temp_dir.path());
        engine.begin_trace_shot();
        engine.dynamic_state = Some(DynamicExecutionState {
            sync_handle: None,
            execution_complete: true,
            terminal_error: Some("dynamic QIS worker failed: interface crashed".to_string()),
            finalized: false,
            terminal_lowering_flushed: false,
        });

        let Err(err) = engine.continue_processing(ByteMessage::builder().build()) else {
            panic!("worker failure must fail the shot");
        };
        assert!(
            err.to_string().contains("dynamic QIS worker failed"),
            "unexpected error: {err}"
        );

        // Sticky: retrying must not complete the shot from the partial state.
        assert!(
            engine
                .continue_processing(ByteMessage::builder().build())
                .is_err()
        );

        // The failed shot must not have emitted the terminal trace marker.
        let wrote_terminal_marker = std::fs::read_dir(temp_dir.path())
            .expect("read trace dir")
            .any(|entry| {
                entry
                    .expect("dir entry")
                    .file_name()
                    .to_string_lossy()
                    .contains("trace_complete")
            });
        assert!(!wrote_terminal_marker);
    }

    /// Emits one late scheduled op on the FIRST drain, then reports empty --
    /// the shape a real lazily scheduling runtime shows after the check has
    /// consumed its held tail. A retry must still fail (sticky), because the
    /// second drain finds an innocently empty scheduler.
    #[derive(Clone, Default)]
    struct LateOpRuntime {
        state: ClassicalState,
        late_op: bool,
    }

    impl QisRuntime for LateOpRuntime {
        fn load_interface(&mut self, _interface: OperationList) -> RuntimeResult<()> {
            Ok(())
        }

        fn execute_until_quantum(&mut self) -> RuntimeResult<Option<Vec<QuantumOp>>> {
            Ok(None)
        }

        fn provide_measurements(
            &mut self,
            _measurements: BTreeMap<usize, bool>,
        ) -> RuntimeResult<()> {
            Ok(())
        }

        fn get_classical_state(&self) -> &ClassicalState {
            &self.state
        }

        fn get_classical_state_mut(&mut self) -> &mut ClassicalState {
            &mut self.state
        }

        fn is_complete(&self) -> bool {
            true
        }

        fn num_qubits(&self) -> usize {
            1
        }

        fn supports_operation_lowering(&self) -> bool {
            true
        }

        fn lower_operations(&mut self, _operations: &[Operation]) -> RuntimeResult<Vec<QuantumOp>> {
            Ok(Vec::new())
        }

        fn drain_pending_operations(&mut self) -> RuntimeResult<Vec<QuantumOp>> {
            if std::mem::replace(&mut self.late_op, false) {
                Ok(vec![QuantumOp::H(0)])
            } else {
                Ok(Vec::new())
            }
        }
    }

    /// Lowering runtime that deliberately does NOT override
    /// `drain_pending_operations`: it must be rejected by the fail-closed
    /// trait default, never certified drained.
    #[derive(Clone, Default)]
    struct NoDrainProtocolRuntime {
        state: ClassicalState,
    }

    impl QisRuntime for NoDrainProtocolRuntime {
        fn load_interface(&mut self, _interface: OperationList) -> RuntimeResult<()> {
            Ok(())
        }

        fn execute_until_quantum(&mut self) -> RuntimeResult<Option<Vec<QuantumOp>>> {
            Ok(None)
        }

        fn provide_measurements(
            &mut self,
            _measurements: BTreeMap<usize, bool>,
        ) -> RuntimeResult<()> {
            Ok(())
        }

        fn get_classical_state(&self) -> &ClassicalState {
            &self.state
        }

        fn get_classical_state_mut(&mut self) -> &mut ClassicalState {
            &mut self.state
        }

        fn is_complete(&self) -> bool {
            true
        }

        fn num_qubits(&self) -> usize {
            1
        }

        fn supports_operation_lowering(&self) -> bool {
            true
        }

        fn lower_operations(&mut self, _operations: &[Operation]) -> RuntimeResult<Vec<QuantumOp>> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn test_lowering_runtime_without_drain_protocol_fails_closed() {
        let mut engine = QisEngine::with_runtime(Box::new(NoDrainProtocolRuntime::default()));
        engine.dynamic_state = Some(DynamicExecutionState {
            sync_handle: None,
            execution_complete: true,
            terminal_error: None,
            finalized: false,
            terminal_lowering_flushed: false,
        });

        let Err(err) = engine.continue_processing(ByteMessage::builder().build()) else {
            panic!("a lowering runtime without the drain protocol must not certify");
        };
        assert!(
            err.to_string()
                .contains("does not implement drain_pending_operations"),
            "unexpected error: {err}"
        );
    }

    /// Runtime whose `shot_end` finalization fails: the completion gate must
    /// fail the shot before the terminal trace marker, and stay failed.
    #[derive(Clone, Default)]
    struct ShotEndFailRuntime {
        state: ClassicalState,
    }

    impl QisRuntime for ShotEndFailRuntime {
        fn load_interface(&mut self, _interface: OperationList) -> RuntimeResult<()> {
            Ok(())
        }

        fn execute_until_quantum(&mut self) -> RuntimeResult<Option<Vec<QuantumOp>>> {
            Ok(None)
        }

        fn provide_measurements(
            &mut self,
            _measurements: BTreeMap<usize, bool>,
        ) -> RuntimeResult<()> {
            Ok(())
        }

        fn get_classical_state(&self) -> &ClassicalState {
            &self.state
        }

        fn get_classical_state_mut(&mut self) -> &mut ClassicalState {
            &mut self.state
        }

        fn is_complete(&self) -> bool {
            true
        }

        fn num_qubits(&self) -> usize {
            1
        }

        fn shot_end(&mut self) -> RuntimeResult<crate::runtime::Shot> {
            Err(crate::runtime::RuntimeError::ExecutionError(
                "invalid final schedule".to_string(),
            ))
        }
    }

    #[test]
    fn test_failing_shot_end_blocks_certification_sticky() {
        let temp_dir = TempDir::new().expect("tempdir");
        let mut engine = QisEngine::with_runtime(Box::new(ShotEndFailRuntime::default()));
        engine.set_operation_trace_dir(temp_dir.path());
        engine.begin_trace_shot();
        engine.dynamic_state = Some(DynamicExecutionState {
            sync_handle: None,
            execution_complete: true,
            terminal_error: None,
            finalized: false,
            terminal_lowering_flushed: false,
        });

        let Err(err) = engine.continue_processing(ByteMessage::builder().build()) else {
            panic!("failing shot_end must fail the shot");
        };
        assert!(
            err.to_string().contains("runtime shot_end failed"),
            "unexpected error: {err}"
        );

        // Sticky across retry.
        assert!(
            engine
                .continue_processing(ByteMessage::builder().build())
                .is_err()
        );

        // No terminal trace marker for the failed shot.
        let wrote_terminal_marker = std::fs::read_dir(temp_dir.path())
            .expect("read trace dir")
            .any(|entry| {
                entry
                    .expect("dir entry")
                    .file_name()
                    .to_string_lossy()
                    .contains("trace_complete")
            });
        assert!(!wrote_terminal_marker);
    }

    /// Runtime whose `shot_end` fails and whose `reset` fails while
    /// `fail_reset` is set, for a reset after a failed shot.
    #[derive(Clone, Default)]
    struct ShotEndAndResetFailRuntime {
        state: ClassicalState,
        fail_reset: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    impl QisRuntime for ShotEndAndResetFailRuntime {
        fn load_interface(&mut self, _interface: OperationList) -> RuntimeResult<()> {
            Ok(())
        }

        fn execute_until_quantum(&mut self) -> RuntimeResult<Option<Vec<QuantumOp>>> {
            Ok(None)
        }

        fn provide_measurements(
            &mut self,
            _measurements: BTreeMap<usize, bool>,
        ) -> RuntimeResult<()> {
            Ok(())
        }

        fn get_classical_state(&self) -> &ClassicalState {
            &self.state
        }

        fn get_classical_state_mut(&mut self) -> &mut ClassicalState {
            &mut self.state
        }

        fn is_complete(&self) -> bool {
            true
        }

        fn num_qubits(&self) -> usize {
            1
        }

        fn shot_end(&mut self) -> RuntimeResult<crate::runtime::Shot> {
            Err(crate::runtime::RuntimeError::ExecutionError(
                "invalid final schedule".to_string(),
            ))
        }

        fn reset(&mut self) -> RuntimeResult<()> {
            if self.fail_reset.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(crate::runtime::RuntimeError::ExecutionError(
                    "exit failed".to_string(),
                ));
            }
            Ok(())
        }
    }

    /// Interface whose `reset` fails while `fail_reset` is set.
    #[derive(Clone, Default)]
    struct FlakyResetInterface {
        fail_reset: std::sync::Arc<std::sync::atomic::AtomicBool>,
        dynamic: bool,
    }

    impl crate::QisInterface for FlakyResetInterface {
        fn load_program(
            &mut self,
            _: &[u8],
            _: crate::ProgramFormat,
        ) -> Result<(), crate::InterfaceError> {
            Ok(())
        }
        fn collect_operations(&mut self) -> Result<OperationList, crate::InterfaceError> {
            Ok(OperationList::new())
        }
        fn execute_with_measurements(
            &mut self,
            _: BTreeMap<usize, bool>,
        ) -> Result<OperationList, crate::InterfaceError> {
            Ok(OperationList::new())
        }
        fn name(&self) -> &'static str {
            "flaky-reset"
        }
        fn reset(&mut self) -> Result<(), crate::InterfaceError> {
            if self.fail_reset.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(crate::InterfaceError::Other(
                    "interface reset failed".into(),
                ));
            }
            Ok(())
        }
        fn supports_dynamic(&self) -> bool {
            self.dynamic
        }
    }

    // #1040: a failed reset after a failed shot must not let `get_results`
    // certify that shot, through every reset entry point; a successful reset
    // through the same entry point recovers fully.
    #[test]
    fn failed_reset_after_failed_shot_never_certifies_results() {
        use std::sync::atomic::Ordering;
        type ResetFn = fn(&mut QisEngine) -> Result<(), PecosError>;
        let resets: [(&str, ResetFn); 3] = [
            ("ControlEngine", |engine| ControlEngine::reset(engine)),
            ("Engine", |engine| Engine::reset(engine)),
            ("ClassicalEngine", |engine| ClassicalEngine::reset(engine)),
        ];
        for (name, reset) in resets {
            let runtime = ShotEndAndResetFailRuntime::default();
            let fail_reset = std::sync::Arc::clone(&runtime.fail_reset);
            fail_reset.store(true, Ordering::SeqCst);
            let mut engine =
                QisEngine::new(Box::new(FlakyResetInterface::default()), Box::new(runtime));
            engine.measurement_results.insert(0, 1);
            engine.dynamic_state = Some(DynamicExecutionState {
                sync_handle: None,
                execution_complete: true,
                terminal_error: None,
                finalized: false,
                terminal_lowering_flushed: false,
            });

            assert!(
                engine
                    .continue_processing(ByteMessage::builder().build())
                    .is_err()
            );
            assert!(engine.get_results().is_err());

            let Err(reset_error) = reset(&mut engine) else {
                panic!("{name}: the runtime reset must fail");
            };
            assert!(
                reset_error.to_string().contains("exit failed"),
                "{name}: {reset_error}"
            );
            let Err(results_error) = engine.get_results() else {
                panic!("{name}: a failed reset must not certify the failed shot's results");
            };
            assert!(
                results_error.to_string().contains("exit failed"),
                "{name}: {results_error}"
            );
            assert!(engine.clone().get_results().is_err(), "{name}");

            fail_reset.store(false, Ordering::SeqCst);
            reset(&mut engine).unwrap();
            let shot = engine.get_results().unwrap();
            assert!(shot.data.is_empty(), "{name}: {shot:?}");
        }
    }

    // An interface reset failure stays latched through `start()`, which resets
    // only the runtime, until a complete reset succeeds.
    #[test]
    fn interface_reset_failure_blocks_start_until_reset_succeeds() {
        use std::sync::atomic::Ordering;
        let interface = FlakyResetInterface::default();
        let fail_reset = std::sync::Arc::clone(&interface.fail_reset);
        fail_reset.store(true, Ordering::SeqCst);
        let mut engine = QisEngine::new(
            Box::new(interface),
            Box::new(ShotEndAndResetFailRuntime::default()),
        );

        let Err(reset_error) = ClassicalEngine::reset(&mut engine) else {
            panic!("the interface reset must fail");
        };
        assert!(
            reset_error.to_string().contains("interface reset failed"),
            "{reset_error}"
        );
        let Err(start_error) = engine.start(()) else {
            panic!("start must refuse while a reset failure is latched");
        };
        assert!(
            start_error.to_string().contains("interface reset failed"),
            "{start_error}"
        );
        assert!(engine.get_results().is_err());

        fail_reset.store(false, Ordering::SeqCst);
        ControlEngine::reset(&mut engine).unwrap();
        assert!(engine.get_results().unwrap().data.is_empty());
        // Unlatched, start proceeds to its own checks again.
        let Err(start_error) = engine.start(()) else {
            panic!("this test interface cannot run a shot");
        };
        assert!(
            start_error.to_string().contains("dynamic-capable"),
            "{start_error}"
        );
    }

    // A failed runtime reset inside `start()` is latched, and `start()` itself
    // never clears the latch; a complete reset does.
    #[test]
    fn start_runtime_reset_failure_is_latched() {
        use std::sync::atomic::Ordering;
        let runtime = ShotEndAndResetFailRuntime::default();
        let fail_reset = std::sync::Arc::clone(&runtime.fail_reset);
        fail_reset.store(true, Ordering::SeqCst);
        let mut engine = QisEngine::new(
            Box::new(FlakyResetInterface {
                dynamic: true,
                ..FlakyResetInterface::default()
            }),
            Box::new(runtime),
        );
        engine.measurement_results.insert(0, 1);

        let Err(first) = engine.start(()) else {
            panic!("start's runtime reset must fail");
        };
        assert!(first.to_string().contains("exit failed"), "{first}");
        assert!(engine.get_results().is_err());

        // The runtime is healthy again, but start alone must not unlatch.
        fail_reset.store(false, Ordering::SeqCst);
        let Err(second) = engine.start(()) else {
            panic!("start must refuse while a reset failure is latched");
        };
        assert!(second.to_string().contains("exit failed"), "{second}");

        Engine::reset(&mut engine).unwrap();
        assert!(engine.get_results().unwrap().data.is_empty());
    }

    /// Counts `shot_end` invocations to pin one-shot finalization.
    #[derive(Clone, Default)]
    struct CountingShotEndRuntime {
        state: ClassicalState,
        shot_end_calls: Arc<Mutex<u32>>,
    }

    impl QisRuntime for CountingShotEndRuntime {
        fn load_interface(&mut self, _interface: OperationList) -> RuntimeResult<()> {
            Ok(())
        }

        fn execute_until_quantum(&mut self) -> RuntimeResult<Option<Vec<QuantumOp>>> {
            Ok(None)
        }

        fn provide_measurements(
            &mut self,
            _measurements: BTreeMap<usize, bool>,
        ) -> RuntimeResult<()> {
            Ok(())
        }

        fn get_classical_state(&self) -> &ClassicalState {
            &self.state
        }

        fn get_classical_state_mut(&mut self) -> &mut ClassicalState {
            &mut self.state
        }

        fn is_complete(&self) -> bool {
            true
        }

        fn num_qubits(&self) -> usize {
            1
        }

        fn shot_end(&mut self) -> RuntimeResult<crate::runtime::Shot> {
            *self.shot_end_calls.lock().expect("counter lock") += 1;
            Ok(crate::runtime::Shot {
                measurements: self.state.measurements.clone(),
                registers: self.state.registers.clone(),
                metadata: BTreeMap::new(),
            })
        }
    }

    #[test]
    fn test_successful_finalization_is_one_shot_across_redundant_polls() {
        let temp_dir = TempDir::new().expect("tempdir");
        let counter = Arc::new(Mutex::new(0u32));
        let runtime = CountingShotEndRuntime {
            shot_end_calls: counter.clone(),
            ..CountingShotEndRuntime::default()
        };
        let mut engine = QisEngine::with_runtime(Box::new(runtime));
        engine.set_operation_trace_dir(temp_dir.path());
        engine.begin_trace_shot();
        engine.dynamic_state = Some(DynamicExecutionState {
            sync_handle: None,
            execution_complete: true,
            terminal_error: None,
            finalized: false,
            terminal_lowering_flushed: false,
        });

        // First completion certifies the shot; a redundant poll must neither
        // call shot_end again nor emit another terminal marker.
        assert!(
            engine
                .continue_processing(ByteMessage::builder().build())
                .is_ok()
        );
        assert!(
            engine
                .continue_processing(ByteMessage::builder().build())
                .is_ok()
        );
        assert_eq!(*counter.lock().expect("counter lock"), 1);

        let terminal_markers = std::fs::read_dir(temp_dir.path())
            .expect("read trace dir")
            .filter(|entry| {
                entry
                    .as_ref()
                    .expect("dir entry")
                    .file_name()
                    .to_string_lossy()
                    .contains("trace_complete")
            })
            .count();
        assert_eq!(terminal_markers, 1);
    }

    /// Errors on every lowering call: the final-tail lowering failure must
    /// latch sticky, because the tail was already consumed from the pending
    /// queue and a retry would otherwise certify a trace with it deleted.
    #[derive(Clone, Default)]
    struct LoweringFailRuntime {
        state: ClassicalState,
    }

    impl QisRuntime for LoweringFailRuntime {
        fn load_interface(&mut self, _interface: OperationList) -> RuntimeResult<()> {
            Ok(())
        }

        fn execute_until_quantum(&mut self) -> RuntimeResult<Option<Vec<QuantumOp>>> {
            Ok(None)
        }

        fn provide_measurements(
            &mut self,
            _measurements: BTreeMap<usize, bool>,
        ) -> RuntimeResult<()> {
            Ok(())
        }

        fn get_classical_state(&self) -> &ClassicalState {
            &self.state
        }

        fn get_classical_state_mut(&mut self) -> &mut ClassicalState {
            &mut self.state
        }

        fn is_complete(&self) -> bool {
            true
        }

        fn num_qubits(&self) -> usize {
            1
        }

        fn supports_operation_lowering(&self) -> bool {
            true
        }

        fn lower_operations(&mut self, _operations: &[Operation]) -> RuntimeResult<Vec<QuantumOp>> {
            Err(crate::runtime::RuntimeError::ExecutionError(
                "scheduler rejected the batch".to_string(),
            ))
        }

        fn drain_pending_operations(&mut self) -> RuntimeResult<Vec<QuantumOp>> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn test_final_tail_lowering_failure_is_sticky_and_never_certifies() {
        let temp_dir = TempDir::new().expect("tempdir");
        let mut engine = QisEngine::with_runtime(Box::new(LoweringFailRuntime::default()));
        engine.set_operation_trace_dir(temp_dir.path());
        engine.begin_trace_shot();
        engine.dynamic_state = Some(DynamicExecutionState {
            sync_handle: None,
            execution_complete: true,
            terminal_error: None,
            finalized: false,
            terminal_lowering_flushed: false,
        });
        engine.pending_dynamic_ops = vec![QuantumOp::H(0).into()];

        let Err(err) = engine.continue_processing(ByteMessage::builder().build()) else {
            panic!("final-tail lowering failure must fail the shot");
        };
        assert!(
            err.to_string().contains("failed to lower operations"),
            "unexpected error: {err}"
        );

        // Sticky: the tail was consumed by mem::take, so a retry sees an
        // empty pending queue -- it must STILL fail, not certify a trace
        // with the tail deleted.
        let Err(retry_err) = engine.continue_processing(ByteMessage::builder().build()) else {
            panic!("retry after tail-lowering failure must not certify the shot");
        };
        assert!(
            retry_err.to_string().contains("failed to lower operations"),
            "unexpected retry error: {retry_err}"
        );

        let wrote_terminal_marker = std::fs::read_dir(temp_dir.path())
            .expect("read trace dir")
            .any(|entry| {
                entry
                    .expect("dir entry")
                    .file_name()
                    .to_string_lossy()
                    .contains("trace_complete")
            });
        assert!(!wrote_terminal_marker);
    }

    #[test]
    fn test_undrained_runtime_scheduler_fails_the_shot_sticky_across_retry() {
        let temp_dir = TempDir::new().expect("tempdir");
        let runtime = LateOpRuntime {
            late_op: true,
            ..LateOpRuntime::default()
        };
        let mut engine = QisEngine::with_runtime(Box::new(runtime));
        engine.set_operation_trace_dir(temp_dir.path());
        engine.begin_trace_shot();
        engine.dynamic_state = Some(DynamicExecutionState {
            sync_handle: None,
            execution_complete: true,
            terminal_error: None,
            finalized: false,
            terminal_lowering_flushed: false,
        });

        let Err(err) = engine.continue_processing(ByteMessage::builder().build()) else {
            panic!("late scheduler operations must fail the shot");
        };
        assert!(
            err.to_string().contains("after the final lowered batch"),
            "unexpected error: {err}"
        );

        // Sticky: the drain consumed the late op, so a retry now finds an
        // empty scheduler -- it must STILL fail, not certify the shot.
        let Err(retry_err) = engine.continue_processing(ByteMessage::builder().build()) else {
            panic!("retry after drain failure must not certify the shot");
        };
        assert!(
            retry_err
                .to_string()
                .contains("after the final lowered batch"),
            "unexpected retry error: {retry_err}"
        );

        // The failed shot must not have emitted the terminal trace marker.
        let wrote_terminal_marker = std::fs::read_dir(temp_dir.path())
            .expect("read trace dir")
            .any(|entry| {
                entry
                    .expect("dir entry")
                    .file_name()
                    .to_string_lossy()
                    .contains("trace_complete")
            });
        assert!(!wrote_terminal_marker);
    }

    #[test]
    fn test_qubit_hint_rejects_too_many_live_physical_slots() {
        let mut engine = QisEngine::with_runtime(Box::new(DummyRuntime::default()));
        engine.set_num_qubits_hint(1);
        let ops = vec![
            Operation::AllocateQubit { id: 81 },
            Operation::AllocateQubit { id: 105 },
        ];

        let Err(err) = engine.operations_to_lowered_commands(&ops) else {
            panic!("allocating beyond the physical qubit hint should error");
        };

        assert!(
            err.to_string().contains("more than the configured 1"),
            "unexpected error: {err}"
        );
    }
}

#[cfg(test)]
mod scheduled_completion_tests {
    use super::*;
    use crate::runtime::{ClassicalState, Result as RuntimeResult};
    use crate::scheduled::{RuntimeScheduledOp as Op, ScheduledBatch, ScheduledMeasurement};
    // Synthetic compiler/interface: the runtime fixture supplies terminal batches.
    // Construction still runs through the public builder and its build() forwarding.
    #[derive(Clone)]
    struct TerminalInterface;
    impl crate::QisInterface for TerminalInterface {
        fn load_program(
            &mut self,
            _: &[u8],
            _: crate::ProgramFormat,
        ) -> Result<(), crate::InterfaceError> {
            Ok(())
        }
        fn collect_operations(&mut self) -> Result<OperationList, crate::InterfaceError> {
            Ok(OperationList::new())
        }
        fn execute_with_measurements(
            &mut self,
            _: BTreeMap<usize, bool>,
        ) -> Result<OperationList, crate::InterfaceError> {
            Ok(OperationList::new())
        }
        fn name(&self) -> &'static str {
            "synthetic-terminal"
        }
        fn reset(&mut self) -> Result<(), crate::InterfaceError> {
            Ok(())
        }
    }
    impl QisInterfaceBuilder for TerminalInterface {
        fn build_from_qis_program(
            &self,
            _: pecos_programs::Qis,
        ) -> Result<OperationList, PecosError> {
            Ok(OperationList::new())
        }
        fn build_from_interface(
            &self,
            interface: OperationList,
        ) -> Result<OperationList, PecosError> {
            Ok(interface)
        }
        fn name(&self) -> &'static str {
            "synthetic-terminal"
        }
        fn create_dynamic_interface_from_qis(
            &self,
            _: pecos_programs::Qis,
        ) -> Result<crate::BoxedInterface, PecosError> {
            Ok(Box::new(self.clone()))
        }
    }
    #[derive(Clone, Default)]
    enum TerminalOutput {
        #[default]
        Feedback,
        Padding,
    }

    #[derive(Clone, Default)]
    struct FeedbackTail {
        state: ClassicalState,
        stage: usize,
        fail_tail: bool,
        output: TerminalOutput,
        with_event: bool,
        ended: bool,
    }
    impl QisRuntime for FeedbackTail {
        fn load_interface(&mut self, _: OperationList) -> RuntimeResult<()> {
            Ok(())
        }
        fn execute_until_quantum(&mut self) -> RuntimeResult<Option<Vec<QuantumOp>>> {
            Ok(None)
        }
        fn provide_measurements(&mut self, values: BTreeMap<usize, bool>) -> RuntimeResult<()> {
            if self.with_event {
                assert_eq!(values.get(&0), Some(&true));
            }
            self.stage = 3;
            Ok(())
        }
        fn get_classical_state(&self) -> &ClassicalState {
            &self.state
        }
        fn get_classical_state_mut(&mut self) -> &mut ClassicalState {
            &mut self.state
        }
        fn is_complete(&self) -> bool {
            true
        }
        fn num_qubits(&self) -> usize {
            1
        }
        fn shot_end(&mut self) -> RuntimeResult<crate::runtime::Shot> {
            if std::mem::replace(&mut self.ended, true) {
                return Err(crate::runtime::RuntimeError::ExecutionError(
                    "shot ended twice".into(),
                ));
            }
            Ok(crate::runtime::Shot::default())
        }
        fn lower_scheduled_operations(
            &mut self,
            operations: &[Operation],
        ) -> RuntimeResult<Vec<ScheduledBatch>> {
            assert!(operations.contains(&QuantumOp::Measure(0, 0).into()));
            self.stage = 1;
            Ok(vec![])
        }
        fn drain_pending_scheduled_operations(&mut self) -> RuntimeResult<Vec<ScheduledBatch>> {
            if self.ended {
                return Err(crate::runtime::RuntimeError::ExecutionError(
                    "drain after shot completion".into(),
                ));
            }
            if matches!(self.output, TerminalOutput::Padding) {
                let index = self.stage;
                self.stage += 1;
                return Ok(vec![ScheduledBatch {
                    runtime_shot_id: 0,
                    batch_index: index,
                    start_time_nanos: index as u64,
                    duration_nanos: 0,
                    operations: vec![Op::Rz {
                        qubit_id: 0,
                        theta: 0.0,
                    }],
                    measurements: vec![],
                }]);
            }
            let (mut ops, mut measurements, index) = match self.stage {
                1 => {
                    self.stage = 2;
                    (
                        vec![Op::Measure {
                            qubit_id: 0,
                            result_id: 0,
                        }],
                        vec![ScheduledMeasurement {
                            operation_index: 0,
                            runtime_result: 0,
                            program_result: 0,
                            leakage_aware: false,
                        }],
                        0,
                    )
                }
                3 => {
                    if self.fail_tail {
                        return Err(crate::runtime::RuntimeError::ExecutionError(
                            "tail drain failed".into(),
                        ));
                    }
                    self.stage = 4;
                    (
                        vec![Op::Rz {
                            qubit_id: 0,
                            theta: 1.0,
                        }],
                        vec![],
                        1,
                    )
                }
                _ => return Ok(vec![]),
            };
            if self.with_event && index == 0 {
                ops.insert(
                    0,
                    Op::Custom {
                        tag: 901,
                        data: vec![1],
                    },
                );
                measurements[0].operation_index = 1;
            }
            Ok(vec![ScheduledBatch {
                runtime_shot_id: 0,
                batch_index: index,
                start_time_nanos: 0,
                duration_nanos: 0,
                operations: ops,
                measurements,
            }])
        }
    }
    #[test]
    fn scheduled_terminal_padding_without_feedback_fails_promptly() {
        for mode in [ScheduledTransport::V3, ScheduledTransport::V4] {
            let mut engine = QisEngine::with_runtime(Box::new(FeedbackTail {
                output: TerminalOutput::Padding,
                ..Default::default()
            }));
            engine.scheduled_transport = mode;
            engine.interface = Some(Box::new(TerminalInterface));
            engine.dynamic_state = Some(DynamicExecutionState {
                sync_handle: None,
                execution_complete: true,
                terminal_error: None,
                finalized: false,
                terminal_lowering_flushed: false,
            });
            assert!(matches!(
                engine
                    .continue_processing(ByteMessage::outcomes_builder().build())
                    .unwrap(),
                EngineStage::NeedsProcessing(_)
            ));
            engine.provide_measurements_terminal(&[]).unwrap();
            let error = engine
                .continue_processing(ByteMessage::outcomes_builder().build())
                .err()
                .expect("padding must fail on round two")
                .to_string();
            assert!(
                error.contains("without measurement feedback") && error.contains("round 2"),
                "{error}"
            );
            assert!(engine.get_results().is_err());
            engine.reset_all().unwrap();
            assert_eq!(engine.scheduled_drain_round, 0);
            assert!(!engine.scheduled_drain_feedback);
        }
    }

    #[test]
    fn scheduled_gate_only_final_tail_completes_without_feedback() {
        for mode in [ScheduledTransport::V3, ScheduledTransport::V4] {
            let mut engine = QisEngine::with_runtime(Box::new(FeedbackTail {
                stage: 3,
                ..Default::default()
            }));
            engine.scheduled_transport = mode;
            engine.dynamic_state = Some(DynamicExecutionState {
                sync_handle: None,
                execution_complete: true,
                terminal_error: None,
                finalized: false,
                terminal_lowering_flushed: false,
            });
            assert!(matches!(
                engine
                    .continue_processing(ByteMessage::outcomes_builder().build())
                    .unwrap(),
                EngineStage::NeedsProcessing(_)
            ));
            assert_eq!(engine.scheduled_drain_round, 1);
            assert!(matches!(
                engine
                    .continue_processing(ByteMessage::outcomes_builder().build())
                    .unwrap(),
                EngineStage::Complete(_)
            ));
            assert_eq!(engine.scheduled_drain_round, 2);
            assert!(engine.get_results().is_ok());
        }
    }

    #[test]
    fn terminal_feedback_must_drain_newly_ready_tail() {
        use pecos_engines::noise::IntoNoiseModel;
        use pecos_engines::runtime_frame::ShotContext;
        use pecos_engines::scheduled_frame::ScheduledIdleZ;
        use pecos_engines::{StateVecEngine, quantum_system::QuantumSystem};
        let mut engine = QisEngine::with_runtime(Box::new(FeedbackTail::default()));
        engine.scheduled_transport = ScheduledTransport::V3;
        engine.dynamic_state = Some(DynamicExecutionState {
            sync_handle: None,
            execution_complete: true,
            terminal_error: None,
            finalized: false,
            terminal_lowering_flushed: false,
        });
        engine
            .lower_operations_terminal(&[QuantumOp::Measure(0, 0).into()])
            .unwrap();
        // Execute the terminal measurement and feedback-triggered tail in a real owner.
        let mut quantum = QuantumSystem::new(
            ScheduledIdleZ::new(1, 0.0, 0.0, 0.0)
                .unwrap()
                .into_noise_model(),
            Box::new(StateVecEngine::new(1)),
        );
        quantum
            .begin_shot(ShotContext {
                run: 1,
                worker: 0,
                shot: 0,
            })
            .unwrap();
        let EngineStage::NeedsProcessing(initial) = engine
            .continue_processing(ByteMessage::outcomes_builder().build())
            .unwrap()
        else {
            panic!("terminal drain must emit its measurement");
        };
        assert_eq!(engine.pending_measurements[&0].outstanding, 1);
        assert_eq!(engine.pending_measurements[&0].unemitted, 0);
        let measured = quantum.process(initial).unwrap();
        assert_eq!(measured.outcomes().unwrap(), vec![0]);
        let EngineStage::NeedsProcessing(commands) = engine.continue_processing(measured).unwrap()
        else {
            panic!("newly ready native tail was silently skipped");
        };
        let reply = quantum.process(commands).unwrap();
        assert!(matches!(
            engine.continue_processing(reply).unwrap(),
            EngineStage::Complete(_)
        ));
        assert_eq!(engine.pending_measurements.len(), 0);
        // A repeated completion poll must not touch an already-ended runtime.
        assert!(matches!(
            engine
                .continue_processing(ByteMessage::outcomes_builder().build())
                .unwrap(),
            EngineStage::Complete(_)
        ));
    }

    #[test]
    fn event_aware_terminal_batches_return_feedback_and_finish_once() {
        use pecos_core::Angle64;
        use pecos_engines::noise::IntoNoiseModel;
        use pecos_engines::runtime_frame::ShotContext;
        use pecos_engines::scheduled_events::{
            ScheduledBatchAdapter, ScheduledEventBatch, ScheduledEventIdleZ, ScheduledEventOp,
            ScheduledGateBuffer,
        };
        use pecos_engines::scheduled_frame::ScheduledIdleZ;
        use pecos_engines::{Gate, StateVecEngine, quantum_system::QuantumSystem};
        struct Flip;
        impl ScheduledBatchAdapter for Flip {
            fn validate(&self, b: &ScheduledEventBatch) -> Result<(), PecosError> {
                for op in &b.operations {
                    if let ScheduledEventOp::Custom { tag, payload } = op
                        && (*tag != 901 || payload.as_slice() != [1])
                    {
                        return Err(PecosError::Input("unsupported synthetic event".into()));
                    }
                }
                Ok(())
            }
            fn translate(
                &mut self,
                b: &ScheduledEventBatch,
                out: &mut ScheduledGateBuffer<'_>,
            ) -> Result<(), PecosError> {
                for op in &b.operations {
                    out.push(match op {
                        ScheduledEventOp::Gate(g) => g.as_ref().clone(),
                        ScheduledEventOp::Custom { .. } => Gate::rxy1q(
                            Angle64::from_radians(std::f64::consts::PI),
                            Angle64::from_radians(0.0),
                            &[0],
                        ),
                    })?;
                }
                Ok(())
            }
        }
        use pecos_engines::ClassicalControlEngineBuilder;
        let mut engine = crate::qis_engine()
            .interface(TerminalInterface)
            .program(pecos_programs::Qis::from_string(
                "define void @qmain() { ret void }",
            ))
            .runtime(FeedbackTail {
                with_event: true,
                ..FeedbackTail::default()
            })
            .scheduled_event_batches(true)
            .build()
            .unwrap();
        engine.dynamic_state = Some(DynamicExecutionState {
            sync_handle: None,
            execution_complete: true,
            terminal_error: None,
            finalized: false,
            terminal_lowering_flushed: false,
        });
        engine
            .lower_operations_terminal(&[QuantumOp::Measure(0, 0).into()])
            .unwrap();
        let mut quantum = QuantumSystem::new(
            ScheduledEventIdleZ::new(ScheduledIdleZ::new(1, 0.0, 0.0, 0.0).unwrap(), |_| {
                Ok(Box::new(Flip))
            })
            .into_noise_model(),
            Box::new(StateVecEngine::new(1)),
        );
        quantum
            .begin_shot(ShotContext {
                run: 1,
                worker: 0,
                shot: 0,
            })
            .unwrap();
        let EngineStage::NeedsProcessing(initial) = engine
            .continue_processing(ByteMessage::outcomes_builder().build())
            .unwrap()
        else {
            panic!("expected initial batch")
        };
        let reply = quantum.process(initial).unwrap();
        assert_eq!(reply.outcomes().unwrap(), vec![1]);
        let EngineStage::NeedsProcessing(tail) = engine.continue_processing(reply).unwrap() else {
            panic!("expected feedback tail")
        };
        let reply = quantum.process(tail).unwrap();
        assert!(matches!(
            engine.continue_processing(reply).unwrap(),
            EngineStage::Complete(_)
        ));
        assert!(matches!(
            engine
                .continue_processing(ByteMessage::outcomes_builder().build())
                .unwrap(),
            EngineStage::Complete(_)
        ));
    }

    #[test]
    fn terminal_feedback_drain_failure_stays_latched() {
        let mut engine = QisEngine::with_runtime(Box::new(FeedbackTail {
            fail_tail: true,
            ..FeedbackTail::default()
        }));
        engine.scheduled_transport = ScheduledTransport::V3;
        engine.dynamic_state = Some(DynamicExecutionState {
            sync_handle: None,
            execution_complete: true,
            terminal_error: None,
            finalized: false,
            terminal_lowering_flushed: false,
        });
        engine
            .lower_operations_terminal(&[QuantumOp::Measure(0, 0).into()])
            .unwrap();
        engine
            .continue_processing(ByteMessage::outcomes_builder().build())
            .unwrap();
        let error = engine
            .continue_processing(ByteMessage::outcomes_builder().add_outcomes(&[0]).build())
            .err()
            .expect("tail drain must fail");
        assert!(error.to_string().contains("tail drain failed"));
        assert!(
            engine
                .continue_processing(ByteMessage::outcomes_builder().build())
                .is_err()
        );
        assert!(!engine.dynamic_state.as_ref().unwrap().finalized);
    }
}

#[cfg(all(test, feature = "selene-runtimes"))]
#[path = "ccengine_terminal_tests.rs"]
mod terminal_lowering_tests;

#[cfg(test)]
#[path = "ccengine_lifecycle_tests.rs"]
mod lifecycle_tests;
