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
//! 2. The main thread returns operations via `generate_commands()`
//! 3. `continue_processing()` receives measurements and signals the worker to continue
//! 4. The worker resumes with the measurement results available

use crate::program::QisInterfaceBuilder;
use crate::qis_interface::{BoxedInterface, DynamicSyncHandle, ProgramFormat};
use crate::runtime::QisRuntime;
use log::{debug, warn};
use pecos_core::Angle64;
use pecos_core::prelude::PecosError;
use pecos_engines::noise::utils::NoiseUtils;
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
}

/// Shared in-memory store for traced QIS operation batches.
pub type OperationTraceStore = Arc<Mutex<Vec<OperationTraceChunk>>>;

/// Result from worker thread - returns both the operations and the interface.
///
/// The error arm also carries the interface back when the worker still holds
/// it, so one failed shot does not permanently strip the engine of its
/// interface. `None` means the interface was genuinely lost (worker died).
type WorkerResult = Result<(OperationList, BoxedInterface), (String, Option<BoxedInterface>)>;

/// Simulator commands plus one metadata record per lowered quantum gate.
struct LoweredCommandBatch {
    commands: ByteMessage,
    gate_metadata: Vec<TraceMetadata>,
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
    work_tx: Sender<DynamicWorkItem>,
    /// Channel to receive results from the worker (wrapped in Mutex for Sync)
    result_rx: Mutex<Receiver<WorkerResult>>,
    /// Thread handle (joined on drop)
    _handle: JoinHandle<()>,
}

impl PersistentDynamicWorker {
    /// Create a new persistent dynamic worker thread
    fn new() -> Self {
        let (work_tx, work_rx) = mpsc::channel::<DynamicWorkItem>();
        let (result_tx, result_rx) = mpsc::channel::<WorkerResult>();

        let handle = std::thread::Builder::new()
            .name("pecos-dynamic-worker".to_string())
            .spawn(move || {
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
                        (Ok(_), Err(e)) => Err((
                            format!("worker teardown (disable_dynamic_mode) failed: {e}"),
                            Some(interface),
                        )),
                        (Err(e), _) => Err((e.to_string(), Some(interface))),
                    };

                    if result_tx.send(send_result).is_err() {
                        // Main thread dropped receiver, exit
                        debug!("Persistent worker: result channel closed, exiting");
                        break;
                    }
                }
                debug!("Persistent dynamic worker exiting");
            })
            .expect("Failed to spawn persistent dynamic worker thread");

        Self {
            work_tx,
            result_rx: Mutex::new(result_rx),
            _handle: handle,
        }
    }

    /// Send a work item to the persistent worker
    fn execute(&self, interface: BoxedInterface) -> Result<(), PecosError> {
        self.work_tx
            .send(DynamicWorkItem { interface })
            .map_err(|_| PecosError::Generic("Persistent worker thread died".to_string()))
    }

    /// Receive the result from the persistent worker (blocking)
    #[allow(dead_code)]
    fn recv_result(&self) -> Result<WorkerResult, PecosError> {
        self.result_rx
            .lock()
            .map_err(|_| PecosError::Generic("Result receiver lock poisoned".to_string()))?
            .recv()
            .map_err(|_| PecosError::Generic("Persistent worker thread died".to_string()))
    }

    /// Try to receive a result without blocking.
    ///
    /// Worker death (channel disconnect) and a poisoned lock are surfaced as
    /// worker errors rather than `None`: `None` must only ever mean "still
    /// running", or the engine would poll an already-dead worker forever.
    fn try_recv_result(&self) -> Option<WorkerResult> {
        let Ok(rx) = self.result_rx.lock() else {
            return Some(Err((
                "dynamic worker result lock poisoned".to_string(),
                None,
            )));
        };
        match rx.try_recv() {
            Ok(result) => Some(result),
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => Some(Err((
                "dynamic worker thread died before returning a result".to_string(),
                None,
            ))),
        }
    }
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

    /// Whether we've started processing
    started: bool,

    /// Tracking measurement result IDs for the current batch
    measurement_mapping: Vec<usize>,

    /// Stored measurement results for `get_results()`
    measurement_results: BTreeMap<usize, bool>,

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
}

impl QisEngine {
    fn parse_measurement_outcomes(message: &ByteMessage) -> Result<Vec<usize>, PecosError> {
        message
            .outcomes()
            .map(|outcomes| outcomes.into_iter().map(|value| value as usize).collect())
            .map_err(|e| PecosError::Generic(format!("Failed to parse measurements: {e}")))
    }

    fn map_measurements(
        measurement_mapping: &[usize],
        measurements: &[usize],
    ) -> Vec<(usize, bool)> {
        measurement_mapping
            .iter()
            .copied()
            .zip(measurements.iter().copied())
            .map(|(result_id, value)| (result_id, value != 0))
            .collect()
    }

    fn store_measurement_updates(&mut self, updates: &[(usize, bool)]) {
        for &(result_id, value) in updates {
            self.measurement_results.insert(result_id, value);
            debug!("QisEngine: Stored measurement result_id={result_id}, value={value}");
        }
    }

    fn provide_measurement_updates_to_runtime(
        &mut self,
        updates: &[(usize, bool)],
    ) -> Result<(), PecosError> {
        if updates.is_empty() {
            return Ok(());
        }
        let measurement_map: BTreeMap<usize, bool> = updates.iter().copied().collect();
        self.runtime
            .provide_measurements(measurement_map)
            .map_err(|e| PecosError::Generic(format!("Failed to provide measurements: {e}")))
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
            started: false,
            measurement_mapping: Vec::new(),
            measurement_results: BTreeMap::new(),
            rng: PecosRng::seed_from_u64(0), // Will be properly seeded via set_seed()
            current_shot_seed: None,
            dynamic_state: None,
            pending_dynamic_ops: Vec::new(),
            simulated_op_count: 0,
            program_bytes: None,
            program_format: None,
            interface_builder: None,
            persistent_worker: None,
            operation_trace_dir: None,
            operation_trace_collector: None,
            trace_engine_id: TRACE_ENGINE_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            trace_shot_index: 0,
            trace_chunk_index: 0,
            command_builder: ByteMessageBuilder::new(),
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
            started: false,
            measurement_mapping: Vec::new(),
            measurement_results: BTreeMap::new(),
            rng: PecosRng::seed_from_u64(0), // Will be properly seeded via set_seed()
            current_shot_seed: None,
            dynamic_state: None,
            pending_dynamic_ops: Vec::new(),
            simulated_op_count: 0,
            program_bytes: None,
            program_format: None,
            interface_builder: None,
            persistent_worker: None,
            operation_trace_dir: None,
            operation_trace_collector: None,
            trace_engine_id: TRACE_ENGINE_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            trace_shot_index: 0,
            trace_chunk_index: 0,
            command_builder: ByteMessageBuilder::new(),
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
        self.num_physical_slots = 0;
    }

    fn allocate_qubit_slot(&mut self, program_id: usize) -> Result<usize, PecosError> {
        if let Some(&slot) = self.active_qubit_slots.get(&program_id) {
            return Ok(slot);
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
                        let slot = self.allocate_qubit_slot(*id)?;
                        builder.pz(&[slot]);
                        Self::push_gate_metadata(&mut gate_metadata, &mut pending_metadata);
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
                            builder.r1xy(
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
                        QuantumOp::ZZ(qubit1, qubit2) => {
                            builder.szz(&[(
                                self.mapped_qubit(*qubit1, qop)?,
                                self.mapped_qubit(*qubit2, qop)?,
                            )]);
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
            gate_metadata,
        });
        self.command_builder = builder;
        message
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
        if self.runtime.supports_operation_lowering() {
            let lowered_ops = self
                .runtime
                .lower_operations_with_metadata(ops)
                .map_err(|e| PecosError::Generic(format!("Runtime lowering error: {e}")))?;
            return self.quantum_ops_to_lowered_commands(lowered_ops);
        }

        self.operations_to_lowered_commands(ops)
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
                        builder.r1xy(
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
                    QuantumOp::ZZ(qubit1, qubit2) => {
                        builder.szz(&[(qubit1, qubit2)]);
                        gate_metadata.push(metadata);
                    }
                    QuantumOp::RZZ(angle, qubit1, qubit2) => {
                        builder.rzz(Angle64::from_radians(angle), &[(qubit1, qubit2)]);
                        gate_metadata.push(metadata);
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
            gate_metadata,
        });
        self.command_builder = builder;
        message
    }

    fn quantum_ops_to_bytemessage(
        &mut self,
        ops: Vec<QuantumOp>,
    ) -> Result<ByteMessage, PecosError> {
        self.quantum_ops_to_lowered_commands(ops.into_iter().map(LoweredQuantumOp::from).collect())
            .map(|lowered| lowered.commands)
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
            started: false,                       // Reset started flag for the clone
            measurement_mapping: Vec::new(),      // Clear for new shot
            measurement_results: BTreeMap::new(), // Clear for new shot
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
            operation_trace_dir: self.operation_trace_dir.clone(),
            operation_trace_collector: self.operation_trace_collector.clone(),
            trace_engine_id: TRACE_ENGINE_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            trace_shot_index: 0,
            trace_chunk_index: 0,
            command_builder: ByteMessageBuilder::new(),
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
                    let measurement_result_ids = if gate_type == "MZ" {
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
            self.persistent_worker = Some(PersistentDynamicWorker::new());
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
    fn set_dynamic_result(&mut self, result_id: u64, value: bool) -> Result<(), PecosError> {
        let state = self
            .dynamic_state
            .as_ref()
            .ok_or_else(|| PecosError::Generic("No dynamic execution in progress".to_string()))?;
        let handle = state
            .sync_handle
            .as_ref()
            .ok_or_else(|| PecosError::Generic("No sync handle available".to_string()))?;

        handle
            .set_measurement_result(result_id, value)
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
    fn get_dynamic_operations(&mut self) -> Option<Vec<Operation>> {
        let state = self.dynamic_state.as_ref()?;
        let handle = state.sync_handle.as_ref()?;
        handle.get_pending_operations().ok()
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
                        state.terminal_error = Some(format!("dynamic QIS worker failed: {e}"));
                    }
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
        self.dynamic_state
            .as_ref()
            .and_then(|state| state.terminal_error.as_ref())
            .map(|err| PecosError::Generic(err.clone()))
    }

    /// Latch a terminal failure for this shot and return it as an error.
    fn latch_terminal_error(&mut self, message: String) -> PecosError {
        if let Some(ref mut state) = self.dynamic_state {
            state.terminal_error = Some(message.clone());
        }
        PecosError::Generic(message)
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
        if let Err(e) = self.runtime.shot_end() {
            return Err(self.latch_terminal_error(format!("runtime shot_end failed: {e}")));
        }
        self.trace_named_result_traces_from_dynamic_handle();
        self.trace_complete_chunk();
        if let Some(ref mut state) = self.dynamic_state {
            state.finalized = true;
        }
        Ok(())
    }

    /// Abort dynamic execution (cleanup)
    fn abort_dynamic_execution(&mut self) {
        // Abort execution via sync handle if available
        if let Some(ref state) = self.dynamic_state
            && let Some(ref handle) = state.sync_handle
        {
            let _ = handle.abort_execution();
        }
        self.dynamic_state = None;
        self.pending_dynamic_ops.clear();
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
        debug!("QisEngine: reset() called");
        self.runtime
            .reset()
            .map_err(|e| PecosError::Generic(format!("Failed to reset runtime: {e}")))?;
        if let Some(ref mut interface) = self.interface {
            interface
                .reset()
                .map_err(crate::interface_impl::interface_error_to_pecos)?;
        }
        self.current_operations = None;
        self.started = false;
        self.measurement_mapping.clear();
        self.measurement_results.clear();
        self.current_shot_seed = None;
        debug!("QisEngine: reset() completed, cleared measurement_results");
        Ok(())
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

    fn generate_commands(&mut self) -> Result<ByteMessage, PecosError> {
        debug!("QisEngine::generate_commands called");

        // Get next batch of quantum operations from runtime
        match self.runtime.execute_until_quantum() {
            Ok(Some(ops)) => {
                debug!("QisEngine: Runtime returned {} operations", ops.len());
                for op in &ops {
                    debug!("QisEngine: Operation: {op:?}");
                }
                let quantum_ops: Vec<QuantumOp> = ops;
                let msg = self.quantum_ops_to_bytemessage(quantum_ops)?;
                debug!(
                    "QisEngine: Generated ByteMessage with {} measurement mappings",
                    self.measurement_mapping.len()
                );

                // Debug: Print the actual ByteMessage content
                debug!("QisEngine: Generated ByteMessage:");
                if let Ok(quantum_ops) = msg.quantum_ops() {
                    debug!("  Quantum ops: {} total", quantum_ops.len());
                    for (i, gate) in quantum_ops.iter().enumerate() {
                        debug!("    Gate {i}: {gate:?}");
                    }
                }
                if let Ok(empty) = msg.is_empty() {
                    debug!("  Is empty: {empty}");
                }

                Ok(msg)
            }
            Ok(None) => {
                debug!("QisEngine: Runtime complete, no more operations");
                Ok(ByteMessage::builder().build())
            }
            Err(e) => {
                debug!("QisEngine: Runtime error: {e}");
                Err(PecosError::Generic(format!("Runtime error: {e}")))
            }
        }
    }

    fn get_results(&self) -> Result<Shot, PecosError> {
        debug!("QisEngine::get_results called");
        debug!(
            "QisEngine: get_results() called, stored results: {:?}",
            self.measurement_results
        );

        // Convert stored measurement results to PECOS shot format
        let mut shot = Shot::default();

        // First, try to get named results from print_bool/print_bool_arr calls
        let mut has_named_results = false;
        if let Some(state) = &self.dynamic_state
            && let Some(handle) = &state.sync_handle
        {
            match handle.get_named_results() {
                Ok(named_results) => {
                    has_named_results = !named_results.is_empty();
                    for (name, values) in named_results {
                        // Convert Vec<bool> to Data
                        // For single values, store as U32; for arrays, store as Vec<U32>
                        if values.len() == 1 {
                            shot.data.insert(name, Data::U32(u32::from(values[0])));
                        } else {
                            // Store as Vec of U32 values (0 or 1)
                            let data_vec: Vec<Data> =
                                values.iter().map(|&b| Data::U32(u32::from(b))).collect();
                            shot.data.insert(name, Data::Vec(data_vec));
                        }
                    }
                    debug!("QisEngine: Added named results to shot");
                }
                Err(e) => {
                    debug!("QisEngine: Failed to get named results: {e}");
                }
            }
        }

        // Only add raw measurements if there are no named results.
        // This handles circuits with variable loop iterations where each shot
        // may produce a different number of raw measurements, but the named
        // results (from result() calls) are consistent.
        if !has_named_results {
            for (result_id, value) in &self.measurement_results {
                shot.data.insert(
                    format!("measurement_{result_id}"),
                    Data::U32(u32::from(*value)),
                );
                debug!(
                    "QisEngine: Added to shot: measurement_{} = {}",
                    result_id,
                    i32::from(*value)
                );
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

        let updates = Self::map_measurements(&self.measurement_mapping, &measurements);
        self.store_measurement_updates(&updates);

        debug!(
            "QisEngine: Final measurement_results: {:?}",
            self.measurement_results
        );

        self.provide_measurement_updates_to_runtime(&updates)
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

        // Clear previous shot's measurement state
        self.measurement_results.clear();
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

        // Reset the runtime to ensure clean state for new shot
        self.runtime
            .reset()
            .map_err(|e| PecosError::Generic(format!("Failed to reset runtime: {e}")))?;

        // Start a new shot with the generated seed
        self.runtime
            .shot_start(0, Some(shot_seed))
            .map_err(|e| PecosError::Generic(format!("Failed to start shot: {e}")))?;

        self.started = true;

        // Start LLVM program in worker thread
        self.start_dynamic_worker()?;

        // Wait for the worker to either need a result or complete
        // Use long timeout as safety net - condvar will wake immediately on signal
        if let Some(result_id) = self.wait_for_result_needed(30_000) {
            debug!("Worker needs result for id={result_id}");
            // Get pending operations
            if let Some(ops) = self.get_dynamic_operations() {
                // Track how many operations we're sending for simulation
                self.simulated_op_count = ops.len();
                if !ops.is_empty() {
                    let lowered = self.lower_operations_to_commands(&ops)?;
                    self.trace_operations_chunk(
                        "pending_start",
                        &ops,
                        Some(result_id),
                        Some(&lowered),
                    );
                    return Ok(EngineStage::NeedsProcessing(lowered.commands));
                }
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
                    let lowered = self.lower_operations_to_commands(&final_ops)?;
                    self.trace_operations_chunk("pending_final", &final_ops, None, Some(&lowered));
                    return Ok(EngineStage::NeedsProcessing(lowered.commands));
                }
            }
            self.finalize_shot_for_certification()?;
            let shot = self.get_results()?;
            return Ok(EngineStage::Complete(shot));
        }

        // Return empty commands while we wait
        Ok(EngineStage::NeedsProcessing(ByteMessage::builder().build()))
    }

    fn continue_processing(
        &mut self,
        input: Self::EngineOutput,
    ) -> Result<EngineStage<Self::EngineInput, Self::Output>, PecosError> {
        debug!("QisEngine::continue_processing called");

        // Verify dynamic state exists (set by start())
        if self.dynamic_state.is_none() {
            return Err(PecosError::Generic(
                "continue_processing called without dynamic state - was start() called?"
                    .to_string(),
            ));
        }

        let measurement_updates = if NoiseUtils::has_measurements(&input) {
            let measurements = Self::parse_measurement_outcomes(&input)?;
            let mapping = std::mem::take(&mut self.measurement_mapping);
            let updates = Self::map_measurements(&mapping, &measurements);
            self.store_measurement_updates(&updates);
            self.provide_measurement_updates_to_runtime(&updates)?;
            updates
        } else {
            Vec::new()
        };

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
                    let lowered = self.lower_operations_to_commands(&final_ops)?;
                    self.trace_operations_chunk("pending_final", &final_ops, None, Some(&lowered));
                    return Ok(EngineStage::NeedsProcessing(lowered.commands));
                }
            }
            self.finalize_shot_for_certification()?;
            let shot = self.get_results()?;
            return Ok(EngineStage::Complete(shot));
        }

        // Provide new measurement values to the dynamic worker thread.
        for &(result_id, value) in &measurement_updates {
            debug!("Stored and providing measurement: result_id={result_id} value={value}");
            self.set_dynamic_result(result_id as u64, value)?;
        }

        // Signal that results are ready
        if !measurement_updates.is_empty() {
            self.signal_dynamic_result_ready()?;
        }

        // Clear measurement mapping for next batch
        self.measurement_mapping.clear();

        // Wait for worker to need more results or complete
        // Condvar wakes immediately on signal; timeout is just a safety net
        if let Some(result_id) = self.wait_for_result_needed(30_000) {
            debug!("Worker needs result for id={result_id}");

            // Check if we already have this result (from a previous batch)
            // Note: result_id is u64 but measurement_results uses usize keys
            // This is safe because result IDs are small sequential integers
            #[allow(clippy::cast_possible_truncation)]
            let result_key = result_id as usize;
            if let Some(&value) = self.measurement_results.get(&result_key) {
                debug!("Result {result_id} already available, signaling immediately");
                // Re-set the result in global storage (in case it was cleared)
                self.set_dynamic_result(result_id, value)?;
                self.signal_dynamic_result_ready()?;
                // Continue loop to wait for next result or completion
            } else {
                // Get newly exported operations.
                if let Some(ops) = self.get_dynamic_operations() {
                    self.simulated_op_count += ops.len();
                    if !ops.is_empty() {
                        let lowered = self.lower_operations_to_commands(&ops)?;
                        self.trace_operations_chunk(
                            "pending_continue",
                            &ops,
                            Some(result_id),
                            Some(&lowered),
                        );
                        return Ok(EngineStage::NeedsProcessing(lowered.commands));
                    }
                }
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
                    let lowered = self.lower_operations_to_commands(&final_ops)?;
                    self.trace_operations_chunk("pending_final", &final_ops, None, Some(&lowered));
                    return Ok(EngineStage::NeedsProcessing(lowered.commands));
                }
            }
            self.finalize_shot_for_certification()?;
            let shot = self.get_results()?;
            return Ok(EngineStage::Complete(shot));
        }

        // Return empty commands while we wait
        Ok(EngineStage::NeedsProcessing(ByteMessage::builder().build()))
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        // Abort any dynamic execution in progress
        self.abort_dynamic_execution();
        // Reset everything
        <Self as Engine>::reset(self)
    }
}

// Tests for QisEngine are in integration tests since they require
// actual interface and runtime implementations.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{ClassicalState, Result as RuntimeResult};
    use tempfile::TempDir;

    #[derive(Clone, Default)]
    struct DummyRuntime {
        state: ClassicalState,
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
            .operations_to_lowered_commands(&ops)
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

        let ops = vec![QuantumOp::H(0).into()];
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
            .operations_to_lowered_commands(&ops)
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
