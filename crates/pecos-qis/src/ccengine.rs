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
use log::debug;
use pecos_core::Angle64;
use pecos_core::prelude::PecosError;
use pecos_engines::noise::utils::NoiseUtils;
use pecos_engines::shot_results::{Data, Shot};
use pecos_engines::{
    ByteMessage, ByteMessageBuilder, ClassicalEngine, ControlEngine, Engine, EngineStage,
};
use pecos_qis_ffi_types::{Operation, OperationCollector as OperationList, QuantumOp};
use pecos_rng::PecosRng;
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::JoinHandle;

/// Result from worker thread - returns both the operations and the interface
type WorkerResult = Result<(OperationList, BoxedInterface), String>;

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

                    // Disable dynamic mode before returning
                    let _ = interface.disable_dynamic_mode();

                    // Send result back to main thread
                    let send_result = result
                        .map(|collector| (collector, interface))
                        .map_err(|e| e.to_string());

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

    /// Try to receive a result without blocking
    fn try_recv_result(&self) -> Option<WorkerResult> {
        self.result_rx.lock().ok()?.try_recv().ok()
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

    /// Number of qubits in the program
    num_qubits: usize,

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
}

impl QisEngine {
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
            num_qubits: 0,
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
            num_qubits: 0,
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

    /// Convert quantum operations to `ByteMessage` for the quantum engine
    fn operations_to_bytemessage(
        &mut self,
        ops: Vec<QuantumOp>,
    ) -> Result<ByteMessage, PecosError> {
        let mut builder = ByteMessageBuilder::new();
        self.measurement_mapping.clear();

        for op in ops {
            match op {
                QuantumOp::H(qubit) => {
                    builder.add_h(&[qubit]);
                }
                QuantumOp::X(qubit) => {
                    builder.add_x(&[qubit]);
                }
                QuantumOp::Y(qubit) => {
                    builder.add_y(&[qubit]);
                }
                QuantumOp::Z(qubit) => {
                    builder.add_z(&[qubit]);
                }
                QuantumOp::S(qubit) => {
                    builder.add_sz(&[qubit]);
                }
                QuantumOp::Sdg(qubit) => {
                    builder.add_szdg(&[qubit]);
                }
                QuantumOp::T(qubit) => {
                    builder.add_t(&[qubit]);
                }
                QuantumOp::Tdg(qubit) => {
                    builder.add_tdg(&[qubit]);
                }
                QuantumOp::RX(angle, qubit) => {
                    builder.add_rx(Angle64::from_radians(angle), &[qubit]);
                }
                QuantumOp::RY(angle, qubit) => {
                    builder.add_ry(Angle64::from_radians(angle), &[qubit]);
                }
                QuantumOp::RZ(angle, qubit) => {
                    builder.add_rz(Angle64::from_radians(angle), &[qubit]);
                }
                QuantumOp::RXY(theta, phi, qubit) => {
                    builder.add_r1xy(
                        Angle64::from_radians(theta),
                        Angle64::from_radians(phi),
                        &[qubit],
                    );
                }
                QuantumOp::CX(control, target) => {
                    builder.add_cx(&[control], &[target]);
                }
                QuantumOp::Measure(qubit, result_id) => {
                    self.measurement_mapping.push(result_id);
                    builder.add_measurements(&[qubit]);
                }
                QuantumOp::ZZ(qubit1, qubit2) => {
                    // ZZ gate is the same as SZZ in PECOS
                    builder.add_szz(&[qubit1], &[qubit2]);
                }
                QuantumOp::RZZ(angle, qubit1, qubit2) => {
                    builder.add_rzz(Angle64::from_radians(angle), &[qubit1], &[qubit2]);
                }
                QuantumOp::Reset(qubit) => {
                    builder.add_prep(&[qubit]);
                }
                _ => {
                    // For other operations, we'd need to add more builder methods
                    // or convert to a generic gate representation
                    return Err(PecosError::Generic(format!(
                        "Unsupported operation: {op:?}"
                    )));
                }
            }
        }

        Ok(builder.build())
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
            num_qubits: self.num_qubits,
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
        }
    }
}

// Helper methods for dynamic execution
impl QisEngine {
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

    /// Get pending operations from the dynamic execution
    ///
    /// This reads from the global storage, which the worker thread
    /// populates before blocking.
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
                    let total_ops = collector.operations.len();
                    debug!(
                        "Worker completed with {} total operations, {} already simulated",
                        total_ops, self.simulated_op_count
                    );
                    // Only store NEW operations (those after what we already simulated)
                    if total_ops > self.simulated_op_count {
                        self.pending_dynamic_ops =
                            collector.operations[self.simulated_op_count..].to_vec();
                        debug!(
                            "Storing {} new operations for final processing",
                            self.pending_dynamic_ops.len()
                        );
                    } else {
                        self.pending_dynamic_ops.clear();
                    }
                    self.interface = Some(interface);
                    if let Some(ref mut state) = self.dynamic_state {
                        state.execution_complete = true;
                    }
                    return true;
                }
                Err(e) => {
                    log::error!("Worker failed: {e}");
                    if let Some(ref mut state) = self.dynamic_state {
                        state.execution_complete = true;
                    }
                    return true;
                }
            }
        }

        false
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

    /// Convert a list of Operations to `QuantumOps` for the quantum engine
    fn operations_to_quantum_ops(ops: &[Operation]) -> Vec<QuantumOp> {
        ops.iter()
            .filter_map(|op| {
                if let Operation::Quantum(qop) = op {
                    Some(qop.clone())
                } else {
                    None
                }
            })
            .collect()
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
        let num_qubits = self.runtime.num_qubits();
        debug!("QisEngine: num_qubits() returning {num_qubits}");
        num_qubits
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
                let msg = self.operations_to_bytemessage(quantum_ops)?;
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
        let measurements = message
            .outcomes()
            .map_err(|e| PecosError::Generic(format!("Failed to parse measurements: {e}")))?;

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

        // Convert to BTreeMap for the runtime and store for get_results()
        let mut measurement_map = BTreeMap::new();
        for (idx, &value) in measurements.iter().enumerate() {
            if idx < self.measurement_mapping.len() {
                let result_id = self.measurement_mapping[idx];
                let bool_value = value != 0;
                measurement_map.insert(result_id, bool_value);

                // Store for get_results()
                self.measurement_results.insert(result_id, bool_value);
                debug!("QisEngine: Stored measurement result_id={result_id}, value={bool_value}");
            }
        }

        debug!(
            "QisEngine: Final measurement_results: {:?}",
            self.measurement_results
        );

        self.runtime
            .provide_measurements(measurement_map)
            .map_err(|e| PecosError::Generic(format!("Failed to provide measurements: {e}")))
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
        debug!("QisEngine: Cleared previous measurement results for new shot");

        // Generate a per-shot seed from our RNG
        let shot_seed = self.rng.next_u64();
        debug!("QisEngine: Generated shot seed {shot_seed}");

        // Store the shot seed for quantum engine access
        self.current_shot_seed = Some(shot_seed);

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
                self.pending_dynamic_ops.clone_from(&ops);
                // Track how many operations we're sending for simulation
                self.simulated_op_count = ops.len();
                debug!(
                    "Tracking {} operations as simulated",
                    self.simulated_op_count
                );
                let quantum_ops = Self::operations_to_quantum_ops(&ops);
                if !quantum_ops.is_empty() {
                    let commands = self.operations_to_bytemessage(quantum_ops)?;
                    return Ok(EngineStage::NeedsProcessing(commands));
                }
            }
        }

        // Check if worker completed without needing any results
        if self.check_worker_complete() {
            // Worker completed but we still need to process any pending operations
            // through the quantum engine (e.g., programs without measurement-dependent conditionals)
            if !self.pending_dynamic_ops.is_empty() {
                let quantum_ops = Self::operations_to_quantum_ops(&self.pending_dynamic_ops);
                self.pending_dynamic_ops.clear();
                if !quantum_ops.is_empty() {
                    debug!(
                        "Worker completed - sending {} final operations to quantum engine",
                        quantum_ops.len()
                    );
                    let commands = self.operations_to_bytemessage(quantum_ops)?;
                    return Ok(EngineStage::NeedsProcessing(commands));
                }
            }
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

        // Process the response from quantum engine
        if NoiseUtils::has_measurements(&input) {
            self.handle_measurements(input.clone())?;
        }

        // First, check if worker already completed (before processing anything else)
        // This avoids unnecessary work if the worker finished
        if self.check_worker_complete() {
            debug!("Worker already complete, finishing shot");
            // Process any final operations
            if !self.pending_dynamic_ops.is_empty() {
                let quantum_ops = Self::operations_to_quantum_ops(&self.pending_dynamic_ops);
                if !quantum_ops.is_empty() {
                    let commands = self.operations_to_bytemessage(quantum_ops)?;
                    self.pending_dynamic_ops.clear();
                    return Ok(EngineStage::NeedsProcessing(commands));
                }
            }
            let shot = self.get_results()?;
            return Ok(EngineStage::Complete(shot));
        }

        // Extract measurements from quantum engine response
        let measurements = input
            .outcomes()
            .map_err(|e| PecosError::Generic(format!("Failed to parse measurements: {e}")))?;

        // Map measurement values to result IDs and provide to worker
        for (idx, &value) in measurements.iter().enumerate() {
            if idx < self.measurement_mapping.len() {
                let result_id = self.measurement_mapping[idx];
                let bool_value = value != 0;
                self.measurement_results.insert(result_id, bool_value);
                debug!(
                    "Stored and providing measurement: result_id={result_id} value={bool_value}"
                );
                // Provide result to worker thread
                self.set_dynamic_result(result_id as u64, bool_value)?;
            }
        }

        // Signal that results are ready
        if !measurements.is_empty() {
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
            if self.measurement_results.contains_key(&result_key) {
                debug!("Result {result_id} already available, signaling immediately");
                // Re-set the result in global storage (in case it was cleared)
                let value = self.measurement_results[&result_key];
                self.set_dynamic_result(result_id, value)?;
                self.signal_dynamic_result_ready()?;
                // Continue loop to wait for next result or completion
            } else {
                // Get pending operations
                if let Some(ops) = self.get_dynamic_operations() {
                    // Only process NEW operations (after what we already simulated)
                    if ops.len() > self.simulated_op_count {
                        let new_ops: Vec<Operation> = ops[self.simulated_op_count..].to_vec();
                        // Update count to include these new operations
                        self.simulated_op_count = ops.len();
                        debug!(
                            "Processing {} new operations, total simulated: {}",
                            new_ops.len(),
                            self.simulated_op_count
                        );
                        let quantum_ops = Self::operations_to_quantum_ops(&new_ops);
                        self.pending_dynamic_ops = new_ops;
                        if !quantum_ops.is_empty() {
                            let commands = self.operations_to_bytemessage(quantum_ops)?;
                            return Ok(EngineStage::NeedsProcessing(commands));
                        }
                    }
                }
            }
        }

        // Check if worker completed after the wait
        if self.check_worker_complete() {
            debug!("Worker completed after wait");
            // Process any final operations
            if !self.pending_dynamic_ops.is_empty() {
                let quantum_ops = Self::operations_to_quantum_ops(&self.pending_dynamic_ops);
                if !quantum_ops.is_empty() {
                    let commands = self.operations_to_bytemessage(quantum_ops)?;
                    self.pending_dynamic_ops.clear();
                    return Ok(EngineStage::NeedsProcessing(commands));
                }
            }
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
