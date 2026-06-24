//! Selene Runtime implementation of `QisRuntime`
//!
//! This wraps a Selene .so runtime plugin and implements the `QisRuntime` trait
//! to provide a Selene-based classical interpreter for QIS programs.

use crate::runtime::{ClassicalState, QisRuntime, Result, RuntimeError, Shot};
use log::{debug, trace};
use pecos_qis_ffi_types::{Operation, OperationCollector, QuantumOp};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CString, c_void};
use std::mem::ManuallyDrop;
use std::path::{Path, PathBuf};
use std::sync::Arc;

type RuntimeInstance = *mut c_void;
type RuntimeGetOperationInstance = *mut c_void;

#[derive(Debug, Clone)]
enum RuntimeScheduledOp {
    Rxy {
        qubit_id: u64,
        theta: f64,
        phi: f64,
    },
    Rz {
        qubit_id: u64,
        theta: f64,
    },
    Rzz {
        qubit_id_1: u64,
        qubit_id_2: u64,
        theta: f64,
    },
    Measure {
        qubit_id: u64,
        result_id: u64,
    },
    MeasureLeaked {
        qubit_id: u64,
        result_id: u64,
    },
    Reset {
        qubit_id: u64,
    },
    Custom,
}

#[derive(Debug, Default)]
struct RuntimeOperationBatch {
    start_time_nanos: u64,
    duration_nanos: u64,
    invoked: bool,
    operations: Vec<RuntimeScheduledOp>,
}

impl RuntimeOperationBatch {
    fn end_time_nanos(&self) -> u64 {
        self.start_time_nanos.saturating_add(self.duration_nanos)
    }
}

#[repr(C)]
struct SeleneRuntimeGetOperationInterface {
    rzz_fn: extern "C" fn(RuntimeGetOperationInstance, u64, u64, f64),
    rxy_fn: extern "C" fn(RuntimeGetOperationInstance, u64, f64, f64),
    rz_fn: extern "C" fn(RuntimeGetOperationInstance, u64, f64),
    measure_fn: extern "C" fn(RuntimeGetOperationInstance, u64, u64),
    measure_leaked_fn: extern "C" fn(RuntimeGetOperationInstance, u64, u64),
    reset_fn: extern "C" fn(RuntimeGetOperationInstance, u64),
    custom_fn: extern "C" fn(RuntimeGetOperationInstance, usize, *const c_void, usize),
    set_batch_time_fn: extern "C" fn(RuntimeGetOperationInstance, u64, u64),
}

extern "C" fn runtime_batch_rxy(
    instance: RuntimeGetOperationInstance,
    qubit_id: u64,
    theta: f64,
    phi: f64,
) {
    let batch = unsafe { &mut *(instance.cast::<RuntimeOperationBatch>()) };
    batch.operations.push(RuntimeScheduledOp::Rxy {
        qubit_id,
        theta,
        phi,
    });
    batch.invoked = true;
}

extern "C" fn runtime_batch_rz(instance: RuntimeGetOperationInstance, qubit_id: u64, theta: f64) {
    let batch = unsafe { &mut *(instance.cast::<RuntimeOperationBatch>()) };
    batch
        .operations
        .push(RuntimeScheduledOp::Rz { qubit_id, theta });
    batch.invoked = true;
}

extern "C" fn runtime_batch_rzz(
    instance: RuntimeGetOperationInstance,
    qubit_id_1: u64,
    qubit_id_2: u64,
    theta: f64,
) {
    let batch = unsafe { &mut *(instance.cast::<RuntimeOperationBatch>()) };
    batch.operations.push(RuntimeScheduledOp::Rzz {
        qubit_id_1,
        qubit_id_2,
        theta,
    });
    batch.invoked = true;
}

extern "C" fn runtime_batch_measure(
    instance: RuntimeGetOperationInstance,
    qubit_id: u64,
    result_id: u64,
) {
    let batch = unsafe { &mut *(instance.cast::<RuntimeOperationBatch>()) };
    batch.operations.push(RuntimeScheduledOp::Measure {
        qubit_id,
        result_id,
    });
    batch.invoked = true;
}

extern "C" fn runtime_batch_measure_leaked(
    instance: RuntimeGetOperationInstance,
    qubit_id: u64,
    result_id: u64,
) {
    let batch = unsafe { &mut *(instance.cast::<RuntimeOperationBatch>()) };
    batch.operations.push(RuntimeScheduledOp::MeasureLeaked {
        qubit_id,
        result_id,
    });
    batch.invoked = true;
}

extern "C" fn runtime_batch_reset(instance: RuntimeGetOperationInstance, qubit_id: u64) {
    let batch = unsafe { &mut *(instance.cast::<RuntimeOperationBatch>()) };
    batch
        .operations
        .push(RuntimeScheduledOp::Reset { qubit_id });
    batch.invoked = true;
}

extern "C" fn runtime_batch_custom(
    instance: RuntimeGetOperationInstance,
    _tag: usize,
    _data: *const c_void,
    _data_len: usize,
) {
    let batch = unsafe { &mut *(instance.cast::<RuntimeOperationBatch>()) };
    batch.operations.push(RuntimeScheduledOp::Custom);
    batch.invoked = true;
}

extern "C" fn runtime_batch_set_time(
    instance: RuntimeGetOperationInstance,
    start_time_nanos: u64,
    duration_nanos: u64,
) {
    let batch = unsafe { &mut *(instance.cast::<RuntimeOperationBatch>()) };
    batch.start_time_nanos = start_time_nanos;
    batch.duration_nanos = duration_nanos;
    batch.invoked = true;
}

static RUNTIME_OPERATION_CALLBACKS: SeleneRuntimeGetOperationInterface =
    SeleneRuntimeGetOperationInterface {
        rzz_fn: runtime_batch_rzz,
        rxy_fn: runtime_batch_rxy,
        rz_fn: runtime_batch_rz,
        measure_fn: runtime_batch_measure,
        measure_leaked_fn: runtime_batch_measure_leaked,
        reset_fn: runtime_batch_reset,
        custom_fn: runtime_batch_custom,
        set_batch_time_fn: runtime_batch_set_time,
    };

/// Selene runtime implementation
///
/// The `library` field is wrapped in `ManuallyDrop` to prevent calling `dlclose()`
/// during process exit. Calling `dlclose()` during shutdown can cause hangs because
/// thread-local storage may already be partially torn down, or other static
/// destructors may be running concurrently.
pub struct SeleneRuntime {
    /// Path to the Selene .so file
    plugin_path: String,

    /// Runtime-plugin init arguments passed to `selene_runtime_init`.
    init_args: Vec<String>,

    /// Additional dynamic library search directories needed by the plugin.
    library_search_dirs: Vec<PathBuf>,

    /// Loaded library (if any)
    /// Wrapped in `ManuallyDrop` to prevent `dlclose()` during process exit.
    #[allow(dead_code)]
    library: Option<ManuallyDrop<Arc<libloading::Library>>>,

    /// Runtime instance pointer
    #[allow(dead_code)]
    instance: Option<*mut c_void>,

    /// Number of qubits the current runtime instance was initialized with.
    initialized_num_qubits: Option<usize>,

    /// Current classical state
    state: ClassicalState,

    /// Operations buffer for batching
    operations_buffer: Vec<QuantumOp>,

    /// Maximum batch size for operations
    batch_size: usize,

    /// Number of qubits
    num_qubits: usize,

    /// Explicit physical runtime capacity requested by the caller.
    ///
    /// Some generated programs use sparse, monotonically increasing logical
    /// handles while guaranteeing a smaller maximum number of live physical
    /// slots. In those cases the `.qubits(...)` hint is the runtime capacity;
    /// if the program actually exceeds it, plugin qalloc fails loudly.
    num_qubits_hint: Option<usize>,

    /// Whether the loaded operation stream uses explicit qalloc/qfree records.
    ///
    /// In this mode program qubit IDs are logical handles, not dense physical
    /// runtime slots. Runtime plugin capacity must therefore follow the maximum
    /// simultaneously-live allocation count instead of `max(program_id) + 1`.
    uses_explicit_qubit_allocation: bool,

    /// Number of allocated result slots
    num_results: usize,

    /// Loaded QIS interface
    interface: Option<OperationCollector>,

    /// Current operation index
    current_op_index: usize,

    /// Flag indicating we need to re-execute with known measurements
    /// Set to true after measurements are provided for dynamic circuits
    needs_reexecution: bool,

    /// Track measurement result IDs that have been seen but not yet resolved
    pending_measurements: Vec<usize>,

    /// Program qubit handles mapped onto runtime qubit handles returned by qalloc.
    program_to_runtime_qubits: BTreeMap<usize, u64>,

    /// Program result IDs mapped onto runtime future IDs returned by measure.
    program_to_runtime_results: BTreeMap<usize, u64>,

    /// Reverse lookup for measurement operations emitted by the runtime plugin.
    runtime_to_program_results: BTreeMap<u64, usize>,

    /// End timestamp of the last scheduled physical operation per runtime qubit.
    last_gate_time_end_nanos: Vec<u64>,

    /// Shot metadata waiting for a lazily loaded runtime plugin.
    pending_shot_start: Option<(u64, Option<u64>)>,
}

// SAFETY: SeleneRuntime owns its instance pointer exclusively.
// WARNING: The Selene FFI runtime may not be thread-safe for concurrent access.
// Sync is required by the QisRuntime/Engine trait but callers must ensure
// single-threaded access to any given instance.
unsafe impl Send for SeleneRuntime {}
unsafe impl Sync for SeleneRuntime {}

impl SeleneRuntime {
    /// Create a new Selene runtime with the given plugin path
    pub fn new(plugin_path: impl AsRef<Path>) -> Self {
        Self {
            plugin_path: plugin_path.as_ref().to_string_lossy().to_string(),
            init_args: Vec::new(),
            library_search_dirs: Vec::new(),
            library: None,
            instance: None,
            initialized_num_qubits: None,
            state: ClassicalState::default(),
            operations_buffer: Vec::new(),
            batch_size: 100,
            num_qubits: 0,
            num_qubits_hint: None,
            uses_explicit_qubit_allocation: false,
            num_results: 0,
            interface: None,
            current_op_index: 0,
            needs_reexecution: false,
            pending_measurements: Vec::new(),
            program_to_runtime_qubits: BTreeMap::new(),
            program_to_runtime_results: BTreeMap::new(),
            runtime_to_program_results: BTreeMap::new(),
            last_gate_time_end_nanos: Vec::new(),
            pending_shot_start: None,
        }
    }

    /// Create a runtime from the generic Selene runtime-plugin shape.
    ///
    /// `init_args` are passed directly to the plugin's `selene_runtime_init`
    /// argc/argv pair. `library_search_dirs` are prepended to the platform
    /// dynamic-library search path before loading the plugin.
    pub fn with_plugin_config(
        plugin_path: impl AsRef<Path>,
        init_args: Vec<String>,
        library_search_dirs: Vec<PathBuf>,
    ) -> Self {
        let mut runtime = Self::new(plugin_path);
        runtime.init_args = init_args;
        runtime.library_search_dirs = library_search_dirs;
        runtime
    }

    /// Check if this runtime needs re-execution with known measurements
    ///
    /// This is set to true after measurements are provided for programs
    /// that may have conditional logic depending on measurement results.
    #[must_use]
    pub fn needs_reexecution(&self) -> bool {
        self.needs_reexecution
    }

    /// Clear the re-execution flag after operations have been reloaded
    pub fn clear_reexecution_flag(&mut self) {
        self.needs_reexecution = false;
    }

    /// Reload operations from a new execution (used for dynamic circuits)
    pub fn reload_operations(&mut self, operations: OperationCollector) {
        debug!(
            "Reloading operations with {} ops (previous: {} ops)",
            operations.operations.len(),
            self.interface.as_ref().map_or(0, |i| i.operations.len())
        );

        // Update capacities from explicit allocation records when available,
        // otherwise fall back to direct program handles used by older QIR/LLVM
        // examples.
        let (num_qubits, num_results) = collector_capacity(&operations);
        self.num_qubits = num_qubits;
        self.num_results = num_results;
        self.uses_explicit_qubit_allocation =
            has_explicit_qubit_allocations(&operations.operations);

        self.interface = Some(operations);
        self.current_op_index = 0;
        self.needs_reexecution = false;
        self.pending_measurements.clear();
    }

    /// Load the Selene plugin
    fn load_plugin(&mut self) -> Result<()> {
        if self.library.is_some() {
            return Ok(());
        }

        self.apply_library_search_dirs()?;
        let plugin_num_qubits = self.plugin_num_qubits();

        debug!(
            "Loading Selene plugin from {} with {} qubits, {} results, and {} init args",
            self.plugin_path,
            plugin_num_qubits,
            self.num_results,
            self.init_args.len()
        );

        unsafe {
            let lib = Arc::new(
                libloading::Library::new(&self.plugin_path)
                    .map_err(|e| RuntimeError::FfiError(format!("Failed to load plugin: {e}")))?,
            );

            // Initialize runtime instance
            let init_fn: libloading::Symbol<
                unsafe extern "C" fn(*mut *mut c_void, u64, u64, u32, *const *const i8) -> i32,
            > = lib
                .get(b"selene_runtime_init")
                .map_err(|e| RuntimeError::FfiError(format!("Missing init function: {e}")))?;

            let c_args = self
                .init_args
                .iter()
                .map(|arg| {
                    CString::new(arg.as_str()).map_err(|_| {
                        RuntimeError::FfiError(format!(
                            "Selene runtime init argument contains NUL byte: {arg:?}"
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let arg_ptrs = c_args.iter().map(|arg| arg.as_ptr()).collect::<Vec<_>>();
            let argv = if arg_ptrs.is_empty() {
                std::ptr::null()
            } else {
                arg_ptrs.as_ptr()
            };

            let mut instance: *mut c_void = std::ptr::null_mut();
            let errno = init_fn(
                &raw mut instance,
                plugin_num_qubits as u64,
                0, // start time
                arg_ptrs.len() as u32,
                argv,
            );

            if errno != 0 {
                return Err(RuntimeError::FfiError(format!(
                    "Init failed with errno {errno}"
                )));
            }

            self.library = Some(ManuallyDrop::new(lib));
            self.instance = Some(instance);
            self.initialized_num_qubits = Some(plugin_num_qubits);
        }

        self.apply_pending_shot_start()?;
        Ok(())
    }

    fn ensure_plugin_capacity(&mut self) -> Result<()> {
        let Some(initialized_num_qubits) = self.initialized_num_qubits else {
            return Ok(());
        };

        let plugin_num_qubits = self.plugin_num_qubits();
        if plugin_num_qubits <= initialized_num_qubits {
            return Ok(());
        }

        debug!(
            "Reinitializing Selene plugin capacity from {} to {} qubits",
            initialized_num_qubits, plugin_num_qubits
        );
        self.reset_plugin_instance()?;
        self.load_plugin()
    }

    fn plugin_num_qubits(&self) -> usize {
        // An explicit hint is authoritative and caps the plugin capacity (a
        // program that exceeds it fails loudly at qalloc by design). The bogus
        // "0 inferred before execution" case is handled upstream in SimBuilder,
        // which no longer freezes that 0 as a hint.
        self.num_qubits_hint.unwrap_or(self.num_qubits)
    }

    fn reset_plugin_instance(&mut self) -> Result<()> {
        if let Some(lib) = &self.library
            && let Some(instance) = self.instance
        {
            unsafe {
                if let Ok(exit_fn) =
                    lib.get::<unsafe extern "C" fn(*mut c_void) -> i32>(b"selene_runtime_exit")
                {
                    let errno = exit_fn(instance);
                    if errno != 0 {
                        return Err(RuntimeError::ExecutionError(format!(
                            "Selene runtime exit failed with errno {errno}"
                        )));
                    }
                }
            }
        }

        self.instance = None;
        self.library = None;
        self.initialized_num_qubits = None;
        self.program_to_runtime_qubits.clear();
        self.program_to_runtime_results.clear();
        self.runtime_to_program_results.clear();
        self.last_gate_time_end_nanos.clear();
        Ok(())
    }

    fn apply_pending_shot_start(&mut self) -> Result<()> {
        let Some((shot_id, seed)) = self.pending_shot_start else {
            return Ok(());
        };
        let Some(lib) = &self.library else {
            return Ok(());
        };
        let Some(instance) = self.instance else {
            return Ok(());
        };

        unsafe {
            if let Ok(shot_start_fn) = lib
                .get::<unsafe extern "C" fn(*mut c_void, u64, u64) -> i32>(
                    b"selene_runtime_shot_start",
                )
            {
                let errno = shot_start_fn(instance, shot_id, seed.unwrap_or(0));
                if errno != 0 {
                    return Err(RuntimeError::ExecutionError(format!(
                        "Shot start failed with errno {errno}"
                    )));
                }
            }
        }

        self.pending_shot_start = None;
        Ok(())
    }

    fn apply_library_search_dirs(&self) -> Result<()> {
        if self.library_search_dirs.is_empty() {
            return Ok(());
        }

        let env_key = if cfg!(target_os = "windows") {
            "PATH"
        } else if cfg!(target_os = "macos") {
            "DYLD_LIBRARY_PATH"
        } else {
            "LD_LIBRARY_PATH"
        };

        let existing = std::env::var_os(env_key).unwrap_or_default();
        let mut paths = self.library_search_dirs.clone();
        paths.extend(std::env::split_paths(&existing));
        let joined = std::env::join_paths(paths).map_err(|e| {
            RuntimeError::FfiError(format!("Invalid Selene runtime library search path: {e}"))
        })?;

        // SAFETY: This mirrors Selene's plugin runtime environment setup. The
        // mutation happens immediately before loading the selected runtime.
        unsafe {
            std::env::set_var(env_key, joined);
        }

        Ok(())
    }

    /// Process operations from the interface sequentially
    ///
    /// This method now breaks at measurement operations to allow the quantum
    /// simulator to execute measurements before continuing. This is essential
    /// for dynamic circuits where conditionals depend on measurement results.
    fn process_interface_ops(&mut self) -> Result<Option<Vec<QuantumOp>>> {
        let interface = self
            .interface
            .as_ref()
            .ok_or(RuntimeError::NoProgramLoaded)?;

        self.operations_buffer.clear();
        self.pending_measurements.clear();

        while self.current_op_index < interface.operations.len() {
            let op = &interface.operations[self.current_op_index];

            match op {
                Operation::Quantum(qop) => {
                    trace!("Processing quantum operation: {qop:?}");
                    self.operations_buffer.push(qop.clone());
                    self.current_op_index += 1;

                    // Check if this is a measurement operation
                    if let QuantumOp::Measure(_, result_id) = qop {
                        self.pending_measurements.push(*result_id);
                        debug!(
                            "Breaking batch after measurement (result_id={result_id}) to wait for results"
                        );
                        // Break the batch after measurements to get results
                        // This enables dynamic circuits with conditionals
                        break;
                    }

                    // Also break if we've reached the batch size limit
                    if self.operations_buffer.len() >= self.batch_size {
                        debug!("Breaking batch at size limit ({})", self.batch_size);
                        break;
                    }
                }
                Operation::AllocateQubit { id } => {
                    trace!("Allocating qubit {id}");
                    self.current_op_index += 1;
                }
                Operation::AllocateResult { id } => {
                    trace!("Allocating result {id}");
                    self.num_results = self.num_results.max(id + 1);
                    self.current_op_index += 1;
                }
                Operation::ReleaseQubit { id } => {
                    trace!("Releasing qubit {id}");
                    let _ = id; // Just track it
                    self.current_op_index += 1;
                }
                Operation::RecordOutput {
                    result_id,
                    register_name,
                } => {
                    trace!(
                        "Recording output: result_id={result_id}, register_name={register_name}"
                    );
                    // Metadata operation - just advance the index
                    // The actual result mapping is handled by the runtime's results collection
                    self.current_op_index += 1;
                }
                Operation::Barrier => {
                    trace!("Barrier encountered");
                    // Barriers don't produce quantum ops but can break batches
                    self.current_op_index += 1;
                    if !self.operations_buffer.is_empty() {
                        // End current batch at barrier
                        break;
                    }
                }
            }
        }

        if self.operations_buffer.is_empty() {
            Ok(None)
        } else {
            trace!(
                "Returning batch of {} quantum operations",
                self.operations_buffer.len()
            );
            Ok(Some(self.operations_buffer.clone()))
        }
    }

    fn runtime_qubit_for_program(&mut self, program_qubit: usize) -> Result<u64> {
        if let Some(&runtime_qubit) = self.program_to_runtime_qubits.get(&program_qubit) {
            return Ok(runtime_qubit);
        }

        self.load_plugin()?;
        let runtime_qubit = self.runtime_qalloc()?;
        self.program_to_runtime_qubits
            .insert(program_qubit, runtime_qubit);
        if !self.uses_explicit_qubit_allocation {
            self.num_qubits = self.num_qubits.max(program_qubit + 1);
        }
        Ok(runtime_qubit)
    }

    fn runtime_qalloc(&self) -> Result<u64> {
        let lib = self
            .library
            .as_ref()
            .ok_or_else(|| RuntimeError::FfiError("Selene runtime is not loaded".to_string()))?;
        let instance = self.instance.ok_or_else(|| {
            RuntimeError::FfiError("Selene runtime is not initialized".to_string())
        })?;

        unsafe {
            let qalloc_fn = lib
                .get::<unsafe extern "C" fn(RuntimeInstance, *mut u64) -> i32>(
                    b"selene_runtime_qalloc",
                )
                .map_err(|e| RuntimeError::FfiError(format!("Missing qalloc function: {e}")))?;
            let mut runtime_qubit = 0;
            let errno = qalloc_fn(instance, &raw mut runtime_qubit);
            if errno != 0 {
                return Err(RuntimeError::FfiError(format!(
                    "qalloc failed with errno {errno}"
                )));
            }
            if runtime_qubit == u64::MAX {
                return Err(RuntimeError::ExecutionError(
                    "Selene runtime failed to allocate a qubit".to_string(),
                ));
            }
            Ok(runtime_qubit)
        }
    }

    fn runtime_qfree(&self, runtime_qubit: u64) -> Result<()> {
        let Some(lib) = &self.library else {
            return Ok(());
        };
        let Some(instance) = self.instance else {
            return Ok(());
        };

        unsafe {
            let qfree_fn = lib
                .get::<unsafe extern "C" fn(RuntimeInstance, u64) -> i32>(b"selene_runtime_qfree")
                .map_err(|e| RuntimeError::FfiError(format!("Missing qfree function: {e}")))?;
            let errno = qfree_fn(instance, runtime_qubit);
            if errno != 0 {
                return Err(RuntimeError::FfiError(format!(
                    "qfree failed with errno {errno}"
                )));
            }
        }

        Ok(())
    }

    fn call_runtime_rxy(&self, runtime_qubit: u64, theta: f64, phi: f64) -> Result<()> {
        let lib = self
            .library
            .as_ref()
            .ok_or_else(|| RuntimeError::FfiError("Selene runtime is not loaded".to_string()))?;
        let instance = self.instance.ok_or_else(|| {
            RuntimeError::FfiError("Selene runtime is not initialized".to_string())
        })?;

        unsafe {
            let rxy_fn = lib
                .get::<unsafe extern "C" fn(RuntimeInstance, u64, f64, f64) -> i32>(
                    b"selene_runtime_rxy_gate",
                )
                .map_err(|e| RuntimeError::FfiError(format!("Missing rxy function: {e}")))?;
            let errno = rxy_fn(instance, runtime_qubit, theta, phi);
            if errno != 0 {
                return Err(RuntimeError::FfiError(format!(
                    "rxy failed with errno {errno}"
                )));
            }
        }

        Ok(())
    }

    fn call_runtime_rz(&self, runtime_qubit: u64, theta: f64) -> Result<()> {
        let lib = self
            .library
            .as_ref()
            .ok_or_else(|| RuntimeError::FfiError("Selene runtime is not loaded".to_string()))?;
        let instance = self.instance.ok_or_else(|| {
            RuntimeError::FfiError("Selene runtime is not initialized".to_string())
        })?;

        unsafe {
            let rz_fn = lib
                .get::<unsafe extern "C" fn(RuntimeInstance, u64, f64) -> i32>(
                    b"selene_runtime_rz_gate",
                )
                .map_err(|e| RuntimeError::FfiError(format!("Missing rz function: {e}")))?;
            let errno = rz_fn(instance, runtime_qubit, theta);
            if errno != 0 {
                return Err(RuntimeError::FfiError(format!(
                    "rz failed with errno {errno}"
                )));
            }
        }

        Ok(())
    }

    fn call_runtime_rzz(
        &self,
        runtime_qubit_1: u64,
        runtime_qubit_2: u64,
        theta: f64,
    ) -> Result<()> {
        let lib = self
            .library
            .as_ref()
            .ok_or_else(|| RuntimeError::FfiError("Selene runtime is not loaded".to_string()))?;
        let instance = self.instance.ok_or_else(|| {
            RuntimeError::FfiError("Selene runtime is not initialized".to_string())
        })?;

        unsafe {
            let rzz_fn = lib
                .get::<unsafe extern "C" fn(RuntimeInstance, u64, u64, f64) -> i32>(
                    b"selene_runtime_rzz_gate",
                )
                .map_err(|e| RuntimeError::FfiError(format!("Missing rzz function: {e}")))?;
            let errno = rzz_fn(instance, runtime_qubit_1, runtime_qubit_2, theta);
            if errno != 0 {
                return Err(RuntimeError::FfiError(format!(
                    "rzz failed with errno {errno}"
                )));
            }
        }

        Ok(())
    }

    fn call_runtime_reset(&self, runtime_qubit: u64) -> Result<()> {
        let lib = self
            .library
            .as_ref()
            .ok_or_else(|| RuntimeError::FfiError("Selene runtime is not loaded".to_string()))?;
        let instance = self.instance.ok_or_else(|| {
            RuntimeError::FfiError("Selene runtime is not initialized".to_string())
        })?;

        unsafe {
            let reset_fn = lib
                .get::<unsafe extern "C" fn(RuntimeInstance, u64) -> i32>(b"selene_runtime_reset")
                .map_err(|e| RuntimeError::FfiError(format!("Missing reset function: {e}")))?;
            let errno = reset_fn(instance, runtime_qubit);
            if errno != 0 {
                return Err(RuntimeError::FfiError(format!(
                    "reset failed with errno {errno}"
                )));
            }
        }

        Ok(())
    }

    fn call_runtime_measure(&mut self, runtime_qubit: u64, program_result: usize) -> Result<()> {
        let lib = self
            .library
            .as_ref()
            .ok_or_else(|| RuntimeError::FfiError("Selene runtime is not loaded".to_string()))?;
        let instance = self.instance.ok_or_else(|| {
            RuntimeError::FfiError("Selene runtime is not initialized".to_string())
        })?;

        let runtime_result = unsafe {
            let measure_fn = lib
                .get::<unsafe extern "C" fn(RuntimeInstance, u64, *mut u64) -> i32>(
                    b"selene_runtime_measure",
                )
                .map_err(|e| RuntimeError::FfiError(format!("Missing measure function: {e}")))?;
            let mut runtime_result = 0;
            let errno = measure_fn(instance, runtime_qubit, &raw mut runtime_result);
            if errno != 0 {
                return Err(RuntimeError::FfiError(format!(
                    "measure failed with errno {errno}"
                )));
            }
            runtime_result
        };

        self.program_to_runtime_results
            .insert(program_result, runtime_result);
        self.runtime_to_program_results
            .insert(runtime_result, program_result);
        self.force_runtime_result(runtime_result)
    }

    fn force_runtime_result(&self, runtime_result: u64) -> Result<()> {
        let lib = self
            .library
            .as_ref()
            .ok_or_else(|| RuntimeError::FfiError("Selene runtime is not loaded".to_string()))?;
        let instance = self.instance.ok_or_else(|| {
            RuntimeError::FfiError("Selene runtime is not initialized".to_string())
        })?;

        unsafe {
            let force_fn = lib
                .get::<unsafe extern "C" fn(RuntimeInstance, u64) -> i32>(
                    b"selene_runtime_force_result",
                )
                .map_err(|e| {
                    RuntimeError::FfiError(format!("Missing force_result function: {e}"))
                })?;
            let errno = force_fn(instance, runtime_result);
            if errno != 0 {
                return Err(RuntimeError::FfiError(format!(
                    "force_result failed with errno {errno}"
                )));
            }
        }

        Ok(())
    }

    fn submit_operation_to_runtime(
        &mut self,
        op: &Operation,
        lowered_ops: &mut Vec<QuantumOp>,
    ) -> Result<()> {
        match op {
            Operation::AllocateQubit { id } => {
                let _ = self.runtime_qubit_for_program(*id)?;
            }
            Operation::AllocateResult { id } => {
                self.num_results = self.num_results.max(id + 1);
            }
            Operation::ReleaseQubit { id } => {
                if let Some(runtime_qubit) = self.program_to_runtime_qubits.remove(id) {
                    self.runtime_qfree(runtime_qubit)?;
                }
            }
            Operation::RecordOutput { .. } | Operation::Barrier => {}
            Operation::Quantum(qop) => self.submit_quantum_op_to_runtime(qop, lowered_ops)?,
        }

        Ok(())
    }

    fn submit_quantum_op_to_runtime(
        &mut self,
        qop: &QuantumOp,
        lowered_ops: &mut Vec<QuantumOp>,
    ) -> Result<()> {
        match qop {
            QuantumOp::RXY(theta, phi, qubit) => {
                let runtime_qubit = self.runtime_qubit_for_program(*qubit)?;
                self.call_runtime_rxy(runtime_qubit, *theta, *phi)?;
            }
            QuantumOp::RZ(theta, qubit) => {
                let runtime_qubit = self.runtime_qubit_for_program(*qubit)?;
                self.call_runtime_rz(runtime_qubit, *theta)?;
            }
            QuantumOp::RZZ(theta, qubit_1, qubit_2) => {
                let runtime_qubit_1 = self.runtime_qubit_for_program(*qubit_1)?;
                let runtime_qubit_2 = self.runtime_qubit_for_program(*qubit_2)?;
                self.call_runtime_rzz(runtime_qubit_1, runtime_qubit_2, *theta)?;
            }
            QuantumOp::Measure(qubit, result_id) => {
                let runtime_qubit = self.runtime_qubit_for_program(*qubit)?;
                self.call_runtime_measure(runtime_qubit, *result_id)?;
                self.program_to_runtime_qubits.remove(qubit);
                self.runtime_qfree(runtime_qubit)?;
            }
            QuantumOp::Reset(qubit) => {
                let runtime_qubit = self.runtime_qubit_for_program(*qubit)?;
                self.call_runtime_reset(runtime_qubit)?;
            }
            _ => {
                lowered_ops.extend(self.drain_runtime_operations()?);
                lowered_ops.push(self.map_passthrough_op_to_runtime_qubits(qop)?);
            }
        }

        Ok(())
    }

    fn map_passthrough_op_to_runtime_qubits(&mut self, qop: &QuantumOp) -> Result<QuantumOp> {
        let mut map = |qubit: usize| -> Result<usize> {
            let runtime_qubit = self.runtime_qubit_for_program(qubit)?;
            usize::try_from(runtime_qubit).map_err(|_| {
                RuntimeError::ExecutionError(format!(
                    "Runtime qubit id {runtime_qubit} does not fit in usize"
                ))
            })
        };

        Ok(match qop {
            QuantumOp::H(qubit) => QuantumOp::H(map(*qubit)?),
            QuantumOp::X(qubit) => QuantumOp::X(map(*qubit)?),
            QuantumOp::Y(qubit) => QuantumOp::Y(map(*qubit)?),
            QuantumOp::Z(qubit) => QuantumOp::Z(map(*qubit)?),
            QuantumOp::S(qubit) => QuantumOp::S(map(*qubit)?),
            QuantumOp::Sdg(qubit) => QuantumOp::Sdg(map(*qubit)?),
            QuantumOp::T(qubit) => QuantumOp::T(map(*qubit)?),
            QuantumOp::Tdg(qubit) => QuantumOp::Tdg(map(*qubit)?),
            QuantumOp::RX(theta, qubit) => QuantumOp::RX(*theta, map(*qubit)?),
            QuantumOp::RY(theta, qubit) => QuantumOp::RY(*theta, map(*qubit)?),
            QuantumOp::CX(control, target) => QuantumOp::CX(map(*control)?, map(*target)?),
            QuantumOp::CY(control, target) => QuantumOp::CY(map(*control)?, map(*target)?),
            QuantumOp::CZ(control, target) => QuantumOp::CZ(map(*control)?, map(*target)?),
            QuantumOp::CH(control, target) => QuantumOp::CH(map(*control)?, map(*target)?),
            QuantumOp::CRZ(theta, control, target) => {
                QuantumOp::CRZ(*theta, map(*control)?, map(*target)?)
            }
            QuantumOp::CCX(control_1, control_2, target) => {
                QuantumOp::CCX(map(*control_1)?, map(*control_2)?, map(*target)?)
            }
            QuantumOp::ZZ(qubit_1, qubit_2) => QuantumOp::ZZ(map(*qubit_1)?, map(*qubit_2)?),
            QuantumOp::Idle(duration, qubit) => QuantumOp::Idle(*duration, map(*qubit)?),
            QuantumOp::RXY(..)
            | QuantumOp::RZ(..)
            | QuantumOp::RZZ(..)
            | QuantumOp::Measure(..)
            | QuantumOp::Reset(..) => qop.clone(),
        })
    }

    fn drain_runtime_operations(&mut self) -> Result<Vec<QuantumOp>> {
        self.load_plugin()?;
        let mut lowered_ops = Vec::new();

        loop {
            let mut batch = RuntimeOperationBatch::default();
            let errno = {
                let lib = self.library.as_ref().ok_or_else(|| {
                    RuntimeError::FfiError("Selene runtime is not loaded".to_string())
                })?;
                let instance = self.instance.ok_or_else(|| {
                    RuntimeError::FfiError("Selene runtime is not initialized".to_string())
                })?;

                unsafe {
                    let get_next_fn = lib
                        .get::<unsafe extern "C" fn(
                            RuntimeInstance,
                            RuntimeGetOperationInstance,
                            *const SeleneRuntimeGetOperationInterface,
                        ) -> i32>(b"selene_runtime_get_next_operations")
                        .map_err(|e| {
                            RuntimeError::FfiError(format!(
                                "Missing get_next_operations function: {e}"
                            ))
                        })?;
                    get_next_fn(
                        instance,
                        (&raw mut batch).cast::<c_void>(),
                        &raw const RUNTIME_OPERATION_CALLBACKS,
                    )
                }
            };

            if errno != 0 {
                return Err(RuntimeError::FfiError(format!(
                    "get_next_operations failed with errno {errno}"
                )));
            }

            if !batch.invoked {
                break;
            }

            lowered_ops.extend(self.convert_runtime_batch(batch)?);
        }

        Ok(lowered_ops)
    }

    fn convert_runtime_batch(&mut self, batch: RuntimeOperationBatch) -> Result<Vec<QuantumOp>> {
        let mut lowered_ops = Vec::new();
        let start_time = batch.start_time_nanos;
        let end_time = batch.end_time_nanos();

        for op in batch.operations {
            match op {
                RuntimeScheduledOp::Rxy {
                    qubit_id,
                    theta,
                    phi,
                } => {
                    let qubit = self.runtime_qubit_to_usize(qubit_id)?;
                    self.push_idle_before(&mut lowered_ops, qubit, start_time)?;
                    lowered_ops.push(QuantumOp::RXY(theta, phi, qubit));
                    self.mark_gate_end(qubit, end_time);
                }
                RuntimeScheduledOp::Rz { qubit_id, theta } => {
                    let qubit = self.runtime_qubit_to_usize(qubit_id)?;
                    self.push_idle_before(&mut lowered_ops, qubit, start_time)?;
                    lowered_ops.push(QuantumOp::RZ(theta, qubit));
                    self.mark_gate_end(qubit, end_time);
                }
                RuntimeScheduledOp::Rzz {
                    qubit_id_1,
                    qubit_id_2,
                    theta,
                } => {
                    let qubit_1 = self.runtime_qubit_to_usize(qubit_id_1)?;
                    let qubit_2 = self.runtime_qubit_to_usize(qubit_id_2)?;
                    self.push_idle_before(&mut lowered_ops, qubit_1, start_time)?;
                    self.push_idle_before(&mut lowered_ops, qubit_2, start_time)?;
                    lowered_ops.push(QuantumOp::RZZ(theta, qubit_1, qubit_2));
                    self.mark_gate_end(qubit_1, end_time);
                    self.mark_gate_end(qubit_2, end_time);
                }
                RuntimeScheduledOp::Measure {
                    qubit_id,
                    result_id,
                }
                | RuntimeScheduledOp::MeasureLeaked {
                    qubit_id,
                    result_id,
                } => {
                    let qubit = self.runtime_qubit_to_usize(qubit_id)?;
                    let program_result = self.runtime_result_to_program_result(result_id)?;
                    self.push_idle_before(&mut lowered_ops, qubit, start_time)?;
                    lowered_ops.push(QuantumOp::Measure(qubit, program_result));
                    self.mark_gate_end(qubit, end_time);
                }
                RuntimeScheduledOp::Reset { qubit_id } => {
                    let qubit = self.runtime_qubit_to_usize(qubit_id)?;
                    lowered_ops.push(QuantumOp::Reset(qubit));
                    self.mark_gate_end(qubit, end_time);
                }
                RuntimeScheduledOp::Custom => {}
            }
        }

        Ok(lowered_ops)
    }

    fn runtime_qubit_to_usize(&mut self, runtime_qubit: u64) -> Result<usize> {
        let qubit = usize::try_from(runtime_qubit).map_err(|_| {
            RuntimeError::ExecutionError(format!(
                "Runtime qubit id {runtime_qubit} does not fit in usize"
            ))
        })?;
        self.ensure_timing_slot(qubit);
        Ok(qubit)
    }

    fn runtime_result_to_program_result(&self, runtime_result: u64) -> Result<usize> {
        if let Some(&program_result) = self.runtime_to_program_results.get(&runtime_result) {
            return Ok(program_result);
        }

        usize::try_from(runtime_result).map_err(|_| {
            RuntimeError::ExecutionError(format!(
                "Runtime result id {runtime_result} does not fit in usize"
            ))
        })
    }

    fn ensure_timing_slot(&mut self, qubit: usize) {
        if self.last_gate_time_end_nanos.len() <= qubit {
            self.last_gate_time_end_nanos.resize(qubit + 1, 0);
        }
    }

    fn push_idle_before(
        &mut self,
        lowered_ops: &mut Vec<QuantumOp>,
        qubit: usize,
        start_time_nanos: u64,
    ) -> Result<()> {
        self.ensure_timing_slot(qubit);
        let last_gate_end = self.last_gate_time_end_nanos[qubit];
        if last_gate_end > start_time_nanos {
            return Err(RuntimeError::ExecutionError(format!(
                "Runtime operation on qubit {qubit} starts before its previous operation ended: {start_time_nanos} < {last_gate_end}"
            )));
        }

        let idle_time = start_time_nanos - last_gate_end;
        if idle_time > 0 {
            lowered_ops.push(QuantumOp::Idle(nanoseconds_to_seconds(idle_time), qubit));
        }

        Ok(())
    }

    fn mark_gate_end(&mut self, qubit: usize, end_time_nanos: u64) {
        self.ensure_timing_slot(qubit);
        self.last_gate_time_end_nanos[qubit] = end_time_nanos;
    }
}

fn nanoseconds_to_seconds(nanoseconds: u64) -> f64 {
    std::time::Duration::from_nanos(nanoseconds).as_secs_f64()
}

impl Clone for SeleneRuntime {
    fn clone(&self) -> Self {
        // For now, create a new instance with the same plugin path
        // The library itself can't be cloned, so we'll reload if needed
        Self {
            plugin_path: self.plugin_path.clone(),
            init_args: self.init_args.clone(),
            library_search_dirs: self.library_search_dirs.clone(),
            library: None,  // Will be reloaded on demand
            instance: None, // Will be recreated on demand
            initialized_num_qubits: None,
            state: self.state.clone(),
            operations_buffer: self.operations_buffer.clone(),
            batch_size: self.batch_size,
            num_qubits: self.num_qubits,
            num_qubits_hint: self.num_qubits_hint,
            uses_explicit_qubit_allocation: self.uses_explicit_qubit_allocation,
            num_results: self.num_results,
            interface: self.interface.clone(),
            current_op_index: self.current_op_index,
            needs_reexecution: self.needs_reexecution,
            pending_measurements: self.pending_measurements.clone(),
            program_to_runtime_qubits: self.program_to_runtime_qubits.clone(),
            program_to_runtime_results: self.program_to_runtime_results.clone(),
            runtime_to_program_results: self.runtime_to_program_results.clone(),
            last_gate_time_end_nanos: self.last_gate_time_end_nanos.clone(),
            pending_shot_start: self.pending_shot_start,
        }
    }
}

fn collector_capacity(interface: &OperationCollector) -> (usize, usize) {
    let uses_explicit_allocations = has_explicit_qubit_allocations(&interface.operations);
    let (mut num_qubits, mut num_results) =
        operation_capacity_with_mode(&interface.operations, uses_explicit_allocations);

    if !uses_explicit_allocations {
        for &qubit in &interface.allocated_qubits {
            include_qubit(&mut num_qubits, qubit);
        }
    }
    for &result in &interface.allocated_results {
        include_result(&mut num_results, result);
    }

    (num_qubits, num_results)
}

fn has_explicit_qubit_allocations(operations: &[Operation]) -> bool {
    operations.iter().any(|op| {
        matches!(
            op,
            Operation::AllocateQubit { .. } | Operation::ReleaseQubit { .. }
        )
    })
}

fn operation_capacity_with_mode(
    operations: &[Operation],
    uses_explicit_allocations: bool,
) -> (usize, usize) {
    let mut num_qubits = 0;
    let mut num_results = 0;
    let mut live_qubits = BTreeSet::new();
    let mut max_live_qubits = 0;

    for op in operations {
        match op {
            Operation::Quantum(qop) if !uses_explicit_allocations => {
                include_quantum_op_capacity(qop, &mut num_qubits, &mut num_results)
            }
            Operation::Quantum(qop) => {
                include_quantum_result_capacity(qop, &mut num_results);
            }
            Operation::AllocateQubit { id } => {
                live_qubits.insert(*id);
                max_live_qubits = max_live_qubits.max(live_qubits.len());
            }
            Operation::ReleaseQubit { id } => {
                live_qubits.remove(id);
            }
            Operation::AllocateResult { id } => include_result(&mut num_results, *id),
            Operation::RecordOutput { result_id, .. } => {
                include_result(&mut num_results, *result_id);
            }
            Operation::Barrier => {}
        }
    }
    if uses_explicit_allocations {
        num_qubits = max_live_qubits;
    }

    (num_qubits, num_results)
}

fn include_quantum_result_capacity(qop: &QuantumOp, num_results: &mut usize) {
    if let QuantumOp::Measure(_, result) = qop {
        include_result(num_results, *result);
    }
}

fn include_quantum_op_capacity(qop: &QuantumOp, num_qubits: &mut usize, num_results: &mut usize) {
    match qop {
        QuantumOp::H(qubit)
        | QuantumOp::X(qubit)
        | QuantumOp::Y(qubit)
        | QuantumOp::Z(qubit)
        | QuantumOp::S(qubit)
        | QuantumOp::Sdg(qubit)
        | QuantumOp::T(qubit)
        | QuantumOp::Tdg(qubit)
        | QuantumOp::RX(_, qubit)
        | QuantumOp::RY(_, qubit)
        | QuantumOp::RZ(_, qubit)
        | QuantumOp::RXY(_, _, qubit)
        | QuantumOp::Idle(_, qubit)
        | QuantumOp::Reset(qubit) => include_qubit(num_qubits, *qubit),
        QuantumOp::CX(qubit_1, qubit_2)
        | QuantumOp::CY(qubit_1, qubit_2)
        | QuantumOp::CZ(qubit_1, qubit_2)
        | QuantumOp::CH(qubit_1, qubit_2)
        | QuantumOp::CRZ(_, qubit_1, qubit_2)
        | QuantumOp::ZZ(qubit_1, qubit_2)
        | QuantumOp::RZZ(_, qubit_1, qubit_2) => {
            include_qubit(num_qubits, *qubit_1);
            include_qubit(num_qubits, *qubit_2);
        }
        QuantumOp::CCX(qubit_1, qubit_2, qubit_3) => {
            include_qubit(num_qubits, *qubit_1);
            include_qubit(num_qubits, *qubit_2);
            include_qubit(num_qubits, *qubit_3);
        }
        QuantumOp::Measure(qubit, result) => {
            include_qubit(num_qubits, *qubit);
            include_result(num_results, *result);
        }
    }
}

fn include_qubit(num_qubits: &mut usize, qubit: usize) {
    *num_qubits = (*num_qubits).max(qubit + 1);
}

fn include_result(num_results: &mut usize, result: usize) {
    *num_results = (*num_results).max(result + 1);
}

impl QisRuntime for SeleneRuntime {
    fn load_interface(&mut self, interface: OperationCollector) -> Result<()> {
        debug!(
            "Loading QIS interface with {} operations",
            interface.operations.len()
        );

        // Count qubits from explicit allocation records when present,
        // otherwise from direct program handles. Some legacy LLVM/QIR inputs
        // use qubit handles like 0 and 1 without emitting allocation calls.
        let (num_qubits, num_results) = collector_capacity(&interface);
        self.num_qubits = num_qubits;
        self.num_results = num_results;
        self.uses_explicit_qubit_allocation = has_explicit_qubit_allocations(&interface.operations);

        debug!(
            "Interface has {} qubits and {} result slots",
            self.num_qubits, self.num_results
        );

        self.interface = Some(interface);
        self.current_op_index = 0;
        self.needs_reexecution = false;
        self.pending_measurements.clear();

        // Don't load the plugin yet - defer until actually needed
        // This allows creating and testing the runtime without a real .so file

        Ok(())
    }

    fn execute_until_quantum(&mut self) -> Result<Option<Vec<QuantumOp>>> {
        // For now, we'll use the simple approach of processing from the interface
        // In a full implementation, we'd call into the Selene runtime's
        // get_next_operations function
        self.process_interface_ops()
    }

    fn supports_operation_lowering(&self) -> bool {
        true
    }

    fn lower_operations(&mut self, operations: &[Operation]) -> Result<Vec<QuantumOp>> {
        if has_explicit_qubit_allocations(operations) {
            self.uses_explicit_qubit_allocation = true;
        }
        let (num_qubits, num_results) =
            operation_capacity_with_mode(operations, self.uses_explicit_qubit_allocation);
        self.num_qubits = self.num_qubits.max(num_qubits);
        self.num_results = self.num_results.max(num_results);
        self.ensure_plugin_capacity()?;
        self.load_plugin()?;
        let mut lowered_ops = Vec::new();

        for op in operations {
            self.submit_operation_to_runtime(op, &mut lowered_ops)?;
        }

        lowered_ops.extend(self.drain_runtime_operations()?);
        Ok(lowered_ops)
    }

    fn provide_measurements(&mut self, measurements: BTreeMap<usize, bool>) -> Result<()> {
        debug!(
            "Received {} measurement results, num_results={}, allocated_results={:?}",
            measurements.len(),
            self.num_results,
            self.interface.as_ref().map(|i| &i.allocated_results)
        );

        // Store measurements in classical state
        for (result_id, value) in &measurements {
            trace!(
                "Measurement result {} = {} (num_results={})",
                result_id, value, self.num_results
            );
            self.state.measurements.insert(*result_id, *value);

            if let Some(runtime_result_id) = self.program_to_runtime_results.get(result_id) {
                if let Some(lib) = &self.library
                    && let Some(instance) = self.instance
                {
                    unsafe {
                        if let Ok(set_result_fn) =
                            lib.get::<unsafe extern "C" fn(*mut c_void, u64, bool) -> i32>(
                                b"selene_runtime_set_bool_result",
                            )
                        {
                            let errno = set_result_fn(instance, *runtime_result_id, *value);
                            if errno != 0 {
                                log::trace!(
                                    "Selene runtime returned error {errno} for result {result_id}"
                                );
                            }
                        }
                    }
                }
            } else {
                log::trace!(
                    "Measurement result {result_id} was not allocated by the Selene runtime, storing locally only"
                );
            }

            if let Some(interface) = &mut self.interface {
                interface.store_result(*result_id, *value);
            }
        }

        // Check if there are remaining operations that might depend on these measurements
        // If so, we need to re-execute the program with the known measurement values
        // so that conditionals can evaluate correctly
        if let Some(interface) = &self.interface {
            let remaining_ops = interface
                .operations
                .len()
                .saturating_sub(self.current_op_index);
            if remaining_ops > 0 && !measurements.is_empty() {
                debug!(
                    "Setting needs_reexecution=true: {} ops remaining after {} measurements",
                    remaining_ops,
                    measurements.len()
                );
                self.needs_reexecution = true;
            }
        }

        Ok(())
    }

    fn get_classical_state(&self) -> &ClassicalState {
        &self.state
    }

    fn get_classical_state_mut(&mut self) -> &mut ClassicalState {
        &mut self.state
    }

    fn is_complete(&self) -> bool {
        self.interface
            .as_ref()
            .is_none_or(|i| self.current_op_index >= i.operations.len())
    }

    fn num_qubits(&self) -> usize {
        self.plugin_num_qubits()
    }

    fn set_num_qubits(&mut self, num_qubits: usize) {
        self.num_qubits_hint = Some(num_qubits);
        self.num_qubits = self.num_qubits.max(num_qubits);
    }

    fn set_batch_size(&mut self, size: usize) {
        self.batch_size = size;
    }

    fn needs_reexecution(&self) -> bool {
        self.needs_reexecution
    }

    fn clear_reexecution_flag(&mut self) {
        self.needs_reexecution = false;
    }

    fn reload_operations(&mut self, operations: OperationCollector) {
        SeleneRuntime::reload_operations(self, operations);
    }

    fn shot_start(&mut self, shot_id: u64, seed: Option<u64>) -> Result<()> {
        // Reset state for new shot
        self.state = ClassicalState::default();
        self.current_op_index = 0;
        self.needs_reexecution = false;
        self.pending_measurements.clear();
        self.program_to_runtime_qubits.clear();
        self.program_to_runtime_results.clear();
        self.runtime_to_program_results.clear();
        self.last_gate_time_end_nanos.clear();
        self.pending_shot_start = Some((shot_id, seed));
        self.apply_pending_shot_start()?;

        Ok(())
    }

    fn shot_end(&mut self) -> Result<Shot> {
        if let Some(lib) = &self.library
            && let Some(instance) = self.instance
        {
            unsafe {
                if let Ok(shot_end_fn) =
                    lib.get::<unsafe extern "C" fn(*mut c_void) -> i32>(b"selene_runtime_shot_end")
                {
                    let _ = shot_end_fn(instance);
                }
            }
        }
        self.pending_shot_start = None;

        // Return the shot with measurements and registers
        let shot = Shot {
            measurements: self.state.measurements.clone(),
            registers: self.state.registers.clone(),
            ..Default::default()
        };

        Ok(shot)
    }

    fn reset(&mut self) -> Result<()> {
        self.reset_plugin_instance()?;
        self.state = ClassicalState::default();
        self.current_op_index = 0;
        self.program_to_runtime_qubits.clear();
        self.program_to_runtime_results.clear();
        self.runtime_to_program_results.clear();
        self.last_gate_time_end_nanos.clear();
        self.pending_shot_start = None;

        Ok(())
    }
}

impl Drop for SeleneRuntime {
    fn drop(&mut self) {
        // Intentionally skip cleanup during drop.
        //
        // IMPORTANT: The FFI call to selene_runtime_exit in reset() can hang
        // during process shutdown because:
        // 1. Thread-local storage may already be partially torn down
        // 2. Other static destructors may be running concurrently
        // 3. The library's internal state may be inconsistent
        //
        // Since drop() is typically called during process exit, it's safe to skip
        // the cleanup and let the OS reclaim all resources. This avoids the
        // intermittent hang that was occurring ~15-20% of the time when running
        // tests in parallel.
        //
        // During normal operation (not process exit), call reset() explicitly
        // before dropping if cleanup is needed.

        // Just clear our local state without making FFI calls
        self.instance = None;
        // Note: We intentionally don't set self.library = None here because
        // the Arc<Library> might be shared, and we don't want to trigger
        // dlclose() during process exit.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selene_runtime_creation() {
        let runtime = SeleneRuntime::new("/path/to/selene.so");
        assert_eq!(runtime.num_qubits(), 0);
        assert!(runtime.is_complete());
    }

    #[test]
    fn test_selene_runtime_plugin_config_clones() {
        let runtime = SeleneRuntime::with_plugin_config(
            "/path/to/selene.so",
            vec!["--duration-ns-rxy=10".to_string()],
            vec![PathBuf::from("/path/to/lib")],
        );
        let cloned = runtime.clone();
        assert_eq!(cloned.init_args, ["--duration-ns-rxy=10"]);
        assert_eq!(cloned.library_search_dirs, [PathBuf::from("/path/to/lib")]);
    }

    #[test]
    fn test_runtime_batch_timing_inserts_idle() {
        let mut runtime = SeleneRuntime::new("/path/to/selene.so");
        let batch = RuntimeOperationBatch {
            start_time_nanos: 20,
            duration_nanos: 5,
            invoked: true,
            operations: vec![RuntimeScheduledOp::Rxy {
                qubit_id: 0,
                theta: 1.0,
                phi: 0.5,
            }],
        };

        let ops = runtime.convert_runtime_batch(batch).unwrap();
        assert_eq!(
            ops,
            vec![QuantumOp::Idle(20e-9, 0), QuantumOp::RXY(1.0, 0.5, 0)]
        );
    }

    #[test]
    fn test_collector_capacity_includes_direct_program_handles() {
        let mut collector = OperationCollector::new();
        collector.queue_operation(QuantumOp::H(0).into());
        collector.queue_operation(QuantumOp::CX(0, 3).into());
        collector.queue_operation(QuantumOp::Measure(3, 7).into());
        collector.queue_operation(Operation::RecordOutput {
            result_id: 7,
            register_name: "c".to_string(),
        });

        assert_eq!(collector_capacity(&collector), (4, 8));
    }

    #[test]
    fn test_collector_capacity_includes_explicit_allocations() {
        let mut collector = OperationCollector::new();
        collector.queue_operation(Operation::AllocateQubit { id: 5 });
        collector.queue_operation(Operation::AllocateResult { id: 2 });
        collector.queue_operation(QuantumOp::H(5).into());

        assert_eq!(collector_capacity(&collector), (1, 3));
    }

    #[test]
    fn test_collector_capacity_uses_max_live_explicit_allocations() {
        let mut collector = OperationCollector::new();
        collector.queue_operation(Operation::AllocateQubit { id: 81 });
        collector.queue_operation(Operation::AllocateQubit { id: 97 });
        collector.queue_operation(QuantumOp::CX(81, 97).into());
        collector.queue_operation(Operation::ReleaseQubit { id: 97 });
        collector.queue_operation(Operation::AllocateQubit { id: 105 });
        collector.queue_operation(QuantumOp::Measure(105, 9).into());

        assert_eq!(collector_capacity(&collector), (2, 10));
    }

    #[test]
    fn test_explicit_qubit_hint_caps_plugin_capacity() {
        let mut runtime = SeleneRuntime::new("/path/to/selene.so");
        runtime.set_num_qubits(98);

        let (num_qubits, _) = operation_capacity_with_mode(
            &[QuantumOp::CX(81, 105).into()],
            runtime.uses_explicit_qubit_allocation,
        );
        runtime.num_qubits = runtime.num_qubits.max(num_qubits);

        assert_eq!(runtime.num_qubits, 106);
        assert_eq!(runtime.plugin_num_qubits(), 98);
        assert_eq!(runtime.num_qubits(), 98);
    }

    #[test]
    fn test_shot_start_defers_until_plugin_load() {
        let mut runtime = SeleneRuntime::new("/path/to/selene.so");
        runtime.shot_start(42, Some(1234)).unwrap();

        assert_eq!(runtime.pending_shot_start, Some((42, Some(1234))));

        runtime.shot_end().unwrap();
        assert_eq!(runtime.pending_shot_start, None);
    }

    #[test]
    fn test_clone_does_not_reuse_initialized_plugin_capacity() {
        let mut runtime = SeleneRuntime::new("/path/to/selene.so");
        runtime.num_qubits = 3;
        runtime.initialized_num_qubits = Some(3);

        let cloned = runtime.clone();

        assert_eq!(cloned.num_qubits, 3);
        assert_eq!(cloned.initialized_num_qubits, None);
    }

    #[test]
    fn test_reset_clears_initialized_plugin_capacity() {
        let mut runtime = SeleneRuntime::new("/path/to/selene.so");
        runtime.initialized_num_qubits = Some(3);

        runtime.reset().unwrap();

        assert_eq!(runtime.initialized_num_qubits, None);
    }
}
