//! Selene Runtime implementation of `QisRuntime`
//!
//! This wraps a Selene .so runtime plugin and implements the `QisRuntime` trait
//! to provide a Selene-based classical interpreter for QIS programs.

use crate::runtime::{ClassicalState, QisRuntime, Result, RuntimeError, Shot};
use crate::scheduled::{
    MAX_OPERATIONS, MAX_PAYLOAD_BYTES, RuntimeScheduledOp, ScheduledBatch, ScheduledMeasurement,
    ScheduledOutput,
};
use log::{debug, trace};
use pecos_qis_ffi_types::{
    LoweredQuantumOp, Operation, OperationCollector, QuantumOp, TraceMetadata,
};
use selene_core::runtime::{RuntimeAPIVersion, plugin::RuntimePluginDescriptorV1};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::ffi::{CString, c_void};
use std::mem::{ManuallyDrop, size_of};
use std::path::{Path, PathBuf};
use std::sync::Arc;

type RuntimeInstance = *mut c_void;
type RuntimeGetOperationInstance = *mut c_void;

#[derive(Debug, Default)]
struct RuntimeOperationBatch {
    start_time_nanos: u64,
    duration_nanos: u64,
    invoked: bool,
    callback_error: Option<&'static str>,
    operations: Vec<RuntimeScheduledOp>,
    extraction_budget: Option<(usize, usize)>,
    payload_bytes: usize,
}

impl RuntimeOperationBatch {
    // Every callback must stop allocating after the first failure. Reserve
    // explicitly so operation-vector growth cannot panic/abort through Vec::push.
    fn reserve_operations(&mut self, additional: usize) -> bool {
        self.invoked = true;
        if self.callback_error.is_some() {
            return false;
        }
        if self
            .extraction_budget
            .is_some_and(|(limit, _)| self.operations.len().saturating_add(additional) > limit)
        {
            self.callback_error = Some("scheduled operation budget exceeded");
        }
        if self.callback_error.is_some() {
            return false;
        }
        if self.operations.try_reserve(additional).is_err() {
            self.callback_error = Some("unable to allocate runtime operation storage");
            return false;
        }
        true
    }

    fn push(&mut self, operation: RuntimeScheduledOp) {
        if self.reserve_operations(1) {
            self.operations.push(operation);
        }
    }

    fn end_time_nanos(&self) -> u64 {
        self.start_time_nanos.saturating_add(self.duration_nanos)
    }
}

#[derive(Debug, Clone)]
struct SourceTraceMetadata {
    op: QuantumOp,
    metadata: TraceMetadata,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SeleneRuntimeGetOperationInterface {
    measure: unsafe extern "C" fn(RuntimeGetOperationInstance, u64, u64),
    measure_leaked: unsafe extern "C" fn(RuntimeGetOperationInstance, u64, u64),
    reset: unsafe extern "C" fn(RuntimeGetOperationInstance, u64),
    custom: unsafe extern "C" fn(RuntimeGetOperationInstance, usize, *const c_void, usize),
    set_batch_time: unsafe extern "C" fn(RuntimeGetOperationInstance, u64, u64),
    rzz: unsafe extern "C" fn(RuntimeGetOperationInstance, u64, u64, f64),
    rxy: unsafe extern "C" fn(RuntimeGetOperationInstance, u64, f64, f64),
    rz: unsafe extern "C" fn(RuntimeGetOperationInstance, u64, f64),
    rpp: unsafe extern "C" fn(RuntimeGetOperationInstance, u64, u64, f64, f64),
}

#[repr(C)]
struct SeleneRuntimeGetOperationHandle {
    instance: RuntimeGetOperationInstance,
    interface: SeleneRuntimeGetOperationInterface,
}

unsafe extern "C" fn runtime_batch_rxy(
    instance: RuntimeGetOperationInstance,
    qubit_id: u64,
    theta: f64,
    phi: f64,
) {
    let batch = unsafe { &mut *(instance.cast::<RuntimeOperationBatch>()) };
    batch.push(RuntimeScheduledOp::Rxy {
        qubit_id,
        theta,
        phi,
    });
    batch.invoked = true;
}

unsafe extern "C" fn runtime_batch_rz(
    instance: RuntimeGetOperationInstance,
    qubit_id: u64,
    theta: f64,
) {
    let batch = unsafe { &mut *(instance.cast::<RuntimeOperationBatch>()) };
    batch.push(RuntimeScheduledOp::Rz { qubit_id, theta });
    batch.invoked = true;
}

unsafe extern "C" fn runtime_batch_rzz(
    instance: RuntimeGetOperationInstance,
    qubit_id_1: u64,
    qubit_id_2: u64,
    theta: f64,
) {
    let batch = unsafe { &mut *(instance.cast::<RuntimeOperationBatch>()) };
    batch.push(RuntimeScheduledOp::Rzz {
        qubit_id_1,
        qubit_id_2,
        theta,
    });
    batch.invoked = true;
}

unsafe extern "C" fn runtime_batch_measure(
    instance: RuntimeGetOperationInstance,
    qubit_id: u64,
    result_id: u64,
) {
    let batch = unsafe { &mut *(instance.cast::<RuntimeOperationBatch>()) };
    batch.push(RuntimeScheduledOp::Measure {
        qubit_id,
        result_id,
    });
    batch.invoked = true;
}

unsafe extern "C" fn runtime_batch_measure_leaked(
    instance: RuntimeGetOperationInstance,
    qubit_id: u64,
    result_id: u64,
) {
    let batch = unsafe { &mut *(instance.cast::<RuntimeOperationBatch>()) };
    batch.push(RuntimeScheduledOp::MeasureLeaked {
        qubit_id,
        result_id,
    });
    batch.invoked = true;
}

unsafe extern "C" fn runtime_batch_reset(instance: RuntimeGetOperationInstance, qubit_id: u64) {
    let batch = unsafe { &mut *(instance.cast::<RuntimeOperationBatch>()) };
    batch.push(RuntimeScheduledOp::Reset { qubit_id });
    batch.invoked = true;
}

// The plugin must return our live batch pointer and provide readable payload
// memory for the duration of this call, as in Selene's BatchExtractor. Foreign
// non-null dangling pointers cannot be validated by the receiver.
unsafe extern "C" fn runtime_batch_custom(
    instance: RuntimeGetOperationInstance,
    tag: usize,
    data: *const c_void,
    data_len: usize,
) {
    let batch = unsafe { &mut *(instance.cast::<RuntimeOperationBatch>()) };
    batch.invoked = true;
    if batch.callback_error.is_some() {
        return;
    }
    if data_len > isize::MAX as usize || (data_len != 0 && data.is_null()) {
        batch.callback_error = Some("invalid custom-event payload pointer/length");
        return;
    }
    if batch
        .extraction_budget
        .is_some_and(|(_, limit)| data_len > limit.saturating_sub(batch.payload_bytes))
    {
        batch.callback_error = Some("scheduled payload budget exceeded");
        return;
    }
    if !batch.reserve_operations(1) {
        return;
    }
    batch.payload_bytes = batch.payload_bytes.saturating_add(data_len);
    let mut owned = Vec::new();
    if owned.try_reserve_exact(data_len).is_err() {
        batch.callback_error = Some("unable to allocate custom-event storage");
        return;
    }
    if data_len != 0 {
        // SAFETY: The plugin owns a readable allocation of data_len bytes until
        // this callback returns. The size and null checks above precede slicing.
        owned.extend_from_slice(unsafe { std::slice::from_raw_parts(data.cast::<u8>(), data_len) });
    }
    batch
        .operations
        .push(RuntimeScheduledOp::Custom { tag, data: owned });
}

unsafe extern "C" fn runtime_batch_set_time(
    instance: RuntimeGetOperationInstance,
    start_time_nanos: u64,
    duration_nanos: u64,
) {
    let batch = unsafe { &mut *(instance.cast::<RuntimeOperationBatch>()) };
    batch.start_time_nanos = start_time_nanos;
    batch.duration_nanos = duration_nanos;
    batch.invoked = true;
}

unsafe extern "C" fn runtime_batch_rpp(
    instance: RuntimeGetOperationInstance,
    qubit_id_1: u64,
    qubit_id_2: u64,
    theta: f64,
    phi: f64,
) {
    let batch = unsafe { &mut *(instance.cast::<RuntimeOperationBatch>()) };
    batch.push(RuntimeScheduledOp::Rpp {
        qubit_id_1,
        qubit_id_2,
        theta,
        phi,
    });
    batch.invoked = true;
}

static RUNTIME_OPERATION_CALLBACKS: SeleneRuntimeGetOperationInterface =
    SeleneRuntimeGetOperationInterface {
        measure: runtime_batch_measure,
        measure_leaked: runtime_batch_measure_leaked,
        reset: runtime_batch_reset,
        custom: runtime_batch_custom,
        set_batch_time: runtime_batch_set_time,
        rzz: runtime_batch_rzz,
        rxy: runtime_batch_rxy,
        rz: runtime_batch_rz,
        rpp: runtime_batch_rpp,
    };

/// An owned opaque event emitted by a Selene runtime, before QIS lowering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeCustomEvent {
    /// Plugin-defined identity; PECOS assigns no meaning to tags.
    pub tag: usize,
    /// Bytes copied before the plugin callback returns.
    pub data: Vec<u8>,
    /// Zero-based runtime batch ordinal within this shot (including ordinary batches).
    pub batch_index: usize,
    /// Zero-based operation position within the original runtime batch.
    pub operation_index: usize,
    /// Original batch start time, in nanoseconds.
    pub start_time_nanos: u64,
    /// Original batch duration, in nanoseconds.
    pub duration_nanos: u64,
}

/// Compatibility policy for events not explicitly recognized as metadata.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RuntimeCustomEventPolicy {
    /// Retain events for inspection without changing ordinary execution (default).
    #[default]
    Capture,
    /// Fail on every event not acknowledged as metadata-only by the handler.
    RejectUnhandled,
}

/// A metadata handler must not acknowledge an unmodeled physical effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeCustomEventDisposition {
    /// The event is understood and has no physical effect to simulate.
    MetadataOnly,
    /// Unknown schema or a physical effect unsupported by this integration.
    Unsupported,
}

type CustomEventHandler =
    Arc<dyn Fn(&RuntimeCustomEvent) -> Result<RuntimeCustomEventDisposition> + Send + Sync>;

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
    /// if the program actually exceeds it, capacity admission rejects the input.
    num_qubits_hint: Option<usize>,

    /// Live program handles whose lifetime was explicitly introduced by qalloc.
    /// Other materialized handles retain legacy release-on-measure semantics.
    explicit_qubit_handles: BTreeSet<usize>,

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

    /// Program results produced by leakage-aware measurements.
    leakage_results: BTreeSet<usize>,

    /// End timestamp of the last scheduled physical operation per runtime qubit.
    last_gate_time_end_nanos: Vec<u64>,

    /// Shot metadata waiting for a lazily loaded runtime plugin.
    pending_shot_start: Option<(u64, Option<u64>)>,

    /// Shot identity actually delivered to the plugin via
    /// `selene_runtime_shot_start`; `selene_runtime_shot_end` takes the same
    /// `(shot_id, seed)` pair, so it must be retained until shot end.
    active_shot: Option<(u64, u64)>,
    custom_events: Vec<RuntimeCustomEvent>,
    runtime_batch_index: usize,
    custom_event_policy: RuntimeCustomEventPolicy,
    custom_event_handler: Option<CustomEventHandler>,
    batch_failure: Option<RuntimeError>,
    scheduled_mode: Option<bool>,
    scheduled_terminal_drained: bool,
    scheduled_output: Option<ScheduledOutput>,
}

// SAFETY: SeleneRuntime owns its instance pointer exclusively.
// WARNING: The Selene FFI runtime may not be thread-safe for concurrent access.
// Sync is required by the QisRuntime/Engine trait but callers must ensure
// single-threaded access to any given instance.
unsafe impl Send for SeleneRuntime {}
unsafe impl Sync for SeleneRuntime {}

impl SeleneRuntime {
    /// Load the Selene 0.3 runtime-plugin descriptor from a library.
    ///
    /// Selene 0.3 intentionally exposes its ABI through one descriptor entry
    /// point instead of exporting every runtime callback as a global symbol.
    unsafe fn runtime_plugin_descriptor(
        library: &libloading::Library,
    ) -> Result<RuntimePluginDescriptorV1> {
        let descriptor = if let Ok(symbol) =
            unsafe { library.get::<*const c_void>(b"selene_runtime_plugin_descriptor_v1") }
        {
            unsafe { symbol.try_as_raw_ptr() }
                .map(|ptr| ptr.cast_const().cast::<RuntimePluginDescriptorV1>())
        } else {
            unsafe {
                library.get::<unsafe extern "C" fn() -> *const RuntimePluginDescriptorV1>(
                    b"selene_runtime_get_plugin_descriptor_v1",
                )
            }
            .ok()
            .map(|accessor| unsafe { accessor() })
        };
        let Some(descriptor) = descriptor.filter(|descriptor| !descriptor.is_null()) else {
            return Err(RuntimeError::FfiError(
                "runtime plugin does not expose a Selene 0.3 descriptor".to_string(),
            ));
        };

        // Read only the header until the plugin establishes that its full
        // descriptor is present. This prevents a short older descriptor from
        // being materialized as the current ABI type.
        let struct_size = unsafe { std::ptr::read_unaligned(descriptor.cast::<u64>()) };
        let expected_size = size_of::<RuntimePluginDescriptorV1>() as u64;
        if struct_size < expected_size {
            return Err(RuntimeError::FfiError(format!(
                "runtime plugin descriptor is too small for Selene 0.3: expected at least {expected_size} bytes, got {struct_size}"
            )));
        }

        let descriptor = unsafe { std::ptr::read_unaligned(descriptor) };
        RuntimeAPIVersion::from(descriptor.api_version)
            .validate()
            .map_err(|error| {
                RuntimeError::FfiError(format!(
                    "runtime plugin has an incompatible Selene API version: {error}"
                ))
            })?;
        Ok(descriptor)
    }

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
            explicit_qubit_handles: BTreeSet::new(),
            num_results: 0,
            interface: None,
            current_op_index: 0,
            needs_reexecution: false,
            pending_measurements: Vec::new(),
            program_to_runtime_qubits: BTreeMap::new(),
            program_to_runtime_results: BTreeMap::new(),
            runtime_to_program_results: BTreeMap::new(),
            leakage_results: BTreeSet::new(),
            last_gate_time_end_nanos: Vec::new(),
            pending_shot_start: None,
            active_shot: None,
            custom_events: Vec::new(),
            runtime_batch_index: 0,
            custom_event_policy: RuntimeCustomEventPolicy::default(),
            custom_event_handler: None,
            batch_failure: None,
            scheduled_mode: None,
            scheduled_terminal_drained: false,
            scheduled_output: None,
        }
    }

    fn lower_native_operations(&mut self, operations: &[Operation]) -> Result<Vec<QuantumOp>> {
        self.check_batch_failure()?;
        self.prepare_runtime_input(operations)?;
        self.with_native_mutation(|runtime| {
            let mut lowered_ops = Vec::new();

            for op in operations {
                if matches!(op, Operation::Barrier) {
                    lowered_ops.extend(runtime.lower_runtime_barrier()?);
                }
                runtime.submit_operation_to_runtime(op, &mut lowered_ops)?;
            }

            lowered_ops.extend(runtime.drain_runtime_operations()?);
            Ok(lowered_ops)
        })
    }

    fn submit_metadata_operations(
        &mut self,
        operations: &[Operation],
    ) -> Result<Vec<LoweredQuantumOp>> {
        let mut lowered_ops = Vec::new();
        let mut source_metadata = VecDeque::new();
        let mut pending_global_metadata = TraceMetadata::new();
        let mut pending_qubit_metadata: BTreeMap<usize, TraceMetadata> = BTreeMap::new();

        for op in operations {
            match op {
                Operation::TraceMetadata { metadata, qubit } => {
                    if let Some(qubit) = qubit {
                        let pending = pending_qubit_metadata.entry(*qubit).or_default();
                        Self::merge_trace_metadata(pending, metadata.clone())?;
                    } else {
                        Self::merge_trace_metadata(&mut pending_global_metadata, metadata.clone())?;
                    }
                }
                Operation::Quantum(qop) => {
                    let metadata = Self::take_pending_trace_metadata_for_source_op(
                        qop,
                        &mut pending_global_metadata,
                        &mut pending_qubit_metadata,
                    )?;
                    if !metadata.is_empty() {
                        let source_op = self.map_quantum_op_to_runtime_qubits(qop)?;
                        source_metadata.push_back(SourceTraceMetadata {
                            op: source_op,
                            metadata,
                        });
                    }
                    let mut emitted_ops = Vec::new();
                    self.submit_operation_to_runtime(op, &mut emitted_ops)?;
                    Self::push_lowered_ops_with_source_metadata(
                        &mut lowered_ops,
                        emitted_ops,
                        &mut source_metadata,
                    );
                }
                Operation::Barrier => {
                    let emitted_ops = self.lower_runtime_barrier()?;
                    Self::push_lowered_ops_with_source_metadata(
                        &mut lowered_ops,
                        emitted_ops,
                        &mut source_metadata,
                    );
                }
                _ => {
                    let mut emitted_ops = Vec::new();
                    self.submit_operation_to_runtime(op, &mut emitted_ops)?;
                    Self::push_lowered_ops_with_source_metadata(
                        &mut lowered_ops,
                        emitted_ops,
                        &mut source_metadata,
                    );
                }
            }
        }

        let emitted_ops = self.drain_runtime_operations()?;
        Self::push_lowered_ops_with_source_metadata(
            &mut lowered_ops,
            emitted_ops,
            &mut source_metadata,
        );

        if !pending_global_metadata.is_empty() {
            return Err(RuntimeError::ExecutionError(format!(
                "trace metadata was not followed by a quantum operation: {pending_global_metadata:?}"
            )));
        }
        Self::fail_if_qubit_metadata_was_not_consumed(&pending_qubit_metadata)?;
        Self::fail_if_metadata_was_not_lowered(&source_metadata)?;

        Ok(lowered_ops)
    }

    fn deliver_measurement_outcomes(&mut self, measurements: &BTreeMap<usize, u32>) -> Result<()> {
        self.check_batch_failure()?;
        // Feedback can make previously blocked native operations ready. A drain
        // preceding this delivery cannot certify the scheduler is still empty.
        if self.scheduled_mode == Some(true) && !measurements.is_empty() {
            self.scheduled_terminal_drained = false;
        }
        debug!(
            "Received {} measurement results, num_results={}, allocated_results={:?}",
            measurements.len(),
            self.num_results,
            self.interface.as_ref().map(|i| &i.allocated_results)
        );

        // Store measurements in classical state
        for (result_id, value) in measurements {
            trace!(
                "Measurement result {} = {} (num_results={})",
                result_id, value, self.num_results
            );
            if *value <= 1 {
                self.state.measurements.insert(*result_id, *value == 1);
            }

            if let Some(runtime_result_id) = self.program_to_runtime_results.get(result_id) {
                if let Some(lib) = &self.library
                    && let Some(instance) = self.instance
                {
                    unsafe {
                        let descriptor = Self::runtime_plugin_descriptor(lib)?;
                        if self.leakage_results.contains(result_id) {
                            let errno = (descriptor.set_u64_result_fn)(
                                instance,
                                *runtime_result_id,
                                u64::from(*value),
                            );
                            if errno != 0 {
                                return Err(RuntimeError::FfiError(format!(
                                    "selene_runtime_set_u64_result failed with errno {errno} \
                                     for result {result_id}"
                                )));
                            }
                        } else {
                            let bool_value = match *value {
                                0 => false,
                                1 => true,
                                _ => {
                                    return Err(RuntimeError::ExecutionError(format!(
                                        "ordinary measurement result {result_id} has non-Boolean outcome {value}"
                                    )));
                                }
                            };
                            // A delivery FAILURE is fatal: the scheduler
                            // would otherwise proceed on stale/default state
                            // while the QIS worker advances on the real bit,
                            // and no later gate can detect the divergence.
                            let errno = (descriptor.set_bool_result_fn)(
                                instance,
                                *runtime_result_id,
                                bool_value,
                            );
                            if errno != 0 {
                                return Err(RuntimeError::FfiError(format!(
                                    "selene_runtime_set_bool_result failed with errno {errno} \
                                     for result {result_id}"
                                )));
                            }
                        }
                    }
                }
            } else {
                log::trace!(
                    "Measurement result {result_id} was not allocated by the Selene runtime, storing locally only"
                );
            }

            if let Some(interface) = &mut self.interface
                && *value <= 1
            {
                interface.store_result(*result_id, *value == 1);
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

    /// Submission, feedback and draining can each fail after partial mutation.
    /// Retrying such a shot is unsafe even if the plugin returns an ordinary error.
    fn with_native_mutation<T>(
        &mut self,
        submit: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| submit(self))) {
            Ok(result) => result.map_err(|error| self.latch_batch_failure(error)),
            Err(payload) => {
                self.latch_batch_failure(RuntimeError::ExecutionError(
                    "native operation panicked; reset required".into(),
                ));
                std::panic::resume_unwind(payload)
            }
        }
    }

    fn drain_native_pending_operations(&mut self) -> Result<Vec<QuantumOp>> {
        self.check_batch_failure()?;
        if self.instance.is_none() {
            // Nothing was ever submitted; do not load the plugin just to drain.
            return Ok(Vec::new());
        }
        self.with_native_mutation(|runtime| {
            // Force the scheduler to release held work before collecting: a plain
            // poll only returns operations the plugin already considers ready, so
            // without the terminal barrier a lazily scheduling runtime could hold
            // a tail batch straight past this check. A plugin without the barrier
            // symbol cannot prove it released held work, so this fails closed
            // (both PECOS-built runtimes export `selene_runtime_global_barrier`).
            if !runtime.call_runtime_global_barrier(0)? {
                return Err(RuntimeError::ExecutionError(
                    "runtime plugin does not export selene_runtime_global_barrier; \
                 cannot force the terminal flush required to verify the \
                 scheduler is drained"
                        .to_string(),
                ));
            }
            runtime.drain_runtime_operations()
        })
    }

    fn select_output_mode(&mut self, scheduled: bool) -> Result<()> {
        self.check_batch_failure()?;
        if self
            .scheduled_mode
            .is_some_and(|previous| previous != scheduled)
        {
            return Err(RuntimeError::ExecutionError(
                "cannot mix flat and scheduled lowering before shot start/reset".into(),
            ));
        }
        self.scheduled_mode = Some(scheduled);
        Ok(())
    }

    fn collect_scheduled(
        &mut self,
        collect: impl FnOnce(&mut Self) -> Result<Vec<QuantumOp>>,
    ) -> Result<Vec<ScheduledBatch>> {
        self.check_batch_failure()?;
        if self.active_shot.is_none() && self.pending_shot_start.is_none() {
            return Err(RuntimeError::ExecutionError(
                "scheduled extraction requires shot_start".into(),
            ));
        }
        let Some(capacity) = self.num_qubits_hint.filter(|capacity| *capacity > 0) else {
            return Err(RuntimeError::ExecutionError(
                "scheduled extraction requires explicit nonzero qubit capacity".into(),
            ));
        };
        if self
            .initialized_num_qubits
            .is_some_and(|initialized| initialized != capacity)
        {
            return Err(RuntimeError::ExecutionError(
                "scheduled capacity changed; reset required".into(),
            ));
        }
        self.select_output_mode(true)?;
        self.scheduled_output = Some(ScheduledOutput::default());
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| collect(self)));
        let output = self
            .scheduled_output
            .take()
            .expect("active scheduled extraction");
        let result = match result {
            Ok(result) => result,
            Err(payload) => {
                self.latch_batch_failure(RuntimeError::ExecutionError(
                    "scheduled extraction panicked; reset required".into(),
                ));
                std::panic::resume_unwind(payload);
            }
        };
        match result {
            Ok(ops) if ops.is_empty() => Ok(output.batches),
            Ok(_) => Err(self.latch_batch_failure(RuntimeError::ExecutionError(
                "unscheduled output in scheduled extraction".into(),
            ))),
            Err(error) => Err(self.latch_batch_failure(error)),
        }
    }

    fn retain_scheduled_batch(&mut self, batch: RuntimeOperationBatch) -> Result<()> {
        let fail = |message: &str| RuntimeError::ExecutionError(message.into());
        if let Some(error) = batch.callback_error {
            return Err(fail(error));
        }
        batch
            .start_time_nanos
            .checked_add(batch.duration_nanos)
            .ok_or_else(|| fail("scheduled end time overflow"))?;
        let shot = self
            .active_shot
            .map(|(id, _)| id)
            .or(self.pending_shot_start.map(|(id, _)| id))
            .ok_or_else(|| fail("scheduled extraction requires shot_start"))?;
        let capacity = self
            .num_qubits_hint
            .ok_or_else(|| fail("scheduled extraction requires explicit capacity"))?;
        let output = self
            .scheduled_output
            .as_mut()
            .ok_or_else(|| fail("no scheduled extraction active"))?;
        if batch.operations.len() > MAX_OPERATIONS {
            return Err(fail("scheduled per-batch operation budget exceeded"));
        }
        let mut bytes = 0usize;
        let mut measurements = Vec::new();
        measurements
            .try_reserve(batch.operations.len())
            .map_err(|_| fail("scheduled measurement allocation failed"))?;
        for (operation_index, op) in batch.operations.iter().enumerate() {
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
                RuntimeScheduledOp::Reset { qubit_id } => (&[*qubit_id], &[]),
                RuntimeScheduledOp::Measure {
                    qubit_id,
                    result_id,
                }
                | RuntimeScheduledOp::MeasureLeaked {
                    qubit_id,
                    result_id,
                } => {
                    let program_result = *self
                        .runtime_to_program_results
                        .get(result_id)
                        .ok_or_else(|| fail("unmapped scheduled measurement result"))?;
                    measurements.push(ScheduledMeasurement {
                        operation_index,
                        runtime_result: *result_id,
                        program_result,
                        leakage_aware: self.leakage_results.contains(&program_result)
                            || matches!(op, RuntimeScheduledOp::MeasureLeaked { .. }),
                    });
                    (&[*qubit_id], &[])
                }
                RuntimeScheduledOp::Custom { data, .. } => {
                    bytes = bytes
                        .checked_add(data.len())
                        .ok_or_else(|| fail("scheduled payload overflow"))?;
                    (&[], &[])
                }
            };
            if angles.iter().any(|a| !a.is_finite())
                || qubits
                    .iter()
                    .any(|q| usize::try_from(*q).map_or(true, |q| q >= capacity))
            {
                return Err(fail("invalid scheduled angle or qubit"));
            }
        }
        if bytes > MAX_PAYLOAD_BYTES {
            return Err(fail("scheduled payload budget exceeded"));
        }
        let next_index = self
            .runtime_batch_index
            .checked_add(1)
            .ok_or_else(|| fail("scheduled batch index overflow"))?;
        output
            .batches
            .try_reserve(1)
            .map_err(|_| fail("scheduled batch allocation failed"))?;
        output.batches.push(ScheduledBatch {
            runtime_shot_id: shot,
            batch_index: self.runtime_batch_index,
            start_time_nanos: batch.start_time_nanos,
            duration_nanos: batch.duration_nanos,
            operations: batch.operations,
            measurements,
        });
        self.runtime_batch_index = next_index;
        Ok(())
    }

    fn check_batch_failure(&self) -> Result<()> {
        match &self.batch_failure {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }

    fn latch_batch_failure(&mut self, error: RuntimeError) -> RuntimeError {
        self.batch_failure.get_or_insert(error).clone()
    }

    /// Configure handling of opaque events. Capture is backward-compatible;
    /// strict emulation should use `RejectUnhandled`. Changing this policy does
    /// not clear a latched batch failure; successful `QisRuntime::reset` is required.
    pub fn set_custom_event_policy(&mut self, policy: RuntimeCustomEventPolicy) {
        self.custom_event_policy = policy;
    }

    /// Install a metadata-only handler, called in emission order outside the C
    /// callback, after the final batch timing is known. Errors always propagate.
    /// Return `Unsupported` for unknown events or unmodeled physical behavior.
    /// The handler is shared across clones; any captured state must be thread-safe
    /// and is not reset by PECOS. This hook does not inject gates or noise.
    /// A handler panic unwinds in Rust, never through the C callback; if caught,
    /// the runtime remains failed until a successful `QisRuntime::reset`.
    pub fn set_custom_event_handler<F>(&mut self, handler: F)
    where
        F: Fn(&RuntimeCustomEvent) -> Result<RuntimeCustomEventDisposition> + Send + Sync + 'static,
    {
        self.custom_event_handler = Some(Arc::new(handler));
    }

    /// Events captured in this shot, including the event causing a handler error.
    /// Cleared on shot start and reset; retained at shot end for inspection.
    #[must_use]
    pub fn custom_events(&self) -> &[RuntimeCustomEvent] {
        &self.custom_events
    }

    /// Drain captured events without resetting their batch ordinals.
    pub fn take_custom_events(&mut self) -> Vec<RuntimeCustomEvent> {
        std::mem::take(&mut self.custom_events)
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

        // Infer the reloaded program's live allocation peak, retaining legacy
        // index-based sizing as a conservative initial-capacity estimate.
        let (num_qubits, num_results) = collector_capacity(&operations);
        self.num_qubits = num_qubits;
        self.num_results = num_results;

        self.interface = Some(operations);
        self.current_op_index = 0;
        self.needs_reexecution = false;
        self.pending_measurements.clear();
    }

    /// Load the Selene plugin
    fn load_plugin(&mut self) -> Result<()> {
        self.check_batch_failure()?;
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
            // RTLD_NOW: an unresolved import fails here with the loader's message
            // instead of terminating the process on its first call.
            #[cfg(unix)]
            let library = libloading::os::unix::Library::open(
                Some(&self.plugin_path),
                libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_LOCAL,
            )
            .map(Into::into);
            #[cfg(not(unix))]
            let library = libloading::Library::new(&self.plugin_path);
            let lib = Arc::new(library.map_err(|e| {
                RuntimeError::FfiError(format!(
                    "Failed to load plugin {}: {}",
                    self.plugin_path,
                    std::error::Error::source(&e).unwrap_or(&e)
                ))
            })?);

            let descriptor = Self::runtime_plugin_descriptor(&lib)?;

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
            let errno = (descriptor.init_fn)(
                &raw mut instance,
                plugin_num_qubits as u64,
                0, // start time
                u32::try_from(arg_ptrs.len()).expect("custom-op argument count exceeds u32"),
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

    fn prepare_runtime_input(&mut self, operations: &[Operation]) -> Result<()> {
        self.check_batch_failure()?;
        let InputCapacity {
            qubits,
            results,
            peak_live,
            duplicate_allocation,
        } = operation_capacity(
            operations,
            self.program_to_runtime_qubits.keys().copied().collect(),
            self.explicit_qubit_handles.clone(),
        );
        if let Some(id) = duplicate_allocation {
            return Err(RuntimeError::ExecutionError(format!(
                "program qubit {id} is already allocated; input rejected before submission"
            )));
        }
        if let Some(requested) = self.num_qubits_hint
            && peak_live > requested
        {
            return Err(RuntimeError::ExecutionError(format!(
                "runtime input requires {peak_live} live qubits but configured capacity is {requested}; \
                 configure sufficient qubits before starting the shot"
            )));
        }
        let required_live_capacity = self.num_qubits_hint.unwrap_or(peak_live);
        if let Some(initialized) = self.initialized_num_qubits
            && required_live_capacity > initialized
        {
            // The plugin ABI has no state-preserving resize. Reinitializing here
            // loses allocations, pending work, results and native shot identity.
            return Err(RuntimeError::ExecutionError(format!(
                "runtime input requires capacity {required_live_capacity} but initialized capacity is {initialized}; \
                 reset and configure sufficient qubits before starting a new shot"
            )));
        }
        self.num_qubits = self.num_qubits.max(qubits);
        self.num_results = self.num_results.max(results);
        // Result bookkeeping and empty barriers need no native instance. Keep
        // capacity inference open until there is qubit demand or an explicit hint.
        if self.instance.is_none() && self.num_qubits_hint.is_none() && self.num_qubits == 0 {
            return Ok(());
        }
        self.load_plugin()
    }

    fn plugin_num_qubits(&self) -> usize {
        self.initialized_num_qubits
            .unwrap_or_else(|| self.num_qubits_hint.unwrap_or(self.num_qubits))
    }

    fn reset_plugin_instance(&mut self) -> Result<()> {
        if let Some(lib) = &self.library
            && let Some(instance) = self.instance
        {
            unsafe {
                if let Some(exit_fn) = Self::runtime_plugin_descriptor(lib)?.exit_fn {
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
        self.explicit_qubit_handles.clear();
        self.program_to_runtime_results.clear();
        self.runtime_to_program_results.clear();
        self.leakage_results.clear();
        self.last_gate_time_end_nanos.clear();
        Ok(())
    }

    fn apply_pending_shot_start(&mut self) -> Result<()> {
        let Some((shot_id, seed)) = self.pending_shot_start else {
            return Ok(());
        };
        if self.library.is_none() || self.instance.is_none() {
            return Ok(());
        }
        self.with_native_mutation(|runtime| {
            let lib = runtime.library.as_ref().expect("checked loaded library");
            let instance = runtime.instance.expect("checked native instance");
            unsafe {
                // The shot lifecycle hooks are part of the certified protocol: a
                // plugin that cannot receive shot boundaries cannot run its own
                // per-shot validation, so a missing symbol fails closed (both
                // PECOS-built runtimes export the full lifecycle).
                let shot_start_fn = Self::runtime_plugin_descriptor(lib)?.shot_start_fn;
                let errno = shot_start_fn(instance, shot_id, seed.unwrap_or(0));
                if errno != 0 {
                    return Err(RuntimeError::ExecutionError(format!(
                        "Shot start failed with errno {errno}"
                    )));
                }
                runtime.active_shot = Some((shot_id, seed.unwrap_or(0)));
            }

            runtime.pending_shot_start = None;
            Ok(())
        })
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
                Operation::TraceMetadata { .. } => {
                    trace!("Trace metadata encountered");
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
            let qalloc_fn = Self::runtime_plugin_descriptor(lib)?.qalloc_fn;
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
            let qfree_fn = Self::runtime_plugin_descriptor(lib)?.qfree_fn;
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
            let rxy_fn = Self::runtime_plugin_descriptor(lib)?.rxy_gate_fn;
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
            let rz_fn = Self::runtime_plugin_descriptor(lib)?.rz_gate_fn;
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
            let rzz_fn = Self::runtime_plugin_descriptor(lib)?.rzz_gate_fn;
            let errno = rzz_fn(instance, runtime_qubit_1, runtime_qubit_2, theta);
            if errno != 0 {
                return Err(RuntimeError::FfiError(format!(
                    "rzz failed with errno {errno}"
                )));
            }
        }

        Ok(())
    }

    fn call_runtime_rpp(
        &self,
        runtime_qubit_1: u64,
        runtime_qubit_2: u64,
        theta: f64,
        phi: f64,
    ) -> Result<()> {
        let lib = self
            .library
            .as_ref()
            .ok_or_else(|| RuntimeError::FfiError("Selene runtime is not loaded".to_string()))?;
        let instance = self.instance.ok_or_else(|| {
            RuntimeError::FfiError("Selene runtime is not initialized".to_string())
        })?;

        unsafe {
            let rpp_fn = Self::runtime_plugin_descriptor(lib)?.rpp_gate_fn;
            let errno = rpp_fn(instance, runtime_qubit_1, runtime_qubit_2, theta, phi);
            if errno != 0 {
                return Err(RuntimeError::FfiError(format!(
                    "rpp failed with errno {errno}"
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
            let reset_fn = Self::runtime_plugin_descriptor(lib)?.reset_fn;
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
            let measure_fn = Self::runtime_plugin_descriptor(lib)?.measure_fn;
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

    fn call_runtime_measure_leaked(
        &mut self,
        runtime_qubit: u64,
        program_result: usize,
    ) -> Result<()> {
        let lib = self
            .library
            .as_ref()
            .ok_or_else(|| RuntimeError::FfiError("Selene runtime is not loaded".to_string()))?;
        let instance = self.instance.ok_or_else(|| {
            RuntimeError::FfiError("Selene runtime is not initialized".to_string())
        })?;

        let runtime_result = unsafe {
            let measure_fn = Self::runtime_plugin_descriptor(lib)?.measure_leaked_fn;
            let mut runtime_result = 0;
            let errno = measure_fn(instance, runtime_qubit, &raw mut runtime_result);
            if errno != 0 {
                return Err(RuntimeError::FfiError(format!(
                    "measure_leaked failed with errno {errno}"
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
            let force_fn = Self::runtime_plugin_descriptor(lib)?.force_result_fn;
            let errno = force_fn(instance, runtime_result);
            if errno != 0 {
                return Err(RuntimeError::FfiError(format!(
                    "force_result failed with errno {errno}"
                )));
            }
        }

        Ok(())
    }

    fn call_runtime_global_barrier(&self, sleep_time: u64) -> Result<bool> {
        let lib = self
            .library
            .as_ref()
            .ok_or_else(|| RuntimeError::FfiError("Selene runtime is not loaded".to_string()))?;
        let instance = self.instance.ok_or_else(|| {
            RuntimeError::FfiError("Selene runtime is not initialized".to_string())
        })?;

        unsafe {
            let global_barrier_fn = Self::runtime_plugin_descriptor(lib)?.global_barrier_fn;
            let errno = global_barrier_fn(instance, sleep_time);
            if errno != 0 {
                return Err(RuntimeError::FfiError(format!(
                    "global_barrier failed with errno {errno}"
                )));
            }
        }

        Ok(true)
    }

    fn lower_runtime_barrier(&mut self) -> Result<Vec<QuantumOp>> {
        if self.instance.is_none() {
            return Ok(Vec::new());
        }

        // Runtime-native barriers keep scheduler-specific ordering decisions
        // inside the plugin. Falling back to a drain preserves compatibility
        // with older plugins that do not expose Selene barrier symbols.
        if self.call_runtime_global_barrier(0)? {
            return Ok(Vec::new());
        }

        self.drain_runtime_operations()
    }

    fn submit_operation_to_runtime(
        &mut self,
        op: &Operation,
        lowered_ops: &mut Vec<QuantumOp>,
    ) -> Result<()> {
        match op {
            Operation::AllocateQubit { id } => {
                debug_assert!(
                    !self.program_to_runtime_qubits.contains_key(id),
                    "duplicate allocations must be rejected by input preflight"
                );
                let _ = self.runtime_qubit_for_program(*id)?;
                self.explicit_qubit_handles.insert(*id);
            }
            Operation::AllocateResult { id } => {
                self.num_results = self.num_results.max(id + 1);
            }
            Operation::ReleaseQubit { id } => {
                self.release_runtime_qubit(*id)?;
            }
            Operation::RecordOutput { .. }
            | Operation::TraceMetadata { .. }
            | Operation::Barrier => {}
            Operation::Quantum(qop) => self.submit_quantum_op_to_runtime(qop, lowered_ops)?,
        }

        Ok(())
    }

    fn map_quantum_op_to_runtime_qubits(&mut self, qop: &QuantumOp) -> Result<QuantumOp> {
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
            QuantumOp::RZ(theta, qubit) => QuantumOp::RZ(*theta, map(*qubit)?),
            QuantumOp::RXY(theta, phi, qubit) => QuantumOp::RXY(*theta, *phi, map(*qubit)?),
            QuantumOp::Idle(duration, qubit) => QuantumOp::Idle(*duration, map(*qubit)?),
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
            QuantumOp::RZZ(theta, qubit_1, qubit_2) => {
                QuantumOp::RZZ(*theta, map(*qubit_1)?, map(*qubit_2)?)
            }
            QuantumOp::RXYXY2Q(theta, phi, qubit_1, qubit_2) => {
                QuantumOp::RXYXY2Q(*theta, *phi, map(*qubit_1)?, map(*qubit_2)?)
            }
            QuantumOp::Measure(qubit, result_id) => QuantumOp::Measure(map(*qubit)?, *result_id),
            QuantumOp::MeasureLeaked(qubit, result_id) => {
                QuantumOp::MeasureLeaked(map(*qubit)?, *result_id)
            }
            QuantumOp::Reset(qubit) => QuantumOp::Reset(map(*qubit)?),
        })
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
            QuantumOp::RXYXY2Q(theta, phi, qubit_1, qubit_2) => {
                let runtime_qubit_1 = self.runtime_qubit_for_program(*qubit_1)?;
                let runtime_qubit_2 = self.runtime_qubit_for_program(*qubit_2)?;
                self.call_runtime_rpp(runtime_qubit_1, runtime_qubit_2, *theta, *phi)?;
            }
            QuantumOp::Measure(qubit, result_id) => {
                let runtime_qubit = self.runtime_qubit_for_program(*qubit)?;
                self.call_runtime_measure(runtime_qubit, *result_id)?;
                self.release_implicitly_measured_qubit(*qubit)?;
            }
            QuantumOp::MeasureLeaked(qubit, result_id) => {
                self.leakage_results.insert(*result_id);
                let runtime_qubit = self.runtime_qubit_for_program(*qubit)?;
                self.call_runtime_measure_leaked(runtime_qubit, *result_id)?;
                self.release_implicitly_measured_qubit(*qubit)?;
            }
            QuantumOp::Reset(qubit) => {
                let runtime_qubit = self.runtime_qubit_for_program(*qubit)?;
                self.call_runtime_reset(runtime_qubit)?;
            }
            _ => {
                lowered_ops.extend(self.drain_runtime_operations()?);
                lowered_ops.push(self.map_quantum_op_to_runtime_qubits(qop)?);
            }
        }

        Ok(())
    }

    fn release_runtime_qubit(&mut self, program_qubit: usize) -> Result<()> {
        if let Some(&runtime_qubit) = self.program_to_runtime_qubits.get(&program_qubit) {
            if let Err(error) = self.runtime_qfree(runtime_qubit) {
                // Native state may have changed even on failure. Retain the map
                // only for diagnostics; no further execution is allowed until reset.
                return Err(self.latch_batch_failure(error));
            }
            self.program_to_runtime_qubits.remove(&program_qubit);
        }
        self.explicit_qubit_handles.remove(&program_qubit);
        Ok(())
    }

    fn release_implicitly_measured_qubit(&mut self, program_qubit: usize) -> Result<()> {
        if !self.explicit_qubit_handles.contains(&program_qubit) {
            self.release_runtime_qubit(program_qubit)?;
        }
        Ok(())
    }

    fn drain_runtime_operations(&mut self) -> Result<Vec<QuantumOp>> {
        self.check_batch_failure()?;
        if self.instance.is_none() {
            return Ok(Vec::new());
        }
        let mut lowered_ops = Vec::new();

        loop {
            if self.scheduled_mode == Some(true) && self.scheduled_output.is_none() {
                return Err(RuntimeError::ExecutionError(
                    "scheduled output requires the scheduled extraction API".into(),
                ));
            }
            let mut batch = RuntimeOperationBatch::default();
            if self.scheduled_output.is_some() {
                batch.extraction_budget = Some((MAX_OPERATIONS, MAX_PAYLOAD_BYTES));
            }
            let errno = {
                let lib = self.library.as_ref().ok_or_else(|| {
                    RuntimeError::FfiError("Selene runtime is not loaded".to_string())
                })?;
                let instance = self.instance.ok_or_else(|| {
                    RuntimeError::FfiError("Selene runtime is not initialized".to_string())
                })?;

                unsafe {
                    let get_next_fn: unsafe extern "C" fn(
                        RuntimeInstance,
                        SeleneRuntimeGetOperationHandle,
                    ) -> i32 = std::mem::transmute(
                        Self::runtime_plugin_descriptor(lib)?.get_next_operations_fn,
                    );
                    get_next_fn(
                        instance,
                        SeleneRuntimeGetOperationHandle {
                            instance: (&raw mut batch).cast::<c_void>(),
                            interface: RUNTIME_OPERATION_CALLBACKS,
                        },
                    )
                }
            };

            if errno != 0 {
                return Err(self.latch_batch_failure(RuntimeError::FfiError(format!(
                    "get_next_operations failed with errno {errno}"
                ))));
            }

            if !batch.invoked {
                break;
            }

            if self.scheduled_output.is_some() {
                if let Err(error) = self.retain_scheduled_batch(batch) {
                    return Err(self.latch_batch_failure(error));
                }
            } else {
                lowered_ops.extend(self.convert_runtime_batch(batch)?);
            }
        }

        Ok(lowered_ops)
    }

    fn push_lowered_ops_with_source_metadata(
        lowered_ops: &mut Vec<LoweredQuantumOp>,
        ops: Vec<QuantumOp>,
        source_metadata: &mut VecDeque<SourceTraceMetadata>,
    ) {
        for op in ops {
            let metadata = if let Some(source_index) = source_metadata
                .iter()
                .position(|source| Self::source_trace_metadata_matches_lowered_op(source, &op))
            {
                source_metadata
                    .remove(source_index)
                    .map(|source| source.metadata)
                    .unwrap_or_default()
            } else {
                TraceMetadata::new()
            };
            lowered_ops.push(LoweredQuantumOp::new(op, metadata));
        }
    }

    fn merge_trace_metadata(target: &mut TraceMetadata, metadata: TraceMetadata) -> Result<()> {
        for (key, value) in metadata {
            if let Some(existing) = target.get(&key)
                && existing != &value
            {
                return Err(RuntimeError::ExecutionError(format!(
                    "conflicting trace metadata for key {key:?}: {existing:?} != {value:?}"
                )));
            }
            target.insert(key, value);
        }
        Ok(())
    }

    fn take_pending_trace_metadata_for_source_op(
        qop: &QuantumOp,
        pending_global_metadata: &mut TraceMetadata,
        pending_qubit_metadata: &mut BTreeMap<usize, TraceMetadata>,
    ) -> Result<TraceMetadata> {
        let mut metadata = std::mem::take(pending_global_metadata);
        let mut consumed_qubits = Vec::new();

        for qubit in Self::quantum_op_qubits(qop) {
            let Some(pending) = pending_qubit_metadata.get(&qubit) else {
                continue;
            };
            if !Self::trace_metadata_can_annotate_source_op(pending, qop) {
                continue;
            }
            Self::merge_trace_metadata(&mut metadata, pending.clone())?;
            consumed_qubits.push(qubit);
        }

        for qubit in consumed_qubits {
            pending_qubit_metadata.remove(&qubit);
        }

        Ok(metadata)
    }

    fn trace_metadata_can_annotate_source_op(metadata: &TraceMetadata, qop: &QuantumOp) -> bool {
        let Some(source_gate) = metadata.get("source_gate").map(String::as_str) else {
            return true;
        };
        match source_gate {
            "SZZ" | "SZZDG" => Self::two_qubit_gate_qubits(qop).is_some(),
            _ => Self::single_qubit_gate_qubit(qop).is_some(),
        }
    }

    fn source_trace_metadata_matches_lowered_op(
        source: &SourceTraceMetadata,
        lowered: &QuantumOp,
    ) -> bool {
        if source.metadata.contains_key("source_gate") {
            return Self::source_gate_metadata_matches_lowered_op(source, lowered);
        }
        Self::source_op_matches_lowered_op(&source.op, lowered)
    }

    fn source_gate_metadata_matches_lowered_op(
        source: &SourceTraceMetadata,
        lowered: &QuantumOp,
    ) -> bool {
        let Some(source_gate) = source.metadata.get("source_gate").map(String::as_str) else {
            return false;
        };
        if matches!(source_gate, "SZZ" | "SZZDG") {
            let Some((source_qubit_1, source_qubit_2)) = Self::two_qubit_gate_qubits(&source.op)
            else {
                return false;
            };
            let Some((lowered_qubit_1, lowered_qubit_2)) = Self::two_qubit_gate_qubits(lowered)
            else {
                return false;
            };
            return Self::same_unordered_pair(
                source_qubit_1,
                source_qubit_2,
                lowered_qubit_1,
                lowered_qubit_2,
            );
        }
        let Some(source_qubit) = Self::single_qubit_gate_qubit(&source.op) else {
            return false;
        };
        let Some(lowered_qubit) = Self::single_qubit_gate_qubit(lowered) else {
            return false;
        };
        source_qubit == lowered_qubit
    }

    fn source_op_matches_lowered_op(source: &QuantumOp, lowered: &QuantumOp) -> bool {
        match (source, lowered) {
            (QuantumOp::H(source_qubit), QuantumOp::H(lowered_qubit))
            | (QuantumOp::X(source_qubit), QuantumOp::X(lowered_qubit))
            | (QuantumOp::Y(source_qubit), QuantumOp::Y(lowered_qubit))
            | (QuantumOp::Z(source_qubit), QuantumOp::Z(lowered_qubit))
            | (QuantumOp::S(source_qubit), QuantumOp::S(lowered_qubit))
            | (QuantumOp::Sdg(source_qubit), QuantumOp::Sdg(lowered_qubit))
            | (QuantumOp::T(source_qubit), QuantumOp::T(lowered_qubit))
            | (QuantumOp::Tdg(source_qubit), QuantumOp::Tdg(lowered_qubit))
            | (QuantumOp::Reset(source_qubit), QuantumOp::Reset(lowered_qubit)) => {
                source_qubit == lowered_qubit
            }
            (
                QuantumOp::RX(source_theta, source_qubit),
                QuantumOp::RX(lowered_theta, lowered_qubit),
            )
            | (
                QuantumOp::RY(source_theta, source_qubit),
                QuantumOp::RY(lowered_theta, lowered_qubit),
            )
            | (
                QuantumOp::RZ(source_theta, source_qubit),
                QuantumOp::RZ(lowered_theta, lowered_qubit),
            )
            | (
                QuantumOp::Idle(source_theta, source_qubit),
                QuantumOp::Idle(lowered_theta, lowered_qubit),
            ) => source_qubit == lowered_qubit && Self::same_float(*source_theta, *lowered_theta),
            (
                QuantumOp::RXY(source_theta, source_phi, source_qubit),
                QuantumOp::RXY(lowered_theta, lowered_phi, lowered_qubit),
            ) => {
                source_qubit == lowered_qubit
                    && Self::same_float(*source_theta, *lowered_theta)
                    && Self::same_float(*source_phi, *lowered_phi)
            }
            (
                QuantumOp::CX(source_control, source_target),
                QuantumOp::CX(lowered_control, lowered_target),
            )
            | (
                QuantumOp::CY(source_control, source_target),
                QuantumOp::CY(lowered_control, lowered_target),
            )
            | (
                QuantumOp::CZ(source_control, source_target),
                QuantumOp::CZ(lowered_control, lowered_target),
            )
            | (
                QuantumOp::CH(source_control, source_target),
                QuantumOp::CH(lowered_control, lowered_target),
            ) => Self::same_pair(
                *source_control,
                *source_target,
                *lowered_control,
                *lowered_target,
            ),
            (
                QuantumOp::CRZ(source_theta, source_control, source_target),
                QuantumOp::CRZ(lowered_theta, lowered_control, lowered_target),
            ) => {
                Self::same_float(*source_theta, *lowered_theta)
                    && Self::same_pair(
                        *source_control,
                        *source_target,
                        *lowered_control,
                        *lowered_target,
                    )
            }
            (
                QuantumOp::CCX(source_control_1, source_control_2, source_target),
                QuantumOp::CCX(lowered_control_1, lowered_control_2, lowered_target),
            ) => {
                (source_control_1, source_control_2, source_target)
                    == (lowered_control_1, lowered_control_2, lowered_target)
            }
            (
                QuantumOp::ZZ(source_qubit_1, source_qubit_2),
                QuantumOp::ZZ(lowered_qubit_1, lowered_qubit_2)
                | QuantumOp::RZZ(_, lowered_qubit_1, lowered_qubit_2),
            ) => Self::same_unordered_pair(
                *source_qubit_1,
                *source_qubit_2,
                *lowered_qubit_1,
                *lowered_qubit_2,
            ),
            (
                QuantumOp::RXYXY2Q(source_theta, source_phi, source_qubit_1, source_qubit_2),
                QuantumOp::RXYXY2Q(lowered_theta, lowered_phi, lowered_qubit_1, lowered_qubit_2),
            ) => {
                Self::same_float(*source_theta, *lowered_theta)
                    && Self::same_float(*source_phi, *lowered_phi)
                    && Self::same_unordered_pair(
                        *source_qubit_1,
                        *source_qubit_2,
                        *lowered_qubit_1,
                        *lowered_qubit_2,
                    )
            }
            (
                QuantumOp::RZZ(source_theta, source_qubit_1, source_qubit_2),
                QuantumOp::RZZ(lowered_theta, lowered_qubit_1, lowered_qubit_2),
            ) => {
                Self::same_float(*source_theta, *lowered_theta)
                    && Self::same_unordered_pair(
                        *source_qubit_1,
                        *source_qubit_2,
                        *lowered_qubit_1,
                        *lowered_qubit_2,
                    )
            }

            (
                QuantumOp::Measure(source_qubit, source_result),
                QuantumOp::Measure(lowered_qubit, lowered_result),
            )
            | (
                QuantumOp::MeasureLeaked(source_qubit, source_result),
                QuantumOp::MeasureLeaked(lowered_qubit, lowered_result),
            ) => source_qubit == lowered_qubit && source_result == lowered_result,
            _ => false,
        }
    }

    fn same_float(left: f64, right: f64) -> bool {
        (left - right).abs() <= 1e-12
    }

    fn same_pair(left_a: usize, left_b: usize, right_a: usize, right_b: usize) -> bool {
        (left_a, left_b) == (right_a, right_b)
    }

    fn same_unordered_pair(left_a: usize, left_b: usize, right_a: usize, right_b: usize) -> bool {
        Self::same_pair(left_a, left_b, right_a, right_b)
            || Self::same_pair(left_a, left_b, right_b, right_a)
    }

    fn quantum_op_qubits(qop: &QuantumOp) -> BTreeSet<usize> {
        let mut qubits = BTreeSet::new();
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
            | QuantumOp::Measure(qubit, _)
            | QuantumOp::MeasureLeaked(qubit, _)
            | QuantumOp::Reset(qubit) => {
                qubits.insert(*qubit);
            }
            QuantumOp::CX(qubit_1, qubit_2)
            | QuantumOp::CY(qubit_1, qubit_2)
            | QuantumOp::CZ(qubit_1, qubit_2)
            | QuantumOp::CH(qubit_1, qubit_2)
            | QuantumOp::CRZ(_, qubit_1, qubit_2)
            | QuantumOp::ZZ(qubit_1, qubit_2)
            | QuantumOp::RZZ(_, qubit_1, qubit_2)
            | QuantumOp::RXYXY2Q(_, _, qubit_1, qubit_2) => {
                qubits.insert(*qubit_1);
                qubits.insert(*qubit_2);
            }
            QuantumOp::CCX(qubit_1, qubit_2, qubit_3) => {
                qubits.insert(*qubit_1);
                qubits.insert(*qubit_2);
                qubits.insert(*qubit_3);
            }
        }
        qubits
    }

    fn single_qubit_gate_qubit(qop: &QuantumOp) -> Option<usize> {
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
            | QuantumOp::RXY(_, _, qubit) => Some(*qubit),
            _ => None,
        }
    }

    fn two_qubit_gate_qubits(qop: &QuantumOp) -> Option<(usize, usize)> {
        match qop {
            QuantumOp::CX(qubit_1, qubit_2)
            | QuantumOp::CY(qubit_1, qubit_2)
            | QuantumOp::CZ(qubit_1, qubit_2)
            | QuantumOp::CH(qubit_1, qubit_2)
            | QuantumOp::CRZ(_, qubit_1, qubit_2)
            | QuantumOp::ZZ(qubit_1, qubit_2)
            | QuantumOp::RZZ(_, qubit_1, qubit_2)
            | QuantumOp::RXYXY2Q(_, _, qubit_1, qubit_2) => Some((*qubit_1, *qubit_2)),
            _ => None,
        }
    }

    fn fail_if_metadata_was_not_lowered(
        source_metadata: &VecDeque<SourceTraceMetadata>,
    ) -> Result<()> {
        let leftover_metadata = source_metadata
            .iter()
            .filter(|source| Self::trace_metadata_requires_lowering(&source.metadata))
            .collect::<Vec<_>>();
        if leftover_metadata.is_empty() {
            return Ok(());
        }
        let examples = leftover_metadata
            .iter()
            .take(3)
            .map(|source| format!("{:?} for {:?}", source.metadata, source.op))
            .collect::<Vec<_>>()
            .join(", ");
        Err(RuntimeError::ExecutionError(format!(
            "runtime lowering did not emit non-idle operations for {} metadata-bearing source operation(s); examples: {examples}",
            leftover_metadata.len()
        )))
    }

    fn trace_metadata_requires_lowering(metadata: &TraceMetadata) -> bool {
        metadata
            .get("source_lowering_required")
            .is_some_and(|value| value.eq_ignore_ascii_case("true"))
    }

    fn fail_if_qubit_metadata_was_not_consumed(
        pending_qubit_metadata: &BTreeMap<usize, TraceMetadata>,
    ) -> Result<()> {
        if pending_qubit_metadata.is_empty() {
            return Ok(());
        }
        let examples = pending_qubit_metadata
            .iter()
            .take(3)
            .map(|(qubit, metadata)| format!("qubit {qubit}: {metadata:?}"))
            .collect::<Vec<_>>()
            .join(", ");
        Err(RuntimeError::ExecutionError(format!(
            "qubit-scoped trace metadata was not followed by a compatible quantum operation for {} qubit(s); examples: {examples}",
            pending_qubit_metadata.len()
        )))
    }

    fn convert_runtime_batch(&mut self, batch: RuntimeOperationBatch) -> Result<Vec<QuantumOp>> {
        self.check_batch_failure()?;
        let result = self.convert_runtime_batch_inner(batch);
        result.map_err(|error| self.latch_batch_failure(error))
    }

    fn convert_runtime_batch_inner(
        &mut self,
        batch: RuntimeOperationBatch,
    ) -> Result<Vec<QuantumOp>> {
        if let Some(error) = batch.callback_error {
            return Err(RuntimeError::FfiError(error.to_string()));
        }
        let batch_index = self.runtime_batch_index;
        self.runtime_batch_index += 1;
        let mut lowered_ops = Vec::new();
        let start_time = batch.start_time_nanos;
        let end_time = batch.end_time_nanos();

        for (operation_index, op) in batch.operations.into_iter().enumerate() {
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
                } => {
                    let qubit = self.runtime_qubit_to_usize(qubit_id)?;
                    let program_result = self.runtime_result_to_program_result(result_id)?;
                    self.push_idle_before(&mut lowered_ops, qubit, start_time)?;
                    if self.leakage_results.contains(&program_result) {
                        // The pinned runtime ABI allocates both Boolean and
                        // leakage-aware futures through `runtime_measure`.
                        // Restore the source result kind after scheduling so
                        // PECOS executes MeasureLeaked and can produce 2.
                        lowered_ops.push(QuantumOp::MeasureLeaked(qubit, program_result));
                    } else {
                        lowered_ops.push(QuantumOp::Measure(qubit, program_result));
                    }
                    self.mark_gate_end(qubit, end_time);
                }
                RuntimeScheduledOp::MeasureLeaked {
                    qubit_id,
                    result_id,
                } => {
                    let qubit = self.runtime_qubit_to_usize(qubit_id)?;
                    let program_result = self.runtime_result_to_program_result(result_id)?;
                    self.leakage_results.insert(program_result);
                    self.push_idle_before(&mut lowered_ops, qubit, start_time)?;
                    lowered_ops.push(QuantumOp::MeasureLeaked(qubit, program_result));
                    self.mark_gate_end(qubit, end_time);
                }
                RuntimeScheduledOp::Reset { qubit_id } => {
                    let qubit = self.runtime_qubit_to_usize(qubit_id)?;
                    lowered_ops.push(QuantumOp::Reset(qubit));
                    self.mark_gate_end(qubit, end_time);
                }
                RuntimeScheduledOp::Rpp {
                    qubit_id_1,
                    qubit_id_2,
                    theta,
                    phi,
                } => {
                    let qubit_1 = self.runtime_qubit_to_usize(qubit_id_1)?;
                    let qubit_2 = self.runtime_qubit_to_usize(qubit_id_2)?;
                    self.push_idle_before(&mut lowered_ops, qubit_1, start_time)?;
                    self.push_idle_before(&mut lowered_ops, qubit_2, start_time)?;
                    lowered_ops.push(QuantumOp::RXYXY2Q(theta, phi, qubit_1, qubit_2));
                    self.mark_gate_end(qubit_1, end_time);
                    self.mark_gate_end(qubit_2, end_time);
                }
                RuntimeScheduledOp::Custom { tag, data } => {
                    let event = RuntimeCustomEvent {
                        tag,
                        data,
                        batch_index,
                        operation_index,
                        start_time_nanos: start_time,
                        duration_nanos: batch.duration_nanos,
                    };
                    self.custom_events.try_reserve(1).map_err(|_| {
                        RuntimeError::ExecutionError(
                            "unable to allocate custom-event history".into(),
                        )
                    })?;
                    self.custom_events.push(event);
                    let event = self.custom_events.last().expect("just inserted");
                    let disposition = if let Some(handler) = &self.custom_event_handler {
                        // If user Rust code panics and its caller catches the unwind,
                        // the consumed event must still leave this shot poisoned.
                        self.batch_failure = Some(RuntimeError::ExecutionError(format!(
                            "runtime custom event handler panicked at batch {batch_index}, operation {operation_index}; reset required"
                        )));
                        let result = handler(event);
                        self.batch_failure = None;
                        result?
                    } else {
                        RuntimeCustomEventDisposition::Unsupported
                    };
                    if disposition == RuntimeCustomEventDisposition::Unsupported
                        && self.custom_event_policy == RuntimeCustomEventPolicy::RejectUnhandled
                    {
                        return Err(RuntimeError::ExecutionError(format!(
                            "unsupported runtime custom event tag {tag} at batch {batch_index}, operation {operation_index}, start {start_time} ns; register a metadata-only handler for understood metadata; physical effects require downstream modeling"
                        )));
                    }
                }
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
        // Clone configuration for a fresh shot, never a native snapshot. An
        // initialized instance may hold allocations, results or scheduler state
        // that cannot be transferred to the fresh plugin, so require reset.
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
            explicit_qubit_handles: self.explicit_qubit_handles.clone(),
            num_results: self.num_results,
            interface: self.interface.clone(),
            current_op_index: self.current_op_index,
            needs_reexecution: self.needs_reexecution,
            pending_measurements: self.pending_measurements.clone(),
            program_to_runtime_qubits: self.program_to_runtime_qubits.clone(),
            program_to_runtime_results: self.program_to_runtime_results.clone(),
            runtime_to_program_results: self.runtime_to_program_results.clone(),
            leakage_results: self.leakage_results.clone(),
            last_gate_time_end_nanos: self.last_gate_time_end_nanos.clone(),
            pending_shot_start: self.pending_shot_start,
            active_shot: self.active_shot,
            custom_events: self.custom_events.clone(),
            runtime_batch_index: self.runtime_batch_index,
            custom_event_policy: self.custom_event_policy,
            custom_event_handler: self.custom_event_handler.clone(),
            batch_failure: self.batch_failure.clone().or_else(|| {
                let kind = if self.scheduled_mode == Some(true) {
                    "scheduled"
                } else if self.instance.is_some() {
                    "native"
                } else {
                    return None;
                };
                Some(RuntimeError::ExecutionError(format!(
                    "cloned {kind} runtime requires reset; live native snapshots are unsupported"
                )))
            }),
            scheduled_mode: self.scheduled_mode,
            scheduled_terminal_drained: false,
            scheduled_output: None,
        }
    }
}

fn collector_capacity(interface: &OperationCollector) -> (usize, usize) {
    let InputCapacity {
        mut qubits,
        mut results,
        ..
    } = operation_capacity(&interface.operations, BTreeSet::new(), BTreeSet::new());
    let explicit: BTreeSet<_> = interface
        .operations
        .iter()
        .filter_map(|op| {
            if let Operation::AllocateQubit { id } = op {
                Some(*id)
            } else {
                None
            }
        })
        .collect();
    for &qubit in &interface.allocated_qubits {
        if !explicit.contains(&qubit) {
            include_qubit(&mut qubits, qubit);
        }
    }
    for &result in &interface.allocated_results {
        include_result(&mut results, result);
    }
    (qubits, results)
}

#[derive(Debug, PartialEq, Eq)]
struct InputCapacity {
    /// Conservative legacy index-based sizing, at least the live peak.
    qubits: usize,
    results: usize,
    peak_live: usize,
    duplicate_allocation: Option<usize>,
}

/// Admission mirrors per-handle submission lifetimes, seeded with all handles
/// still live from earlier input. Collector sizing ignores invalid allocations;
/// submission admission rejects them before mutating native state.
fn operation_capacity(
    operations: &[Operation],
    mut live: BTreeSet<usize>,
    mut explicit: BTreeSet<usize>,
) -> InputCapacity {
    let mut qubits = 0;
    let mut results = 0;
    let mut peak = live.len();
    let mut duplicate_allocation = None;
    for op in operations {
        match op {
            Operation::AllocateQubit { id } => {
                if !live.insert(*id) {
                    duplicate_allocation.get_or_insert(*id);
                }
                explicit.insert(*id);
                peak = peak.max(live.len());
            }
            Operation::ReleaseQubit { id } => {
                live.remove(id);
                explicit.remove(id);
            }
            Operation::Quantum(qop) => {
                for_each_quantum_qubit(qop, |qubit| {
                    live.insert(qubit);
                    if !explicit.contains(&qubit) {
                        include_qubit(&mut qubits, qubit);
                    }
                });
                peak = peak.max(live.len());
                include_quantum_result_capacity(qop, &mut results);
                if let QuantumOp::Measure(qubit, _) | QuantumOp::MeasureLeaked(qubit, _) = qop
                    && !explicit.contains(qubit)
                {
                    live.remove(qubit);
                }
            }
            Operation::AllocateResult { id } => include_result(&mut results, *id),
            Operation::RecordOutput { result_id, .. } => include_result(&mut results, *result_id),
            Operation::TraceMetadata { .. } | Operation::Barrier => {}
        }
    }
    InputCapacity {
        qubits: qubits.max(peak),
        results,
        peak_live: peak,
        duplicate_allocation,
    }
}

fn include_quantum_result_capacity(qop: &QuantumOp, num_results: &mut usize) {
    match qop {
        QuantumOp::Measure(_, result) | QuantumOp::MeasureLeaked(_, result) => {
            include_result(num_results, *result);
        }
        _ => {}
    }
}

fn for_each_quantum_qubit(qop: &QuantumOp, mut include: impl FnMut(usize)) {
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
        | QuantumOp::Reset(qubit) => include(*qubit),
        QuantumOp::CX(qubit_1, qubit_2)
        | QuantumOp::CY(qubit_1, qubit_2)
        | QuantumOp::CZ(qubit_1, qubit_2)
        | QuantumOp::CH(qubit_1, qubit_2)
        | QuantumOp::CRZ(_, qubit_1, qubit_2)
        | QuantumOp::ZZ(qubit_1, qubit_2)
        | QuantumOp::RZZ(_, qubit_1, qubit_2)
        | QuantumOp::RXYXY2Q(_, _, qubit_1, qubit_2) => {
            include(*qubit_1);
            include(*qubit_2);
        }
        QuantumOp::CCX(qubit_1, qubit_2, qubit_3) => {
            include(*qubit_1);
            include(*qubit_2);
            include(*qubit_3);
        }
        QuantumOp::Measure(qubit, _) | QuantumOp::MeasureLeaked(qubit, _) => {
            include(*qubit);
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
        self.check_batch_failure()?;
        debug!(
            "Loading QIS interface with {} operations",
            interface.operations.len()
        );

        // Infer peak live allocations, including mixed explicit and implicit
        // handles. Conservative legacy index sizing is used only to choose the
        // initial capacity; runtime handles are allocated independently.
        let (num_qubits, num_results) = collector_capacity(&interface);
        self.num_qubits = num_qubits;
        self.num_results = num_results;

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
        self.select_output_mode(false)?;
        // For now, we'll use the simple approach of processing from the interface
        // In a full implementation, we'd call into the Selene runtime's
        // get_next_operations function
        self.process_interface_ops()
    }

    fn supports_operation_lowering(&self) -> bool {
        true
    }

    fn drain_pending_operations(&mut self) -> Result<Vec<QuantumOp>> {
        self.select_output_mode(false)?;
        self.drain_native_pending_operations()
    }

    /// Extract native batches without flattening, idle insertion or event handling.
    /// See [`crate::scheduled`] for the extraction-only contract.
    ///
    /// Requires `shot_start` and explicit `set_num_qubits` capacity. Only native
    /// RXY/RZ/RZZ/RXYXY2Q, reset, measurements,
    /// allocation/release and barriers are accepted. Source trace metadata is
    /// rejected rather than silently lost. Flat and scheduled lowering cannot
    /// be mixed within a shot. Each native batch admits at most 4096 operations
    /// and 256 KiB of opaque payload. Returned batch counts are not capped;
    /// aggregate memory grows with the native schedule. No history is kept after return.
    ///
    /// # Errors
    /// Rejects unsupported inputs before submission. Extraction failures after
    /// submission poison the runtime until successful reset. This is not rollback
    /// of native scheduler state. Runtime-local shot IDs are not host worker IDs.
    fn lower_scheduled_operations(
        &mut self,
        operations: &[Operation],
    ) -> Result<Vec<ScheduledBatch>> {
        self.check_batch_failure()?;
        for op in operations {
            let ids: &[usize] = match op {
                Operation::AllocateQubit { id }
                | Operation::AllocateResult { id }
                | Operation::ReleaseQubit { id } => &[*id],
                Operation::Quantum(
                    QuantumOp::RXY(_, _, q) | QuantumOp::RZ(_, q) | QuantumOp::Reset(q),
                ) => &[*q],
                Operation::Quantum(
                    QuantumOp::RZZ(_, a, b)
                    | QuantumOp::RXYXY2Q(_, _, a, b)
                    | QuantumOp::Measure(a, b)
                    | QuantumOp::MeasureLeaked(a, b),
                ) => &[*a, *b],
                _ => &[],
            };
            if ids.iter().any(|id| id.checked_add(1).is_none()) {
                return Err(RuntimeError::ExecutionError(
                    "scheduled source identifier overflow".into(),
                ));
            }
            let angles: &[f64] = match op {
                Operation::Quantum(QuantumOp::RXY(a, b, _) | QuantumOp::RXYXY2Q(a, b, _, _)) => {
                    &[*a, *b]
                }
                Operation::Quantum(QuantumOp::RZ(a, _) | QuantumOp::RZZ(a, _, _)) => &[*a],
                _ => &[],
            };
            if angles.iter().any(|angle| !angle.is_finite()) {
                return Err(RuntimeError::ExecutionError(
                    "non-finite scheduled source angle".into(),
                ));
            }
            match op {
                Operation::Quantum(
                    QuantumOp::RXY(..)
                    | QuantumOp::RXYXY2Q(..)
                    | QuantumOp::RZ(..)
                    | QuantumOp::RZZ(..)
                    | QuantumOp::Reset(_)
                    | QuantumOp::Measure(_, _)
                    | QuantumOp::MeasureLeaked(_, _),
                )
                | Operation::AllocateQubit { .. }
                | Operation::AllocateResult { .. }
                | Operation::ReleaseQubit { .. }
                | Operation::Barrier => {}
                _ => {
                    return Err(RuntimeError::ExecutionError(
                        "unsupported scheduled extraction input".into(),
                    ));
                }
            }
        }
        self.collect_scheduled(|runtime| {
            if !operations.is_empty() {
                runtime.scheduled_terminal_drained = false;
            }
            runtime.lower_native_operations(operations)
        })
    }

    /// Force the native terminal barrier and return any remaining scheduled batches.
    /// Call before shot completion and consume all returned work. This does not
    /// execute it or certify a physics consumer. Same per-batch budgets as extraction.
    ///
    /// # Errors
    /// Fails if a terminal flush is unsupported or extraction fails. Post-submission
    /// failures remain latched until reset.
    fn drain_pending_scheduled_operations(&mut self) -> Result<Vec<ScheduledBatch>> {
        let batches = self.collect_scheduled(Self::drain_native_pending_operations)?;
        self.scheduled_terminal_drained = true;
        Ok(batches)
    }

    fn lower_operations(&mut self, operations: &[Operation]) -> Result<Vec<QuantumOp>> {
        self.select_output_mode(false)?;
        self.lower_native_operations(operations)
    }

    fn lower_operations_with_metadata(
        &mut self,
        operations: &[Operation],
    ) -> Result<Vec<LoweredQuantumOp>> {
        self.select_output_mode(false)?;
        self.prepare_runtime_input(operations)?;
        self.with_native_mutation(|runtime| runtime.submit_metadata_operations(operations))
    }

    fn provide_measurements(&mut self, measurements: BTreeMap<usize, bool>) -> Result<()> {
        self.provide_measurement_outcomes(
            measurements
                .into_iter()
                .map(|(result_id, value)| (result_id, u32::from(value)))
                .collect(),
        )
    }

    fn provide_measurement_outcomes(&mut self, measurements: BTreeMap<usize, u32>) -> Result<()> {
        self.check_batch_failure()?;
        self.with_native_mutation(|runtime| runtime.deliver_measurement_outcomes(&measurements))
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
        self.check_batch_failure()?;
        if self.scheduled_mode == Some(true) && !self.scheduled_terminal_drained {
            return Err(RuntimeError::ExecutionError(
                "drain the scheduled shot or reset before starting another shot".into(),
            ));
        }
        // Reset state for new shot
        self.state = ClassicalState::default();
        self.current_op_index = 0;
        self.needs_reexecution = false;
        self.pending_measurements.clear();
        self.program_to_runtime_qubits.clear();
        self.explicit_qubit_handles.clear();
        self.program_to_runtime_results.clear();
        self.runtime_to_program_results.clear();
        self.leakage_results.clear();
        self.last_gate_time_end_nanos.clear();
        self.custom_events.clear();
        self.scheduled_mode = None;
        self.scheduled_terminal_drained = false;
        self.scheduled_output = None;
        self.runtime_batch_index = 0;
        self.pending_shot_start = Some((shot_id, seed));
        self.apply_pending_shot_start()?;

        Ok(())
    }

    fn shot_end(&mut self) -> Result<Shot> {
        self.check_batch_failure()?;
        if self.scheduled_mode == Some(true) && !self.scheduled_terminal_drained {
            return Err(RuntimeError::ExecutionError(
                "scheduled shot requires a successful terminal drain before shot_end".into(),
            ));
        }
        // Only end a shot the plugin actually started; the pinned Selene ABI
        // is `selene_runtime_shot_end(instance, shot_id, seed)`, mirroring
        // shot_start, so the delivered identity pair is replayed here.
        // `take()` clears the active shot unconditionally: a clone that
        // inherited `active_shot` without a loaded plugin has no FFI shot to
        // end, but must not carry the stale identity forward.
        if let Some((_shot_id, _seed)) = self.active_shot.take()
            && let Some(lib) = &self.library
            && let Some(instance) = self.instance
        {
            unsafe {
                // Missing shot_end fails closed like the other lifecycle
                // hooks: the plugin's own finalization validation is part of
                // what shot completion certifies.
                let shot_end_fn = Self::runtime_plugin_descriptor(lib)?.shot_end_fn;
                let errno = shot_end_fn(instance);
                if errno != 0 {
                    return Err(RuntimeError::FfiError(format!(
                        "selene_runtime_shot_end failed with errno {errno}"
                    )));
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
        self.batch_failure = None;
        self.state = ClassicalState::default();
        self.current_op_index = 0;
        self.program_to_runtime_qubits.clear();
        self.explicit_qubit_handles.clear();
        self.program_to_runtime_results.clear();
        self.runtime_to_program_results.clear();
        self.leakage_results.clear();
        self.last_gate_time_end_nanos.clear();
        self.custom_events.clear();
        self.scheduled_mode = None;
        self.scheduled_terminal_drained = false;
        self.scheduled_output = None;
        self.runtime_batch_index = 0;
        self.pending_shot_start = None;
        self.active_shot = None;

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

    /// Discriminates on Linux; macOS binds the fixture's import at load either way.
    #[cfg(unix)]
    #[test]
    fn selene_runtime_rejects_unresolved_import_at_load() {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("missing_import.c");
        let library = directory.path().join(if cfg!(target_os = "macos") {
            "missing_import.dylib"
        } else {
            "missing_import.so"
        });
        std::fs::write(
            &source,
            "extern void pecos_test_missing_import(void);\n\
             void pecos_test_export(void) { pecos_test_missing_import(); }\n",
        )
        .unwrap();
        let mut compiler = std::process::Command::new("cc");
        compiler.args(["-shared", "-fPIC"]);
        #[cfg(target_os = "linux")]
        compiler.arg("-Wl,-z,lazy");
        #[cfg(target_os = "macos")]
        compiler.arg("-Wl,-undefined,dynamic_lookup");
        let output = compiler
            .arg(&source)
            .arg("-o")
            .arg(&library)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "C fixture compilation failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let mut runtime = SeleneRuntime::new(&library);
        let error = runtime
            .lower_operations(&[Operation::AllocateQubit { id: 0 }])
            .unwrap_err();
        assert!(
            error.to_string().contains("pecos_test_missing_import"),
            "unexpected loader error: {error}"
        );
    }

    #[test]
    fn custom_rejection_remains_terminal_until_reset() {
        let mut runtime = SeleneRuntime::new("synthetic-runtime.so");
        runtime.set_custom_event_policy(RuntimeCustomEventPolicy::RejectUnhandled);
        let mut batch = RuntimeOperationBatch::default();
        unsafe {
            runtime_batch_custom((&raw mut batch).cast(), 8401, std::ptr::null(), 0);
        }
        let error = runtime
            .convert_runtime_batch(batch)
            .unwrap_err()
            .to_string();
        assert_eq!(runtime.shot_end().unwrap_err().to_string(), error);
        runtime.take_custom_events();
        runtime.set_custom_event_policy(RuntimeCustomEventPolicy::Capture);
        runtime.set_custom_event_handler(|_| Ok(RuntimeCustomEventDisposition::MetadataOnly));
        assert_eq!(
            runtime.drain_pending_operations().unwrap_err().to_string(),
            error
        );
        assert_eq!(
            runtime.lower_operations(&[]).unwrap_err().to_string(),
            error
        );
        assert_eq!(
            runtime
                .lower_operations_with_metadata(&[])
                .unwrap_err()
                .to_string(),
            error
        );
        assert_eq!(runtime.clone().shot_end().unwrap_err().to_string(), error);
        assert_eq!(runtime.shot_start(2, None).unwrap_err().to_string(), error);
        runtime.reset().unwrap();
        runtime.shot_start(2, None).unwrap();
        assert_eq!(runtime.custom_events(), []);
        assert!(runtime.shot_end().is_ok());
    }

    #[test]
    fn failed_custom_callback_stops_following_batch_appends() {
        let mut batch = RuntimeOperationBatch::default();
        unsafe {
            let instance = (&raw mut batch).cast();
            runtime_batch_custom(instance, 8401, std::ptr::null(), 1);
            runtime_batch_rxy(instance, 0, 0.25, 0.5);
            runtime_batch_rz(instance, 0, 0.25);
            runtime_batch_rzz(instance, 0, 1, 0.25);
            runtime_batch_rpp(instance, 0, 1, 0.25, 0.5);
            runtime_batch_reset(instance, 0);
            runtime_batch_measure(instance, 0, 1);
            runtime_batch_measure_leaked(instance, 0, 2);
            runtime_batch_custom(instance, 8402, std::ptr::null(), 0);
        }
        assert_eq!(
            batch.callback_error,
            Some("invalid custom-event payload pointer/length")
        );
        assert_eq!(batch.operations, []);
        assert_eq!(batch.operations.capacity(), 0);
    }

    #[test]
    fn custom_handler_error_and_panic_cannot_certify_a_shot() {
        for panic_in_handler in [false, true] {
            let mut runtime = SeleneRuntime::new("synthetic-runtime.so");
            runtime.set_custom_event_handler(move |_| {
                assert!(!panic_in_handler, "synthetic handler panic");
                Err(RuntimeError::ExecutionError(
                    "synthetic handler error".into(),
                ))
            });
            let mut batch = RuntimeOperationBatch::default();
            unsafe {
                runtime_batch_custom((&raw mut batch).cast(), 8403, std::ptr::null(), 0);
            }
            // The handler runs in Rust after C returns. A downstream caller may
            // catch its panic, but must not then resume the consumed batch.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                runtime.convert_runtime_batch(batch)
            }));
            if panic_in_handler {
                assert!(result.is_err());
            } else {
                assert!(result.unwrap().is_err());
            }
            assert_eq!(runtime.custom_events().len(), 1);
            assert!(runtime.shot_end().is_err());
            assert!(
                runtime
                    .provide_measurement_outcomes(BTreeMap::new())
                    .is_err()
            );
            assert!(runtime.execute_until_quantum().is_err());
            runtime.reset().unwrap();
            assert!(runtime.shot_end().is_ok());
        }
    }

    #[test]
    fn custom_allocation_failure_is_sticky_without_further_callback_work() {
        let mut batch = RuntimeOperationBatch::default();
        // Deterministic capacity overflow exercises the fallible reserve path
        // without allocating large memory or passing an unreadable payload.
        assert!(!batch.reserve_operations(usize::MAX));
        unsafe {
            runtime_batch_custom((&raw mut batch).cast(), 8404, std::ptr::null(), 0);
            runtime_batch_rz((&raw mut batch).cast(), 0, 0.25);
        }
        assert_eq!(batch.operations, []);
        let mut runtime = SeleneRuntime::new("synthetic-runtime.so");
        let error = runtime
            .convert_runtime_batch(batch)
            .unwrap_err()
            .to_string();
        assert!(error.contains("unable to allocate runtime operation storage"));
        assert_eq!(runtime.shot_end().unwrap_err().to_string(), error);
        runtime.reset().unwrap();
        assert!(runtime.shot_end().is_ok());
    }

    #[test]
    fn custom_callback_preserves_owned_payload() {
        let mut batch = RuntimeOperationBatch::default();
        let mut payload = vec![17_u8, 29, 43];
        unsafe {
            runtime_batch_custom(
                (&raw mut batch).cast(),
                7301,
                payload.as_ptr().cast(),
                payload.len(),
            );
        }
        payload.fill(0);
        drop(payload);
        assert!(matches!(&batch.operations[..],
            [RuntimeScheduledOp::Custom { tag: 7301, data }] if data == &[17, 29, 43]));
    }

    #[test]
    fn custom_events_round_trip_through_selene_batch_extractor() {
        use selene_core::operation::{BatchOperation, Operation, plugin::BatchExtractor};
        let source = BatchOperation::runtime(
            vec![
                Operation::Custom {
                    custom_tag: 7301,
                    data: vec![17, 29].into_boxed_slice(),
                },
                Operation::RZGate {
                    qubit_id: 0,
                    theta: 0.25,
                },
                Operation::Custom {
                    custom_tag: 7302,
                    data: vec![43].into_boxed_slice(),
                },
            ],
            20.into(),
            5.into(),
        );
        let mut extractor = BatchExtractor::from_batch_operation(source);
        let input = extractor.runtime_batch_extraction();
        let mut batch = RuntimeOperationBatch::default();
        let output = SeleneRuntimeGetOperationHandle {
            instance: (&raw mut batch).cast(),
            interface: RUNTIME_OPERATION_CALLBACKS,
        };
        // Exercise the same repr(C) handle conversion as the plugin drain path.
        unsafe {
            (input.interface.extract_fn)(
                input.instance,
                std::mem::transmute::<
                    SeleneRuntimeGetOperationHandle,
                    selene_core::operation::plugin::RuntimeGetOperationHandle,
                >(output),
            );
        }
        drop(extractor);
        let mut runtime = SeleneRuntime::new("synthetic-runtime.so");
        runtime.convert_runtime_batch(batch).unwrap();
        assert_eq!(
            runtime
                .custom_events()
                .iter()
                .map(|event| (
                    event.tag,
                    event.operation_index,
                    event.start_time_nanos,
                    event.duration_nanos
                ))
                .collect::<Vec<_>>(),
            [(7301, 0, 20, 5), (7302, 2, 20, 5)]
        );
        assert_eq!(runtime.custom_events()[0].data, [17, 29]);
        assert_eq!(runtime.custom_events()[1].data, [43]);
    }

    fn synthetic_custom_batch(tag: usize) -> RuntimeOperationBatch {
        let mut batch = RuntimeOperationBatch::default();
        let payload = [17_u8, 29, 43];
        unsafe {
            runtime_batch_rz((&raw mut batch).cast(), 0, 0.25);
            runtime_batch_custom(
                (&raw mut batch).cast(),
                tag,
                payload.as_ptr().cast(),
                payload.len(),
            );
            runtime_batch_measure((&raw mut batch).cast(), 1, 901);
            // Timing is allowed to arrive after the operations.
            runtime_batch_set_time((&raw mut batch).cast(), 20, 5);
        }
        batch
    }

    #[test]
    fn custom_capture_preserves_order_timing_and_measurement_routing() {
        let mut runtime = SeleneRuntime::new("synthetic-runtime.so");
        runtime.runtime_to_program_results.insert(901, 7);
        let ops = runtime
            .convert_runtime_batch(synthetic_custom_batch(7301))
            .unwrap();
        assert_eq!(
            ops,
            vec![
                QuantumOp::Idle(20e-9, 0),
                QuantumOp::RZ(0.25, 0),
                QuantumOp::Idle(20e-9, 1),
                QuantumOp::Measure(1, 7)
            ]
        );
        assert_eq!(runtime.last_gate_time_end_nanos, [25, 25]);
        runtime
            .convert_runtime_batch(RuntimeOperationBatch::default())
            .unwrap();
        let mut next_batch = synthetic_custom_batch(7302);
        next_batch.start_time_nanos = 30;
        let next_ops = runtime.convert_runtime_batch(next_batch).unwrap();
        assert_eq!(
            next_ops,
            vec![
                QuantumOp::Idle(5e-9, 0),
                QuantumOp::RZ(0.25, 0),
                QuantumOp::Idle(5e-9, 1),
                QuantumOp::Measure(1, 7)
            ]
        );
        let events = runtime.take_custom_events();
        assert_eq!(events.len(), 2);
        assert_eq!(
            events[0],
            RuntimeCustomEvent {
                tag: 7301,
                data: vec![17, 29, 43],
                batch_index: 0,
                operation_index: 1,
                start_time_nanos: 20,
                duration_nanos: 5,
            }
        );
        assert_eq!(events[1].tag, 7302);
        assert_eq!(events[1].batch_index, 2);
        assert_eq!(runtime.custom_events(), []);
        assert_eq!(runtime.runtime_batch_index, 3);
    }

    #[test]
    fn custom_strict_policy_requires_explicit_metadata_acknowledgement() {
        let mut runtime = SeleneRuntime::new("synthetic-runtime.so");
        runtime.runtime_to_program_results.insert(901, 7);
        runtime.set_custom_event_policy(RuntimeCustomEventPolicy::RejectUnhandled);
        let error = runtime
            .convert_runtime_batch(synthetic_custom_batch(7301))
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("tag 7301 at batch 0, operation 1, start 20 ns")
        );
        assert_eq!(runtime.custom_events().len(), 1);
        runtime.set_custom_event_handler(|event| {
            assert_eq!(event.data, [17, 29, 43]);
            assert_eq!(event.start_time_nanos, 20);
            Ok(if event.tag == 7301 {
                RuntimeCustomEventDisposition::MetadataOnly
            } else {
                RuntimeCustomEventDisposition::Unsupported
            })
        });
        runtime.reset().unwrap();
        runtime.runtime_to_program_results.insert(901, 7);
        runtime
            .convert_runtime_batch(synthetic_custom_batch(7301))
            .unwrap();
        runtime.reset().unwrap();
        runtime.runtime_to_program_results.insert(901, 7);
        assert!(
            runtime
                .convert_runtime_batch(synthetic_custom_batch(7302))
                .is_err()
        );
        // Compatibility mode retains unsupported events and handler failures still propagate.
        runtime.set_custom_event_policy(RuntimeCustomEventPolicy::Capture);
        runtime.reset().unwrap();
        runtime.runtime_to_program_results.insert(901, 7);
        runtime
            .convert_runtime_batch(synthetic_custom_batch(7302))
            .unwrap();
        runtime.set_custom_event_handler(|_| {
            Err(RuntimeError::ExecutionError(
                "synthetic handler failure".into(),
            ))
        });
        runtime.reset().unwrap();
        runtime.runtime_to_program_results.insert(901, 7);
        assert!(
            runtime
                .convert_runtime_batch(synthetic_custom_batch(7301))
                .unwrap_err()
                .to_string()
                .contains("synthetic handler failure")
        );
    }

    #[test]
    fn custom_callback_validates_lengths_without_dereferencing_invalid_input() {
        for (ptr, len) in [
            (std::ptr::null(), 1),
            (std::ptr::dangling::<u8>().cast(), usize::MAX),
        ] {
            let mut batch = RuntimeOperationBatch::default();
            unsafe {
                runtime_batch_custom((&raw mut batch).cast(), 7301, ptr, len);
            }
            let mut runtime = SeleneRuntime::new("synthetic-runtime.so");
            assert!(matches!(
                runtime.convert_runtime_batch(batch),
                Err(RuntimeError::FfiError(_))
            ));
        }
        let mut batch = RuntimeOperationBatch::default();
        unsafe {
            runtime_batch_custom((&raw mut batch).cast(), 7301, std::ptr::null(), 0);
        }
        let mut runtime = SeleneRuntime::new("synthetic-runtime.so");
        runtime.convert_runtime_batch(batch).unwrap();
        let data = &runtime.custom_events()[0].data;
        assert!(
            data.is_empty(),
            "expected no custom event data, got {data:?}"
        );
    }

    #[test]
    fn custom_capture_lifecycle_and_cloned_configuration() {
        let mut runtime = SeleneRuntime::new("synthetic-runtime.so");
        runtime.runtime_to_program_results.insert(901, 7);
        runtime.set_custom_event_policy(RuntimeCustomEventPolicy::RejectUnhandled);
        runtime.set_custom_event_handler(|_| Ok(RuntimeCustomEventDisposition::MetadataOnly));
        runtime.last_gate_time_end_nanos.clear();
        runtime
            .convert_runtime_batch(synthetic_custom_batch(7301))
            .unwrap();
        runtime.shot_end().unwrap();
        assert_eq!(runtime.custom_events().len(), 1);
        let mut cloned = runtime.clone();
        cloned.shot_start(2, Some(13)).unwrap();
        assert_eq!(cloned.custom_events(), []);
        assert_eq!(cloned.runtime_batch_index, 0);
        assert_eq!(
            cloned.custom_event_policy,
            RuntimeCustomEventPolicy::RejectUnhandled
        );
        assert!(cloned.custom_event_handler.is_some());
        assert_eq!(runtime.custom_events().len(), 1);
        runtime.reset().unwrap();
        assert_eq!(runtime.custom_events(), []);
        assert_eq!(runtime.runtime_batch_index, 0);
    }

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
            callback_error: None,
            extraction_budget: None,
            payload_bytes: 0,
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

    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn test_rxyxy2q_round_trip_through_simple_runtime() {
        let mut runtime = crate::selene_runtimes::selene_simple_runtime().unwrap();
        let metadata = TraceMetadata::from([("source_label".to_string(), "xyxy".to_string())]);
        let lowered = runtime
            .lower_operations_with_metadata(&[
                Operation::AllocateQubit { id: 7 },
                Operation::AllocateQubit { id: 3 },
                Operation::TraceMetadata {
                    metadata: metadata.clone(),
                    qubit: Some(7),
                },
                QuantumOp::RXYXY2Q(-0.73, 0.41, 3, 7).into(),
            ])
            .unwrap();
        // The runtime assigns physical qubits in allocation order. The
        // source handles are deliberately different so we check the mapping
        // and the callback, as well as the two angles and source metadata.
        let gates = lowered
            .iter()
            .filter(|gate| matches!(gate.op, QuantumOp::RXYXY2Q(..)))
            .collect::<Vec<_>>();
        assert_eq!(gates.len(), 1, "{lowered:?}");
        assert_eq!(gates[0].op, QuantumOp::RXYXY2Q(-0.73, 0.41, 1, 0));
        assert_eq!(gates[0].metadata, metadata);
    }

    #[cfg(feature = "selene-runtimes")]
    #[derive(Clone, Copy, Debug)]
    enum LoweringRoute {
        Flat,
        Metadata,
        Scheduled,
    }

    #[cfg(feature = "selene-runtimes")]
    impl LoweringRoute {
        const ALL: [Self; 3] = [Self::Flat, Self::Metadata, Self::Scheduled];

        fn lower(self, runtime: &mut SeleneRuntime, ops: &[Operation]) -> Result<()> {
            match self {
                Self::Flat => runtime.lower_operations(ops).map(|_| ()),
                Self::Metadata => runtime.lower_operations_with_metadata(ops).map(|_| ()),
                Self::Scheduled => runtime.lower_scheduled_operations(ops).map(|_| ()),
            }
        }

        fn drain(self, runtime: &mut SeleneRuntime) {
            match self {
                Self::Flat | Self::Metadata => {
                    runtime.drain_pending_operations().unwrap();
                }
                Self::Scheduled => {
                    runtime.drain_pending_scheduled_operations().unwrap();
                }
            }
        }
    }

    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn explicit_measurement_lifetime_survives_until_release() {
        for mode in LoweringRoute::ALL {
            for leaked in [false, true] {
                for mut runtime in [
                    crate::selene_runtimes::selene_simple_runtime().unwrap(),
                    crate::selene_runtimes::selene_soft_rz_runtime().unwrap(),
                ] {
                    runtime.set_num_qubits(1);
                    runtime.shot_start(0, Some(7)).unwrap();
                    mode.lower(
                        &mut runtime,
                        &[
                            Operation::AllocateQubit { id: 71 },
                            QuantumOp::Reset(71).into(),
                        ],
                    )
                    .unwrap();
                    let original = runtime.program_to_runtime_qubits[&71];
                    for result in 0..2 {
                        let measurement = if leaked {
                            QuantumOp::MeasureLeaked(71, result)
                        } else {
                            QuantumOp::Measure(71, result)
                        };
                        // Separate submissions exercise persistence after feedback,
                        // including a continuation with no allocation records.
                        mode.lower(&mut runtime, &[measurement.into()]).unwrap();
                        assert_eq!(
                            runtime.program_to_runtime_qubits.get(&71),
                            Some(&original),
                            "measurement released a live handle: mode={mode:?}, leaked={leaked}"
                        );
                        runtime
                            .provide_measurement_outcomes(BTreeMap::from([(result, 0)]))
                            .unwrap();
                        mode.lower(&mut runtime, &[QuantumOp::Reset(71).into()])
                            .unwrap();
                        assert_eq!(runtime.program_to_runtime_qubits[&71], original);
                    }
                    mode.lower(&mut runtime, &[Operation::ReleaseQubit { id: 71 }])
                        .unwrap();
                    assert!(!runtime.program_to_runtime_qubits.contains_key(&71));
                    // A one-slot runtime can allocate again only after release.
                    mode.lower(&mut runtime, &[Operation::AllocateQubit { id: 93 }])
                        .unwrap();
                    assert_eq!(runtime.program_to_runtime_qubits.len(), 1);
                    mode.lower(&mut runtime, &[Operation::ReleaseQubit { id: 93 }])
                        .unwrap();
                    mode.drain(&mut runtime);
                    runtime.shot_end().unwrap();
                }
            }
        }
    }

    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn legacy_measurement_still_releases_implicit_handles() {
        for mode in LoweringRoute::ALL {
            for measurement in [QuantumOp::Measure(0, 0), QuantumOp::MeasureLeaked(0, 0)] {
                let mut runtime = crate::selene_runtimes::selene_simple_runtime().unwrap();
                runtime.set_num_qubits(1);
                runtime.shot_start(0, None).unwrap();
                mode.lower(&mut runtime, &[measurement.into()]).unwrap();
                assert!(runtime.program_to_runtime_qubits.is_empty());
            }
        }
    }

    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn explicit_lifetime_mode_does_not_leak_into_a_new_legacy_shot() {
        for mode in LoweringRoute::ALL {
            for reset in [false, true] {
                let mut runtime = crate::selene_runtimes::selene_simple_runtime().unwrap();
                runtime.set_num_qubits(1);
                runtime.shot_start(0, None).unwrap();
                mode.lower(
                    &mut runtime,
                    &[
                        Operation::AllocateQubit { id: 71 },
                        Operation::ReleaseQubit { id: 71 },
                    ],
                )
                .unwrap();
                mode.drain(&mut runtime);
                runtime.shot_end().unwrap();
                if reset {
                    runtime.reset().unwrap();
                }
                runtime.shot_start(1, None).unwrap();
                mode.lower(&mut runtime, &[QuantumOp::Measure(0, 0).into()])
                    .unwrap();
                assert!(runtime.program_to_runtime_qubits.is_empty());
            }
        }
    }

    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn continuation_capacity_preserves_native_state() {
        for mode in LoweringRoute::ALL {
            for grow_by in [1, 2] {
                let mut runtime = crate::selene_runtimes::selene_simple_runtime().unwrap();
                if matches!(mode, LoweringRoute::Scheduled) {
                    runtime.set_num_qubits(1);
                }
                runtime.shot_start(4, Some(9)).unwrap();
                mode.lower(
                    &mut runtime,
                    &[
                        Operation::AllocateQubit { id: 0 },
                        QuantumOp::Measure(0, 0).into(),
                    ],
                )
                .unwrap();
                runtime
                    .provide_measurement_outcomes(BTreeMap::from([(0, 1)]))
                    .unwrap();
                let handles = runtime.program_to_runtime_qubits.clone();
                let instance = runtime.instance;
                let active = runtime.active_shot;
                let ops = (1..=grow_by)
                    .map(|id| Operation::AllocateQubit { id })
                    .collect::<Vec<_>>();
                let error = mode
                    .lower(&mut runtime, &ops)
                    .expect_err("live plugin must not resize");
                assert!(error.to_string().contains("capacity"), "{error}");
                assert_eq!(runtime.program_to_runtime_qubits, handles);
                assert_eq!(runtime.instance, instance);
                assert_eq!(runtime.active_shot, active);
                assert_eq!(runtime.initialized_num_qubits, Some(1));
                assert_eq!(
                    runtime.get_classical_state().measurements.get(&0),
                    Some(&true)
                );
                if matches!(mode, LoweringRoute::Scheduled) {
                    assert!(
                        mode.lower(&mut runtime, &[Operation::ReleaseQubit { id: 0 }])
                            .is_err()
                    );
                    assert!(runtime.shot_end().is_err());
                    runtime.reset().unwrap();
                    runtime.shot_start(5, None).unwrap();
                    mode.lower(&mut runtime, &[Operation::AllocateQubit { id: 0 }])
                        .unwrap();
                }
                mode.lower(&mut runtime, &[Operation::ReleaseQubit { id: 0 }])
                    .unwrap();
                mode.drain(&mut runtime);
                runtime.shot_end().unwrap();
            }
        }
    }

    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn mixed_handles_keep_individual_lifetimes() {
        let mut runtime = crate::selene_runtimes::selene_simple_runtime().unwrap();
        runtime.set_num_qubits(2);
        runtime.shot_start(0, None).unwrap();
        runtime
            .lower_operations(&[
                QuantumOp::Reset(0).into(),
                Operation::AllocateQubit { id: 71 },
                QuantumOp::Measure(0, 0).into(),
                QuantumOp::Measure(71, 1).into(),
            ])
            .unwrap();
        assert!(!runtime.program_to_runtime_qubits.contains_key(&0));
        assert!(runtime.program_to_runtime_qubits.contains_key(&71));
    }

    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn loaded_collector_does_not_define_streamed_lifetimes() {
        let mut runtime = crate::selene_runtimes::selene_simple_runtime().unwrap();
        let mut collector = OperationCollector::new();
        collector.queue_operation(Operation::AllocateQubit { id: 71 });
        runtime.load_interface(collector).unwrap();
        runtime.shot_start(0, None).unwrap();
        runtime
            .lower_operations(&[QuantumOp::Measure(0, 0).into()])
            .unwrap();
        assert!(runtime.program_to_runtime_qubits.is_empty());
    }

    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn failed_release_latches_uncertain_native_state() {
        let mut runtime = crate::selene_runtimes::selene_simple_runtime().unwrap();
        runtime.set_num_qubits(1);
        runtime.shot_start(0, None).unwrap();
        runtime
            .lower_operations(&[Operation::AllocateQubit { id: 0 }])
            .unwrap();
        // Force a deterministic public-plugin qfree failure without invalid pointers.
        runtime.program_to_runtime_qubits.insert(0, u64::MAX);
        assert!(
            runtime
                .lower_operations(&[Operation::ReleaseQubit { id: 0 }])
                .is_err()
        );
        assert!(runtime.shot_end().is_err());
        assert!(runtime.lower_operations(&[]).is_err());
        assert!(runtime.clone().shot_end().is_err());
        runtime.reset().unwrap();
        runtime.shot_start(1, None).unwrap();
        runtime
            .lower_operations(&[Operation::AllocateQubit { id: 0 }])
            .unwrap();
        assert_eq!(runtime.explicit_qubit_handles, BTreeSet::from([0]));
    }

    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn no_hint_continuations_use_preloaded_whole_program_capacity() {
        for mode in [LoweringRoute::Flat, LoweringRoute::Metadata] {
            let mut runtime = crate::selene_runtimes::selene_simple_runtime().unwrap();
            let first = [
                Operation::AllocateQubit { id: 71 },
                QuantumOp::Measure(71, 0).into(),
            ];
            let second = [
                Operation::AllocateQubit { id: 93 },
                QuantumOp::Measure(93, 1).into(),
                Operation::ReleaseQubit { id: 93 },
                Operation::ReleaseQubit { id: 71 },
            ];
            let mut collector = OperationCollector::new();
            for op in first.iter().chain(&second) {
                collector.queue_operation(op.clone());
            }
            runtime.load_interface(collector).unwrap();
            runtime.shot_start(4, Some(9)).unwrap();
            mode.lower(&mut runtime, &first).unwrap();
            runtime
                .provide_measurement_outcomes(BTreeMap::from([(0, 1)]))
                .unwrap();
            let instance = runtime.instance;
            mode.lower(&mut runtime, &second).unwrap();
            assert_eq!(runtime.instance, instance);
            assert_eq!(runtime.initialized_num_qubits, Some(2));
            assert_eq!(runtime.active_shot, Some((4, 9)));
            assert!(runtime.program_to_runtime_qubits.is_empty());
            runtime
                .provide_measurement_outcomes(BTreeMap::from([(1, 0)]))
                .unwrap();
            mode.drain(&mut runtime);
            assert_eq!(
                runtime.shot_end().unwrap().measurements,
                BTreeMap::from([(0, true), (1, false)])
            );
        }
    }

    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn unreleased_measurement_handles_do_not_leak_across_completed_public_shots() {
        for mode in LoweringRoute::ALL {
            for mut runtime in [
                crate::selene_runtimes::selene_simple_runtime().unwrap(),
                crate::selene_runtimes::selene_soft_rz_runtime().unwrap(),
            ] {
                runtime.set_num_qubits(1);
                for shot in 0..3 {
                    runtime.shot_start(shot, None).unwrap();
                    mode.lower(
                        &mut runtime,
                        &[
                            Operation::AllocateQubit { id: 71 },
                            QuantumOp::Reset(71).into(),
                            QuantumOp::Measure(71, 0).into(),
                        ],
                    )
                    .unwrap();
                    runtime
                        .provide_measurement_outcomes(BTreeMap::from([(0, 0)]))
                        .unwrap();
                    assert_eq!(runtime.program_to_runtime_qubits.len(), 1);
                    mode.drain(&mut runtime);
                    runtime.shot_end().unwrap();
                }
            }
        }
    }

    #[test]
    fn duplicate_admission_follows_ordered_handle_lifetimes() {
        for (ops, duplicate) in [
            (
                vec![
                    Operation::AllocateQubit { id: 7 },
                    Operation::AllocateQubit { id: 7 },
                ],
                true,
            ),
            (
                vec![
                    QuantumOp::Reset(7).into(),
                    Operation::AllocateQubit { id: 7 },
                ],
                true,
            ),
            (
                vec![
                    Operation::AllocateQubit { id: 7 },
                    QuantumOp::Measure(7, 0).into(),
                    Operation::AllocateQubit { id: 7 },
                ],
                true,
            ),
            (
                vec![
                    Operation::AllocateQubit { id: 7 },
                    Operation::ReleaseQubit { id: 7 },
                    Operation::AllocateQubit { id: 7 },
                ],
                false,
            ),
            (
                vec![
                    QuantumOp::Measure(7, 0).into(),
                    Operation::AllocateQubit { id: 7 },
                ],
                false,
            ),
        ] {
            let capacity = operation_capacity(&ops, BTreeSet::new(), BTreeSet::new());
            assert_eq!(capacity.duplicate_allocation, duplicate.then_some(7));
        }
    }

    #[test]
    fn mixed_capacity_counts_implicit_handles_and_retained_explicit_allocations() {
        let ops = [
            QuantumOp::Reset(0).into(),
            Operation::AllocateQubit { id: 71 },
            QuantumOp::Measure(0, 0).into(),
            Operation::AllocateQubit { id: 93 },
            QuantumOp::Measure(71, 1).into(),
        ];
        assert_eq!(
            operation_capacity(&ops, BTreeSet::new(), BTreeSet::new()),
            InputCapacity {
                qubits: 2,
                results: 2,
                peak_live: 2,
                duplicate_allocation: None
            }
        );
        let live = BTreeSet::from([71, 93]);
        assert_eq!(
            operation_capacity(&[Operation::AllocateQubit { id: 105 }], live.clone(), live),
            InputCapacity {
                qubits: 3,
                results: 0,
                peak_live: 3,
                duplicate_allocation: None
            }
        );
    }

    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn legacy_handle_indices_do_not_require_native_growth() {
        for mode in [LoweringRoute::Flat, LoweringRoute::Metadata] {
            let mut runtime = crate::selene_runtimes::selene_simple_runtime().unwrap();
            runtime.shot_start(0, None).unwrap();
            mode.lower(&mut runtime, &[QuantumOp::Measure(0, 0).into()])
                .unwrap();
            runtime
                .provide_measurement_outcomes(BTreeMap::from([(0, 0)]))
                .unwrap();
            let instance = runtime.instance;
            mode.lower(
                &mut runtime,
                &[QuantumOp::H(71).into(), QuantumOp::Measure(71, 1).into()],
            )
            .unwrap();
            assert_eq!(runtime.instance, instance);
            assert_eq!(runtime.initialized_num_qubits, Some(1));
            assert_eq!(runtime.num_qubits(), 1);
            runtime
                .provide_measurement_outcomes(BTreeMap::from([(1, 1)]))
                .unwrap();
            let error = mode
                .lower(&mut runtime, &[QuantumOp::CX(71, 93).into()])
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("requires capacity 2 but initialized capacity is 1")
            );
            assert_eq!(runtime.instance, instance);
            assert!(runtime.program_to_runtime_qubits.is_empty());
        }
    }

    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn empty_input_does_not_initialize_zero_capacity() {
        for mode in [LoweringRoute::Flat, LoweringRoute::Metadata] {
            let mut runtime = crate::selene_runtimes::selene_simple_runtime().unwrap();
            runtime.shot_start(0, None).unwrap();
            mode.lower(&mut runtime, &[]).unwrap();
            assert!(runtime.instance.is_none());
            mode.lower(
                &mut runtime,
                &[
                    Operation::AllocateQubit { id: 71 },
                    QuantumOp::Measure(71, 0).into(),
                    Operation::ReleaseQubit { id: 71 },
                ],
            )
            .unwrap();
            assert_eq!(runtime.initialized_num_qubits, Some(1));
        }
    }

    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn duplicate_live_allocation_obeys_route_recovery_policy() {
        for mode in LoweringRoute::ALL {
            let mut runtime = crate::selene_runtimes::selene_simple_runtime().unwrap();
            runtime.set_num_qubits(1);
            runtime.shot_start(0, None).unwrap();
            mode.lower(
                &mut runtime,
                &[
                    Operation::AllocateQubit { id: 71 },
                    QuantumOp::Measure(71, 0).into(),
                ],
            )
            .unwrap();
            runtime
                .provide_measurement_outcomes(BTreeMap::from([(0, 0)]))
                .unwrap();
            assert!(
                mode.lower(&mut runtime, &[Operation::AllocateQubit { id: 71 }])
                    .is_err()
            );
            if matches!(mode, LoweringRoute::Scheduled) {
                assert!(
                    mode.lower(&mut runtime, &[Operation::ReleaseQubit { id: 71 }])
                        .is_err()
                );
                assert!(runtime.shot_end().is_err());
                runtime.reset().unwrap();
                runtime.shot_start(1, None).unwrap();
            } else {
                mode.lower(&mut runtime, &[Operation::ReleaseQubit { id: 71 }])
                    .unwrap();
            }
            mode.lower(
                &mut runtime,
                &[
                    Operation::AllocateQubit { id: 71 },
                    Operation::ReleaseQubit { id: 71 },
                    Operation::AllocateQubit { id: 71 },
                ],
            )
            .unwrap();
        }
    }

    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn native_operation_errors_latch_after_partial_input() {
        for mode in [LoweringRoute::Flat, LoweringRoute::Metadata] {
            for failure in 0..5 {
                let mut runtime = crate::selene_runtimes::selene_simple_runtime().unwrap();
                runtime.set_num_qubits(2);
                runtime.shot_start(0, None).unwrap();
                mode.lower(&mut runtime, &[Operation::AllocateQubit { id: 0 }])
                    .unwrap();
                let bad_op: Operation = if failure == 0 {
                    // Occupy the spare native slot without updating the host map.
                    runtime.runtime_qalloc().unwrap();
                    Operation::AllocateQubit { id: 1 }
                } else {
                    // A valid library rejects an out-of-range integer handle.
                    runtime.program_to_runtime_qubits.insert(1, u64::MAX);
                    match failure {
                        1 => QuantumOp::Reset(1),
                        2 => QuantumOp::Measure(1, 0),
                        3 => QuantumOp::MeasureLeaked(1, 0),
                        _ => QuantumOp::RXY(0.5, 0.0, 1),
                    }
                    .into()
                };
                assert!(
                    mode.lower(&mut runtime, &[QuantumOp::RXY(0.5, 0.0, 0).into(), bad_op])
                        .is_err()
                );
                assert!(runtime.shot_end().is_err());
                assert!(runtime.clone().shot_end().is_err());
                assert!(mode.lower(&mut runtime, &[]).is_err());
                runtime.reset().unwrap();
                runtime.shot_start(1, None).unwrap();
                mode.lower(&mut runtime, &[Operation::AllocateQubit { id: 0 }])
                    .unwrap();
            }
        }
    }

    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn duplicate_preflight_preserves_flat_native_state() {
        for mode in [LoweringRoute::Flat, LoweringRoute::Metadata] {
            let mut runtime = crate::selene_runtimes::selene_simple_runtime().unwrap();
            runtime.set_num_qubits(2);
            runtime.shot_start(0, None).unwrap();
            mode.lower(&mut runtime, &[Operation::AllocateQubit { id: 0 }])
                .unwrap();
            let handles = runtime.program_to_runtime_qubits.clone();
            let instance = runtime.instance;
            let error = mode
                .lower(
                    &mut runtime,
                    &[
                        Operation::AllocateQubit { id: 1 },
                        Operation::AllocateQubit { id: 0 },
                    ],
                )
                .unwrap_err();
            assert!(error.to_string().contains("already allocated"));
            assert_eq!(runtime.program_to_runtime_qubits, handles);
            assert_eq!(runtime.instance, instance);
            assert!(runtime.batch_failure.is_none());
            mode.lower(
                &mut runtime,
                &[
                    Operation::ReleaseQubit { id: 0 },
                    Operation::AllocateQubit { id: 0 },
                    Operation::ReleaseQubit { id: 0 },
                ],
            )
            .unwrap();
            mode.drain(&mut runtime);
            runtime.shot_end().unwrap();
        }
    }

    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn failed_lazy_shot_start_requires_reset() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        static FAIL_START: AtomicBool = AtomicBool::new(true);
        static START_CALLS: AtomicUsize = AtomicUsize::new(0);
        unsafe extern "C" fn injected_start(_: *mut c_void, _: u64, _: u64) -> i32 {
            START_CALLS.fetch_add(1, Ordering::SeqCst);
            if FAIL_START.load(Ordering::SeqCst) {
                42
            } else {
                0
            }
        }

        // Export a descriptor accessor from a tiny test library. Its descriptor
        // lives in this process and delegates every other callback to the real
        // public runtime; no ABI layout duplication or invalid pointers are used.
        let executable = std::env::current_exe().unwrap();
        let plugin = crate::selene_runtimes::find_library_in_dir(
            executable.parent().unwrap(),
            pecos_qis_test_runtime::LIBRARY_NAME,
        )
        .expect("Cargo-built runtime fixture beside the test executable");
        let public_runtime = crate::selene_runtimes::selene_simple_runtime().unwrap();
        // SAFETY: Both libraries stay loaded and the boxed descriptor stays alive
        // until all runtimes have been reset. Every callback has its original ABI.
        let public_library =
            unsafe { libloading::Library::new(&public_runtime.plugin_path).unwrap() };
        let mut descriptor =
            Box::new(unsafe { SeleneRuntime::runtime_plugin_descriptor(&public_library).unwrap() });
        descriptor.shot_start_fn = injected_start;
        let fixture = unsafe { libloading::Library::new(&plugin).unwrap() };
        unsafe {
            let set = fixture
                .get::<unsafe extern "C" fn(*mut c_void)>(b"set_descriptor")
                .unwrap();
            set((&raw mut *descriptor).cast());
        }
        for mode in LoweringRoute::ALL {
            FAIL_START.store(true, Ordering::SeqCst);
            START_CALLS.store(0, Ordering::SeqCst);
            let mut runtime = SeleneRuntime::new(&plugin);
            runtime.init_args.clone_from(&public_runtime.init_args);
            runtime.set_num_qubits(1);
            runtime.shot_start(3, Some(5)).unwrap();
            let error = mode
                .lower(&mut runtime, &[Operation::AllocateQubit { id: 0 }])
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("Shot start failed with errno 42")
            );
            assert!(runtime.instance.is_some());
            assert!(runtime.active_shot.is_none());
            FAIL_START.store(false, Ordering::SeqCst);
            assert!(
                mode.lower(&mut runtime, &[Operation::AllocateQubit { id: 0 }])
                    .is_err()
            );
            assert!(runtime.shot_end().is_err());
            assert!(runtime.clone().shot_end().is_err());
            assert_eq!(START_CALLS.load(Ordering::SeqCst), 1);
            runtime.reset().unwrap();
            runtime.shot_start(4, Some(6)).unwrap();
            mode.lower(&mut runtime, &[Operation::AllocateQubit { id: 0 }])
                .unwrap();
            assert_eq!(runtime.active_shot, Some((4, 6)));
            assert_eq!(START_CALLS.load(Ordering::SeqCst), 2);
            mode.drain(&mut runtime);
            runtime.shot_end().unwrap();
            runtime.reset().unwrap();
        }
    }

    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn result_only_input_defers_native_capacity() {
        for mode in [LoweringRoute::Flat, LoweringRoute::Metadata] {
            let mut runtime = crate::selene_runtimes::selene_simple_runtime().unwrap();
            runtime.shot_start(7, Some(11)).unwrap();
            mode.lower(
                &mut runtime,
                &[Operation::AllocateResult { id: 5 }, Operation::Barrier],
            )
            .unwrap();
            assert!(runtime.instance.is_none());
            assert_eq!(runtime.num_results, 6);
            assert_eq!(runtime.pending_shot_start, Some((7, Some(11))));
            mode.lower(
                &mut runtime,
                &[
                    Operation::AllocateQubit { id: 71 },
                    QuantumOp::Measure(71, 5).into(),
                    Operation::ReleaseQubit { id: 71 },
                ],
            )
            .unwrap();
            assert_eq!(runtime.initialized_num_qubits, Some(1));
            assert_eq!(runtime.active_shot, Some((7, 11)));
            runtime
                .provide_measurement_outcomes(BTreeMap::from([(5, 1)]))
                .unwrap();
            mode.drain(&mut runtime);
            assert!(runtime.shot_end().unwrap().measurements[&5]);
        }
    }

    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn live_native_clone_requires_reset_before_reallocation() {
        for mode in LoweringRoute::ALL {
            for measured in [false, true] {
                let mut runtime = crate::selene_runtimes::selene_simple_runtime().unwrap();
                runtime.set_num_qubits(2);
                runtime.shot_start(0, None).unwrap();
                mode.lower(
                    &mut runtime,
                    &[
                        Operation::AllocateQubit { id: 71 },
                        QuantumOp::RXY(0.5, 0.0, 71).into(),
                    ],
                )
                .unwrap();
                if measured {
                    mode.lower(&mut runtime, &[QuantumOp::Measure(71, 0).into()])
                        .unwrap();
                    runtime
                        .provide_measurement_outcomes(BTreeMap::from([(0, 0)]))
                        .unwrap();
                }
                let mut cloned = runtime.clone();
                assert!(
                    mode.lower(&mut cloned, &[Operation::AllocateQubit { id: 72 }])
                        .is_err()
                );
                assert!(cloned.shot_end().is_err());
                assert!(cloned.clone().shot_start(1, None).is_err());
                // The original remains usable; only the unsupported snapshot is rejected.
                mode.lower(&mut runtime, &[Operation::AllocateQubit { id: 72 }])
                    .unwrap();
                assert_ne!(
                    runtime.program_to_runtime_qubits[&71],
                    runtime.program_to_runtime_qubits[&72]
                );
                cloned.reset().unwrap();
                cloned.shot_start(1, None).unwrap();
                mode.lower(
                    &mut cloned,
                    &[
                        Operation::AllocateQubit { id: 71 },
                        Operation::AllocateQubit { id: 72 },
                    ],
                )
                .unwrap();
                assert_ne!(
                    cloned.program_to_runtime_qubits[&71],
                    cloned.program_to_runtime_qubits[&72]
                );
            }
        }
    }

    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn failed_feedback_delivery_requires_reset() {
        for mode in LoweringRoute::ALL {
            for leaked in [false, true] {
                let mut runtime = crate::selene_runtimes::selene_simple_runtime().unwrap();
                runtime.set_num_qubits(1);
                runtime.shot_start(0, None).unwrap();
                let measurement = if leaked {
                    QuantumOp::MeasureLeaked(0, 0)
                } else {
                    QuantumOp::Measure(0, 0)
                };
                mode.lower(
                    &mut runtime,
                    &[Operation::AllocateQubit { id: 0 }, measurement.into()],
                )
                .unwrap();
                // The real plugin rejects an unknown integer result ID.
                runtime.program_to_runtime_results.insert(0, u64::MAX);
                let error = runtime
                    .provide_measurement_outcomes(BTreeMap::from([(0, 1)]))
                    .unwrap_err();
                assert!(error.to_string().contains("set_"));
                assert!(
                    runtime
                        .provide_measurement_outcomes(BTreeMap::new())
                        .is_err()
                );
                assert!(runtime.shot_end().is_err());
                assert!(runtime.clone().shot_end().is_err());
                runtime.reset().unwrap();
                runtime.shot_start(1, None).unwrap();
                mode.lower(&mut runtime, &[QuantumOp::Measure(0, 0).into()])
                    .unwrap();
                runtime
                    .provide_measurement_outcomes(BTreeMap::from([(0, 0)]))
                    .unwrap();
                mode.drain(&mut runtime);
                runtime.shot_end().unwrap();
            }
        }
    }

    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn failed_terminal_barrier_requires_reset() {
        for mode in [LoweringRoute::Flat, LoweringRoute::Metadata] {
            let mut runtime = crate::selene_runtimes::selene_simple_runtime().unwrap();
            runtime.set_num_qubits(1);
            runtime.shot_start(0, None).unwrap();
            mode.lower(&mut runtime, &[Operation::AllocateQubit { id: 0 }])
                .unwrap();
            // Simulate an unavailable barrier API without supplying invalid FFI pointers.
            let library = runtime.library.take();
            let error = runtime.drain_pending_operations().unwrap_err();
            runtime.library = library;
            assert!(error.to_string().contains("runtime is not loaded"));
            assert!(runtime.shot_end().is_err());
            assert!(runtime.drain_pending_operations().is_err());
            assert!(runtime.clone().shot_end().is_err());
            runtime.reset().unwrap();
            runtime.shot_start(1, None).unwrap();
            mode.lower(&mut runtime, &[Operation::AllocateQubit { id: 0 }])
                .unwrap();
            mode.drain(&mut runtime);
            runtime.shot_end().unwrap();
        }
    }

    #[cfg(feature = "selene-runtimes")]
    #[test]
    fn caught_submission_panic_cannot_resume_native_state() {
        let mut runtime = crate::selene_runtimes::selene_simple_runtime().unwrap();
        runtime.set_num_qubits(1);
        runtime.shot_start(0, None).unwrap();
        runtime
            .lower_operations(&[Operation::AllocateQubit { id: 0 }])
            .unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.with_native_mutation::<()>(|runtime| {
                runtime.call_runtime_rxy(0, 0.5, 0.0)?;
                panic!("synthetic panic after native mutation");
            })
        }));
        assert!(result.is_err());
        assert!(runtime.shot_end().is_err());
        assert!(runtime.lower_operations(&[]).is_err());
        runtime.reset().unwrap();
        runtime.shot_start(1, None).unwrap();
        runtime
            .lower_operations(&[Operation::AllocateQubit { id: 0 }])
            .unwrap();
    }

    #[test]
    fn test_runtime_batch_rpp_preserves_angles_and_timing() {
        let mut runtime = SeleneRuntime::new("/path/to/selene.so");
        runtime.last_gate_time_end_nanos = vec![5, 10, 0];
        let ops = runtime
            .convert_runtime_batch(RuntimeOperationBatch {
                start_time_nanos: 20,
                duration_nanos: 7,
                invoked: true,
                callback_error: None,
                extraction_budget: None,
                payload_bytes: 0,
                operations: vec![RuntimeScheduledOp::Rpp {
                    qubit_id_1: 1,
                    qubit_id_2: 0,
                    theta: -0.73,
                    phi: 0.41,
                }],
            })
            .unwrap();

        // Both qubits wait until 20ns, but their preceding gates ended at
        // different times. The RPP should become one gate with both angles.
        assert_eq!(
            ops,
            vec![
                QuantumOp::Idle(10e-9, 1),
                QuantumOp::Idle(15e-9, 0),
                QuantumOp::RXYXY2Q(-0.73, 0.41, 1, 0),
            ]
        );
        assert_eq!(runtime.last_gate_time_end_nanos, vec![27, 27, 0]);
        let mut following = Vec::new();
        runtime.push_idle_before(&mut following, 0, 30).unwrap();
        runtime.push_idle_before(&mut following, 1, 30).unwrap();
        assert_eq!(
            following,
            vec![QuantumOp::Idle(3e-9, 0), QuantumOp::Idle(3e-9, 1)]
        );
    }

    #[test]
    fn test_rxyxy2q_metadata_matches_both_angles_and_qubits() {
        let source = QuantumOp::RXYXY2Q(-0.73, 0.41, 2, 5);
        assert!(SeleneRuntime::source_op_matches_lowered_op(
            &source,
            &QuantumOp::RXYXY2Q(-0.73, 0.41, 5, 2),
        ));
        for other in [
            QuantumOp::RXYXY2Q(0.73, 0.41, 2, 5),
            QuantumOp::RXYXY2Q(-0.73, -0.41, 2, 5),
            QuantumOp::RXYXY2Q(-0.73, 0.41, 2, 4),
        ] {
            assert!(!SeleneRuntime::source_op_matches_lowered_op(
                &source, &other
            ));
        }
        let metadata = TraceMetadata::from([("source_label".to_string(), "xyxy".to_string())]);
        let mut pending = VecDeque::from([SourceTraceMetadata {
            op: source.clone(),
            metadata: metadata.clone(),
        }]);
        let mut lowered = Vec::new();
        SeleneRuntime::push_lowered_ops_with_source_metadata(
            &mut lowered,
            vec![QuantumOp::Idle(3e-9, 2), source.clone()],
            &mut pending,
        );
        assert!(pending.is_empty());
        assert!(lowered[0].metadata.is_empty());
        assert_eq!(lowered[1], LoweredQuantumOp::new(source.clone(), metadata));
        assert_eq!(
            SeleneRuntime::quantum_op_qubits(&source),
            BTreeSet::from([2, 5])
        );
        assert_eq!(SeleneRuntime::two_qubit_gate_qubits(&source), Some((2, 5)));
        let mut collector = OperationCollector::default();
        collector.operations.push(source.into());
        assert_eq!(collector_capacity(&collector), (6, 0));
    }

    #[test]
    fn test_source_metadata_attaches_to_first_non_idle_lowered_op() {
        let mut metadata = TraceMetadata::new();
        metadata.insert("source_label".to_string(), "probe:szz-host".to_string());
        let mut source_metadata = VecDeque::from([SourceTraceMetadata {
            op: QuantumOp::RZZ(0.5, 0, 1),
            metadata,
        }]);
        let mut lowered_ops = Vec::new();

        SeleneRuntime::push_lowered_ops_with_source_metadata(
            &mut lowered_ops,
            vec![QuantumOp::Idle(20e-9, 0), QuantumOp::RZZ(0.5, 0, 1)],
            &mut source_metadata,
        );

        assert!(lowered_ops[0].metadata.is_empty());
        assert_eq!(
            lowered_ops[1]
                .metadata
                .get("source_label")
                .map(String::as_str),
            Some("probe:szz-host")
        );
        assert!(source_metadata.is_empty());
    }

    #[test]
    fn test_source_idle_metadata_can_attach_to_idle_op() {
        let mut metadata = TraceMetadata::new();
        metadata.insert("source_label".to_string(), "probe:idle".to_string());
        let mut source_metadata = VecDeque::from([SourceTraceMetadata {
            op: QuantumOp::Idle(20e-9, 0),
            metadata,
        }]);
        let mut lowered_ops = Vec::new();

        SeleneRuntime::push_lowered_ops_with_source_metadata(
            &mut lowered_ops,
            vec![QuantumOp::Idle(20e-9, 0)],
            &mut source_metadata,
        );

        assert_eq!(
            lowered_ops[0]
                .metadata
                .get("source_label")
                .map(String::as_str),
            Some("probe:idle")
        );
        assert!(source_metadata.is_empty());
    }

    #[test]
    fn test_unmatched_metadata_does_not_block_later_compatible_metadata() {
        let mut rz_metadata = TraceMetadata::new();
        rz_metadata.insert("source_label".to_string(), "probe:virtual-rz".to_string());
        let mut rzz_metadata = TraceMetadata::new();
        rzz_metadata.insert("source_label".to_string(), "probe:szz-host".to_string());
        let mut source_metadata = VecDeque::from([
            SourceTraceMetadata {
                op: QuantumOp::RZ(0.25, 0),
                metadata: rz_metadata,
            },
            SourceTraceMetadata {
                op: QuantumOp::RZZ(0.5, 0, 1),
                metadata: rzz_metadata,
            },
        ]);
        let mut lowered_ops = Vec::new();

        SeleneRuntime::push_lowered_ops_with_source_metadata(
            &mut lowered_ops,
            vec![QuantumOp::RZZ(0.5, 0, 1)],
            &mut source_metadata,
        );

        assert_eq!(
            lowered_ops[0]
                .metadata
                .get("source_label")
                .map(String::as_str),
            Some("probe:szz-host")
        );
        assert_eq!(source_metadata.len(), 1);
        assert_eq!(
            source_metadata[0]
                .metadata
                .get("source_label")
                .map(String::as_str),
            Some("probe:virtual-rz")
        );
    }

    #[test]
    fn test_source_metadata_can_attach_after_runtime_reordering() {
        let mut first_metadata = TraceMetadata::new();
        first_metadata.insert("source_label".to_string(), "probe:first".to_string());
        let mut second_metadata = TraceMetadata::new();
        second_metadata.insert("source_label".to_string(), "probe:second".to_string());
        let mut source_metadata = VecDeque::from([
            SourceTraceMetadata {
                op: QuantumOp::RZZ(0.5, 0, 1),
                metadata: first_metadata,
            },
            SourceTraceMetadata {
                op: QuantumOp::RZZ(-0.5, 2, 3),
                metadata: second_metadata,
            },
        ]);
        let mut lowered_ops = Vec::new();

        SeleneRuntime::push_lowered_ops_with_source_metadata(
            &mut lowered_ops,
            vec![QuantumOp::RZZ(-0.5, 2, 3), QuantumOp::RZZ(0.5, 0, 1)],
            &mut source_metadata,
        );

        assert_eq!(
            lowered_ops[0]
                .metadata
                .get("source_label")
                .map(String::as_str),
            Some("probe:second")
        );
        assert_eq!(
            lowered_ops[1]
                .metadata
                .get("source_label")
                .map(String::as_str),
            Some("probe:first")
        );
        assert!(source_metadata.is_empty());
    }

    #[test]
    fn test_source_gate_metadata_matches_runtime_normalized_single_qubit_pulse() {
        let mut metadata = TraceMetadata::new();
        metadata.insert("source_gate".to_string(), "H".to_string());
        metadata.insert("source_label".to_string(), "probe:h-prefix".to_string());
        let mut source_metadata = VecDeque::from([SourceTraceMetadata {
            op: QuantumOp::RXY(std::f64::consts::FRAC_PI_2, -std::f64::consts::FRAC_PI_2, 2),
            metadata,
        }]);
        let mut lowered_ops = Vec::new();

        SeleneRuntime::push_lowered_ops_with_source_metadata(
            &mut lowered_ops,
            vec![QuantumOp::RXY(std::f64::consts::FRAC_PI_2, 0.0, 2)],
            &mut source_metadata,
        );

        assert_eq!(
            lowered_ops[0]
                .metadata
                .get("source_label")
                .map(String::as_str),
            Some("probe:h-prefix")
        );
        assert!(source_metadata.is_empty());
    }

    #[test]
    fn test_qubit_scoped_metadata_waits_for_compatible_source_op() {
        let mut h_metadata = TraceMetadata::new();
        h_metadata.insert("source_gate".to_string(), "H".to_string());
        h_metadata.insert("source_label".to_string(), "probe:h-prefix".to_string());
        let mut szz_metadata = TraceMetadata::new();
        szz_metadata.insert("source_gate".to_string(), "SZZ".to_string());
        szz_metadata.insert("source_label".to_string(), "probe:szz-host".to_string());

        let mut pending_global_metadata = TraceMetadata::new();
        let mut pending_qubit_metadata = BTreeMap::from([(1, szz_metadata), (8, h_metadata)]);

        let rxy_metadata = SeleneRuntime::take_pending_trace_metadata_for_source_op(
            &QuantumOp::RXY(std::f64::consts::FRAC_PI_2, 0.0, 8),
            &mut pending_global_metadata,
            &mut pending_qubit_metadata,
        )
        .expect("take metadata for RXY");
        assert_eq!(
            rxy_metadata.get("source_label").map(String::as_str),
            Some("probe:h-prefix")
        );
        assert!(pending_qubit_metadata.contains_key(&1));
        assert!(!pending_qubit_metadata.contains_key(&8));

        let rzz_metadata = SeleneRuntime::take_pending_trace_metadata_for_source_op(
            &QuantumOp::RZZ(-std::f64::consts::FRAC_PI_2, 9, 1),
            &mut pending_global_metadata,
            &mut pending_qubit_metadata,
        )
        .expect("take metadata for RZZ");
        assert_eq!(
            rzz_metadata.get("source_label").map(String::as_str),
            Some("probe:szz-host")
        );
        assert!(pending_qubit_metadata.is_empty());
    }

    #[test]
    fn test_conflicting_trace_metadata_fails_loudly() {
        let mut left_metadata = TraceMetadata::new();
        left_metadata.insert("source_label".to_string(), "probe:left".to_string());
        let mut right_metadata = TraceMetadata::new();
        right_metadata.insert("source_label".to_string(), "probe:right".to_string());

        let mut pending_global_metadata = TraceMetadata::new();
        let mut pending_qubit_metadata = BTreeMap::from([(0, left_metadata), (1, right_metadata)]);

        let error = SeleneRuntime::take_pending_trace_metadata_for_source_op(
            &QuantumOp::RZZ(std::f64::consts::FRAC_PI_2, 0, 1),
            &mut pending_global_metadata,
            &mut pending_qubit_metadata,
        )
        .expect_err("conflicting source labels should fail");
        assert!(error.to_string().contains("conflicting trace metadata"));
    }

    #[test]
    fn test_optional_unlowered_trace_metadata_is_allowed() {
        let mut metadata = TraceMetadata::new();
        metadata.insert(
            "source_label".to_string(),
            "probe:optimized-away-prefix".to_string(),
        );
        let source_metadata = VecDeque::from([SourceTraceMetadata {
            op: QuantumOp::RXY(std::f64::consts::FRAC_PI_2, 0.0, 4),
            metadata,
        }]);

        SeleneRuntime::fail_if_metadata_was_not_lowered(&source_metadata)
            .expect("optional metadata may be optimized away by the runtime");
    }

    #[test]
    fn test_required_unlowered_trace_metadata_fails_loudly() {
        let mut metadata = TraceMetadata::new();
        metadata.insert(
            "source_label".to_string(),
            "probe:required-host".to_string(),
        );
        metadata.insert("source_lowering_required".to_string(), "true".to_string());
        let source_metadata = VecDeque::from([SourceTraceMetadata {
            op: QuantumOp::RZZ(std::f64::consts::FRAC_PI_2, 0, 1),
            metadata,
        }]);

        let error = SeleneRuntime::fail_if_metadata_was_not_lowered(&source_metadata)
            .expect_err("required metadata should fail when it is not lowered");
        assert!(error.to_string().contains("required-host"));
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

        let InputCapacity {
            qubits: num_qubits, ..
        } = operation_capacity(
            &[QuantumOp::CX(81, 105).into()],
            BTreeSet::new(),
            BTreeSet::new(),
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

#[cfg(test)]
#[path = "scheduled_tests.rs"]
mod scheduled_tests;
