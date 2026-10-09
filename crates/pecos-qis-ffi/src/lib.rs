//! Minimal QIS Interface for Fast Linking
//!
//! Minimal FFI interface needed to link QIS (Quantum Instruction Set)
//! programs with Rust functions. It's designed to be lightweight and compile quickly.
//!
//! The interface collects quantum operations during program execution without performing
//! any simulation or complex state management. These operations are later processed by
//! a `QisRuntime` implementation.
//!
//! For dynamic circuits (conditionals depending on measurement results), a quantum executor
//! callback can be registered that will execute pending operations when a measurement result
//! is needed but not yet available.
//!
//! # Parallel Execution Support
//!
//! This crate supports parallel execution of multiple quantum programs (e.g., Monte Carlo
//! simulations) by using per-execution contexts. Each execution creates its own
//! `ExecutionContext` which isolates state between parallel executions.
//!
//! To use parallel execution:
//! 1. Create an `ExecutionContext` with `pecos_create_execution_context()`
//! 2. Register it on the worker thread with `pecos_register_execution_context()`
//! 3. Run the quantum program
//! 4. Unregister with `pecos_register_execution_context(null)`
//! 5. Destroy with `pecos_destroy_execution_context()`

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};

pub mod ffi;
mod random;
mod selene;

#[cfg(test)]
mod test_env;

#[cfg(test)]
mod cancellation_tests;

#[cfg(test)]
mod dynamic_read_tests;

#[cfg(test)]
mod named_results_tests;

#[cfg(test)]
mod random_tests;

// --- Per-Execution Context for Parallel Execution Support ---

/// Cancellation is terminal for an execution context.
#[derive(Debug, Default, PartialEq, Eq)]
pub enum CancellationState {
    #[default]
    Running,
    Requested,
}

/// State for dynamic circuit synchronization
#[derive(Debug, Default)]
pub struct DynamicSyncState {
    /// Cancellation belongs to this context and participates in the wait predicate.
    pub cancellation: CancellationState,
    /// Set to true when a measurement result is available
    pub result_ready: bool,
    /// Set to true when `___read_future_bool` needs a result
    pub need_result: bool,
    /// Set to true when the worker thread has completed
    pub worker_complete: bool,
}

/// Per-execution context for dynamic circuit coordination
///
/// This struct contains all the state needed for a single quantum program execution.
/// Each parallel execution (e.g., each Monte Carlo shot) should have its own context.
///
/// The context is thread-safe and can be shared between the main thread and worker thread
/// via Arc or raw pointers.
///
/// Nested execution wrappers and re-entrant FFI calls while a context mutex is
/// held are unsupported: each thread has one jump buffer and non-reentrant mutexes.
pub struct ExecutionContext {
    /// Cheap checkpoint mirror of the guarded cancellation state. Never reset.
    pub cancel_requested: AtomicBool,
    /// Flag indicating dynamic execution mode is active
    pub dynamic_mode_active: AtomicBool,
    /// The result ID that is being waited for
    pub waiting_for_result: AtomicU64,
    /// Mutex for signaling between worker and main thread
    pub sync_state: Mutex<DynamicSyncState>,
    /// Condvar for synchronization
    pub sync_condvar: Condvar,
    /// Storage for pending operations (shared between threads)
    pub pending_ops: Mutex<Vec<Operation>>,
    /// Storage for measurement outcomes (shared between threads).
    ///
    /// Ordinary measurements use 0/1. Leakage-aware measurements may also use 2.
    pub measurement_results: Mutex<Vec<Option<u64>>>,
    /// Result slots with a measurement queued or completed in this shot.
    measured_results: Mutex<BTreeSet<usize>>,
    /// Typed storage for named results from all `print_*` entry points.
    pub named_results: Mutex<BTreeMap<String, NamedResult>>,
    /// Runtime provenance for bool outputs and scalar integer 0/1 calls.
    pub named_result_traces: Mutex<Vec<NamedResultTrace>>,
    /// Result IDs read since the last named output consumed them.
    pub pending_result_reads: Mutex<Vec<usize>>,
    /// First termination or output error; reset before the next program starts.
    pub program_error: Mutex<Option<ProgramError>>,
    /// Program-seeded classical RNG; absent until seeded in the current shot.
    program_rng: Mutex<Option<pecos_random::PCGRandom>>,
    /// Live libc allocations owned by this program execution, keyed by address.
    pub program_allocations: Mutex<BTreeSet<usize>>,
    /// Number of allocations actually freed for this context.
    pub program_allocation_frees: AtomicUsize,
}

static LIVE_PROGRAM_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static TOTAL_PROGRAM_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

/// Number of live tracked program allocations across this FFI library.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_get_live_allocation_count() -> usize {
    LIVE_PROGRAM_ALLOCATIONS.load(Ordering::SeqCst)
}

/// Number of program allocations made across this FFI library.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_get_total_allocation_count() -> usize {
    TOTAL_PROGRAM_ALLOCATIONS.load(Ordering::SeqCst)
}

/// Number of allocations actually freed for the registered context.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_get_context_allocation_free_count() -> usize {
    get_execution_context().map_or(0, |ctx| {
        unsafe { &*ctx }
            .program_allocation_frees
            .load(Ordering::SeqCst)
    })
}

/// Free any outstanding program allocations. Called at both shot boundaries.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_cleanup_program_allocations() {
    if let Some(ctx) = get_execution_context() {
        unsafe { &*ctx }.release_program_allocations();
    }
}

impl ExecutionContext {
    /// Create a new execution context with default state
    #[must_use]
    pub fn new() -> Self {
        Self {
            cancel_requested: AtomicBool::new(false),
            dynamic_mode_active: AtomicBool::new(false),
            waiting_for_result: AtomicU64::new(u64::MAX),
            sync_state: Mutex::new(DynamicSyncState::default()),
            sync_condvar: Condvar::new(),
            pending_ops: Mutex::new(Vec::new()),
            measurement_results: Mutex::new(Vec::new()),
            measured_results: Mutex::new(BTreeSet::new()),
            named_results: Mutex::new(BTreeMap::new()),
            named_result_traces: Mutex::new(Vec::new()),
            pending_result_reads: Mutex::new(Vec::new()),
            program_error: Mutex::new(None),
            program_rng: Mutex::new(None),
            program_allocations: Mutex::new(BTreeSet::new()),
            program_allocation_frees: AtomicUsize::new(0),
        }
    }

    /// Reset the context to initial state (for reuse)
    pub fn reset(&self) {
        self.reset_outputs();
        self.dynamic_mode_active.store(false, Ordering::SeqCst);
        self.waiting_for_result.store(u64::MAX, Ordering::SeqCst);
        if let Ok(mut state) = self.sync_state.lock() {
            state.result_ready = false;
            state.need_result = false;
            state.worker_complete = false;
        }
        if let Ok(mut ops) = self.pending_ops.lock() {
            ops.clear();
        }
    }

    /// Reset per-shot measurement slots and output without changing dynamic synchronization.
    pub fn reset_outputs(&self) {
        self.clear_program_error();
        if let Ok(mut measured) = self.measured_results.lock() {
            measured.clear();
        }
        if let Ok(mut results) = self.measurement_results.lock() {
            results.clear();
        }
        self.reset_program_rng();
        self.release_program_allocations();
        if let Ok(mut named) = self.named_results.lock() {
            named.clear();
        }
        if let Ok(mut traces) = self.named_result_traces.lock() {
            traces.clear();
        }
        if let Ok(mut reads) = self.pending_result_reads.lock() {
            reads.clear();
        }
    }

    /// Free allocations skipped by program cleanup on normal return or transfer.
    pub fn release_program_allocations(&self) {
        let allocations = {
            let mut allocations = match self.program_allocations.lock() {
                Ok(allocations) => allocations,
                Err(poisoned) => {
                    self.record_program_error(ProgramError::InvalidInput {
                        entry: "pecos_cleanup_program_allocations".to_string(),
                        detail: "poisoned allocation set".to_string(),
                    });
                    // Ownership remains known even if a prior panic poisoned the lock.
                    poisoned.into_inner()
                }
            };
            std::mem::take(&mut *allocations)
        };
        for address in allocations {
            unsafe { self.free_program_allocation(std::ptr::with_exposed_provenance_mut(address)) };
        }
    }

    /// # Safety
    /// The allocation must have been removed from this context's live set.
    unsafe fn free_program_allocation(&self, ptr: *mut libc::c_void) {
        unsafe { libc::free(ptr) };
        self.program_allocation_frees.fetch_add(1, Ordering::SeqCst);
        LIVE_PROGRAM_ALLOCATIONS.fetch_sub(1, Ordering::SeqCst);
    }

    /// Record that program execution read a runtime measurement result.
    pub fn record_result_read(&self, result_id: usize) {
        if let Ok(mut reads) = self.pending_result_reads.lock() {
            reads.push(result_id);
        } else {
            log::error!("ExecutionContext::record_result_read failed to acquire lock");
        }
    }

    fn take_result_reads(&self, count: usize) -> Vec<usize> {
        if count == 0 {
            return Vec::new();
        }
        let Ok(mut reads) = self.pending_result_reads.lock() else {
            log::error!("ExecutionContext::take_result_reads failed to acquire lock");
            return Vec::new();
        };
        if reads.len() != count {
            log::warn!(
                "Named result output expected exactly {count} result read(s), but {} were recorded; refusing ambiguous provenance",
                reads.len()
            );
            reads.clear();
            return Vec::new();
        }
        reads.drain(..).collect()
    }

    fn store_named_result_trace(&self, name: &str, values: &[bool], result_ids: Vec<usize>) {
        if let Ok(mut traces) = self.named_result_traces.lock() {
            traces.push(NamedResultTrace {
                name: name.to_string(),
                values: values.to_vec(),
                result_ids,
            });
        } else {
            log::error!(
                "ExecutionContext::store_named_result_trace failed to acquire lock for '{name}'"
            );
        }
    }

    /// Clear the previous shot's error before entering the program.
    pub fn clear_program_error(&self) {
        if let Ok(mut error) = self.program_error.lock() {
            *error = None;
        }
    }

    /// Keep the first program error so subsequent output cannot hide it.
    pub fn record_program_error(&self, error: ProgramError) {
        if let Ok(mut stored) = self.program_error.lock() {
            stored.get_or_insert(error);
        }
    }

    /// Append one call's values, consuming reads only for nonempty bool calls.
    /// `is_scalar` describes the entry point, not the number of elements.
    pub fn store_named_result(&self, name: &str, values: NamedResult, is_scalar: bool) {
        let Ok(error) = self.program_error.lock() else {
            return;
        };
        if error.is_some() {
            return;
        }
        drop(error);
        // DEM detector convention: scalar integer 0/1 calls retain bool traces
        // in the single ideal tracing run, but never claim measurement provenance.
        // Integer arrays never trace, including one-element arrays; the old
        // integer detector convention had only scalar calls. Static certification
        // recognizes only real bool outputs.
        // Only real bool calls drain reads; an empty bool array has no reads to
        // drain and keeps an empty trace. Storage follows declared call types.
        let is_bool_call = matches!(&values, NamedResult::Bool(_));
        let bool_values = match &values {
            NamedResult::Bool(values) => Some(values.clone()),
            _ if values.is_empty() => None,
            NamedResult::I64(values) if is_scalar && matches!(values.as_slice(), [0 | 1]) => {
                Some(values.iter().map(|&value| value == 1).collect())
            }
            NamedResult::U64(values) if is_scalar && matches!(values.as_slice(), [0 | 1]) => {
                Some(values.iter().map(|&value| value == 1).collect())
            }
            _ => None,
        };
        let stored = match self.named_results.lock() {
            Ok(mut named) => match named.entry(name.to_string()) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(values);
                    Ok(())
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    entry.get_mut().append(values).map_err(|(existing, incoming)| {
                        format!("Named result '{name}' has element type {existing}, but received {incoming}")
                    })
                }
            },
            Err(error) => Err(format!("Failed to store named result '{name}': {error}")),
        };
        if let Err(message) = stored {
            self.record_program_error(ProgramError::NamedResult(message));
            return;
        }
        if let Some(values) = bool_values {
            let result_ids = if is_bool_call {
                self.take_result_reads(values.len())
            } else {
                Vec::new()
            };
            self.store_named_result_trace(name, &values, result_ids);
        }
    }

    /// Store a named result (single bool value).
    pub fn store_named_bool(&self, name: &str, value: bool) {
        self.store_named_result(name, NamedResult::Bool(vec![value]), true);
    }

    /// Store a named result array (multiple bool values).
    pub fn store_named_array(&self, name: &str, values: &[bool]) {
        self.store_named_result(name, NamedResult::Bool(values.to_vec()), false);
    }

    /// Get all named results (returns a clone)
    #[must_use]
    pub fn get_named_results(&self) -> BTreeMap<String, NamedResult> {
        self.named_results
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default()
    }

    /// Get all named result provenance records (returns a clone)
    #[must_use]
    pub fn get_named_result_traces(&self) -> Vec<NamedResultTrace> {
        self.named_result_traces
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default()
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

// Thread-local storage for the current execution context
thread_local! {
    /// Thread-local storage for the per-execution context
    /// This is set by the worker thread before calling qmain
    static EXECUTION_CONTEXT: RefCell<Option<*mut ExecutionContext>> = const { RefCell::new(None) };
}

/// Register an execution context for the current thread
///
/// This should be called on the worker thread before starting execution.
/// Pass null to unregister the context.
///
/// # Safety
/// The pointer must be valid for the duration of execution, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_register_execution_context(ctx: *mut ExecutionContext) {
    log::debug!("pecos_register_execution_context called: ctx={ctx:?}");
    EXECUTION_CONTEXT.with(|ec| {
        *ec.borrow_mut() = if ctx.is_null() { None } else { Some(ctx) };
    });
}

/// Create a new execution context
///
/// Returns a pointer to a newly allocated `ExecutionContext`.
/// The caller is responsible for freeing this via `pecos_destroy_execution_context`.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_create_execution_context() -> *mut ExecutionContext {
    log::debug!("pecos_create_execution_context called");
    Box::into_raw(Box::new(ExecutionContext::new()))
}

/// Clear a previous program error at shot start, without resetting dynamic synchronization.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_clear_program_error() {
    if let Some(ctx) = get_execution_context() {
        unsafe { &*ctx }.clear_program_error();
    }
}

/// Return the recorded program error as JSON, or null if the shot has no error.
/// Free the returned string with `pecos_free_named_results_json`.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_get_program_error_json() -> *mut std::ffi::c_char {
    let Some(ctx) = get_execution_context() else {
        return std::ptr::null_mut();
    };
    let ctx = unsafe { &*ctx };
    let error = match ctx.program_error.lock() {
        Ok(error) => error.clone(),
        Err(error) => Some(ProgramError::NamedResult(format!(
            "Failed to read program error: {error}"
        ))),
    };
    let Some(error) = error else {
        return std::ptr::null_mut();
    };
    let Ok(json) = serde_json::to_string(&error) else {
        return std::ptr::null_mut();
    };
    match std::ffi::CString::new(json) {
        Ok(json) => json.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Whether the current program terminated with a normal exit.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_program_exited() -> bool {
    let Some(ctx) = get_execution_context() else {
        return false;
    };
    unsafe { &*ctx }
        .program_error
        .lock()
        .is_ok_and(|error| matches!(*error, Some(ProgramError::Exit { .. })))
}

/// Destroy an execution context
///
/// # Safety
/// The pointer must have been created by `pecos_create_execution_context` and
/// have no remaining users. Registrations on other threads must already have
/// been cleared or replaced. Only a matching registration on this thread is
/// cleared; neither another thread's TLS nor a replacement context is touched.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_destroy_execution_context(ctx: *mut ExecutionContext) {
    // Drop may run during TLS teardown. Do not initialize TLS or log here.
    let _ = EXECUTION_CONTEXT.try_with(|ec| {
        if *ec.borrow() == Some(ctx) {
            *ec.borrow_mut() = None;
        }
    });
    if !ctx.is_null() {
        // SAFETY: ptr was allocated by Box::into_raw in pecos_create_execution_context
        unsafe { &*ctx }.release_program_allocations();
        drop(unsafe { Box::from_raw(ctx) });
    }
}

/// Get the current execution context for this thread
///
/// Returns the registered context if available, otherwise returns None.
fn get_execution_context() -> Option<*mut ExecutionContext> {
    EXECUTION_CONTEXT.with(|ec| *ec.borrow())
}

// Re-export all types from pecos-qis-ffi-types
pub use pecos_qis_ffi_types::{
    NamedResult, NamedResultTrace, Operation, OperationCollector, OperationList, ProgramError,
    QuantumOp, TraceMetadata,
};

/// Type alias for the quantum executor callback
///
/// This callback is called when `___read_future_bool` needs a measurement result
/// that hasn't been computed yet. The callback should:
/// 1. Take the pending operations from the collector
/// 2. Execute them on a quantum simulator
/// 3. Return the measurement results as a map of `result_id` -> value
///
/// The callback receives:
/// - A mutable reference to the operation collector
/// - Returns a map of measurement results
pub type QuantumExecutorCallback =
    Box<dyn Fn(&mut OperationCollector) -> BTreeMap<usize, bool> + Send>;

thread_local! {
    /// Thread-local storage for the current operation collector
    static INTERFACE: RefCell<OperationCollector> = RefCell::new(OperationCollector::new());

    /// Thread-local storage for the quantum executor callback
    /// This is called when a measurement result is needed but not available
    static EXECUTOR: RefCell<Option<QuantumExecutorCallback>> = const { RefCell::new(None) };
}

/// Get the thread-local operation collector
pub fn with_interface<F, R>(f: F) -> R
where
    F: FnOnce(&mut OperationCollector) -> R,
{
    INTERFACE.with(|interface| f(&mut interface.borrow_mut()))
}

/// Reset the thread-local operation collector
pub fn reset_interface() {
    if let Some(ctx) = get_execution_context() {
        unsafe { &*ctx }.reset_outputs();
    }
    with_interface(OperationCollector::reset);
    // Also reset the collection mode read counter for loop termination
    ffi::reset_collection_read_count();
}

/// Get a clone of the thread-local operation collector
#[must_use]
pub fn get_interface_clone() -> OperationCollector {
    with_interface(|interface| interface.clone())
}

/// Take the thread-local operation collector, leaving an empty collector behind.
#[must_use]
pub fn take_interface() -> OperationCollector {
    with_interface(std::mem::take)
}

/// Set measurement results in the thread-local operation collector
pub fn set_measurements(measurements: impl IntoIterator<Item = (usize, bool)>) {
    with_interface(|interface| interface.set_measurement_results(measurements));
}

/// Set the quantum executor callback for dynamic circuit execution
///
/// This callback is called when `___read_future_bool` needs a measurement result
/// that hasn't been simulated yet. The callback should execute pending quantum
/// operations and return measurement results.
///
/// # Example
/// ```
/// use pecos_qis_ffi::set_quantum_executor;
/// use std::collections::{BTreeMap, BTreeSet};
///
/// set_quantum_executor(|collector| {
///     let _ops = collector.take_operations();
///     BTreeMap::new() // return measurement results
/// });
/// ```
pub fn set_quantum_executor<F>(executor: F)
where
    F: Fn(&mut OperationCollector) -> BTreeMap<usize, bool> + Send + 'static,
{
    log::debug!("set_quantum_executor called");
    EXECUTOR.with(|e| *e.borrow_mut() = Some(Box::new(executor)));
}

/// Clear the quantum executor callback
pub fn clear_quantum_executor() {
    EXECUTOR.with(|e| *e.borrow_mut() = None);
}

/// Execute pending operations and get measurement results
///
/// This is called by `___read_future_bool` when a result is needed but not available.
/// Returns true if execution happened (and results were stored), false if no executor is set.
#[must_use]
pub fn execute_pending_and_get_results() -> bool {
    log::debug!("execute_pending_and_get_results called");
    EXECUTOR.with(|executor| {
        let executor_ref = executor.borrow();
        if let Some(exec) = executor_ref.as_ref() {
            // Execute pending operations
            let results = INTERFACE.with(|interface| exec(&mut interface.borrow_mut()));

            // Store the results
            INTERFACE.with(|interface| {
                let mut iface = interface.borrow_mut();
                for (result_id, value) in results {
                    iface.store_result(result_id, value);
                }
            });
            true
        } else {
            log::debug!("No executor set");
            false
        }
    })
}

// --- FFI functions for cross-library dynamic circuit coordination ---

/// Enable dynamic mode on an explicit context.
///
/// Returns 0 on success, 1 for a null context, 2 for a poisoned lock,
/// and 3 for invalid data or serialization failure.
///
/// # Safety
/// A non-null context must remain live throughout the call. Any output pointer
/// must be valid and writable. No thread-local registration is consulted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_enable_dynamic_mode_with_context(ctx: *mut ExecutionContext) -> i32 {
    // SAFETY: The caller keeps the context alive.
    let Some(ctx) = (unsafe { ctx.as_ref() }) else {
        return 1;
    };
    let Ok(mut state) = ctx.sync_state.lock() else {
        return 2;
    };
    let Ok(mut results) = ctx.measurement_results.lock() else {
        return 2;
    };
    let Ok(mut ops) = ctx.pending_ops.lock() else {
        return 2;
    };
    state.result_ready = false;
    state.need_result = false;
    state.worker_complete = false;
    results.clear();
    ops.clear();
    ctx.dynamic_mode_active.store(true, Ordering::SeqCst);
    0
}

/// Disable dynamic mode on an explicit context.
///
/// Returns 0 on success, 1 for a null context, 2 for a poisoned lock,
/// and 3 for invalid data or serialization failure.
///
/// # Safety
/// A non-null context must remain live throughout the call. Any output pointer
/// must be valid and writable. No thread-local registration is consulted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_disable_dynamic_mode_with_context(
    ctx: *mut ExecutionContext,
) -> i32 {
    // SAFETY: The caller keeps the context alive.
    let Some(ctx) = (unsafe { ctx.as_ref() }) else {
        return 1;
    };
    ctx.dynamic_mode_active.store(false, Ordering::SeqCst);
    let (mut state, status) = match ctx.sync_state.lock() {
        Ok(state) => (state, 0),
        Err(poisoned) => (poisoned.into_inner(), 2),
    };
    state.worker_complete = true;
    drop(state);
    ctx.sync_condvar.notify_all();
    status
}

/// Signal result ready on an explicit context.
///
/// Returns 0 on success, 1 for a null context, 2 for a poisoned lock,
/// and 3 for invalid data or serialization failure.
///
/// # Safety
/// A non-null context must remain live throughout the call. Any output pointer
/// must be valid and writable. No thread-local registration is consulted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_signal_result_ready_with_context(ctx: *mut ExecutionContext) -> i32 {
    // SAFETY: The caller keeps the context alive.
    let Some(ctx) = (unsafe { ctx.as_ref() }) else {
        return 1;
    };
    let Ok(mut state) = ctx.sync_state.lock() else {
        return 2;
    };
    state.result_ready = true;
    state.need_result = false;
    drop(state);
    ctx.sync_condvar.notify_all();
    0
}

/// Set measurement result on an explicit context.
///
/// Returns 0 on success, 1 for a null context, 2 for a poisoned lock,
/// and 3 for invalid data or serialization failure.
///
/// # Safety
/// A non-null context must remain live throughout the call. Any output pointer
/// must be valid and writable. No thread-local registration is consulted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_set_measurement_result_with_context(
    ctx: *mut ExecutionContext,
    result_id: u64,
    value: bool,
) -> i32 {
    // SAFETY: The caller keeps the context alive.
    unsafe { pecos_set_measurement_outcome_with_context(ctx, result_id, u64::from(value)) }
}

/// Set measurement outcome on an explicit context.
///
/// Returns 0 on success, 1 for a null context, 2 for a poisoned lock,
/// and 3 for invalid data or serialization failure.
///
/// # Safety
/// A non-null context must remain live throughout the call. Any output pointer
/// must be valid and writable. No thread-local registration is consulted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_set_measurement_outcome_with_context(
    ctx: *mut ExecutionContext,
    result_id: u64,
    value: u64,
) -> i32 {
    // SAFETY: The caller keeps the context alive.
    let Some(ctx) = (unsafe { ctx.as_ref() }) else {
        return 1;
    };
    let Ok(index) = usize::try_from(result_id) else {
        return 3;
    };
    let Some(len) = index.checked_add(1) else {
        return 3;
    };
    if value > 2 {
        return 3;
    }
    let Ok(mut results) = ctx.measurement_results.lock() else {
        return 2;
    };
    if results.len() < len {
        results.resize(len, None);
    }
    results[index] = Some(value);
    0
}

/// Wait for need result on an explicit context.
///
/// Returns 0 on success, 1 for a null context, 2 for a poisoned lock,
/// and 3 for invalid data or serialization failure. Output is `u64::MAX` on timeout or completion.
///
/// # Safety
/// A non-null context must remain live throughout the call. Any output pointer
/// must be valid and writable. No thread-local registration is consulted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_wait_for_need_result_with_context(
    ctx: *mut ExecutionContext,
    timeout_ms: u64,
    output: *mut u64,
) -> i32 {
    // SAFETY: The caller provides writable output storage.
    unsafe {
        *output = u64::MAX;
    }
    // SAFETY: The caller keeps the context alive.
    let Some(ctx) = (unsafe { ctx.as_ref() }) else {
        return 1;
    };
    let Ok(mut state) = ctx.sync_state.lock() else {
        return 2;
    };
    while !state.need_result && !state.worker_complete {
        match ctx
            .sync_condvar
            .wait_timeout(state, std::time::Duration::from_millis(timeout_ms))
        {
            Ok((next, timeout)) => {
                state = next;
                if timeout.timed_out() {
                    return 0;
                }
            }
            Err(_) => return 2,
        }
    }
    if !state.worker_complete && state.need_result {
        // SAFETY: The caller provides writable output storage.
        unsafe {
            *output = ctx.waiting_for_result.load(Ordering::SeqCst);
        }
    }
    0
}

/// Get pending operations on an explicit context.
///
/// Returns 0 on success, 1 for a null context, 2 for a poisoned lock,
/// and 3 for invalid data or serialization failure. An empty queue returns an allocated empty collector.
/// Free the output with `pecos_free_operations`.
///
/// # Safety
/// A non-null context must remain live throughout the call. Any output pointer
/// must be valid and writable. No thread-local registration is consulted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_get_pending_operations_with_context(
    ctx: *mut ExecutionContext,
    output: *mut *mut OperationCollector,
) -> i32 {
    // SAFETY: The caller provides writable output storage.
    unsafe {
        *output = std::ptr::null_mut();
    }
    // SAFETY: The caller keeps the context alive.
    let Some(ctx) = (unsafe { ctx.as_ref() }) else {
        return 1;
    };
    let Ok(mut pending) = ctx.pending_ops.lock() else {
        return 2;
    };
    let mut collector = OperationCollector::new();
    collector.operations = std::mem::take(&mut *pending);
    // SAFETY: The caller provides writable output storage.
    unsafe {
        *output = Box::into_raw(Box::new(collector));
    }
    0
}

/// Get named results json on an explicit context.
///
/// Returns 0 on success, 1 for a null context, 2 for a poisoned lock,
/// and 3 for invalid data or serialization failure. Empty results return null.
/// Free the output with `pecos_free_named_results_json`.
///
/// # Safety
/// A non-null context must remain live throughout the call. Any output pointer
/// must be valid and writable. No thread-local registration is consulted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_get_named_results_json_with_context(
    ctx: *mut ExecutionContext,
    output: *mut *mut std::ffi::c_char,
) -> i32 {
    // SAFETY: The caller provides writable output storage.
    unsafe {
        *output = std::ptr::null_mut();
    }
    // SAFETY: The caller keeps the context alive.
    let Some(ctx) = (unsafe { ctx.as_ref() }) else {
        return 1;
    };
    let Ok(values) = ctx.named_results.lock() else {
        return 2;
    };
    if values.is_empty() {
        return 0;
    }
    let Ok(json) = serde_json::to_string(&*values) else {
        return 3;
    };
    let Ok(json) = std::ffi::CString::new(json) else {
        return 3;
    };
    // SAFETY: The caller provides writable output storage.
    unsafe {
        *output = json.into_raw();
    }
    0
}

/// Get named result traces json on an explicit context.
///
/// Returns 0 on success, 1 for a null context, 2 for a poisoned lock,
/// and 3 for invalid data or serialization failure. Empty results return null.
/// Free the output with `pecos_free_named_results_json`.
///
/// # Safety
/// A non-null context must remain live throughout the call. Any output pointer
/// must be valid and writable. No thread-local registration is consulted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_get_named_result_traces_json_with_context(
    ctx: *mut ExecutionContext,
    output: *mut *mut std::ffi::c_char,
) -> i32 {
    // SAFETY: The caller provides writable output storage.
    unsafe {
        *output = std::ptr::null_mut();
    }
    // SAFETY: The caller keeps the context alive.
    let Some(ctx) = (unsafe { ctx.as_ref() }) else {
        return 1;
    };
    let Ok(values) = ctx.named_result_traces.lock() else {
        return 2;
    };
    if values.is_empty() {
        return 0;
    }
    let Ok(json) = serde_json::to_string(&*values) else {
        return 3;
    };
    let Ok(json) = std::ffi::CString::new(json) else {
        return 3;
    };
    // SAFETY: The caller provides writable output storage.
    unsafe {
        *output = json.into_raw();
    }
    0
}

/// Enable dynamic execution mode (called via FFI from executor)
///
/// Requires a per-execution context to be registered via `pecos_register_execution_context`.
/// If no context is registered, this is a no-op.
///
/// # Safety
/// This function is safe to call from any thread.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_enable_dynamic_mode() {
    let ctx = get_execution_context().unwrap_or(std::ptr::null_mut());
    // SAFETY: The registered context is live for this call.
    unsafe { pecos_enable_dynamic_mode_with_context(ctx) };
}

/// Request cooperative cancellation on an explicit execution context.
///
/// Returns 0 on success, 1 for a null context, and 2 when the synchronization
/// state is poisoned. A poisoned state is still cancelled and its waiters woken,
/// so a reader with no timeout can always be reached; the error only reports it.
///
/// # Safety
/// A non-null context must remain live for this call. It need not be registered
/// on the calling thread. Cancellation lasts for the entire context lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_abort_dynamic_execution(ctx: *mut ExecutionContext) -> i32 {
    // SAFETY: The caller keeps a non-null context alive for this call.
    let Some(ctx) = (unsafe { ctx.as_ref() }) else {
        return 1;
    };
    let (mut state, status) = match ctx.sync_state.lock() {
        Ok(state) => (state, 0),
        Err(poisoned) => (poisoned.into_inner(), 2),
    };
    state.cancellation = CancellationState::Requested;
    ctx.cancel_requested.store(true, Ordering::Release);
    drop(state);
    ctx.sync_condvar.notify_all();
    status
}

/// Disable dynamic execution mode (called via FFI from executor)
///
/// This also signals completion so the main thread wakes up.
///
/// Requires a per-execution context to be registered via `pecos_register_execution_context`.
/// If no context is registered, this is a no-op.
///
/// # Safety
/// This function is safe to call from any thread.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_disable_dynamic_mode() {
    let ctx = get_execution_context().unwrap_or(std::ptr::null_mut());
    // SAFETY: The registered context is live for this call.
    unsafe { pecos_disable_dynamic_mode_with_context(ctx) };
}

/// Check if a result is needed (called by main thread to check if worker is waiting)
///
/// Returns the result ID being waited for, or `u64::MAX` if no result is needed
/// or no execution context is registered.
///
/// # Safety
/// This function is safe to call from any thread.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_check_need_result() -> u64 {
    if let Some(ctx) = get_execution_context() {
        // SAFETY: Context is valid for duration of execution
        let ctx = unsafe { &*ctx };
        if let Ok(state) = ctx.sync_state.lock()
            && state.need_result
        {
            return ctx.waiting_for_result.load(Ordering::SeqCst);
        }
    }
    u64::MAX
}

/// Wait for a result to be needed or worker to complete (called by main thread)
///
/// Blocks until the worker thread needs a measurement result OR completes.
/// Returns the result ID that is needed, or `u64::MAX` if worker completed,
/// timeout, or no execution context is registered.
///
/// # Safety
/// This function is safe to call from any thread.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_wait_for_need_result(timeout_ms: u64) -> u64 {
    let ctx = get_execution_context().unwrap_or(std::ptr::null_mut());
    let mut result = u64::MAX;
    // SAFETY: The registered context is live for this call.
    unsafe { pecos_wait_for_need_result_with_context(ctx, timeout_ms, &raw mut result) };
    result
}

/// Check if worker has completed
///
/// Returns false if no execution context is registered.
///
/// # Safety
/// This function is safe to call from any thread.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_is_worker_complete() -> bool {
    if let Some(ctx) = get_execution_context() {
        // SAFETY: Context is valid for duration of execution
        let ctx = unsafe { &*ctx };
        if let Ok(state) = ctx.sync_state.lock() {
            return state.worker_complete;
        }
    }
    false
}

/// Signal that a measurement result is ready (called by main thread after simulation)
///
/// If no execution context is registered, this is a no-op and logs a warning.
///
/// # Safety
/// This function is safe to call from any thread.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_signal_result_ready() {
    let ctx = get_execution_context().unwrap_or(std::ptr::null_mut());
    // SAFETY: The registered context is live for this call.
    unsafe { pecos_signal_result_ready_with_context(ctx) };
}

/// Why the worker's unbounded measurement wait ended.
#[derive(Debug, PartialEq, Eq)]
pub enum ResultWaitOutcome {
    Ready,
    Cancelled,
    WorkerComplete,
    Failed(&'static str),
}

/// Wait for host readiness, cancellation, completion, or a synchronization failure.
/// There is no deadline: slow simulation does not invalidate a measurement.
#[must_use]
pub fn wait_for_result_ready(result_id: u64) -> ResultWaitOutcome {
    let Some(ctx) = get_execution_context() else {
        return ResultWaitOutcome::Failed("no execution context");
    };
    // SAFETY: The registered context is live throughout execution.
    let ctx = unsafe { &*ctx };
    let Ok(mut state) = ctx.sync_state.lock() else {
        return ResultWaitOutcome::Failed("poisoned synchronization state");
    };
    if state.cancellation == CancellationState::Requested {
        return ResultWaitOutcome::Cancelled;
    }
    if state.worker_complete {
        return ResultWaitOutcome::WorkerComplete;
    }

    // Export must succeed before publishing a request the host could act on.
    let exported = INTERFACE.with(|interface| {
        let mut iface = interface.borrow_mut();
        let Ok(mut pending) = ctx.pending_ops.lock() else {
            return false;
        };
        if pending.is_empty() {
            std::mem::swap(&mut *pending, &mut iface.operations);
        } else {
            pending.append(&mut iface.operations);
        }
        true
    });
    if !exported {
        return ResultWaitOutcome::Failed("poisoned pending operations");
    }
    ctx.waiting_for_result.store(result_id, Ordering::SeqCst);
    state.need_result = true;
    state.result_ready = false;
    ctx.sync_condvar.notify_all();

    let Ok(state) = ctx.sync_condvar.wait_while(state, |state| {
        !state.result_ready
            && !state.worker_complete
            && state.cancellation == CancellationState::Running
    }) else {
        return ResultWaitOutcome::Failed("poisoned synchronization state");
    };
    // Cancellation wins even if an outcome was published at the same time.
    if state.cancellation == CancellationState::Requested {
        ResultWaitOutcome::Cancelled
    } else if state.worker_complete {
        ResultWaitOutcome::WorkerComplete
    } else {
        ResultWaitOutcome::Ready
    }
}

/// Queue a result-producing operation and invalidate the worker-owned slot.
/// All measurement entry points use this boundary, including Selene wrappers.
fn queue_measurement(qubit: usize, result_id: usize, leaked: bool) {
    if is_dynamic_mode_active()
        && let Some(ctx) = get_execution_context()
    {
        // SAFETY: The registered context is live throughout execution.
        let ctx = unsafe { &*ctx };
        if let Ok(mut measured) = ctx.measured_results.lock() {
            measured.insert(result_id);
        }
        if let Ok(mut results) = ctx.measurement_results.lock()
            && let Some(result) = results.get_mut(result_id)
        {
            *result = None;
        }
    }
    with_interface(|iface| {
        let op = if leaked {
            QuantumOp::MeasureLeaked(qubit, result_id)
        } else {
            QuantumOp::Measure(qubit, result_id)
        };
        iface.queue_operation(op.into());
    });
}

/// Typed lookup for the dynamic path; absence and lock failure are distinct.
fn lookup_measurement_outcome(result_id: u64) -> Result<Option<u64>, &'static str> {
    let ctx = get_execution_context().ok_or("no execution context")?;
    let index = usize::try_from(result_id).map_err(|_| "result ID does not fit in usize")?;
    // SAFETY: The registered context is live throughout execution.
    let ctx = unsafe { &*ctx };
    let measured = ctx
        .measured_results
        .lock()
        .map_err(|_| "poisoned measured result slots")?;
    if !measured.contains(&index) {
        return Err("never measured");
    }
    drop(measured);
    let results = ctx
        .measurement_results
        .lock()
        .map_err(|_| "poisoned measurement outcomes")?;
    Ok(results.get(index).copied().flatten())
}

/// Read once through the dynamic protocol, recording any owned error before
/// returning to a destructor-free FFI frame that can transfer to the guard.
fn read_dynamic_result(result_id: u64, boolean: bool) -> Option<u64> {
    let read = || -> Result<Option<u64>, &'static str> {
        let mut outcome = lookup_measurement_outcome(result_id)?;
        if outcome.is_none() {
            match wait_for_result_ready(result_id) {
                ResultWaitOutcome::Ready => {
                    outcome = lookup_measurement_outcome(result_id)?;
                    if outcome.is_none() {
                        return Err("ready without a stored outcome");
                    }
                }
                ResultWaitOutcome::Cancelled => return Ok(None),
                ResultWaitOutcome::WorkerComplete => return Err("worker complete before outcome"),
                ResultWaitOutcome::Failed(reason) => return Err(reason),
            }
        }
        if boolean && outcome.is_some_and(|value| value > 1) {
            return Err("outcome is not representable as bool");
        }
        Ok(outcome)
    };
    match read() {
        Ok(outcome) => outcome,
        Err(reason) => {
            if let Some(ctx) = get_execution_context() {
                // SAFETY: The registered context is live throughout execution.
                let ctx = unsafe { &*ctx };
                let error = if ctx.cancel_requested.load(Ordering::Acquire) {
                    ProgramError::Cancelled
                } else {
                    ProgramError::ResultUnavailable {
                        result_id,
                        reason: reason.to_owned(),
                    }
                };
                ctx.record_program_error(error);
            }
            None
        }
    }
}

/// Check if dynamic mode is active
///
/// Returns false if no execution context is registered.
#[must_use]
pub fn is_dynamic_mode_active() -> bool {
    if let Some(ctx) = get_execution_context() {
        // SAFETY: Context is valid for duration of execution
        let ctx = unsafe { &*ctx };
        ctx.dynamic_mode_active.load(Ordering::SeqCst)
    } else {
        false
    }
}

/// Get a measurement result from the execution context (for cross-thread access)
///
/// This is used by the worker thread to get results set by the main thread.
/// Returns None if no execution context is registered.
#[must_use]
pub fn get_measurement_outcome(result_id: u64) -> Option<u64> {
    let ctx = get_execution_context()?;
    let result_index = usize::try_from(result_id).ok()?;
    // SAFETY: Context is valid for duration of execution
    let ctx = unsafe { &*ctx };
    if let Ok(results) = ctx.measurement_results.lock() {
        let value = results.get(result_index).copied().flatten();
        log::debug!("get_measurement_result: result_id={result_id}, value={value:?}");
        value
    } else {
        None
    }
}

/// Get a Boolean measurement result from the execution context.
///
/// Returns `None` for a leakage outcome instead of silently treating 2 as true.
#[must_use]
pub fn get_measurement_result(result_id: u64) -> Option<bool> {
    match get_measurement_outcome(result_id)? {
        0 => Some(false),
        1 => Some(true),
        value => {
            log::error!(
                "get_measurement_result: result_id={result_id} has non-Boolean outcome {value}"
            );
            None
        }
    }
}

/// Set a measurement result via FFI (called by main thread after simulation)
///
/// This stores in the execution context so worker thread can access it.
/// If no execution context is registered, this is a no-op.
///
/// # Safety
/// This function is safe to call from any thread.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_set_measurement_result(result_id: u64, value: bool) {
    let ctx = get_execution_context().unwrap_or(std::ptr::null_mut());
    // SAFETY: The registered context is live for this call.
    unsafe { pecos_set_measurement_result_with_context(ctx, result_id, value) };
}

/// Set an integer-valued measurement outcome via FFI.
///
/// Ordinary measurement results are 0/1; leakage-aware results may also be 2.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_set_measurement_outcome(result_id: u64, value: u64) {
    let ctx = get_execution_context().unwrap_or(std::ptr::null_mut());
    // SAFETY: The registered context is live for this call.
    unsafe { pecos_set_measurement_outcome_with_context(ctx, result_id, value) };
}

/// Clear pending operations in the thread-local collector
///
/// # Safety
/// This function is safe to call from any thread.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_clear_pending_operations() {
    INTERFACE.with(|interface| {
        interface.borrow_mut().operations.clear();
    });
}

/// Get pending operations from execution context (for cross-thread access)
///
/// Returns a pointer to a newly allocated `OperationCollector` with the pending operations.
/// The caller is responsible for freeing this via `pecos_free_operations`.
/// Returns an allocated empty collector when no operations are pending.
/// Returns null when no context is registered or the pending-operations lock is poisoned.
///
/// # Safety
/// This function is safe to call from any thread. The returned pointer must be freed.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_get_pending_operations() -> *mut OperationCollector {
    let ctx = get_execution_context().unwrap_or(std::ptr::null_mut());
    let mut output = std::ptr::null_mut();
    // SAFETY: The registered context is live for this call.
    unsafe { pecos_get_pending_operations_with_context(ctx, &raw mut output) };
    output
}

/// Free an `OperationCollector` allocated by `pecos_get_pending_operations`
///
/// # Safety
/// The pointer must have been allocated by `pecos_get_pending_operations`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_free_operations(ptr: *mut OperationCollector) {
    if !ptr.is_null() {
        // SAFETY: ptr was allocated by Box::into_raw in pecos_get_pending_operations
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Get named results from execution context as JSON
///
/// Returns a pointer to a heap-allocated null-terminated JSON string containing
/// the named results. Format: `{"name1": [true, false, ...], "name2": [...], ...}`
///
/// The caller must free the returned string using `pecos_free_named_results_json`.
/// Returns null if no context is registered or results are empty.
///
/// # Safety
/// This function is safe to call from any thread. The returned pointer must be freed.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_get_named_results_json() -> *mut std::ffi::c_char {
    let ctx = get_execution_context().unwrap_or(std::ptr::null_mut());
    let mut output = std::ptr::null_mut();
    // SAFETY: The registered context is live for this call.
    unsafe { pecos_get_named_results_json_with_context(ctx, &raw mut output) };
    output
}

/// Get named result runtime provenance from execution context as JSON.
///
/// Returns a pointer to a heap-allocated null-terminated JSON string containing
/// records of `result(...)` calls with the measurement result IDs used to
/// produce each output value.
///
/// The caller must free the returned string using `pecos_free_named_results_json`.
/// Returns null if no context is registered or traces are empty.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_get_named_result_traces_json() -> *mut std::ffi::c_char {
    let ctx = get_execution_context().unwrap_or(std::ptr::null_mut());
    let mut output = std::ptr::null_mut();
    // SAFETY: The registered context is live for this call.
    unsafe { pecos_get_named_result_traces_json_with_context(ctx, &raw mut output) };
    output
}

/// Free a JSON string returned by the named-result or program-error exports.
///
/// # Safety
/// The pointer must have been allocated by `pecos_get_named_results_json`,
/// `pecos_get_named_result_traces_json`, or `pecos_get_program_error_json`, and
/// must not have been freed already.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_free_named_results_json(ptr: *mut std::ffi::c_char) {
    if !ptr.is_null() {
        // SAFETY: All three JSON exports allocate with CString::into_raw.
        drop(unsafe { std::ffi::CString::from_raw(ptr) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;

    const TEST_SYNC_TIMEOUT_MS: u64 = 5_000;

    /// Helper to create and register an execution context for tests
    fn setup_context() -> *mut ExecutionContext {
        let ctx = pecos_create_execution_context();
        unsafe { pecos_register_execution_context(ctx) };
        ctx
    }

    /// Helper to unregister and destroy a context
    fn teardown_context(ctx: *mut ExecutionContext) {
        unsafe {
            pecos_register_execution_context(std::ptr::null_mut());
            pecos_destroy_execution_context(ctx);
        }
    }

    #[test]
    fn explicit_exports_ignore_tls_and_preserve_statuses() {
        let registered = setup_context();
        let mut owned = ExecutionContext::new();
        let ctx = &raw mut owned;
        let mut result = u64::MAX;
        let mut ops = std::ptr::null_mut();
        let mut json = std::ptr::null_mut();
        // All output pointers are writable and both contexts remain live.
        unsafe {
            assert_eq!(pecos_enable_dynamic_mode_with_context(ctx), 0);
            assert!(!(*registered).dynamic_mode_active.load(Ordering::SeqCst));
            assert_eq!(pecos_set_measurement_result_with_context(ctx, 0, true), 0);
            assert_eq!(pecos_set_measurement_outcome_with_context(ctx, 1, 2), 0);
            assert_eq!(
                *owned.measurement_results.lock().unwrap(),
                [Some(1), Some(2)]
            );
            assert!((*registered).measurement_results.lock().unwrap().is_empty());
            owned.waiting_for_result.store(42, Ordering::SeqCst);
            owned.sync_state.lock().unwrap().need_result = true;
            assert_eq!(
                pecos_wait_for_need_result_with_context(ctx, 0, &raw mut result),
                0
            );
            assert_eq!(result, 42);
            assert_eq!(pecos_signal_result_ready_with_context(ctx), 0);
            assert!(owned.sync_state.lock().unwrap().result_ready);
            assert!(!(*registered).sync_state.lock().unwrap().result_ready);
            owned
                .pending_ops
                .lock()
                .unwrap()
                .push(Operation::AllocateQubit { id: 3 });
            assert_eq!(
                pecos_get_pending_operations_with_context(ctx, &raw mut ops),
                0
            );
            assert_eq!((*ops).operations, [Operation::AllocateQubit { id: 3 }]);
            pecos_free_operations(ops);
            assert_eq!(
                pecos_get_pending_operations_with_context(ctx, &raw mut ops),
                0
            );
            assert!(!ops.is_null());
            assert_eq!((*ops).operations, []);
            pecos_free_operations(ops);
            owned
                .named_results
                .lock()
                .unwrap()
                .insert("answer".into(), NamedResult::U64(vec![42]));
            owned
                .named_result_traces
                .lock()
                .unwrap()
                .push(NamedResultTrace {
                    name: "answer".into(),
                    values: vec![true],
                    result_ids: vec![0],
                });
            assert_eq!(
                pecos_get_named_results_json_with_context(ctx, &raw mut json),
                0
            );
            assert!(
                std::ffi::CStr::from_ptr(json)
                    .to_str()
                    .unwrap()
                    .contains("42")
            );
            pecos_free_named_results_json(json);
            assert_eq!(
                pecos_get_named_result_traces_json_with_context(ctx, &raw mut json),
                0
            );
            assert!(
                std::ffi::CStr::from_ptr(json)
                    .to_str()
                    .unwrap()
                    .contains("answer")
            );
            pecos_free_named_results_json(json);
            assert_eq!(pecos_disable_dynamic_mode_with_context(ctx), 0);
            assert!(owned.sync_state.lock().unwrap().worker_complete);
            assert!(!(*registered).sync_state.lock().unwrap().worker_complete);
            let null = std::ptr::null_mut();
            assert_eq!(pecos_enable_dynamic_mode_with_context(null), 1);
            assert_eq!(pecos_disable_dynamic_mode_with_context(null), 1);
            assert_eq!(pecos_set_measurement_result_with_context(null, 0, true), 1);
            assert_eq!(pecos_set_measurement_outcome_with_context(null, 0, 2), 1);
            assert_eq!(pecos_signal_result_ready_with_context(null), 1);
            assert_eq!(
                pecos_wait_for_need_result_with_context(null, 0, &raw mut result),
                1
            );
            assert_eq!(result, u64::MAX);
            assert_eq!(
                pecos_get_pending_operations_with_context(null, &raw mut ops),
                1
            );
            assert!(ops.is_null());
            assert_eq!(
                pecos_get_named_results_json_with_context(null, &raw mut json),
                1
            );
            assert!(json.is_null());
            assert_eq!(
                pecos_get_named_result_traces_json_with_context(null, &raw mut json),
                1
            );
        }
        teardown_context(registered);
    }

    #[test]
    fn explicit_exports_report_poisoned_locks() {
        fn poison<T>(mutex: &Mutex<T>) {
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _guard = mutex.lock().unwrap();
                    panic!("poison test lock");
                }))
                .is_err()
            );
        }
        let mut owned = ExecutionContext::new();
        let ctx = &raw mut owned;
        poison(&owned.sync_state);
        poison(&owned.measurement_results);
        poison(&owned.pending_ops);
        poison(&owned.named_results);
        poison(&owned.named_result_traces);
        let mut result = 0;
        let mut ops = std::ptr::null_mut();
        let mut json = std::ptr::null_mut();
        unsafe {
            assert_eq!(pecos_enable_dynamic_mode_with_context(ctx), 2);
            assert_eq!(pecos_disable_dynamic_mode_with_context(ctx), 2);
            assert_eq!(pecos_set_measurement_result_with_context(ctx, 0, true), 2);
            assert_eq!(pecos_set_measurement_outcome_with_context(ctx, 0, 2), 2);
            assert_eq!(pecos_signal_result_ready_with_context(ctx), 2);
            assert_eq!(
                pecos_wait_for_need_result_with_context(ctx, 0, &raw mut result),
                2
            );
            assert_eq!(
                pecos_get_pending_operations_with_context(ctx, &raw mut ops),
                2
            );
            assert!(ops.is_null());
            assert_eq!(
                pecos_get_named_results_json_with_context(ctx, &raw mut json),
                2
            );
            assert!(json.is_null());
            assert_eq!(
                pecos_get_named_result_traces_json_with_context(ctx, &raw mut json),
                2
            );
            assert!(json.is_null());
        }
    }

    // --- ExecutionContext tests ---

    #[test]
    fn test_execution_context_creation() {
        let ctx = pecos_create_execution_context();
        assert!(!ctx.is_null());

        // Verify initial state
        let context = unsafe { &*ctx };
        assert!(!context.dynamic_mode_active.load(Ordering::SeqCst));
        assert_eq!(context.waiting_for_result.load(Ordering::SeqCst), u64::MAX);

        unsafe { pecos_destroy_execution_context(ctx) };
    }

    #[test]
    fn test_execution_context_reset() {
        let ctx = pecos_create_execution_context();
        let context = unsafe { &*ctx };

        // Set some state
        context.dynamic_mode_active.store(true, Ordering::SeqCst);
        context.waiting_for_result.store(42, Ordering::SeqCst);
        if let Ok(mut results) = context.measurement_results.lock() {
            results.resize(1, None);
            results[0] = Some(1);
        }
        if let Ok(mut ops) = context.pending_ops.lock() {
            ops.push(Operation::AllocateQubit { id: 0 });
        }

        // Reset
        context.reset();

        // Verify reset
        assert!(!context.dynamic_mode_active.load(Ordering::SeqCst));
        assert_eq!(context.waiting_for_result.load(Ordering::SeqCst), u64::MAX);
        if let Ok(results) = context.measurement_results.lock() {
            assert!(results.is_empty());
        }
        if let Ok(ops) = context.pending_ops.lock() {
            assert!(ops.is_empty());
        }

        unsafe { pecos_destroy_execution_context(ctx) };
    }

    #[test]
    fn test_register_unregister_context() {
        let ctx = setup_context();

        // Verify context is registered
        assert!(get_execution_context().is_some());

        // Unregister
        unsafe { pecos_register_execution_context(std::ptr::null_mut()) };
        assert!(get_execution_context().is_none());

        unsafe { pecos_destroy_execution_context(ctx) };
    }

    // --- Dynamic mode tests (with context) ---

    #[test]
    fn test_enable_disable_dynamic_mode() {
        let ctx = setup_context();

        assert!(!is_dynamic_mode_active());

        pecos_enable_dynamic_mode();
        assert!(is_dynamic_mode_active());

        pecos_disable_dynamic_mode();
        assert!(!is_dynamic_mode_active());

        teardown_context(ctx);
    }

    #[test]
    fn test_worker_complete_signaling() {
        let ctx = setup_context();

        pecos_enable_dynamic_mode();

        // Initially worker is not complete
        assert!(!pecos_is_worker_complete());

        // Disable dynamic mode signals completion
        pecos_disable_dynamic_mode();

        assert!(pecos_is_worker_complete());

        teardown_context(ctx);
    }

    #[test]
    fn test_measurement_result_storage() {
        let ctx = setup_context();

        // Set a measurement result
        pecos_set_measurement_result(42, true);
        pecos_set_measurement_result(43, false);

        // Retrieve results
        assert_eq!(get_measurement_result(42), Some(true));
        assert_eq!(get_measurement_result(43), Some(false));
        assert_eq!(get_measurement_result(99), None);

        teardown_context(ctx);
    }

    #[test]
    fn test_enable_clears_previous_state() {
        let ctx = setup_context();

        // Set some state
        pecos_set_measurement_result(1, true);
        let context = unsafe { &*ctx };
        if let Ok(mut ops) = context.pending_ops.lock() {
            ops.push(Operation::AllocateQubit { id: 0 });
        }

        // Enable dynamic mode should clear state
        pecos_enable_dynamic_mode();

        assert_eq!(get_measurement_result(1), None);
        if let Ok(ops) = context.pending_ops.lock() {
            assert!(ops.is_empty());
        }

        teardown_context(ctx);
    }

    #[test]
    fn test_check_need_result_when_not_needed() {
        let ctx = setup_context();

        // When no result is needed, should return MAX
        assert_eq!(pecos_check_need_result(), u64::MAX);

        teardown_context(ctx);
    }

    #[test]
    fn test_check_need_result_no_context() {
        // When no context is registered, should return MAX
        assert_eq!(pecos_check_need_result(), u64::MAX);
    }

    #[test]
    fn test_wait_for_need_result_timeout() {
        let ctx = setup_context();

        pecos_enable_dynamic_mode();

        // With short timeout and no worker requesting results, should timeout
        let result = pecos_wait_for_need_result(10);
        assert_eq!(result, u64::MAX);

        teardown_context(ctx);
    }

    #[test]
    fn test_wait_for_need_result_worker_complete() {
        let ctx = setup_context();

        pecos_enable_dynamic_mode();

        // Simulate worker completing immediately
        pecos_disable_dynamic_mode();

        // Should return MAX because worker completed
        let result = pecos_wait_for_need_result(100);
        assert_eq!(result, u64::MAX);

        teardown_context(ctx);
    }

    // --- Cross-thread tests with shared context ---

    #[test]
    fn test_cross_thread_result_signaling() {
        use std::sync::Barrier;
        use std::time::Duration;

        // Create context that will be shared between threads
        let ctx = pecos_create_execution_context();
        let ctx_ptr = ctx as usize; // Convert to usize for Send

        // Register on main thread first
        unsafe { pecos_register_execution_context(ctx) };
        pecos_enable_dynamic_mode();

        // Use a barrier to ensure proper synchronization
        let barrier = Arc::new(Barrier::new(2));
        let worker_barrier = Arc::clone(&barrier);

        // Spawn a "worker" thread that requests a result
        let worker = thread::spawn(move || {
            // Register the same context on worker thread
            let ctx = ctx_ptr as *mut ExecutionContext;
            unsafe { pecos_register_execution_context(ctx) };

            let context = unsafe { &*ctx };

            // Signal that we need result 5
            context.waiting_for_result.store(5, Ordering::SeqCst);
            if let Ok(mut state) = context.sync_state.lock() {
                state.need_result = true;
            }
            context.sync_condvar.notify_all();

            // Sync with main thread - ensure it can see our signal
            worker_barrier.wait();

            // Wait for the result (with timeout)
            let timeout = Duration::from_secs(1);
            let mut state = context.sync_state.lock().unwrap();
            while !state.result_ready {
                let result = context.sync_condvar.wait_timeout(state, timeout).unwrap();
                state = result.0;
                if result.1.timed_out() {
                    unsafe { pecos_register_execution_context(std::ptr::null_mut()) };
                    return None;
                }
            }

            let result = get_measurement_result(5);
            unsafe { pecos_register_execution_context(std::ptr::null_mut()) };
            result
        });

        // Main thread: wait for worker to signal it needs result
        barrier.wait();

        // Now worker has definitely set need_result
        let needed_id = pecos_wait_for_need_result(TEST_SYNC_TIMEOUT_MS);
        assert_eq!(needed_id, 5);

        // Provide the result
        pecos_set_measurement_result(5, true);
        pecos_signal_result_ready();

        // Worker should receive the result
        let result = worker.join().unwrap();
        assert_eq!(result, Some(true));

        // Cleanup on main thread
        unsafe {
            pecos_register_execution_context(std::ptr::null_mut());
            pecos_destroy_execution_context(ctx);
        }
    }

    #[test]
    fn test_pending_operations_storage() {
        if !crate::test_env::run_test_in_child("tests::test_pending_operations_storage") {
            return;
        }
        let ctx = setup_context();

        let context = unsafe { &*ctx };

        // Store some operations in context storage
        if let Ok(mut ops) = context.pending_ops.lock() {
            ops.push(Operation::AllocateQubit { id: 0 });
            ops.push(Operation::AllocateQubit { id: 1 });
        }

        // Get pending operations
        let ptr = pecos_get_pending_operations();
        // SAFETY: `pecos_get_pending_operations` returns null or a leaked
        // `Box<OperationCollector>` (properly initialised + aligned);
        // ownership stays here until `pecos_free_operations` below.
        let collector =
            unsafe { ptr.as_ref() }.expect("pecos_get_pending_operations returned null");
        assert_eq!(collector.operations.len(), 2);

        // Free the collector
        unsafe { pecos_free_operations(ptr) };

        // Second read should be empty because the handoff drains pending ops.
        let ptr = pecos_get_pending_operations();
        assert!(!ptr.is_null());
        assert_eq!(unsafe { &*ptr }.operations, []);
        unsafe { pecos_free_operations(ptr) };

        teardown_context(ctx);
    }

    #[test]
    fn test_pending_operations_empty() {
        if !crate::test_env::run_test_in_child("tests::test_pending_operations_empty") {
            return;
        }
        let ctx = setup_context();

        // Successful empty imports are distinct from unavailable imports.
        let ptr = pecos_get_pending_operations();
        assert!(!ptr.is_null());
        assert_eq!(unsafe { &*ptr }.operations, []);
        unsafe { pecos_free_operations(ptr) };

        teardown_context(ctx);
    }

    #[test]
    fn test_pending_operations_no_context() {
        // When no context is registered, should return null
        let ptr = pecos_get_pending_operations();
        assert!(ptr.is_null());
    }

    // --- Thread-local interface tests (don't require execution context) ---

    #[test]
    fn test_interface_reset() {
        // Store some operations
        with_interface(|iface| {
            iface.operations.push(Operation::AllocateQubit { id: 0 });
        });

        // Verify operation was stored
        let count = with_interface(|iface| iface.operations.len());
        assert_eq!(count, 1);

        // Reset
        reset_interface();

        // Should be empty
        let count = with_interface(|iface| iface.operations.len());
        assert_eq!(count, 0);
    }

    #[test]
    fn test_set_measurements() {
        reset_interface();

        // Set measurements
        set_measurements([(0, true), (1, false)]);

        // Verify via interface
        let result_0 = with_interface(|iface| iface.get_result(0));
        let result_1 = with_interface(|iface| iface.get_result(1));

        assert_eq!(result_0, Some(true));
        assert_eq!(result_1, Some(false));
    }

    #[test]
    fn test_quantum_executor_callback() {
        reset_interface();

        // Set up an executor that returns fixed results
        set_quantum_executor(|_collector| {
            let mut results = BTreeMap::new();
            results.insert(0, true);
            results.insert(1, false);
            results
        });

        // Execute should succeed
        let executed = execute_pending_and_get_results();
        assert!(executed);

        // Results should be stored
        let result_0 = with_interface(|iface| iface.get_result(0));
        assert_eq!(result_0, Some(true));

        // Clear executor
        clear_quantum_executor();

        // Now execute should fail (no executor)
        let executed = execute_pending_and_get_results();
        assert!(!executed);
    }

    #[test]
    fn test_get_interface_clone() {
        reset_interface();

        with_interface(|iface| {
            iface.queue_operation(Operation::AllocateQubit { id: 0 });
        });

        let clone = get_interface_clone();
        assert_eq!(clone.operations.len(), 1);
    }

    #[test]
    fn test_clear_pending_operations() {
        reset_interface();

        with_interface(|iface| {
            iface.queue_operation(Operation::AllocateQubit { id: 0 });
            iface.queue_operation(Operation::Quantum(QuantumOp::H(0)));
        });

        pecos_clear_pending_operations();

        with_interface(|iface| {
            assert_eq!(iface.operations, []);
        });
    }

    // --- wait_for_result_ready tests ---

    #[test]
    fn test_wait_for_result_ready_no_context() {
        if !crate::test_env::run_test_in_child("tests::test_wait_for_result_ready_no_context") {
            return;
        }
        // Without a context, report the failure immediately
        let result = wait_for_result_ready(0);
        assert_eq!(result, ResultWaitOutcome::Failed("no execution context"));
    }

    #[test]
    fn test_wait_for_result_ready_worker_complete() {
        if !crate::test_env::run_test_in_child("tests::test_wait_for_result_ready_worker_complete")
        {
            return;
        }
        let ctx = setup_context();

        pecos_enable_dynamic_mode();

        unsafe { &*ctx }.sync_state.lock().unwrap().worker_complete = true;
        assert_eq!(wait_for_result_ready(0), ResultWaitOutcome::WorkerComplete);

        teardown_context(ctx);
    }

    #[test]
    fn test_wait_for_result_ready_exports_operations() {
        if !crate::test_env::run_test_in_child(
            "tests::test_wait_for_result_ready_exports_operations",
        ) {
            return;
        }

        // Create context that will be shared between threads
        let ctx = pecos_create_execution_context();
        let ctx_ptr = ctx as usize;

        unsafe { pecos_register_execution_context(ctx) };
        pecos_enable_dynamic_mode();

        // Store some operations in the thread-local interface
        with_interface(|iface| {
            iface.queue_operation(Operation::AllocateQubit { id: 0 });
            iface.queue_operation(Operation::Quantum(QuantumOp::H(0)));
        });

        let barrier = Arc::new(Barrier::new(2));
        let worker_barrier = Arc::clone(&barrier);

        // Spawn a thread that waits for need_result then signals ready
        let handle = thread::spawn(move || {
            let ctx = ctx_ptr as *mut ExecutionContext;
            unsafe { pecos_register_execution_context(ctx) };

            // Sync with main
            worker_barrier.wait();

            // Wait for main thread to signal it needs a result
            let needed_id = pecos_wait_for_need_result(TEST_SYNC_TIMEOUT_MS);
            assert_eq!(needed_id, 5);
            pecos_signal_result_ready();

            unsafe { pecos_register_execution_context(std::ptr::null_mut()) };
        });

        barrier.wait();

        // Wait for result - this should export operations to context storage
        let result = wait_for_result_ready(5);
        assert_eq!(result, ResultWaitOutcome::Ready);

        // Verify operations were exported to context storage
        let context = unsafe { &*ctx };
        if let Ok(ops) = context.pending_ops.lock() {
            assert_eq!(ops.len(), 2);
        }

        handle.join().unwrap();

        unsafe {
            pecos_register_execution_context(std::ptr::null_mut());
            pecos_destroy_execution_context(ctx);
        }
    }

    #[test]
    fn test_wait_for_result_ready_signals_need() {
        if !crate::test_env::run_test_in_child("tests::test_wait_for_result_ready_signals_need") {
            return;
        }

        // Create context that will be shared between threads
        let ctx = pecos_create_execution_context();
        let ctx_ptr = ctx as usize;

        unsafe { pecos_register_execution_context(ctx) };
        pecos_enable_dynamic_mode();

        let barrier = Arc::new(Barrier::new(2));
        let worker_barrier = Arc::clone(&barrier);

        // Spawn a worker that will call wait_for_result_ready
        let worker = thread::spawn(move || {
            let ctx = ctx_ptr as *mut ExecutionContext;
            unsafe { pecos_register_execution_context(ctx) };

            worker_barrier.wait();

            let result = wait_for_result_ready(42);

            unsafe { pecos_register_execution_context(std::ptr::null_mut()) };
            result
        });

        barrier.wait();

        // Main thread: wait for worker to signal it needs result
        let needed = pecos_wait_for_need_result(TEST_SYNC_TIMEOUT_MS);
        assert_eq!(needed, 42);

        // Signal result ready
        pecos_signal_result_ready();

        let result = worker.join().unwrap();
        assert_eq!(result, ResultWaitOutcome::Ready);

        unsafe {
            pecos_register_execution_context(std::ptr::null_mut());
            pecos_destroy_execution_context(ctx);
        }
    }

    #[test]
    fn test_wait_for_result_ready_full_cycle() {
        if !crate::test_env::run_test_in_child("tests::test_wait_for_result_ready_full_cycle") {
            return;
        }

        // Create context that will be shared between threads
        let ctx = pecos_create_execution_context();
        let ctx_ptr = ctx as usize;

        unsafe { pecos_register_execution_context(ctx) };
        pecos_enable_dynamic_mode();

        let barrier = Arc::new(Barrier::new(2));
        let worker_barrier = Arc::clone(&barrier);

        // Spawn a worker that requests a result
        let worker = thread::spawn(move || {
            let ctx = ctx_ptr as *mut ExecutionContext;
            unsafe { pecos_register_execution_context(ctx) };

            with_interface(|iface| {
                iface.queue_operation(Operation::AllocateQubit { id: 0 });
                iface.queue_operation(Operation::Quantum(QuantumOp::Measure(0, 0)));
            });

            worker_barrier.wait();

            // This will export ops and wait for result
            let result = if wait_for_result_ready(0) == ResultWaitOutcome::Ready {
                get_measurement_result(0)
            } else {
                None
            };

            unsafe { pecos_register_execution_context(std::ptr::null_mut()) };
            result
        });

        barrier.wait();

        // Main thread: wait for worker to need result
        let needed_id = pecos_wait_for_need_result(TEST_SYNC_TIMEOUT_MS);
        assert_eq!(needed_id, 0);

        // Verify operations were exported
        let ops_ptr = pecos_get_pending_operations();
        // SAFETY: see `pecos_get_pending_operations` -- null-or-leaked-Box invariant.
        let ops = unsafe { ops_ptr.as_ref() }.expect("pecos_get_pending_operations returned null");
        assert_eq!(ops.operations.len(), 2);
        unsafe { pecos_free_operations(ops_ptr) };

        // Provide the measurement result
        pecos_set_measurement_result(0, true);
        pecos_signal_result_ready();

        // Worker should get the result
        let result = worker.join().unwrap();
        assert_eq!(result, Some(true));

        unsafe {
            pecos_register_execution_context(std::ptr::null_mut());
            pecos_destroy_execution_context(ctx);
        }
    }

    #[test]
    fn test_wait_for_result_ready_exports_only_new_operations() {
        if !crate::test_env::run_test_in_child(
            "tests::test_wait_for_result_ready_exports_only_new_operations",
        ) {
            return;
        }

        let ctx = pecos_create_execution_context();
        let ctx_ptr = ctx as usize;

        unsafe { pecos_register_execution_context(ctx) };
        pecos_enable_dynamic_mode();

        let barrier = Arc::new(Barrier::new(2));
        let worker_barrier = Arc::clone(&barrier);

        let worker = thread::spawn(move || {
            let ctx = ctx_ptr as *mut ExecutionContext;
            unsafe { pecos_register_execution_context(ctx) };

            with_interface(|iface| {
                iface.queue_operation(Operation::AllocateQubit { id: 0 });
            });

            worker_barrier.wait();

            assert_eq!(wait_for_result_ready(0), ResultWaitOutcome::Ready);

            with_interface(|iface| {
                assert_eq!(iface.operations, []);
            });

            with_interface(|iface| {
                iface.queue_operation(Operation::Quantum(QuantumOp::H(0)));
            });

            assert_eq!(wait_for_result_ready(1), ResultWaitOutcome::Ready);

            with_interface(|iface| {
                assert_eq!(iface.operations, []);
            });

            unsafe { pecos_register_execution_context(std::ptr::null_mut()) };
        });

        barrier.wait();

        let needed_id = pecos_wait_for_need_result(TEST_SYNC_TIMEOUT_MS);
        assert_eq!(needed_id, 0);
        let ops_ptr = pecos_get_pending_operations();
        // SAFETY: see `pecos_get_pending_operations` -- null-or-leaked-Box invariant.
        let ops = unsafe { ops_ptr.as_ref() }.expect("pecos_get_pending_operations returned null");
        assert_eq!(ops.operations, vec![Operation::AllocateQubit { id: 0 }]);
        unsafe { pecos_free_operations(ops_ptr) };
        pecos_signal_result_ready();

        let needed_id = pecos_wait_for_need_result(TEST_SYNC_TIMEOUT_MS);
        assert_eq!(needed_id, 1);
        let ops_ptr = pecos_get_pending_operations();
        // SAFETY: see `pecos_get_pending_operations` -- null-or-leaked-Box invariant.
        let ops = unsafe { ops_ptr.as_ref() }.expect("pecos_get_pending_operations returned null");
        assert_eq!(ops.operations, vec![Operation::Quantum(QuantumOp::H(0))]);
        unsafe { pecos_free_operations(ops_ptr) };
        pecos_signal_result_ready();

        worker.join().unwrap();

        unsafe {
            pecos_register_execution_context(std::ptr::null_mut());
            pecos_destroy_execution_context(ctx);
        }
    }

    #[test]
    fn test_is_dynamic_mode_active() {
        let ctx = setup_context();

        assert!(!is_dynamic_mode_active());

        pecos_enable_dynamic_mode();
        assert!(is_dynamic_mode_active());

        pecos_disable_dynamic_mode();
        assert!(!is_dynamic_mode_active());

        teardown_context(ctx);
    }

    #[test]
    fn test_is_dynamic_mode_active_no_context() {
        // Without context, should return false
        assert!(!is_dynamic_mode_active());
    }

    #[test]
    fn test_get_measurement_result() {
        let ctx = setup_context();

        // Initially no results
        assert_eq!(get_measurement_result(0), None);

        // Set a result
        pecos_set_measurement_result(0, true);
        assert_eq!(get_measurement_result(0), Some(true));

        pecos_set_measurement_result(1, false);
        assert_eq!(get_measurement_result(1), Some(false));

        // Non-existent result
        assert_eq!(get_measurement_result(999), None);

        teardown_context(ctx);
    }

    #[test]
    fn test_get_measurement_result_no_context() {
        // Without context, should return None
        assert_eq!(get_measurement_result(0), None);
    }

    // --- Parallel execution isolation tests ---

    #[test]
    fn test_parallel_contexts_are_isolated() {
        use std::sync::Barrier;

        let barrier = Arc::new(Barrier::new(2));
        let barrier1 = Arc::clone(&barrier);
        let barrier2 = Arc::clone(&barrier);

        // Spawn two threads, each with their own context
        let thread1 = thread::spawn(move || {
            let ctx = pecos_create_execution_context();
            unsafe { pecos_register_execution_context(ctx) };

            pecos_enable_dynamic_mode();
            pecos_set_measurement_result(0, true);

            barrier1.wait(); // Sync point 1
            barrier1.wait(); // Sync point 2

            // Should still have our value
            let result = get_measurement_result(0);

            unsafe {
                pecos_register_execution_context(std::ptr::null_mut());
                pecos_destroy_execution_context(ctx);
            }

            result
        });

        let thread2 = thread::spawn(move || {
            let ctx = pecos_create_execution_context();
            unsafe { pecos_register_execution_context(ctx) };

            pecos_enable_dynamic_mode();
            pecos_set_measurement_result(0, false);

            barrier2.wait(); // Sync point 1
            barrier2.wait(); // Sync point 2

            // Should have our own value, not thread1's
            let result = get_measurement_result(0);

            unsafe {
                pecos_register_execution_context(std::ptr::null_mut());
                pecos_destroy_execution_context(ctx);
            }

            result
        });

        let result1 = thread1.join().unwrap();
        let result2 = thread2.join().unwrap();

        // Each thread should have its own isolated value
        assert_eq!(result1, Some(true));
        assert_eq!(result2, Some(false));
    }
}
