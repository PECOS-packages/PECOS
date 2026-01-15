//! Minimal QIS Interface for Fast Linking
//!
//! This crate provides the minimal FFI interface needed to link QIS (Quantum Instruction Set)
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
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};

pub mod ffi;

// =============================================================================
// Per-Execution Context for Parallel Execution Support
// =============================================================================

/// State for dynamic circuit synchronization
#[derive(Debug, Default)]
pub struct DynamicSyncState {
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
pub struct ExecutionContext {
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
    /// Storage for measurement results (shared between threads)
    pub measurement_results: Mutex<BTreeMap<u64, bool>>,
    /// Storage for named results from `print_bool`/`print_bool_arr` (e.g., "synx", "final")
    pub named_results: Mutex<BTreeMap<String, Vec<bool>>>,
}

impl ExecutionContext {
    /// Create a new execution context with default state
    #[must_use]
    pub fn new() -> Self {
        Self {
            dynamic_mode_active: AtomicBool::new(false),
            waiting_for_result: AtomicU64::new(u64::MAX),
            sync_state: Mutex::new(DynamicSyncState::default()),
            sync_condvar: Condvar::new(),
            pending_ops: Mutex::new(Vec::new()),
            measurement_results: Mutex::new(BTreeMap::new()),
            named_results: Mutex::new(BTreeMap::new()),
        }
    }

    /// Reset the context to initial state (for reuse)
    pub fn reset(&self) {
        self.dynamic_mode_active.store(false, Ordering::SeqCst);
        self.waiting_for_result.store(u64::MAX, Ordering::SeqCst);
        if let Ok(mut state) = self.sync_state.lock() {
            state.result_ready = false;
            state.need_result = false;
            state.worker_complete = false;
        }
        if let Ok(mut results) = self.measurement_results.lock() {
            results.clear();
        }
        if let Ok(mut ops) = self.pending_ops.lock() {
            ops.clear();
        }
        if let Ok(mut named) = self.named_results.lock() {
            named.clear();
        }
    }

    /// Store a named result (single bool value)
    pub fn store_named_bool(&self, name: &str, value: bool) {
        let thread_id = std::thread::current().id();
        if let Ok(mut named) = self.named_results.lock() {
            let entry = named.entry(name.to_string()).or_default();
            entry.push(value);
            log::debug!(
                "ExecutionContext::store_named_bool: thread {:?} stored '{}' = {} (now {} values: {:?})",
                thread_id,
                name,
                value,
                entry.len(),
                entry
            );
        } else {
            log::error!(
                "ExecutionContext::store_named_bool: thread {thread_id:?} failed to acquire lock for '{name}'"
            );
        }
    }

    /// Store a named result array (multiple bool values)
    pub fn store_named_array(&self, name: &str, values: &[bool]) {
        if let Ok(mut named) = self.named_results.lock() {
            let entry = named.entry(name.to_string()).or_default();
            entry.extend_from_slice(values);
        }
    }

    /// Get all named results (returns a clone)
    #[must_use]
    pub fn get_named_results(&self) -> BTreeMap<String, Vec<bool>> {
        self.named_results
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

/// Destroy an execution context
///
/// # Safety
/// The pointer must have been created by `pecos_create_execution_context`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_destroy_execution_context(ctx: *mut ExecutionContext) {
    log::debug!("pecos_destroy_execution_context called: ctx={ctx:?}");
    if !ctx.is_null() {
        // SAFETY: ptr was allocated by Box::into_raw in pecos_create_execution_context
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
pub use pecos_qis_ffi_types::{Operation, OperationCollector, OperationList, QuantumOp};

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
    with_interface(OperationCollector::reset);
    // Also reset the collection mode read counter for loop termination
    ffi::reset_collection_read_count();
}

/// Get a clone of the thread-local operation collector
#[must_use]
pub fn get_interface_clone() -> OperationCollector {
    with_interface(|interface| interface.clone())
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
/// ```ignore
/// set_quantum_executor(|collector| {
///     let ops = collector.take_operations();
///     let results = my_simulator.execute(ops);
///     results
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

// =============================================================================
// FFI functions for cross-library dynamic circuit coordination
// =============================================================================

/// Enable dynamic execution mode (called via FFI from executor)
///
/// Requires a per-execution context to be registered via `pecos_register_execution_context`.
/// If no context is registered, this is a no-op and logs a warning.
///
/// # Safety
/// This function is safe to call from any thread.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_enable_dynamic_mode() {
    log::debug!("pecos_enable_dynamic_mode called");

    if let Some(ctx) = get_execution_context() {
        // SAFETY: Context is valid for duration of execution
        let ctx = unsafe { &*ctx };
        ctx.dynamic_mode_active.store(true, Ordering::SeqCst);
        // Reset sync state
        if let Ok(mut state) = ctx.sync_state.lock() {
            state.result_ready = false;
            state.need_result = false;
            state.worker_complete = false;
        }
        // Clear storage for new shot
        if let Ok(mut results) = ctx.measurement_results.lock() {
            results.clear();
        }
        if let Ok(mut ops) = ctx.pending_ops.lock() {
            ops.clear();
        }
        log::debug!("pecos_enable_dynamic_mode: enabled");
    } else {
        log::warn!("pecos_enable_dynamic_mode: no execution context registered");
    }
}

/// Disable dynamic execution mode (called via FFI from executor)
///
/// This also signals completion so the main thread wakes up.
///
/// Requires a per-execution context to be registered via `pecos_register_execution_context`.
/// If no context is registered, this is a no-op and logs a warning.
///
/// # Safety
/// This function is safe to call from any thread.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_disable_dynamic_mode() {
    log::debug!("pecos_disable_dynamic_mode called");

    if let Some(ctx) = get_execution_context() {
        // SAFETY: Context is valid for duration of execution
        let ctx = unsafe { &*ctx };
        ctx.dynamic_mode_active.store(false, Ordering::SeqCst);
        // Signal worker completion so main thread wakes up
        if let Ok(mut state) = ctx.sync_state.lock() {
            state.worker_complete = true;
        }
        ctx.sync_condvar.notify_all();
        log::debug!("pecos_disable_dynamic_mode: disabled");
    } else {
        log::warn!("pecos_disable_dynamic_mode: no execution context registered");
    }
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
    use std::time::Duration;

    let timeout = Duration::from_millis(timeout_ms);

    let Some(ctx) = get_execution_context() else {
        log::warn!("pecos_wait_for_need_result: no execution context registered");
        return u64::MAX;
    };

    // SAFETY: Context is valid for duration of execution
    let ctx = unsafe { &*ctx };

    let Ok(mut state) = ctx.sync_state.lock() else {
        return u64::MAX;
    };

    // Wait until either: need_result is true, worker_complete is true, or timeout
    while !state.need_result && !state.worker_complete {
        let result = ctx.sync_condvar.wait_timeout(state, timeout);
        match result {
            Ok((s, timed_out)) => {
                state = s;
                if timed_out.timed_out() {
                    log::debug!("pecos_wait_for_need_result: timeout");
                    return u64::MAX;
                }
            }
            Err(_) => return u64::MAX,
        }
    }

    if state.worker_complete {
        log::debug!("pecos_wait_for_need_result: worker complete");
        u64::MAX
    } else if state.need_result {
        let result_id = ctx.waiting_for_result.load(Ordering::SeqCst);
        log::debug!("pecos_wait_for_need_result: got result_id={result_id}");
        result_id
    } else {
        u64::MAX
    }
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
    log::debug!("pecos_signal_result_ready called");

    if let Some(ctx) = get_execution_context() {
        // SAFETY: Context is valid for duration of execution
        let ctx = unsafe { &*ctx };
        if let Ok(mut state) = ctx.sync_state.lock() {
            state.result_ready = true;
            state.need_result = false;
        }
        ctx.sync_condvar.notify_all();
        log::debug!("pecos_signal_result_ready: signaled");
    } else {
        log::warn!("pecos_signal_result_ready: no execution context registered");
    }
}

/// Wait for a result to be ready (called by worker thread inside `___read_future_bool`)
///
/// Returns true if result is ready, false on timeout or if no context is registered.
#[must_use]
pub fn wait_for_result_ready(result_id: u64, timeout_ms: u64) -> bool {
    use std::time::Duration;

    log::debug!("wait_for_result_ready: result_id={result_id}, timeout={timeout_ms}ms");

    let Some(ctx) = get_execution_context() else {
        log::warn!("wait_for_result_ready: no execution context registered");
        return false;
    };

    // SAFETY: Context is valid for duration of execution
    let ctx = unsafe { &*ctx };

    // Export pending operations to context storage before blocking
    // This allows the main thread to access them
    INTERFACE.with(|interface| {
        let iface = interface.borrow();
        if let Ok(mut pending) = ctx.pending_ops.lock() {
            pending.clear();
            pending.extend(iface.operations.iter().cloned());
            log::debug!(
                "wait_for_result_ready: exported {} pending operations",
                pending.len()
            );
        }
    });

    // Signal that we need a result
    ctx.waiting_for_result.store(result_id, Ordering::SeqCst);
    if let Ok(mut state) = ctx.sync_state.lock() {
        state.need_result = true;
        state.result_ready = false;
    }
    ctx.sync_condvar.notify_all();

    // Wait for result to be ready
    let timeout = Duration::from_millis(timeout_ms);
    let Ok(mut state) = ctx.sync_state.lock() else {
        return false;
    };

    if !state.result_ready {
        let result = ctx.sync_condvar.wait_timeout(state, timeout);
        state = match result {
            Ok((s, _)) => s,
            Err(_) => return false,
        };
    }

    log::debug!("wait_for_result_ready: result_ready={}", state.result_ready);
    state.result_ready
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
pub fn get_measurement_result(result_id: u64) -> Option<bool> {
    let ctx = get_execution_context()?;
    // SAFETY: Context is valid for duration of execution
    let ctx = unsafe { &*ctx };
    if let Ok(results) = ctx.measurement_results.lock() {
        let value = results.get(&result_id).copied();
        log::debug!("get_measurement_result: result_id={result_id}, value={value:?}");
        value
    } else {
        None
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
    log::debug!("pecos_set_measurement_result: result_id={result_id}, value={value}");
    if let Some(ctx) = get_execution_context() {
        // SAFETY: Context is valid for duration of execution
        let ctx = unsafe { &*ctx };
        if let Ok(mut results) = ctx.measurement_results.lock() {
            results.insert(result_id, value);
        }
    } else {
        log::warn!("pecos_set_measurement_result: no execution context registered");
    }
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
/// Returns null if no operations are available or no context is registered.
///
/// # Safety
/// This function is safe to call from any thread. The returned pointer must be freed.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_get_pending_operations() -> *mut OperationCollector {
    let Some(ctx) = get_execution_context() else {
        log::warn!("pecos_get_pending_operations: no execution context registered");
        return std::ptr::null_mut();
    };

    // SAFETY: Context is valid for duration of execution
    let ctx = unsafe { &*ctx };

    let ops = match ctx.pending_ops.lock() {
        Ok(pending) => {
            log::debug!("pecos_get_pending_operations: {} operations", pending.len());
            pending.clone()
        }
        Err(_) => return std::ptr::null_mut(),
    };

    if ops.is_empty() {
        return std::ptr::null_mut();
    }

    let mut collector = OperationCollector::new();
    collector.operations = ops;
    Box::into_raw(Box::new(collector))
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
    let thread_id = std::thread::current().id();
    log::debug!("pecos_get_named_results_json: called from thread {thread_id:?}");

    let Some(ctx) = get_execution_context() else {
        log::warn!(
            "pecos_get_named_results_json: no execution context registered on thread {thread_id:?}"
        );
        return std::ptr::null_mut();
    };

    log::debug!("pecos_get_named_results_json: found context {ctx:?} on thread {thread_id:?}");

    // SAFETY: Context is valid for duration of execution
    let ctx = unsafe { &*ctx };
    let named_results = ctx.get_named_results();

    if named_results.is_empty() {
        log::debug!(
            "pecos_get_named_results_json: no named results in context on thread {thread_id:?}"
        );
        return std::ptr::null_mut();
    }

    // Log details about what we're returning
    log::debug!(
        "pecos_get_named_results_json: thread {:?} returning {} keys: {:?}",
        thread_id,
        named_results.len(),
        named_results.keys().collect::<Vec<_>>()
    );
    for (key, values) in &named_results {
        log::debug!("  {} -> {} values: {:?}", key, values.len(), values);
    }

    // Serialize to JSON
    let json = match serde_json::to_string(&named_results) {
        Ok(s) => s,
        Err(e) => {
            log::error!("pecos_get_named_results_json: serialization error: {e}");
            return std::ptr::null_mut();
        }
    };

    log::debug!(
        "pecos_get_named_results_json: returning {} bytes",
        json.len()
    );

    // Convert to C string
    match std::ffi::CString::new(json) {
        Ok(cstr) => cstr.into_raw(),
        Err(e) => {
            log::error!("pecos_get_named_results_json: CString error: {e}");
            std::ptr::null_mut()
        }
    }
}

/// Free a JSON string allocated by `pecos_get_named_results_json`
///
/// # Safety
/// The pointer must have been allocated by `pecos_get_named_results_json`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_free_named_results_json(ptr: *mut std::ffi::c_char) {
    if !ptr.is_null() {
        // SAFETY: ptr was allocated by CString::into_raw in pecos_get_named_results_json
        drop(unsafe { std::ffi::CString::from_raw(ptr) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

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

    // =========================================================================
    // ExecutionContext tests
    // =========================================================================

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
            results.insert(0, true);
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

    // =========================================================================
    // Dynamic mode tests (with context)
    // =========================================================================

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

    // =========================================================================
    // Cross-thread tests with shared context
    // =========================================================================

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
            let timeout = Duration::from_millis(1000);
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
        let needed_id = pecos_wait_for_need_result(500);
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
        let ctx = setup_context();

        let context = unsafe { &*ctx };

        // Store some operations in context storage
        if let Ok(mut ops) = context.pending_ops.lock() {
            ops.push(Operation::AllocateQubit { id: 0 });
            ops.push(Operation::AllocateQubit { id: 1 });
        }

        // Get pending operations
        let ptr = pecos_get_pending_operations();
        assert!(!ptr.is_null());

        // Verify operations
        let collector = unsafe { &*ptr };
        assert_eq!(collector.operations.len(), 2);

        // Free the collector
        unsafe { pecos_free_operations(ptr) };

        teardown_context(ctx);
    }

    #[test]
    fn test_pending_operations_empty() {
        let ctx = setup_context();

        // When no operations, should return null
        let ptr = pecos_get_pending_operations();
        assert!(ptr.is_null());

        teardown_context(ctx);
    }

    #[test]
    fn test_pending_operations_no_context() {
        // When no context is registered, should return null
        let ptr = pecos_get_pending_operations();
        assert!(ptr.is_null());
    }

    // =========================================================================
    // Thread-local interface tests (don't require execution context)
    // =========================================================================

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
            assert!(iface.operations.is_empty());
        });
    }

    // =========================================================================
    // wait_for_result_ready tests
    // =========================================================================

    #[test]
    fn test_wait_for_result_ready_no_context() {
        // Without a context, should return false immediately
        let result = wait_for_result_ready(0, 10);
        assert!(!result);
    }

    #[test]
    fn test_wait_for_result_ready_timeout() {
        let ctx = setup_context();

        pecos_enable_dynamic_mode();

        // No one will signal result_ready, so this should timeout
        let result = wait_for_result_ready(0, 10);
        assert!(!result);

        teardown_context(ctx);
    }

    #[test]
    fn test_wait_for_result_ready_exports_operations() {
        use std::sync::Barrier;

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
            let needed_id = pecos_wait_for_need_result(500);
            assert_eq!(needed_id, 5);
            pecos_signal_result_ready();

            unsafe { pecos_register_execution_context(std::ptr::null_mut()) };
        });

        barrier.wait();

        // Wait for result - this should export operations to context storage
        let result = wait_for_result_ready(5, 500);
        assert!(result);

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
        use std::sync::Barrier;

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

            let result = wait_for_result_ready(42, 500);

            unsafe { pecos_register_execution_context(std::ptr::null_mut()) };
            result
        });

        barrier.wait();

        // Main thread: wait for worker to signal it needs result
        let needed = pecos_wait_for_need_result(500);
        assert_eq!(needed, 42);

        // Signal result ready
        pecos_signal_result_ready();

        let result = worker.join().unwrap();
        assert!(result);

        unsafe {
            pecos_register_execution_context(std::ptr::null_mut());
            pecos_destroy_execution_context(ctx);
        }
    }

    #[test]
    fn test_wait_for_result_ready_full_cycle() {
        use std::sync::Barrier;

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
            let result = if wait_for_result_ready(0, 500) {
                get_measurement_result(0)
            } else {
                None
            };

            unsafe { pecos_register_execution_context(std::ptr::null_mut()) };
            result
        });

        barrier.wait();

        // Main thread: wait for worker to need result
        let needed_id = pecos_wait_for_need_result(500);
        assert_eq!(needed_id, 0);

        // Verify operations were exported
        let ops_ptr = pecos_get_pending_operations();
        assert!(!ops_ptr.is_null());
        let ops = unsafe { &*ops_ptr };
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

    // =========================================================================
    // Parallel execution isolation tests
    // =========================================================================

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
