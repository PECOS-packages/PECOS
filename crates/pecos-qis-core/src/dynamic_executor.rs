//! Dynamic Execution Coordinator for QIS
//!
//! This module provides thread synchronization for dynamic circuit execution.
//! When an LLVM program needs measurement results that aren't yet available
//! (e.g., for conditionals depending on measurements), the execution needs to
//! pause, let the quantum simulator run, and then resume with the results.
//!
//! The coordinator manages communication between:
//! - Worker thread: Runs the LLVM program
//! - Main thread: Runs `QisEngine` methods (`generate_commands`, `continue_processing`)

use log::debug;
use pecos_qis_ffi_types::{Operation, OperationCollector};
use std::collections::BTreeMap;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

/// Messages from the worker thread to the main thread
#[derive(Debug)]
pub enum WorkerMessage {
    /// Pending operations are ready for quantum execution
    OperationsReady(Vec<Operation>),
    /// LLVM execution completed successfully
    ExecutionComplete(OperationCollector),
    /// LLVM execution failed with an error
    ExecutionFailed(String),
}

/// Messages from the main thread to the worker thread
#[derive(Debug)]
pub enum MainMessage {
    /// Measurement results from quantum execution
    MeasurementResults(BTreeMap<usize, bool>),
    /// Signal to abort execution
    Abort,
}

/// State shared between the LLVM execution callback and the coordinator
struct SharedState {
    /// Operations collected so far
    pending_operations: Vec<Operation>,
    /// Measurement results provided by main thread
    measurement_results: BTreeMap<usize, bool>,
    /// Flag indicating worker is waiting for measurements
    waiting_for_measurements: bool,
    /// Flag indicating execution should abort
    should_abort: bool,
}

/// Handle for the worker thread to communicate with the main thread
///
/// This is passed to the quantum executor callback in pecos-qis-ffi
pub struct WorkerHandle {
    /// Sender to send messages to main thread
    to_main: Sender<WorkerMessage>,
    /// Receiver to get messages from main thread (reserved for future use)
    #[allow(dead_code)]
    from_main: Receiver<MainMessage>,
    /// Shared state for operation collection
    state: Arc<Mutex<SharedState>>,
    /// Condvar to wait for measurement results
    condvar: Arc<Condvar>,
}

impl WorkerHandle {
    /// Called when `___read_future_bool` needs a measurement result
    ///
    /// This function:
    /// 1. Takes the pending operations collected so far
    /// 2. Sends them to the main thread
    /// 3. Blocks waiting for measurement results
    /// 4. Returns the results
    ///
    /// # Panics
    /// Panics if the internal mutex is poisoned.
    #[must_use]
    pub fn request_measurements(&self) -> BTreeMap<usize, bool> {
        let ops = {
            let mut state = self.state.lock().unwrap();
            std::mem::take(&mut state.pending_operations)
        };

        debug!(
            "Worker requesting measurements, sending {} operations",
            ops.len()
        );

        // Send operations to main thread
        if self
            .to_main
            .send(WorkerMessage::OperationsReady(ops))
            .is_err()
        {
            // Main thread disconnected - return empty results
            debug!("Main thread disconnected, returning empty results");
            return BTreeMap::new();
        }

        // Wait for measurement results
        let mut state = self.state.lock().unwrap();
        state.waiting_for_measurements = true;

        while state.waiting_for_measurements && !state.should_abort {
            state = self.condvar.wait(state).unwrap();
        }

        if state.should_abort {
            debug!("Execution aborted");
            return BTreeMap::new();
        }

        let results = state.measurement_results.clone();
        debug!("Worker received {} measurement results", results.len());
        results
    }

    /// Store an operation (called by FFI functions)
    ///
    /// # Panics
    /// Panics if the internal mutex is poisoned.
    pub fn store_operation(&self, op: Operation) {
        let mut state = self.state.lock().unwrap();
        state.pending_operations.push(op);
    }

    /// Check if execution should abort
    ///
    /// # Panics
    /// Panics if the internal mutex is poisoned.
    #[must_use]
    pub fn should_abort(&self) -> bool {
        self.state.lock().unwrap().should_abort
    }
}

/// Handle for the main thread (`QisEngine`) to coordinate with the worker
pub struct MainHandle {
    /// Receiver to get messages from worker thread
    from_worker: Receiver<WorkerMessage>,
    /// Sender to send messages to worker thread
    to_worker: Sender<MainMessage>,
    /// Shared state for providing measurements
    state: Arc<Mutex<SharedState>>,
    /// Condvar to signal worker thread
    condvar: Arc<Condvar>,
    /// Worker thread handle
    worker_thread: Option<JoinHandle<()>>,
}

impl MainHandle {
    /// Wait for operations from the worker thread
    ///
    /// Returns `Some(operations)` if more operations are pending,
    /// or `None` if execution is complete.
    #[must_use]
    pub fn wait_for_operations(&self) -> Option<Vec<Operation>> {
        match self.from_worker.recv() {
            Ok(WorkerMessage::OperationsReady(ops)) => {
                debug!("Main received {} operations from worker", ops.len());
                Some(ops)
            }
            Ok(WorkerMessage::ExecutionComplete(collector)) => {
                debug!(
                    "Main received execution complete with {} total operations",
                    collector.operations.len()
                );
                None
            }
            Ok(WorkerMessage::ExecutionFailed(err)) => {
                log::error!("Worker execution failed: {err}");
                None
            }
            Err(_) => {
                debug!("Worker channel closed");
                None
            }
        }
    }

    /// Provide measurement results to the worker thread
    ///
    /// # Panics
    /// Panics if the internal mutex is poisoned.
    pub fn provide_measurements(&self, measurements: BTreeMap<usize, bool>) {
        debug!(
            "Main providing {} measurements to worker",
            measurements.len()
        );

        {
            let mut state = self.state.lock().unwrap();
            state.measurement_results = measurements.clone();
            state.waiting_for_measurements = false;
        }

        // Also send via channel for backup
        let _ = self
            .to_worker
            .send(MainMessage::MeasurementResults(measurements));

        // Signal the worker thread
        self.condvar.notify_one();
    }

    /// Signal the worker to abort execution
    ///
    /// # Panics
    /// Panics if the internal mutex is poisoned.
    pub fn abort(&self) {
        {
            let mut state = self.state.lock().unwrap();
            state.should_abort = true;
            state.waiting_for_measurements = false;
        }

        let _ = self.to_worker.send(MainMessage::Abort);
        self.condvar.notify_one();
    }

    /// Join the worker thread
    ///
    /// # Errors
    /// Returns an error if the worker thread panicked.
    pub fn join(mut self) -> Result<(), String> {
        if let Some(handle) = self.worker_thread.take() {
            handle
                .join()
                .map_err(|e| format!("Worker thread panicked: {e:?}"))
        } else {
            Ok(())
        }
    }

    /// Check if there are pending operations (non-blocking)
    #[must_use]
    pub fn try_recv_operations(&self) -> Option<Vec<Operation>> {
        match self.from_worker.try_recv() {
            Ok(WorkerMessage::OperationsReady(ops)) => Some(ops),
            Ok(WorkerMessage::ExecutionComplete(_) | WorkerMessage::ExecutionFailed(_))
            | Err(_) => None,
        }
    }
}

impl Drop for MainHandle {
    fn drop(&mut self) {
        // Signal abort and wait for worker
        self.abort();
        if let Some(handle) = self.worker_thread.take() {
            let _ = handle.join();
        }
    }
}

/// Create a pair of handles for coordinating dynamic execution
///
/// Returns (`worker_handle`, `main_handle`) where:
/// - `worker_handle` should be used in the LLVM execution thread
/// - `main_handle` should be used by `QisEngine`
#[must_use]
pub fn create_coordinator() -> (WorkerHandle, MainHandle) {
    let (to_main_tx, to_main_rx) = mpsc::channel();
    let (to_worker_tx, to_worker_rx) = mpsc::channel();

    let state = Arc::new(Mutex::new(SharedState {
        pending_operations: Vec::new(),
        measurement_results: BTreeMap::new(),
        waiting_for_measurements: false,
        should_abort: false,
    }));
    let condvar = Arc::new(Condvar::new());

    let worker_handle = WorkerHandle {
        to_main: to_main_tx,
        from_main: to_worker_rx,
        state: Arc::clone(&state),
        condvar: Arc::clone(&condvar),
    };

    let main_handle = MainHandle {
        from_worker: to_main_rx,
        to_worker: to_worker_tx,
        state,
        condvar,
        worker_thread: None,
    };

    (worker_handle, main_handle)
}

/// Start dynamic LLVM execution on a worker thread
///
/// This function:
/// 1. Creates the coordinator handles
/// 2. Spawns a worker thread that runs the LLVM execution
/// 3. Returns the main handle for `QisEngine` to use
///
/// The `execute_fn` should call the interface's `collect_operations()` or similar,
/// using the `WorkerHandle` for measurement coordination.
pub fn start_dynamic_execution<F>(execute_fn: F) -> MainHandle
where
    F: FnOnce(WorkerHandle) -> Result<OperationCollector, String> + Send + 'static,
{
    let (worker_handle, mut main_handle) = create_coordinator();
    let to_main = worker_handle.to_main.clone();

    let worker_thread = thread::spawn(move || match execute_fn(worker_handle) {
        Ok(collector) => {
            let _ = to_main.send(WorkerMessage::ExecutionComplete(collector));
        }
        Err(err) => {
            let _ = to_main.send(WorkerMessage::ExecutionFailed(err));
        }
    });

    main_handle.worker_thread = Some(worker_thread);
    main_handle
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::time::Duration;

    #[test]
    fn test_coordinator_creation() {
        let (worker, _main) = create_coordinator();

        // Worker can store operations
        worker.store_operation(Operation::AllocateQubit { id: 0 });

        // Check state
        let state = worker.state.lock().unwrap();
        assert_eq!(state.pending_operations.len(), 1);
    }

    #[test]
    fn test_store_multiple_operations() {
        let (worker, _main) = create_coordinator();

        worker.store_operation(Operation::AllocateQubit { id: 0 });
        worker.store_operation(Operation::AllocateQubit { id: 1 });
        worker.store_operation(Operation::ReleaseQubit { id: 0 });

        let state = worker.state.lock().unwrap();
        assert_eq!(state.pending_operations.len(), 3);
    }

    #[test]
    fn test_should_abort_initially_false() {
        let (worker, _main) = create_coordinator();
        assert!(!worker.should_abort());
    }

    #[test]
    fn test_abort_sets_flag() {
        let (worker, main) = create_coordinator();

        main.abort();

        assert!(worker.should_abort());
    }

    #[test]
    fn test_try_recv_operations_empty() {
        let (_worker, main) = create_coordinator();

        // No operations sent, should return None
        let result = main.try_recv_operations();
        assert!(result.is_none());
    }

    #[test]
    fn test_provide_measurements() {
        let (worker, main) = create_coordinator();

        let mut measurements = BTreeMap::new();
        measurements.insert(0, true);
        measurements.insert(1, false);

        main.provide_measurements(measurements.clone());

        let state = worker.state.lock().unwrap();
        assert_eq!(state.measurement_results.get(&0), Some(&true));
        assert_eq!(state.measurement_results.get(&1), Some(&false));
        assert!(!state.waiting_for_measurements);
    }

    #[test]
    fn test_full_measurement_request_cycle() {
        let barrier = Arc::new(Barrier::new(2));
        let worker_barrier = Arc::clone(&barrier);

        let main = start_dynamic_execution(move |worker| {
            // Store some operations
            worker.store_operation(Operation::AllocateQubit { id: 0 });
            worker.store_operation(Operation::AllocateQubit { id: 1 });

            // Wait for main thread
            worker_barrier.wait();

            // Request measurements (this will block until main provides them)
            let results = worker.request_measurements();

            assert_eq!(results.get(&0), Some(&true));
            assert_eq!(results.get(&1), Some(&false));

            Ok(OperationCollector::new())
        });

        // Sync with worker
        barrier.wait();

        // Wait for worker to send operations
        if let Some(ops) = main.wait_for_operations() {
            assert_eq!(ops.len(), 2);

            // Provide measurements
            let mut measurements = BTreeMap::new();
            measurements.insert(0, true);
            measurements.insert(1, false);
            main.provide_measurements(measurements);
        }

        // Wait for completion
        main.join().unwrap();
    }

    #[test]
    fn test_start_dynamic_execution_success() {
        let main = start_dynamic_execution(|_worker| {
            let mut collector = OperationCollector::new();
            collector
                .operations
                .push(Operation::AllocateQubit { id: 0 });
            Ok(collector)
        });

        // Wait for completion - should receive ExecutionComplete
        let result = main.wait_for_operations();
        assert!(result.is_none()); // None means complete, not operations ready

        main.join().unwrap();
    }

    #[test]
    fn test_start_dynamic_execution_error() {
        let main = start_dynamic_execution(|_worker| Err("Test error".to_string()));

        // Wait for completion - should receive ExecutionFailed
        let result = main.wait_for_operations();
        assert!(result.is_none());

        main.join().unwrap();
    }

    #[test]
    fn test_abort_wakes_waiting_worker() {
        let barrier = Arc::new(Barrier::new(2));
        let worker_barrier = Arc::clone(&barrier);

        let main = start_dynamic_execution(move |worker| {
            worker_barrier.wait();

            // Request measurements - this will block
            let results = worker.request_measurements();

            // Should get empty results because of abort
            assert!(results.is_empty());

            Ok(OperationCollector::new())
        });

        barrier.wait();

        // Give worker time to start waiting
        thread::sleep(Duration::from_millis(10));

        // Abort
        main.abort();

        // Join should succeed
        main.join().unwrap();
    }

    #[test]
    fn test_worker_message_debug() {
        let msg = WorkerMessage::OperationsReady(vec![Operation::AllocateQubit { id: 0 }]);
        let debug_str = format!("{msg:?}");
        assert!(debug_str.contains("OperationsReady"));

        let msg = WorkerMessage::ExecutionComplete(OperationCollector::new());
        let debug_str = format!("{msg:?}");
        assert!(debug_str.contains("ExecutionComplete"));

        let msg = WorkerMessage::ExecutionFailed("error".to_string());
        let debug_str = format!("{msg:?}");
        assert!(debug_str.contains("ExecutionFailed"));
    }

    #[test]
    fn test_main_message_debug() {
        let msg = MainMessage::MeasurementResults(BTreeMap::new());
        let debug_str = format!("{msg:?}");
        assert!(debug_str.contains("MeasurementResults"));

        let msg = MainMessage::Abort;
        let debug_str = format!("{msg:?}");
        assert!(debug_str.contains("Abort"));
    }

    #[test]
    fn test_drop_aborts_worker() {
        let barrier = Arc::new(Barrier::new(2));
        let worker_barrier = Arc::clone(&barrier);
        let completed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let completed_clone = Arc::clone(&completed);

        {
            let _main = start_dynamic_execution(move |worker| {
                worker_barrier.wait();

                // Check should_abort in a loop
                for _ in 0..100 {
                    if worker.should_abort() {
                        completed_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                        return Ok(OperationCollector::new());
                    }
                    thread::sleep(Duration::from_millis(10));
                }

                Ok(OperationCollector::new())
            });

            barrier.wait();
            // main is dropped here, which should call abort
        }

        // Worker should have seen the abort
        thread::sleep(Duration::from_millis(50));
        assert!(completed.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn test_join_without_worker_thread() {
        let (_worker, main) = create_coordinator();
        // main.worker_thread is None
        assert!(main.join().is_ok());
    }

    // Edge case tests for nested conditionals

    #[test]
    fn test_multiple_measurement_requests() {
        // Simulates nested conditionals where multiple measurement results are needed
        let barrier = Arc::new(Barrier::new(2));
        let worker_barrier = Arc::clone(&barrier);

        let main = start_dynamic_execution(move |worker| {
            // First round of operations
            worker.store_operation(Operation::AllocateQubit { id: 0 });

            worker_barrier.wait();

            // First measurement request
            let results1 = worker.request_measurements();
            assert_eq!(results1.get(&0), Some(&true));

            // Second round of operations (based on first measurement)
            worker.store_operation(Operation::AllocateQubit { id: 1 });

            // Second measurement request
            let results2 = worker.request_measurements();
            assert_eq!(results2.get(&1), Some(&false));

            // Third round based on both measurements
            worker.store_operation(Operation::ReleaseQubit { id: 0 });
            worker.store_operation(Operation::ReleaseQubit { id: 1 });

            Ok(OperationCollector::new())
        });

        barrier.wait();

        // First round
        if let Some(ops) = main.wait_for_operations() {
            assert_eq!(ops.len(), 1);
            let mut measurements = BTreeMap::new();
            measurements.insert(0, true);
            main.provide_measurements(measurements);
        }

        // Second round
        if let Some(ops) = main.wait_for_operations() {
            assert_eq!(ops.len(), 1);
            let mut measurements = BTreeMap::new();
            measurements.insert(1, false);
            main.provide_measurements(measurements);
        }

        main.join().unwrap();
    }

    #[test]
    fn test_operations_accumulated_between_requests() {
        // Verifies that operations are properly accumulated between measurement requests
        let barrier = Arc::new(Barrier::new(2));
        let worker_barrier = Arc::clone(&barrier);

        let main = start_dynamic_execution(move |worker| {
            // Store multiple operations before first request
            worker.store_operation(Operation::AllocateQubit { id: 0 });
            worker.store_operation(Operation::AllocateQubit { id: 1 });
            worker.store_operation(Operation::AllocateQubit { id: 2 });

            worker_barrier.wait();

            let _ = worker.request_measurements();

            // Store more operations after first result
            worker.store_operation(Operation::ReleaseQubit { id: 0 });
            worker.store_operation(Operation::ReleaseQubit { id: 1 });

            let _ = worker.request_measurements();

            Ok(OperationCollector::new())
        });

        barrier.wait();

        // First batch should have 3 operations
        if let Some(ops) = main.wait_for_operations() {
            assert_eq!(ops.len(), 3);
            main.provide_measurements(BTreeMap::new());
        }

        // Second batch should have 2 operations
        if let Some(ops) = main.wait_for_operations() {
            assert_eq!(ops.len(), 2);
            main.provide_measurements(BTreeMap::new());
        }

        main.join().unwrap();
    }

    #[test]
    fn test_main_disconnect_during_request() {
        // Test that worker handles main thread disconnecting gracefully
        let barrier = Arc::new(Barrier::new(2));
        let worker_barrier = Arc::clone(&barrier);
        let got_empty_results = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let got_empty_results_clone = Arc::clone(&got_empty_results);

        let (worker, main_handle) = create_coordinator();

        // Spawn worker manually to control main handle lifetime
        let worker_thread = thread::spawn(move || {
            worker_barrier.wait();

            // This request should return empty because main disconnected
            let results = worker.request_measurements();
            if results.is_empty() {
                got_empty_results_clone.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        });

        barrier.wait();

        // Drop main handle (simulates disconnect)
        drop(main_handle);

        worker_thread.join().unwrap();

        assert!(got_empty_results.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn test_empty_operations_request() {
        // Test requesting measurements with no pending operations
        let barrier = Arc::new(Barrier::new(2));
        let worker_barrier = Arc::clone(&barrier);

        let main = start_dynamic_execution(move |worker| {
            worker_barrier.wait();

            // Request measurements with no operations stored
            let results = worker.request_measurements();
            assert!(results.is_empty()); // Main provides empty results

            Ok(OperationCollector::new())
        });

        barrier.wait();

        // Should receive empty operations list
        if let Some(ops) = main.wait_for_operations() {
            assert!(ops.is_empty());
            main.provide_measurements(BTreeMap::new());
        }

        main.join().unwrap();
    }

    #[test]
    fn test_measurement_results_cleared_between_requests() {
        // Verify that old measurement results don't leak into new requests
        let barrier = Arc::new(Barrier::new(2));
        let worker_barrier = Arc::clone(&barrier);

        let main = start_dynamic_execution(move |worker| {
            worker.store_operation(Operation::AllocateQubit { id: 0 });
            worker_barrier.wait();

            let results1 = worker.request_measurements();
            assert_eq!(results1.get(&0), Some(&true));
            assert_eq!(results1.get(&1), None); // Result 1 not provided in first round

            worker.store_operation(Operation::AllocateQubit { id: 1 });

            let results2 = worker.request_measurements();
            // Should only have result 1, not result 0 from previous round
            assert_eq!(results2.get(&1), Some(&false));
            // Result 0 from previous round should be replaced
            assert_eq!(results2.get(&0), None);

            Ok(OperationCollector::new())
        });

        barrier.wait();

        // First round - only provide result 0
        if main.wait_for_operations().is_some() {
            let mut measurements = BTreeMap::new();
            measurements.insert(0, true);
            main.provide_measurements(measurements);
        }

        // Second round - only provide result 1
        if main.wait_for_operations().is_some() {
            let mut measurements = BTreeMap::new();
            measurements.insert(1, false);
            main.provide_measurements(measurements);
        }

        main.join().unwrap();
    }
}
