// Thread-exit witnesses are scoped to an engine and its clones, so assertions
// remain valid when unrelated tests create workers in the same process.
use super::*;
use std::cell::RefCell;
use std::sync::Once;

#[derive(Default)]
pub(crate) struct WorkerCounts {
    pub(crate) live: AtomicU64,
    pub(crate) started: AtomicU64,
    pub(crate) joined: AtomicU64,
    pub(crate) returned: AtomicU64,
}

impl WorkerCounts {
    pub(crate) fn assert_joined(&self) {
        assert_eq!(self.live.load(Ordering::SeqCst), 0, "worker still live");
        let started = self.started.load(Ordering::SeqCst);
        assert!(started > 0, "test must create a worker");
        assert_eq!(self.joined.load(Ordering::SeqCst), started);
    }
}

pub(crate) struct WorkerExitGuard(Arc<WorkerCounts>);

impl WorkerExitGuard {
    pub(crate) fn new(counts: Arc<WorkerCounts>) -> Self {
        counts.live.fetch_add(1, Ordering::SeqCst);
        counts.started.fetch_add(1, Ordering::SeqCst);
        Self(counts)
    }
}

impl Drop for WorkerExitGuard {
    fn drop(&mut self) {
        self.0.live.fetch_sub(1, Ordering::SeqCst);
    }
}

thread_local! {
    static WARNINGS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

struct WarningLogger;
impl log::Log for WarningLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() == log::Level::Warn
    }
    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            WARNINGS.with_borrow_mut(|warnings| warnings.push(record.args().to_string()));
        }
    }
    fn flush(&self) {}
}

fn capture_warnings() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        log::set_logger(&WarningLogger).unwrap();
        log::set_max_level(log::LevelFilter::Warn);
    });
    WARNINGS.with_borrow_mut(Vec::clear);
}

fn warnings() -> String {
    WARNINGS.with_borrow(|warnings| warnings.join("\n"))
}

#[derive(Clone, Copy)]
enum WorkerPanic {
    Borrowed,
    Owned,
    NonString,
}

struct BlockingInterface {
    release: Mutex<Receiver<()>>,
    entered: Sender<()>,
    panic: Option<WorkerPanic>,
}

impl crate::QisInterface for BlockingInterface {
    fn load_program(&mut self, _: &[u8], _: ProgramFormat) -> Result<(), InterfaceError> {
        Ok(())
    }
    fn collect_operations(&mut self) -> Result<OperationList, InterfaceError> {
        self.entered.send(()).unwrap();
        self.release.lock().unwrap().recv().unwrap();
        match self.panic {
            Some(WorkerPanic::Borrowed) => panic!("injected worker panic"),
            Some(WorkerPanic::Owned) => std::panic::panic_any(String::from("owned worker panic")),
            Some(WorkerPanic::NonString) => std::panic::panic_any(42_u8),
            None => {}
        }
        Ok(OperationList::new())
    }
    fn execute_with_measurements(
        &mut self,
        _: BTreeMap<usize, bool>,
    ) -> Result<OperationList, InterfaceError> {
        self.collect_operations()
    }
    fn name(&self) -> &'static str {
        "blocking drop fixture"
    }
    fn reset(&mut self) -> Result<(), InterfaceError> {
        panic!("drop must not reset the interface")
    }
    fn disable_dynamic_mode(&mut self) -> Result<(), InterfaceError> {
        Ok(())
    }
}

fn blocked_worker(
    panic: Option<WorkerPanic>,
) -> (PersistentDynamicWorker, Sender<()>, Arc<WorkerCounts>) {
    let counts = Arc::new(WorkerCounts::default());
    let mut worker = PersistentDynamicWorker::new(Arc::clone(&counts));
    worker.drop_timeout = Duration::from_millis(20);
    let (release, receiver) = mpsc::channel();
    let (entered, entry) = mpsc::channel();
    worker
        .execute(Box::new(BlockingInterface {
            release: Mutex::new(receiver),
            entered,
            panic,
        }))
        .unwrap();
    entry.recv_timeout(Duration::from_secs(2)).unwrap();
    (worker, release, counts)
}

fn release_later(release: Sender<()>) -> JoinHandle<()> {
    std::thread::spawn(move || {
        // Even an unconditional join must return, so deadline regressions fail
        // the elapsed-time assertion instead of hanging the test process.
        std::thread::sleep(Duration::from_millis(1_500));
        release.send(()).unwrap();
    })
}

#[test]
fn drop_bounds_unfinished_worker_and_warns() {
    capture_warnings();
    let (worker, release, counts) = blocked_worker(None);
    let bound = worker.drop_timeout;
    let release_thread = release_later(release);
    let started = Instant::now();
    drop(worker);
    let elapsed = started.elapsed();
    let live_at_drop = counts.live.load(Ordering::SeqCst);
    release_thread.join().unwrap();
    wait_for_exit(&counts);
    assert!(elapsed >= bound);
    assert!(elapsed < Duration::from_secs(1));
    assert_eq!(live_at_drop, 1);
    assert_eq!(counts.joined.load(Ordering::SeqCst), 0);
    let warning = warnings();
    assert!(warning.contains("did not finish within 20ms; detaching"));
    assert!(!warning.contains("noncooperative"));
}

fn wait_for_exit(counts: &WorkerCounts) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while counts.live.load(Ordering::SeqCst) != 0 {
        assert!(Instant::now() < deadline, "worker did not exit");
        std::thread::yield_now();
    }
}

#[test]
fn drop_joins_panicked_worker_and_warns() {
    for (panic, expected) in [
        (
            WorkerPanic::Borrowed,
            "Dynamic worker panicked: injected worker panic",
        ),
        (
            WorkerPanic::Owned,
            "Dynamic worker panicked: owned worker panic",
        ),
        (
            WorkerPanic::NonString,
            "Dynamic worker panicked with a non-string payload",
        ),
    ] {
        capture_warnings();
        let (worker, release, counts) = blocked_worker(Some(panic));
        release.send(()).unwrap();
        // Backtrace resolution holds a process-wide lock and can be slow under
        // CI load. This test exercises joining a panic, not a shutdown deadline.
        let deadline = Instant::now() + Duration::from_secs(60);
        while !worker.handle.as_ref().unwrap().is_finished() {
            assert!(Instant::now() < deadline, "panicked worker did not exit");
            std::thread::sleep(Duration::from_millis(1));
        }
        drop(worker);
        counts.assert_joined();
        assert!(warnings().contains(expected), "{}", warnings());
    }
}

#[test]
fn drop_after_abort_failure_still_closes_work_channel() {
    capture_warnings();
    let (engine, _result_sender, fail, calls) = reset_tests::running();
    let counts = Arc::clone(&engine.worker_counts);
    fail.store(true, Ordering::SeqCst);
    drop(engine);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    counts.assert_joined();
    assert!(
        warnings().contains("Failed to abort dynamic execution during engine drop: abort failed")
    );
}

#[test]
fn drop_after_abort_failure_is_bounded_and_reports_failure_on_detach() {
    capture_warnings();
    let (mut engine, _result_sender, fail, calls) = reset_tests::running();
    let (worker, release, counts) = blocked_worker(None);
    engine.persistent_worker = Some(worker);
    fail.store(true, Ordering::SeqCst);
    let release_thread = release_later(release);
    let started = Instant::now();
    drop(engine);
    let elapsed = started.elapsed();
    let live_at_drop = counts.live.load(Ordering::SeqCst);
    release_thread.join().unwrap();
    wait_for_exit(&counts);
    assert!(elapsed < Duration::from_secs(1));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(live_at_drop, 1);
    assert!(
        warnings().contains("did not finish within 20ms; detaching; abort failed: abort failed")
    );
}

#[test]
fn drop_completed_shot_does_not_abort() {
    let (mut engine, _result_sender, fail, calls) = reset_tests::running();
    let counts = Arc::clone(&engine.worker_counts);
    engine.dynamic_state.as_mut().unwrap().execution_complete = true;
    fail.store(true, Ordering::SeqCst);
    drop(engine);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    counts.assert_joined();
}

#[derive(Clone, Default)]
struct ResetCountingRuntime {
    state: ClassicalState,
    resets: Arc<AtomicU64>,
}

impl QisRuntime for ResetCountingRuntime {
    fn load_interface(&mut self, _: OperationList) -> RuntimeResult<()> {
        Ok(())
    }
    fn execute_until_quantum(&mut self) -> RuntimeResult<Option<Vec<QuantumOp>>> {
        Ok(None)
    }
    fn provide_measurements(&mut self, _: BTreeMap<usize, bool>) -> RuntimeResult<()> {
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
    fn reset(&mut self) -> RuntimeResult<()> {
        self.resets.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[derive(Default)]
struct ResetCountingInterface {
    resets: Arc<AtomicU64>,
    exits: Arc<AtomicU64>,
}

impl crate::QisInterface for ResetCountingInterface {
    fn load_program(&mut self, _: &[u8], _: ProgramFormat) -> Result<(), InterfaceError> {
        Ok(())
    }
    fn collect_operations(&mut self) -> Result<OperationList, InterfaceError> {
        Ok(OperationList::new())
    }
    fn execute_with_measurements(
        &mut self,
        _: BTreeMap<usize, bool>,
    ) -> Result<OperationList, InterfaceError> {
        self.collect_operations()
    }
    fn name(&self) -> &'static str {
        "reset counting drop fixture"
    }
    fn reset(&mut self) -> Result<(), InterfaceError> {
        self.resets.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn disable_dynamic_mode(&mut self) -> Result<(), InterfaceError> {
        self.exits.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[test]
fn drop_never_resets_host_runtime_or_interface() {
    let runtime = ResetCountingRuntime::default();
    let interface = ResetCountingInterface::default();
    let runtime_resets = Arc::clone(&runtime.resets);
    let interface_resets = Arc::clone(&interface.resets);
    let interface_exits = Arc::clone(&interface.exits);
    // Both objects stay on the engine, so a reset during drop reaches them.
    let engine = QisEngine::new(Box::new(interface), Box::new(runtime));
    drop(engine);
    assert_eq!(runtime_resets.load(Ordering::SeqCst), 0);
    assert_eq!(interface_resets.load(Ordering::SeqCst), 0);
    assert_eq!(interface_exits.load(Ordering::SeqCst), 0);
}
