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

struct BlockingInterface {
    release: Mutex<Receiver<()>>,
    entered: Sender<()>,
    panic: bool,
}

impl crate::QisInterface for BlockingInterface {
    fn load_program(&mut self, _: &[u8], _: ProgramFormat) -> Result<(), InterfaceError> {
        Ok(())
    }
    fn collect_operations(&mut self) -> Result<OperationList, InterfaceError> {
        self.entered.send(()).unwrap();
        self.release.lock().unwrap().recv().unwrap();
        assert!(!self.panic, "injected worker panic");
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

fn blocked_worker(panic: bool) -> (PersistentDynamicWorker, Sender<()>, Arc<WorkerCounts>) {
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

#[test]
fn drop_bounds_unfinished_worker_and_warns() {
    capture_warnings();
    let (worker, release, counts) = blocked_worker(false);
    let bound = worker.drop_timeout;
    let started = Instant::now();
    drop(worker);
    let elapsed = started.elapsed();
    assert!(elapsed >= bound);
    assert!(elapsed < Duration::from_secs(1));
    assert_eq!(counts.live.load(Ordering::SeqCst), 1);
    assert_eq!(counts.joined.load(Ordering::SeqCst), 0);
    let warning = warnings();
    assert!(warning.contains("did not finish within 20ms; detaching"));
    assert!(!warning.contains("noncooperative"));
    // Release the deliberately detached fixture so it cannot leak into later tests.
    release.send(()).unwrap();
    wait_for_exit(&counts);
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
    capture_warnings();
    let (mut worker, release, counts) = blocked_worker(true);
    worker.drop_timeout = Duration::from_secs(2);
    release.send(()).unwrap();
    drop(worker);
    counts.assert_joined();
    assert!(warnings().contains("Dynamic worker panicked"));
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
    let (worker, release, counts) = blocked_worker(false);
    engine.persistent_worker = Some(worker);
    fail.store(true, Ordering::SeqCst);
    let started = Instant::now();
    drop(engine);
    assert!(started.elapsed() < Duration::from_secs(1));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(counts.live.load(Ordering::SeqCst), 1);
    assert!(
        warnings().contains("did not finish within 20ms; detaching; abort failed: abort failed")
    );
    release.send(()).unwrap();
    wait_for_exit(&counts);
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
