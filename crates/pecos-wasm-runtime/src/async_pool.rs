//! A per-execution-thread pool. Jobs own all WASM state they access.

use std::cell::Cell;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::thread::{self, JoinHandle};

pub(crate) const MAX_WORKERS: usize = 64;
pub(crate) type Job = Box<dyn FnOnce() + Send + 'static>;

#[derive(Default)]
struct Progress {
    // Includes queued, dequeued, running, and not-yet-dropped jobs.
    outstanding: usize,
    cancelling: bool,
}

#[derive(Default)]
pub(crate) struct Control {
    progress: Mutex<Progress>,
    changed: Condvar,
    #[cfg(test)]
    before_claim: Mutex<Option<Job>>,
}

impl Control {
    fn done(&self) {
        let mut progress = self
            .progress
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        progress.outstanding = progress
            .outstanding
            .checked_sub(1)
            .expect("WASM async job accounting: every finished job was submitted");
        self.changed.notify_all();
    }

    #[cfg(test)]
    pub(crate) fn wait_for_cancel(&self) {
        let progress = self.progress.lock().unwrap();
        drop(
            self.changed
                .wait_while(progress, |p| !p.cancelling)
                .unwrap(),
        );
    }
}

// A recoverable panic in a job or in its captures must not kill a worker. Even
// the panic payload can have a panicking destructor. Attempt its drop under
// protection; forget a secondary payload to prevent an unbounded chain of
// destructor panics. A destructor that panics while a panic is already unwinding
// still aborts the process; catch_unwind cannot recover from that.
fn protected(operation: impl FnOnce()) {
    if let Err(payload) = catch_unwind(AssertUnwindSafe(operation))
        && let Err(secondary) = catch_unwind(AssertUnwindSafe(|| drop(payload)))
    {
        std::mem::forget(secondary);
    }
}

pub(crate) struct Pool {
    sender: Option<mpsc::Sender<Job>>,
    workers: Vec<JoinHandle<()>>,
    control: Arc<Control>,
    // Cancel's guarantees hold only if the owner alone submits and cancels:
    // !Sync keeps a pool from being shared, while Send still lets it be dropped elsewhere.
    _owner: PhantomData<Cell<()>>,
}

impl Pool {
    pub(crate) fn new(count: usize) -> Result<Self, String> {
        Self::with_spawner(count, |job| {
            thread::Builder::new()
                .name("pecos-wasm-call".into())
                .spawn(job)
        })
    }

    fn with_spawner(
        count: usize,
        mut spawn: impl FnMut(Job) -> std::io::Result<JoinHandle<()>>,
    ) -> Result<Self, String> {
        if !(1..=MAX_WORKERS).contains(&count) {
            return Err(format!(
                "WASM async worker count must be between 1 and {MAX_WORKERS}; got {count}"
            ));
        }
        let (sender, receiver) = mpsc::channel::<Job>();
        let receiver = Arc::new(Mutex::new(receiver));
        let mut pool = Self {
            sender: Some(sender),
            workers: Vec::new(),
            control: Arc::default(),
            _owner: PhantomData,
        };
        for _ in 0..count {
            let receiver = Arc::clone(&receiver);
            let control = Arc::clone(&pool.control);
            let worker = spawn(Box::new(move || {
                loop {
                    // Never hold the queue lock while running or discarding a job.
                    let job = {
                        let receiver = receiver
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        receiver.recv()
                    };
                    let Ok(job) = job else { break };
                    #[cfg(test)]
                    {
                        let hook = control.before_claim.lock().unwrap().take();
                        if let Some(hook) = hook {
                            hook();
                        }
                    }
                    // Claim and cancel use the same lock. A dequeued job is still
                    // unclaimed until this decision, and remains counted until
                    // execution/discard and panic-payload cleanup have finished.
                    let claimed = !control
                        .progress
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .cancelling;
                    protected(|| {
                        if claimed {
                            job();
                        } else {
                            drop(job);
                        }
                    });
                    control.done();
                }
            }))
            .map_err(|error| format!("Cannot start WASM async worker: {error}"))?;
            pool.workers.push(worker);
        }
        Ok(pool)
    }

    pub(crate) fn size(&self) -> usize {
        self.workers.len()
    }

    pub(crate) fn submit(&self, job: Job) -> Result<(), String> {
        let sender = self
            .sender
            .as_ref()
            .ok_or("WASM async executor is closed")?;
        {
            let mut progress = self
                .control
                .progress
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            progress.outstanding += 1;
        }
        if let Err(error) = sender.send(job) {
            protected(|| drop(error.0));
            self.control.done();
            return Err("WASM async executor disconnected".into());
        }
        Ok(())
    }

    /// Discard every unclaimed job and wait for all prior jobs' captures to be
    /// released. Claimed jobs finish normally. Only the owner submits/cancels.
    /// WASM execution has per-phase deadlines; arbitrary Rust blocking is unbounded.
    /// Completed replies belong to the caller, not to this pool.
    pub(crate) fn cancel(&self) {
        let mut progress = self
            .control
            .progress
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        progress.cancelling = true;
        // No production code waits for this transition; tests observe it.
        self.control.changed.notify_all();
        progress = self
            .control
            .changed
            .wait_while(progress, |p| p.outstanding != 0)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        progress.cancelling = false;
    }

    #[cfg(test)]
    pub(crate) fn control(&self) -> Arc<Control> {
        Arc::clone(&self.control)
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.cancel();
        // Close before joining: idle workers must be able to leave recv().
        drop(self.sender.take());
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

enum Slot<T> {
    Waiting,
    Sent(T),
    Closed,
}

struct Reply<T> {
    slot: Mutex<Slot<T>>,
    ready: Condvar,
}

impl<T> Reply<T> {
    fn lock(&self) -> std::sync::MutexGuard<'_, Slot<T>> {
        self.slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Completes a [`ReplyReceiver`] once; dropping it unsent disconnects the receiver.
pub(crate) struct ReplySender<T>(Arc<Reply<T>>);

/// Waits for one value from its [`ReplySender`].
pub(crate) struct ReplyReceiver<T>(Arc<Reply<T>>);

/// A one-shot reply channel that blocks on a `Mutex` and `Condvar`, never on
/// thread parking. On Apple targets std parks threads (as `mpsc::Receiver::recv`
/// does) on libdispatch semaphores backed by Mach ports, which a forked child
/// does not inherit: the forking thread crashes if it parked before the fork and
/// parks again in the child. Apple's `Mutex` and `Condvar` are pthread objects
/// and keep working in the child. The job queue can stay an `mpsc` channel: only
/// workers wait on it, and a child never reuses an inherited worker.
pub(crate) fn reply<T>() -> (ReplySender<T>, ReplyReceiver<T>) {
    let reply = Arc::new(Reply {
        slot: Mutex::new(Slot::Waiting),
        ready: Condvar::new(),
    });
    (ReplySender(Arc::clone(&reply)), ReplyReceiver(reply))
}

impl<T> ReplySender<T> {
    /// Never blocks, whether or not the receiver still exists.
    pub(crate) fn send(self, value: T) {
        *self.0.lock() = Slot::Sent(value);
        self.0.ready.notify_one();
    }
}

impl<T> Drop for ReplySender<T> {
    fn drop(&mut self) {
        let mut slot = self.0.lock();
        if matches!(*slot, Slot::Waiting) {
            *slot = Slot::Closed;
            self.0.ready.notify_one();
        }
    }
}

impl<T> ReplyReceiver<T> {
    /// Returns `None` if the sender was dropped without sending.
    pub(crate) fn recv(self) -> Option<T> {
        let slot = self.0.lock();
        let mut slot = self
            .0
            .ready
            .wait_while(slot, |slot| matches!(slot, Slot::Waiting))
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match std::mem::replace(&mut *slot, Slot::Closed) {
            Slot::Sent(value) => Some(value),
            Slot::Waiting | Slot::Closed => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::watchdog;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    #[test]
    fn overlapping_jobs_release_queue_lock() {
        if !watchdog("async_pool::tests::overlapping_jobs_release_queue_lock") {
            return;
        }
        let pool = Pool::new(2).unwrap();
        let (started, starts) = mpsc::channel();
        let mut releases = Vec::new();
        for _ in 0..2 {
            let started = started.clone();
            let (release, wait) = mpsc::channel();
            releases.push(release);
            pool.submit(Box::new(move || {
                started.send(()).unwrap();
                wait.recv().unwrap();
            }))
            .unwrap();
        }
        starts.recv_timeout(Duration::from_secs(5)).unwrap();
        starts.recv_timeout(Duration::from_secs(5)).unwrap();
        for release in releases {
            release.send(()).unwrap();
        }
        drop(pool);
    }

    #[test]
    fn shutdown_discards_queue_and_joins_running_job() {
        if !watchdog("async_pool::tests::shutdown_discards_queue_and_joins_running_job") {
            return;
        }
        let pool = Pool::new(1).unwrap();
        let control = pool.control();
        let (started, start) = mpsc::channel();
        let (release, wait) = mpsc::channel();
        let (reply, abandoned) = mpsc::channel();
        drop(abandoned);
        pool.submit(Box::new(move || {
            started.send(()).unwrap();
            wait.recv().unwrap();
            let _ = reply.send(());
        }))
        .unwrap();
        start.recv().unwrap();
        let executed = Arc::new(AtomicBool::new(false));
        let queued = Arc::clone(&executed);
        let (reply, receiver) = mpsc::channel::<()>();
        pool.submit(Box::new(move || {
            queued.store(true, Ordering::Release);
            let _ = reply.send(());
        }))
        .unwrap();
        let cleanup = thread::spawn(move || drop(pool));
        control.wait_for_cancel();
        release.send(()).unwrap();
        cleanup.join().unwrap();
        assert!(!executed.load(Ordering::Acquire));
        assert!(receiver.recv().is_err());
    }

    #[test]
    fn panic_disconnects_reply_without_losing_worker() {
        if !watchdog("async_pool::tests::panic_disconnects_reply_without_losing_worker") {
            return;
        }
        let pool = Pool::new(1).unwrap();
        let (sender, receiver) = reply::<()>();
        pool.submit(Box::new(move || {
            let _sender = sender;
            panic!("injected job panic");
        }))
        .unwrap();
        assert!(receiver.recv().is_none());
        let (sender, receiver) = reply();
        pool.submit(Box::new(move || sender.send(42))).unwrap();
        assert_eq!(receiver.recv(), Some(42));
    }

    #[test]
    fn reply_wakes_a_waiting_receiver() {
        if !watchdog("async_pool::tests::reply_wakes_a_waiting_receiver") {
            return;
        }
        for send in [true, false] {
            let (sender, receiver) = reply();
            let completer = thread::spawn(move || {
                // Let the receiver reach its wait before completing.
                thread::sleep(Duration::from_millis(50));
                if send {
                    sender.send(7);
                } else {
                    drop(sender);
                }
            });
            assert_eq!(receiver.recv(), send.then_some(7));
            completer.join().unwrap();
        }
        let (sender, receiver) = reply();
        sender.send(7);
        assert_eq!(receiver.recv(), Some(7));
        let (sender, receiver) = reply::<i32>();
        drop(sender);
        assert_eq!(receiver.recv(), None);
        let (sender, receiver) = reply();
        drop(receiver);
        sender.send(7);
    }

    #[test]
    fn partial_startup_failure_joins_started_workers() {
        if !watchdog("async_pool::tests::partial_startup_failure_joins_started_workers") {
            return;
        }
        let exited = Arc::new(AtomicBool::new(false));
        let mut started = 0;
        let result = Pool::with_spawner(2, |job| {
            started += 1;
            if started == 2 {
                return Err(std::io::Error::other("injected spawn failure"));
            }
            let exited = Arc::clone(&exited);
            thread::Builder::new().spawn(move || {
                job();
                exited.store(true, Ordering::Release);
            })
        });
        assert!(result.is_err());
        assert!(exited.load(Ordering::Acquire));
        assert_eq!(
            Pool::new(0).err().unwrap(),
            "WASM async worker count must be between 1 and 64; got 0"
        );
        assert_eq!(
            Pool::new(MAX_WORKERS + 1).err().unwrap(),
            "WASM async worker count must be between 1 and 64; got 65"
        );
    }
}

#[cfg(test)]
mod cancellation_tests {
    use super::*;
    use crate::test_support::watchdog;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct Capture(Arc<AtomicBool>);
    impl Drop for Capture {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }
    struct PanicOnDrop;
    impl Drop for PanicOnDrop {
        fn drop(&mut self) {
            panic!("injected capture/payload drop panic");
        }
    }

    fn identity(pool: &Pool) -> thread::ThreadId {
        let (reply, receive) = mpsc::channel();
        pool.submit(Box::new(move || {
            reply.send(thread::current().id()).unwrap();
        }))
        .unwrap();
        receive.recv().unwrap()
    }

    #[test]
    fn idle_and_repeated_cancel() {
        if !watchdog("async_pool::cancellation_tests::idle_and_repeated_cancel") {
            return;
        }
        let pool = Pool::new(1).unwrap();
        pool.cancel(); // No job has been submitted: this pool is deterministically idle.
        let original = identity(&pool);
        for _ in 0..5 {
            pool.cancel();
            assert_eq!(identity(&pool), original);
        }
    }

    #[test]
    fn dequeued_unclaimed_job_is_discarded() {
        if !watchdog("async_pool::cancellation_tests::dequeued_unclaimed_job_is_discarded") {
            return;
        }
        let pool = Pool::new(1).unwrap();
        let original = identity(&pool);
        let (dequeued, reached) = mpsc::channel();
        let (release, wait) = mpsc::channel();
        *pool.control.before_claim.lock().unwrap() = Some(Box::new(move || {
            dequeued.send(()).unwrap();
            wait.recv().unwrap();
        }));
        let dropped = Arc::new(AtomicBool::new(false));
        let capture = Capture(Arc::clone(&dropped));
        let ran = Arc::new(AtomicBool::new(false));
        let executed = Arc::clone(&ran);
        pool.submit(Box::new(move || {
            executed.store(true, Ordering::Release);
            drop(capture);
        }))
        .unwrap();
        reached.recv().unwrap();
        let control = pool.control();
        let controller = thread::spawn(move || {
            control.wait_for_cancel();
            release.send(()).unwrap();
        });
        pool.cancel();
        assert!(!ran.load(Ordering::Acquire));
        assert!(dropped.load(Ordering::Acquire));
        controller.join().unwrap();
        assert_eq!(identity(&pool), original);
    }

    #[test]
    fn discarded_capture_panic_preserves_worker() {
        if !watchdog("async_pool::cancellation_tests::discarded_capture_panic_preserves_worker") {
            return;
        }
        let pool = Pool::new(1).unwrap();
        let original = identity(&pool);
        let (started, starts) = mpsc::channel();
        let (release, wait) = mpsc::channel();
        pool.submit(Box::new(move || {
            started.send(()).unwrap();
            wait.recv().unwrap();
        }))
        .unwrap();
        starts.recv().unwrap();
        let dropped = Arc::new(AtomicBool::new(false));
        let captures = (PanicOnDrop, Capture(Arc::clone(&dropped)));
        let ran = Arc::new(AtomicBool::new(false));
        let executed = Arc::clone(&ran);
        pool.submit(Box::new(move || {
            executed.store(true, Ordering::Release);
            drop(captures);
        }))
        .unwrap();
        let control = pool.control();
        let controller = thread::spawn(move || {
            control.wait_for_cancel();
            release.send(()).unwrap();
        });
        pool.cancel();
        assert!(!ran.load(Ordering::Acquire));
        assert!(dropped.load(Ordering::Acquire));
        controller.join().unwrap();
        assert_eq!(identity(&pool), original);
    }

    #[test]
    fn running_panic_during_cancel_preserves_worker() {
        if !watchdog("async_pool::cancellation_tests::running_panic_during_cancel_preserves_worker")
        {
            return;
        }
        let pool = Pool::new(1).unwrap();
        let original = identity(&pool);
        let (started, starts) = mpsc::channel();
        let (release, wait) = mpsc::channel();
        let (reply, receive) = mpsc::channel::<()>();
        pool.submit(Box::new(move || {
            let _reply = reply;
            started.send(()).unwrap();
            wait.recv().unwrap();
            std::panic::panic_any(PanicOnDrop);
        }))
        .unwrap();
        starts.recv().unwrap();
        let control = pool.control();
        let controller = thread::spawn(move || {
            control.wait_for_cancel();
            release.send(()).unwrap();
        });
        pool.cancel();
        assert!(receive.recv().is_err());
        controller.join().unwrap();
        assert_eq!(identity(&pool), original);
    }

    #[test]
    fn queued_jobs_survive_running_panic() {
        if !watchdog("async_pool::cancellation_tests::queued_jobs_survive_running_panic") {
            return;
        }
        let pool = Pool::new(1).unwrap();
        let original = identity(&pool);
        let (started, starts) = mpsc::channel();
        let (release, wait) = mpsc::channel();
        let (reply, receive) = mpsc::channel::<()>();
        pool.submit(Box::new(move || {
            let _reply = reply;
            started.send(()).unwrap();
            wait.recv().unwrap();
            panic!("injected non-WASM panic");
        }))
        .unwrap();
        starts.recv().unwrap();
        let mut replies = Vec::new();
        for value in 0..4 {
            let (reply, receive) = mpsc::channel();
            replies.push(receive);
            pool.submit(Box::new(move || reply.send(value).unwrap()))
                .unwrap();
        }
        release.send(()).unwrap();
        assert!(receive.recv().is_err());
        for (value, reply) in replies.into_iter().enumerate() {
            assert_eq!(reply.recv().unwrap(), value);
        }
        assert_eq!(identity(&pool), original);
    }
}
