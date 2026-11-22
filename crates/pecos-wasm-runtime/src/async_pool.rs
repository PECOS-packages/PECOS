//! A per-execution-thread pool. Jobs own all WASM state they access.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};

pub(crate) const MAX_WORKERS: usize = 64;
pub(crate) type Job = Box<dyn FnOnce() + Send + 'static>;

pub(crate) struct Pool {
    sender: Option<mpsc::Sender<Job>>,
    workers: Vec<JoinHandle<()>>,
    stopping: Arc<AtomicBool>,
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
                "WASM async worker count must be between 1 and {MAX_WORKERS}"
            ));
        }
        let (sender, receiver) = mpsc::channel::<Job>();
        let receiver = Arc::new(Mutex::new(receiver));
        let mut pool = Self {
            sender: Some(sender),
            workers: Vec::new(),
            stopping: Arc::new(AtomicBool::new(false)),
        };
        for _ in 0..count {
            let receiver = Arc::clone(&receiver);
            let stopping = Arc::clone(&pool.stopping);
            let worker = spawn(Box::new(move || {
                loop {
                    // Never hold the queue lock while running a job.
                    let job = {
                        let receiver = receiver
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        receiver.recv()
                    };
                    let Ok(job) = job else { break };
                    if stopping.load(Ordering::Acquire) {
                        // Dropping queued captures also disconnects their reply channels.
                        drop(job);
                        continue;
                    }
                    // A failed job drops its owned state/reply sender. The caller
                    // observes disconnection; this worker remains usable.
                    let _ = catch_unwind(AssertUnwindSafe(job));
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
        self.sender
            .as_ref()
            .ok_or("WASM async executor is closed")?
            .send(job)
            .map_err(|_| "WASM async executor disconnected".into())
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.stopping.store(true, Ordering::Release);
        // Close before joining: idle workers must be able to leave recv().
        drop(self.sender.take());
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    // Lifecycle bugs must fail rather than hang the full test process.
    fn watchdog(name: &str) -> bool {
        if std::env::var("PECOS_WASM_POOL_TEST").as_deref() == Ok(name) {
            return true;
        }
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([name, "--exact"])
            .env("PECOS_WASM_POOL_TEST", name)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(status.success(), "lifecycle child failed: {name}");
                return false;
            }
            if Instant::now() >= deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("lifecycle child timed out: {name}");
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn owner_thread_exit_joins_pool() {
        if !watchdog("async_pool::tests::owner_thread_exit_joins_pool") {
            return;
        }
        // Exercise the actual runtime TLS destructor, including an abandoned
        // running job and a queued job, not just an explicit Pool::drop.
        for active in [false, true] {
            let finished = Arc::new(AtomicBool::new(false));
            let observed = Arc::clone(&finished);
            let (started, starts) = mpsc::channel();
            let owner = thread::spawn(move || {
                let pool = Pool::new(1).unwrap();
                if active {
                    pool.submit(Box::new(move || {
                        started.send(()).unwrap();
                        thread::sleep(Duration::from_millis(20));
                        finished.store(true, Ordering::Release);
                    }))
                    .unwrap();
                    starts.recv_timeout(Duration::from_secs(5)).unwrap();
                    pool.submit(Box::new(|| {})).unwrap();
                }
                crate::STATE.with(|state| state.borrow_mut().pool = Some(pool));
                // No explicit reset: leaving the owner must join its pool.
            });
            owner.join().unwrap();
            if active {
                assert!(observed.load(Ordering::Acquire));
            }
        }
    }

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
        let stopping = Arc::clone(&pool.stopping);
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
        while !stopping.load(Ordering::Acquire) {
            thread::yield_now();
        }
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
        let (reply, receiver) = mpsc::channel::<()>();
        pool.submit(Box::new(move || {
            let _reply = reply;
            panic!("injected job panic");
        }))
        .unwrap();
        assert!(receiver.recv().is_err());
        let (reply, receiver) = mpsc::channel();
        pool.submit(Box::new(move || {
            reply.send(42).unwrap();
        }))
        .unwrap();
        assert_eq!(receiver.recv().unwrap(), 42);
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
        assert!(Pool::new(0).is_err());
        assert!(Pool::new(MAX_WORKERS + 1).is_err());
    }
}
