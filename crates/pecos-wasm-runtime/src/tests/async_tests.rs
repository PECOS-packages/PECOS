use super::*;
use crate::test_support::watchdog;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier, mpsc};
use std::thread;
use std::time::{Duration, Instant};

fn submit(job: impl FnOnce() + Send + 'static) {
    with_pool(|pool| pool.unwrap().submit(Box::new(job)).unwrap());
}

fn initialize(workers: usize) {
    register(0, MODULE).unwrap();
    let context = acquire(0).unwrap();
    let function = lookup(0, "increment", 0, 1).unwrap();
    call_async(context, function, &[], workers).unwrap();
    assert_eq!(read(context, 1).unwrap(), [11]);
    release(context).unwrap();
}

fn identities(count: usize) -> std::collections::HashSet<thread::ThreadId> {
    let barrier = Arc::new(Barrier::new(count));
    let (send, receive) = mpsc::channel();
    for _ in 0..count {
        let barrier = Arc::clone(&barrier);
        let send = send.clone();
        submit(move || {
            send.send(thread::current().id()).unwrap();
            barrier.wait();
        });
    }
    (0..count).map(|_| receive.recv().unwrap()).collect()
}

#[test]
fn pool_survives_resets() {
    if !watchdog("tests::async_tests::pool_survives_resets") {
        return;
    }
    initialize(2);
    // Thread ids are never reused, so equal ids mean no worker was respawned.
    let original = identities(2);
    assert_eq!(original.len(), 2);
    for _ in 0..4 {
        reset().unwrap();
        initialize(2);
        assert_eq!(identities(2), original);
    }
}

struct Dropped(Arc<AtomicBool>);
impl Drop for Dropped {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

fn cancellation_route(route: &str) {
    reset().unwrap();
    begin_worker_shot([1; 32]).unwrap();
    initialize(1);
    let original = identities(1);
    let control = with_pool(|pool| pool.unwrap().control());
    let (started, starts) = mpsc::channel();
    let (release, wait) = mpsc::channel();
    let finished = Arc::new(AtomicBool::new(false));
    let observed = Arc::clone(&finished);
    submit(move || {
        started.send(()).unwrap();
        wait.recv().unwrap();
        observed.store(true, Ordering::Release);
    });
    starts.recv().unwrap();
    let discarded = Arc::new(AtomicBool::new(false));
    let capture = Dropped(Arc::clone(&discarded));
    let ran = Arc::new(AtomicBool::new(false));
    let executed = Arc::clone(&ran);
    submit(move || {
        executed.store(true, Ordering::Release);
        drop(capture);
    });
    let controller = thread::spawn(move || {
        control.wait_for_cancel();
        release.send(()).unwrap();
    });
    match route {
        "reset" => {
            reset().unwrap();
        }
        "changed_key" => {
            begin_worker_shot([2; 32]).unwrap();
        }
        "unfinished" => {
            begin_worker_shot([1; 32]).unwrap();
        }
        "boundary" => {
            assert_eq!(boundary(-1, || crate::release(999)), -1);
        }
        _ => unreachable!(),
    }
    assert!(finished.load(Ordering::Acquire));
    assert!(discarded.load(Ordering::Acquire));
    assert!(!ran.load(Ordering::Acquire));
    controller.join().unwrap();
    assert_eq!(identities(1), original);
}

macro_rules! route_test {
    ($name:ident, $route:literal) => {
        #[test]
        fn $name() {
            if !watchdog(concat!("tests::async_tests::", stringify!($name))) {
                return;
            }
            cancellation_route($route);
        }
    };
}
route_test!(reset_cancels, "reset");
route_test!(changed_key_cancels, "changed_key");
route_test!(unfinished_shot_cancels, "unfinished");
route_test!(boundary_cancels, "boundary");

#[test]
fn worker_count_follows_state_lifetime() {
    if !watchdog("tests::async_tests::worker_count_follows_state_lifetime") {
        return;
    }
    for count in [1, 3, 2] {
        reset().unwrap();
        begin_worker_shot([3; 32]).unwrap();
        initialize(count);
        with_pool(|pool| assert_eq!(pool.unwrap().size(), count));
        assert_eq!(identities(count).len(), count);
        for next_shot in [false, true] {
            if next_shot {
                finish_worker_shot().unwrap();
                begin_worker_shot([3; 32]).unwrap();
                register(0, MODULE).unwrap();
            }
            let context = acquire(0).unwrap();
            let function = lookup(0, "increment", 0, 1).unwrap();
            assert_eq!(
                call_async(context, function, &[], count + 1).unwrap_err(),
                "WASM async worker count changed without reset"
            );
            release(context).unwrap();
        }
    }
}

#[test]
fn queue_wait_is_not_execution_timeout() {
    if !watchdog("tests::async_tests::queue_wait_is_not_execution_timeout") {
        return;
    }
    initialize(1);
    let (started, starts) = mpsc::channel();
    let (release, wait) = mpsc::channel();
    submit(move || {
        started.send(()).unwrap();
        wait.recv().unwrap();
    });
    starts.recv().unwrap();
    // Both instantiation (including start) and the export execute inside each job.
    let busy = br#"(module
        (func $work (export "work") (result i64) (local $n i32)
            i32.const 1000 local.set $n
            (loop $again local.get $n i32.const 1 i32.sub local.tee $n br_if $again)
            i64.const 7)
        (func $start call $work drop) (start $start))"#;
    register_with_timeout(1, busy, 0.1).unwrap();
    let function = lookup(1, "work", 0, 1).unwrap();
    let mut contexts = Vec::new();
    for index in 1..5 {
        let context = acquire(index).unwrap();
        call_async(context, function, &[], 1).unwrap();
        contexts.push(context);
    }
    let submitted = Instant::now();
    let controller = thread::spawn(move || {
        let hold = Duration::from_millis(250);
        while submitted.elapsed() < hold {
            thread::sleep(hold.saturating_sub(submitted.elapsed()));
        }
        release.send(()).unwrap();
    });
    for context in contexts {
        assert_eq!(read(context, 1).unwrap(), [7]);
    }
    controller.join().unwrap();
}

#[test]
fn worker_shot_recovers_after_async_trap() {
    if !watchdog("tests::async_tests::worker_shot_recovers_after_async_trap") {
        return;
    }
    for abi_failure in [false, true] {
        reset().unwrap();
        begin_worker_shot([9; 32]).unwrap();
        initialize(1);
        let context = acquire(0).unwrap();
        let trap = lookup(0, "trap", 0, 0).unwrap();
        call_async(context, trap, &[], 1).unwrap();
        if abi_failure {
            assert_eq!(boundary(-1, || read(context, 0).map(|_| 0)), -1);
        } else {
            assert!(read(context, 0).is_err());
        }
        begin_worker_shot([9; 32]).unwrap();
        STATE.with(|state| assert!(state.borrow().contexts.is_empty()));
        initialize(1);
        finish_worker_shot().unwrap();
    }
}

// Worker-thread state whose destructor finishes only after a delay: a join
// returns after it, so the flag is set; a missing join returns first.
struct SlowExit(Arc<AtomicBool>);
impl Drop for SlowExit {
    fn drop(&mut self) {
        thread::sleep(Duration::from_millis(100));
        self.0.store(true, Ordering::Release);
    }
}

thread_local! {
    static SLOW_EXIT: std::cell::RefCell<Option<SlowExit>> = const { std::cell::RefCell::new(None) };
}

#[test]
fn owner_thread_exit_joins_pool() {
    if !watchdog("tests::async_tests::owner_thread_exit_joins_pool") {
        return;
    }
    for active in [false, true] {
        let finished = Arc::new(AtomicBool::new(false));
        let observed = Arc::clone(&finished);
        let discarded = Arc::new(AtomicBool::new(false));
        let capture = Dropped(Arc::clone(&discarded));
        let worker_exited = Arc::new(AtomicBool::new(false));
        let exit_flag = Arc::clone(&worker_exited);
        thread::spawn(move || {
            initialize(1); // Initialize STATE, COMPILED, and POOL through runtime calls.
            let original = identities(1);
            reset().unwrap();
            assert_eq!(identities(1), original);
            let (installed, install) = mpsc::channel();
            submit(move || {
                SLOW_EXIT.with(|slot| *slot.borrow_mut() = Some(SlowExit(exit_flag)));
                installed.send(()).unwrap();
            });
            install.recv().unwrap();
            if active {
                let (started, starts) = mpsc::channel();
                let (release, wait) = mpsc::channel();
                let control = with_pool(|pool| pool.unwrap().control());
                submit(move || {
                    started.send(()).unwrap();
                    wait.recv().unwrap();
                    finished.store(true, Ordering::Release);
                });
                starts.recv().unwrap();
                submit(move || drop(capture));
                thread::spawn(move || {
                    control.wait_for_cancel();
                    release.send(()).unwrap();
                });
            }
            // Only real TLS destructors cancel and join the surviving pool.
        })
        .join()
        .unwrap();
        assert!(
            worker_exited.load(Ordering::Acquire),
            "owner exit returned before its worker thread ended"
        );
        if active {
            assert!(discarded.load(Ordering::Acquire));
            assert!(observed.load(Ordering::Acquire));
        }
    }
}

pub(crate) mod abi {
    crate::export_wasm_runtime!();
}

#[test]
fn public_abi_failures_clean_up_without_reset() {
    if !watchdog("tests::async_tests::public_abi_failures_clean_up_without_reset") {
        return;
    }
    for failure in ["second_read", "unread_release", "null_arguments"] {
        initialize(1);
        let context = abi::pecos_wasm_v1_acquire(0);
        // SAFETY: Each non-null pointer below covers the exact supplied length.
        unsafe {
            let function = abi::pecos_wasm_v1_lookup(0, b"increment".as_ptr(), 9, 0, 1);
            assert_ne!(function, 0);
            let mut output = 0;
            if failure == "null_arguments" {
                assert_eq!(
                    abi::pecos_wasm_v2_call(context, function, std::ptr::null(), 1, 1),
                    -1
                );
                let error = abi::pecos_wasm_v1_error();
                let message = std::slice::from_raw_parts(error.add(1), usize::from(*error));
                assert_eq!(message, b"Null WASM arguments");
            } else {
                assert_eq!(
                    abi::pecos_wasm_v2_call(context, function, std::ptr::null(), 0, 1),
                    0
                );
                if failure == "second_read" {
                    assert_eq!(abi::pecos_wasm_v1_read(context, &raw mut output, 1), 0);
                    assert_eq!(abi::pecos_wasm_v1_read(context, &raw mut output, 1), -1);
                } else {
                    assert_eq!(abi::pecos_wasm_v1_release(context), -1);
                }
            }
        }
        // No reset between failure and re-registration/recovery.
        initialize(1);
        reset().unwrap();
    }
}
