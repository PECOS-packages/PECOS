use super::async_tests::abi;
use crate::COMPILED;
use crate::test_support::watchdog;
use pecos_wasm::{ForeignObject, WasmForeignObject};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::thread;
use std::time::Duration;

const MODULE: &[u8] = br#"(module
    (func (export "add") (param i64 i64) (result i64) local.get 0 local.get 1 i64.add)
    (func (export "spin") (loop br 0)))"#;
const OTHER: &[u8] = br#"(module
    (func (export "add") (param i64 i64) (result i64) local.get 0 local.get 1 i64.add))"#;
const TRAPS: &[u8] = br#"(module (memory 1)
    (func (export "unreachable") unreachable)
    (func (export "bounds") i32.const 65536 i64.load drop))"#;

fn error() -> String {
    // SAFETY: The ABI returns its live TLS buffer prefixed by its byte length.
    unsafe {
        let pointer = abi::pecos_wasm_v1_error();
        String::from_utf8_lossy(std::slice::from_raw_parts(
            pointer.add(1),
            usize::from(*pointer),
        ))
        .into_owned()
    }
}

fn register(bytes: &[u8], timeout: f64) {
    // SAFETY: bytes covers the supplied length.
    assert_eq!(
        unsafe { abi::pecos_wasm_v2_register(0, bytes.as_ptr(), bytes.len(), timeout) },
        0,
        "register: {}",
        error()
    );
}

fn lookup(name: &str, inputs: usize, outputs: usize) -> u64 {
    // SAFETY: name covers the supplied UTF-8 byte length.
    let function =
        unsafe { abi::pecos_wasm_v1_lookup(0, name.as_ptr(), name.len(), inputs, outputs) };
    assert_ne!(function, 0, "lookup {name}: {}", error());
    function
}

fn acquire() -> u64 {
    let context = abi::pecos_wasm_v1_acquire(0);
    assert_ne!(context, 0, "acquire: {}", error());
    context
}

fn call(context: u64, function: u64, arguments: &[i64], asynchronous: bool) -> i32 {
    // SAFETY: arguments covers the supplied number of i64 values.
    unsafe {
        if asynchronous {
            abi::pecos_wasm_v2_call(context, function, arguments.as_ptr(), arguments.len(), 1)
        } else {
            abi::pecos_wasm_v1_call(context, function, arguments.as_ptr(), arguments.len())
        }
    }
}

fn read(context: u64, output: &mut [i64]) -> i32 {
    // SAFETY: output covers the supplied number of writable i64 values.
    unsafe { abi::pecos_wasm_v1_read(context, output.as_mut_ptr(), output.len()) }
}

fn add(context: u64, asynchronous: bool) {
    let function = lookup("add", 2, 1);
    assert_eq!(
        call(context, function, &[19, 23], asynchronous),
        0,
        "{}",
        error()
    );
    let mut output = [0];
    assert_eq!(read(context, &mut output), 0, "{}", error());
    assert_eq!(output, [42]);
}

fn fresh_add(bytes: &[u8], timeout: f64, asynchronous: bool) -> u64 {
    register(bytes, timeout);
    let context = acquire();
    add(context, asynchronous);
    context
}

fn timeout() {
    register(MODULE, 0.2);
    let function = lookup("spin", 0, 0);
    let context = acquire();
    assert_eq!(call(context, function, &[], false), -1);
    assert!(error().contains("timed out"), "{}", error());
}

// The test runs alone in the watchdog subprocess. Before fork, the caller has
// completed host-side Wasmtime work; outstanding workers are parked in guest code.
fn fork_child(normal_exit: bool, body: impl FnOnce()) {
    // SAFETY: All callers satisfy the documented host-side quiescence condition.
    let pid = unsafe { libc::fork() };
    assert!(pid >= 0, "fork: {}", std::io::Error::last_os_error());
    if pid == 0 {
        // SAFETY: Bound even a child stuck in TLS destruction; it owns output pipes.
        unsafe { libc::alarm(10) };
        let result = catch_unwind(AssertUnwindSafe(body));
        let status = if let Err(payload) = result {
            let reason = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("non-string panic payload");
            let message = format!("fork child failed: {}\n", reason.replace('\n', " "));
            // SAFETY: message is live for this write to the watchdog's stderr pipe.
            unsafe { libc::write(2, message.as_ptr().cast(), message.len()) };
            1
        } else {
            0
        };
        // SAFETY: Never return to the test harness in the forked process. Only the
        // exit tests deliberately run TLS destructors of this inherited thread.
        unsafe {
            if normal_exit && status == 0 {
                libc::exit(0);
            }
            libc::_exit(status);
        }
    }
    let mut status = 0;
    // SAFETY: pid identifies our child and status is a writable wait status.
    assert_eq!(unsafe { libc::waitpid(pid, &raw mut status, 0) }, pid);
    let decoded = if libc::WIFEXITED(status) {
        format!("exit code {}", libc::WEXITSTATUS(status))
    } else if libc::WIFSIGNALED(status) {
        format!("terminating signal {}", libc::WTERMSIG(status))
    } else {
        format!("wait status {status:#x}")
    };
    assert!(
        libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0,
        "fork child: {decoded}"
    );
}

#[test]
fn async_after_fork() {
    if !watchdog("tests::fork_tests::async_after_fork") {
        return;
    }
    fresh_add(MODULE, 0.2, true);
    assert_eq!(abi::pecos_wasm_v1_reset(), 0);
    fork_child(false, || {
        fresh_add(MODULE, 0.2, true);
    });
    let context = fresh_add(MODULE, 0.2, true);
    let spin = lookup("spin", 0, 0);
    assert_eq!(call(context, spin, &[], false), -1);
    assert!(error().contains("timed out"), "{}", error());
}

#[test]
fn timeout_after_fork() {
    if !watchdog("tests::fork_tests::timeout_after_fork") {
        return;
    }
    fresh_add(OTHER, 0.2, false);
    assert_eq!(abi::pecos_wasm_v1_reset(), 0);
    timeout();
    COMPILED.with(|cache| assert_eq!(cache.borrow().compilations, 2));
    fork_child(false, || {
        timeout();
        COMPILED.with(|cache| assert_eq!(cache.borrow().compilations, 2));
    });
}

fn outstanding() -> u64 {
    let context = fresh_add(MODULE, 3.0, true);
    let spin = lookup("spin", 0, 0);
    assert_eq!(call(context, spin, &[], true), 0, "{}", error());
    // Reuse the already-created instance: no instantiation is in flight at fork.
    thread::sleep(Duration::from_millis(200));
    context
}

#[test]
fn pending_read_after_fork() {
    if !watchdog("tests::fork_tests::pending_read_after_fork") {
        return;
    }
    let context = outstanding();
    fork_child(false, || {
        assert_eq!(read(context, &mut []), -1);
        assert!(error().contains("Unknown WASM context"), "{}", error());
        fresh_add(MODULE, 0.2, true);
    });
    assert_eq!(read(context, &mut []), -1);
    assert!(error().contains("timed out"), "{}", error());
}

#[test]
fn exit_after_fork_idle() {
    if !watchdog("tests::fork_tests::exit_after_fork_idle") {
        return;
    }
    fresh_add(MODULE, 0.2, true);
    fork_child(true, || {});
}

#[test]
fn exit_after_fork_outstanding() {
    if !watchdog("tests::fork_tests::exit_after_fork_outstanding") {
        return;
    }
    outstanding();
    fork_child(true, || {});
    // Normal parent thread exit cancels and joins its still-live worker.
}

fn trap(name: &str, asynchronous: bool) {
    // Every preceding trap reset the ABI's registrations, handles and contexts.
    register(TRAPS, 0.2);
    let function = lookup(name, 0, 0);
    let context = acquire();
    let status = call(context, function, &[], asynchronous);
    if asynchronous {
        assert_eq!(status, 0, "{}", error());
        assert_eq!(read(context, &mut []), -1);
    } else {
        assert_eq!(status, -1);
    }
    let message = error();
    assert!(message.contains("trapped"), "{message}");
    assert!(!message.contains("panic"), "{message}");
    assert!(!message.contains("disconnected"), "{message}");
}

#[test]
fn trap_after_fork() {
    if !watchdog("tests::fork_tests::trap_after_fork") {
        return;
    }
    for name in ["unreachable", "bounds"] {
        trap(name, false);
    }
    fork_child(false, || {
        for asynchronous in [false, true] {
            for name in ["unreachable", "bounds"] {
                trap(name, asynchronous);
            }
        }
    });
}

#[test]
fn inherited_foreign_object_after_fork() {
    if !watchdog("tests::fork_tests::inherited_foreign_object_after_fork") {
        return;
    }
    let mut object = WasmForeignObject::from_bytes_with_limits(MODULE, 0.2, None).unwrap();
    let mut check = || {
        assert_eq!(object.exec("add", &[19, 23]).unwrap(), [42]);
        let error = object.exec("spin", &[]).unwrap_err().to_string();
        assert!(error.contains("timed out"), "{error}");
    };
    check();
    fork_child(false, check);
}

#[test]
fn new_module_after_fork() {
    if !watchdog("tests::fork_tests::new_module_after_fork") {
        return;
    }
    fresh_add(MODULE, 0.2, false);
    fork_child(false, || {
        // OTHER has never been registered or compiled in this parent.
        let context = fresh_add(OTHER, 0.2, false);
        add(context, true);
        COMPILED.with(|cache| assert_eq!(cache.borrow().compilations, 2));
    });
}
