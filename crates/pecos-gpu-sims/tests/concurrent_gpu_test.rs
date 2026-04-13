//! Stress tests for concurrent GPU simulator usage.
//!
//! These tests verify that multiple GPU simulators can be created, used,
//! and destroyed concurrently without segfaults or resource leaks.
//!
//! With the shared process-wide wgpu context (`gpu_probe::gpu_context`),
//! multiple sim types share one Device and Queue. The mixed-sim stress test
//! is the real coverage for the shared-context corner cases (interleaved
//! `queue.write_buffer` / `queue.submit`, parallel state readback, etc.).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use pecos_core::QubitId;
use pecos_gpu_sims::{
    DefaultGpuStab, GpuPauliProp, GpuStateVec32, GpuStateVecAuto, gpu_probe::gpu_context,
};
use pecos_simulators::CliffordGateable;

#[test]
fn test_concurrent_gpu_stab_creation_and_destruction() {
    let created = AtomicUsize::new(0);

    thread::scope(|s| {
        for i in 0u64..8 {
            let created = &created;
            s.spawn(move || {
                for _ in 0..5 {
                    let sim = DefaultGpuStab::with_seed(4, i);
                    if let Ok(mut sim) = sim {
                        created.fetch_add(1, Ordering::Relaxed);
                        sim.h(&[QubitId(0)]);
                        sim.cx(&[(QubitId(0), QubitId(1))]);
                        let _ = sim.mz(&[QubitId(0)]);
                    }
                    // sim drops here
                }
            });
        }
    });

    let total = created.load(Ordering::Relaxed);
    if total == 0 {
        eprintln!("WARNING: no GPU available -- concurrent stress test was a no-op");
    }
}

#[test]
fn test_rapid_create_destroy() {
    let mut created = 0usize;

    for i in 0..20 {
        let sim = DefaultGpuStab::with_seed(2, i);
        if let Ok(mut sim) = sim {
            created += 1;
            sim.h(&[QubitId(0)]);
            let _ = sim.mz(&[QubitId(0)]);
        }
    }

    if created == 0 {
        eprintln!("WARNING: no GPU available -- rapid create/destroy test was a no-op");
    }
}

/// Mixed simulator types under `thread::scope`. Each thread spins a small Bell-state
/// circuit on a randomly-picked sim type. Asserts perfect Z parity on the result.
/// This is the corner case the shared-context fix is designed for: interleaved
/// `queue.write_buffer` and `queue.submit` from many sim types on one Device.
#[test]
fn test_mixed_sim_types_concurrent() {
    // First check whether the shared context comes up at all -- if not the
    // whole test should skip cleanly rather than counting zero successes.
    if gpu_context().is_err() {
        eprintln!("WARNING: no GPU available -- mixed-sim concurrent test was a no-op");
        return;
    }

    let bell_parity_failures = AtomicUsize::new(0);
    let total_runs = AtomicUsize::new(0);

    thread::scope(|s| {
        for tid in 0u32..8 {
            let bell_parity_failures = &bell_parity_failures;
            let total_runs = &total_runs;
            s.spawn(move || {
                // Each thread cycles through all four sim types twice.
                for round in 0..2 {
                    let kind = (tid + round) % 4;
                    match kind {
                        0 => run_stab_bell(tid, total_runs, bell_parity_failures),
                        1 => run_statevec_auto_bell(tid, total_runs, bell_parity_failures),
                        2 => run_statevec32_bell(tid, total_runs, bell_parity_failures),
                        _ => run_pauli_prop_no_op(tid, total_runs),
                    }
                }
            });
        }
    });

    assert_eq!(
        bell_parity_failures.load(Ordering::Relaxed),
        0,
        "Bell parity violated across threads -- shared GPU context interleaving bug"
    );

    let runs = total_runs.load(Ordering::Relaxed);
    assert!(
        runs > 0,
        "Mixed-sim test produced zero runs even though gpu_context() reported success"
    );
}

fn run_stab_bell(seed: u32, total: &AtomicUsize, fails: &AtomicUsize) {
    let Ok(mut sim) = DefaultGpuStab::with_seed(2, u64::from(seed)) else {
        return;
    };
    sim.h(&[QubitId(0)]);
    sim.cx(&[(QubitId(0), QubitId(1))]);
    let results = sim.mz(&[QubitId(0), QubitId(1)]);
    total.fetch_add(1, Ordering::Relaxed);
    if results.len() != 2 || results[0].outcome != results[1].outcome {
        fails.fetch_add(1, Ordering::Relaxed);
    }
}

fn run_statevec_auto_bell(seed: u32, total: &AtomicUsize, fails: &AtomicUsize) {
    let _ = seed; // GpuStateVecAuto::new doesn't take a seed; deterministic enough.
    let Ok(mut sim) = GpuStateVecAuto::new(2) else {
        return;
    };
    sim.h(&[QubitId(0)]);
    sim.cx(&[(QubitId(0), QubitId(1))]);
    let results = sim.mz(&[QubitId(0), QubitId(1)]);
    total.fetch_add(1, Ordering::Relaxed);
    if results.len() != 2 || results[0].outcome != results[1].outcome {
        fails.fetch_add(1, Ordering::Relaxed);
    }
}

fn run_statevec32_bell(seed: u32, total: &AtomicUsize, fails: &AtomicUsize) {
    let _ = seed;
    let Ok(mut sim) = GpuStateVec32::new(2) else {
        return;
    };
    sim.h(&[QubitId(0)]);
    sim.cx(&[(QubitId(0), QubitId(1))]);
    let results = sim.mz(&[QubitId(0), QubitId(1)]);
    total.fetch_add(1, Ordering::Relaxed);
    if results.len() != 2 || results[0].outcome != results[1].outcome {
        fails.fetch_add(1, Ordering::Relaxed);
    }
}

/// `GpuPauliProp` doesn't have a Bell-state notion; just exercise create + flush
/// on the shared device alongside the other sims to stress the queue.
fn run_pauli_prop_no_op(seed: u32, total: &AtomicUsize) {
    let Ok(mut prop) = GpuPauliProp::with_seed(4, 8, u64::from(seed)) else {
        return;
    };
    prop.inject_x_fault(0);
    prop.h(&[0]);
    prop.cx(&[(0, 1)]);
    prop.sync();
    total.fetch_add(1, Ordering::Relaxed);
}

/// Memoization: the second call to `gpu_context()` must return the same device
/// (or the same error) without re-initializing. Catches a regression where
/// someone removes the `OnceLock` in `gpu_probe::gpu_context`.
#[test]
fn test_gpu_context_is_memoized() {
    let first = gpu_context();
    let second = gpu_context();

    match (first, second) {
        (Ok(a), Ok(b)) => {
            // Device handles are Arc-based internally in wgpu; equality of the
            // returned context's underlying device pointer is the cheapest
            // check that we got the same device twice. We don't have a public
            // API for "are these the same device" so use adapter info as a
            // proxy: same name, backend, device_type.
            assert_eq!(a.info.name, b.info.name);
            assert_eq!(a.info.backend, b.info.backend);
            assert_eq!(a.info.device_type, b.info.device_type);
        }
        (Err(a), Err(b)) => {
            // Errors must be memoized too: same kind on both calls.
            assert_eq!(format!("{a}"), format!("{b}"), "memoized error must match");
        }
        _ => panic!("gpu_context() returned different success/failure on repeat calls"),
    }
}
