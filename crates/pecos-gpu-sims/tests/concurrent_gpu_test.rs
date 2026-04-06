//! Stress tests for concurrent GPU simulator usage.
//!
//! These tests verify that multiple GPU simulators can be created, used,
//! and destroyed concurrently without segfaults or resource leaks.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use pecos_core::QubitId;
use pecos_gpu_sims::DefaultGpuStab;
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
                    // sim drops here -- Drop should sync device
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
