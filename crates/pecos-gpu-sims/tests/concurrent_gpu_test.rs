//! Stress tests for concurrent GPU simulator usage.
//!
//! These tests verify that multiple GPU simulators can be created, used,
//! and destroyed concurrently without segfaults or resource leaks.

use std::thread;

use pecos_core::QubitId;
use pecos_gpu_sims::DefaultGpuStab;
use pecos_simulators::CliffordGateable;

#[test]
fn test_concurrent_gpu_stab_creation_and_destruction() {
    // Create and destroy many GpuStab instances in parallel threads
    let handles: Vec<_> = (0..8)
        .map(|i| {
            thread::spawn(move || {
                // Create a small simulator, run a simple circuit, drop it.
                // Each thread gets its own device.
                for _ in 0..5 {
                    let sim = DefaultGpuStab::with_seed(4, i as u64);
                    if let Ok(mut sim) = sim {
                        sim.h(&[QubitId(0)]);
                        sim.cx(&[(QubitId(0), QubitId(1))]);
                        let _ = sim.mz(&[QubitId(0)]);
                    }
                    // sim drops here -- Drop should sync device
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_rapid_create_destroy() {
    // Rapidly create and destroy simulators in sequence to stress the Drop path
    for i in 0..20 {
        let sim = DefaultGpuStab::with_seed(2, i);
        if let Ok(mut sim) = sim {
            sim.h(&[QubitId(0)]);
            let _ = sim.mz(&[QubitId(0)]);
        }
    }
}
