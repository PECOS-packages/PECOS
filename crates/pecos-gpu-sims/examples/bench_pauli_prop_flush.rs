//! Quick wall-time benchmark for `GpuPauliProp` flush throughput.
//!
//! Used to verify the inter-gate poll removal didn't regress correctness
//! cost; expect dispatches/sec to scale with circuit length, not block on
//! poll between gates.

use pecos_gpu_sims::GpuPauliProp;
use std::time::Instant;

fn main() {
    let n_qubits = 32usize;
    let n_shots = 1024u32;
    let n_iters = 500;

    let Ok(mut prop) = GpuPauliProp::with_seed(n_qubits, n_shots, 42) else {
        eprintln!("no GPU; skipping benchmark");
        return;
    };

    // Warm-up: pipelines, allocators, etc.
    for _ in 0..50 {
        prop.h(&[0]);
    }
    prop.sync();

    let start = Instant::now();
    for _ in 0..n_iters {
        // 7 dispatches per iter: 4-qubit H, 2-pair CX, 1-pair CX.
        prop.h(&[0, 1, 2, 3]);
        prop.cx(&[(0, 1), (2, 3)]);
        prop.cx(&[(1, 2)]);
    }
    prop.sync();
    let elapsed = start.elapsed();

    let total = u64::try_from(n_iters * 7).unwrap();
    #[allow(clippy::cast_precision_loss)] // bench output, not numerically critical
    let us_per = (elapsed.as_micros() as f64) / (total as f64);
    #[allow(clippy::cast_precision_loss)]
    let dispatch_per_s = (total as f64) / elapsed.as_secs_f64();
    println!(
        "{total} dispatches in {elapsed:?} -> {us_per:.2} us/dispatch ({dispatch_per_s:.0} dispatch/s)"
    );
}
