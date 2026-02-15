// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Test comparing dispatch paths for bulk RNG operations.

#![allow(clippy::cast_precision_loss)]

use pecos_rng::{PecosRng, Rng, RngBulkExt, RngProbabilityExt};
use std::time::Instant;

/// Simulates OLD `measurement_sampler` path (before `RngBulkExt`)
fn old_path<R: Rng + RngProbabilityExt>(rng: &mut R, dest: &mut [u64]) {
    rng.fill_u64(dest); // Dispatches to trait default (loop)
}

/// Simulates NEW `measurement_sampler` path
fn new_path<R: Rng + RngBulkExt>(rng: &mut R, dest: &mut [u64]) {
    rng.fill_u64_bulk(dest); // Dispatches to explicit impl (optimized)
}

#[test]
#[ignore = "Performance test - run explicitly with: cargo test -p pecos-rng -- --ignored"]
fn compare_dispatch_paths() {
    const ITERATIONS: usize = 1000;
    const SIZE: usize = 10000;

    let mut data = vec![0u64; SIZE];

    // NEW path (RngBulkExt)
    let mut rng = PecosRng::seed_from_u64(42);
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        new_path(&mut rng, &mut data);
    }
    let new_elapsed = start.elapsed();

    // OLD path (RngProbabilityExt)
    let mut rng = PecosRng::seed_from_u64(42);
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        old_path(&mut rng, &mut data);
    }
    let old_elapsed = start.elapsed();

    println!("\n=== Dispatch Path Comparison ===");
    println!("NEW (RngBulkExt):        {new_elapsed:?}");
    println!("OLD (RngProbabilityExt): {old_elapsed:?}");
    println!(
        "Speedup: {:.2}x",
        old_elapsed.as_nanos() as f64 / new_elapsed.as_nanos() as f64
    );

    // The new path should be faster
    // Note: This is a performance test - results vary with system load
    // Run explicitly with: cargo test -p pecos-rng -- --ignored
    assert!(
        new_elapsed < old_elapsed,
        "New path should be faster: new={new_elapsed:?}, old={old_elapsed:?}"
    );
}
