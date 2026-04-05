// Copyright 2026 The PECOS Developers
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

// profiling calculations use count as f64
#![allow(clippy::cast_precision_loss)]
//! Profile `VecSet` operations to identify optimization opportunities.
//!
//! Run with:
//!   cargo run --release --example `profile_vecset` -p pecos-core

use pecos_core::{Set, SortedVecSet, VecSet};
use std::hint::black_box;

fn main() {
    let iterations = 100_000;

    println!("VecSet vs SortedVecSet Operation Profiling");
    println!("==========================================\n");

    // Test different set sizes
    for size in [4, 8, 16, 32, 64] {
        println!("Set size: {size}");

        // Create test sets - VecSet
        let set_a: VecSet<usize> = (0..size).collect();
        let set_b: VecSet<usize> = (size / 2..size + size / 2).collect();

        // Create test sets - SortedVecSet
        let sorted_a: SortedVecSet = (0..size).collect();
        let sorted_b: SortedVecSet = (size / 2..size + size / 2).collect();

        // Profile toggle (symmetric_difference_item_update)
        let mut test_set = set_a.clone();
        let start = std::time::Instant::now();
        for i in 0..iterations {
            test_set.symmetric_difference_item_update(&(i % size));
        }
        let toggle_time = start.elapsed();
        black_box(&test_set);

        // Profile xor_assign (symmetric_difference_update)
        let start = std::time::Instant::now();
        for _ in 0..iterations / 10 {
            let mut test_set = set_a.clone();
            test_set.symmetric_difference_update(&set_b);
            black_box(&test_set);
        }
        let xor_time = start.elapsed();

        // Profile contains
        let start = std::time::Instant::now();
        for i in 0..iterations {
            black_box(set_a.contains(&(i % (size * 2))));
        }
        let contains_time = start.elapsed();

        // Profile intersection_count
        let start = std::time::Instant::now();
        for _ in 0..iterations / 10 {
            black_box(set_a.intersection_count(&set_b));
        }
        let intersection_time = start.elapsed();

        // Profile iteration
        let start = std::time::Instant::now();
        for _ in 0..iterations / 10 {
            let mut sum = 0usize;
            for &x in &set_a {
                sum = sum.wrapping_add(x);
            }
            black_box(sum);
        }
        let iter_time = start.elapsed();

        // Profile SortedVecSet toggle
        let mut sorted_test = sorted_a.clone();
        let start = std::time::Instant::now();
        for i in 0..iterations {
            sorted_test.toggle(i % size);
        }
        let sorted_toggle_time = start.elapsed();
        black_box(&sorted_test);

        // Profile SortedVecSet xor_assign
        let start = std::time::Instant::now();
        for _ in 0..iterations / 10 {
            let mut test_set = sorted_a.clone();
            test_set.xor_assign(&sorted_b);
            black_box(&test_set);
        }
        let sorted_xor_time = start.elapsed();

        // Profile SortedVecSet contains
        let start = std::time::Instant::now();
        for i in 0..iterations {
            black_box(sorted_a.contains(i % (size * 2)));
        }
        let sorted_contains_time = start.elapsed();

        println!("  VecSet:");
        println!(
            "    toggle:       {:>8.1} ns/op",
            toggle_time.as_nanos() as f64 / iterations as f64
        );
        println!(
            "    xor_assign:   {:>8.1} ns/op",
            xor_time.as_nanos() as f64 / (iterations / 10) as f64
        );
        println!(
            "    contains:     {:>8.1} ns/op",
            contains_time.as_nanos() as f64 / iterations as f64
        );
        println!(
            "    intersection: {:>8.1} ns/op",
            intersection_time.as_nanos() as f64 / (iterations / 10) as f64
        );
        println!(
            "    iteration:    {:>8.1} ns/op",
            iter_time.as_nanos() as f64 / (iterations / 10) as f64
        );
        println!("  SortedVecSet:");
        println!(
            "    toggle:       {:>8.1} ns/op",
            sorted_toggle_time.as_nanos() as f64 / iterations as f64
        );
        println!(
            "    xor_assign:   {:>8.1} ns/op",
            sorted_xor_time.as_nanos() as f64 / (iterations / 10) as f64
        );
        println!(
            "    contains:     {:>8.1} ns/op",
            sorted_contains_time.as_nanos() as f64 / iterations as f64
        );
        println!();
    }

    // Now compare with sorted approach
    println!("\nSorted VecSet Comparison");
    println!("========================\n");

    for size in [8, 16, 32, 64] {
        println!("Set size: {size}");

        // Unsorted
        let unsorted: Vec<usize> = (0..size).map(|i| (i * 7) % size).collect();
        // Sorted
        let mut sorted = unsorted.clone();
        sorted.sort_unstable();

        // Profile linear search vs binary search for contains
        let target = size / 2;

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            black_box(unsorted.contains(&target));
        }
        let linear_time = start.elapsed();

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            black_box(sorted.binary_search(&target).is_ok());
        }
        let binary_time = start.elapsed();

        println!(
            "  linear search:  {:>8.1} ns/op",
            linear_time.as_nanos() as f64 / iterations as f64
        );
        println!(
            "  binary search:  {:>8.1} ns/op",
            binary_time.as_nanos() as f64 / iterations as f64
        );
        println!(
            "  speedup:        {:>8.2}x",
            linear_time.as_nanos() as f64 / binary_time.as_nanos() as f64
        );
        println!();
    }

    // Profile merge-based XOR vs position-based XOR
    println!("\nMerge-based XOR Comparison");
    println!("==========================\n");

    for size in [8, 16, 32, 64] {
        println!("Set size: {size}");

        let mut sorted_a: Vec<usize> = (0..size).collect();
        let mut sorted_b: Vec<usize> = (size / 2..size + size / 2).collect();
        sorted_a.sort_unstable();
        sorted_b.sort_unstable();

        // Position-based XOR (current VecSet approach)
        let start = std::time::Instant::now();
        for _ in 0..iterations / 10 {
            let mut result = sorted_a.clone();
            for &item in &sorted_b {
                if let Some(pos) = result.iter().position(|&x| x == item) {
                    result.swap_remove(pos);
                } else {
                    result.push(item);
                }
            }
            black_box(&result);
        }
        let position_time = start.elapsed();

        // Merge-based XOR (sorted approach)
        let start = std::time::Instant::now();
        for _ in 0..iterations / 10 {
            let result = merge_xor(&sorted_a, &sorted_b);
            black_box(&result);
        }
        let merge_time = start.elapsed();

        println!(
            "  position-based: {:>8.1} ns/op",
            position_time.as_nanos() as f64 / (iterations / 10) as f64
        );
        println!(
            "  merge-based:    {:>8.1} ns/op",
            merge_time.as_nanos() as f64 / (iterations / 10) as f64
        );
        println!(
            "  speedup:        {:>8.2}x",
            position_time.as_nanos() as f64 / merge_time.as_nanos() as f64
        );
        println!();
    }
}

/// Merge-based XOR for sorted arrays - O(n+m) instead of O(n*m)
fn merge_xor(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                result.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                // Element in both - XOR cancels, skip both
                i += 1;
                j += 1;
            }
        }
    }

    // Add remaining elements
    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);

    result
}
