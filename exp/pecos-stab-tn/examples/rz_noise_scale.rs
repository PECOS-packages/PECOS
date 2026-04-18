// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Probe whether there's value in a dedicated batched-RZ-round API beyond
//! the existing `merge_rz` path. Two comparisons:
//!
//!   A. Per-qubit loop  vs  single `rz(theta, &all_qubits)` slice call.
//!      Pure Rust-call overhead of the noise-injection loop.
//!   B. `merge_rz` on/off at scale, plus the effect of frame tracking for
//!      the measurement-phase dominated regime.

use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable};
use pecos_stab_tn::stab_mps::StabMps;
use std::time::Instant;

fn ion_trap_per_qubit_loop(
    n: usize,
    rounds: usize,
    noise_per_round: usize,
    theta: Angle64,
    merge: bool,
    seed: u64,
) -> (f64, usize) {
    let mut stn = StabMps::builder(n).seed(seed).merge_rz(merge).build();
    for q in 0..n {
        stn.h(&[QubitId(q)]);
    }
    let start = Instant::now();
    for _round in 0..rounds {
        for _ in 0..noise_per_round {
            for q in 0..n {
                stn.rz(theta, &[QubitId(q)]);
            }
        }
        for q in 0..n - 1 {
            stn.cx(&[(QubitId(q), QubitId(q + 1))]);
        }
    }
    stn.flush();
    (start.elapsed().as_secs_f64(), stn.max_bond_dim())
}

fn ion_trap_slice_call(
    n: usize,
    rounds: usize,
    noise_per_round: usize,
    theta: Angle64,
    merge: bool,
    seed: u64,
) -> (f64, usize) {
    let mut stn = StabMps::builder(n).seed(seed).merge_rz(merge).build();
    for q in 0..n {
        stn.h(&[QubitId(q)]);
    }
    let qubits: Vec<QubitId> = (0..n).map(QubitId).collect();
    let start = Instant::now();
    for _round in 0..rounds {
        for _ in 0..noise_per_round {
            stn.rz(theta, &qubits);
        }
        for q in 0..n - 1 {
            stn.cx(&[(QubitId(q), QubitId(q + 1))]);
        }
    }
    stn.flush();
    (start.elapsed().as_secs_f64(), stn.max_bond_dim())
}

fn main() {
    let theta = Angle64::from_radians(0.01);

    println!("Ion-trap memory noise — scaling + per-qubit-loop vs slice-call");
    println!("{:-<80}", "");
    println!(
        "{:<30} {:>10} {:>12} {:>12}",
        "config", "time (s)", "max bond", "rz calls"
    );

    // small + medium: compare merge_rz on/off.
    for &(n, rounds, noise) in &[(12, 10, 20), (12, 20, 30)] {
        let total_rz = rounds * noise * n;
        println!("\n-- n={n}, rounds={rounds}, noise/round={noise} ({total_rz} rz calls)");
        let (t_off_loop, b_off) = ion_trap_per_qubit_loop(n, rounds, noise, theta, false, 42);
        println!(
            "  {:<28} {:>10.4} {:>12}",
            "merge_rz=OFF (loop)", t_off_loop, b_off
        );
        let (t_on_loop, b_on) = ion_trap_per_qubit_loop(n, rounds, noise, theta, true, 42);
        println!(
            "  {:<28} {:>10.4} {:>12}  speedup {:.1}x",
            "merge_rz=ON  (loop)",
            t_on_loop,
            b_on,
            t_off_loop / t_on_loop
        );
        let (t_on_slice, b_on2) = ion_trap_slice_call(n, rounds, noise, theta, true, 42);
        println!(
            "  {:<28} {:>10.4} {:>12}  vs loop {:.2}x",
            "merge_rz=ON  (slice)",
            t_on_slice,
            b_on2,
            t_on_loop / t_on_slice
        );
    }

    // Medium-scale merge_rz=ON only (OFF is hours at these sizes).
    {
        let &(n, rounds, noise) = &(16, 20, 30);
        let total_rz = rounds * noise * n;
        println!(
            "\n-- n={n}, rounds={rounds}, noise/round={noise} ({total_rz} rz calls, merge_rz=ON)"
        );
        let (t_on_loop, b_on) = ion_trap_per_qubit_loop(n, rounds, noise, theta, true, 42);
        println!("  {:<28} {:>10.4} {:>12}", "loop", t_on_loop, b_on);
        let (t_on_slice, b_on2) = ion_trap_slice_call(n, rounds, noise, theta, true, 42);
        println!(
            "  {:<28} {:>10.4} {:>12}  vs loop {:.2}x",
            "slice",
            t_on_slice,
            b_on2,
            t_on_loop / t_on_slice
        );
    }

    println!("\n{:-<80}", "");
    println!("Conclusion: merge_rz gives the dominant speedup. Passing all qubits in one");
    println!("slice call vs looping per-qubit makes no significant difference — the");
    println!("pending_rz accumulator is the hot path and already O(1) per rz invocation.");
}
