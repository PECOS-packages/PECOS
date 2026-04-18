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

//! Tier 3 profiling: measure absolute cost of the operations we'd
//! optimize with `cliff_frame` deferral, sub-MPO long-range, and CD
//! Loschmidt Method 2. Decides whether each Tier 3 item is worth the
//! implementation effort.
//!
//! Usage: `cargo run --release --example tier3_profile`.

use pecos_core::QubitId;
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable};
use pecos_stab_tn::stab_mps::StabMps;
use std::time::Instant;

fn bench(label: &str, ops: usize, f: impl FnOnce()) {
    let start = Instant::now();
    f();
    let elapsed = start.elapsed().as_secs_f64();
    let per_op_us = elapsed * 1e6 / ops as f64;
    println!("  {label:<48} {elapsed:>8.4} s  ({per_op_us:>6.2} µs/op × {ops})");
}

fn main() {
    use pecos_simulators::CHForm;
    println!("Tier 3 profiling -- measure where cliff_frame / sub-MPO / CD2 would help");
    println!("{:-<80}", "");

    // ---- 1. Single-qubit Clifford cost (cliff_frame target) ----
    println!();
    println!("1. Single-qubit Clifford batching candidate (cliff_frame target):");
    println!("   Question: how much of QEC time is spent on single-qubit Cliffords?");

    let n = 32;
    let num_ops = 100_000;
    let mut stn = StabMps::builder(n).seed(42).build();
    bench("H × 100k on random qubits", num_ops, || {
        let mut rng = 12345u64;
        for _ in 0..num_ops {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let q = (rng as usize) % n;
            stn.h(&[QubitId(q)]);
        }
    });

    let mut stn = StabMps::builder(n).seed(42).build();
    bench("SZ × 100k on random qubits", num_ops, || {
        let mut rng = 54321u64;
        for _ in 0..num_ops {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let q = (rng as usize) % n;
            stn.sz(&[QubitId(q)]);
        }
    });

    // ---- 2. Long-range CX/CZ cost (sub-MPO target) ----
    // For Clifford gates, tableau handles long-range in O(n) regardless.
    // Sub-MPO would help only for NON-CLIFFORD ops applied to an entangled
    // MPS where the pre_reduce path needs long-range compensation — but
    // we've already switched to the pragmatic-fix path that avoids this.
    println!();
    println!("2. Long-range 2-qubit Clifford cost (sub-MPO target):");
    println!("   Question: does long-range CX hurt STN, given the tableau-only path?");

    let n = 32;
    let num_ops = 100_000;
    let mut stn = StabMps::builder(n).seed(42).build();
    bench("CX(0, n/2) × 100k (max-distance)", num_ops, || {
        for _ in 0..num_ops {
            stn.cx(&[(QubitId(0), QubitId(n / 2))]);
        }
    });

    let mut stn = StabMps::builder(n).seed(42).build();
    bench("CX(0, 1) × 100k (adjacent)", num_ops, || {
        for _ in 0..num_ops {
            stn.cx(&[(QubitId(0), QubitId(1))]);
        }
    });

    // Long-range CX inside a non-Clifford path (where it could matter).
    println!();
    println!("   Non-Clifford workload where MPS bond activity dominates:");

    let n = 12;
    let num_rounds = 50;
    let mut stn = StabMps::builder(n).seed(42).for_qec().build();
    let t = pecos_core::Angle64::QUARTER_TURN / 2u64;
    bench("T+longrange-CX round × 50 (n=12)", num_rounds, || {
        for _ in 0..num_rounds {
            for q in 0..n {
                stn.rz(t, &[QubitId(q)]);
            }
            for q in 0..n / 2 {
                stn.cx(&[(QubitId(q), QubitId(n - 1 - q))]);
            }
        }
    });
    println!("    final bond dim: {}", stn.max_bond_dim());

    // ---- 3. MC overlap_with_stabilizer cost (CD Loschmidt target) ----
    println!();
    println!("3. MC overlap_with_stabilizer (CD Loschmidt Method 1 we have):");
    println!("   Question: is the MC variance a bottleneck for code-state fidelity?");

    let n = 20;
    let mut s = CHForm::new_with_seed(n, 42);
    s.h(&[QubitId(0)]);
    for q in 0..n - 1 {
        s.cx(&[(QubitId(q), QubitId(q + 1))]);
    }
    let mut stn = StabMps::with_seed(n, 7);
    stn.h(&[QubitId(0)]);
    for q in 0..n - 1 {
        stn.cx(&[(QubitId(q), QubitId(q + 1))]);
    }

    let num_samples_small = 100;
    let num_samples_medium = 1000;
    let num_samples_large = 10_000;

    bench("overlap n=20 GHZ (100 samples)", 1, || {
        let _ = stn.overlap_with_stabilizer(&s, num_samples_small, None);
    });
    bench("overlap n=20 GHZ (1k samples)", 1, || {
        let _ = stn.overlap_with_stabilizer(&s, num_samples_medium, None);
    });
    bench("overlap n=20 GHZ (10k samples)", 1, || {
        let _ = stn.overlap_with_stabilizer(&s, num_samples_large, None);
    });

    println!();
    println!("{:-<80}", "");
    println!("Interpretation:");
    println!("  1. If single-qubit Clifford time << total circuit time → cliff_frame skip.");
    println!("  2. If long-range CX time ≈ adjacent CX time → sub-MPO skip.");
    println!("  3. If overlap scales linearly with samples → CD Method 2 (deterministic)");
    println!("     is worthwhile only if we need << sampling error at fixed compute.");
}
