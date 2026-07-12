// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Golden-value tests locking the seeded output streams of the PECOS RNGs.
//!
//! Seeded reproducibility is load-bearing: users record seeds to reproduce
//! simulation runs. These values were captured from the rapidhash 4.4
//! `RapidRng`-backed implementation; any change to them is a breaking change
//! to seeded reproducibility and must be deliberate (and release-noted), not
//! an accident of a dependency swap.

use pecos_random::{ParallelRapidRng, PecosScalarRng};

#[test]
fn parallel_rapid_rng_seed_42_stream_is_stable() {
    let mut rng = ParallelRapidRng::seed_from_u64(42);
    let mut out = [0u64; 8];
    rng.fill_u64(&mut out);
    assert_eq!(out, GOLDEN_PARALLEL_SEED_42);
}

#[test]
fn scalar_rng_seed_42_stream_is_stable() {
    let mut rng = PecosScalarRng::seed_from_u64(42);
    let out: Vec<u64> = (0..8).map(|_| rng.next_u64()).collect();
    assert_eq!(out, GOLDEN_SCALAR_SEED_42);
}

// Captured from the rapidhash 4.4 implementation (pre-rapidrand migration).
// Regenerate ONLY for a deliberate, release-noted reproducibility break.
const GOLDEN_PARALLEL_SEED_42: [u64; 8] = [
    15898102487349570925,
    4075293041860347929,
    2961045109388800320,
    7198908955497073420,
    12155105407659006943,
    9311516336095720243,
    17539611694522935563,
    3088116801777502571,
];
const GOLDEN_SCALAR_SEED_42: [u64; 8] = [
    15898102487349570925,
    12155105407659006943,
    9267879203684296501,
    11858079087261110352,
    4827150399489690183,
    17360650844478325545,
    12169232296622201862,
    13970818894244782829,
];
