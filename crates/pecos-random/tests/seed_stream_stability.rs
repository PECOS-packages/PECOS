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
    15_898_102_487_349_570_925,
    4_075_293_041_860_347_929,
    2_961_045_109_388_800_320,
    7_198_908_955_497_073_420,
    12_155_105_407_659_006_943,
    9_311_516_336_095_720_243,
    17_539_611_694_522_935_563,
    3_088_116_801_777_502_571,
];
const GOLDEN_SCALAR_SEED_42: [u64; 8] = [
    15_898_102_487_349_570_925,
    12_155_105_407_659_006_943,
    9_267_879_203_684_296_501,
    11_858_079_087_261_110_352,
    4_827_150_399_489_690_183,
    17_360_650_844_478_325_545,
    12_169_232_296_622_201_862,
    13_970_818_894_244_782_829,
];
