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
//! an accident of a dependency swap. The `derive_seed` values were captured
//! from the `DefaultHasher`-based implementation that its fixed hash replaced.

use pecos_random::{ParallelRapidRng, PecosScalarRng, derive_seed};

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

#[test]
fn derive_seed_values_are_stable() {
    let long = "x".repeat(100);
    for (base, purpose, expected) in [
        (42, "noise", 4_853_411_367_290_695_164),
        (42, "measurement", 3_392_018_874_687_868_882),
        (0, "", 3_410_042_138_156_595_269),
        (u64::MAX, "noise_model", 7_354_911_115_145_454_653),
        // The purposes HybridEngine and QuantumSystem derive component seeds with.
        (42, "classical_engine", 10_633_337_267_899_434_960),
        (42, "quantum_system", 6_768_816_764_539_692_319),
        (42, "quantum_engine", 10_075_785_296_346_441_876),
        (7, "worker_0", 13_979_981_128_284_556_205),
        (123, "quantum", 6_503_438_428_033_344_226),
        // Length 2: with the rows above, every byte length mod 8 is covered.
        (3, "xy", 12_234_639_505_916_456_087),
        (
            1,
            "h\u{e9}llo \u{2713} \u{91cf}\u{5b50}",
            6_045_238_936_066_559_765,
        ),
        (5, long.as_str(), 10_847_697_514_060_669_651),
        // Lengths 7 and 8: with the 0xFF terminator, one fills a block exactly.
        (9, "abcdefg", 12_570_786_958_402_476_676),
        (9, "abcdefgh", 1_102_260_455_885_514_446),
    ] {
        assert_eq!(
            derive_seed(base, purpose),
            expected,
            "derive_seed({base}, {purpose:?})"
        );
    }
}
