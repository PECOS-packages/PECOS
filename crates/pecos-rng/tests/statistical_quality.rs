// Copyright 2025 The PECOS Developers
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

//! Statistical quality tests for RNG implementations.
//!
//! These tests use the `random_tester` crate (based on ENT) to verify that
//! our RNG implementations produce statistically good random numbers.
//!
//! Tests include:
//! - Shannon entropy (should be close to 8.0 bits per byte for ideal randomness)
//! - Chi-square distribution (p-value should be between 0.01 and 0.99)
//! - Mean value (should be close to 127.5 for uniform bytes)
//! - Monte Carlo pi estimation (should be close to pi)
//! - Serial correlation (should be close to 0.0)

use random_tester::{
    ChiSquareCalculation, EntropyTester, MeanCalculation, MonteCarloCalculation,
    SerialCorrelationCoefficientCalculation, ShannonCalculation,
};

/// Number of bytes to generate for testing (1 MB)
const TEST_BYTES: usize = 1_000_000;

/// Results from running all statistical tests
#[derive(Debug)]
struct RngTestResults {
    name: &'static str,
    shannon_entropy: f64,
    chi_square_p: f64,
    mean: f64,
    monte_carlo_pi: f64,
    serial_correlation: f64,
}

impl RngTestResults {
    /// Check if all results are within acceptable bounds
    fn is_acceptable(&self) -> bool {
        // Shannon entropy: for random bytes, should be close to 8.0 bits
        let shannon_ok = self.shannon_entropy > 7.9 && self.shannon_entropy <= 8.0;

        // Chi-square p-value: should be between 0.01 and 0.99
        // Values outside this range suggest non-randomness
        let chi_ok = self.chi_square_p > 0.01 && self.chi_square_p < 0.99;

        // Mean: for uniform bytes [0,255], expected mean is 127.5
        // Allow some deviation (say, within 1%)
        let mean_ok = (self.mean - 127.5).abs() < 1.5;

        // Monte Carlo pi: should be close to 3.14159...
        // Allow ~1% error
        let pi_ok = (self.monte_carlo_pi - std::f64::consts::PI).abs() < 0.05;

        // Serial correlation: should be very close to 0 for independent samples
        let serial_ok = self.serial_correlation.abs() < 0.01;

        shannon_ok && chi_ok && mean_ok && pi_ok && serial_ok
    }

    fn print_report(&self) {
        println!("\n=== {} Statistical Quality Report ===", self.name);
        println!(
            "Shannon entropy:     {:.6} bits/byte (ideal: 8.0)",
            self.shannon_entropy
        );
        println!(
            "Chi-square p-value:  {:.6} (acceptable: 0.01-0.99)",
            self.chi_square_p
        );
        println!("Mean value:          {:.6} (ideal: 127.5)", self.mean);
        println!(
            "Monte Carlo pi:      {:.6} (actual: {:.6})",
            self.monte_carlo_pi,
            std::f64::consts::PI
        );
        println!(
            "Serial correlation:  {:.6} (ideal: 0.0)",
            self.serial_correlation
        );
        println!(
            "Overall:             {}",
            if self.is_acceptable() { "PASS" } else { "FAIL" }
        );
    }
}

/// Run all statistical tests on a byte slice
fn run_tests(name: &'static str, data: &[u8]) -> RngTestResults {
    let mut shannon = ShannonCalculation::default();
    let mut chi = ChiSquareCalculation::default();
    let mut mean = MeanCalculation::default();
    let mut monte_carlo = MonteCarloCalculation::default();
    let mut serial = SerialCorrelationCoefficientCalculation::default();

    shannon.update(data);
    chi.update(data);
    mean.update(data);
    monte_carlo.update(data);
    serial.update(data);

    RngTestResults {
        name,
        shannon_entropy: shannon.finalize(),
        chi_square_p: chi.finalize(),
        mean: mean.finalize(),
        monte_carlo_pi: monte_carlo.finalize(),
        serial_correlation: serial.finalize(),
    }
}

/// Generate random bytes using `PCG64Fast`
fn generate_pcg64fast_bytes(seed: u64, count: usize) -> Vec<u8> {
    use pecos_rng::prelude::PCG64Fast;
    let mut rng = PCG64Fast::seed_from_u64(seed);
    let mut bytes = vec![0u8; count];
    rng.fill_bytes(&mut bytes);
    bytes
}

/// Generate random bytes using `PCGRandom`
fn generate_pcgrandom_bytes(seed: u64, count: usize) -> Vec<u8> {
    use pecos_rng::prelude::PCGRandom;
    let mut rng = PCGRandom::seed_from_u64(seed);
    let mut bytes = vec![0u8; count];
    rng.fill_bytes(&mut bytes);
    bytes
}

/// Generate random bytes using `RapidRng`
fn generate_rapidrng_bytes(seed: u64, count: usize) -> Vec<u8> {
    use rand::RngCore;
    use rapidhash::rng::RapidRng;
    let mut rng = RapidRng::new(seed);
    let mut bytes = vec![0u8; count];
    rng.fill_bytes(&mut bytes);
    bytes
}

/// Generate random bytes using Xoshiro256++
fn generate_xoshiro_bytes(seed: u64, count: usize) -> Vec<u8> {
    use rand::RngCore;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut bytes = vec![0u8; count];
    rng.fill_bytes(&mut bytes);
    bytes
}

/// Generate random bytes using `PecosRng` (SIMD Xoshiro256++)
fn generate_pecosrng_bytes(seed: u64, count: usize) -> Vec<u8> {
    use pecos_rng::prelude::{PecosRng, RngCore};
    let mut rng = PecosRng::seed_from_u64(seed);
    let mut bytes = vec![0u8; count];
    rng.fill_bytes(&mut bytes);
    bytes
}

/// Generate random bytes using `PecosRng` (parallel `RapidRng`)
fn generate_pecosfastrng_bytes(seed: u64, count: usize) -> Vec<u8> {
    use pecos_rng::prelude::{PecosRng, RngCore};
    let mut rng = PecosRng::seed_from_u64(seed);
    let mut bytes = vec![0u8; count];
    rng.fill_bytes(&mut bytes);
    bytes
}

// ============================================================================
// Tests for PCG64Fast
// ============================================================================

#[test]
fn test_pcg64fast_statistical_quality() {
    let data = generate_pcg64fast_bytes(42, TEST_BYTES);
    let results = run_tests("PCG64Fast", &data);
    results.print_report();
    assert!(
        results.is_acceptable(),
        "PCG64Fast failed statistical quality tests"
    );
}

#[test]
fn test_pcg64fast_multiple_seeds() {
    // Test with multiple seeds to ensure consistency
    for seed in [1, 42, 12345, 98765, 314_159_265] {
        let data = generate_pcg64fast_bytes(seed, TEST_BYTES);
        let results = run_tests("PCG64Fast", &data);
        assert!(
            results.is_acceptable(),
            "PCG64Fast failed with seed {seed}: {results:?}"
        );
    }
}

// ============================================================================
// Tests for PCGRandom (original PCG32)
// ============================================================================

#[test]
fn test_pcgrandom_statistical_quality() {
    let data = generate_pcgrandom_bytes(42, TEST_BYTES);
    let results = run_tests("PCGRandom", &data);
    results.print_report();
    assert!(
        results.is_acceptable(),
        "PCGRandom failed statistical quality tests"
    );
}

// ============================================================================
// Tests for PecosRng (SIMD Xoshiro256++)
// ============================================================================

#[test]
fn test_pecosrng_statistical_quality() {
    let data = generate_pecosrng_bytes(42, TEST_BYTES);
    let results = run_tests("PecosRng", &data);
    results.print_report();
    assert!(
        results.is_acceptable(),
        "PecosRng failed statistical quality tests"
    );
}

// ============================================================================
// Tests for PecosRng (parallel RapidRng)
// ============================================================================

#[test]
fn test_pecosfastrng_statistical_quality() {
    let data = generate_pecosfastrng_bytes(42, TEST_BYTES);
    let results = run_tests("PecosRng", &data);
    results.print_report();
    assert!(
        results.is_acceptable(),
        "PecosRng failed statistical quality tests"
    );
}

// ============================================================================
// Comparison tests with other well-known RNGs
// ============================================================================

#[test]
fn test_rapidrng_statistical_quality() {
    let data = generate_rapidrng_bytes(42, TEST_BYTES);
    let results = run_tests("RapidRng", &data);
    results.print_report();
    assert!(
        results.is_acceptable(),
        "RapidRng failed statistical quality tests"
    );
}

#[test]
fn test_xoshiro_statistical_quality() {
    let data = generate_xoshiro_bytes(42, TEST_BYTES);
    let results = run_tests("Xoshiro256++", &data);
    results.print_report();
    assert!(
        results.is_acceptable(),
        "Xoshiro256++ failed statistical quality tests"
    );
}

// ============================================================================
// Comparison report (run with `cargo test -- --nocapture`)
// ============================================================================

#[test]
fn comparison_report() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║           RNG Statistical Quality Comparison Report              ║");
    println!("║                  (1 MB of random data each)                      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let rngs: Vec<(&str, Vec<u8>)> = vec![
        ("PCG64Fast", generate_pcg64fast_bytes(42, TEST_BYTES)),
        ("PCGRandom", generate_pcgrandom_bytes(42, TEST_BYTES)),
        ("RapidRng", generate_rapidrng_bytes(42, TEST_BYTES)),
        ("Xoshiro256++", generate_xoshiro_bytes(42, TEST_BYTES)),
        ("PecosRng", generate_pecosrng_bytes(42, TEST_BYTES)),
        ("PecosRng", generate_pecosfastrng_bytes(42, TEST_BYTES)),
    ];

    let mut all_pass = true;
    for (name, data) in &rngs {
        let results = run_tests(name, data);
        results.print_report();
        if !results.is_acceptable() {
            all_pass = false;
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!(
        "Overall: {}",
        if all_pass {
            "ALL TESTS PASSED"
        } else {
            "SOME TESTS FAILED"
        }
    );
    println!("═══════════════════════════════════════════════════════════════════\n");
}
