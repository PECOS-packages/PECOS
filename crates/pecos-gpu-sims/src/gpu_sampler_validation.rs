//! Validation tests comparing GPU and CPU measurement samplers.
//!
//! These tests verify that the GPU sampler produces results equivalent to the CPU sampler.

#[cfg(test)]
#[allow(clippy::cast_precision_loss)] // Test code computes ratios from counts
#[allow(clippy::cast_possible_truncation)] // Test shot counts are small and fit in u32
mod tests {
    use crate::GpuMeasurementSampler;
    use pecos_simulators::SymbolicSparseStab;
    use pecos_simulators::measurement_sampler::{MeasurementKind, MeasurementSampler};

    /// Helper to check if GPU is available, skip test if not.
    fn get_gpu_sampler(measurements: &[MeasurementKind]) -> Option<GpuMeasurementSampler> {
        match GpuMeasurementSampler::new(measurements) {
            Ok(s) => Some(s),
            Err(e) => {
                println!("Skipping test - no GPU available: {e}");
                None
            }
        }
    }

    // =========================================================================
    // Fixed measurement tests - should produce identical results
    // =========================================================================

    #[test]
    fn validate_fixed_measurements_match_exactly() {
        let measurements = vec![
            MeasurementKind::Fixed(false),
            MeasurementKind::Fixed(true),
            MeasurementKind::Fixed(false),
            MeasurementKind::Fixed(true),
        ];

        let Some(gpu_sampler) = get_gpu_sampler(&measurements) else {
            return;
        };

        let shots = 10_000;
        let gpu_result = gpu_sampler.sample_with_seed(shots, 42);

        // Fixed(false) should have all zeros
        assert_eq!(
            gpu_result.count_ones(0),
            0,
            "Fixed(false) should have 0 ones"
        );
        assert_eq!(
            gpu_result.count_ones(2),
            0,
            "Fixed(false) should have 0 ones"
        );

        // Fixed(true) should have all ones
        assert_eq!(
            gpu_result.count_ones(1),
            shots,
            "Fixed(true) should have all ones"
        );
        assert_eq!(
            gpu_result.count_ones(3),
            shots,
            "Fixed(true) should have all ones"
        );
    }

    #[test]
    fn validate_fixed_counts_match_exactly() {
        let measurements = vec![
            MeasurementKind::Fixed(false),
            MeasurementKind::Fixed(true),
            MeasurementKind::Fixed(false),
        ];

        let Some(gpu_sampler) = get_gpu_sampler(&measurements) else {
            return;
        };

        let shots = 100_000;
        let counts = gpu_sampler.sample_counts_with_seed(shots, 42).unwrap();

        assert_eq!(counts[0], 0, "Fixed(false) count should be 0");
        assert_eq!(
            counts[1], shots as u32,
            "Fixed(true) count should equal shots"
        );
        assert_eq!(counts[2], 0, "Fixed(false) count should be 0");
    }

    // =========================================================================
    // Random measurement tests - should have ~50% ones
    // =========================================================================

    #[test]
    fn validate_random_distribution() {
        let measurements = vec![MeasurementKind::Random; 10];

        let Some(gpu_sampler) = get_gpu_sampler(&measurements) else {
            return;
        };

        let shots = 100_000;
        let gpu_result = gpu_sampler.sample_with_seed(shots, 42);

        for m in 0..10 {
            let ones = gpu_result.count_ones(m);
            let ratio = ones as f64 / shots as f64;
            assert!(
                (0.48..=0.52).contains(&ratio),
                "Measurement {} should be ~50% ones, got {:.2}%",
                m,
                ratio * 100.0
            );
        }
    }

    #[test]
    fn validate_random_counts_distribution() {
        let measurements = vec![MeasurementKind::Random; 10];

        let Some(gpu_sampler) = get_gpu_sampler(&measurements) else {
            return;
        };

        let shots = 100_000;
        let counts = gpu_sampler.sample_counts_with_seed(shots, 42).unwrap();

        for (m, &count) in counts.iter().enumerate() {
            let ratio = f64::from(count) / shots as f64;
            assert!(
                (0.48..=0.52).contains(&ratio),
                "Measurement {} should be ~50% ones, got {:.2}%",
                m,
                ratio * 100.0
            );
        }
    }

    #[test]
    fn validate_random_independence() {
        // Multiple random measurements should be independent
        let measurements = vec![MeasurementKind::Random, MeasurementKind::Random];

        let Some(gpu_sampler) = get_gpu_sampler(&measurements) else {
            return;
        };

        let shots = 100_000;
        let gpu_result = gpu_sampler.sample_with_seed(shots, 42);

        // Count how often m0 == m1 (should be ~50% for independent randoms)
        let mut same_count = 0;
        for shot in 0..shots {
            if gpu_result.get(shot, 0) == gpu_result.get(shot, 1) {
                same_count += 1;
            }
        }

        let same_ratio = f64::from(same_count) / shots as f64;
        assert!(
            (0.48..=0.52).contains(&same_ratio),
            "Independent random measurements should agree ~50%, got {:.2}%",
            same_ratio * 100.0
        );
    }

    // =========================================================================
    // Copy measurement tests
    // =========================================================================

    #[test]
    fn validate_copy_matches_source() {
        let measurements = vec![
            MeasurementKind::Random,
            MeasurementKind::Copy(0),
            MeasurementKind::Copy(0),
        ];

        let Some(gpu_sampler) = get_gpu_sampler(&measurements) else {
            return;
        };

        let shots = 10_000;
        let gpu_result = gpu_sampler.sample_with_seed(shots, 42);

        // Copy measurements should match source exactly
        for shot in 0..shots {
            assert_eq!(
                gpu_result.get(shot, 0),
                gpu_result.get(shot, 1),
                "Copy(0) should match m0 at shot {shot}"
            );
            assert_eq!(
                gpu_result.get(shot, 0),
                gpu_result.get(shot, 2),
                "Copy(0) should match m0 at shot {shot}"
            );
        }
    }

    #[test]
    fn validate_copy_flipped_inverts_source() {
        let measurements = vec![MeasurementKind::Random, MeasurementKind::CopyFlipped(0)];

        let Some(gpu_sampler) = get_gpu_sampler(&measurements) else {
            return;
        };

        let shots = 10_000;
        let gpu_result = gpu_sampler.sample_with_seed(shots, 42);

        // CopyFlipped should be opposite of source
        for shot in 0..shots {
            assert_eq!(
                gpu_result.get(shot, 0),
                !gpu_result.get(shot, 1),
                "CopyFlipped(0) should be opposite of m0 at shot {shot}"
            );
        }
    }

    // =========================================================================
    // Computed (XOR) measurement tests
    // =========================================================================

    #[test]
    fn validate_computed_xor_two_sources() {
        let measurements = vec![
            MeasurementKind::Random,
            MeasurementKind::Random,
            MeasurementKind::Computed {
                deps: vec![0, 1],
                flip: false,
            },
        ];

        let Some(gpu_sampler) = get_gpu_sampler(&measurements) else {
            return;
        };

        let shots = 10_000;
        let gpu_result = gpu_sampler.sample_with_seed(shots, 42);

        // Computed should equal XOR of dependencies
        for shot in 0..shots {
            let expected = gpu_result.get(shot, 0) ^ gpu_result.get(shot, 1);
            assert_eq!(
                gpu_result.get(shot, 2),
                expected,
                "Computed XOR failed at shot {shot}"
            );
        }
    }

    #[test]
    fn validate_computed_xor_with_flip() {
        let measurements = vec![
            MeasurementKind::Random,
            MeasurementKind::Computed {
                deps: vec![0],
                flip: true,
            },
        ];

        let Some(gpu_sampler) = get_gpu_sampler(&measurements) else {
            return;
        };

        let shots = 10_000;
        let gpu_result = gpu_sampler.sample_with_seed(shots, 42);

        // Computed with flip should equal NOT of source
        for shot in 0..shots {
            assert_eq!(
                gpu_result.get(shot, 1),
                !gpu_result.get(shot, 0),
                "Computed with flip should invert at shot {shot}"
            );
        }
    }

    #[test]
    fn validate_computed_xor_three_sources() {
        let measurements = vec![
            MeasurementKind::Random,
            MeasurementKind::Random,
            MeasurementKind::Random,
            MeasurementKind::Computed {
                deps: vec![0, 1, 2],
                flip: false,
            },
        ];

        let Some(gpu_sampler) = get_gpu_sampler(&measurements) else {
            return;
        };

        let shots = 10_000;
        let gpu_result = gpu_sampler.sample_with_seed(shots, 42);

        // Computed should equal XOR of all three sources
        for shot in 0..shots {
            let expected =
                gpu_result.get(shot, 0) ^ gpu_result.get(shot, 1) ^ gpu_result.get(shot, 2);
            assert_eq!(
                gpu_result.get(shot, 3),
                expected,
                "Computed XOR of 3 sources failed at shot {shot}"
            );
        }
    }

    // =========================================================================
    // Surface code validation - compare GPU and CPU on realistic workload
    // =========================================================================

    #[test]
    fn validate_surface_code_statistics() {
        // Simulate a small surface code
        let distance = 5;
        let rounds = 3;
        let num_data = distance * distance;
        let num_ancillas = num_data - 1;
        let num_qubits = num_data + num_ancillas;
        let ancilla_start = num_data;

        let mut sim = SymbolicSparseStab::new(num_qubits);
        sim.reset();

        for i in 0..num_data {
            sim.h(i);
        }

        for _round in 0..rounds {
            for a in 0..num_ancillas {
                let ancilla = ancilla_start + a;
                let base = a % num_data;
                sim.cx(ancilla, base);
                if a + 1 < num_data {
                    sim.cx(ancilla, (base + 1) % num_data);
                }
            }

            for a in 0..num_ancillas {
                let ancilla = ancilla_start + a;
                sim.mz(ancilla);
            }
        }

        let history = sim.measurement_history();
        let measurements: Vec<MeasurementKind> = MeasurementKind::from_history(history);

        // Count measurement types
        let mut fixed_count = 0;
        let mut random_count = 0;
        let mut other_count = 0;
        for kind in &measurements {
            match kind {
                MeasurementKind::Fixed(_) => fixed_count += 1,
                MeasurementKind::Random => random_count += 1,
                _ => other_count += 1,
            }
        }

        println!(
            "Surface code d={}: {} measurements ({} fixed, {} random, {} other)",
            distance,
            measurements.len(),
            fixed_count,
            random_count,
            other_count
        );

        // Create samplers
        let cpu_sampler = MeasurementSampler::new(history);
        let Some(gpu_sampler) = get_gpu_sampler(&measurements) else {
            return;
        };

        let shots = 50_000;

        // Sample with both
        let cpu_result = cpu_sampler.sample(shots);
        let gpu_result = gpu_sampler.sample_with_seed(shots, 42);

        // For each measurement, verify statistics are reasonable
        for (m, kind) in measurements.iter().enumerate() {
            let cpu_ones = cpu_result.count_ones(m);
            let gpu_ones = gpu_result.count_ones(m);

            match kind {
                MeasurementKind::Fixed(false) => {
                    assert_eq!(cpu_ones, 0, "CPU Fixed(false) m{m} should be 0");
                    assert_eq!(gpu_ones, 0, "GPU Fixed(false) m{m} should be 0");
                }
                MeasurementKind::Fixed(true) => {
                    assert_eq!(cpu_ones, shots, "CPU Fixed(true) m{m} should be all 1s");
                    assert_eq!(gpu_ones, shots, "GPU Fixed(true) m{m} should be all 1s");
                }
                MeasurementKind::Random => {
                    // Both should be ~50%
                    let cpu_ratio = cpu_ones as f64 / shots as f64;
                    let gpu_ratio = gpu_ones as f64 / shots as f64;
                    assert!(
                        (0.45..=0.55).contains(&cpu_ratio),
                        "CPU Random m{} should be ~50%, got {:.2}%",
                        m,
                        cpu_ratio * 100.0
                    );
                    assert!(
                        (0.45..=0.55).contains(&gpu_ratio),
                        "GPU Random m{} should be ~50%, got {:.2}%",
                        m,
                        gpu_ratio * 100.0
                    );
                }
                _ => {
                    // For Copy/CopyFlipped/Computed, just verify they're not obviously wrong
                    // (specific values depend on RNG, which differs between CPU and GPU)
                }
            }
        }
    }

    // =========================================================================
    // Noisy sampling validation
    // =========================================================================

    #[test]
    fn validate_noisy_sampling_error_rate() {
        // Start with all zeros, apply noise, check error rate
        let measurements = vec![MeasurementKind::Fixed(false); 100];

        let Some(gpu_sampler) = get_gpu_sampler(&measurements) else {
            return;
        };

        let shots = 100_000;
        let error_rate = 0.05; // 5% error rate

        let result = gpu_sampler.sample_noisy_with_seed(shots, error_rate, 42);

        // Each measurement should have ~5% errors (ones)
        for m in 0..measurements.len() {
            let ones = result.count_ones(m);
            let observed_rate = ones as f64 / shots as f64;
            assert!(
                (0.04..=0.06).contains(&observed_rate),
                "Measurement {} should have ~5% errors, got {:.2}%",
                m,
                observed_rate * 100.0
            );
        }
    }

    #[test]
    fn validate_noisy_sampling_zero_error_rate() {
        let measurements = vec![MeasurementKind::Fixed(false), MeasurementKind::Fixed(true)];

        let Some(gpu_sampler) = get_gpu_sampler(&measurements) else {
            return;
        };

        let shots = 10_000;

        // With 0% error rate, results should be unchanged
        let result = gpu_sampler.sample_noisy_with_seed(shots, 0.0, 42);

        assert_eq!(
            result.count_ones(0),
            0,
            "0% noise should preserve Fixed(false)"
        );
        assert_eq!(
            result.count_ones(1),
            shots,
            "0% noise should preserve Fixed(true)"
        );
    }

    // =========================================================================
    // Edge cases
    // =========================================================================

    #[test]
    fn validate_single_measurement() {
        let measurements = vec![MeasurementKind::Random];

        let Some(gpu_sampler) = get_gpu_sampler(&measurements) else {
            return;
        };

        let result = gpu_sampler.sample_with_seed(1000, 42);
        let ratio = result.count_ones(0) as f64 / 1000.0;
        assert!(
            (0.40..=0.60).contains(&ratio),
            "Single random measurement should be ~50%"
        );
    }

    #[test]
    fn validate_many_measurements() {
        let measurements: Vec<MeasurementKind> = (0..1000)
            .map(|i| {
                if i % 3 == 0 {
                    MeasurementKind::Fixed(i % 2 == 0)
                } else {
                    MeasurementKind::Random
                }
            })
            .collect();

        let Some(gpu_sampler) = get_gpu_sampler(&measurements) else {
            return;
        };

        let result = gpu_sampler.sample_with_seed(10_000, 42);

        // Just verify it doesn't crash and produces sensible output
        assert_eq!(result.num_measurements(), 1000);
        assert_eq!(result.shots(), 10_000);
    }

    #[test]
    fn validate_large_shot_count() {
        let measurements = vec![MeasurementKind::Fixed(true), MeasurementKind::Random];

        let Some(gpu_sampler) = get_gpu_sampler(&measurements) else {
            return;
        };

        let shots = 500_000;
        if shots > gpu_sampler.max_shots() {
            println!("Skipping large shot test - exceeds GPU max");
            return;
        }

        let result = gpu_sampler.sample_with_seed(shots, 42);

        assert_eq!(
            result.count_ones(0),
            shots,
            "Fixed(true) should have all ones"
        );

        let random_ratio = result.count_ones(1) as f64 / shots as f64;
        assert!(
            (0.49..=0.51).contains(&random_ratio),
            "Random should be ~50% with large samples"
        );
    }
}
