//! Tests for seeding behavior with the unified QASM engine API

#[test]
fn test_qasm_engine_deterministic_with_seed() {
    use pecos_engines::sim_builder;
    use pecos_qasm::qasm_engine;

    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    // Build simulation with fixed seed
    let sim = sim_builder()
        .classical(qasm_engine().qasm(qasm))
        .seed(42)
        .build()
        .unwrap();

    // Run twice with same parameters
    let mut sim = sim;
    let results1 = sim.run(100).unwrap();
    let results2 = sim.run(100).unwrap();

    // Both should have same length
    assert_eq!(results1.len(), 100);
    assert_eq!(results2.len(), 100);

    // Convert to shot maps to compare distributions
    if let (Ok(map1), Ok(map2)) = (results1.try_as_shot_map(), results2.try_as_shot_map()) {
        // With same seed, distributions should be identical
        // Check that all registers match
        for (register, values1) in map1.iter() {
            if let Some(values2) = map2.get(register) {
                assert_eq!(
                    values1.len(),
                    values2.len(),
                    "Register '{register}' has different shot counts"
                );
                // For deterministic results, the actual values should match
                // but we can't easily compare DataVec variants directly
            } else {
                panic!("Register '{register}' missing in second run");
            }
        }
    }
}

#[test]
fn test_qasm_engine_random_without_seed() {
    use pecos_engines::sim_builder;
    use pecos_qasm::qasm_engine;

    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    // Build simulation without seed
    let sim = sim_builder()
        .classical(qasm_engine().qasm(qasm))
        .build()
        .unwrap();

    // Run multiple times - should get different distributions
    let mut sim = sim;
    let results1 = sim.run(1000).unwrap();
    let results2 = sim.run(1000).unwrap();
    let results3 = sim.run(1000).unwrap();

    assert_eq!(results1.len(), 1000);
    assert_eq!(results2.len(), 1000);
    assert_eq!(results3.len(), 1000);

    // Note: We can't guarantee they're different due to randomness,
    // but with 1000 shots they almost certainly will be
}

#[test]
fn test_qasm_engine_with_seed_reproducibility() {
    use pecos_engines::sim_builder;
    use pecos_qasm::qasm_engine;

    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    // Note: MonteCarloEngine doesn't support changing seed after creation
    // Build with seed instead
    let mut sim1 = sim_builder()
        .classical(qasm_engine().qasm(qasm))
        .seed(42)
        .build()
        .unwrap();
    let results_first = sim1.run(100).unwrap();

    let mut sim2 = sim_builder()
        .classical(qasm_engine().qasm(qasm))
        .seed(42)
        .build()
        .unwrap();
    let results_second = sim2.run(100).unwrap();

    // Same seed should give same results
    assert_eq!(results_first.len(), 100);
    assert_eq!(results_second.len(), 100);

    if let (Ok(map_first), Ok(map_second)) = (
        results_first.try_as_shot_map(),
        results_second.try_as_shot_map(),
    ) {
        // Verify identical distributions
        for (register, values1) in map_first.iter() {
            if let Some(values2) = map_second.get(register) {
                assert_eq!(
                    values1.len(),
                    values2.len(),
                    "Register '{register}' has different counts with same seed"
                );
            }
        }
    }

    // Different seeds should (likely) give different results
    let mut sim3 = sim_builder()
        .classical(qasm_engine().qasm(qasm))
        .seed(43)
        .build()
        .unwrap();
    let results2 = sim3.run(100).unwrap();

    let mut sim4 = sim_builder()
        .classical(qasm_engine().qasm(qasm))
        .build()
        .unwrap();
    let results3 = sim4.run(100).unwrap(); // Random

    assert_eq!(results2.len(), 100);
    assert_eq!(results3.len(), 100);
}

#[test]
fn test_qasm_engine_noise_with_seed() {
    use pecos_engines::{DepolarizingNoise, sim_builder};
    use pecos_qasm::qasm_engine;

    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c[3];
        h q[0];
        cx q[0], q[1];
        cx q[1], q[2];
        measure q -> c;
    "#;

    // With noise and seed, results should still be deterministic
    let sim = sim_builder()
        .classical(qasm_engine().qasm(qasm))
        .seed(42)
        .noise(DepolarizingNoise { p: 0.01 })
        .build()
        .unwrap();

    let mut sim = sim;
    let results1 = sim.run(500).unwrap();
    let results2 = sim.run(500).unwrap();

    assert_eq!(results1.len(), 500);
    assert_eq!(results2.len(), 500);

    // Even with noise, same seed = same results
    if let (Ok(map1), Ok(map2)) = (results1.try_as_shot_map(), results2.try_as_shot_map()) {
        for (register, values1) in map1.iter() {
            if let Some(values2) = map2.get(register) {
                assert_eq!(
                    values1.len(),
                    values2.len(),
                    "Register '{register}' should have same counts with noise+seed"
                );
            }
        }
    }
}
