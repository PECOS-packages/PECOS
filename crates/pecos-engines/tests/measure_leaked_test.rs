use pecos_engines::byte_message::ByteMessageBuilder;
use pecos_engines::noise::general::GeneralNoiseModel;
use pecos_engines::quantum::StateVecEngine;
use pecos_engines::{Engine, QuantumSystem};

#[test]
fn test_measure_leaked_basic_functionality() {
    // Create a simple 2-qubit system with no noise
    let engine = Box::new(StateVecEngine::new(2));
    let mut system = QuantumSystem::new_without_noise(engine);

    // Test 1: MeasureLeaked behaves like Measure without leakage
    let mut builder = ByteMessageBuilder::new();
    let _ = builder.for_quantum_operations();
    builder.h(&[0]); // Create superposition
    builder.measure_leakages(&[0, 1]); // MeasureLeaked on both qubits

    let circuit = builder.build();
    let result = system.process(circuit).unwrap();
    let outcomes = result.outcomes().unwrap();

    assert_eq!(outcomes.len(), 2);
    assert!(
        outcomes[0] <= 1,
        "MeasureLeaked without leakage should return 0 or 1"
    );
    assert_eq!(outcomes[1], 0, "Qubit 1 should be in |0⟩ state");
}

#[test]
fn test_measure_leaked_with_general_noise_model() {
    // Create a noise model
    let mut noise_model = GeneralNoiseModel::builder()
        .with_prep_probability(0.0)
        .with_meas_0_probability(0.0)
        .with_meas_1_probability(0.0)
        .with_p1_probability(0.0)
        .with_p2_probability(0.0)
        .with_seed(42)
        .build();

    // Manually mark qubits as leaked
    noise_model.mark_as_leaked(0);
    noise_model.mark_as_leaked(2);

    let engine = Box::new(StateVecEngine::new(3));
    let mut system = QuantumSystem::new(Box::new(noise_model), engine);

    // Create measurement circuit
    let mut builder = ByteMessageBuilder::new();
    let _ = builder.for_quantum_operations();

    // Mix of regular Measure and MeasureLeaked
    builder.mz(&[0]); // Regular measure on leaked qubit 0
    builder.measure_leakages(&[1]); // MeasureLeaked on non-leaked qubit 1
    builder.measure_leakages(&[2]); // MeasureLeaked on leaked qubit 2

    let circuit = builder.build();
    let result = system.process(circuit).unwrap();
    let outcomes = result.outcomes().unwrap();

    assert_eq!(outcomes.len(), 3);
    assert_eq!(
        outcomes[0], 1,
        "Regular Measure on leaked qubit should return 1"
    );
    assert_eq!(
        outcomes[1], 0,
        "MeasureLeaked on non-leaked qubit should return 0"
    );
    assert_eq!(
        outcomes[2], 2,
        "MeasureLeaked on leaked qubit should return 2"
    );
}

#[test]
fn test_measure_leaked_preserves_quantum_state() {
    // Verify that MeasureLeaked correctly measures quantum states when no leakage
    let engine = Box::new(StateVecEngine::new(2));
    let mut system = QuantumSystem::new_without_noise(engine);

    // Create Bell state
    let mut builder = ByteMessageBuilder::new();
    let _ = builder.for_quantum_operations();
    builder.h(&[0]);
    builder.cx(&[(0, 1)]);
    builder.measure_leakages(&[0, 1]);

    let circuit = builder.build();

    // Run multiple times to check correlation
    let mut same_results = 0;
    let runs = 100;

    for _ in 0..runs {
        system.reset().unwrap();
        let result = system.process(circuit.clone()).unwrap();
        let outcomes = result.outcomes().unwrap();

        assert_eq!(outcomes.len(), 2);
        assert!(
            outcomes[0] <= 1 && outcomes[1] <= 1,
            "No leakage should occur"
        );

        if outcomes[0] == outcomes[1] {
            same_results += 1;
        }
    }

    // Bell state should have perfect correlation
    assert_eq!(
        same_results, runs,
        "Bell state measurements should be perfectly correlated"
    );
}

#[test]
fn test_measure_leaked_sequential_measurements() {
    // Test that leaked state persists across multiple measurements
    let mut noise_model = GeneralNoiseModel::builder()
        .with_prep_probability(0.0)
        .with_meas_0_probability(0.0)
        .with_meas_1_probability(0.0)
        .with_p1_probability(0.0)
        .with_p2_probability(0.0)
        .with_seed(42)
        .build();

    // Manually mark qubit as leaked
    noise_model.mark_as_leaked(0);

    let engine = Box::new(StateVecEngine::new(1));
    let mut system = QuantumSystem::new(Box::new(noise_model), engine);

    // First circuit: measure the leaked qubit
    let mut builder = ByteMessageBuilder::new();
    let _ = builder.for_quantum_operations();
    builder.measure_leakages(&[0]);

    let circuit1 = builder.build();
    let result1 = system.process(circuit1).unwrap();
    let outcomes1 = result1.outcomes().unwrap();

    assert_eq!(
        outcomes1[0], 2,
        "First MeasureLeaked should return 2 for leaked qubit"
    );

    // Second circuit: measure the same qubit again (should still be leaked)
    let mut builder = ByteMessageBuilder::new();
    let _ = builder.for_quantum_operations();
    builder.measure_leakages(&[0]);

    let circuit2 = builder.build();
    let result2 = system.process(circuit2).unwrap();
    let outcomes2 = result2.outcomes().unwrap();

    assert_eq!(
        outcomes2[0], 2,
        "Second MeasureLeaked should still return 2"
    );

    // Third circuit: regular measurement (should return 1)
    let mut builder = ByteMessageBuilder::new();
    let _ = builder.for_quantum_operations();
    builder.mz(&[0]);

    let circuit3 = builder.build();
    let result3 = system.process(circuit3).unwrap();
    let outcomes3 = result3.outcomes().unwrap();

    assert_eq!(
        outcomes3[0], 1,
        "Regular Measure should return 1 for leaked qubit"
    );
}

#[test]
fn test_measure_leaked_with_prep_unleaks() {
    // Test that Prep operation unleaks qubits
    let mut noise_model = GeneralNoiseModel::builder()
        .with_prep_probability(0.0)
        .with_meas_0_probability(0.0)
        .with_meas_1_probability(0.0)
        .with_p1_probability(0.0)
        .with_p2_probability(0.0)
        .with_seed(42)
        .build();

    // Manually mark qubit as leaked
    noise_model.mark_as_leaked(0);

    let engine = Box::new(StateVecEngine::new(1));
    let mut system = QuantumSystem::new(Box::new(noise_model), engine);

    // First circuit: measure the leaked qubit
    let mut builder = ByteMessageBuilder::new();
    let _ = builder.for_quantum_operations();
    builder.measure_leakages(&[0]); // Should return 2

    let circuit1 = builder.build();
    let result1 = system.process(circuit1).unwrap();
    let outcomes1 = result1.outcomes().unwrap();

    assert_eq!(outcomes1.len(), 1);
    assert_eq!(outcomes1[0], 2, "MeasureLeaked before prep should return 2");

    // Second circuit: prep (unleak) and measure again
    let mut builder = ByteMessageBuilder::new();
    let _ = builder.for_quantum_operations();
    builder.pz(&[0]); // Unleak the qubit
    builder.measure_leakages(&[0]); // Should return 0 (back to |0⟩)

    let circuit2 = builder.build();
    let result2 = system.process(circuit2).unwrap();
    let outcomes2 = result2.outcomes().unwrap();

    assert_eq!(outcomes2.len(), 1);
    assert_eq!(
        outcomes2[0], 0,
        "MeasureLeaked after prep should return 0 (unleaked)"
    );
}
