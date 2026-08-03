use pecos_engines::byte_message::ByteMessageBuilder;
use pecos_engines::quantum::QuantumEngine;
use pecos_engines::quantum::{SparseStabEngine, StabVecEngine, StateVecEngine};
use pecos_engines::{Engine, QuantumSystem};

/// MPZ measures, records the outcome, and leaves the qubit prepared in |0>.
/// All three executor paths are exercised: the state-vector engine, the
/// Clifford-message loop (sparse stabilizer), and the general-message loop
/// (stab-vec) each dispatch through a different gate match.
#[test]
fn mpz_records_the_flip_and_resets_the_qubit() {
    let engines: Vec<(&str, Box<dyn QuantumEngine>)> = vec![
        ("state_vec", Box::new(StateVecEngine::new(1))),
        ("sparse_stab", Box::new(SparseStabEngine::new(1))),
        ("stab_vec", Box::new(StabVecEngine::new(1))),
    ];
    for (name, engine) in engines {
        let mut system = QuantumSystem::new_without_noise(engine);

        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.x(&[0]);
        builder.mpz(&[0]);
        builder.mz(&[0]);

        let circuit = builder.build();
        let result = system.process(circuit).unwrap();
        let outcomes = result.outcomes().unwrap();

        assert_eq!(
            outcomes,
            vec![1, 0],
            "{name}: the MPZ reads the flipped state and resets it; a plain \
             MZ here would read [1, 1]"
        );
    }
}

/// The general noise model tracks MPZ as a measurement: a noiseless run must
/// complete with the right outcome count (it previously died on an
/// outcome-count mismatch after routing MPZ through 1q-depolarizing noise).
#[test]
fn mpz_runs_through_the_general_noise_model() {
    use pecos_engines::noise::general::GeneralNoiseModel;

    let noise = GeneralNoiseModel::builder()
        .with_prep_probability(0.0)
        .with_meas_0_probability(0.0)
        .with_meas_1_probability(0.0)
        .with_p1_probability(0.0)
        .with_p2_probability(0.0)
        .with_seed(7)
        .build();
    let engine = Box::new(StateVecEngine::new(1));
    let mut system = QuantumSystem::new(Box::new(noise), engine);

    let mut builder = ByteMessageBuilder::new();
    let _ = builder.for_quantum_operations();
    builder.x(&[0]);
    builder.mpz(&[0]);
    builder.mz(&[0]);

    let circuit = builder.build();
    let result = system.process(circuit).unwrap();
    let outcomes = result.outcomes().unwrap();
    assert_eq!(outcomes, vec![1, 0]);
}
