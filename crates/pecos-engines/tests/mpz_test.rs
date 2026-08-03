use pecos_engines::byte_message::ByteMessageBuilder;
use pecos_engines::quantum::StateVecEngine;
use pecos_engines::{Engine, QuantumSystem};

/// MPZ measures, records the outcome, and leaves the qubit prepared in |0>.
#[test]
fn mpz_records_the_flip_and_resets_the_qubit() {
    let engine = Box::new(StateVecEngine::new(1));
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
        "the MPZ reads the flipped state and resets it; a plain MZ here \
         would read [1, 1]"
    );
}
