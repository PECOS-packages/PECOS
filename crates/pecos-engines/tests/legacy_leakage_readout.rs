use pecos_engines::noise::GeneralNoiseModel;
use pecos_engines::{
    ByteMessage, ControlEngine, Engine, EngineStage, GateType, StateVecEngine,
    quantum_system::QuantumSystem,
};

#[test]
fn legacy_readouts_keep_leakage_before_later_reset_or_leak() {
    for ternary in [false, true] {
        for initially_leaked in [false, true] {
            let mut model = GeneralNoiseModel::builder()
                .with_p_prep(1.0)
                .with_prep_leak_ratio(if initially_leaked { 0.0 } else { 1.0 })
                .build();
            if initially_leaked {
                model.mark_as_leaked(0);
            }
            let mut system = QuantumSystem::new(Box::new(model), Box::new(StateVecEngine::new(1)));
            let mut input = ByteMessage::quantum_operations_builder();
            if ternary {
                input.measure_leakages(&[0]);
            } else {
                input.mz(&[0]);
            }
            input.pz(&[0]);
            if ternary {
                input.measure_leakages(&[0]);
            } else {
                input.mz(&[0]);
            }
            let output = system.process(input.build()).unwrap().outcomes().unwrap();
            let leaked = if ternary { 2 } else { 1 };
            assert_eq!(
                output,
                if initially_leaked {
                    vec![leaked, 1]
                } else {
                    vec![0, leaked]
                }
            );
        }
    }
}

#[test]
fn mpz_reads_leakage_then_clears_it_even_when_noiseless() {
    for noiseless in [false, true] {
        let mut builder = GeneralNoiseModel::builder();
        if noiseless {
            builder = builder.with_noiseless_gate(GateType::MPZ);
        }
        let mut model = builder.build();
        model.mark_as_leaked(0);
        let mut system = QuantumSystem::new(Box::new(model), Box::new(StateVecEngine::new(1)));
        let input = ByteMessage::quantum_operations_builder()
            .mpz(&[0])
            .measure_leakages(&[0])
            .x(&[0])
            .mz(&[0])
            .build();
        assert_eq!(
            system.process(input).unwrap().outcomes().unwrap(),
            vec![1, 0, 1]
        );
    }
}

#[test]
fn crosstalk_continuation_cannot_rewrite_an_already_executed_readout() {
    use std::collections::BTreeMap;
    let model = GeneralNoiseModel::builder()
        .with_p_meas_crosstalk(1.0)
        .with_p_meas_crosstalk_model(&BTreeMap::from([
            ("0->L".into(), 1.0),
            ("1->L".into(), 1.0),
        ]))
        .build();
    let mut system = QuantumSystem::new(Box::new(model), Box::new(StateVecEngine::new(2)));
    // The payload inserts a measurement of q0. Its transition only executes in
    // the continuation, after the user readout in this first simulator call.
    let input = ByteMessage::quantum_operations_builder()
        .pz(&[0, 1])
        .meas_crosstalk_global_payload(&[1])
        .measure_leakages(&[0])
        .build();
    assert_eq!(system.process(input).unwrap().outcomes().unwrap(), vec![0]);
    let next = ByteMessage::quantum_operations_builder()
        .measure_leakages(&[0])
        .build();
    assert_eq!(system.process(next).unwrap().outcomes().unwrap(), vec![2]);
}

#[test]
fn preparation_crosstalk_results_do_not_shift_user_readouts() {
    let mut model = GeneralNoiseModel::builder()
        .with_p_prep_crosstalk(1.0)
        .build();
    // Register q1 as a crosstalk victim, then leak q0 after preparation.
    let first = ByteMessage::quantum_operations_builder()
        .pz(&[0, 1])
        .build();
    model.start(first).unwrap();
    assert!(matches!(
        model
            .continue_processing(ByteMessage::create_empty())
            .unwrap(),
        EngineStage::Complete(_)
    ));
    model.mark_as_leaked(0);
    let mut system = QuantumSystem::new(Box::new(model), Box::new(StateVecEngine::new(2)));
    let input = ByteMessage::quantum_operations_builder()
        .measure_leakages(&[0])
        .pz(&[0])
        .measure_leakages(&[0])
        .build();
    assert_eq!(
        system.process(input).unwrap().outcomes().unwrap(),
        vec![2, 0]
    );
}
