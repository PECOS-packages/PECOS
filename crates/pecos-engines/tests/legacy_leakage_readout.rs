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
fn crosstalk_transitions_precede_later_readout_and_reset() {
    use std::collections::BTreeMap;
    for local in [false, true] {
        for leak in [false, true] {
            for reset in [false, true] {
                let transitions = if leak {
                    [("0->L", 1.0), ("1->L", 1.0)]
                } else {
                    [("0->1", 1.0), ("1->0", 1.0)]
                };
                let model = GeneralNoiseModel::builder()
                    .with_p_meas_crosstalk(1.0)
                    .with_p_meas_crosstalk_model(
                        &transitions
                            .into_iter()
                            .map(|(k, v)| (k.into(), v))
                            .collect::<BTreeMap<_, _>>(),
                    )
                    .build();
                let mut system =
                    QuantumSystem::new(Box::new(model), Box::new(StateVecEngine::new(2)));
                let mut input = ByteMessage::quantum_operations_builder();
                input.pz(&[0, 1]).measure_leakages(&[0]);
                if local {
                    input.meas_crosstalk_local_payload(&[0]);
                } else {
                    input.meas_crosstalk_global_payload(&[1]);
                }
                if reset {
                    input.pz(&[0]);
                }
                input.measure_leakages(&[0]);
                let want = if reset {
                    0
                } else if leak {
                    2
                } else {
                    1
                };
                assert_eq!(
                    system.process(input.build()).unwrap().outcomes().unwrap(),
                    vec![0, want],
                    "local={local}, leak={leak}, reset={reset}"
                );
                let next = ByteMessage::quantum_operations_builder()
                    .measure_leakages(&[0])
                    .build();
                assert_eq!(
                    system.process(next).unwrap().outcomes().unwrap(),
                    vec![want]
                );
            }
        }
    }
}

#[test]
fn multiple_crosstalk_sites_apply_each_transition_once() {
    use std::collections::BTreeMap;
    let model = GeneralNoiseModel::builder()
        .with_p_meas_crosstalk(1.0)
        .with_p_meas_crosstalk_model(&BTreeMap::from([
            ("0->1".into(), 1.0),
            ("1->0".into(), 1.0),
        ]))
        .build();
    let mut system = QuantumSystem::new(Box::new(model), Box::new(StateVecEngine::new(1)));
    let input = ByteMessage::quantum_operations_builder()
        .pz(&[0])
        .meas_crosstalk_local_payload(&[0])
        .mz(&[0])
        .meas_crosstalk_local_payload(&[0])
        .mz(&[0])
        .build();
    assert_eq!(
        system.process(input).unwrap().outcomes().unwrap(),
        vec![1, 0]
    );
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

#[test]
fn crosstalk_error_and_clone_require_reset_before_resuming() {
    let mut model = GeneralNoiseModel::builder()
        .with_p_meas_crosstalk(1.0)
        .build();
    let input = ByteMessage::quantum_operations_builder()
        .pz(&[0])
        .meas_crosstalk_local_payload(&[0])
        .x(&[0])
        .mz(&[0])
        .build();
    let EngineStage::NeedsProcessing(first) = model.start(input.clone()).unwrap() else {
        panic!("expected measurement dispatch");
    };
    assert_eq!(
        first
            .quantum_ops()
            .unwrap()
            .iter()
            .map(|g| g.gate_type)
            .collect::<Vec<_>>(),
        vec![GateType::PZ, GateType::MZ]
    );
    assert!(model.start(input.clone()).is_err());
    assert!(model.apply_noise_on_start(&input).is_err());
    assert!(
        model
            .continue_processing(ByteMessage::create_empty())
            .is_err()
    );
    let mut cloned = model.clone();
    for noise in [&mut model, &mut cloned] {
        let reply = ByteMessage::outcomes_builder().add_outcomes(&[0]).build();
        assert!(noise.continue_processing(reply).is_err());
        assert!(noise.start(input.clone()).is_err());
        noise.reset().unwrap();
        // No stale X or old user outcomes may escape after recovery.
        let mut system =
            QuantumSystem::new(Box::new(noise.clone()), Box::new(StateVecEngine::new(1)));
        assert_eq!(
            system
                .process(ByteMessage::quantum_operations_builder().mz(&[0]).build())
                .unwrap()
                .outcomes()
                .unwrap(),
            vec![0]
        );
    }
}

#[test]
fn crosstalk_boundaries_match_explicit_dispatch_without_renoising_transitions() {
    use std::collections::BTreeMap;
    for seed in 0..32 {
        let model = GeneralNoiseModel::builder()
            .with_p1(1.0)
            .with_p_meas_crosstalk(1.0)
            .with_p_meas_crosstalk_model(&BTreeMap::from([
                ("0->1".into(), 1.0),
                ("1->0".into(), 1.0),
            ]))
            .build();
        let mut whole =
            QuantumSystem::new(Box::new(model.clone()), Box::new(StateVecEngine::new(1)));
        let mut split = QuantumSystem::new(Box::new(model), Box::new(StateVecEngine::new(1)));
        whole.set_seed(seed);
        split.set_seed(seed);
        let input = ByteMessage::quantum_operations_builder()
            .pz(&[0])
            .meas_crosstalk_local_payload(&[0])
            .mz(&[0])
            .build();
        assert_eq!(whole.process(input).unwrap().outcomes().unwrap(), vec![1]);
        split
            .process(
                ByteMessage::quantum_operations_builder()
                    .pz(&[0])
                    .meas_crosstalk_local_payload(&[0])
                    .build(),
            )
            .unwrap();
        assert_eq!(
            split
                .process(ByteMessage::quantum_operations_builder().mz(&[0]).build())
                .unwrap()
                .outcomes()
                .unwrap(),
            vec![1]
        );
        assert_eq!(
            format!("{:?}", whole.noise_model().rng()),
            format!("{:?}", split.noise_model().rng())
        );
    }
}

#[test]
fn crosstalk_without_injected_victims_keeps_one_dispatch() {
    for (probability, prepare) in [(0.0, true), (1.0, false)] {
        let mut noise = GeneralNoiseModel::builder()
            .with_p_meas_crosstalk(probability)
            .build();
        let mut input = ByteMessage::quantum_operations_builder();
        if prepare {
            input.pz(&[0]);
        }
        input.meas_crosstalk_local_payload(&[0]).x(&[0]).mz(&[0]);
        let EngineStage::NeedsProcessing(commands) = noise.start(input.build()).unwrap() else {
            panic!("expected commands");
        };
        let mut simulator = StateVecEngine::new(1);
        let reply = simulator.process(commands).unwrap();
        let EngineStage::Complete(outcomes) = noise.continue_processing(reply).unwrap() else {
            panic!("no victim should introduce no extra dispatch");
        };
        assert_eq!(outcomes.outcomes().unwrap(), vec![1]);
    }
}

#[test]
fn cloned_pending_crosstalk_resumes_independently() {
    use std::collections::BTreeMap;
    let mut noise = GeneralNoiseModel::builder()
        .with_p_meas_crosstalk(1.0)
        .with_p_meas_crosstalk_model(&BTreeMap::from([
            ("0->1".into(), 1.0),
            ("1->0".into(), 1.0),
        ]))
        .build();
    let input = ByteMessage::quantum_operations_builder()
        .pz(&[0])
        .mz(&[0])
        .meas_crosstalk_local_payload(&[0])
        .mz(&[0])
        .build();
    let EngineStage::NeedsProcessing(first) = noise.start(input).unwrap() else {
        panic!("expected commands");
    };
    let mut simulator = StateVecEngine::new(1);
    let reply = simulator.process(first).unwrap();
    let mut states = Vec::new();
    for mut copy in [noise.clone(), noise] {
        let mut sim = simulator.clone();
        let mut stage = copy.continue_processing(reply.clone()).unwrap();
        loop {
            match stage {
                EngineStage::NeedsProcessing(commands) => {
                    stage = copy
                        .continue_processing(sim.process(commands).unwrap())
                        .unwrap();
                }
                EngineStage::Complete(outcomes) => {
                    assert_eq!(outcomes.outcomes().unwrap(), vec![0, 1]);
                    states.push(format!("{:?}", pecos_core::RngManageable::rng(&copy)));
                    break;
                }
            }
        }
    }
    assert_eq!(states[0], states[1]);
}

#[test]
fn abandoned_transition_dispatch_blocks_new_starts_until_ack_or_reset() {
    use std::collections::BTreeMap;
    let mut noise = GeneralNoiseModel::builder()
        .with_p_meas_crosstalk(1.0)
        .with_p_meas_crosstalk_model(&BTreeMap::from([
            ("0->1".into(), 1.0),
            ("1->0".into(), 1.0),
        ]))
        .build();
    let input = ByteMessage::quantum_operations_builder()
        .pz(&[0])
        .meas_crosstalk_local_payload(&[0])
        .build();
    noise.start(input.clone()).unwrap();
    let reply = ByteMessage::outcomes_builder().add_outcomes(&[0]).build();
    assert!(matches!(
        noise.continue_processing(reply).unwrap(),
        EngineStage::NeedsProcessing(_)
    ));
    // No user results and no remaining source gates: the transition itself still
    // awaits simulator acknowledgement, including after cloning the controller.
    for mut copy in [noise.clone(), noise] {
        assert!(copy.start(input.clone()).is_err());
        assert!(copy.apply_noise_on_start(&input).is_err());
        assert!(matches!(
            copy.continue_processing(ByteMessage::create_empty())
                .unwrap(),
            EngineStage::Complete(_)
        ));
        copy.start(input.clone()).unwrap();
        copy.reset().unwrap();
        assert!(copy.start(input.clone()).is_ok());
    }
}
