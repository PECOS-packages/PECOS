use pecos_engines::byte_message::ByteMessage;
use pecos_engines::noise::{
    ChannelValidationError, GeneralNoiseModel, PauliLeakageChannel, PauliLeakageEvent,
    QubitTransitionState, StateTransitionChannel, TwoQubitPauliLeakageChannel,
    TwoQubitStateTransitionChannel,
};
use pecos_engines::quantum::StateVecEngine;
use pecos_engines::{Engine, EngineSystem, QuantumSystem};
use std::collections::BTreeMap;

#[test]
fn typed_and_string_configurations_agree_and_return_structured_errors() {
    use QubitTransitionState::{Leakage, One, Zero};
    let typed = StateTransitionChannel::from_rows(
        0.3,
        &BTreeMap::from([
            (Zero, BTreeMap::from([(One, 1.0)])),
            (One, BTreeMap::from([(Leakage, 1.0)])),
        ]),
    )
    .unwrap();
    let strings = StateTransitionChannel::from_transitions(
        0.3,
        &BTreeMap::from([("0->1".into(), 1.0), ("1->L".into(), 1.0)]),
    )
    .unwrap();
    assert_eq!(typed.transitions(), strings.transitions());
    assert!(matches!(
        StateTransitionChannel::from_rows(
            1.0,
            &BTreeMap::from([(Zero, BTreeMap::from([(Zero, 1.0)]))])
        ),
        Err(ChannelValidationError::IncompleteComputationalRows(patterns))
            if patterns == BTreeMap::from([("C".into(), vec!["1".into()])])
    ));
    assert!(matches!(
        StateTransitionChannel::leak_recovery(2.0, 0.5),
        Err(ChannelValidationError::InvalidProbability(2.0))
    ));
    let identity =
        PauliLeakageChannel::from_events(1.0, &BTreeMap::from([(PauliLeakageEvent::I, 1.0)]))
            .unwrap();
    assert_eq!(identity.events(), &BTreeMap::from([("I".into(), 1.0)]));
    let joint = TwoQubitStateTransitionChannel::from_rows(
        1.0,
        &BTreeMap::from([([Leakage, Leakage], BTreeMap::from([([Zero, One], 1.0)]))]),
    )
    .unwrap();
    assert_eq!(joint.transitions()["LL"]["01"], 1.0);
    let events = TwoQubitPauliLeakageChannel::from_events(
        1.0,
        &BTreeMap::from([([PauliLeakageEvent::X, PauliLeakageEvent::Leakage], 1.0)]),
    )
    .unwrap();
    assert_eq!(events.events()["XL"], 1.0);
}

fn interference_ones(channel: &StateTransitionChannel) -> usize {
    let noise = GeneralNoiseModel::builder()
        .add_p1_transition_channel_before_gate(channel)
        .build();
    let mut system = QuantumSystem::new(Box::new(noise), Box::new(StateVecEngine::new(1)));
    system.set_seed(41);
    let circuit = ByteMessage::quantum_operations_builder()
        .h(&[0])
        .h(&[0])
        .mz(&[0])
        .build();
    let mut ones = 0;
    for _ in 0..512 {
        system.reset().unwrap();
        let result = system.process_as_system(circuit.clone()).unwrap();
        let outcomes = result.outcomes().unwrap();
        assert_eq!(
            outcomes.len(),
            1,
            "internal MZ must not become a program result"
        );
        ones += usize::from(outcomes[0] == 1);
    }
    ones
}

#[test]
fn explicit_identity_rows_collapse_but_omitted_computational_rows_preserve_coherence() {
    let identity = StateTransitionChannel::from_transitions(
        1.0,
        &BTreeMap::from([("0->0".into(), 1.0), ("1->1".into(), 1.0)]),
    )
    .unwrap();
    assert!((190..322).contains(&interference_ones(&identity)));
    assert_eq!(
        interference_ones(&StateTransitionChannel::leak_recovery(1.0, 0.5).unwrap()),
        0
    );
    let disabled =
        StateTransitionChannel::from_transition_dict(0.0, identity.transition_dict()).unwrap();
    assert_eq!(interference_ones(&disabled), 0);
}

#[test]
fn wildcard_identity_preserves_interference_on_the_other_operand() {
    let recover = TwoQubitStateTransitionChannel::from_transitions(
        1.0,
        &BTreeMap::from([("*L->*0".into(), 1.0)]),
    )
    .unwrap();
    let leak = TwoQubitPauliLeakageChannel::from_events(
        1.0,
        &BTreeMap::from([([PauliLeakageEvent::I, PauliLeakageEvent::Leakage], 1.0)]),
    )
    .unwrap();
    let noise = GeneralNoiseModel::builder()
        .add_p2_pauli_leakage_step_before_gate(&pecos_engines::noise::P2PauliLeakageStep::joint(
            leak,
        ))
        .add_p2_transition_channel_before_gate(&recover)
        .build();
    let mut system = QuantumSystem::new(Box::new(noise), Box::new(StateVecEngine::new(2)));
    system.set_seed(7);
    // The recovered second operand is zero, so CZ leaves the first operand's |+> intact.
    let circuit = ByteMessage::quantum_operations_builder()
        .h(&[0])
        .cz(&[(0, 1)])
        .h(&[0])
        .mz(&[0, 1])
        .build();
    for _ in 0..32 {
        system.reset().unwrap();
        assert_eq!(
            system
                .process_as_system(circuit.clone())
                .unwrap()
                .outcomes()
                .unwrap(),
            vec![0, 0]
        );
    }
}

#[test]
fn wildcard_rows_within_normalization_tolerance_remain_valid_when_combined() {
    let channel = TwoQubitStateTransitionChannel::from_transitions(
        1.0,
        &BTreeMap::from([
            ("*L->*0".into(), 0.500_004),
            ("*L->*1".into(), 0.500_004),
            ("L*->0*".into(), 0.500_004),
            ("L*->1*".into(), 0.500_004),
        ]),
    );
    assert!(channel.is_ok());
}

#[test]
fn individual_weights_above_one_are_rejected_even_within_sum_tolerance() {
    for weight in [1.000_001, f64::from_bits(1.0_f64.to_bits() + 1)] {
        let errors = [
            StateTransitionChannel::from_transitions(
                1.0,
                &BTreeMap::from([("L->0".into(), weight)]),
            )
            .unwrap_err(),
            TwoQubitStateTransitionChannel::from_transitions(
                1.0,
                &BTreeMap::from([("*L->*0".into(), weight)]),
            )
            .unwrap_err(),
            PauliLeakageChannel::new(1.0, &BTreeMap::from([("X".into(), weight)])).unwrap_err(),
            TwoQubitPauliLeakageChannel::new(1.0, &BTreeMap::from([("XL".into(), weight)]))
                .unwrap_err(),
        ];
        for error in errors {
            assert!(
                matches!(error, ChannelValidationError::InvalidWeight { value, .. } if value.to_bits() == weight.to_bits())
            );
        }
    }
}

#[test]
fn runtime_frames_reject_hooks_until_channel_barriers_have_boundary_support() {
    use pecos_engines::runtime_frame::{FrameLimits, RuntimeNoise};
    let recovery = StateTransitionChannel::leak_recovery(1.0, 0.5).unwrap();
    let builder = GeneralNoiseModel::builder().add_p2_transition_channel_before_gate(&recovery);
    let error = RuntimeNoise::new(builder, 2, FrameLimits::default())
        .err()
        .unwrap();
    assert!(error.to_string().contains("gate-hook channels"));
}

#[test]
fn pauli_identity_hooks_build_and_preserve_interference() {
    use pecos_engines::noise::P2PauliLeakageStep;
    let single = PauliLeakageChannel::new(1.0, &BTreeMap::from([("I".into(), 1.0)])).unwrap();
    let joint = P2PauliLeakageStep::joint(
        TwoQubitPauliLeakageChannel::new(1.0, &BTreeMap::from([("II".into(), 1.0)])).unwrap(),
    );
    let noise = GeneralNoiseModel::builder()
        .add_p1_pauli_leakage_channel_before_gate(&single)
        .add_p1_pauli_leakage_channel_after_gate(&single)
        .add_p2_pauli_leakage_step_before_gate(&joint)
        .add_p2_pauli_leakage_step_after_gate(&joint)
        .add_p2_pauli_leakage_channel_before_gate(&single)
        .add_p2_pauli_leakage_channel_after_gate(&single)
        .build();
    let mut system = QuantumSystem::new(Box::new(noise), Box::new(StateVecEngine::new(2)));
    system.set_seed(41);
    let circuit = ByteMessage::quantum_operations_builder()
        .h(&[0])
        .cz(&[(0, 1)])
        .h(&[0])
        .mz(&[0, 1])
        .build();
    for _ in 0..64 {
        system.reset().unwrap();
        assert_eq!(
            system
                .process_as_system(circuit.clone())
                .unwrap()
                .outcomes()
                .unwrap(),
            vec![0, 0]
        );
    }
}

#[test]
fn mixed_pauli_identity_hooks_preserve_event_probabilities() {
    use pecos_engines::noise::P2PauliLeakageStep;
    for joint in [false, true] {
        for before in [false, true] {
            let mut builder = GeneralNoiseModel::builder();
            if joint {
                let step = P2PauliLeakageStep::joint(
                    TwoQubitPauliLeakageChannel::new(
                        0.8,
                        &BTreeMap::from([("II".into(), 0.75), ("XI".into(), 0.25)]),
                    )
                    .unwrap(),
                );
                builder = if before {
                    builder.add_p2_pauli_leakage_step_before_gate(&step)
                } else {
                    builder.add_p2_pauli_leakage_step_after_gate(&step)
                };
            } else {
                let channel = PauliLeakageChannel::new(
                    0.8,
                    &BTreeMap::from([("I".into(), 0.75), ("X".into(), 0.25)]),
                )
                .unwrap();
                builder = if before {
                    builder.add_p1_pauli_leakage_channel_before_gate(&channel)
                } else {
                    builder.add_p1_pauli_leakage_channel_after_gate(&channel)
                };
            }
            let mut system =
                QuantumSystem::new(Box::new(builder.build()), Box::new(StateVecEngine::new(2)));
            system.set_seed(41);
            let mut circuit = ByteMessage::quantum_operations_builder();
            if joint {
                circuit.cz(&[(0, 1)]);
            } else {
                circuit.z(&[0]);
            }
            let circuit = circuit.mz(&[0, 1]).build();
            let mut ones = 0;
            for _ in 0..1024 {
                system.reset().unwrap();
                let outcomes = system
                    .process_as_system(circuit.clone())
                    .unwrap()
                    .outcomes()
                    .unwrap();
                assert_eq!(outcomes.len(), 2);
                assert_eq!(outcomes[1], 0);
                ones += usize::from(outcomes[0] == 1);
            }
            // P(X) = outer probability * event weight = 0.2, not 0.8.
            assert!(
                (150..260).contains(&ones),
                "joint={joint}, before={before}: {ones}"
            );
        }
    }
}

fn leakage_pattern_channel() -> TwoQubitStateTransitionChannel {
    TwoQubitStateTransitionChannel::from_transitions(
        1.0,
        &BTreeMap::from([("0L->LL".into(), 1.0), ("1L->1L".into(), 1.0)]),
    )
    .unwrap()
}

#[test]
fn joint_hook_without_matching_leakage_pattern_emits_no_measurement() {
    let mut noise = GeneralNoiseModel::builder()
        .add_p2_transition_channel_before_gate(leakage_pattern_channel())
        .build();
    let circuit = ByteMessage::quantum_operations_builder()
        .cz(&[(0, 1)])
        .build();
    assert_eq!(
        noise
            .apply_noise_on_start(&circuit)
            .unwrap()
            .quantum_ops()
            .unwrap(),
        circuit.quantum_ops().unwrap()
    );
}

#[test]
fn joint_hook_without_matching_leakage_pattern_preserves_coherence() {
    let noise = GeneralNoiseModel::builder()
        .add_p2_transition_channel_before_gate(leakage_pattern_channel())
        .build();
    let mut system = QuantumSystem::new(Box::new(noise), Box::new(StateVecEngine::new(2)));
    system.set_seed(41);
    let circuit = ByteMessage::quantum_operations_builder()
        .h(&[0])
        .cz(&[(0, 1)])
        .h(&[0])
        .mz(&[0])
        .build();
    for _ in 0..128 {
        system.reset().unwrap();
        assert_eq!(
            system
                .process_as_system(circuit.clone())
                .unwrap()
                .outcomes()
                .unwrap(),
            vec![0]
        );
    }
}

#[test]
fn hook_scaling_rejects_overscaled_probabilities_and_accepts_boundary() {
    for (p, gate_scale, valid) in [
        (0.0, 1.0, true),
        (0.5, 1.0, true),
        (0.8, 1.0, false),
        // Grouping matters: p * (MAX * 2) is non-finite, while (p * MAX) * 2 is valid.
        (1e-310, f64::MAX, false),
        (0.0, f64::MAX, false),
    ] {
        let single = StateTransitionChannel::leak_recovery(p, 1.0).unwrap();
        let joint = TwoQubitStateTransitionChannel::from_transitions(
            p,
            &BTreeMap::from([("LL->00".into(), 1.0)]),
        )
        .unwrap();
        let pauli = PauliLeakageChannel::new(p, &BTreeMap::from([("L".into(), 1.0)])).unwrap();
        let joint_pauli = pecos_engines::noise::P2PauliLeakageStep::joint(
            TwoQubitPauliLeakageChannel::new(p, &BTreeMap::from([("LL".into(), 1.0)])).unwrap(),
        );
        let base = GeneralNoiseModel::builder()
            .with_scale(2.0)
            .with_p1_scale(gate_scale)
            .with_p2_scale(gate_scale);
        let builders = vec![
            base.clone().add_p1_transition_channel_before_gate(&single),
            base.clone().add_p1_transition_channel_after_gate(&single),
            base.clone()
                .add_p1_pauli_leakage_channel_before_gate(&pauli),
            base.clone().add_p1_pauli_leakage_channel_after_gate(&pauli),
            base.clone().add_p2_transition_channel_before_gate(&joint),
            base.clone().add_p2_transition_channel_after_gate(&joint),
            base.clone().add_p2_transition_channel_before_gate(&single),
            base.clone().add_p2_transition_channel_after_gate(&single),
            base.clone()
                .add_p2_pauli_leakage_step_before_gate(&joint_pauli),
            base.clone()
                .add_p2_pauli_leakage_step_after_gate(&joint_pauli),
            base.clone()
                .add_p2_pauli_leakage_channel_before_gate(&pauli),
            base.add_p2_pauli_leakage_channel_after_gate(&pauli),
        ];
        for builder in builders {
            assert_eq!(builder.validate_configuration().is_ok(), valid);
            if valid {
                let _ = builder.build();
            }
        }
    }
}

#[test]
fn concrete_joint_rows_require_every_computational_completion() {
    for (present, missing, pattern) in [
        (vec!["0L"], vec!["1L"], "CL"),
        (vec!["1L"], vec!["0L"], "CL"),
        (vec!["L0"], vec!["L1"], "LC"),
        (vec!["L1"], vec!["L0"], "LC"),
        (vec!["00"], vec!["01", "10", "11"], "CC"),
        (vec!["01", "10", "11"], vec!["00"], "CC"),
    ] {
        let mut rows = present
            .into_iter()
            .map(|s| (s.to_string(), BTreeMap::from([(s.to_string(), 1.0)])))
            .collect::<BTreeMap<_, _>>();
        let error = TwoQubitStateTransitionChannel::new(1.0, &rows)
            .unwrap_err()
            .to_string();
        assert!(error.contains(&format!("pattern {pattern}:")), "{error}");
        for source in missing {
            assert!(error.contains(source), "{error} must name {source}");
            rows.insert(
                source.to_string(),
                BTreeMap::from([(source.to_string(), 1.0)]),
            );
        }
        assert!(TwoQubitStateTransitionChannel::new(1.0, &rows).is_ok());
    }
    assert!(
        TwoQubitStateTransitionChannel::from_transitions(
            1.0,
            &BTreeMap::from([("LL->00".into(), 1.0)])
        )
        .is_ok()
    );
}

#[test]
fn valid_one_qubit_tensor_products_have_complete_joint_rows() {
    use pecos_engines::noise::TransitionDict;
    // These are all nonempty valid source-row sets for one qubit.
    let factors = [vec!["L"], vec!["0", "1"], vec!["0", "1", "L"]];
    for left in &factors {
        for right in &factors {
            let dict = |sources: &Vec<&str>| {
                TransitionDict::new(
                    &sources
                        .iter()
                        .map(|s| (s.to_string(), BTreeMap::from([("L".into(), 1.0)])))
                        .collect(),
                )
                .unwrap()
            };
            let product = dict(left).tensor(&dict(right));
            assert!(TransitionDict::new(product.transitions()).is_ok());
        }
    }
}

#[test]
fn resumed_hook_reaches_measurement_crosstalk_and_returns_user_result() {
    let identity = StateTransitionChannel::from_transitions(
        1.0,
        &BTreeMap::from([("0->0".into(), 1.0), ("1->1".into(), 1.0)]),
    )
    .unwrap();
    let noise = GeneralNoiseModel::builder()
        .add_p1_transition_channel_before_gate(&identity)
        .with_p_meas_crosstalk_local(1.0)
        .with_p_meas_crosstalk_model(&BTreeMap::from([
            ("0->1".into(), 1.0),
            ("1->0".into(), 1.0),
        ]))
        .build();
    let mut system = QuantumSystem::new(Box::new(noise), Box::new(StateVecEngine::new(1)));
    let circuit = ByteMessage::quantum_operations_builder()
        .pz(&[0])
        .z(&[0])
        .meas_crosstalk_local_payload(&[0])
        .mz(&[0])
        .build();
    assert_eq!(
        system
            .process_as_system(circuit)
            .unwrap()
            .outcomes()
            .unwrap(),
        vec![1]
    );
}

#[test]
fn wildcard_and_tensor_channels_measure_only_the_computational_factor() {
    use pecos_engines::Gate;
    use pecos_engines::noise::TransitionDict;
    let identity =
        TransitionDict::from_transitions(&BTreeMap::from([("L->L".into(), 1.0)])).unwrap();
    let flip = TransitionDict::from_transitions(&BTreeMap::from([
        ("0->1".into(), 1.0),
        ("1->0".into(), 1.0),
    ]))
    .unwrap();
    for (dict, measured) in [
        (identity.tensor(&flip), 1),
        (flip.tensor(&identity), 0),
        (
            TransitionDict::from_transitions(&BTreeMap::from([
                ("*0->*1".into(), 1.0),
                ("*1->*0".into(), 1.0),
            ]))
            .unwrap(),
            1,
        ),
        (
            TransitionDict::from_transitions(&BTreeMap::from([
                ("0*->1*".into(), 1.0),
                ("1*->0*".into(), 1.0),
            ]))
            .unwrap(),
            0,
        ),
    ] {
        let channel = TwoQubitStateTransitionChannel::from_transition_dict(1.0, &dict).unwrap();
        let mut noise = GeneralNoiseModel::builder()
            .add_p2_transition_channel_before_gate(channel)
            .build();
        let input = ByteMessage::quantum_operations_builder()
            .cz(&[(0, 1)])
            .build();
        let emitted = noise.apply_noise_on_start(&input).unwrap();
        assert_eq!(emitted.quantum_ops().unwrap(), vec![Gate::mz(&[measured])]);
        let mut reply = ByteMessage::outcomes_builder();
        reply.add_outcomes(&[0]);
        let resumed = noise
            .apply_noise_on_continue_processing(reply.build())
            .unwrap();
        assert_eq!(
            resumed.quantum_ops().unwrap(),
            vec![Gate::x(&[measured]), Gate::cz(&[(0, 1)])]
        );
    }
}

#[test]
fn independent_hook_scaling_checks_unequal_legs_in_both_positions() {
    use pecos_engines::noise::{P2PauliLeakageStep, P2TransitionStep};
    for (first, second, valid) in [
        (0.0, 0.5, true),
        (0.2, 0.5, true),
        (0.8, 0.2, false),
        (0.2, 0.8, false),
    ] {
        let transition = P2TransitionStep::independent(
            StateTransitionChannel::leak_recovery(first, 1.0).unwrap(),
            StateTransitionChannel::leak_recovery(second, 1.0).unwrap(),
        );
        let events = BTreeMap::from([("L".into(), 1.0)]);
        let pauli = P2PauliLeakageStep::independent(
            PauliLeakageChannel::new(first, &events).unwrap(),
            PauliLeakageChannel::new(second, &events).unwrap(),
        );
        let base = GeneralNoiseModel::builder()
            .with_p2_scale(0.5)
            .with_scale(4.0);
        for builder in [
            base.clone().add_p2_transition_step_before_gate(&transition),
            base.clone().add_p2_transition_step_after_gate(&transition),
            base.clone().add_p2_pauli_leakage_step_before_gate(&pauli),
            base.add_p2_pauli_leakage_step_after_gate(&pauli),
        ] {
            assert_eq!(builder.validate_configuration().is_ok(), valid);
            if valid {
                let _ = builder.build();
            }
        }
    }
}

#[test]
fn joint_row_error_groups_missing_sources_by_leakage_pattern() {
    let error = TwoQubitStateTransitionChannel::from_transitions(
        1.0,
        &BTreeMap::from([
            ("0L->LL".into(), 1.0),
            ("L0->LL".into(), 1.0),
            ("00->LL".into(), 1.0),
        ]),
    )
    .unwrap_err();
    assert_eq!(
        error,
        ChannelValidationError::IncompleteComputationalRows(BTreeMap::from([
            ("CC".into(), vec!["01".into(), "10".into(), "11".into()]),
            ("CL".into(), vec!["1L".into()]),
            ("LC".into(), vec!["L1".into()]),
        ]))
    );
    let error = error.to_string();
    for expected in [
        "pattern CL: missing source rows 1L",
        "pattern LC: missing source rows L1",
        "pattern CC: missing source rows 01, 10, 11",
    ] {
        assert!(error.contains(expected), "{error}");
    }
}
