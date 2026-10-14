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
        Err(ChannelValidationError::IncompleteComputationalRows)
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
        &BTreeMap::from([("*L->*0".into(), 1.000_009), ("L*->0*".into(), 1.000_009)]),
    );
    assert!(channel.is_ok());
}
