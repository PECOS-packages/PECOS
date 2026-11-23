//! Non-collapsing detection, positional outcomes, and exact noise RNG accounting.
use pecos_core::RngManageable;
use pecos_engines::noise::{
    BiasedDepolarizingNoiseModel, DepolarizingNoiseModel, GeneralNoiseConfig, GeneralNoiseModel,
    GeneralNoiseModelBuilder, LossSite, NoiseRng, PauliLeakageChannel, QubitLoss, ReadoutMatrix,
    StateTransitionChannel,
};
use pecos_engines::quantum::{CoinTossEngine, SparseStabEngine, StabVecEngine, StateVecEngine};
use pecos_engines::{ByteMessage, ControlEngine, Engine, EngineStage, Gate, GateType};
use std::collections::BTreeMap;

const FLAVORS: [GateType; 3] = [
    GateType::DetectLeaked,
    GateType::DetectLost,
    GateType::DetectLeakedLost,
];
fn program(gates: &[Gate]) -> ByteMessage {
    let mut builder = ByteMessage::quantum_operations_builder();
    builder.add_gate_commands(gates);
    builder.build()
}
fn detect(kind: GateType, qs: &[usize]) -> Gate {
    Gate::simple(kind, qs.iter().copied().map(Into::into).collect::<Vec<_>>())
}
fn ideal<E: Engine<Input = ByteMessage, Output = ByteMessage>>(engine: &mut E) {
    for flavor in FLAVORS {
        engine.reset().unwrap();
        let result = engine
            .process(program(&[
                Gate::h(&[0]),
                detect(flavor, &[0]),
                Gate::h(&[0]),
                Gate::mz(&[0]),
            ]))
            .unwrap();
        assert_eq!(result.outcomes().unwrap(), [0, 0], "{flavor}");
        engine.reset().unwrap();
        let result = engine
            .process(program(&[
                Gate::x(&[1, 3]),
                Gate::mz(&[1]),
                detect(flavor, &[0, 2]),
                Gate::mz(&[3]),
            ]))
            .unwrap();
        assert_eq!(result.outcomes().unwrap(), [1, 0, 0, 1]);
        assert!(engine.process(program(&[detect(flavor, &[4])])).is_err());
    }
}
#[test]
fn clifford_detect_does_not_collapse() {
    for seed in 0..64 {
        ideal(&mut SparseStabEngine::with_seed(4, seed));
    }
}
#[test]
fn general_detect_does_not_collapse() {
    for seed in 0..64 {
        ideal(&mut StabVecEngine::with_seed(4, seed));
    }
}
#[test]
fn statevector_detect_does_not_collapse() {
    for seed in 0..64 {
        ideal(&mut StateVecEngine::with_seed(4, seed));
    }
}
#[test]
fn coin_toss_detection_and_byte_builder() {
    let mut b = ByteMessage::quantum_operations_builder();
    b.detect_leaked(&[2, 0])
        .detect_lost(&[1])
        .detect_leaked_lost(&[3]);
    let input = b.build();
    assert_eq!(input.quantum_ops().unwrap().len(), 4);
    let mut engine = CoinTossEngine::with_prob_and_seed(4, 1.0, 7);
    assert_eq!(engine.process(input).unwrap().outcomes().unwrap(), [0; 4]);
    assert!(engine.process(program(&[Gate::detect_lost(&[4])])).is_err());
}
fn weights(entries: &[(&str, f64)]) -> BTreeMap<String, f64> {
    entries.iter().map(|(k, v)| ((*k).into(), *v)).collect()
}
fn all_noise() -> GeneralNoiseModelBuilder {
    let hook = PauliLeakageChannel::new(1.0, &weights(&[("X", 1.0)])).unwrap();
    GeneralNoiseModel::builder()
        .with_seed(7919)
        .with_loss(
            QubitLoss::builder()
                .with_default_recovery(false)
                .with_idle_rate(1.0)
                .with_measurement(1.0)
                .with_prep(0.2)
                .add_events_channel(LossSite::BeforeSingleQubit, 0.2, &weights(&[("L", 1.0)]))
                .add_events_channel(LossSite::AfterSingleQubit, 0.2, &weights(&[("L", 1.0)]))
                .add_recovery_channel(LossSite::BeforeSingleQubit, 0.3, 0.4)
                .add_recovery_channel(LossSite::AfterSingleQubit, 0.3, 0.4)
                .build()
                .unwrap(),
        )
        .with_p1(1.0)
        .with_p2(0.3)
        .with_p1_emission_ratio(0.5)
        .with_p2_emission_ratio(0.5)
        .with_p_prep(0.3)
        .with_prep_leak_ratio(0.5)
        .with_p_prep_crosstalk(0.4)
        .with_p_meas_crosstalk(0.4)
        .with_p_idle_linear(0.2, &weights(&[("X", 1.0)]))
        .with_p_idle_sin_squared(0.2, &weights(&[("Z", 1.0)]))
        .with_p_idle_coherent(0.2, &weights(&[("RX", 1.0)]))
        .add_p1_pauli_leakage_channel_before_gate(&hook)
        .add_p1_pauli_leakage_channel_after_gate(&hook)
        .with_p_meas_0(1.0)
        .with_p_meas_1(1.0)
}
fn run(model: &mut GeneralNoiseModel, gates: &[Gate]) -> (Vec<u32>, Vec<Gate>) {
    let mut engine = StateVecEngine::with_seed(4, 65537);
    let mut stage = model.start(program(gates)).unwrap();
    let mut emitted = vec![];
    loop {
        match stage {
            EngineStage::NeedsProcessing(commands) => {
                emitted.extend(commands.quantum_ops().unwrap());
                stage = model
                    .continue_processing(engine.process(commands).unwrap())
                    .unwrap();
            }
            EngineStage::Complete(reply) => return (reply.outcomes().unwrap(), emitted),
        }
    }
}
fn tail(model: &impl RngManageable<Rng = pecos_random::PecosRng>) -> [u64; 4] {
    let mut rng = model.rng().clone();
    std::array::from_fn(|_| rng.next_u64())
}
fn prepare_locations(model: &mut GeneralNoiseModel) {
    run(model, &[Gate::idle(100.0, vec![0.into()])]);
    model.mark_as_leaked(1);
}
fn nominal(flavor: GateType) -> [u32; 3] {
    match flavor {
        GateType::DetectLeaked => [0, 2, 0],
        GateType::DetectLost => [3, 0, 0],
        GateType::DetectLeakedLost => [3, 2, 0],
        _ => unreachable!(),
    }
}
#[test]
fn detection_nominal_no_faults_hooks_flips_or_rng_draws() {
    for flavor in FLAVORS {
        let mut model = all_noise().build();
        prepare_locations(&mut model);
        let before = tail(&model);
        let gate = detect(flavor, &[0, 1, 2]);
        let (outcomes, emitted) = run(&mut model, std::slice::from_ref(&gate));
        assert_eq!(outcomes, nominal(flavor));
        assert_eq!(emitted, [gate]); // Includes both p1 hooks at probability one.
        assert_eq!(tail(&model), before);
        // Measurement loss must not fire on present or leaked operands.
        assert_eq!(
            run(&mut model, &[Gate::detect_leaked_lost(&[0, 1, 2])]).0,
            [3, 2, 0]
        );
        assert_eq!(tail(&model), before);
    }
}
#[test]
fn detection_readout_exactly_one_draw_per_configured_location() {
    for flavor in FLAVORS {
        for (qubit, row) in [(0, "lost"), (1, "leaked"), (2, "present")] {
            for noiseless in [false, true] {
                let rows = BTreeMap::from([(row.into(), weights(&[("0", 1.0)]))]);
                let mut builder =
                    all_noise().with_readout(ReadoutMatrix::new(flavor, &rows).unwrap());
                if noiseless {
                    builder = builder.with_noiseless_gate(flavor);
                }
                let mut model = builder.build();
                prepare_locations(&mut model);
                let mut expected_rng = NoiseRng::new(model.rng().clone());
                if !noiseless {
                    expected_rng.random_float();
                }
                let expected_tail =
                    std::array::from_fn::<_, 4, _>(|_| expected_rng.inner_mut().next_u64());
                let gate = detect(flavor, &[qubit]);
                let (outcome, emitted) = run(&mut model, std::slice::from_ref(&gate));
                assert_eq!(
                    outcome,
                    [if noiseless { nominal(flavor)[qubit] } else { 0 }]
                );
                assert_eq!(emitted, [gate]);
                assert_eq!(
                    tail(&model),
                    expected_tail,
                    "{flavor} {row} noiseless={noiseless}"
                );
            }
        }
    }
}
#[test]
fn detection_present_readout_returns_non_nominal_outcomes() {
    for (flavor, reported) in [
        (GateType::DetectLeaked, 2),
        (GateType::DetectLost, 3),
        (GateType::DetectLeakedLost, 2),
        (GateType::DetectLeakedLost, 3),
    ] {
        let rows = BTreeMap::from([(
            "present".into(),
            BTreeMap::from([(reported.to_string(), 1.0)]),
        )]);
        let mut model = all_noise()
            .with_readout(ReadoutMatrix::new(flavor, &rows).unwrap())
            .build();
        prepare_locations(&mut model);
        let mut expected_rng = NoiseRng::new(model.rng().clone());
        expected_rng.random_float();
        let expected_tail = std::array::from_fn::<_, 4, _>(|_| expected_rng.inner_mut().next_u64());
        assert_eq!(run(&mut model, &[detect(flavor, &[2])]).0, [reported]);
        assert_eq!(tail(&model), expected_tail);
    }
}
#[test]
fn detection_under_general_noise_does_not_collapse_and_loss_off_is_present() {
    for flavor in FLAVORS {
        for seed in 0..64 {
            let mut model = GeneralNoiseModel::builder()
                .with_seed(seed)
                .with_loss(QubitLoss::builder().with_measurement(1.0).build().unwrap())
                .build();
            assert_eq!(
                run(
                    &mut model,
                    &[
                        Gate::h(&[0]),
                        detect(flavor, &[0]),
                        Gate::h(&[0]),
                        Gate::mz(&[0])
                    ]
                )
                .0,
                [0, 0]
            );
        }
        let mut model = GeneralNoiseModel::builder().with_p_meas_0(1.0).build();
        assert_eq!(run(&mut model, &[detect(flavor, &[0])]).0, [0]);
    }
}
#[test]
fn detection_is_queued_after_pending_transition() {
    let channel =
        StateTransitionChannel::from_transitions(1.0, &weights(&[("0->L", 1.0), ("1->L", 1.0)]))
            .unwrap();
    let mut model = GeneralNoiseModel::builder()
        .add_p1_transition_channel_after_gate(&channel)
        .build();
    assert_eq!(
        run(&mut model, &[Gate::x(&[0]), Gate::detect_leaked(&[0])]).0,
        [2]
    );
}
#[test]
fn biased_detection_positions_and_rng_tail() {
    for seed in [7, 7919] {
        for flavor in FLAVORS {
            let mut model = BiasedDepolarizingNoiseModel::new(0.0, 1.0, 1.0, 1.0, 1.0);
            model.set_seed(seed);
            let mut reference = model.clone();
            let mut sim = StateVecEngine::new(4);
            let EngineStage::NeedsProcessing(commands) = model
                .start(program(&[
                    Gate::mz(&[0]),
                    detect(flavor, &[1, 2]),
                    Gate::mz(&[3]),
                ]))
                .unwrap()
            else {
                panic!()
            };
            let EngineStage::Complete(reply) = model
                .continue_processing(sim.process(commands).unwrap())
                .unwrap()
            else {
                panic!()
            };
            assert_eq!(reply.outcomes().unwrap(), [1, 0, 0, 1]);
            let EngineStage::NeedsProcessing(commands) =
                reference.start(program(&[Gate::mz(&[0, 3])])).unwrap()
            else {
                panic!()
            };
            reference
                .continue_processing(sim.process(commands).unwrap())
                .unwrap();
            assert_eq!(tail(&model), tail(&reference));
        }
    }
}
#[test]
fn biased_detection_record_validation_reset_and_consumption() {
    let mut model = BiasedDepolarizingNoiseModel::new_uniform(0.0);
    model.start(program(&[Gate::detect_lost(&[0, 1])])).unwrap();
    let mut reply = ByteMessage::outcomes_builder();
    reply.add_outcomes(&[0]);
    let error = model
        .continue_processing(reply.build())
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("(1)") && error.contains("(2)"), "{error}");
    // Consumed on error, reset, and replaced by a subsequent detect-free start.
    for mode in 0..3 {
        model.start(program(&[Gate::detect_lost(&[0, 1])])).unwrap();
        match mode {
            0 => {
                model.reset().unwrap();
            }
            1 => {
                model.start(program(&[Gate::mz(&[0])])).unwrap();
            }
            _ => {}
        }
        let mut b = ByteMessage::outcomes_builder();
        b.add_outcomes(&[0]);
        assert_eq!(model.continue_processing(b.build()).is_ok(), mode != 2);
        let mut b = ByteMessage::outcomes_builder();
        b.add_outcomes(&[0]);
        assert!(model.continue_processing(b.build()).is_ok());
    }
}
#[test]
fn depolarizing_forwards_detection_without_noise_or_draws() {
    let mut model = DepolarizingNoiseModel::new_uniform(1.0);
    let before = tail(&model);
    let gates = FLAVORS.map(|kind| detect(kind, &[0, 1]));
    let EngineStage::NeedsProcessing(commands) = model.start(program(&gates)).unwrap() else {
        panic!()
    };
    assert_eq!(commands.quantum_ops().unwrap(), gates);
    assert_eq!(tail(&model), before);
}
#[test]
fn detection_readout_validation_rust_and_json() {
    for flavor in FLAVORS {
        for row in ["present", "leaked", "lost", "0", "1"] {
            let rows = BTreeMap::from([(row.into(), weights(&[("0", 1.0)]))]);
            let valid = !matches!(row, "0" | "1");
            let matrix = ReadoutMatrix::new(flavor, &rows);
            assert_eq!(matrix.is_ok(), valid);
            if let Ok(matrix) = matrix {
                assert_eq!(
                    GeneralNoiseModel::builder()
                        .with_readout(matrix)
                        .try_build()
                        .is_ok(),
                    row != "lost"
                );
            } else {
                assert!(
                    matrix
                        .unwrap_err()
                        .to_string()
                        .contains("use row present, leaked, or lost")
                );
            }
            for loss_on in [false, true] {
                let mut config =
                    serde_json::json!({"measurement": {"readout": {flavor.to_string(): rows}}});
                if loss_on {
                    config["loss"] = serde_json::json!({"enabled": true, "measurement": 0.0});
                }
                let parsed: GeneralNoiseConfig = serde_json::from_value(config).unwrap();
                assert_eq!(
                    parsed.build().is_ok(),
                    valid && (loss_on || row != "lost"),
                    "{flavor} {row} loss={loss_on}"
                );
            }
        }
        for column in [
            "1",
            if flavor == GateType::DetectLost {
                "2"
            } else {
                "3"
            },
        ] {
            if flavor == GateType::DetectLeakedLost && column == "3" {
                continue;
            }
            assert!(
                ReadoutMatrix::new(
                    flavor,
                    &BTreeMap::from([("present".into(), weights(&[(column, 1.0)]))])
                )
                .is_err()
            );
        }
    }
    for flavor in [
        GateType::MZ,
        GateType::MeasureZLeaked,
        GateType::MeasureZLost,
        GateType::MeasureZLeakedLost,
    ] {
        let error = ReadoutMatrix::new(
            flavor,
            &BTreeMap::from([("present".into(), weights(&[("0", 1.0)]))]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("use row 0, 1, leaked, or lost"));
    }
}

#[test]
fn detection_json_rejects_out_of_alphabet_columns_and_z_present_rows() {
    for (flavor, row, column) in [
        (GateType::DetectLeaked, "present", "1"),
        (GateType::DetectLeaked, "present", "3"),
        (GateType::DetectLost, "leaked", "2"),
        (GateType::DetectLeakedLost, "present", "1"),
        (GateType::MZ, "present", "0"),
        (GateType::MeasureZLeaked, "present", "0"),
        (GateType::MeasureZLost, "present", "0"),
        (GateType::MeasureZLeakedLost, "present", "0"),
    ] {
        let config = serde_json::json!({"measurement": {"readout": {flavor.to_string(): {row: {column: 1.0}}}}});
        let error = GeneralNoiseConfig::from_json(&config.to_string())
            .unwrap()
            .build()
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains(&flavor.to_string()));
        assert!(error.contains(if flavor.is_detection() {
            "alphabet"
        } else {
            "use row 0, 1, leaked, or lost"
        }));
    }
}
