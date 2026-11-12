//! Per-flavor readout semantics, stream compatibility, and configuration boundaries.
use pecos_core::RngManageable;
use pecos_engines::noise::{
    GeneralNoiseConfig, GeneralNoiseModel, GeneralNoiseModelBuilder, ReadoutMatrix,
    StateTransitionChannel,
};
use pecos_engines::quantum::StateVecEngine;
use pecos_engines::runtime_frame::{FrameLimits, RuntimeNoise};
use pecos_engines::{ByteMessage, ControlEngine, Engine, EngineStage, Gate, GateType};
use std::collections::BTreeMap;

type Rows = BTreeMap<String, BTreeMap<String, f64>>;

fn rows(json: &str) -> Rows {
    serde_json::from_str(json).unwrap()
}
fn matrix(flavor: GateType, json: &str) -> ReadoutMatrix {
    ReadoutMatrix::new(flavor, &rows(json)).unwrap()
}
fn builder() -> GeneralNoiseModelBuilder {
    GeneralNoiseModel::builder().with_seed(7919)
}
fn program(gates: &[Gate]) -> ByteMessage {
    let mut b = ByteMessage::quantum_operations_builder();
    for gate in gates {
        b.add_gate_command(gate);
    }
    b.build()
}
fn measure(flavor: GateType, qubit: usize) -> Gate {
    Gate::simple(flavor, vec![qubit.into()])
}
fn tail(model: &GeneralNoiseModel) -> [u64; 4] {
    let mut rng = model.rng().clone();
    std::array::from_fn(|_| rng.next_u64())
}
fn run(model: &mut GeneralNoiseModel, gates: &[Gate]) -> (Vec<u32>, Vec<Vec<Gate>>, [u64; 4]) {
    let mut sim = StateVecEngine::with_seed(3, 65537);
    let mut stage = model.start(program(gates)).unwrap();
    let mut emitted = Vec::new();
    loop {
        match stage {
            EngineStage::NeedsProcessing(commands) => {
                emitted.push(commands.quantum_ops().unwrap());
                stage = model
                    .continue_processing(sim.process(commands).unwrap())
                    .unwrap();
            }
            EngineStage::Complete(results) => {
                return (results.outcomes().unwrap(), emitted, tail(model));
            }
        }
    }
}

#[test]
fn default_heralds_preserve_outcomes_commands_and_rng() {
    for (new, old) in [
        (GateType::MeasureZLost, GateType::MZ),
        (GateType::MeasureZLeakedLost, GateType::MeasureZLeaked),
    ] {
        for seed in 0..32 {
            let mut reference = builder()
                .with_seed(seed)
                .with_p_meas_0(0.27)
                .with_p_meas_1(0.61)
                .with_p1(0.3)
                .build();
            reference.mark_as_leaked(2);
            let mut candidate = reference.clone();
            let gates = |flavor| {
                vec![
                    Gate::h(&[0]),
                    Gate::x(&[1]),
                    measure(flavor, 0),
                    measure(flavor, 1),
                    measure(flavor, 2),
                ]
            };
            let expected = run(&mut reference, &gates(old));
            let mut actual = run(&mut candidate, &gates(new));
            for command in actual.1.iter_mut().flatten() {
                if command.gate_type == new {
                    command.gate_type = old;
                }
            }
            assert_eq!(actual, expected, "{new}, seed {seed}");
        }
    }
}

#[test]
fn leaked_default_does_not_use_configured_one_row_and_flavors_are_isolated() {
    let b = builder()
        .with_p_meas_1(1.0)
        .with_readout(matrix(GateType::MZ, r#"{"1":{"1":1.0}}"#));
    let mut model = b.clone().build();
    model.mark_as_leaked(1);
    assert_eq!(
        run(&mut model, &[Gate::x(&[0]), Gate::mz(&[0, 1])]).0,
        [1, 0]
    );
    let mut other = b
        .clone()
        .with_readout(matrix(GateType::MeasureZLost, r#"{"0":{"3":1.0}}"#))
        .build();
    let mut original = b.build();
    assert_eq!(
        run(&mut other, &[Gate::mz(&[0])]),
        run(&mut original, &[Gate::mz(&[0])])
    );
}

#[test]
fn deterministic_row_still_consumes_exactly_one_draw() {
    let mut original = builder().build();
    let mut configured = builder()
        .with_readout(matrix(GateType::MZ, r#"{"0":{"0":1.0}}"#))
        .build();
    let before = tail(&original);
    let result = run(&mut original, &[Gate::mz(&[0])]);
    assert_ne!(before, result.2);
    assert_eq!(run(&mut configured, &[Gate::mz(&[0])]), result);
}

#[test]
fn noiseless_flavors_are_independent_and_do_not_draw() {
    let flavors = [
        GateType::MZ,
        GateType::MeasureZLost,
        GateType::MeasureZLeaked,
        GateType::MeasureZLeakedLost,
    ];
    for noiseless in flavors {
        for configured in [false, true] {
            for measured in flavors {
                let mut setup = builder().with_p_meas_0(1.0).with_noiseless_gate(noiseless);
                if configured {
                    setup = setup.with_readout(matrix(measured, r#"{"0":{"1":1.0}}"#));
                }
                let mut model = setup.build();
                let before = run(&mut model.clone(), &[]).2;
                let result = run(&mut model, &[measure(measured, 0)]);
                assert_eq!(
                    result.0,
                    [u32::from(measured != noiseless)],
                    "{measured}, noiseless {noiseless}"
                );
                assert_eq!(result.2 == before, measured == noiseless);
            }
        }
        let mut model = builder()
            .with_p_meas_1(1.0)
            .with_noiseless_gate(noiseless)
            .with_readout(matrix(noiseless, r#"{"leaked":{"0":1.0}}"#))
            .build();
        model.mark_as_leaked(0);
        let before = run(&mut model.clone(), &[]).2;
        let nominal = if matches!(
            noiseless,
            GateType::MeasureZLeaked | GateType::MeasureZLeakedLost
        ) {
            2
        } else {
            1
        };
        let result = run(&mut model, &[measure(noiseless, 0)]);
        assert_eq!(result.0, [nominal]);
        assert_eq!(result.2, before);
    }
    // MPZ's own noiseless flag suppresses gate faults, but readout is keyed on MZ.
    for (key, expected) in [(GateType::MZ, 0), (GateType::MPZ, 1)] {
        let mut model = builder()
            .with_noiseless_gate(key)
            .with_readout(matrix(GateType::MZ, r#"{"0":{"1":1.0}}"#))
            .build();
        assert_eq!(run(&mut model, &[Gate::mpz(&[0])]).0, [expected]);
    }
}

#[test]
fn injected_transition_and_crosstalk_measurements_skip_readout() {
    let set_one = StateTransitionChannel::from_transitions(
        1.0,
        &BTreeMap::from([("0->1".into(), 1.0), ("1->1".into(), 1.0)]),
    )
    .unwrap();
    let base = builder().add_p1_transition_channel_before_gate(&set_one);
    let configured = base
        .clone()
        .with_readout(matrix(GateType::MZ, r#"{"0":{"1":1.0},"1":{"0":1.0}}"#));
    let gates = [Gate::z(&[0]), Gate::measure_z_leaked(&[0])];
    let expected = run(&mut base.build(), &gates);
    assert_eq!(expected.0, [1]);
    assert!(
        expected
            .1
            .iter()
            .flatten()
            .any(|gate| gate.gate_type == GateType::X)
    );
    assert!(
        expected.1.len() > 1,
        "hook must defer through a measurement"
    );
    let actual = run(&mut configured.build(), &gates);
    assert_eq!(
        actual.0, expected.0,
        "injected measurement destination changed"
    );
    assert_eq!(
        actual.1, expected.1,
        "injected measurement commands changed"
    );
    assert_eq!(
        actual.2, expected.2,
        "injected measurement RNG tail changed"
    );

    let base = builder()
        .with_p_meas_crosstalk_local(1.0)
        .with_p_meas_crosstalk_model(&BTreeMap::from([
            ("0->1".into(), 1.0),
            ("1->1".into(), 1.0),
        ]));
    let configured = base
        .clone()
        .with_readout(matrix(GateType::MZ, r#"{"0":{"1":1.0}}"#));
    let gates = [
        Gate::pz(&[0]),
        Gate::meas_crosstalk_local_payload(&[0]),
        Gate::measure_z_leaked(&[0]),
    ];
    let expected = run(&mut base.build(), &gates);
    assert_eq!(expected.0, [1]);
    assert!(
        expected
            .1
            .iter()
            .flatten()
            .any(|gate| gate.gate_type == GateType::X)
    );
    let actual = run(&mut configured.build(), &gates);
    assert_eq!(
        actual.0, expected.0,
        "injected measurement destination changed"
    );
    assert_eq!(
        actual.1, expected.1,
        "injected measurement commands changed"
    );
    assert_eq!(
        actual.2, expected.2,
        "injected measurement RNG tail changed"
    );
}

#[test]
fn scaling_affects_only_unconfigured_rows() {
    let mut model = builder()
        .with_scale(0.5)
        .with_p_meas_1(1.0)
        .with_readout(matrix(GateType::MZ, r#"{"0":{"1":1.0}}"#))
        .build();
    let mut zeros = 0;
    for _ in 0..2000 {
        let result = run(&mut model, &[Gate::x(&[1]), Gate::mz(&[0, 1])]).0;
        assert_eq!(result[0], 1);
        zeros += usize::from(result[1] == 0);
    }
    assert!((850..1150).contains(&zeros), "scaled default row: {zeros}");
}

#[test]
fn validation_names_flavor_row_value_and_fix() {
    let mut failures = Vec::new();
    let mut check = |flavor: GateType, row: &str, value: &str, fix: &str, input: &Rows| {
        let error = ReadoutMatrix::new(flavor, input).unwrap_err().to_string();
        let required = [
            flavor.to_string(),
            format!("row {row:?}"),
            format!("invalid value {value}"),
            fix.into(),
        ];
        let missing: Vec<_> = required
            .iter()
            .filter(|part| !error.contains(part.as_str()))
            .collect();
        if !missing.is_empty() {
            failures.push(format!(
                "{flavor} {row:?} {value}: missing {missing:?} in {error}"
            ));
        }
    };
    for (json, row, value, fix) in [
        (r#"{"0":{"2":1.0}}"#, "0", r#"column "2" = 1"#, "alphabet"),
        (
            r#"{"0":{"0":-0.1,"1":1.0}}"#,
            "0",
            r#"column "0" = -0.1"#,
            "[0, 1]",
        ),
        (r#"{"1":{"0":1.1}}"#, "1", r#"column "0" = 1.1"#, "[0, 1]"),
        (r#"{"leaked":{"0":0.9}}"#, "leaked", "sum 0.9", "sum to 1"),
        ("{}", "<flavor>", "{}", "at least one row"),
        (r#"{"0":{}}"#, "0", "{}", "at least one outcome"),
        (r#"{"typo":{"0":1.0}}"#, "typo", r#""typo""#, "use row"),
        (r#"{"0":{"00":1.0}}"#, "0", r#"column "00" = 1"#, "alphabet"),
    ] {
        check(GateType::MZ, row, value, fix, &rows(json));
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let input = BTreeMap::from([("0".into(), BTreeMap::from([("0".into(), value)]))]);
        check(
            GateType::MZ,
            "0",
            &format!("column \"0\" = {value}"),
            "finite",
            &input,
        );
    }
    for (gate, fix) in [
        (GateType::MPZ, "MPZ uses MZ's readout matrix; configure MZ"),
        (GateType::MX, "has no readout matrix; flavors are"),
        (GateType::MeasureFree, "has no readout matrix; flavors are"),
        (GateType::X, "has no readout matrix; flavors are"),
    ] {
        check(
            gate,
            "<flavor>",
            &gate.to_string(),
            fix,
            &rows(r#"{"0":{"0":1.0}}"#),
        );
    }
    for (gate, forbidden) in [
        (GateType::MeasureZLost, "2"),
        (GateType::MeasureZLeaked, "3"),
    ] {
        let input = BTreeMap::from([("0".into(), BTreeMap::from([(forbidden.into(), 1.0)]))]);
        check(
            gate,
            "0",
            &format!("column {forbidden:?} = 1"),
            "alphabet",
            &input,
        );
    }
    assert!(
        failures.is_empty(),
        "{} invalid cases failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn lost_rows_construct_but_fail_at_build() {
    for flavor in [
        GateType::MZ,
        GateType::MeasureZLeaked,
        GateType::MeasureZLost,
        GateType::MeasureZLeakedLost,
    ] {
        let matrix = matrix(flavor, r#"{"lost":{"0":1.0}}"#);
        let error = builder()
            .with_readout(matrix)
            .try_build()
            .unwrap_err()
            .to_string();
        let json =
            format!(r#"{{"measurement":{{"readout":{{"{flavor}":{{"lost":{{"0":1.0}}}}}}}}}}"#);
        let json_error = GeneralNoiseConfig::from_json(&json)
            .unwrap()
            .build()
            .unwrap_err()
            .to_string();
        for message in [error, json_error] {
            for part in [
                &flavor.to_string(),
                "lost",
                "value",
                "qubit loss, which is not configured",
                "remove",
            ] {
                assert!(message.contains(part), "{message}");
            }
        }
    }
}

#[test]
fn json_and_builder_replacement_aliases_and_preservation() {
    let a = matrix(GateType::MZ, r#"{"0":{"1":1.0}}"#);
    let b = matrix(GateType::MeasureZLost, r#"{"0":{"3":1.0}}"#);
    let rust = builder().with_readout(a.clone()).with_readout(b.clone());
    let config = GeneralNoiseConfig::from_json(
        r#"{"measurement":{"readout":{"Measure":{"0":{"1":1.0}},"measurezlost":{"0":{"3":1.0}}}}}"#,
    )
    .unwrap();
    let gates = [Gate::mz(&[0]), Gate::measure_z_lost(&[1])];
    let expected = run(&mut rust.clone().build(), &gates);
    assert_eq!(expected.0, [1, 3]);
    assert_eq!(
        run(&mut config.apply_to(builder()).unwrap().build(), &gates),
        expected
    );
    let absent = GeneralNoiseConfig::from_json("{}").unwrap();
    assert_eq!(
        run(&mut absent.apply_to(rust.clone()).unwrap().build(), &gates),
        expected
    );
    let empty = GeneralNoiseConfig::from_json(r#"{"measurement":{"readout":{}}}"#).unwrap();
    assert!(
        !empty
            .apply_to(rust.clone())
            .unwrap()
            .build()
            .has_configured_readout()
    );
    let replace =
        GeneralNoiseConfig::from_json(r#"{"measurement":{"readout":{"MZ":{"0":{"0":1.0}}}}}"#)
            .unwrap();
    assert_eq!(
        run(&mut replace.apply_to(rust.clone()).unwrap().build(), &gates).0,
        [0, 0]
    );
    let replaced = rust.with_readout(matrix(GateType::MZ, r#"{"1":{"1":1.0}}"#));
    assert_eq!(run(&mut replaced.build(), &gates).0, [0, 3]);
    let duplicate = GeneralNoiseConfig::from_json(
        r#"{"measurement":{"readout":{"MZ":{"0":{"0":1.0}},"Measure":{"0":{"0":1.0}}}}}"#,
    )
    .unwrap()
    .build()
    .unwrap_err()
    .to_string();
    assert!(
        duplicate.contains("MZ") && duplicate.contains("Measure") && duplicate.contains("both")
    );
    assert!(GeneralNoiseConfig::from_json(r#"{"measurement":{"readout_typo":{}}}"#).is_err());
    let unknown =
        GeneralNoiseConfig::from_json(r#"{"measurement":{"readout":{"unknown":{"0":{"0":1.0}}}}}"#)
            .unwrap()
            .build()
            .unwrap_err()
            .to_string();
    assert!(unknown.contains("unknown") && unknown.contains("flavor"));
}

#[test]
fn configured_readout_rejects_simple_and_runtime_profiles() {
    let configured = builder().with_readout(matrix(GateType::MZ, r#"{"0":{"0":1.0}}"#));
    assert!(configured.clone().build().has_configured_readout());
    assert!(builder().simple_probabilities().is_some());
    assert!(configured.simple_probabilities().is_none());
    assert!(configured.pauli_with_angle_scaling().is_none());
    let error = RuntimeNoise::new(configured, 1, FrameLimits::default())
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("readout matrices") && error.contains("remove measurement.readout"),
        "{error}"
    );
}

#[test]
fn json_readout_errors_preserve_the_users_flavor_key() {
    let error =
        GeneralNoiseConfig::from_json(r#"{"measurement":{"readout":{"Measure":{"0":{"2":1.0}}}}}"#)
            .unwrap()
            .build()
            .unwrap_err()
            .to_string();
    assert_eq!(error.matches("Input error:").count(), 1, "{error}");
    for part in [
        r#"key "Measure""#,
        "readout MZ",
        r#"row "0""#,
        r#"column "2" = 1"#,
    ] {
        assert!(error.contains(part), "{error} lacks {part}");
    }
}
