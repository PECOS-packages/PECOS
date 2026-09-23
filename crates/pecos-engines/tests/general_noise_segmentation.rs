//! Contract investigation on the existing model, not an event bridge.
use pecos_core::RngManageable;
use pecos_engines::noise::GeneralNoiseModel;
use pecos_engines::quantum::StateVecEngine;
use pecos_engines::{ByteMessage, ControlEngine, Engine, EngineStage, QuantumSystem};
use std::collections::BTreeMap;

fn program(ops: &str) -> ByteMessage {
    let mut b = ByteMessage::quantum_operations_builder();
    for op in ops.chars() {
        match op {
            'M' => {
                b.mz(&[0]);
            }
            'L' => {
                b.measure_leakages(&[0]);
            }
            'P' => {
                b.pz(&[0]);
            }
            'X' => {
                b.x(&[0]);
            }
            'I' => {
                b.idle(0.25, &[0]);
            }
            _ => panic!("invalid fixture"),
        }
    }
    b.build()
}

fn rng_tail(system: &QuantumSystem) -> [u64; 8] {
    let mut rng = system.noise_model().rng().clone();
    std::array::from_fn(|_| rng.next_u64())
}

#[test]
fn general_noise_segmentation_changes_leakage_readout() {
    let mut model = GeneralNoiseModel::default();
    model.mark_as_leaked(0);
    let mut whole = QuantumSystem::new(Box::new(model.clone()), Box::new(StateVecEngine::new(1)));
    let mut split = QuantumSystem::new(Box::new(model), Box::new(StateVecEngine::new(1)));
    // Preparation is processed at start, but leakage readout at continuation.
    assert_eq!(
        whole.process(program("LP")).unwrap().outcomes().unwrap(),
        [0]
    );
    assert_eq!(
        split.process(program("L")).unwrap().outcomes().unwrap(),
        [2]
    );
    split.process(program("P")).unwrap();
    // This is a deterministic distribution change, not just RNG reordering.
}

#[test]
fn general_noise_segmentation_reorders_gate_and_readout_randomness() {
    let mut changed_outcomes = 0;
    let mut changed_rng = 0;
    for seed in 0..64 {
        let model = GeneralNoiseModel::builder()
            .with_p1(0.375)
            .with_p_meas_0(0.625)
            .with_p_meas_1(0.25)
            .with_seed(seed)
            .build();
        let mut whole =
            QuantumSystem::new(Box::new(model.clone()), Box::new(StateVecEngine::new(1)));
        let mut split = QuantumSystem::new(Box::new(model), Box::new(StateVecEngine::new(1)));
        let expected = whole.process(program("MXM")).unwrap().outcomes().unwrap();
        let mut actual = split.process(program("M")).unwrap().outcomes().unwrap();
        actual.extend(split.process(program("XM")).unwrap().outcomes().unwrap());
        changed_outcomes += usize::from(expected != actual);
        changed_rng += usize::from(rng_tail(&whole) != rng_tail(&split));
    }
    assert!(
        changed_outcomes > 0,
        "segmentation unexpectedly preserved all outcomes"
    );
    assert!(
        changed_rng > 0,
        "segmentation unexpectedly preserved all RNG tails"
    );
}

#[test]
fn measurement_free_prefix_splits_preserve_seeded_results_in_this_profile() {
    // Positive control only: it does not certify other profiles or boundaries.
    // All original gates are retained, with no lifecycle reset at a boundary.
    for seed in 0..64 {
        for boundary in 1..5 {
            let model = GeneralNoiseModel::builder()
                .with_p_prep(0.125)
                .with_p1(0.375)
                .with_p_idle_linear(0.125, &BTreeMap::from([("X".to_owned(), 1.0)]))
                .with_p_meas_0(0.625)
                .with_p_meas_1(0.25)
                .with_seed(seed)
                .build();
            let mut whole =
                QuantumSystem::new(Box::new(model.clone()), Box::new(StateVecEngine::new(1)));
            let mut split = QuantumSystem::new(Box::new(model), Box::new(StateVecEngine::new(1)));
            let source = "PIXXM";
            let expected = whole.process(program(source)).unwrap().outcomes().unwrap();
            assert!(
                split
                    .process(program(&source[..boundary]))
                    .unwrap()
                    .outcomes()
                    .unwrap()
                    .is_empty()
            );
            let actual = split
                .process(program(&source[boundary..]))
                .unwrap()
                .outcomes()
                .unwrap();
            assert_eq!(actual, expected, "seed {seed}, boundary {boundary}");
            assert_eq!(
                rng_tail(&split),
                rng_tail(&whole),
                "seed {seed}, boundary {boundary}"
            );
        }
    }
}

#[test]
fn splitting_an_idle_can_change_the_noise_distribution() {
    let mut differences = 0;
    for seed in 0..64 {
        let model = GeneralNoiseModel::builder()
            .with_p_idle_sin_squared(
                std::f64::consts::FRAC_PI_2,
                &BTreeMap::from([("X".to_owned(), 1.0)]),
            )
            .with_seed(seed)
            .build();
        let mut whole =
            QuantumSystem::new(Box::new(model.clone()), Box::new(StateVecEngine::new(1)));
        let mut split = QuantumSystem::new(Box::new(model), Box::new(StateVecEngine::new(1)));
        let mut b = ByteMessage::quantum_operations_builder();
        b.idle(1.0, &[0]).mz(&[0]);
        let expected = whole.process(b.build()).unwrap().outcomes().unwrap();
        let mut b = ByteMessage::quantum_operations_builder();
        b.idle(0.5, &[0]);
        split.process(b.build()).unwrap();
        let mut b = ByteMessage::quantum_operations_builder();
        b.idle(0.5, &[0]).mz(&[0]);
        let actual = split.process(b.build()).unwrap().outcomes().unwrap();
        assert_eq!(expected, [1]);
        differences += usize::from(expected != actual);
    }
    assert!(differences > 0);
}

fn crosstalk_model() -> GeneralNoiseModel {
    GeneralNoiseModel::builder()
        .with_p_meas_crosstalk_local(1.0)
        .with_p_meas_crosstalk_model(&BTreeMap::from([
            ("0->1".to_owned(), 1.0),
            ("1->0".to_owned(), 1.0),
        ]))
        .with_seed(19)
        .build()
}

fn pending_crosstalk(model: &mut GeneralNoiseModel) {
    let mut b = ByteMessage::quantum_operations_builder();
    b.pz(&[0, 1]);
    b.mz(&[0]);
    b.meas_crosstalk_local_payload(&[1]);
    let EngineStage::NeedsProcessing(commands) = model.start(b.build()).unwrap() else {
        panic!("expected simulator work");
    };
    let mut sim = StateVecEngine::new(2);
    let outcomes = sim.process(commands).unwrap();
    let EngineStage::NeedsProcessing(effects) = model.continue_processing(outcomes).unwrap() else {
        panic!("expected crosstalk continuation");
    };
    assert!(!effects.quantum_ops().unwrap().is_empty());
}

#[test]
fn reset_after_continuation_error_discards_pending_results() {
    let mut model = crosstalk_model();
    pending_crosstalk(&mut model);
    // The simulator/controller exchange fails while a user outcome is retained.
    let unexpected_outcome = ByteMessage::outcomes_builder().add_outcomes(&[0]).build();
    assert!(model.continue_processing(unexpected_outcome).is_err());
    model.reset().unwrap();
    let mut system = QuantumSystem::new(Box::new(model), Box::new(StateVecEngine::new(2)));
    assert_eq!(
        system.process(program("M")).unwrap().outcomes().unwrap(),
        [0]
    );
}

#[test]
fn resetting_clone_does_not_clear_original_pending_results() {
    let mut original = crosstalk_model();
    pending_crosstalk(&mut original);
    let mut cloned = original.clone();
    cloned.reset().unwrap();
    let empty_outcomes = ByteMessage::outcomes_builder().build();
    let EngineStage::Complete(result) = original.continue_processing(empty_outcomes).unwrap()
    else {
        panic!("expected completion");
    };
    assert_eq!(result.outcomes().unwrap(), [0]);
    assert!(cloned.start(program("M")).is_ok());
}

#[test]
fn reset_workers_match_fresh_models_with_identical_seeds() {
    let template = crosstalk_model();
    let workers: Vec<_> = (0..4)
        .map(|seed| {
            let mut model = template.clone();
            std::thread::spawn(move || {
                model.mark_as_leaked(0);
                model.reset().unwrap();
                model.set_seed(seed);
                let mut system =
                    QuantumSystem::new(Box::new(model), Box::new(StateVecEngine::new(2)));
                let result = system.process(program("LXM")).unwrap().outcomes().unwrap();
                (seed, result, rng_tail(&system))
            })
        })
        .collect();
    for worker in workers {
        let (seed, result, tail) = worker.join().unwrap();
        let mut model = crosstalk_model();
        model.set_seed(seed);
        let mut reference = QuantumSystem::new(Box::new(model), Box::new(StateVecEngine::new(2)));
        assert_eq!(
            reference
                .process(program("LXM"))
                .unwrap()
                .outcomes()
                .unwrap(),
            result
        );
        assert_eq!(rng_tail(&reference), tail);
    }
}
