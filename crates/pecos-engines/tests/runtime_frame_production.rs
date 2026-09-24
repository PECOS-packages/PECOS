//! All execution here enters exported QuantumSystem/HybridEngine/SimBuilder APIs.
use pecos_core::{QubitId, RngManageable};
use pecos_engines::noise::{GeneralNoiseModelBuilder, IntoNoiseModel};
use pecos_engines::runtime_frame::{
    FLIP, FrameLimits, FrameRecord, METADATA, RuntimeNoise, ShotContext, encode_frame,
};
use pecos_engines::*;
use std::any::Any;
use std::collections::BTreeMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

fn context(shot: usize) -> ShotContext {
    ShotContext {
        run: 41,
        worker: 3,
        shot,
    }
}
fn message(gates: &[Gate]) -> ByteMessage {
    let mut b = ByteMessage::quantum_operations_builder();
    for gate in gates {
        b.add_gate_command(gate);
    }
    b.build()
}
fn frame(gates: &[Gate], metadata: bool) -> ByteMessage {
    let mut records = vec![];
    for gate in gates {
        if metadata {
            records.push(FrameRecord::Event {
                id: METADATA,
                target: 0,
            });
        }
        records.push(FrameRecord::gate(gate.clone()));
    }
    if metadata {
        records.push(FrameRecord::Event {
            id: METADATA,
            target: 0,
        });
    }
    encode_frame(&records).unwrap()
}
fn setup(builder: GeneralNoiseModelBuilder, seed: u64) -> QuantumSystem {
    let noise = RuntimeNoise::new(builder, 2, FrameLimits::default())
        .unwrap()
        .into_noise_model();
    let mut system = QuantumSystem::new(noise, Box::new(StateVecEngine::new(2)));
    system.set_seed(seed);
    system.begin_shot(context(0)).unwrap();
    system
}
fn assert_rng_equal(left: &QuantumSystem, right: &QuantumSystem) {
    // Debug includes the complete RNG state, not merely a short next-word sample.
    assert_eq!(
        format!("{:?}", left.noise_model().rng()),
        format!("{:?}", right.noise_model().rng())
    );
    let left = left
        .quantum_engine()
        .as_any()
        .downcast_ref::<StateVecEngine>()
        .unwrap();
    let right = right
        .quantum_engine()
        .as_any()
        .downcast_ref::<StateVecEngine>()
        .unwrap();
    assert_eq!(format!("{:?}", left.rng()), format!("{:?}", right.rng()));
}

#[test]
fn metadata_matches_ordinary_execution_across_inputs_leakage_idle_and_readout() {
    let builder = GeneralNoiseModel::builder()
        .with_p1(0.375)
        .with_p_meas_0(0.625)
        .with_p_prep(1.0)
        .with_prep_leak_ratio(1.0)
        .with_p_idle_sin_squared(
            std::f64::consts::FRAC_PI_2,
            &BTreeMap::from([("X".into(), 1.0)]),
        );
    let inputs = [
        vec![Gate::pz(&[0])], // persist leakage to a second input
        vec![
            Gate::measure_leaked(&[0]),
            Gate::pz(&[0]),
            Gate::measure_leaked(&[0]),
        ],
        vec![
            Gate::idle(1.0, vec![QubitId(1)]),
            Gate::h(&[1]),
            Gate::mz(&[1]),
            Gate::x(&[1]),
            Gate::mz(&[1]),
        ],
    ];
    for seed in 0..64 {
        let mut annotated = setup(builder.clone(), seed);
        let mut plain = setup(builder.clone(), seed);
        let mut legacy = QuantumSystem::new(
            Box::new(builder.clone().build()),
            Box::new(StateVecEngine::new(2)),
        );
        legacy.set_seed(seed);
        for input in &inputs {
            let expected = legacy.process(message(input)).unwrap().outcomes().unwrap();
            assert_eq!(
                annotated
                    .process(frame(input, true))
                    .unwrap()
                    .outcomes()
                    .unwrap(),
                expected
            );
            assert_eq!(
                plain
                    .process(frame(input, false))
                    .unwrap()
                    .outcomes()
                    .unwrap(),
                expected
            );
            assert_rng_equal(&annotated, &legacy);
            assert_rng_equal(&plain, &legacy);
            assert_eq!(annotated.shot_context(), Some(context(0)));
        }
    }
}

#[test]
fn physical_effect_is_after_raw_measurement_and_is_not_renoised() {
    let gate_boundary = encode_frame(&[
        FrameRecord::gate(Gate::h(&[0])),
        FrameRecord::Event {
            id: FLIP,
            target: 0,
        },
        FrameRecord::gate(Gate::h(&[0])),
        FrameRecord::gate(Gate::mz(&[0])),
    ])
    .unwrap();
    // Moving the X before the first H or after the second H would return one.
    assert_eq!(
        setup(GeneralNoiseModel::builder(), 7)
            .process(gate_boundary)
            .unwrap()
            .outcomes()
            .unwrap(),
        [0]
    );
    let records = [
        FrameRecord::gate(Gate::mz(&[0])),
        FrameRecord::Event {
            id: FLIP,
            target: 0,
        },
        FrameRecord::gate(Gate::mz(&[0])),
    ];
    let mut system = setup(GeneralNoiseModel::builder().with_p1(1.0), 7);
    let mut reference = setup(GeneralNoiseModel::builder().with_p1(1.0), 7);
    assert_eq!(
        system
            .process(encode_frame(&records).unwrap())
            .unwrap()
            .outcomes()
            .unwrap(),
        [0, 1]
    );
    reference
        .process(frame(&[Gate::mz(&[0]), Gate::mz(&[0])], false))
        .unwrap();
    assert_rng_equal(&system, &reference);
    // Persistent physical state remains across original inputs, not just events.
    assert_eq!(
        system
            .process(frame(&[Gate::mz(&[0])], true))
            .unwrap()
            .outcomes()
            .unwrap(),
        [1]
    );
    let mut cloned = system.clone();
    assert!(cloned.process(frame(&[Gate::mz(&[0])], false)).is_err());
    cloned.begin_shot(context(1)).unwrap(); // explicit identity; retain snapshot physics
    assert_eq!(
        cloned
            .process(frame(&[Gate::mz(&[0])], true))
            .unwrap()
            .outcomes()
            .unwrap(),
        [1]
    );
    system.reset().unwrap();
    system.begin_shot(context(2)).unwrap();
    assert_eq!(
        system
            .process(frame(&[Gate::mz(&[0])], false))
            .unwrap()
            .outcomes()
            .unwrap(),
        [0]
    );
}

#[test]
fn crosstalk_completion_occurs_once_after_all_internal_yields() {
    let builder = GeneralNoiseModel::builder()
        .with_p_meas_crosstalk_local(1.0)
        .with_p_meas_crosstalk_model(&BTreeMap::from([
            ("0->1".into(), 1.0),
            ("1->0".into(), 1.0),
        ]));
    let payload = Gate::new(
        GateType::MeasCrosstalkLocalPayload,
        vec![],
        vec![],
        vec![QubitId(1)],
    );
    let first = [Gate::pz(&[0]), Gate::pz(&[1]), Gate::mz(&[0]), payload];
    let mut system = setup(builder.clone(), 8);
    let mut legacy =
        QuantumSystem::new(Box::new(builder.build()), Box::new(StateVecEngine::new(2)));
    legacy.set_seed(8);
    for input in [&first[..], &[Gate::mz(&[1])][..]] {
        assert_eq!(
            system
                .process(frame(input, true))
                .unwrap()
                .outcomes()
                .unwrap(),
            legacy.process(message(input)).unwrap().outcomes().unwrap()
        );
        assert_rng_equal(&system, &legacy);
    }
    // A second completion would flip back to zero.
    assert_eq!(
        system
            .process(frame(&[Gate::mz(&[1])], false))
            .unwrap()
            .outcomes()
            .unwrap(),
        [1]
    );
}

#[test]
fn invalid_suffixes_versions_ids_profiles_and_budgets_reject_before_prefix_effects() {
    let good = frame(&[Gate::x(&[0]), Gate::mz(&[0])], false);
    let base = good.as_bytes().to_vec();
    let mut invalid = vec![];
    for (offset, value) in [(4, 99), (40, 250), (41, 1), (48, 99), (52, 99)] {
        let mut bytes = base.clone();
        bytes[offset] = value;
        invalid.push(bytes);
    }
    let mut trailing = base.clone();
    trailing.push(0);
    invalid.push(trailing);
    for length in 0..base.len() {
        invalid.push(base[..length].to_vec());
    }
    let mut unknown_event = encode_frame(&[
        FrameRecord::gate(Gate::x(&[0])),
        FrameRecord::Event {
            id: METADATA,
            target: 0,
        },
    ])
    .unwrap()
    .into_bytes();
    unknown_event[48..52].copy_from_slice(&9999u32.to_le_bytes());
    invalid.push(unknown_event);
    for bytes in invalid {
        let mut system = setup(GeneralNoiseModel::builder().with_p1(0.3), 19);
        let reference = system.clone();
        assert!(system.process(ByteMessage::new(&bytes)).is_err());
        assert_rng_equal(&system, &reference);
        assert_eq!(
            system
                .process(frame(&[Gate::mz(&[0])], false))
                .unwrap()
                .outcomes()
                .unwrap(),
            [0]
        );
    }
    let mut unsupported = setup(GeneralNoiseModel::builder().with_prep_leak_ratio(0.5), 1);
    let bad = encode_frame(&[
        FrameRecord::gate(Gate::x(&[0])),
        FrameRecord::Event {
            id: FLIP,
            target: 0,
        },
    ])
    .unwrap();
    assert!(unsupported.process(bad.clone()).is_err());
    assert_eq!(
        unsupported
            .process(frame(&[Gate::mz(&[0])], false))
            .unwrap()
            .outcomes()
            .unwrap(),
        [0]
    );
    let mut consumer = QuantumSystem::new_without_noise(Box::new(StateVecEngine::new(2)));
    assert!(consumer.process(bad).is_err());
    assert_eq!(
        consumer
            .process(message(&[Gate::mz(&[0])]))
            .unwrap()
            .outcomes()
            .unwrap(),
        [0]
    );
    let cfg = RuntimeNoise::new(
        GeneralNoiseModel::builder(),
        2,
        FrameLimits {
            records: 2,
            expanded_operations: 48,
        },
    )
    .unwrap();
    let mut limited = QuantumSystem::new(cfg.into_noise_model(), Box::new(StateVecEngine::new(2)));
    limited.begin_shot(context(0)).unwrap();
    assert!(limited.process(good).is_err()); // two-record bound >48
    assert_eq!(
        limited
            .process(frame(&[Gate::mz(&[0])], false))
            .unwrap()
            .outcomes()
            .unwrap(),
        [0]
    );
    let mut identified = Gate::mz(&[0]);
    identified.meas_ids.push(pecos_core::MeasId::from_raw(4));
    assert!(encode_frame(&[FrameRecord::gate(identified)]).is_err());
    assert!(
        RuntimeNoise::new(
            GeneralNoiseModel::builder()
                .with_p1(0.75)
                .with_p1_scale(2.0),
            2,
            FrameLimits::default()
        )
        .is_err()
    );
    assert!(
        encode_frame(&vec![
            FrameRecord::Event {
                id: METADATA,
                target: 0
            };
            129
        ])
        .is_err()
    );
}

#[derive(Clone, Debug)]
struct FailingSimulator {
    fail_process: Arc<AtomicBool>,
    fail_reset: Arc<AtomicBool>,
}
impl Engine for FailingSimulator {
    type Input = ByteMessage;
    type Output = ByteMessage;
    fn process(&mut self, _: ByteMessage) -> Result<ByteMessage, PecosError> {
        if self.fail_process.load(Ordering::SeqCst) {
            Err(PecosError::Processing("synthetic execution failure".into()))
        } else {
            Ok(ByteMessage::outcomes_builder().build())
        }
    }
    fn reset(&mut self) -> Result<(), PecosError> {
        if self.fail_reset.load(Ordering::SeqCst) {
            Err(PecosError::Processing("synthetic reset failure".into()))
        } else {
            Ok(())
        }
    }
}
impl QuantumEngine for FailingSimulator {
    fn set_seed(&mut self, _: u64) {}
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[test]
fn errors_and_failed_reset_poison_owner_and_clones() {
    let fail_process = Arc::new(AtomicBool::new(true));
    let fail_reset = Arc::new(AtomicBool::new(true));
    let sim = FailingSimulator {
        fail_process: fail_process.clone(),
        fail_reset: fail_reset.clone(),
    };
    let cfg = RuntimeNoise::new(GeneralNoiseModel::builder(), 2, FrameLimits::default()).unwrap();
    let mut system = QuantumSystem::new(cfg.into_noise_model(), Box::new(sim));
    system.begin_shot(context(0)).unwrap();
    let msg = message(&[Gate::x(&[0])]);
    assert!(system.process(msg.clone()).is_err());
    fail_process.store(false, Ordering::SeqCst);
    let mut clone = system.clone();
    assert!(system.process(msg.clone()).is_err());
    assert!(clone.begin_shot(context(1)).is_err());
    assert!(system.reset().is_err());
    assert!(system.begin_shot(context(2)).is_err());
    fail_reset.store(false, Ordering::SeqCst);
    system.reset().unwrap();
    system.begin_shot(context(3)).unwrap();
    system.process(msg).unwrap();
}

#[derive(Clone)]
struct Script {
    inputs: Vec<ByteMessage>,
    position: usize,
    outputs: Vec<u32>,
    fail_reset: Arc<AtomicBool>,
    fail_after_input: bool,
}
impl Engine for Script {
    type Input = ();
    type Output = Shot;
    fn process(&mut self, (): ()) -> Result<Shot, PecosError> {
        self.get_results()
    }
    fn reset(&mut self) -> Result<(), PecosError> {
        if self.fail_reset.load(Ordering::SeqCst) {
            return Err(PecosError::Processing("classical reset failure".into()));
        }
        self.position = 0;
        self.outputs.clear();
        Ok(())
    }
}
impl ClassicalEngine for Script {
    fn num_qubits(&self) -> usize {
        2
    }
    fn generate_commands(&mut self) -> Result<ByteMessage, PecosError> {
        Ok(self.inputs[self.position].clone())
    }
    fn handle_measurements(&mut self, reply: ByteMessage) -> Result<(), PecosError> {
        self.outputs.extend(reply.outcomes()?);
        Ok(())
    }
    fn get_results(&self) -> Result<Shot, PecosError> {
        let mut shot = Shot::default();
        for (i, value) in self.outputs.iter().enumerate() {
            shot.data.insert(format!("m{i}"), Data::U32(*value));
        }
        Ok(shot)
    }
    fn compile(&self) -> Result<(), PecosError> {
        Ok(())
    }
    fn reset(&mut self) -> Result<(), PecosError> {
        Engine::reset(self)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
impl ControlEngine for Script {
    type Input = ();
    type Output = Shot;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;
    fn start(&mut self, (): ()) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
        Ok(EngineStage::NeedsProcessing(self.generate_commands()?))
    }
    fn continue_processing(
        &mut self,
        reply: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
        self.handle_measurements(reply)?;
        if self.fail_after_input {
            return Err(PecosError::Processing(
                "classical continuation failure".into(),
            ));
        }
        self.position += 1;
        if self.position == self.inputs.len() {
            Ok(EngineStage::Complete(self.get_results()?))
        } else {
            Ok(EngineStage::NeedsProcessing(self.generate_commands()?))
        }
    }
    fn reset(&mut self) -> Result<(), PecosError> {
        Engine::reset(self)
    }
}
struct ScriptBuilder(Script);
impl ClassicalControlEngineBuilder for ScriptBuilder {
    type Engine = Script;
    fn build(self) -> Result<Script, PecosError> {
        Ok(self.0)
    }
}
fn script(metadata: bool) -> Script {
    Script {
        inputs: vec![
            encode_frame(&[
                FrameRecord::gate(Gate::mz(&[0])),
                FrameRecord::Event {
                    id: FLIP,
                    target: 0,
                },
            ])
            .unwrap(),
            frame(&[Gate::mz(&[0])], metadata),
        ],
        position: 0,
        outputs: vec![],
        fail_reset: Arc::new(AtomicBool::new(false)),
        fail_after_input: false,
    }
}
#[test]
fn sim_builder_multiple_inputs_and_workers_follow_the_production_route() {
    for metadata in [false, true] {
        let mut sim = sim_builder()
            .classical(ScriptBuilder(script(metadata)))
            .quantum(state_vector())
            .qubits(2)
            .noise(
                RuntimeNoise::new(GeneralNoiseModel::builder(), 2, FrameLimits::default()).unwrap(),
            )
            .seed(17)
            .workers(4)
            .build()
            .unwrap();
        for _ in 0..2 {
            let shots = sim.run(24).unwrap();
            for shot in shots.shots {
                assert_eq!(shot.data["m0"], Data::U32(0));
                assert_eq!(shot.data["m1"], Data::U32(1));
            }
        }
    }
}

#[test]
fn classical_failure_requires_whole_host_reset_not_quantum_reset_only() {
    let mut controller = script(true);
    controller.fail_after_input = true;
    let fail = controller.fail_reset.clone();
    let mut host = hybrid::HybridEngineBuilder::new()
        .with_classical_engine(Box::new(controller))
        .with_quantum_system(setup(GeneralNoiseModel::builder(), 1))
        .build();
    host.reset().unwrap();
    assert!(host.run_shot_with_context(context(0)).is_err());
    fail.store(true, Ordering::SeqCst);
    assert!(host.reset().is_err());
    host.quantum_system.reset().unwrap();
    assert!(host.quantum_system.begin_shot(context(1)).is_err());
    fail.store(false, Ordering::SeqCst);
    host.classical_engine
        .as_any_mut()
        .downcast_mut::<Script>()
        .unwrap()
        .fail_after_input = false;
    host.reset().unwrap();
    let result = host.run_shot_with_context(context(2)).unwrap();
    assert_eq!(result.data["m1"], Data::U32(1));
    assert_eq!(host.quantum_system.shot_context(), Some(context(2)));
}

#[test]
fn historical_leakage_boundary_and_unsplit_nonlinear_idle_are_preserved() {
    for (axis, expected) in [("L", vec![0, 0]), ("X", vec![1])] {
        let builder = GeneralNoiseModel::builder().with_p_idle_sin_squared(
            std::f64::consts::FRAC_PI_2,
            &BTreeMap::from([(axis.into(), 1.0)]),
        );
        let mut system = setup(builder.clone(), 21);
        let mut legacy =
            QuantumSystem::new(Box::new(builder.build()), Box::new(StateVecEngine::new(2)));
        legacy.set_seed(21);
        let idle = [Gate::idle(1.0, vec![QubitId(0)])];
        system.process(frame(&idle, true)).unwrap();
        legacy.process(message(&idle)).unwrap();
        let next = if axis == "L" {
            vec![
                Gate::measure_leaked(&[0]),
                Gate::pz(&[0]),
                Gate::measure_leaked(&[0]),
            ]
        } else {
            vec![Gate::mz(&[0])]
        };
        assert_eq!(
            system
                .process(frame(&next, true))
                .unwrap()
                .outcomes()
                .unwrap(),
            expected
        );
        assert_eq!(
            legacy.process(message(&next)).unwrap().outcomes().unwrap(),
            expected
        );
        assert_rng_equal(&system, &legacy);
    }
}

#[test]
fn noisy_workers_replay_identically_with_metadata_and_explicit_identities() {
    let build = |metadata| {
        let mut s = script(metadata);
        s.inputs[0] = frame(&[Gate::h(&[0]), Gate::mz(&[0])], metadata);
        sim_builder()
            .classical(ScriptBuilder(s))
            .quantum(state_vector())
            .qubits(2)
            .noise(
                RuntimeNoise::new(
                    GeneralNoiseModel::builder()
                        .with_p1(0.375)
                        .with_p_meas(0.25),
                    2,
                    FrameLimits::default(),
                )
                .unwrap(),
            )
            .seed(83)
            .workers(4)
            .build()
            .unwrap()
    };
    let mut annotated = build(true);
    let mut unannotated = build(false);
    for _ in 0..3 {
        assert_eq!(
            annotated.run(64).unwrap().shots,
            unannotated.run(64).unwrap().shots
        );
    }
    let threads: Vec<_> = (0..4)
        .map(|worker| {
            std::thread::spawn(move || {
                let ctx = ShotContext {
                    run: 91,
                    worker,
                    shot: 7,
                };
                let mut host = hybrid::HybridEngineBuilder::new()
                    .with_classical_engine(Box::new(script(true)))
                    .with_quantum_system(setup(GeneralNoiseModel::builder(), worker as u64))
                    .build();
                host.reset().unwrap();
                let result = host.run_shot_with_context(ctx).unwrap();
                assert_eq!(result.data["m1"], Data::U32(1));
                assert_eq!(host.quantum_system.shot_context(), Some(ctx));
                ctx
            })
        })
        .collect();
    let identities: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    for (worker, ctx) in identities.iter().enumerate() {
        assert_eq!(ctx.worker, worker);
    }
}

#[test]
fn capacity_edges_and_mutable_access_require_explicit_readmission() {
    let records = vec![
        FrameRecord::Event {
            id: METADATA,
            target: 0
        };
        128
    ];
    let exact = encode_frame(&records).unwrap();
    let mut system = setup(GeneralNoiseModel::builder(), 3);
    system.process(exact.clone()).unwrap();
    let mut over = exact.into_bytes();
    over[8..12].copy_from_slice(&129u32.to_le_bytes());
    assert!(system.process(ByteMessage::new(&over)).is_err());
    let mut bad_idle =
        frame(&[Gate::x(&[0]), Gate::idle(1.0, vec![QubitId(0)])], false).into_bytes();
    bad_idle[56..64].copy_from_slice(&f64::NAN.to_le_bytes());
    assert!(system.process(ByteMessage::new(&bad_idle)).is_err());
    assert_eq!(
        system
            .process(frame(&[Gate::mz(&[0])], false))
            .unwrap()
            .outcomes()
            .unwrap(),
        [0]
    );
    let _ = system.noise_model_mut();
    assert!(system.begin_shot(context(1)).is_err());
    system.reset().unwrap();
    system.begin_shot(context(1)).unwrap();
    system.process(frame(&[Gate::mz(&[0])], false)).unwrap();
    let mut unsupported_sim = QuantumSystem::new(
        RuntimeNoise::new(GeneralNoiseModel::builder(), 2, FrameLimits::default())
            .unwrap()
            .into_noise_model(),
        Box::new(StabilizerEngine::new(2)),
    );
    unsupported_sim.begin_shot(context(0)).unwrap();
    assert!(
        unsupported_sim
            .process(frame(&[Gate::x(&[0])], false))
            .is_err()
    );
    let mut too_small = QuantumSystem::new(
        RuntimeNoise::new(GeneralNoiseModel::builder(), 2, FrameLimits::default())
            .unwrap()
            .into_noise_model(),
        Box::new(StateVecEngine::new(1)),
    );
    too_small.begin_shot(context(0)).unwrap();
    assert!(too_small.process(frame(&[Gate::x(&[0])], false)).is_err());
}

#[test]
fn resolved_configuration_and_idle_arithmetic_reject_nonfinite_values() {
    for builder in [
        GeneralNoiseModel::builder().with_scale(f64::NAN),
        GeneralNoiseModel::builder()
            .with_scale(1e200)
            .with_p1_scale(1e200),
    ] {
        assert!(RuntimeNoise::new(builder, 2, FrameLimits::default()).is_err());
    }
    let builder = GeneralNoiseModel::builder()
        .with_p_idle_sin_squared(1e300, &BTreeMap::from([("X".into(), 1.0)]));
    let mut system = setup(builder, 3);
    let bad = frame(&[Gate::x(&[0]), Gate::idle(1e100, vec![QubitId(0)])], false);
    assert!(system.process(bad).is_err());
    assert_eq!(
        system
            .process(frame(&[Gate::mz(&[0])], false))
            .unwrap()
            .outcomes()
            .unwrap(),
        [0]
    );
}
