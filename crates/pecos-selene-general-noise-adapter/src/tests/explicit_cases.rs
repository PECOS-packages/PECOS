use super::*;
use crate::{
    CrosstalkExposure as Exposure, CrosstalkPlacement as Placement, ExplicitCrosstalkErrorModel,
};
use pecos_core::errors::PecosError;
use pecos_engines::GeneralNoiseConfig;
use pecos_engines::prelude::{ControlEngine, PassThroughNoiseModel};

fn config(model: &str) -> GeneralNoiseConfig {
    GeneralNoiseConfig::from_json(model).unwrap()
}
fn flip() -> GeneralNoiseConfig {
    config(
        r#"{"measurement":{"local_crosstalk_probability":1.0,"global_crosstalk_probability":1.0,"crosstalk_model":{"0->1":1.0,"1->0":1.0}}}"#,
    )
}
fn setup(config: &GeneralNoiseConfig) -> (ExplicitCrosstalkErrorModel, ClassicalSimulator) {
    let mut model = ExplicitCrosstalkErrorModel::new(2, config).unwrap();
    model.shot_start(0, 17).unwrap();
    let mut simulator = ClassicalSimulator::with_qubits(2);
    model
        .handle_operations(
            runtime_batch(
                vec![
                    Operation::Reset { qubit_id: 0 },
                    Operation::Reset { qubit_id: 1 },
                ],
                0,
                1,
            ),
            &[],
            &mut simulator,
        )
        .unwrap();
    (model, simulator)
}
fn measure(q: u64, result_id: u64) -> Operation {
    Operation::Measure {
        qubit_id: q,
        result_id,
    }
}
fn local(index: usize, qubits: Vec<usize>) -> Placement {
    Placement {
        before_operation: index,
        exposure: Exposure::Local(qubits),
    }
}

#[test]
fn explicit_order_repetition_and_end_placement_preserve_user_results() {
    let (mut model, mut simulator) = setup(&flip());
    let result = model
        .handle_operations(
            runtime_batch(vec![measure(0, 71), measure(1, 93)], 1, 1),
            &[
                local(0, vec![0]),
                local(1, vec![1]),
                local(1, vec![1]),
                Placement {
                    before_operation: 2,
                    exposure: Exposure::GlobalExcluding(vec![0]),
                },
            ],
            &mut simulator,
        )
        .unwrap();
    assert_eq!(
        result
            .bool_results
            .iter()
            .map(|r| (r.result_id, r.value))
            .collect::<Vec<_>>(),
        vec![(71, true), (93, false)]
    );
    assert!(result.u64_results.is_empty());
    // The trailing global effect is retained and executes after user measurement.
    let next = model
        .handle_operations(
            runtime_batch(vec![measure(1, u64::MAX)], 2, 1),
            &[],
            &mut simulator,
        )
        .unwrap();
    assert_eq!(next.bool_results[0].result_id, u64::MAX);
    assert!(next.bool_results[0].value);
    model.shot_end().unwrap();
}

#[test]
fn empty_explicit_list_disables_automatic_exposures() {
    let (mut model, mut simulator) = setup(&flip());
    model
        .handle_operations(
            runtime_batch(vec![measure(0, 0)], 1, 1),
            &[],
            &mut simulator,
        )
        .unwrap();
    let out = model
        .handle_operations(
            runtime_batch(vec![measure(1, 1)], 2, 1),
            &[],
            &mut simulator,
        )
        .unwrap();
    assert!(!out.bool_results[0].value);
    assert!(
        !simulator
            .received
            .iter()
            .flatten()
            .any(|op| matches!(op, Operation::RXYGate { .. }))
    );
}

#[test]
fn leakage_exposure_survives_batches_and_does_not_leak_internal_ids() {
    let cfg = config(
        r#"{"measurement":{"local_crosstalk_probability":1.0,"crosstalk_model":{"0->L":1.0,"1->L":1.0}}}"#,
    );
    let (mut model, mut simulator) = setup(&cfg);
    let out = model
        .handle_operations(
            runtime_batch(vec![], 1, 100),
            &[local(0, vec![1])],
            &mut simulator,
        )
        .unwrap();
    assert!(out.bool_results.is_empty() && out.u64_results.is_empty());
    let out = model
        .handle_operations(
            runtime_batch(
                vec![
                    Operation::MeasureLeaked {
                        qubit_id: 1,
                        result_id: 301,
                    },
                    measure(1, 302),
                ],
                101,
                1,
            ),
            &[],
            &mut simulator,
        )
        .unwrap();
    assert_eq!(out.u64_results.len(), 1);
    assert_eq!(
        (out.u64_results[0].result_id, out.u64_results[0].value),
        (301, 2)
    );
    assert_eq!(out.bool_results.len(), 1);
    assert_eq!(
        (out.bool_results[0].result_id, out.bool_results[0].value),
        (302, true)
    );
    model.shot_end().unwrap();
    model.shot_start(1, 17).unwrap();
    simulator = ClassicalSimulator::with_qubits(2);
    let out = model
        .handle_operations(
            // No reset first: shot_end alone must clear the leaked state.
            runtime_batch(
                vec![Operation::MeasureLeaked {
                    qubit_id: 1,
                    result_id: 303,
                }],
                1000,
                1,
            ),
            &[],
            &mut simulator,
        )
        .unwrap();
    assert_eq!(out.u64_results[0].value, 0);
}

#[test]
fn exposure_batches_do_not_split_nonlinear_idle_or_advance_occupancy() {
    let cfg =
        config(r#"{"idle":{"sin_squared_rate":1.5707963267948966,"sin_squared_model":{"X":1.0}}}"#);
    let (mut model, mut simulator) = setup(&cfg);
    model
        .handle_operations(
            runtime_batch(vec![], 250_000_001, 500_000_000),
            &[local(0, vec![1])],
            &mut simulator,
        )
        .unwrap();
    let out = model
        .handle_operations(
            runtime_batch(vec![measure(1, 40)], 1_000_000_001, 1),
            &[],
            &mut simulator,
        )
        .unwrap();
    // One complete second produces a certain X; splitting/advancing at the
    // exposure would change this deterministic result.
    assert!(out.bool_results[0].value);
}

#[test]
fn whole_batch_rejection_does_not_mutate_clock_rng_or_buffer() {
    let invalid_ops = vec![
        vec![
            measure(0, 3),
            Operation::RZGate {
                qubit_id: 1,
                theta: f64::NAN,
            },
        ],
        vec![measure(0, 3), measure(1, 3)],
        vec![
            measure(0, 3),
            Operation::RZZGate {
                qubit_id_1: 1,
                qubit_id_2: 1,
                theta: 0.5,
            },
        ],
        vec![measure(0, 3), measure(2, 4)],
        vec![
            measure(0, 3),
            Operation::Custom {
                custom_tag: 77,
                data: vec![1].into(),
            },
        ],
    ];
    for ops in invalid_ops {
        let (mut model, mut simulator) = setup(&flip());
        let before = simulator.received.clone();
        assert!(
            model
                .handle_operations(
                    runtime_batch(ops, 9, 1),
                    &[local(0, vec![1])],
                    &mut simulator
                )
                .is_err()
        );
        assert_eq!(simulator.received, before);
        // Valid earlier time proves failed admission did not update occupancy;
        // no pending measurement or exposure from the rejected batch survives.
        let out = model
            .handle_operations(
                runtime_batch(vec![measure(1, 8)], 1, 1),
                &[],
                &mut simulator,
            )
            .unwrap();
        assert!(!out.bool_results[0].value);
        model.shot_end().unwrap();
    }
}

#[test]
fn malformed_placements_reject_without_dispatch_and_name_the_bad_value() {
    for (placements, message) in [
        (
            vec![local(2, vec![0])],
            "crosstalk placement 0 is at position 2, but the batch has 1 operations",
        ),
        (
            vec![local(1, vec![0]), local(0, vec![1])],
            "crosstalk placement 1 is at position 0 after a placement at position 1",
        ),
        (
            vec![local(0, vec![0, 0])],
            "crosstalk placement 0 lists qubit 0 more than once",
        ),
        (
            vec![local(0, vec![]), local(0, vec![2])],
            "crosstalk placement 1 lists qubit 2, but the simulation has 2 qubits",
        ),
        (
            vec![Placement {
                before_operation: 0,
                exposure: Exposure::GlobalExcluding(vec![1, 1]),
            }],
            "crosstalk placement 0 lists qubit 1 more than once",
        ),
    ] {
        let (mut model, mut simulator) = setup(&flip());
        let before = simulator.received.clone();
        let error = model
            .handle_operations(
                runtime_batch(vec![measure(0, 7)], 1, 1),
                &placements,
                &mut simulator,
            )
            .err()
            .expect("malformed placement must be rejected");
        assert!(error.to_string().contains(message), "{error}");
        assert_eq!(simulator.received, before);
        model.shot_end().unwrap();
    }
}

#[test]
fn large_batches_and_per_measurement_global_exposures_are_admitted() {
    fn resets(n_qubits: usize) -> Vec<Operation> {
        (0..u64::try_from(n_qubits).unwrap())
            .map(|qubit_id| Operation::Reset { qubit_id })
            .collect()
    }
    // One reset layer across every qubit is a single ordinary batch.
    let mut model = ExplicitCrosstalkErrorModel::new(5000, &flip()).unwrap();
    model.shot_start(0, 3).unwrap();
    model
        .handle_operations(runtime_batch(resets(5000), 0, 1), &[], &mut ZeroSimulator)
        .unwrap();
    model.shot_end().unwrap();

    // Each measurement exposes every other prepared qubit globally.
    let n_qubits = 200;
    let mut model = ExplicitCrosstalkErrorModel::new(n_qubits, &flip()).unwrap();
    model.shot_start(0, 3).unwrap();
    model
        .handle_operations(
            runtime_batch(resets(n_qubits), 0, 1),
            &[],
            &mut ZeroSimulator,
        )
        .unwrap();
    let measurements = (0..u64::try_from(n_qubits).unwrap())
        .map(|q| measure(q, q))
        .collect();
    let placements = (0..n_qubits)
        .map(|index| Placement {
            before_operation: index,
            exposure: Exposure::GlobalExcluding(vec![index]),
        })
        .collect::<Vec<_>>();
    let out = model
        .handle_operations(
            runtime_batch(measurements, 1, 1),
            &placements,
            &mut ZeroSimulator,
        )
        .unwrap();
    assert_eq!(out.bool_results.len(), n_qubits);
    model.shot_end().unwrap();
}

#[test]
fn invalid_timing_rejected_before_dispatch() {
    let (mut model, mut simulator) = setup(&flip());
    let before = simulator.received.clone();
    for batch in [
        runtime_batch(vec![measure(0, 4)], 0, 1),
        runtime_batch(vec![measure(0, 4)], u64::MAX, 1),
        BatchOperation::error_model(vec![measure(0, 4)]),
    ] {
        assert!(model.handle_operations(batch, &[], &mut simulator).is_err());
        assert_eq!(simulator.received, before);
    }
}

struct FailingSimulator {
    panic: bool,
    calls: usize,
}
impl SimulatorInterface for FailingSimulator {
    fn exit(&mut self) -> Result<()> {
        Ok(())
    }
    fn shot_start(&mut self, _: u64, _: u64) -> Result<()> {
        Ok(())
    }
    fn shot_end(&mut self) -> Result<()> {
        Ok(())
    }
    fn handle_operations(&mut self, _: BatchOperation) -> Result<BatchResult> {
        self.calls += 1;
        assert!(!self.panic, "synthetic simulator panic");
        anyhow::bail!("synthetic simulator failure")
    }
    fn get_metric(&mut self, _: u8) -> Result<Option<(String, MetricValue)>> {
        Ok(None)
    }
}
#[test]
fn execution_error_or_caught_panic_requires_fresh_model_and_simulator() {
    for panic in [false, true] {
        let (mut model, _) = setup(&flip());
        let mut simulator = FailingSimulator { panic, calls: 0 };
        let attempted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            model.handle_operations(
                runtime_batch(vec![measure(0, 4)], 1, 1),
                &[local(0, vec![0])],
                &mut simulator,
            )
        }));
        assert!(attempted.is_err() || attempted.unwrap().is_err());
        assert_eq!(simulator.calls, 1);
        assert!(model.shot_end().is_err());
        assert!(model.shot_start(1, 7).is_err());
        assert!(
            model
                .handle_operations(
                    runtime_batch(vec![measure(0, 4)], 2, 1),
                    &[],
                    &mut simulator
                )
                .is_err()
        );
        assert_eq!(simulator.calls, 1);
        let (mut fresh, mut reset_simulator) = setup(&flip());
        let out = fresh
            .handle_operations(
                runtime_batch(vec![measure(0, 4)], 1, 1),
                &[],
                &mut reset_simulator,
            )
            .unwrap();
        assert!(!out.bool_results[0].value);
    }
}

#[test]
fn explicit_lifecycle_cannot_restart_an_active_shot() {
    let mut model = ExplicitCrosstalkErrorModel::new(2, &flip()).unwrap();
    assert!(model.shot_end().is_err());
    assert!(
        model
            .handle_operations(runtime_batch(vec![], 0, 0), &[], &mut ZeroSimulator)
            .is_err()
    );
    model.shot_start(0, 1).unwrap();
    assert!(model.shot_start(1, 2).is_err());
    model.shot_end().unwrap();
    model.shot_start(1, 2).unwrap();
    model.shot_end().unwrap();
}

#[test]
fn seeded_explicit_batches_match_unsplit_direct_controller_with_hooks() {
    let cfg = config(
        r#"{"measurement":{"local_crosstalk_probability":0.41,"global_crosstalk_probability":0.63,"crosstalk_model":{"0->1":0.8,"0->L":0.2,"1->0":0.7,"1->L":0.3}},"single_qubit":{"hooks_before":[{"kind":"transition","probability":0.7,"transitions":{"0->1":0.4,"0->0":0.6,"1->0":0.3,"1->1":0.7}}]}}"#,
    );
    for seed in 0..32 {
        let mut model = ExplicitCrosstalkErrorModel::new(2, &cfg).unwrap();
        model.shot_start(0, seed).unwrap();
        let mut direct: Box<dyn NoiseModel> = Box::new(cfg.build().unwrap());
        direct.set_seed(seed);
        let mut simulator = ClassicalSimulator::with_qubits(2);
        let mut direct_simulator = ClassicalSimulator::with_qubits(2);
        let mut b = ByteMessage::quantum_operations_builder();
        b.pz(&[0]);
        b.pz(&[1]);
        b.meas_crosstalk_local_payload(&[1]);
        b.meas_crosstalk_local_payload(&[0]);
        b.rz(Angle64::try_from_radians(0.5).unwrap(), &[0]);
        b.mz(&[1]);
        b.meas_crosstalk_global_payload(&[1]);
        b.measure_leakages(&[0]);
        b.meas_crosstalk_local_payload(&[1]);
        let expected = process_direct_model(direct.as_mut(), b.build(), &mut direct_simulator)
            .outcomes()
            .unwrap();
        let out = model
            .handle_operations(
                runtime_batch(
                    vec![
                        Operation::Reset { qubit_id: 0 },
                        Operation::Reset { qubit_id: 1 },
                        Operation::RZGate {
                            qubit_id: 0,
                            theta: 0.5,
                        },
                        measure(1, 8),
                        Operation::MeasureLeaked {
                            qubit_id: 0,
                            result_id: 19,
                        },
                    ],
                    0,
                    9,
                ),
                &[
                    local(2, vec![1]),
                    local(2, vec![0]),
                    Placement {
                        before_operation: 4,
                        exposure: Exposure::GlobalExcluding(vec![1]),
                    },
                    local(5, vec![1]),
                ],
                &mut simulator,
            )
            .unwrap();
        assert_eq!(
            vec![
                u32::from(out.bool_results[0].value),
                u32::try_from(out.u64_results[0].value).unwrap()
            ],
            expected,
            "seed {seed}"
        );
        assert_eq!(simulator.received, direct_simulator.received, "seed {seed}");
    }
}

#[test]
fn exposure_precedes_the_deferred_idle_of_its_operation() {
    // A full second of idle is a certain X; the exposure flips only |0> to |1>.
    let cfg = config(
        r#"{"idle":{"sin_squared_rate":1.5707963267948966,"sin_squared_model":{"X":1.0}},"measurement":{"local_crosstalk_probability":1.0,"crosstalk_model":{"0->1":1.0,"1->1":1.0}}}"#,
    );
    let (mut model, mut simulator) = setup(&cfg);
    let out = model
        .handle_operations(
            runtime_batch(vec![measure(1, 50)], 1_000_000_001, 1),
            &[local(0, vec![1])],
            &mut simulator,
        )
        .unwrap();
    // Exposure then idle: |0> -> |1> -> |0>. Idle then exposure would leave |1>.
    assert!(!out.bool_results[0].value);
}

#[test]
fn rejected_admission_preserves_the_noise_stream() {
    let cfg = config(
        r#"{"measurement":{"local_crosstalk_probability":0.41,"global_crosstalk_probability":0.63,"crosstalk_model":{"0->1":0.8,"0->L":0.2,"1->0":0.7,"1->L":0.3}}}"#,
    );
    let run = |reject_midway: bool| {
        let mut model = ExplicitCrosstalkErrorModel::new(2, &cfg).unwrap();
        model.shot_start(0, 5).unwrap();
        let mut simulator = ClassicalSimulator::with_qubits(2);
        let mut values = Vec::new();
        for batch in 0..8_u64 {
            if reject_midway && batch == 4 {
                assert!(
                    model
                        .handle_operations(
                            runtime_batch(vec![measure(0, 99)], batch * 10, 1),
                            &[local(0, vec![1, 1])],
                            &mut simulator,
                        )
                        .is_err()
                );
            }
            let out = model
                .handle_operations(
                    runtime_batch(
                        vec![
                            Operation::Reset { qubit_id: 0 },
                            Operation::Reset { qubit_id: 1 },
                            measure(0, 2 * batch),
                            measure(1, 2 * batch + 1),
                        ],
                        batch * 10,
                        1,
                    ),
                    &[
                        local(2, vec![1]),
                        Placement {
                            before_operation: 3,
                            exposure: Exposure::GlobalExcluding(vec![1]),
                        },
                    ],
                    &mut simulator,
                )
                .unwrap();
            values.extend(out.bool_results.iter().map(|r| (r.result_id, r.value)));
        }
        model.shot_end().unwrap();
        (values, simulator.received)
    };
    assert_eq!(run(true), run(false));
}

#[test]
fn crosstalk_payload_lists_fit_one_record_on_both_paths() {
    let n_qubits = 300;
    for (exposure, admitted) in [
        (Exposure::Local((0..255).collect()), true),
        (Exposure::GlobalExcluding((0..255).collect()), true),
        (Exposure::Local((0..256).collect()), false),
        (Exposure::GlobalExcluding((0..256).collect()), false),
    ] {
        let mut model = ExplicitCrosstalkErrorModel::new(n_qubits, &flip()).unwrap();
        model.shot_start(0, 1).unwrap();
        let result = model.handle_operations(
            runtime_batch(vec![], 0, 0),
            &[Placement {
                before_operation: 0,
                exposure,
            }],
            &mut ZeroSimulator,
        );
        if admitted {
            assert!(result.is_ok());
        } else {
            let error = result.err().expect("oversized payload must be rejected");
            assert!(
                error
                    .to_string()
                    .contains("lists 256 qubits, but one crosstalk payload holds at most 255"),
                "{error}"
            );
            model.shot_end().unwrap();
        }
    }

    // The automatic path encodes measured qubits and local victims the same way.
    let measurements =
        |count: u64| -> Vec<Operation> { (0..count).map(|q| measure(q, q)).collect() };
    let groups = format!(
        r#"{{"model":{{}},"local_groups":[{:?}]}}"#,
        (0..n_qubits).collect::<Vec<_>>()
    );
    for (config_json, count, message) in [
        (r#"{"model":{}}"#, 255, None),
        (
            r#"{"model":{}}"#,
            256,
            Some(
                "batch measures 256 qubits, but one measurement-crosstalk payload holds at most 255",
            ),
        ),
        (
            groups.as_str(),
            1,
            Some(
                "batch has 299 local crosstalk victims, but one measurement-crosstalk payload holds at most 255",
            ),
        ),
    ] {
        let mut model = build_error_model(config_json, n_qubits);
        model.shot_start(0, 1).unwrap();
        let result =
            model.handle_operations(runtime_batch(measurements(count), 0, 1), &mut ZeroSimulator);
        match message {
            None => assert!(result.is_ok()),
            Some(message) => {
                let error = result.err().expect("oversized payload must be rejected");
                assert!(error.to_string().contains(message), "{error}");
            }
        }
    }
}

#[derive(Clone, Default)]
struct ResetFailsModel(PassThroughNoiseModel);

impl NoiseModel for ResetFailsModel {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl pecos_core::RngManageable for ResetFailsModel {
    type Rng = <PassThroughNoiseModel as pecos_core::RngManageable>::Rng;

    fn set_rng(&mut self, rng: Self::Rng) {
        self.0.set_rng(rng);
    }

    fn rng(&self) -> &Self::Rng {
        self.0.rng()
    }

    fn rng_mut(&mut self) -> &mut Self::Rng {
        self.0.rng_mut()
    }
}

impl ControlEngine for ResetFailsModel {
    type Input = ByteMessage;
    type Output = ByteMessage;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    fn start(
        &mut self,
        input: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, ByteMessage>, PecosError> {
        self.0.start(input)
    }

    fn continue_processing(
        &mut self,
        result: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, ByteMessage>, PecosError> {
        self.0.continue_processing(result)
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        Err(PecosError::Generic("synthetic reset failure".into()))
    }
}

#[test]
fn failed_shot_end_poisons_the_explicit_owner() {
    let mut model = ExplicitCrosstalkErrorModel::from_inner(GeneralNoiseErrorModel {
        model: Box::new(ResetFailsModel::default()),
        last_operation_end: vec![None; 1],
        local_groups: Vec::new(),
    });
    model.shot_start(0, 1).unwrap();
    assert!(model.shot_end().is_err());
    assert!(
        model
            .handle_operations(runtime_batch(vec![], 0, 0), &[], &mut ZeroSimulator)
            .is_err()
    );
    assert!(model.shot_start(1, 1).is_err());
}
