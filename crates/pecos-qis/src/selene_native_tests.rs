//! Runtime regressions for native submission and queue ordering.
use super::*;
use crate::selene_native::tests::apply;
use pecos_core::QubitId;
use pecos_simulators::{CliffordGateable, StateVecAoS};
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
enum Route {
    Flat,
    Metadata,
    Scheduled,
}
impl Route {
    const ALL: [Self; 3] = [Self::Flat, Self::Metadata, Self::Scheduled];

    fn lower(self, runtime: &mut SeleneRuntime, ops: &[Operation]) -> Vec<QuantumOp> {
        match self {
            Self::Flat => runtime.lower_operations(ops).unwrap(),
            Self::Metadata => runtime
                .lower_operations_with_metadata(ops)
                .unwrap()
                .into_iter()
                .map(|op| op.op)
                .collect(),
            Self::Scheduled => runtime
                .lower_scheduled_operations(ops)
                .unwrap()
                .into_iter()
                .flat_map(|batch| {
                    batch
                        .operations
                        .iter()
                        .enumerate()
                        .map(|(index, op)| match *op {
                            RuntimeScheduledOp::Rxy {
                                qubit_id,
                                theta,
                                phi,
                            } => QuantumOp::RXY(theta, phi, usize::try_from(qubit_id).unwrap()),
                            RuntimeScheduledOp::Rz { qubit_id, theta } => {
                                QuantumOp::RZ(theta, usize::try_from(qubit_id).unwrap())
                            }
                            RuntimeScheduledOp::Rzz {
                                qubit_id_1,
                                qubit_id_2,
                                theta,
                            } => QuantumOp::RZZ(
                                theta,
                                usize::try_from(qubit_id_1).unwrap(),
                                usize::try_from(qubit_id_2).unwrap(),
                            ),
                            RuntimeScheduledOp::Reset { qubit_id } => {
                                QuantumOp::Reset(usize::try_from(qubit_id).unwrap())
                            }
                            RuntimeScheduledOp::Measure { qubit_id, .. }
                            | RuntimeScheduledOp::MeasureLeaked { qubit_id, .. } => {
                                let measurement = batch
                                    .measurements
                                    .iter()
                                    .find(|m| m.operation_index == index)
                                    .unwrap();
                                if measurement.leakage_aware {
                                    QuantumOp::MeasureLeaked(
                                        usize::try_from(qubit_id).unwrap(),
                                        measurement.program_result,
                                    )
                                } else {
                                    QuantumOp::Measure(
                                        usize::try_from(qubit_id).unwrap(),
                                        measurement.program_result,
                                    )
                                }
                            }
                            _ => panic!("unexpected native output {op:?}"),
                        })
                        .collect::<Vec<_>>()
                })
                .collect(),
        }
    }
}

fn start(mut runtime: SeleneRuntime, qubits: usize) -> SeleneRuntime {
    runtime.set_num_qubits(qubits);
    runtime.shot_start(0, Some(7)).unwrap();
    runtime
}

fn simulate(sim: &mut StateVecAoS, ops: &[QuantumOp]) -> BTreeMap<usize, u32> {
    let mut results = BTreeMap::new();
    for op in ops {
        match *op {
            QuantumOp::Measure(q, r) | QuantumOp::MeasureLeaked(q, r) => {
                results.insert(r, u32::from(sim.mz(&[QubitId(q)])[0].outcome));
            }
            _ => apply(sim, op),
        }
    }
    results
}

#[test]
fn soft_rz_h_rz_h_preserves_virtual_phase_on_every_route() {
    for route in Route::ALL {
        let mut runtime = start(crate::selene_runtimes::selene_soft_rz_runtime().unwrap(), 1);
        let mut sim = StateVecAoS::new(1);
        let preparation = route.lower(
            &mut runtime,
            &[
                Operation::AllocateQubit { id: 0 },
                QuantumOp::Reset(0).into(),
                QuantumOp::Measure(0, 7).into(),
            ],
        );
        runtime
            .provide_measurement_outcomes(simulate(&mut sim, &preparation))
            .unwrap();
        let lowered = route.lower(
            &mut runtime,
            &[
                QuantumOp::H(0).into(),
                QuantumOp::RZ(PI, 0).into(),
                QuantumOp::H(0).into(),
                QuantumOp::Measure(0, 8).into(),
            ],
        );
        assert_eq!(
            simulate(&mut sim, &lowered)[&8],
            1,
            "{route:?}: {lowered:?}"
        );
    }
}

#[test]
fn soft_rz_reset_precedes_decomposed_x_on_every_route() {
    for route in Route::ALL {
        let mut runtime = start(crate::selene_runtimes::selene_soft_rz_runtime().unwrap(), 1);
        let lowered = route.lower(
            &mut runtime,
            &[
                Operation::AllocateQubit { id: 0 },
                QuantumOp::Reset(0).into(),
                QuantumOp::X(0).into(),
                QuantumOp::Measure(0, 0).into(),
            ],
        );
        let reset = lowered
            .iter()
            .position(|op| matches!(op, QuantumOp::Reset(0)))
            .unwrap();
        let pulse = lowered
            .iter()
            .position(|op| matches!(op, QuantumOp::RXY(..)))
            .unwrap();
        assert!(reset < pulse, "{route:?}: {lowered:?}");
        assert_eq!(simulate(&mut StateVecAoS::new(1), &lowered)[&0], 1);
    }
}

fn assert_native_program(route: Route) {
    let mut runtime = start(crate::selene_runtimes::selene_simple_runtime().unwrap(), 3);
    let lowered = route.lower(
        &mut runtime,
        &[
            Operation::AllocateQubit { id: 0 },
            Operation::AllocateQubit { id: 1 },
            Operation::AllocateQubit { id: 2 },
            QuantumOp::Reset(0).into(),
            QuantumOp::H(0).into(),
            QuantumOp::CX(0, 1).into(),
            QuantumOp::CCX(0, 1, 2).into(),
            QuantumOp::Measure(2, 3).into(),
            Operation::Barrier,
        ],
    );
    assert!(lowered.iter().any(|op| matches!(op, QuantumOp::RXY(..))));
    assert!(lowered.iter().any(|op| matches!(op, QuantumOp::RZZ(..))));
    for op in lowered {
        assert!(
            matches!(
                op,
                QuantumOp::RXY(..)
                    | QuantumOp::RZ(..)
                    | QuantumOp::RZZ(..)
                    | QuantumOp::Reset(..)
                    | QuantumOp::Measure(..)
                    | QuantumOp::Idle(..)
            ),
            "{route:?}: {op:?}"
        );
    }
}

#[test]
fn simple_runtime_emits_only_native_gates() {
    assert_native_program(Route::Flat);
    assert_native_program(Route::Metadata);
}

#[test]
fn scheduled_route_accepts_h_cx_and_ccx_as_native_batches() {
    assert_native_program(Route::Scheduled);
}

#[test]
fn soft_rz_measure_leaked_releases_result_before_unrelated_x() {
    for route in Route::ALL {
        let mut runtime = start(crate::selene_runtimes::selene_soft_rz_runtime().unwrap(), 2);
        let lowered = route.lower(
            &mut runtime,
            &[
                Operation::AllocateQubit { id: 0 },
                QuantumOp::Reset(0).into(),
                Operation::AllocateQubit { id: 1 },
                QuantumOp::Reset(1).into(),
                QuantumOp::MeasureLeaked(0, 17).into(),
                QuantumOp::X(1).into(),
            ],
        );
        let results = simulate(&mut StateVecAoS::new(2), &lowered);
        assert_eq!(results.get(&17), Some(&0), "{route:?}: {lowered:?}");
        runtime.provide_measurement_outcomes(results).unwrap();
        assert_eq!(
            runtime.get_classical_state().measurements.get(&17),
            Some(&false)
        );
    }
}

#[test]
fn idle_releases_queued_work_on_its_qubit_before_passthrough() {
    for route in [Route::Flat, Route::Metadata] {
        let mut runtime = start(crate::selene_runtimes::selene_soft_rz_runtime().unwrap(), 1);
        let lowered = route.lower(
            &mut runtime,
            &[
                Operation::AllocateQubit { id: 0 },
                QuantumOp::Reset(0).into(),
                QuantumOp::X(0).into(),
                QuantumOp::Idle(0.001, 0).into(),
            ],
        );
        assert!(
            matches!(lowered.last(), Some(QuantumOp::Idle(0.001, 0))),
            "{lowered:?}"
        );
        assert!(
            lowered.iter().any(|op| matches!(op, QuantumOp::RXY(..))),
            "{lowered:?}"
        );
        assert!(
            lowered.iter().any(|op| matches!(op, QuantumOp::Reset(..))),
            "{lowered:?}"
        );
    }
}

#[test]
fn decomposed_source_metadata_reaches_native_output() {
    for gate in [
        QuantumOp::H(7),
        QuantumOp::CX(7, 3),
        QuantumOp::CCX(7, 3, 9),
    ] {
        let mut runtime = start(crate::selene_runtimes::selene_simple_runtime().unwrap(), 3);
        let metadata = TraceMetadata::from([("source_label".into(), "decomposed".into())]);
        let lowered = runtime
            .lower_operations_with_metadata(&[
                Operation::AllocateQubit { id: 7 },
                Operation::AllocateQubit { id: 3 },
                Operation::AllocateQubit { id: 9 },
                Operation::TraceMetadata {
                    metadata: metadata.clone(),
                    qubit: None,
                },
                gate.into(),
                Operation::Barrier,
            ])
            .unwrap();
        assert_eq!(
            lowered.iter().filter(|op| op.metadata == metadata).count(),
            1
        );
    }
}

#[test]
fn legacy_execution_submits_native_gates_and_flushes_terminal_tail() {
    for mut runtime in [
        start(crate::selene_runtimes::selene_simple_runtime().unwrap(), 1),
        start(crate::selene_runtimes::selene_soft_rz_runtime().unwrap(), 1),
    ] {
        let mut collector = OperationCollector::default();
        collector.operations = vec![
            Operation::AllocateQubit { id: 0 },
            QuantumOp::Reset(0).into(),
            QuantumOp::H(0).into(),
            QuantumOp::RZ(PI, 0).into(),
            QuantumOp::H(0).into(),
            QuantumOp::MeasureLeaked(0, 4).into(),
            QuantumOp::X(0).into(),
        ];
        runtime.load_interface(collector).unwrap();
        runtime.set_batch_size(1);
        let mut sim = StateVecAoS::new(1);
        let mut all = Vec::new();
        let mut results = BTreeMap::new();
        while let Some(ops) = runtime.execute_until_quantum().unwrap() {
            let outcomes = simulate(&mut sim, &ops);
            runtime
                .provide_measurement_outcomes(outcomes.clone())
                .unwrap();
            results.extend(outcomes);
            all.extend(ops);
        }
        assert_eq!(results[&4], 1);
        assert!(sim.probability(0) > 1.0 - 1e-10, "terminal X was lost");
        assert!(
            all.iter()
                .all(|op| !matches!(op, QuantumOp::H(..) | QuantumOp::X(..)))
        );
        assert_eq!(runtime.drain_pending_operations().unwrap(), Vec::new());
    }
}

#[test]
fn decomposed_labels_stay_on_their_anchor_pulses_for_both_runtimes() {
    for (soft, runtime) in [
        (
            false,
            crate::selene_runtimes::selene_simple_runtime().unwrap(),
        ),
        (
            true,
            crate::selene_runtimes::selene_soft_rz_runtime().unwrap(),
        ),
    ] {
        let mut runtime = start(runtime, 2);
        let label = |name: &str| Operation::TraceMetadata {
            metadata: TraceMetadata::from([("source_label".into(), name.into())]),
            qubit: None,
        };
        let lowered = runtime
            .lower_operations_with_metadata(&[
                Operation::AllocateQubit { id: 0 },
                Operation::AllocateQubit { id: 1 },
                QuantumOp::Reset(0).into(),
                QuantumOp::Reset(1).into(),
                QuantumOp::RZ(0.37, 0).into(),
                QuantumOp::H(0).into(),
                label("H"),
                QuantumOp::H(0).into(),
                label("CX"),
                QuantumOp::CX(1, 0).into(),
                label("X-a"),
                QuantumOp::X(0).into(),
                label("X-b"),
                QuantumOp::X(0).into(),
                QuantumOp::Y(0).into(),
                label("X-c"),
                QuantumOp::X(0).into(),
                Operation::Barrier,
            ])
            .unwrap();
        let pulses: Vec<_> = lowered
            .iter()
            .filter(|op| matches!(op.op, QuantumOp::RXY(..)))
            .collect();
        let expected_labels = [
            None,
            Some("H"),
            Some("CX"),
            None,
            Some("X-a"),
            Some("X-b"),
            None,
            Some("X-c"),
        ];
        assert_eq!(pulses.len(), expected_labels.len());
        for (pulse, label) in pulses.iter().zip(expected_labels) {
            assert_eq!(
                pulse.metadata.get("source_label").map(String::as_str),
                label,
                "soft={soft}: {lowered:?}"
            );
        }
        assert_eq!(
            lowered.iter().filter(|op| !op.metadata.is_empty()).count(),
            5
        );
        let half = std::f64::consts::FRAC_PI_2;
        let axes = [-half, -half, half, PI, 0.0, 0.0, half, 0.0];
        let phases = [
            0.37,
            0.37 + PI,
            0.37 + 2.0 * PI,
            0.37 + 2.0 * PI,
            0.37 + 1.5 * PI,
            0.37 + 1.5 * PI,
            0.37 + 1.5 * PI,
            0.37 + 1.5 * PI,
        ];
        for ((pulse, axis), phase) in pulses.iter().zip(axes).zip(phases) {
            let QuantumOp::RXY(_, phi, 0) = pulse.op else {
                panic!("wrong anchor {pulse:?}")
            };
            assert!((phi - (axis - if soft { phase } else { 0.0 })).abs() < 1e-12);
        }
    }
}

#[test]
fn native_metadata_rejects_an_unexplained_pulse_axis() {
    let source = SourceTraceMetadata {
        op: QuantumOp::RXY(PI, 0.0, 0),
        metadata: TraceMetadata::new(),
        native_match: true,
        folded_phi: Some(-0.37),
    };
    assert!(SeleneRuntime::source_trace_metadata_matches_lowered_op(
        &source,
        &QuantumOp::RXY(PI, 0.0, 0)
    ));
    assert!(SeleneRuntime::source_trace_metadata_matches_lowered_op(
        &source,
        &QuantumOp::RXY(PI, -0.37, 0)
    ));
    assert!(!SeleneRuntime::source_trace_metadata_matches_lowered_op(
        &source,
        &QuantumOp::RXY(PI, 0.81, 0)
    ));
}

#[test]
fn unsupported_native_gate_error_names_source_qubits_and_runtime() {
    for route in Route::ALL {
        let mut runtime = start(crate::selene_runtimes::selene_soft_rz_runtime().unwrap(), 2);
        let operations = [QuantumOp::RXYXY2Q(0.25, 0.5, 0, 1).into()];
        let error = match route {
            Route::Flat => runtime.lower_operations(&operations).unwrap_err(),
            Route::Metadata => runtime
                .lower_operations_with_metadata(&operations)
                .unwrap_err(),
            Route::Scheduled => runtime.lower_scheduled_operations(&operations).unwrap_err(),
        }
        .to_string();
        assert!(error.contains("RXYXY2Q(0.25, 0.5, 0, 1)"), "{error}");
        assert!(error.contains("qubits {0, 1}"), "{error}");
        assert!(error.contains(&runtime.plugin_path), "{error}");
        assert!(error.contains("rpp failed with errno"), "{error}");
        assert!(runtime.shot_end().is_err());
    }
}

#[test]
fn scheduled_idle_error_explains_missing_native_entry_point() {
    let mut runtime = start(crate::selene_runtimes::selene_simple_runtime().unwrap(), 1);
    let error = runtime
        .lower_scheduled_operations(&[QuantumOp::Idle(0.25, 0).into()])
        .unwrap_err()
        .to_string();
    assert!(error.contains("Idle(0.25, 0)"), "{error}");
    assert!(
        error.contains("scheduled extraction has no native idle entry point"),
        "{error}"
    );
    assert!(error.contains(&runtime.plugin_path), "{error}");
}

#[test]
fn emitted_custom_events_require_default_acknowledgement_on_every_route() {
    use std::sync::atomic::{AtomicBool, Ordering};
    static EMITTED: AtomicBool = AtomicBool::new(false);
    unsafe extern "C" fn emit_custom(
        _: RuntimeInstance,
        output: SeleneRuntimeGetOperationHandle,
    ) -> i32 {
        if !EMITTED.swap(true, Ordering::SeqCst) {
            let bytes = [17_u8, 29];
            unsafe {
                (output.interface.rz)(output.instance, 0, 0.2);
                (output.interface.custom)(
                    output.instance,
                    7301,
                    bytes.as_ptr().cast(),
                    bytes.len(),
                );
            }
        }
        0
    }
    let executable = std::env::current_exe().unwrap();
    let source = crate::selene_runtimes::find_library_in_dir(
        executable.parent().unwrap(),
        pecos_qis_test_runtime::LIBRARY_NAME,
    )
    .unwrap();
    // Isolate the fixture's descriptor slot from other parallel descriptor tests.
    let directory = tempfile::tempdir().unwrap();
    let plugin = directory.path().join(source.file_name().unwrap());
    std::fs::copy(&source, &plugin).unwrap();
    let public = crate::selene_runtimes::selene_simple_runtime().unwrap();
    // Both libraries and the boxed descriptor outlive all native instances.
    let library = unsafe { libloading::Library::new(&public.plugin_path).unwrap() };
    let mut descriptor =
        Box::new(unsafe { SeleneRuntime::runtime_plugin_descriptor(&library).unwrap() });
    // The same repr(C) callback-handle conversion used by drain_runtime_operations.
    descriptor.get_next_operations_fn = unsafe {
        std::mem::transmute::<
            unsafe extern "C" fn(RuntimeInstance, SeleneRuntimeGetOperationHandle) -> i32,
            unsafe extern "C" fn(
                RuntimeInstance,
                selene_core::operation::plugin::RuntimeGetOperationHandle,
            ) -> i32,
        >(emit_custom)
    };
    let fixture = unsafe { libloading::Library::new(&plugin).unwrap() };
    unsafe {
        let set = fixture
            .get::<unsafe extern "C" fn(*mut c_void)>(b"set_descriptor")
            .unwrap();
        set((&raw mut *descriptor).cast());
    }
    for route in Route::ALL {
        for policy in ["default", "capture", "metadata"] {
            EMITTED.store(false, Ordering::SeqCst);
            let mut runtime = SeleneRuntime::new(&plugin);
            runtime.init_args.clone_from(&public.init_args);
            let mut runtime = start(runtime, 1);
            match policy {
                "capture" => runtime.set_custom_event_policy(RuntimeCustomEventPolicy::Capture),
                "metadata" => runtime.set_custom_event_handler(|event| {
                    assert_eq!(event.tag, 7301);
                    assert_eq!(event.data, [17, 29]);
                    Ok(RuntimeCustomEventDisposition::MetadataOnly)
                }),
                _ => {}
            }
            let operations = [Operation::AllocateQubit { id: 0 }];
            let result = match route {
                Route::Flat => runtime.lower_operations(&operations).map(|_| ()),
                Route::Metadata => runtime.lower_operations_with_metadata(&operations).map(|_| ()),
                Route::Scheduled => runtime.lower_scheduled_operations(&operations).map(|batches| {
                    assert!(matches!(&batches[0].operations[1], RuntimeScheduledOp::Custom { tag: 7301, data } if data == &[17, 29]));
                }),
            };
            if policy == "default" {
                let error = result.unwrap_err().to_string();
                assert!(
                    error.contains("tag 7301 at batch 0, operation 1"),
                    "{error}"
                );
                assert!(error.contains(&runtime.plugin_path), "{error}");
                assert!(runtime.shot_end().is_err());
            } else {
                result.unwrap();
                if !matches!(route, Route::Scheduled) {
                    assert_eq!(runtime.custom_events()[0].tag, 7301);
                    assert_eq!(runtime.custom_events()[0].data, [17, 29]);
                }
            }
            runtime.reset().unwrap();
        }
    }
}

#[test]
fn queued_metadata_records_survive_calls_and_flat_transitions() {
    for soft in [false, true] {
        for flat_prefix in [false, true] {
            let mut runtime = start(
                if soft {
                    crate::selene_runtimes::selene_soft_rz_runtime().unwrap()
                } else {
                    crate::selene_runtimes::selene_simple_runtime().unwrap()
                },
                1,
            );
            let prefix = [Operation::AllocateQubit { id: 0 }, QuantumOp::H(0).into()];
            if flat_prefix {
                runtime.lower_operations(&prefix).unwrap();
            } else {
                runtime.lower_operations_with_metadata(&prefix).unwrap();
            }
            let label = |name: &str| Operation::TraceMetadata {
                metadata: TraceMetadata::from([("source_label".into(), name.into())]),
                qubit: None,
            };
            let h = runtime
                .lower_operations_with_metadata(&[
                    label("H"),
                    QuantumOp::H(0).into(),
                    Operation::Barrier,
                ])
                .unwrap();
            let pulses: Vec<_> = h
                .iter()
                .filter(|op| matches!(op.op, QuantumOp::RXY(..)))
                .collect();
            assert_eq!(pulses.len(), if soft { 2 } else { 1 });
            assert_eq!(
                pulses
                    .last()
                    .unwrap()
                    .metadata
                    .get("source_label")
                    .map(String::as_str),
                Some("H")
            );
            if soft {
                assert_eq!(pulses[0].metadata, TraceMetadata::new());
            }
            let mut x = runtime
                .lower_operations_with_metadata(&[label("first X"), QuantumOp::X(0).into()])
                .unwrap();
            x.extend(
                runtime
                    .lower_operations_with_metadata(&[
                        label("second X"),
                        QuantumOp::X(0).into(),
                        Operation::Barrier,
                    ])
                    .unwrap(),
            );
            let labels: Vec<_> = x
                .iter()
                .filter_map(|op| op.metadata.get("source_label").map(String::as_str))
                .collect();
            assert_eq!(labels, ["first X", "second X"]);
        }
    }
}
