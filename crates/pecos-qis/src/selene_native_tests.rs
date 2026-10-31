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

    // These gate/feedback tests explicitly read every measurement they submit.
    fn lower_and_read(self, runtime: &mut SeleneRuntime, ops: &[Operation]) -> Vec<QuantumOp> {
        let mut output = self.lower_raw(runtime, ops);
        for op in ops {
            if let Operation::Quantum(
                QuantumOp::Measure(_, result) | QuantumOp::MeasureLeaked(_, result),
            ) = op
            {
                if runtime.has_pending_feedback() {
                    break;
                }
                runtime.force_result(*result).unwrap();
                output.extend(self.lower_raw(runtime, &[]));
            }
        }
        output
    }

    fn lower_raw(self, runtime: &mut SeleneRuntime, ops: &[Operation]) -> Vec<QuantumOp> {
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
                            RuntimeScheduledOp::Rpp {
                                qubit_id_1,
                                qubit_id_2,
                                theta,
                                phi,
                            } => QuantumOp::RXYXY2Q(
                                theta,
                                phi,
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
                            RuntimeScheduledOp::Custom { .. } => {
                                panic!("unexpected native output {op:?}")
                            }
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

fn simulate(sim: &mut StateVecAoS, ops: &[QuantumOp]) -> Vec<(usize, u32)> {
    let mut results = Vec::new();
    for op in ops {
        match *op {
            QuantumOp::Measure(q, r) | QuantumOp::MeasureLeaked(q, r) => {
                results.push((r, u32::from(sim.mz(&[QubitId(q)])[0].outcome)));
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
        let preparation = route.lower_and_read(
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
        let lowered = route.lower_and_read(
            &mut runtime,
            &[
                QuantumOp::H(0).into(),
                QuantumOp::RZ(PI, 0).into(),
                QuantumOp::H(0).into(),
                QuantumOp::Measure(0, 8).into(),
            ],
        );
        assert_eq!(
            simulate(&mut sim, &lowered)
                .into_iter()
                .collect::<BTreeMap<_, _>>()[&8],
            1,
            "{route:?}: {lowered:?}"
        );
    }
}

#[test]
fn soft_rz_reset_precedes_decomposed_x_on_every_route() {
    for route in Route::ALL {
        let mut runtime = start(crate::selene_runtimes::selene_soft_rz_runtime().unwrap(), 1);
        let lowered = route.lower_and_read(
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
        assert_eq!(
            simulate(&mut StateVecAoS::new(1), &lowered)
                .into_iter()
                .collect::<BTreeMap<_, _>>()[&0],
            1
        );
    }
}

fn assert_native_program(route: Route) {
    let mut runtime = start(crate::selene_runtimes::selene_simple_runtime().unwrap(), 3);
    let lowered = route.lower_and_read(
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
fn soft_rz_unsupported_leaked_force_does_not_flush_a_reused_qubit() {
    // The pinned plugin only recognizes ordinary measurements in force_result.
    // A compatibility barrier would flush through the later lifetime instead.
    for route in Route::ALL {
        let mut runtime = start(crate::selene_runtimes::selene_soft_rz_runtime().unwrap(), 1);
        let lowered = route.lower_raw(
            &mut runtime,
            &[
                Operation::AllocateQubit { id: 7 },
                QuantumOp::Reset(7).into(),
                QuantumOp::MeasureLeaked(7, 31).into(),
                Operation::ReleaseQubit { id: 7 },
                Operation::AllocateQubit { id: 9 },
                QuantumOp::Reset(9).into(),
                QuantumOp::X(9).into(),
                QuantumOp::Measure(9, 47).into(),
            ],
        );
        assert!(lowered.is_empty(), "{route:?}: {lowered:?}");
        runtime.force_result(31).unwrap();
        let lowered = route.lower_raw(&mut runtime, &[]);
        assert!(lowered.is_empty(), "{route:?}: {lowered:?}");
    }
}

#[test]
fn idle_delegates_to_local_barrier_without_inventing_a_physical_delay() {
    for route in [Route::Flat, Route::Metadata] {
        let mut runtime = start(crate::selene_runtimes::selene_soft_rz_runtime().unwrap(), 1);
        let lowered = route.lower_and_read(
            &mut runtime,
            &[
                Operation::AllocateQubit { id: 0 },
                QuantumOp::Reset(0).into(),
                QuantumOp::X(0).into(),
                QuantumOp::Idle(0.001, 0).into(),
            ],
        );
        assert!(
            !lowered
                .iter()
                .any(|op| matches!(op, QuantumOp::Idle(0.001, 0))),
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
fn reduced_native_set_rejects_before_any_plugin_submission() {
    for route in Route::ALL {
        let mut runtime = start(
            SeleneRuntime::new("limited-runtime.so").with_native_gate_set(
                RuntimeNativeGateSet::new([RuntimeNativeGate::Rxy, RuntimeNativeGate::Rz]),
            ),
            2,
        );
        let operations = [
            Operation::AllocateQubit { id: 0 },
            QuantumOp::H(0).into(),
            QuantumOp::CX(0, 1).into(),
        ];
        let error = match route {
            Route::Flat => runtime.lower_operations(&operations).unwrap_err(),
            Route::Metadata => runtime
                .lower_operations_with_metadata(&operations)
                .unwrap_err(),
            Route::Scheduled => runtime.lower_scheduled_operations(&operations).unwrap_err(),
        }
        .to_string();
        for field in [
            "CX(0, 1)",
            "qubits {0, 1}",
            "limited-runtime.so",
            "native set {rxy, rz}",
            "needs missing entry point(s) {rzz}",
            "with_native_gate_set",
        ] {
            assert!(error.contains(field), "{error}");
        }
        assert!(runtime.library.is_none());
        assert!(runtime.instance.is_none());
        assert!(runtime.program_to_runtime_qubits.is_empty());
    }
}

#[test]
fn scheduled_idle_delegates_to_runtime_barrier() {
    let mut runtime = start(crate::selene_runtimes::selene_simple_runtime().unwrap(), 1);
    assert_eq!(
        runtime
            .lower_scheduled_operations(&[QuantumOp::Idle(0.25, 0).into()])
            .unwrap(),
        []
    );
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
                assert!(error.contains("set_custom_event_handler"), "{error}");
                assert!(
                    error.contains("RuntimeCustomEventPolicy::Capture"),
                    "{error}"
                );
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

#[test]
fn soft_rz_outstanding_metadata_is_bounded_across_h_chunks() {
    for labelled in [false, true] {
        let mut runtime = start(crate::selene_runtimes::selene_soft_rz_runtime().unwrap(), 1);
        runtime
            .lower_operations(&[Operation::AllocateQubit { id: 0 }])
            .unwrap();
        for chunk in 0..160 {
            let mut ops = Vec::new();
            if labelled {
                // An absorbed, optional anchor must retire at the next pulse,
                // even when that pulse also has its own label.
                ops.extend([
                    Operation::TraceMetadata {
                        metadata: TraceMetadata::from([("source_label".into(), "Z-only".into())]),
                        qubit: None,
                    },
                    QuantumOp::S(0).into(),
                ]);
            }
            for _ in 0..50 {
                if labelled {
                    ops.push(Operation::TraceMetadata {
                        metadata: TraceMetadata::from([("source_label".into(), "H".into())]),
                        qubit: None,
                    });
                }
                ops.push(QuantumOp::H(0).into());
            }
            ops.push(Operation::Barrier);
            if labelled {
                let lowered = runtime.lower_operations_with_metadata(&ops).unwrap();
                assert_eq!(
                    lowered.iter().filter(|op| !op.metadata.is_empty()).count(),
                    50
                );
            } else {
                runtime.lower_operations(&ops).unwrap();
                assert!(
                    runtime
                        .source_trace_metadata
                        .iter()
                        .all(|record| matches!(record.op, QuantumOp::RZ(..)))
                );
            }
            assert!(
                runtime.source_trace_metadata.len() <= 2,
                "chunk {chunk}: {} outstanding records",
                runtime.source_trace_metadata.len()
            );
        }
        if !labelled {
            runtime.lower_operations(&[QuantumOp::H(0).into()]).unwrap();
            assert!(
                !runtime.source_trace_metadata.is_empty(),
                "unlabelled queued work must guard against later labels"
            );
        }
    }
}

#[test]
fn z_only_metadata_has_no_anchor_when_virtual_z_is_absorbed() {
    for gate in [
        QuantumOp::Z(0),
        QuantumOp::S(0),
        QuantumOp::Sdg(0),
        QuantumOp::T(0),
        QuantumOp::Tdg(0),
        QuantumOp::RZ(0.0, 0),
        QuantumOp::RZ(-0.73, 0),
    ] {
        for soft in [false, true] {
            for required in [false, true] {
                let mut runtime = start(
                    if soft {
                        crate::selene_runtimes::selene_soft_rz_runtime().unwrap()
                    } else {
                        crate::selene_runtimes::selene_simple_runtime().unwrap()
                    },
                    1,
                );
                let output = runtime.lower_operations_with_metadata(&[
                    Operation::AllocateQubit { id: 0 },
                    Operation::TraceMetadata {
                        metadata: TraceMetadata::from([
                            ("source_label".into(), "Z-only".into()),
                            ("source_lowering_required".into(), required.to_string()),
                        ]),
                        qubit: None,
                    },
                    gate.clone().into(),
                    Operation::TraceMetadata {
                        metadata: TraceMetadata::from([("source_label".into(), "next X".into())]),
                        qubit: None,
                    },
                    QuantumOp::X(0).into(),
                    Operation::Barrier,
                ]);
                if soft && required {
                    let error = output.unwrap_err().to_string();
                    assert!(
                        error.contains("Z-only") && error.contains("did not emit"),
                        "{error}"
                    );
                } else {
                    let output = output.unwrap();
                    let labelled: Vec<_> =
                        output.iter().filter(|op| !op.metadata.is_empty()).collect();
                    assert_eq!(labelled.len(), if soft { 1 } else { 2 }, "{gate:?}");
                    if !soft {
                        assert!(matches!(labelled[0].op, QuantumOp::RZ(_, 0)));
                        assert_eq!(labelled[0].metadata["source_label"], "Z-only");
                    }
                    let pulse = labelled.last().unwrap();
                    assert!(matches!(pulse.op, QuantumOp::RXY(_, _, 0)));
                    assert_eq!(pulse.metadata["source_label"], "next X");
                    assert!(runtime.source_trace_metadata.is_empty());
                }
            }
        }
    }
}

#[test]
fn rpp_uses_each_runtime_native_set_on_every_route() {
    for route in Route::ALL {
        for soft in [false, true] {
            let runtime = if soft {
                crate::selene_runtimes::selene_soft_rz_runtime().unwrap()
            } else {
                crate::selene_runtimes::selene_simple_runtime().unwrap()
            };
            let mut runtime = start(runtime, 2);
            let output = route.lower_and_read(
                &mut runtime,
                &[
                    Operation::AllocateQubit { id: 0 },
                    Operation::AllocateQubit { id: 1 },
                    QuantumOp::Reset(0).into(),
                    QuantumOp::Reset(1).into(),
                    QuantumOp::RXYXY2Q(PI, -0.61, 0, 1).into(),
                    QuantumOp::Measure(0, 0).into(),
                    QuantumOp::Measure(1, 1).into(),
                    Operation::Barrier,
                ],
            );
            assert_eq!(
                output
                    .iter()
                    .filter(|op| matches!(op, QuantumOp::RXYXY2Q(..)))
                    .count(),
                usize::from(!soft)
            );
            if soft {
                assert_eq!(
                    output
                        .iter()
                        .filter(|op| matches!(op, QuantumOp::RXY(..)))
                        .count(),
                    4
                );
                assert_eq!(
                    output
                        .iter()
                        .filter(|op| matches!(op, QuantumOp::RZZ(..)))
                        .count(),
                    1
                );
            }
            let mut sim = StateVecAoS::new(2);
            let mut results = simulate(&mut sim, &output);
            runtime
                .provide_measurement_outcomes(results.clone())
                .unwrap();
            let mut tail = route.lower_raw(&mut runtime, &[]);
            if !runtime.has_pending_feedback() {
                runtime.force_result(1).unwrap();
                tail.extend(route.lower_raw(&mut runtime, &[]));
            }
            results.extend(simulate(&mut sim, &tail));
            assert_eq!(results, [(0, 1), (1, 1)]);
        }
    }
}

#[test]
fn native_set_identity_survives_named_construction_clone_and_reset() {
    assert_eq!(
        SeleneRuntime::new("custom.so").native_gate_set(),
        RuntimeNativeGateSet::ALL
    );
    for (name, expected) in [
        ("selene_simple_runtime", RuntimeNativeGateSet::ALL),
        ("selene_soft_rz_runtime", RuntimeNativeGateSet::RXY_RZ_RZZ),
    ] {
        let mut runtime = crate::selene_runtimes::selene_runtime_auto(name)
            .unwrap()
            .clone();
        assert_eq!(runtime.native_gate_set(), expected);
        runtime.reset().unwrap();
        assert_eq!(runtime.native_gate_set(), expected);
    }
    let set = RuntimeNativeGateSet::new([RuntimeNativeGate::Rpp]);
    let runtime =
        SeleneRuntime::with_plugin_config("custom.so", vec![], vec![]).with_native_gate_set(set);
    assert_eq!(runtime.clone().native_gate_set(), set);
}

#[test]
fn scheduled_release_polls_between_native_submissions_without_barriers() {
    use selene_core::operation::plugin::RuntimeGetOperationHandle;
    use std::cell::{Cell, RefCell};

    thread_local! {
        static ORIGINAL: Cell<Option<RuntimePluginDescriptorV1>> = const { Cell::new(None) };
        static CALLS: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
    }
    unsafe extern "C" fn extract(
        instance: RuntimeInstance,
        output: SeleneRuntimeGetOperationHandle,
    ) -> i32 {
        CALLS.with_borrow_mut(|calls| calls.push("extract"));
        let original = ORIGINAL.get().unwrap();
        // Use the same repr(C) handle conversion as production extraction.
        let forward = unsafe {
            std::mem::transmute::<
                unsafe extern "C" fn(RuntimeInstance, RuntimeGetOperationHandle) -> i32,
                unsafe extern "C" fn(RuntimeInstance, SeleneRuntimeGetOperationHandle) -> i32,
            >(original.get_next_operations_fn)
        };
        unsafe { forward(instance, output) }
    }
    unsafe extern "C" fn reset(instance: RuntimeInstance, qubit: u64) -> i32 {
        CALLS.with_borrow_mut(|calls| calls.push("reset"));
        let original = ORIGINAL.get().unwrap();
        unsafe { (original.reset_fn)(instance, qubit) }
    }
    unsafe extern "C" fn release(instance: RuntimeInstance, qubit: u64) -> i32 {
        CALLS.with_borrow_mut(|calls| calls.push("release"));
        let original = ORIGINAL.get().unwrap();
        unsafe { (original.qfree_fn)(instance, qubit) }
    }
    unsafe extern "C" fn barrier(
        instance: RuntimeInstance,
        qubits: *const u64,
        count: u64,
        delay: u64,
    ) -> i32 {
        CALLS.with_borrow_mut(|calls| calls.push("barrier"));
        let original = ORIGINAL.get().unwrap();
        unsafe { (original.local_barrier_fn)(instance, qubits, count, delay) }
    }
    unsafe extern "C" fn global_barrier(instance: RuntimeInstance, delay: u64) -> i32 {
        CALLS.with_borrow_mut(|calls| calls.push("global_barrier"));
        let original = ORIGINAL.get().unwrap();
        unsafe { (original.global_barrier_fn)(instance, delay) }
    }
    let executable = std::env::current_exe().unwrap();
    let source = crate::selene_runtimes::find_library_in_dir(
        executable.parent().unwrap(),
        pecos_qis_test_runtime::LIBRARY_NAME,
    )
    .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let plugin = directory.path().join(source.file_name().unwrap());
    std::fs::copy(source, &plugin).unwrap();
    let public = crate::selene_runtimes::selene_soft_rz_runtime().unwrap();
    // The libraries and descriptor outlive the runtime and all forwarding callbacks.
    let library = unsafe { libloading::Library::new(&public.plugin_path).unwrap() };
    let mut descriptor =
        Box::new(unsafe { SeleneRuntime::runtime_plugin_descriptor(&library).unwrap() });
    ORIGINAL.set(Some(*descriptor));
    descriptor.get_next_operations_fn = unsafe {
        std::mem::transmute::<
            unsafe extern "C" fn(RuntimeInstance, SeleneRuntimeGetOperationHandle) -> i32,
            unsafe extern "C" fn(RuntimeInstance, RuntimeGetOperationHandle) -> i32,
        >(extract)
    };
    descriptor.reset_fn = reset;
    descriptor.qfree_fn = release;
    descriptor.local_barrier_fn = barrier;
    descriptor.global_barrier_fn = global_barrier;
    let fixture = unsafe { libloading::Library::new(&plugin).unwrap() };
    unsafe {
        let set = fixture
            .get::<unsafe extern "C" fn(*mut c_void)>(b"set_descriptor")
            .unwrap();
        set((&raw mut *descriptor).cast());
    }
    let mut runtime = SeleneRuntime::new(&plugin);
    runtime.init_args.clone_from(&public.init_args);
    let mut runtime = start(runtime, 1);
    CALLS.with_borrow_mut(Vec::clear);
    let mut batches = runtime
        .lower_scheduled_operations(&[
            Operation::AllocateQubit { id: 7 },
            QuantumOp::Reset(7).into(),
            QuantumOp::RXY(0.25, 0.0, 7).into(),
            QuantumOp::Measure(7, 31).into(),
            Operation::ReleaseQubit { id: 7 },
            Operation::AllocateQubit { id: 9 },
            QuantumOp::Reset(9).into(),
            QuantumOp::Measure(9, 47).into(),
        ])
        .unwrap();
    let calls = CALLS.with_borrow(Clone::clone);
    let submissions: Vec<_> = calls
        .iter()
        .copied()
        .filter(|call| *call != "extract")
        .collect();
    assert_eq!(submissions, ["reset", "release", "reset"], "{calls:?}");
    for (index, call) in calls.iter().enumerate() {
        if *call != "extract" {
            assert_eq!(calls.get(index + 1), Some(&"extract"), "{calls:?}");
        }
    }
    assert!(batches.is_empty(), "submission must remain lazy");
    runtime.force_result(47).unwrap();
    batches.extend(runtime.lower_scheduled_operations(&[]).unwrap());
    assert_eq!(batches.iter().flat_map(|b| &b.measurements).count(), 1);
    runtime.provide_measurement_outcomes(vec![(31, 0)]).unwrap();
    batches.extend(runtime.lower_scheduled_operations(&[]).unwrap());
    let results: Vec<_> = batches
        .iter()
        .flat_map(|batch| &batch.measurements)
        .map(|m| m.program_result)
        .collect();
    assert_eq!(results, [31, 47]);
    let gates: Vec<_> = batches.iter().flat_map(|batch| &batch.operations).collect();
    assert_eq!(
        gates.len(),
        5,
        "both lifetimes' work must survive release and native-slot reuse"
    );
    assert!(matches!(
        gates[0],
        RuntimeScheduledOp::Reset { qubit_id: 0 }
    ));
    assert!(matches!(
        gates[1],
        RuntimeScheduledOp::Rxy { qubit_id: 0, .. }
    ));
    assert!(matches!(
        gates[2],
        RuntimeScheduledOp::Measure { qubit_id: 0, .. }
    ));
    assert!(matches!(
        gates[3],
        RuntimeScheduledOp::Reset { qubit_id: 0 }
    ));
    assert!(matches!(
        gates[4],
        RuntimeScheduledOp::Measure { qubit_id: 0, .. }
    ));
    runtime
        .provide_measurement_outcomes(Vec::from([(47, 1)]))
        .unwrap();
    runtime.drain_pending_scheduled_operations().unwrap();
    runtime.shot_end().unwrap();
}

mod measurement_followups {
    use super::*;
    use std::cell::{Cell, RefCell};

    thread_local! {
        static FORCED: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
        static FORCE_RELEASES: Cell<bool> = const { Cell::new(false) };
        static LOCAL_BARRIERS: Cell<usize> = const { Cell::new(0) };
        static SUBMITTED: RefCell<Vec<(u64, bool)>> = const { RefCell::new(Vec::new()) };
        static DELIVERED: RefCell<Vec<(u64, bool, u64)>> = const { RefCell::new(Vec::new()) };
        static EMITTED: Cell<usize> = const { Cell::new(0) };
        static RELEASE: Cell<usize> = const { Cell::new(2) };
        static REVERSE: Cell<bool> = const { Cell::new(false) };
        static FAIL_SETTER: Cell<bool> = const { Cell::new(false) };
        static PAD: Cell<usize> = const { Cell::new(0) };
        static REQUIRE_FEEDBACK: Cell<bool> = const { Cell::new(false) };
        static MIN_SUBMITTED: Cell<usize> = const { Cell::new(2) };
        static GETTER_VALUE: Cell<Option<u64>> = const { Cell::new(None) };
        static FORCE_VALUE: Cell<Option<u64>> = const { Cell::new(None) };
        static FAIL_GETTER: Cell<bool> = const { Cell::new(false) };
        static INVERT_FEEDBACK: Cell<bool> = const { Cell::new(false) };
        static END_RELEASES: Cell<bool> = const { Cell::new(false) };
        static END_CALLS: Cell<usize> = const { Cell::new(0) };
        static END_TAIL_EMITTED: Cell<bool> = const { Cell::new(false) };
        static PULSE_RELEASES: Cell<bool> = const { Cell::new(false) };
        static NATIVE_CALLS: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
        static REFCOUNTS: RefCell<Vec<(bool, u64)>> = const { RefCell::new(Vec::new()) };
        static FAIL_REFCOUNT: Cell<bool> = const { Cell::new(false) };

        static ORIGINAL: Cell<Option<RuntimePluginDescriptorV1>> = const { Cell::new(None) };
    }
    fn allocate(leaked: bool, output: *mut u64) -> i32 {
        if REQUIRE_FEEDBACK.get() && EMITTED.get() > DELIVERED.with_borrow(Vec::len) {
            return 78;
        }
        SUBMITTED.with_borrow_mut(|submitted| {
            let id = 100 + submitted.len() as u64;
            submitted.push((id, leaked));
            unsafe { *output = id };
        });
        0
    }
    unsafe extern "C" fn measure(_: RuntimeInstance, _: u64, out: *mut u64) -> i32 {
        allocate(false, out)
    }
    unsafe extern "C" fn measure_leaked(_: RuntimeInstance, _: u64, out: *mut u64) -> i32 {
        allocate(true, out)
    }
    unsafe extern "C" fn force(_: RuntimeInstance, result: u64) -> i32 {
        FORCED.with_borrow_mut(|calls| calls.push(result));
        if let Some(value) = FORCE_VALUE.get() {
            GETTER_VALUE.set(Some(value));
        }
        if FORCE_RELEASES.get() {
            RELEASE.set(
                RELEASE
                    .get()
                    .max(usize::try_from(result - 100 + 1).unwrap()),
            );
        }
        0
    }
    unsafe extern "C" fn local(_: RuntimeInstance, _: *const u64, _: u64, _: u64) -> i32 {
        LOCAL_BARRIERS.set(LOCAL_BARRIERS.get() + 1);
        0
    }
    unsafe extern "C" fn set_bool(_: RuntimeInstance, id: u64, value: bool) -> i32 {
        DELIVERED.with_borrow_mut(|calls| calls.push((id, false, u64::from(value))));
        if INVERT_FEEDBACK.get() {
            GETTER_VALUE.set(Some(u64::from(!value)));
        }
        i32::from(FAIL_SETTER.get())
    }
    unsafe extern "C" fn set_u64(_: RuntimeInstance, id: u64, value: u64) -> i32 {
        DELIVERED.with_borrow_mut(|calls| calls.push((id, true, value)));
        i32::from(FAIL_SETTER.get())
    }
    unsafe extern "C" fn extract(_: RuntimeInstance, out: SeleneRuntimeGetOperationHandle) -> i32 {
        if REQUIRE_FEEDBACK.get() && EMITTED.get() > DELIVERED.with_borrow(Vec::len) {
            return 77;
        }
        if PAD.get() > 0 {
            PAD.set(PAD.get() - 1);
            unsafe { (out.interface.set_batch_time)(out.instance, 0, 0) };
            return 0;
        }
        if END_RELEASES.get()
            && END_CALLS.get() == 1
            && DELIVERED.with_borrow(Vec::len) == 2
            && !END_TAIL_EMITTED.replace(true)
        {
            unsafe {
                (out.interface.set_batch_time)(out.instance, 2, 0);
                (out.interface.rxy)(out.instance, 0, PI, 0.0);
            }
            return 0;
        }
        let emitted = EMITTED.get();
        let next = SUBMITTED.with_borrow(|submitted| {
            if submitted.len() < MIN_SUBMITTED.get()
                || emitted >= RELEASE.get()
                || emitted >= submitted.len()
            {
                return None;
            }
            Some(submitted[if REVERSE.get() { 1 - emitted } else { emitted }])
        });
        if let Some((id, _)) = next {
            // Emit ordinary Measure even for leaked source futures, as supported
            // by the runtime ABI. Kind must follow this native ID, not its slot.
            unsafe {
                (out.interface.set_batch_time)(out.instance, emitted as u64, 0);
                (out.interface.measure)(out.instance, 0, id);
            }
            EMITTED.set(emitted + 1);
        }
        0
    }
    unsafe extern "C" fn get_bool(_: RuntimeInstance, _: u64, out: *mut i8) -> i32 {
        unsafe {
            *out = GETTER_VALUE.get().map_or(-1, |v| i8::from(v != 0));
        }
        i32::from(FAIL_GETTER.get())
    }
    unsafe extern "C" fn get_u64(_: RuntimeInstance, _: u64, out: *mut u64) -> i32 {
        unsafe {
            *out = GETTER_VALUE.get().unwrap_or(u64::MAX);
        }
        i32::from(FAIL_GETTER.get())
    }
    unsafe extern "C" fn end(instance: RuntimeInstance) -> i32 {
        END_CALLS.set(END_CALLS.get() + 1);
        if END_RELEASES.get() {
            RELEASE.set(2);
            0
        } else {
            unsafe { (ORIGINAL.get().unwrap().shot_end_fn)(instance) }
        }
    }
    unsafe extern "C" fn rxy(instance: RuntimeInstance, q: u64, theta: f64, phi: f64) -> i32 {
        if REQUIRE_FEEDBACK.get() && EMITTED.get() > DELIVERED.with_borrow(Vec::len) {
            return 79;
        }
        NATIVE_CALLS.with_borrow_mut(|v| v.push("rxy"));
        if PULSE_RELEASES.get() {
            RELEASE.set(1);
        }
        unsafe { (ORIGINAL.get().unwrap().rxy_gate_fn)(instance, q, theta, phi) }
    }
    unsafe extern "C" fn rz(instance: RuntimeInstance, q: u64, theta: f64) -> i32 {
        if REQUIRE_FEEDBACK.get() && EMITTED.get() > DELIVERED.with_borrow(Vec::len) {
            return 80;
        }
        NATIVE_CALLS.with_borrow_mut(|v| v.push("rz"));
        unsafe { (ORIGINAL.get().unwrap().rz_gate_fn)(instance, q, theta) }
    }

    #[test]
    fn submission_waits_for_feedback_before_the_next_native_call() {
        for route in Route::ALL {
            with_descriptor(|mut runtime| {
                MIN_SUBMITTED.set(1);
                REQUIRE_FEEDBACK.set(true);
                let first = route.lower_raw(
                    &mut runtime,
                    &[
                        QuantumOp::Measure(0, 7).into(),
                        QuantumOp::Measure(0, 8).into(),
                    ],
                );
                assert_eq!(first, [QuantumOp::Measure(0, 7)]);
                assert_eq!(SUBMITTED.with_borrow(Vec::len), 1);
                runtime.provide_measurement_outcomes(vec![(7, 0)]).unwrap();
                assert!(
                    route
                        .lower_raw(&mut runtime, &[])
                        .contains(&QuantumOp::Measure(0, 8))
                );
                assert_eq!(SUBMITTED.with_borrow(Vec::len), 2);
            });
        }
    }

    #[test]
    fn feedback_can_interrupt_a_decomposed_gate() {
        for route in Route::ALL {
            with_descriptor(|mut runtime| {
                MIN_SUBMITTED.set(1);
                REQUIRE_FEEDBACK.set(true);
                PULSE_RELEASES.set(true);
                RELEASE.set(0);
                let first = route.lower_raw(
                    &mut runtime,
                    &[QuantumOp::Measure(0, 7).into(), QuantumOp::H(0).into()],
                );
                assert_eq!(first, [QuantumOp::Measure(0, 7)]);
                assert_eq!(NATIVE_CALLS.with_borrow(Clone::clone), ["rxy"]);
                runtime.provide_measurement_outcomes(vec![(7, 0)]).unwrap();
                route.lower_raw(&mut runtime, &[]);
                assert_eq!(NATIVE_CALLS.with_borrow(Clone::clone), ["rxy", "rz"]);
            });
        }
    }

    #[test]
    fn end_feedback_finishes_before_instance_reuse_on_every_route() {
        for route in Route::ALL {
            with_descriptor(|mut runtime| {
                MIN_SUBMITTED.set(1);
                REQUIRE_FEEDBACK.set(true);
                RELEASE.set(0);
                END_RELEASES.set(true);
                assert_eq!(
                    route.lower_raw(
                        &mut runtime,
                        &[
                            QuantumOp::Measure(0, 7).into(),
                            QuantumOp::Measure(0, 8).into(),
                        ]
                    ),
                    Vec::<QuantumOp>::new()
                );
                match route {
                    Route::Flat | Route::Metadata => assert_eq!(
                        runtime.finish_operations().unwrap(),
                        [LoweredQuantumOp::from(QuantumOp::Measure(0, 7))],
                    ),
                    Route::Scheduled => assert_eq!(
                        runtime
                            .finish_scheduled_operations()
                            .unwrap()
                            .iter()
                            .flat_map(|batch| &batch.measurements)
                            .map(|m| m.program_result)
                            .collect::<Vec<_>>(),
                        [7],
                    ),
                }
                let instance = runtime.instance;
                assert!(runtime.shot_start(1, None).is_err());
                runtime.provide_measurement_outcomes(vec![(7, 0)]).unwrap();
                assert!(
                    route
                        .lower_raw(&mut runtime, &[])
                        .contains(&QuantumOp::Measure(0, 8))
                );
                runtime.provide_measurement_outcomes(vec![(8, 0)]).unwrap();
                assert!(
                    route
                        .lower_raw(&mut runtime, &[])
                        .contains(&QuantumOp::RXY(PI, 0.0, 0))
                );
                assert_eq!(END_CALLS.get(), 1);
                runtime.shot_start(1, None).unwrap();
                assert_eq!(runtime.instance, instance);
            });
        }
    }

    #[test]
    fn native_getters_preserve_pending_values_and_failures() {
        for leaked in [false, true] {
            with_descriptor(|mut runtime| {
                RELEASE.set(0);
                let op = if leaked {
                    QuantumOp::MeasureLeaked(0, 7)
                } else {
                    QuantumOp::Measure(0, 7)
                };
                Route::Flat.lower_raw(&mut runtime, &[op.into()]);
                assert_eq!(runtime.read_result(7).unwrap(), None);
                let value = if leaked { 1_u64 << 40 } else { 1 };
                GETTER_VALUE.set(Some(value));
                assert_eq!(runtime.read_result(7).unwrap(), Some(value));
                FAIL_GETTER.set(true);
                let error = runtime.read_result(7).unwrap_err().to_string();
                assert!(
                    error.contains("native result read failed with errno 1"),
                    "{error}"
                );
                FAIL_GETTER.set(false);
                assert!(
                    runtime.read_result(7).is_err(),
                    "a native failure must remain latched"
                );
            });
        }
    }

    // Real worker execution proves that physical feedback cannot bypass a
    // runtime's getter, including futures resolved without physical operations.
    #[test]
    fn engine_reads_runtime_values_and_processes_end_feedback() {
        use pecos_engines::{ClassicalEngine, ControlEngine, Engine, EngineStage, StateVecEngine};
        let _lock = crate::test_env::ENV_MUTEX.lock().unwrap();
        for mode in [0, 1, 2, 3] {
            with_descriptor(|mut runtime| {
                runtime.reset().unwrap();
                MIN_SUBMITTED.set(1);
                REQUIRE_FEEDBACK.set(true);
                RELEASE.set(if mode == 0 { 2 } else { 0 });
                INVERT_FEEDBACK.set(mode == 0);
                FORCE_VALUE.set(match mode {
                    1 => Some(1),
                    3 => Some(1_u64 << 40),
                    _ => None,
                });
                END_RELEASES.set(mode == 2);
                let tail = if mode == 2 {
                    "%n = call i32 @__quantum__qis__m__body(i64 0, i64 8)\nret i64 0"
                } else if mode == 3 {
                    "%id = call i64 @___lazy_measure_leaked(i64 0)\n%value = call i64 @___read_future_uint(i64 %id)\ncall void @print_uint_selene(ptr @label, i64 8, i64 %value)\nret i64 0"
                } else {
                    "%read = call i32 @__quantum__rt__result_get_one(i64 7)\n%value = zext i32 %read to i64\ncall void @print_uint_selene(ptr @label, i64 8, i64 %value)\nret i64 0"
                };
                let measure = if mode == 3 {
                    ""
                } else {
                    "%m = call i32 @__quantum__qis__m__body(i64 0, i64 7)"
                };
                let source = format!(
                    r#"
                    @label = private constant [8 x i8] c"observed"
                    declare void @print_uint_selene(ptr, i64, i64)
                    declare i64 @___lazy_measure_leaked(i64)
                    declare i64 @___read_future_uint(i64)
                    declare i32 @__quantum__qis__m__body(i64, i64)
                    declare i32 @__quantum__rt__result_get_one(i64)
                    define i64 @qmain(i64 %shot) {{
                        {measure}
                        {tail}
                    }}
                "#
                );
                let mut engine = crate::QisEngine::new(
                    Box::new(crate::QisHeliosInterface::new()),
                    Box::new(runtime),
                );
                engine.set_num_qubits_hint(1);
                engine
                    .load_program(source.as_bytes(), crate::ProgramFormat::LlvmIrText)
                    .unwrap();
                let mut quantum = StateVecEngine::new(1);
                let mut stage = engine.start(()).unwrap();
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
                loop {
                    assert!(std::time::Instant::now() < deadline);
                    match stage {
                        EngineStage::NeedsProcessing(commands) => {
                            stage = engine
                                .continue_processing(quantum.process(commands).unwrap())
                                .unwrap();
                        }
                        EngineStage::Complete(shot) => {
                            if mode != 2 {
                                assert_eq!(
                                    shot.data["observed"],
                                    pecos_engines::shot_results::Data::U64(if mode == 3 {
                                        1_u64 << 40
                                    } else {
                                        1
                                    })
                                );
                            }
                            break;
                        }
                    }
                }
                assert_eq!(END_CALLS.get(), 1);
                assert_eq!(LOCAL_BARRIERS.get(), 0);
                assert_eq!(
                    FORCED.with_borrow(Clone::clone),
                    if mode == 1 || mode == 3 {
                        vec![100]
                    } else {
                        vec![]
                    }
                );
                assert_eq!(
                    DELIVERED.with_borrow(Vec::len),
                    if mode == 1 || mode == 3 {
                        0
                    } else if mode == 0 {
                        1
                    } else {
                        2
                    }
                );
                if mode == 2 {
                    assert!(END_TAIL_EMITTED.get());
                }
            });
        }
    }

    #[test]
    fn read_targets_latest_native_future_for_reused_program_slot() {
        with_descriptor(|mut runtime| {
            RELEASE.set(0);
            let ops = Route::Flat.lower_raw(
                &mut runtime,
                &[
                    QuantumOp::Measure(0, 7).into(),
                    QuantumOp::Measure(0, 7).into(),
                ],
            );
            assert_eq!(ops, Vec::<QuantumOp>::new());
            assert!(FORCED.with_borrow(Vec::is_empty));
            runtime.force_result(7).unwrap();
            assert_eq!(FORCED.with_borrow(Clone::clone), [101]);
        });
    }

    #[test]
    fn forcing_either_measurement_kind_preserves_later_qubit_lifetime() {
        for route in Route::ALL {
            for leaked in [false, true] {
                with_descriptor(|mut runtime| {
                    RELEASE.set(0);
                    FORCE_RELEASES.set(true);
                    REQUIRE_FEEDBACK.set(true);
                    let first = if leaked {
                        QuantumOp::MeasureLeaked(7, 31)
                    } else {
                        QuantumOp::Measure(7, 31)
                    };
                    let ops = route.lower_raw(
                        &mut runtime,
                        &[
                            Operation::AllocateQubit { id: 7 },
                            first.into(),
                            Operation::ReleaseQubit { id: 7 },
                            Operation::AllocateQubit { id: 9 },
                            QuantumOp::Measure(9, 47).into(),
                        ],
                    );
                    assert!(ops.is_empty(), "{route:?}: {ops:?}");
                    assert!(FORCED.with_borrow(Vec::is_empty));
                    runtime.force_result(31).unwrap();
                    assert_eq!(FORCED.with_borrow(Clone::clone), [100]);
                    assert_eq!(LOCAL_BARRIERS.get(), 0);
                    let first = if leaked {
                        QuantumOp::MeasureLeaked(0, 31)
                    } else {
                        QuantumOp::Measure(0, 31)
                    };
                    assert_eq!(route.lower_raw(&mut runtime, &[]), [first]);
                    let value = if leaked { 2 } else { 1 };
                    runtime
                        .provide_measurement_outcomes(vec![(31, value)])
                        .unwrap();
                    assert_eq!(route.lower_raw(&mut runtime, &[]), Vec::<QuantumOp>::new());
                    runtime.force_result(47).unwrap();
                    assert_eq!(FORCED.with_borrow(Clone::clone), [100, 101]);
                    assert_eq!(LOCAL_BARRIERS.get(), 0);
                    let expected = match route {
                        Route::Flat | Route::Metadata => {
                            vec![QuantumOp::Idle(1e-9, 0), QuantumOp::Measure(0, 47)]
                        }
                        Route::Scheduled => vec![QuantumOp::Measure(0, 47)],
                    };
                    assert_eq!(route.lower_raw(&mut runtime, &[]), expected);
                    runtime.provide_measurement_outcomes(vec![(47, 0)]).unwrap();
                    assert_eq!(route.lower_raw(&mut runtime, &[]), Vec::<QuantumOp>::new());
                    assert_eq!(
                        DELIVERED.with_borrow(Clone::clone),
                        [(100, leaked, value.into()), (101, false, 0)]
                    );
                });
            }
        }
    }

    unsafe extern "C" fn increment(_: RuntimeInstance, future: u64) -> i32 {
        REFCOUNTS.with_borrow_mut(|calls| calls.push((true, future)));
        RELEASE.set(2);
        if FAIL_REFCOUNT.get() { 81 } else { 0 }
    }
    unsafe extern "C" fn decrement(_: RuntimeInstance, future: u64) -> i32 {
        REFCOUNTS.with_borrow_mut(|calls| calls.push((false, future)));
        if FAIL_REFCOUNT.get() {
            return 82;
        }
        if REQUIRE_FEEDBACK.get() && DELIVERED.with_borrow(Vec::len) != 2 {
            83
        } else {
            0
        }
    }

    #[test]
    fn refcounts_use_native_futures_and_wait_for_released_measurements() {
        for route in Route::ALL {
            with_descriptor(|mut runtime| {
                RELEASE.set(0);
                REQUIRE_FEEDBACK.set(true);
                let first = route.lower_raw(
                    &mut runtime,
                    &[
                        QuantumOp::Measure(0, 7).into(),
                        QuantumOp::MeasureLeaked(0, 8).into(),
                        Operation::RuntimeRequest {
                            request: RuntimeRequest::IncrementFutureRefcount { result_id: 7 },
                            result_id: Some(9),
                        },
                        Operation::RuntimeRequest {
                            request: RuntimeRequest::DecrementFutureRefcount { result_id: 8 },
                            result_id: Some(10),
                        },
                    ],
                );
                assert!(first.contains(&QuantumOp::Measure(0, 7)));
                assert_eq!(REFCOUNTS.with_borrow(Clone::clone), [(true, 100)]);
                runtime.provide_measurement_outcomes(vec![(7, 0)]).unwrap();
                assert!(
                    route
                        .lower_raw(&mut runtime, &[])
                        .contains(&QuantumOp::MeasureLeaked(0, 8))
                );
                assert_eq!(REFCOUNTS.with_borrow(Clone::clone), [(true, 100)]);
                runtime.provide_measurement_outcomes(vec![(8, 2)]).unwrap();
                assert_eq!(route.lower_raw(&mut runtime, &[]), []);
                assert_eq!(
                    REFCOUNTS.with_borrow(Clone::clone),
                    [(true, 100), (false, 101)]
                );
                assert_eq!(runtime.read_result(10).unwrap(), Some(0));
                assert!(FORCED.with_borrow(Vec::is_empty));
                assert_eq!(LOCAL_BARRIERS.get(), 0);
            });
        }
    }

    #[test]
    fn invalid_refcounts_fail_without_fallback() {
        for action in 0..3 {
            with_descriptor(|mut runtime| {
                RELEASE.set(0);
                runtime
                    .lower_operations(&[QuantumOp::Measure(0, 7).into()])
                    .unwrap();
                FAIL_REFCOUNT.set(true);
                let request = match action {
                    0 => RuntimeRequest::IncrementFutureRefcount { result_id: 7 },
                    1 => RuntimeRequest::DecrementFutureRefcount { result_id: 7 },
                    _ => RuntimeRequest::IncrementFutureRefcount { result_id: 99 },
                };
                let error = runtime
                    .lower_operations(&[Operation::RuntimeRequest {
                        request,
                        result_id: None,
                    }])
                    .unwrap_err()
                    .to_string();
                let expected = match action {
                    0 => "increment_future_refcount failed with errno 81",
                    1 => "decrement_future_refcount failed with errno 82",
                    _ => "unknown measurement future 99",
                };
                assert!(error.contains(expected), "{error}");
                assert!(runtime.lower_operations(&[]).is_err());
                assert!(FORCED.with_borrow(Vec::is_empty));
            });
        }
    }

    unsafe extern "C" fn custom_request(
        _: RuntimeInstance,
        tag: u64,
        data: *const c_void,
        len: usize,
        out: *mut u64,
    ) -> i32 {
        let data = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), len) };
        if data != [0, 255, 7] {
            return 71;
        }
        if tag == 99 {
            return 79;
        }
        if tag == 88 && DELIVERED.with_borrow(Vec::len) != 2 {
            return 72;
        }
        if tag == 42 {
            RELEASE.set(2);
        }
        unsafe {
            *out = u64::MAX;
        }
        0
    }

    #[test]
    fn custom_requests_release_work_and_preserve_full_width_responses() {
        for route in Route::ALL {
            with_descriptor(|mut runtime| {
                RELEASE.set(0);
                REQUIRE_FEEDBACK.set(true);
                let request = |tag, result_id| Operation::RuntimeRequest {
                    request: RuntimeRequest::Custom {
                        tag,
                        data: vec![0, 255, 7],
                    },
                    result_id: Some(result_id),
                };
                let first = route.lower_raw(
                    &mut runtime,
                    &[
                        QuantumOp::Measure(0, 7).into(),
                        QuantumOp::Measure(0, 8).into(),
                        request(42, 9),
                        request(88, 10),
                    ],
                );
                let internal = pecos_qis_ffi_types::RUNTIME_OWNED_MEASUREMENT_ID;
                assert!(first.contains(&QuantumOp::Measure(0, internal)));
                assert!(!runtime.runtime_responses.contains_key(&10));
                runtime
                    .provide_measurement_outcomes(vec![(internal, 0)])
                    .unwrap();
                assert!(
                    route
                        .lower_raw(&mut runtime, &[])
                        .contains(&QuantumOp::Measure(0, internal))
                );
                assert!(!runtime.runtime_responses.contains_key(&10));
                runtime
                    .provide_measurement_outcomes(vec![(internal, 1)])
                    .unwrap();
                assert_eq!(route.lower_raw(&mut runtime, &[]), []);
                assert_eq!(runtime.read_result(9).unwrap(), Some(u64::MAX));
                assert_eq!(runtime.read_result(10).unwrap(), Some(u64::MAX));
                assert!(FORCED.with_borrow(Vec::is_empty));
            });
        }
    }

    #[test]
    fn later_measurement_reusing_a_request_slot_reads_the_native_future() {
        for leaked in [false, true] {
            with_descriptor(|mut runtime| {
                RELEASE.set(0);
                Route::Flat.lower_raw(
                    &mut runtime,
                    &[Operation::RuntimeRequest {
                        request: RuntimeRequest::Custom {
                            tag: 0,
                            data: vec![0, 255, 7],
                        },
                        result_id: Some(7),
                    }],
                );
                assert_eq!(runtime.read_result(7).unwrap(), Some(u64::MAX));
                GETTER_VALUE.set(Some(1));
                Route::Flat.lower_raw(
                    &mut runtime,
                    &[if leaked {
                        QuantumOp::MeasureLeaked(0, 7)
                    } else {
                        QuantumOp::Measure(0, 7)
                    }
                    .into()],
                );
                assert_eq!(runtime.read_result(7).unwrap(), Some(1));
            });
        }
    }

    #[test]
    fn custom_request_errors_are_terminal() {
        with_descriptor(|mut runtime| {
            let error = runtime
                .lower_operations(&[Operation::RuntimeRequest {
                    request: RuntimeRequest::Custom {
                        tag: 99,
                        data: vec![0, 255, 7],
                    },
                    result_id: Some(0),
                }])
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("custom_call failed with errno 79"),
                "{error}"
            );
            assert!(runtime.lower_operations(&[]).is_err());
        });
    }

    #[test]
    fn engine_waits_for_custom_request_feedback_before_resuming_program() {
        use pecos_engines::{ClassicalEngine, ControlEngine, Engine, EngineStage, StateVecEngine};
        let _lock = crate::test_env::ENV_MUTEX.lock().unwrap();
        with_descriptor(|mut runtime| {
            runtime.reset().unwrap();
            RELEASE.set(0);
            REQUIRE_FEEDBACK.set(true);
            let source = format!(
                r#"
                @payload = private constant [3 x i8] c"\00\FF\07"
                @label = private constant [8 x i8] c"observed"
                declare i32 @__quantum__qis__m__body(i64, i64)
                {declare_selene_custom_runtime_call}
                declare void @print_uint_selene(ptr, i64, i64)
                define i64 @qmain(i64 %shot) {{
                    %m = call i32 @__quantum__qis__m__body(i64 0, i64 0)
                    %n = call i32 @__quantum__qis__m__body(i64 0, i64 1)
                    {call_a}
                    {call_b}
                    %value = extractvalue {{ i32, i64 }} %b, 1
                    call void @print_uint_selene(ptr @label, i64 8, i64 %value)
                    ret i64 0
                }}
            "#,
                declare_selene_custom_runtime_call = crate::selene_test_ir::declaration(
                    "selene_custom_runtime_call",
                    "ptr, i64, ptr, i64"
                ),
                call_a = crate::selene_test_ir::call(
                    "a",
                    "selene_custom_runtime_call",
                    "ptr null, i64 42, ptr @payload, i64 3"
                ),
                call_b = crate::selene_test_ir::call(
                    "b",
                    "selene_custom_runtime_call",
                    "ptr null, i64 88, ptr @payload, i64 3"
                ),
            );
            let mut engine = crate::QisEngine::new(
                Box::new(crate::QisHeliosInterface::new()),
                Box::new(runtime),
            );
            engine.set_num_qubits_hint(1);
            engine
                .load_program(source.as_bytes(), crate::ProgramFormat::LlvmIrText)
                .unwrap();
            let mut quantum = StateVecEngine::new(1);
            let mut stage = engine.start(()).unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            loop {
                assert!(std::time::Instant::now() < deadline);
                match stage {
                    EngineStage::NeedsProcessing(commands) => {
                        stage = engine
                            .continue_processing(quantum.process(commands).unwrap())
                            .unwrap();
                    }
                    EngineStage::Complete(shot) => {
                        assert_eq!(
                            shot.data["observed"],
                            pecos_engines::shot_results::Data::U64(u64::MAX)
                        );
                        break;
                    }
                }
            }
            assert_eq!(DELIVERED.with_borrow(Vec::len), 2);
            assert!(FORCED.with_borrow(Vec::is_empty));
        });
    }

    #[test]
    fn engine_waits_for_refcount_feedback_before_resuming_program() {
        use pecos_engines::{ClassicalEngine, ControlEngine, Engine, EngineStage, StateVecEngine};
        let _lock = crate::test_env::ENV_MUTEX.lock().unwrap();
        with_descriptor(|mut runtime| {
            runtime.reset().unwrap();
            RELEASE.set(0);
            REQUIRE_FEEDBACK.set(true);
            let source = format!(
                r#"
                @payload = private constant [3 x i8] c"\00\FF\07"
                @label = private constant [8 x i8] c"observed"
                declare i64 @___lazy_measure(i64)
                {declare_selene_custom_runtime_call}
                declare void @___inc_future_refcount(i64)
                declare void @___dec_future_refcount(i64)
                declare void @print_uint_selene(ptr, i64, i64)
                define i64 @qmain(i64 %shot) {{
                    %m = call i64 @___lazy_measure(i64 0)
                    %n = call i64 @___lazy_measure(i64 0)
                    call void @___inc_future_refcount(i64 %m)
                    call void @___dec_future_refcount(i64 %n)
                    {call_b}
                    %value = extractvalue {{ i32, i64 }} %b, 1
                    call void @print_uint_selene(ptr @label, i64 8, i64 %value)
                    ret i64 0
                }}
            "#,
                declare_selene_custom_runtime_call = crate::selene_test_ir::declaration(
                    "selene_custom_runtime_call",
                    "ptr, i64, ptr, i64"
                ),
                call_b = crate::selene_test_ir::call(
                    "b",
                    "selene_custom_runtime_call",
                    "ptr null, i64 88, ptr @payload, i64 3"
                ),
            );
            let mut engine = crate::QisEngine::new(
                Box::new(crate::QisHeliosInterface::new()),
                Box::new(runtime),
            );
            engine.set_num_qubits_hint(1);
            engine
                .load_program(source.as_bytes(), crate::ProgramFormat::LlvmIrText)
                .unwrap();
            let mut quantum = StateVecEngine::new(1);
            let mut stage = engine.start(()).unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            loop {
                assert!(std::time::Instant::now() < deadline);
                match stage {
                    EngineStage::NeedsProcessing(commands) => {
                        stage = engine
                            .continue_processing(quantum.process(commands).unwrap())
                            .unwrap();
                    }
                    EngineStage::Complete(shot) => {
                        assert_eq!(
                            shot.data["observed"],
                            pecos_engines::shot_results::Data::U64(u64::MAX)
                        );
                        break;
                    }
                }
            }
            assert_eq!(
                REFCOUNTS.with_borrow(Clone::clone),
                [(true, 100), (false, 101)]
            );
            assert_eq!(DELIVERED.with_borrow(Vec::len), 2);
            assert!(FORCED.with_borrow(Vec::is_empty));
        });
    }

    fn with_descriptor(run: impl FnOnce(SeleneRuntime)) {
        let executable = std::env::current_exe().unwrap();
        let source = crate::selene_runtimes::find_library_in_dir(
            executable.parent().unwrap(),
            pecos_qis_test_runtime::LIBRARY_NAME,
        )
        .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let plugin = directory.path().join(source.file_name().unwrap());
        std::fs::copy(source, &plugin).unwrap();
        let public = crate::selene_runtimes::selene_simple_runtime().unwrap();
        let library = unsafe { libloading::Library::new(&public.plugin_path).unwrap() };
        let mut descriptor =
            Box::new(unsafe { SeleneRuntime::runtime_plugin_descriptor(&library).unwrap() });
        ORIGINAL.set(Some(*descriptor));
        descriptor.get_bool_result_fn = get_bool;
        descriptor.custom_call_fn = Some(custom_request);
        descriptor.increment_future_refcount_fn = increment;
        descriptor.decrement_future_refcount_fn = decrement;
        descriptor.get_u64_result_fn = get_u64;
        descriptor.shot_end_fn = end;
        descriptor.rxy_gate_fn = rxy;
        descriptor.rz_gate_fn = rz;
        descriptor.measure_fn = measure;
        descriptor.measure_leaked_fn = measure_leaked;
        descriptor.force_result_fn = force;
        descriptor.local_barrier_fn = local;
        descriptor.global_barrier_fn = force;
        descriptor.set_bool_result_fn = set_bool;
        descriptor.set_u64_result_fn = set_u64;
        descriptor.get_next_operations_fn = unsafe {
            std::mem::transmute::<
                unsafe extern "C" fn(RuntimeInstance, SeleneRuntimeGetOperationHandle) -> i32,
                unsafe extern "C" fn(
                    RuntimeInstance,
                    selene_core::operation::plugin::RuntimeGetOperationHandle,
                ) -> i32,
            >(extract)
        };
        let fixture = unsafe { libloading::Library::new(&plugin).unwrap() };
        unsafe {
            fixture
                .get::<unsafe extern "C" fn(*mut c_void)>(b"set_descriptor")
                .unwrap()((&raw mut *descriptor).cast());
        };
        FORCED.with_borrow_mut(Vec::clear);
        FORCE_RELEASES.set(false);
        LOCAL_BARRIERS.set(0);
        SUBMITTED.with_borrow_mut(Vec::clear);
        DELIVERED.with_borrow_mut(Vec::clear);
        EMITTED.set(0);
        RELEASE.set(2);
        REVERSE.set(false);
        FAIL_SETTER.set(false);
        PAD.set(0);
        REQUIRE_FEEDBACK.set(false);
        MIN_SUBMITTED.set(2);
        GETTER_VALUE.set(None);
        FORCE_VALUE.set(None);
        FAIL_GETTER.set(false);
        INVERT_FEEDBACK.set(false);
        END_RELEASES.set(false);
        END_CALLS.set(0);
        END_TAIL_EMITTED.set(false);
        PULSE_RELEASES.set(false);
        NATIVE_CALLS.with_borrow_mut(Vec::clear);
        REFCOUNTS.with_borrow_mut(Vec::clear);
        FAIL_REFCOUNT.set(false);
        let mut runtime = SeleneRuntime::new(plugin);
        runtime.init_args.clone_from(&public.init_args);
        run(start(runtime, 1));
    }

    #[test]
    fn extraction_yields_until_each_measurement_batch_receives_feedback() {
        for route in Route::ALL {
            with_descriptor(|mut runtime| {
                REQUIRE_FEEDBACK.set(true);
                let first = route.lower_raw(
                    &mut runtime,
                    &[
                        QuantumOp::Measure(0, 7).into(),
                        QuantumOp::Measure(0, 8).into(),
                    ],
                );
                assert_eq!(
                    first
                        .iter()
                        .filter(|op| matches!(op, QuantumOp::Measure(..)))
                        .count(),
                    1
                );
                assert_eq!(route.lower_raw(&mut runtime, &[]), Vec::<QuantumOp>::new());
                runtime.provide_measurement_outcomes(vec![(7, 1)]).unwrap();
                let second = route.lower_raw(&mut runtime, &[]);
                assert!(second.contains(&QuantumOp::Measure(0, 8)));
                runtime.provide_measurement_outcomes(vec![(8, 0)]).unwrap();
                assert_eq!(route.lower_raw(&mut runtime, &[]), Vec::<QuantumOp>::new());
                assert_eq!(
                    DELIVERED.with_borrow(Clone::clone),
                    [(100, false, 1), (101, false, 0)]
                );
            });
        }
    }

    #[test]
    fn reused_slot_delivers_each_native_kind_in_emission_order() {
        for route in Route::ALL {
            for first_leaked in [false, true] {
                for deferred in [false, true] {
                    for reverse in [false, true] {
                        with_descriptor(|mut runtime| {
                            REVERSE.set(reverse);
                            RELEASE.set(if deferred { 1 } else { 2 });
                            let measure = |leaked| {
                                if leaked {
                                    QuantumOp::MeasureLeaked(0, 7)
                                } else {
                                    QuantumOp::Measure(0, 7)
                                }
                            };
                            let ops = route.lower_and_read(
                                &mut runtime,
                                &[
                                    Operation::AllocateQubit { id: 0 },
                                    measure(first_leaked).into(),
                                    measure(!first_leaked).into(),
                                ],
                            );
                            let feedback = |ops: &[QuantumOp]| {
                                ops.iter()
                                    .filter_map(|op| match op {
                                        QuantumOp::Measure(_, id) => Some((*id, 1)),
                                        QuantumOp::MeasureLeaked(_, id) => Some((*id, 2)),
                                        _ => None,
                                    })
                                    .collect::<Vec<_>>()
                            };
                            assert_eq!(feedback(&ops).len(), 1);
                            assert_eq!(route.lower_raw(&mut runtime, &[]), Vec::<QuantumOp>::new());
                            runtime
                                .provide_measurement_outcomes(feedback(&ops))
                                .unwrap();
                            {
                                RELEASE.set(2);
                                let tail =
                                    route.lower_and_read(&mut runtime, &[Operation::Barrier]);
                                assert_eq!(feedback(&tail).len(), 1);
                                runtime
                                    .provide_measurement_outcomes(feedback(&tail))
                                    .unwrap();
                            }
                            let mut expected = vec![
                                (100, first_leaked, if first_leaked { 2 } else { 1 }),
                                (101, !first_leaked, if first_leaked { 1 } else { 2 }),
                            ];
                            if reverse {
                                expected.reverse();
                            }
                            assert_eq!(
                                DELIVERED.with_borrow(Clone::clone),
                                expected,
                                "{route:?}, deferred={deferred}"
                            );
                            assert!(runtime.emitted_measurements[&7].is_empty());
                        });
                    }
                }
            }
        }
    }

    #[test]
    fn native_measurement_setter_failure_is_latched() {
        for leaked in [false, true] {
            with_descriptor(|mut runtime| {
                Route::Flat.lower_and_read(
                    &mut runtime,
                    &[
                        Operation::AllocateQubit { id: 0 },
                        if leaked {
                            QuantumOp::MeasureLeaked(0, 7)
                        } else {
                            QuantumOp::Measure(0, 7)
                        }
                        .into(),
                        QuantumOp::Measure(0, 8).into(),
                    ],
                );
                FAIL_SETTER.set(true);
                let error = runtime
                    .provide_measurement_outcomes(vec![(7, if leaked { 2 } else { 1 })])
                    .unwrap_err()
                    .to_string();
                assert!(
                    error.contains(if leaked {
                        "set_u64_result"
                    } else {
                        "set_bool_result"
                    }),
                    "{error}"
                );
                assert_eq!(
                    runtime
                        .provide_measurement_outcomes(vec![(8, 0)])
                        .unwrap_err()
                        .to_string(),
                    error
                );
                assert_eq!(DELIVERED.with_borrow(Vec::len), 1);
            });
        }
    }

    fn assert_forced_native_drain_at_v6_bound(extra: usize) {
        use crate::scheduled_transport::{ScheduledTransport, encode_mode};
        use pecos_engines::scheduled_frame::MAX_SCHEDULE_BYTES;
        let count = (MAX_SCHEDULE_BYTES - 16) / 40;
        with_descriptor(|mut runtime| {
            runtime
                .lower_scheduled_operations(&[Operation::AllocateQubit { id: 0 }])
                .unwrap();
            PAD.set(count + extra);
            let result = runtime.drain_pending_scheduled_operations();
            if extra == 0 {
                let batches = result.unwrap();
                assert_eq!(batches.len(), count);
                let (wire, _) = encode_mode(batches, 0, ScheduledTransport::V6).unwrap();
                assert_eq!(wire.as_bytes().len(), 16 + count * 40);
            } else {
                let Err(error) = result else {
                    panic!("oversized native drain must fail");
                };
                let error = error.to_string();
                assert!(error.contains("event transport limit"), "{error}");
                assert_eq!(
                    runtime
                        .drain_pending_scheduled_operations()
                        .unwrap_err()
                        .to_string(),
                    error
                );
            }
            assert_eq!(PAD.get(), 0, "must reach the V6 boundary");
        });
    }

    #[test]
    fn forced_native_drain_just_under_v6_bound_succeeds() {
        assert_forced_native_drain_at_v6_bound(0);
    }

    #[test]
    fn forced_native_drain_has_aggregate_budget() {
        assert_forced_native_drain_at_v6_bound(1);
    }

    #[test]
    fn barrier_collection_has_aggregate_budget() {
        with_descriptor(|mut runtime| {
            runtime
                .lower_scheduled_operations(&[Operation::AllocateQubit { id: 0 }])
                .unwrap();
            PAD.set((pecos_engines::scheduled_frame::MAX_SCHEDULE_BYTES - 16) / 40 + 1);
            let Err(error) = runtime.lower_scheduled_operations(&[Operation::Barrier]) else {
                panic!("unbounded Barrier output must fail");
            };
            let error = error.to_string();
            assert!(error.contains("event transport limit"), "{error}");
            assert_eq!(PAD.get(), 0);
            assert_eq!(
                runtime
                    .lower_scheduled_operations(&[])
                    .unwrap_err()
                    .to_string(),
                error
            );
        });
    }
}

#[test]
fn shot_end_rejects_undelivered_native_measurements() {
    for route in Route::ALL {
        let mut runtime = start(crate::selene_runtimes::selene_simple_runtime().unwrap(), 1);
        let ops = route.lower_and_read(&mut runtime, &[QuantumOp::Measure(0, 7).into()]);
        assert_eq!(
            ops.iter()
                .filter(|op| matches!(op, QuantumOp::Measure(..)))
                .count(),
            1
        );
        match route {
            Route::Scheduled => {
                runtime.drain_pending_scheduled_operations().unwrap();
            }
            _ => {
                runtime.drain_pending_operations().unwrap();
            }
        }
        let error = runtime.shot_end().unwrap_err().to_string();
        assert!(
            error.contains("native measurements awaiting outcomes"),
            "{error}"
        );
        assert_eq!(runtime.shot_end().unwrap_err().to_string(), error);
        assert_eq!(
            runtime
                .provide_measurement_outcomes(vec![(7, 0)])
                .unwrap_err()
                .to_string(),
            error
        );
        runtime.reset().unwrap();
        runtime.shot_start(1, Some(7)).unwrap();
        route.lower_and_read(&mut runtime, &[QuantumOp::Measure(0, 7).into()]);
        runtime.provide_measurement_outcomes(vec![(7, 1)]).unwrap();
        match route {
            Route::Scheduled => {
                runtime.drain_pending_scheduled_operations().unwrap();
            }
            _ => {
                runtime.drain_pending_operations().unwrap();
            }
        }
        assert!(runtime.shot_end().unwrap().measurements[&7]);
    }
}

#[test]
fn lazy_measurements_wait_for_a_result_read_on_every_route() {
    for route in Route::ALL {
        let mut runtime = start(crate::selene_runtimes::selene_soft_rz_runtime().unwrap(), 2);
        let initial = route.lower_raw(
            &mut runtime,
            &[
                Operation::AllocateQubit { id: 7 },
                Operation::AllocateQubit { id: 9 },
                QuantumOp::Reset(7).into(),
                QuantumOp::X(7).into(),
                QuantumOp::Measure(7, 31).into(),
                Operation::ReleaseQubit { id: 7 },
                QuantumOp::Reset(9).into(),
                QuantumOp::X(9).into(),
                QuantumOp::Measure(9, 47).into(),
            ],
        );
        assert!(
            initial.is_empty(),
            "measurement submission forced work: {route:?}: {initial:?}"
        );
        assert!(runtime.force_result(31).unwrap());
        let first = route.lower_raw(&mut runtime, &[]);
        assert!(
            first
                .iter()
                .any(|op| matches!(op, QuantumOp::Measure(_, 31)))
        );
        assert!(
            !first
                .iter()
                .any(|op| matches!(op, QuantumOp::Measure(_, 47)))
        );
        runtime.provide_measurement_outcomes(vec![(31, 1)]).unwrap();
        assert_eq!(route.lower_raw(&mut runtime, &[]), Vec::<QuantumOp>::new());
        assert!(runtime.force_result(47).unwrap());
        let second = route.lower_raw(&mut runtime, &[]);
        assert!(
            second
                .iter()
                .any(|op| matches!(op, QuantumOp::Measure(_, 47)))
        );
        runtime.provide_measurement_outcomes(vec![(47, 1)]).unwrap();
        if matches!(route, Route::Scheduled) {
            runtime.drain_pending_scheduled_operations().unwrap();
        } else {
            runtime.drain_pending_operations().unwrap();
        }
        runtime.shot_end().unwrap();
    }
}

#[test]
fn explicit_delays_follow_runtime_timestamps_on_all_routes() {
    use selene_core::runtime::{
        RuntimeInterface, RuntimeInterfaceFactory, plugin::RuntimePluginInterface,
    };

    for soft in [false, true] {
        let make = || {
            start(
                if soft {
                    crate::selene_runtimes::selene_soft_rz_runtime().unwrap()
                } else {
                    crate::selene_runtimes::selene_simple_runtime().unwrap()
                },
                2,
            )
        };
        let requests = [
            QuantumOp::Reset(0).into(),
            QuantumOp::Reset(1).into(),
            Operation::RuntimeRequest {
                request: RuntimeRequest::SimulateDelay { nanoseconds: 101 },
                result_id: None,
            },
            Operation::RuntimeRequest {
                request: RuntimeRequest::LocalBarrier {
                    qubits: vec![1],
                    sleep_ns: 23,
                },
                result_id: None,
            },
            QuantumOp::RXY(0.5, 0.0, 1).into(),
            Operation::RuntimeRequest {
                request: RuntimeRequest::GlobalBarrier { sleep_ns: 37 },
                result_id: None,
            },
            QuantumOp::Idle(11e-9, 0).into(),
            QuantumOp::RXY(0.5, 0.0, 0).into(),
            Operation::Barrier,
        ];
        let mut scheduled = make();
        let batches = scheduled.lower_scheduled_operations(&requests).unwrap();
        let mut reference = RuntimePluginInterface::new_from_file(&scheduled.plugin_path)
            .unwrap()
            .init(2, 0.into(), &scheduled.init_args)
            .unwrap();
        reference.shot_start(0, 7).unwrap();
        let mut direct = Vec::new();
        let mut qubits = BTreeMap::new();
        let drain = |reference: &mut selene_core::runtime::plugin::RuntimePlugin,
                     output: &mut Vec<_>| {
            while let Some(batch) = reference.get_next_operations().unwrap() {
                let timing = batch.runtime_source().unwrap();
                let ops = batch
                    .iter_ops()
                    .map(|op| match *op {
                        selene_core::operation::Operation::Reset { qubit_id } => {
                            RuntimeScheduledOp::Reset { qubit_id }
                        }
                        selene_core::operation::Operation::RXYGate {
                            qubit_id,
                            theta,
                            phi,
                        } => RuntimeScheduledOp::Rxy {
                            qubit_id,
                            theta,
                            phi,
                        },
                        _ => panic!("unexpected direct Selene operation {op:?}"),
                    })
                    .collect::<Vec<_>>();
                output.push((u64::from(timing.start()), u64::from(timing.duration()), ops));
            }
        };
        drain(&mut reference, &mut direct);
        for op in &requests {
            if let Operation::Quantum(qop) = op {
                for_each_quantum_qubit(qop, |q| {
                    qubits.entry(q).or_insert_with(|| {
                        let native = reference.qalloc().unwrap();
                        drain(&mut reference, &mut direct);
                        native
                    });
                });
            }
            match op {
                Operation::Quantum(QuantumOp::Reset(q)) => reference.reset(qubits[q]).unwrap(),
                Operation::Quantum(QuantumOp::RXY(theta, phi, q)) => {
                    reference.rxy_gate(qubits[q], *theta, *phi).unwrap();
                }
                Operation::Quantum(QuantumOp::Idle(_, q)) => {
                    reference.local_barrier(&[qubits[q]], 11).unwrap();
                }
                Operation::RuntimeRequest {
                    request:
                        RuntimeRequest::LocalBarrier {
                            qubits: ids,
                            sleep_ns,
                        },
                    ..
                } => reference
                    .local_barrier(
                        &ids.iter().map(|q| qubits[q]).collect::<Vec<_>>(),
                        *sleep_ns,
                    )
                    .unwrap(),
                Operation::RuntimeRequest {
                    request: RuntimeRequest::GlobalBarrier { sleep_ns },
                    ..
                } => reference.global_barrier(*sleep_ns).unwrap(),
                Operation::RuntimeRequest {
                    request: RuntimeRequest::SimulateDelay { nanoseconds },
                    ..
                } => reference.simulate_delay(*nanoseconds).unwrap(),
                Operation::Barrier => reference.global_barrier(0).unwrap(),
                _ => panic!("unexpected request {op:?}"),
            }
            drain(&mut reference, &mut direct);
        }
        let actual: Vec<_> = batches
            .iter()
            .map(|b| (b.start_time_nanos, b.duration_nanos, b.operations.clone()))
            .collect();
        assert_eq!(
            actual, direct,
            "PECOS must match Selene's plugin hosting calls"
        );
        let mut expected = Vec::new();
        let mut end = BTreeMap::new();
        for batch in &batches {
            for op in &batch.operations {
                let (q, gate) = match *op {
                    RuntimeScheduledOp::Reset { qubit_id } => (
                        qubit_id,
                        QuantumOp::Reset(usize::try_from(qubit_id).unwrap()),
                    ),
                    RuntimeScheduledOp::Rxy {
                        qubit_id,
                        theta,
                        phi,
                    } => (
                        qubit_id,
                        QuantumOp::RXY(theta, phi, usize::try_from(qubit_id).unwrap()),
                    ),
                    _ => panic!("unexpected operation {op:?}"),
                };
                let previous = end.get(&q).copied().unwrap_or(0);
                if batch.start_time_nanos > previous {
                    expected.push(QuantumOp::Idle(
                        nanoseconds_to_seconds(batch.start_time_nanos - previous),
                        usize::try_from(q).unwrap(),
                    ));
                }
                expected.push(gate);
                end.insert(q, batch.start_time_nanos + batch.duration_nanos);
            }
        }
        // These public runtimes currently ignore barrier sleep arguments.
        // Compare the same program with zero sleeps: PECOS must not add them.
        let without_delays: Vec<_> = requests
            .iter()
            .cloned()
            .map(|op| match op {
                Operation::RuntimeRequest {
                    request: RuntimeRequest::LocalBarrier { qubits, .. },
                    result_id,
                } => Operation::RuntimeRequest {
                    request: RuntimeRequest::LocalBarrier {
                        qubits,
                        sleep_ns: 0,
                    },
                    result_id,
                },
                Operation::RuntimeRequest {
                    request: RuntimeRequest::GlobalBarrier { .. },
                    result_id,
                } => Operation::RuntimeRequest {
                    request: RuntimeRequest::GlobalBarrier { sleep_ns: 0 },
                    result_id,
                },
                Operation::Quantum(QuantumOp::Idle(_, q)) => QuantumOp::Idle(0.0, q).into(),
                other => other,
            })
            .collect();
        assert_eq!(
            make().lower_scheduled_operations(&without_delays).unwrap(),
            batches
        );
        for route in [Route::Flat, Route::Metadata] {
            assert_eq!(
                route.lower_raw(&mut make(), &requests),
                expected,
                "{soft} {route:?}"
            );
        }
    }
}

#[test]
fn invalid_source_delays_fail_before_native_submission() {
    for duration in [-1.0, f64::NAN, f64::INFINITY, 0.5e-9, 1e-25, 1e30] {
        let mut runtime = crate::selene_runtimes::selene_simple_runtime().unwrap();
        let error = runtime
            .lower_operations(&[
                QuantumOp::Reset(0).into(),
                QuantumOp::Idle(duration, 0).into(),
            ])
            .unwrap_err();
        assert!(error.to_string().contains("nanoseconds"));
        assert!(!runtime.native_shot.work_submitted);
    }
    for (seconds, nanos) in [(0.0, 0), (11e-9, 11), (1e-6, 1000)] {
        assert_eq!(seconds_to_nanoseconds(seconds).unwrap(), nanos);
    }
}

#[test]
fn pecos_runtime_targeted_forcing_retains_unrelated_work_on_every_route() {
    for route in Route::ALL {
        let mut runtime = start(crate::selene_runtimes::pecos_test_runtime(&[]), 2);
        let operations = [
            QuantumOp::X(0).into(),
            QuantumOp::MeasureLeaked(0, 7).into(),
            QuantumOp::X(1).into(),
            QuantumOp::Measure(1, 8).into(),
        ];
        assert_eq!(
            route.lower_raw(&mut runtime, &operations),
            Vec::<QuantumOp>::new()
        );
        assert!(runtime.force_result(7).unwrap());
        let first = route.lower_raw(&mut runtime, &[]);
        assert!(
            first
                .iter()
                .any(|op| matches!(op, QuantumOp::MeasureLeaked(0, 7)))
        );
        assert!(
            !first
                .iter()
                .any(|op| matches!(op, QuantumOp::Measure(1, 8)))
        );
        runtime.provide_measurement_outcomes(vec![(7, 2)]).unwrap();
        assert_eq!(runtime.read_result(7).unwrap(), Some(2));
        assert!(runtime.force_result(7).unwrap());
        assert_eq!(route.lower_raw(&mut runtime, &[]), Vec::<QuantumOp>::new());
        assert!(runtime.force_result(8).unwrap());
        assert!(
            route
                .lower_raw(&mut runtime, &[])
                .iter()
                .any(|op| matches!(op, QuantumOp::Measure(1, 8)))
        );
        runtime.provide_measurement_outcomes(vec![(8, 1)]).unwrap();
        if matches!(route, Route::Scheduled) {
            runtime.drain_pending_scheduled_operations().unwrap();
        }
        runtime.shot_end().unwrap();
    }
}

#[test]
fn pecos_runtime_reused_slot_reads_the_latest_native_future() {
    for route in Route::ALL {
        let mut runtime = start(crate::selene_runtimes::pecos_test_runtime(&[]), 2);
        let ops = [
            QuantumOp::Measure(0, 3).into(),
            QuantumOp::X(0).into(),
            QuantumOp::Measure(0, 3).into(),
        ];
        assert_eq!(route.lower_raw(&mut runtime, &ops), Vec::<QuantumOp>::new());
        runtime.force_result(3).unwrap();
        for value in [0, 1] {
            assert_eq!(runtime.read_result(3).unwrap(), None);
            let emitted = route.lower_raw(&mut runtime, &[]);
            assert_eq!(
                emitted
                    .iter()
                    .filter(|op| matches!(op, QuantumOp::Measure(_, 3)))
                    .count(),
                1
            );
            runtime
                .provide_measurement_outcomes(vec![(3, value)])
                .unwrap();
        }
        assert_eq!(runtime.read_result(3).unwrap(), Some(1));
        runtime.force_result(3).unwrap();
        assert_eq!(route.lower_raw(&mut runtime, &[]), Vec::<QuantumOp>::new());
    }
}

#[test]
fn pecos_runtime_pressure_drains_between_decomposed_native_gates() {
    for route in Route::ALL {
        let mut runtime = start(
            crate::selene_runtimes::pecos_test_runtime(&["--max-queued-batches=1"]),
            2,
        );
        let emitted = route.lower_raw(
            &mut runtime,
            &[
                QuantumOp::H(0).into(),
                QuantumOp::CX(0, 1).into(),
                QuantumOp::Measure(1, 0).into(),
            ],
        );
        assert!(
            emitted
                .iter()
                .any(|op| matches!(op, QuantumOp::Measure(_, 0)))
        );
        runtime.provide_measurement_outcomes(vec![(0, 0)]).unwrap();
        assert_eq!(runtime.read_result(0).unwrap(), Some(0));
    }
}
