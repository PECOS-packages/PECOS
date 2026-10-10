//! Public propagation must never silently skip a non-Clifford gate.
use pecos_core::{Angle64, Gate, QubitId};
use pecos_qec::fault_tolerance::propagator::CountingRecorder;
use pecos_qec::fault_tolerance::propagator::dag::PropagationBuffers;
use pecos_qec::fault_tolerance::*;
use pecos_quantum::{DagCircuit, GateType, TickCircuit};
use pecos_simulators::PauliProp;
use std::collections::BinaryHeap;

fn circuits(gate: Gate) -> (TickCircuit, DagCircuit) {
    let mut tick = TickCircuit::new();
    tick.tick().h(&[0]);
    tick.tick().try_add_gate(gate.clone()).unwrap();
    tick.tick().mz(&[0]);
    let mut dag = DagCircuit::new();
    dag.h(&[0]);
    dag.add_gate_auto_wire(gate);
    dag.mz(&[0]);
    (tick, dag)
}
fn fault() -> PauliFault {
    PauliFault::new(
        SpacetimeLocation {
            tick: 1,
            qubits: vec![QubitId(0)],
            before: false,
            gate_type: GateType::H,
            gate_index: 0,
        },
        vec![1],
    )
}

fn unsupported_gates() -> [Gate; 2] {
    [
        Gate::rz(Angle64::from_radians(0.3), &[0]),
        Gate::simple(GateType::T, vec![QubitId(0)]),
    ]
}

fn reject_tick<T>(
    mut run: impl FnMut(&TickCircuit, &mut PauliProp, Direction) -> Result<T, UnsupportedGateError>,
) {
    for gate in unsupported_gates() {
        let expected = UnsupportedGateError {
            gate_type: gate.gate_type,
            angles: gate.angles.to_vec(),
            location: UnsupportedGateLocation::Tick {
                tick: 1,
                gate_in_tick: 0,
            },
            qubits: vec![0],
        };
        let (tick, _) = circuits(gate);
        for direction in [Direction::Forward, Direction::Backward] {
            let mut prop = PauliProp::new();
            prop.track_x(&[0]);
            assert_eq!(
                run(&tick, &mut prop, direction)
                    .err()
                    .expect("must reject unsupported gate"),
                expected
            );
        }
    }
}

fn reject_dag<T>(
    mut run: impl FnMut(&DagCircuit, &mut PauliProp, Direction) -> Result<T, UnsupportedGateError>,
) {
    for gate in unsupported_gates() {
        let expected = UnsupportedGateError {
            gate_type: gate.gate_type,
            angles: gate.angles.to_vec(),
            location: UnsupportedGateLocation::DagNode { node: 1 },
            qubits: vec![0],
        };
        let (_, dag) = circuits(gate);
        for direction in [Direction::Forward, Direction::Backward] {
            let mut prop = PauliProp::new();
            prop.track_x(&[0]);
            assert_eq!(
                run(&dag, &mut prop, direction)
                    .err()
                    .expect("must reject unsupported gate"),
                expected
            );
        }
    }
}

macro_rules! rejections {
    ($check:ident; $($name:ident => $run:expr),* $(,)?) => {
        $(#[test] fn $name() { $check($run); })*
    };
}

rejections! { reject_tick;
    through_circuit_rejects_non_cliffords => propagate_through_circuit,
    tick_range_rejects_non_cliffords => |tick, prop, direction| propagate_tick_range(tick, prop, 0, 2, direction),
    backward_tick_rejects_non_cliffords => |tick, prop, _| propagate_backward_from_tick(tick, prop, 2),
    fault_backward_rejects_non_cliffords => |tick, _, _| propagate_fault_backward(tick, &fault()),
    observable_backward_rejects_non_cliffords => |tick, _, _| propagate_observable_backward(tick, &[0], &[], 2),
    tick_map_rejects_non_cliffords => |tick, _, _| TickFaultAnalyzer::new(tick).map(|checker| checker.build_influence_map()),
    tick_tracked_map_rejects_non_cliffords => |tick, _, _| TickFaultAnalyzer::new(tick).map(|checker| checker.build_influence_map_with_tracked_paulis(&[(&[0], &[])])),
    fault_forward_rejects_non_cliffords => |tick, _, _| propagate_fault(tick, &PauliFault { location: SpacetimeLocation {tick: 0, ..fault().location}, ..fault() }),
    faults_forward_rejects_non_cliffords => |tick, _, _| propagate_faults(tick, &FaultConfiguration::new()),
    checker_check_logical_error_rejects_non_cliffords => |tick, _, _| PauliPropChecker::new(tick).map(|checker| checker.check_logical_error(&[0], &[])),
    checker_check_multiple_logicals_rejects_non_cliffords => |tick, _, _| PauliPropChecker::new(tick).map(|checker| checker.check_multiple_logicals(&[(&[0], &[])])),
    checker_check_error_weight_rejects_non_cliffords => |tick, _, _| PauliPropChecker::new(tick).map(|checker| checker.check_error_weight(1)),
    checker_check_syndrome_detection_rejects_non_cliffords => |tick, _, _| PauliPropChecker::new(tick).map(|checker| checker.check_syndrome_detection(&[0], &[], true)),
    checker_analyze_all_faults_rejects_non_cliffords => |tick, _, _| PauliPropChecker::new(tick).map(|checker| checker.analyze_all_faults(&[0], &[], &[(&[0], &[])])),
    checker_analyze_fault_tolerance_rejects_non_cliffords => |tick, _, _| PauliPropChecker::new(tick).map(|checker| checker.analyze_fault_tolerance(&[0], &[], &[(&[0], &[])], true)),
    checker_is_fault_tolerant_rejects_non_cliffords => |tick, _, _| PauliPropChecker::new(tick).map(|checker| checker.is_fault_tolerant(&[0], &[], &[(&[0], &[])])),
    checker_analyze_decoder_requirements_rejects_non_cliffords => |tick, _, _| PauliPropChecker::new(tick).map(|checker| checker.analyze_decoder_requirements(&[0], &[], &[(&[0], &[])])),
    checker_analyze_with_syndrome_history_rejects_non_cliffords => |tick, _, _| PauliPropChecker::new(tick).map(|checker| checker.analyze_with_syndrome_history(&[(&[0], &[])])),
    checker_analyze_with_input_faults_rejects_non_cliffords => |tick, _, _| PauliPropChecker::new(tick).map(|checker| checker.analyze_with_input_faults(&[0], &[], &[(&[0], &[])], true)),
    checker_is_fault_tolerant_with_inputs_rejects_non_cliffords => |tick, _, _| PauliPropChecker::new(tick).map(|checker| checker.is_fault_tolerant_with_inputs(&[0], &[], &[(&[0], &[])])),
    checker_analyze_with_follow_up_rejects_non_cliffords => |tick, _, _| PauliPropChecker::new(tick).map(|checker| checker.analyze_with_follow_up(&[0], &[], &[(&[0], &[])], &FollowUpConfig::new())),
    checker_is_gadget_fault_tolerant_rejects_non_cliffords => |tick, _, _| PauliPropChecker::new(tick).map(|checker| checker.is_gadget_fault_tolerant(&[0], &[], &[(&[0], &[])], &FollowUpConfig::new())),
    checker_verify_flag_fault_tolerance_rejects_non_cliffords => |tick, _, _| PauliPropChecker::new(tick).map(|checker| checker.verify_flag_fault_tolerance(&[0], &[], (&[0], &[]), 1)),
    checker_diagnose_hook_errors_rejects_non_cliffords => |tick, _, _| PauliPropChecker::new(tick).map(|checker| checker.diagnose_hook_errors(&[0], &[0], &[], &[(&[0], &[])], 1)),
    gadget_analyze_rejects_non_cliffords => |tick, _, _| GadgetChecker::from_circuit(tick).map(|checker| checker.analyze(1)),
    gadget_analyze_with_options_rejects_non_cliffords => |tick, _, _| GadgetChecker::from_circuit(tick).map(|checker| checker.analyze_with_options(1, true)),
    gadget_analyze_decoder_requirements_rejects_non_cliffords => |tick, _, _| GadgetChecker::from_circuit(tick).map(|checker| checker.analyze_decoder_requirements(1)),
    gadget_analyze_with_follow_up_rejects_non_cliffords => |tick, _, _| GadgetChecker::from_circuit(tick).map(|checker| checker.analyze_with_follow_up(1, &GadgetFollowUpConfig::new(vec![]))),
    gadget_analyze_with_syndrome_history_rejects_non_cliffords => |tick, _, _| GadgetChecker::from_circuit(tick).map(|checker| checker.analyze_with_syndrome_history(1)),
    fault_checker_circuit_fault_distance_rejects_non_cliffords => |tick, _, _| FaultChecker::new(tick).map(|checker| checker.circuit_fault_distance(&[0], &[], &[(&[0], &[])], 1)),
    fault_checker_per_logical_circuit_fault_distances_rejects_non_cliffords => |tick, _, _| FaultChecker::new(tick).map(|checker| checker.per_logical_circuit_fault_distances(&[0], &[], &[(&[0], &[])], 1)),
    fault_checker_analyze_fault_categories_rejects_non_cliffords => |tick, _, _| FaultChecker::new(tick).map(|checker| checker.analyze_fault_categories(&[0], &[], &[(&[0], &[])], true)),
    fault_checker_check_undetectable_logical_errors_rejects_non_cliffords => |tick, _, _| FaultChecker::new(tick).map(|checker| checker.check_undetectable_logical_errors(&[0], &[], &[(&[0], &[])])),
    fault_checker_check_undetectable_errors_rejects_non_cliffords => |tick, _, _| FaultChecker::new(tick).map(|checker| checker.check_undetectable_errors(&[0], &[])),
    fault_checker_check_output_weight_expansion_rejects_non_cliffords => |tick, _, _| FaultChecker::new(tick).map(|checker| checker.check_output_weight_expansion(&[0], 1)),
    run_circuit_rejects_non_cliffords => |tick, _, _| run_circuit_with_faults(tick, &mut PauliProp::new(), &FaultConfiguration::new()),
    fault_checker_check_rejects_non_cliffords => |tick, _, _| FaultChecker::new(tick).map(|checker| checker.check(|_: &PauliProp| false, PauliProp::new)),
    fault_checker_simulator_rejects_non_cliffords => |tick, _, _| FaultChecker::new(tick).map(|checker| checker.check_with_simulator(&[0], &[], &[(&[0], &[])], |_: &PauliProp, _| false, PauliProp::new))
}

rejections! { reject_dag;
    analyzer_map_rejects_non_cliffords => |dag, _, _| DagFaultAnalyzer::new(dag).build_influence_map(),
    sparse_dag_rejects_non_cliffords => propagate_sparse_dag,
    through_dag_rejects_non_cliffords => propagate_through_dag,
    backward_node_rejects_non_cliffords => |dag, prop, _| propagate_backward_from_node(dag, prop, 2),
    dag_sparse_rejects_non_cliffords => |dag, prop, direction| DagPropagator::new(dag).propagate_sparse(prop, direction),
    dag_dense_rejects_non_cliffords => |dag, prop, direction| DagPropagator::new(dag).propagate_dense(prop, direction),
    dag_backward_rejects_non_cliffords => |dag, prop, _| DagPropagator::new(dag).propagate_backward_from(prop, 2),
    analyzer_generic_rejects_non_cliffords => |dag, _, _| DagFaultAnalyzer::new(dag).propagate_from_measurement_generic(2, 0, 0, 0, &mut CountingRecorder::default(), &mut PropagationBuffers { visited: vec![false;3], active_qubits: vec![false], heap: BinaryHeap::new() }),
    analyzer_all_rejects_non_cliffords => |dag, _, _| DagFaultAnalyzer::new(dag).propagate_all(&mut CountingRecorder::default()),
    analyzer_parallel_rejects_non_cliffords => |dag, _, _| DagFaultAnalyzer::new(dag).propagate_all_parallel(),
    analyzer_forest_rejects_non_cliffords => |dag, _, _| DagFaultAnalyzer::new(dag).propagate_all_forest()
}

#[test]
fn pauli_frame_rejects_non_cliffords() {
    for gate in [
        Gate::rz(Angle64::from_radians(0.3), &[0]),
        Gate::simple(GateType::T, vec![QubitId(0)]),
    ] {
        let expected_type = format!("unsupported gate {:?}", gate.gate_type);
        let (tick, dag) = circuits(gate);
        let message = (PauliFrameLookup::from_circuit(&dag, &[], &[]))
            .expect_err("must reject unsupported gate")
            .to_string();
        assert!(message.contains(&expected_type));
        assert!(message.contains("DAG node 1"));
        let _ = (&tick, &dag);
    }
}

#[test]
fn correction_cycle_rejects_non_cliffords() {
    for gate in [
        Gate::rz(Angle64::from_radians(0.3), &[0]),
        Gate::simple(GateType::T, vec![QubitId(0)]),
    ] {
        let expected_type = format!("unsupported gate {:?}", gate.gate_type);
        let (tick, dag) = circuits(gate);
        let message = (run_correction_cycle(
            &tick,
            &FaultConfiguration::new(),
            &ErrorCorrectionConfig::new(),
            &mut LookupTableDecoder::three_qubit_bitflip(),
        ))
        .expect_err("must reject unsupported gate")
        .to_string();
        assert!(message.contains(&expected_type));
        assert!(message.contains("tick 1 gate 0"));
        let _ = (&tick, &dag);
    }
}

#[test]
fn correction_check_rejects_non_cliffords() {
    for gate in [
        Gate::rz(Angle64::from_radians(0.3), &[0]),
        Gate::simple(GateType::T, vec![QubitId(0)]),
    ] {
        let expected_type = format!("unsupported gate {:?}", gate.gate_type);
        let (tick, dag) = circuits(gate);
        let message = (ErrorCorrectionChecker::new(&tick).map(|checker| {
            checker.check(
                &mut LookupTableDecoder::three_qubit_bitflip(),
                FaultCheckConfig::default(),
                false,
            )
        }))
        .expect_err("must reject unsupported gate")
        .to_string();
        assert!(message.contains(&expected_type));
        assert!(message.contains("tick 1 gate 0"));
        let _ = (&tick, &dag);
    }
}

#[test]
fn clifford_walkers_preserve_pauli_results() {
    type TickWalk = fn(&TickCircuit, &mut PauliProp, Direction) -> Result<(), UnsupportedGateError>;
    type DagWalk = fn(&DagCircuit, &mut PauliProp, Direction) -> Result<(), UnsupportedGateError>;
    let (tick, dag) = circuits(Gate::sz(&[0]));
    let tick_walks: [TickWalk; 2] = [propagate_through_circuit, |c, p, d| {
        propagate_tick_range(c, p, 0, 2, d)
    }];
    let dag_walks: [DagWalk; 4] = [
        propagate_through_dag,
        propagate_sparse_dag,
        |c, p, d| DagPropagator::new(c).propagate_dense(p, d),
        |c, p, d| DagPropagator::new(c).propagate_sparse(p, d),
    ];
    for direction in [Direction::Forward, Direction::Backward] {
        let expected = match direction {
            Direction::Forward => (false, false),
            Direction::Backward => (true, false),
        };
        for walk in tick_walks {
            let mut prop = PauliProp::new();
            match direction {
                Direction::Forward => prop.track_x(&[0]),
                Direction::Backward => prop.track_z(&[0]),
            }
            walk(&tick, &mut prop, direction).unwrap();
            assert_eq!((prop.contains_x(0), prop.contains_z(0)), expected);
        }
        for walk in dag_walks {
            let mut prop = PauliProp::new();
            match direction {
                Direction::Forward => prop.track_x(&[0]),
                Direction::Backward => prop.track_z(&[0]),
            }
            walk(&dag, &mut prop, direction).unwrap();
            assert_eq!((prop.contains_x(0), prop.contains_z(0)), expected);
        }
    }
    let fault_back = propagate_fault_backward(&tick, &fault()).unwrap();
    assert!(fault_back.contains_x(0) && fault_back.contains_z(0));
    let backwards = [
        propagate_observable_backward(&tick, &[], &[0], 2).unwrap(),
        {
            let mut p = PauliProp::new();
            p.track_z(&[0]);
            propagate_backward_from_tick(&tick, &mut p, 2).unwrap();
            p
        },
        {
            let mut p = PauliProp::new();
            p.track_z(&[0]);
            propagate_backward_from_node(&dag, &mut p, 2).unwrap();
            p
        },
        {
            let mut p = PauliProp::new();
            p.track_z(&[0]);
            DagPropagator::new(&dag)
                .propagate_backward_from(&mut p, 2)
                .unwrap();
            p
        },
    ];
    for prop in backwards {
        assert!(prop.contains_x(0) && !prop.contains_z(0));
    }
    let fault = PauliFault {
        location: SpacetimeLocation {
            tick: 0,
            ..fault().location
        },
        ..fault()
    };
    for prop in [
        propagate_fault(&tick, &fault).unwrap(),
        propagate_faults(&tick, &FaultConfiguration::with_faults(vec![fault])).unwrap(),
    ] {
        assert!(prop.contains_x(0));
        assert!(!prop.contains_z(0));
    }
}

#[test]
fn clifford_analyzers_preserve_influences_and_order() {
    let (tick, dag) = circuits(Gate::sz(&[0]));
    let analyzer = DagFaultAnalyzer::new(&dag);
    let map = analyzer.build_influence_map().unwrap();
    assert_eq!(map.locations.len(), 3);
    let mut generic = CountingRecorder::default();
    analyzer
        .propagate_from_measurement_generic(
            2,
            0,
            0,
            0,
            &mut generic,
            &mut PropagationBuffers {
                visited: vec![false; 3],
                active_qubits: vec![false],
                heap: BinaryHeap::new(),
            },
        )
        .unwrap();
    let mut serial = CountingRecorder::default();
    analyzer.propagate_all(&mut serial).unwrap();
    assert_eq!(generic.count, 3);
    assert_eq!(serial.count, generic.count);
    assert_eq!(serial.by_pauli, generic.by_pauli);
    for influences in [
        analyzer.propagate_all_parallel().unwrap().into_soa(),
        analyzer.propagate_all_forest().unwrap().into_soa(),
    ] {
        for location in 0..3 {
            for pauli in [
                pecos_qec::fault_tolerance::propagator::Pauli::X,
                pecos_qec::fault_tolerance::propagator::Pauli::Y,
                pecos_qec::fault_tolerance::propagator::Pauli::Z,
            ] {
                assert_eq!(
                    influences.detectors(location, pauli),
                    map.influences.detectors(location, pauli)
                );
            }
        }
    }
    let analyzer = TickFaultAnalyzer::new(&tick).unwrap();
    let plain = analyzer.build_influence_map();
    let tracked = analyzer.build_influence_map_with_tracked_paulis(&[(&[], &[0])]);
    assert_eq!(plain.num_fault_locations(), 3);
    for (location, influence) in &plain.influences {
        assert_eq!(
            influence.detector_flips,
            tracked.influences[location].detector_flips
        );
    }
}

#[test]
fn errors_follow_traversal_order_and_original_batch_coordinates() {
    let mut circuit = TickCircuit::new();
    circuit.tick().h(&[0]);
    circuit.tick().h(&[1]).t(&[0]);
    circuit.tick().rz(Angle64::from_radians(0.3), &[0]);
    for direction in [Direction::Forward, Direction::Backward] {
        let error =
            propagate_through_circuit(&circuit, &mut PauliProp::new(), direction).unwrap_err();
        let location = match direction {
            Direction::Forward => UnsupportedGateLocation::Tick {
                tick: 1,
                gate_in_tick: 1,
            },
            Direction::Backward => UnsupportedGateLocation::Tick {
                tick: 2,
                gate_in_tick: 0,
            },
        };
        assert_eq!(error.location, location);
    }
    // A restricted walk does not inspect gates outside its requested range.
    propagate_tick_range(&circuit, &mut PauliProp::new(), 0, 0, Direction::Forward).unwrap();
    let dag = DagCircuit::try_from(&circuit).unwrap();
    for direction in [Direction::Forward, Direction::Backward] {
        let dense = propagate_through_dag(&dag, &mut PauliProp::new(), direction).unwrap_err();
        let sparse = propagate_sparse_dag(&dag, &mut PauliProp::new(), direction).unwrap_err();
        assert_eq!(dense, sparse); // Empty support must not bypass validation.
    }
}

#[test]
fn analyzer_checks_before_no_measurement_shortcuts_or_buffer_access() {
    for gate in unsupported_gates() {
        let mut dag = DagCircuit::new();
        dag.add_gate_auto_wire(gate);
        let analyzer = DagFaultAnalyzer::new(&dag);
        let mut recorder = CountingRecorder::default();
        assert!(analyzer.propagate_all(&mut recorder).is_err());
        assert!(analyzer.propagate_all_parallel().is_err());
        assert!(analyzer.propagate_all_forest().is_err());
        let mut work = PropagationBuffers {
            visited: vec![],
            active_qubits: vec![],
            heap: BinaryHeap::new(),
        };
        assert!(
            analyzer
                .propagate_from_measurement_generic(100, 100, 0, 0, &mut recorder, &mut work)
                .is_err()
        );
        assert_eq!(recorder.count, 0);
    }
}
