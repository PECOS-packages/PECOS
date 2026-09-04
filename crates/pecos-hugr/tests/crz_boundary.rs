// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

use pecos_engines::Engine;
use pecos_engines::byte_message::builder::ByteMessageBuilder;
use pecos_engines::hybrid::HybridEngineBuilder;
use pecos_engines::quantum::{DenseStateVecEngine, StateVecEngine};
use pecos_hugr::HugrEngine;
use pecos_quantum::hugr_convert::hugr_to_dag_circuit;
use tket::TketOp;
use tket::extension::rotation::{ConstRotation, RotationOp, RotationOpBuilder};
use tket::hugr::Hugr;
use tket::hugr::builder::{DFGBuilder, Dataflow, DataflowHugr};
use tket::hugr::std_extensions::arithmetic::float_types::ConstF64;
use tket::hugr::types::Signature;

fn crz_hugr(theta: f64, basis: usize) -> Hugr {
    let mut builder = DFGBuilder::new(Signature::new(vec![], vec![])).expect("create HUGR");
    let mut target = builder
        .add_dataflow_op(TketOp::QAlloc, vec![])
        .expect("allocate target")
        .outputs()
        .next()
        .expect("target output");
    let mut control = builder
        .add_dataflow_op(TketOp::QAlloc, vec![])
        .expect("allocate control")
        .outputs()
        .next()
        .expect("control output");
    if basis & 1 != 0 {
        target = builder
            .add_dataflow_op(TketOp::X, vec![target])
            .expect("prepare target")
            .outputs()
            .next()
            .expect("target output");
    }
    if basis & 2 != 0 {
        control = builder
            .add_dataflow_op(TketOp::X, vec![control])
            .expect("prepare control")
            .outputs()
            .next()
            .expect("control output");
    }
    let rotation = builder.add_load_value(
        ConstRotation::new(theta / std::f64::consts::PI).expect("finite HUGR rotation"),
    );
    let mut outputs = builder
        .add_dataflow_op(TketOp::CRz, vec![control, target, rotation])
        .expect("add CRz")
        .outputs();
    control = outputs.next().expect("control output");
    target = outputs.next().expect("target output");
    builder
        .add_dataflow_op(TketOp::QFree, vec![target])
        .expect("free target");
    builder
        .add_dataflow_op(TketOp::QFree, vec![control])
        .expect("free control");
    builder
        .finish_hugr_with_outputs(vec![])
        .expect("finish HUGR")
}

fn execute(theta: f64, basis: usize) -> Vec<(f64, f64)> {
    let dag = hugr_to_dag_circuit(&crz_hugr(theta, basis)).expect("convert HUGR to DAG");
    let mut builder = ByteMessageBuilder::new();
    let _ = builder.for_quantum_operations();
    for node in dag.topological_order() {
        if let Some(gate) = dag.gate(node) {
            builder.add_gate_command(gate);
        }
    }
    let mut simulator = StateVecEngine::new(2);
    simulator
        .process(builder.build())
        .expect("execute converted circuit");
    simulator
        .simulator_mut()
        .state()
        .iter()
        .map(|amplitude| (amplitude.re, amplitude.im))
        .collect()
}

fn dynamic_radd_crz_hugr() -> Hugr {
    let mut builder = DFGBuilder::new(Signature::new(vec![], vec![])).expect("create HUGR");
    let target = builder
        .add_dataflow_op(TketOp::QAlloc, vec![])
        .expect("allocate target")
        .out_wire(0);
    let control = builder
        .add_dataflow_op(TketOp::QAlloc, vec![])
        .expect("allocate control")
        .out_wire(0);
    let control = builder
        .add_dataflow_op(TketOp::H, [control])
        .expect("prepare superposed control")
        .out_wire(0);

    let halfturns_a = builder.add_load_value(ConstF64::new(1.5));
    let halfturns_b = builder.add_load_value(ConstF64::new(1.5));
    let rotation_a = builder
        .add_from_halfturns_unchecked(halfturns_a)
        .expect("construct first rotation");
    let rotation_b = builder
        .add_from_halfturns_unchecked(halfturns_b)
        .expect("construct second rotation");
    let rotation_sum = builder
        .add_dataflow_op(RotationOp::radd, [rotation_a, rotation_b])
        .expect("add rotations")
        .out_wire(0);

    let crz = builder
        .add_dataflow_op(TketOp::CRz, [control, target, rotation_sum])
        .expect("apply CRz");
    builder
        .add_dataflow_op(TketOp::QFree, [crz.out_wire(1)])
        .expect("free target");
    builder
        .add_dataflow_op(TketOp::QFree, [crz.out_wire(0)])
        .expect("free control");
    builder
        .finish_hugr_with_outputs(vec![])
        .expect("finish HUGR")
}

fn execute_dynamic_radd_crz() -> Vec<(f64, f64)> {
    let hugr_engine = HugrEngine::from_hugr(dynamic_radd_crz_hugr());
    let mut hybrid = HybridEngineBuilder::new()
        .with_classical_engine(Box::new(hugr_engine))
        .with_quantum_engine(Box::new(DenseStateVecEngine::new(2)))
        .build();
    hybrid.run_shot().expect("execute dynamic CRz HUGR");

    let state_vector = hybrid
        .quantum_system
        .quantum_engine_mut()
        .as_any_mut()
        .downcast_mut::<DenseStateVecEngine>()
        .expect("dense state-vector engine")
        .simulator_mut()
        .state();
    state_vector
        .iter()
        .map(|amplitude| (amplitude.re, amplitude.im))
        .collect()
}

fn expected_superposed_control(theta: f64) -> [(f64, f64); 4] {
    let amplitude = std::f64::consts::FRAC_1_SQRT_2;
    let controlled_amplitude = (
        amplitude * (-theta / 2.0).cos(),
        amplitude * (-theta / 2.0).sin(),
    );
    [
        (amplitude, 0.0),
        (0.0, 0.0),
        controlled_amplitude,
        (0.0, 0.0),
    ]
}

fn state_matches_up_to_global_phase(actual: &[(f64, f64)], expected: &[(f64, f64)]) -> bool {
    let reference = std::f64::consts::FRAC_1_SQRT_2;
    let phase = (actual[0].0 / reference, actual[0].1 / reference);
    actual.iter().zip(expected).all(|(&(re, im), expected)| {
        let normalized = (re * phase.0 + im * phase.1, im * phase.0 - re * phase.1);
        (normalized.0 - expected.0).abs() < 1e-12 && (normalized.1 - expected.1).abs() < 1e-12
    })
}

#[test]
fn dynamic_radd_preserves_crz_winding() {
    let actual = execute_dynamic_radd_crz();
    let expected_three_pi = expected_superposed_control(3.0 * std::f64::consts::PI);
    let wrong_pi = expected_superposed_control(std::f64::consts::PI);

    assert!(
        state_matches_up_to_global_phase(&actual, &expected_three_pi),
        "dynamic CRz state {actual:?} did not match CRz(3π) on |+0>"
    );
    assert!(
        !state_matches_up_to_global_phase(&actual, &wrong_pi),
        "dynamic CRz state {actual:?} incorrectly matched CRz(π) on |+0>"
    );
}

#[test]
fn rust_hugr_crz_boundary_preserves_full_matrix() {
    for theta in [
        -std::f64::consts::PI,
        std::f64::consts::PI / 3.0,
        std::f64::consts::PI,
        std::f64::consts::TAU,
        3.0 * std::f64::consts::PI,
    ] {
        let mut actual = vec![vec![(0.0, 0.0); 4]; 4];
        for column in 0..4 {
            for (row_values, amplitude) in actual.iter_mut().zip(execute(theta, column)) {
                row_values[column] = amplitude;
            }
        }
        let half = theta / 2.0;
        let reference = [
            (1.0, 0.0),
            (1.0, 0.0),
            (half.cos(), -half.sin()),
            (half.cos(), half.sin()),
        ];
        let phase = actual[0][0];
        let phase_norm = phase.0 * phase.0 + phase.1 * phase.1;
        assert!(
            (phase_norm - 1.0).abs() < 1e-12,
            "theta={theta}, phase={phase:?}"
        );
        if theta.abs() <= std::f64::consts::PI {
            assert!((phase.0 - 1.0).abs() < 1e-12 && phase.1.abs() < 1e-12);
        } else {
            assert!((phase.0.abs() - 1.0).abs() < 1e-12 && phase.1.abs() < 1e-12);
        }
        for (row, row_values) in actual.iter().enumerate() {
            for (column, &value) in row_values.iter().enumerate() {
                let normalized = (
                    (value.0 * phase.0 + value.1 * phase.1) / phase_norm,
                    (value.1 * phase.0 - value.0 * phase.1) / phase_norm,
                );
                let expected = if row == column {
                    reference[row]
                } else {
                    (0.0, 0.0)
                };
                assert!(
                    (normalized.0 - expected.0).abs() < 1e-12
                        && (normalized.1 - expected.1).abs() < 1e-12,
                    "theta={theta}, entry=({row}, {column}), actual={normalized:?}, expected={expected:?}, phase={phase:?}"
                );
            }
        }
    }
}
