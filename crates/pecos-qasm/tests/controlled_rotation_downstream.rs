//! Exercise consumers of the phase-shaped U emitted by qelib1 controlled rotations.

use pecos_core::gate_type::GateType;
use pecos_core::{Angle64, Gate};
use pecos_engines::noise::GeneralNoiseModel;
use pecos_engines::prelude::state_vector;
use pecos_engines::{ClassicalEngine, sim_builder};
use pecos_phir_json::phir_json_engine;
use pecos_qasm::{QASMEngine, qasm_engine, qasm_to_phir_json};
use pecos_qec::fault_tolerance::dem_builder::{DemSampler, NoiseConfig};
use pecos_quantum::DagCircuit;
use pecos_quantum::hugr_convert::dag_circuit_to_hugr;
use pecos_quantum::pass::{CancelInverses, CircuitPass, StripIdentities};
use std::str::FromStr;

fn qasm(body: &str) -> String {
    format!("OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2]; creg c[2]; {body}")
}

fn emitted_gates(body: &str) -> Vec<Gate> {
    let mut engine = QASMEngine::from_str(&qasm(body)).expect("QASM must parse");
    let mut gates = Vec::new();
    loop {
        let batch = engine
            .generate_commands()
            .expect("QASM must emit commands")
            .quantum_ops()
            .expect("commands must contain valid gates");
        if batch.is_empty() {
            break;
        }
        gates.extend(batch);
    }
    gates
}

fn controlled_rotation_dag(name: &str, angle: &str) -> DagCircuit {
    let gates = emitted_gates(&format!("{name}({angle}) q[0],q[1];"));
    // Guard the integration boundary: these tests must actually reach consumers with U.
    let phases: Vec<_> = gates
        .iter()
        .filter(|gate| gate.gate_type == GateType::U)
        .collect();
    assert_eq!(phases.len(), 2, "{name} must emit two U gates");
    for gate in phases {
        assert_eq!(gate.angles.len(), 3);
        assert_eq!(gate.angles[0], Angle64::ZERO);
        assert_eq!(gate.angles[1], Angle64::ZERO);
    }
    let mut dag = DagCircuit::new();
    for gate in gates {
        dag.add_gate_auto_wire(gate);
    }
    assert!(dag.wire_count() > 0, "the emitted circuit must be wired");
    dag
}

fn noisy_qasm_outcomes(body: &str, p1: f64) -> Vec<u64> {
    let results = sim_builder()
        .classical(qasm_engine().qasm(qasm(body)))
        .quantum(state_vector())
        .noise(
            GeneralNoiseModel::builder()
                .with_p1(p1)
                .with_noiseless_gate(GateType::RZ),
        )
        .seed(42)
        .workers(1)
        .run(1000)
        .expect("QASM simulation must execute");
    results
        .try_as_shot_map()
        .unwrap()
        .try_bits_as_u64("c")
        .unwrap()
}

#[test]
fn qasm_crz_inherits_noiseless_rz() {
    controlled_rotation_dag("crz", "0.7");
    let outcomes = noisy_qasm_outcomes("crz(0.7) q[0],q[1]; measure q -> c;", 0.2);
    assert_eq!(outcomes, vec![0; 1000]);
}

#[test]
fn qasm_general_u_remains_noisy_with_noiseless_rz() {
    let body = "U(pi,0,0.7) q[1]; measure q -> c;";
    let gates = emitted_gates("U(pi,0,0.7) q[1];");
    assert_eq!(gates.len(), 1);
    assert_eq!(gates[0].gate_type, GateType::U);
    assert_ne!(gates[0].angles[0], Angle64::ZERO);
    // p1 must be NON-ZERO here: with no noise configured the outcome is
    // deterministic whatever the classification, so a zero rate cannot
    // distinguish "general U is exempt" from "general U is noisy". The
    // exemption keys on the phase-shaped form, so this U(theta=pi) must still
    // pick up faults and therefore must NOT be deterministic.
    let outcomes = noisy_qasm_outcomes(body, 0.2);
    assert_eq!(outcomes.len(), 1000);
    assert!(
        outcomes.iter().any(|outcome| *outcome != 2),
        "a general U must remain noisy when only RZ is noiseless, but every shot gave 2"
    );
    let noisy = noisy_qasm_outcomes(body, 0.2);
    assert_eq!(noisy.len(), 1000);
    assert!(
        noisy.contains(&0),
        "general U must still receive bit-flip noise"
    );
    assert!(
        noisy.contains(&2),
        "some shots must retain the ideal outcome"
    );
    assert!(noisy.iter().all(|&value| value == 0 || value == 2));
}

#[test]
fn qasm_crz_pi_builds_dem_sampler() {
    let dag = controlled_rotation_dag("crz", "pi");
    DemSampler::from_circuit(&dag, &NoiseConfig::uniform(0.01))
        .expect("DEM sampler must accept the gates emitted by QASM crz(pi)");
}

fn assert_json_execution(name: &str) {
    // Check both control states so the controlled operation actually takes effect.
    for (preparation, expected) in [("", 0), ("x q[0];", if name == "crz" { 1 } else { 3 })] {
        let source = qasm(&format!(
            "{preparation} {name}(pi) q[0],q[1]; measure q -> c;"
        ));
        let json = qasm_to_phir_json(&source).expect("QASM to PHIR/JSON conversion must succeed");
        let ops = json["ops"].as_array().unwrap();
        assert_eq!(ops.iter().filter(|op| op["qop"] == "U").count(), 2);
        if name == "cry" {
            assert!(ops.iter().any(|op| op["qop"] == "SX"));
            assert!(ops.iter().any(|op| op["qop"] == "SXdg"));
        }
        let results = sim_builder()
            .classical(phir_json_engine().json(&json.to_string()).unwrap())
            .quantum(state_vector())
            .seed(42)
            .workers(1)
            .run(32)
            .unwrap_or_else(|error| panic!("QASM {name}(pi) JSON execution failed: {error}"));
        assert_eq!(results.len(), 32);
        for shot in &results.shots {
            assert_eq!(shot.data["c"].as_u32(), Some(expected), "{name}: {shot:?}");
        }
    }
}

#[test]
fn qasm_crz_pi_json_executes() {
    assert_json_execution("crz");
}

#[test]
fn qasm_crx_pi_json_executes() {
    assert_json_execution("crx");
}

#[test]
fn qasm_cry_pi_json_executes() {
    assert_json_execution("cry");
}

#[test]
fn qasm_controlled_rotations_export_to_hugr() {
    for name in ["crz", "crx", "cry"] {
        for angle in ["pi", "0.7"] {
            let dag = controlled_rotation_dag(name, angle);
            dag_circuit_to_hugr(&dag)
                .unwrap_or_else(|error| panic!("QASM {name}({angle}) HUGR export failed: {error}"));
        }
    }
}

#[test]
fn qasm_crz_zero_optimizes_away() {
    let mut dag = controlled_rotation_dag("crz", "0");
    assert_eq!(dag.gate_count(), 4);
    StripIdentities.apply_dag(&mut dag);
    assert_eq!(dag.gate_count(), 2, "only the CX pair should remain");
    CancelInverses.apply_dag(&mut dag);
    assert_eq!(dag.gate_count(), 0);
}
