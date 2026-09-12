use pecos_core::gate_type::GateType;
use pecos_engines::noise::GeneralNoiseModel;
use pecos_engines::quantum::StateVecEngine;
use pecos_engines::{ByteMessage, ControlEngine, Engine, EngineStage};
use pecos_phir_json::v0_1::engine::PhirJsonEngine;
use pecos_qasm::qasm_to_phir_json::qasm_to_phir_json;

fn execute(body: &str) -> Vec<u32> {
    let qasm = format!("OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; {body}");
    let json = qasm_to_phir_json(&qasm).unwrap();
    let mut engine = PhirJsonEngine::from_program(serde_json::from_value(json).unwrap()).unwrap();
    let EngineStage::NeedsProcessing(message) = engine.start(()).unwrap() else {
        panic!("expected quantum operations")
    };
    let mut model = GeneralNoiseModel::builder()
        .with_seed(7)
        .with_p1(0.5)
        .with_noiseless_gate(GateType::RZ)
        .with_noiseless_gate(GateType::SX)
        .with_noiseless_gate(GateType::SXdg)
        .build();
    let noise = &mut model;
    let mut outcomes = Vec::new();
    for _ in 0..64 {
        let mut sim = StateVecEngine::new(1);
        sim.process(noise.apply_noise_on_start(&message).unwrap())
            .unwrap();
        let mut measure = ByteMessage::quantum_operations_builder();
        measure.mz(&[0]);
        outcomes.extend(sim.process(measure.build()).unwrap().outcomes().unwrap());
    }
    outcomes
}

#[test]
fn phase_u_json_round_trip() {
    assert!(execute("U(0,0,pi/2) q[0];").iter().all(|&bit| bit == 0));
}

#[test]
fn sx_sxdg_json_round_trip() {
    assert!(execute("sx q[0]; sxdg q[0];").iter().all(|&bit| bit == 0));
    for gate in ["sx", "sxdg"] {
        let outcomes = execute(&format!("{gate} q[0];"));
        assert!(outcomes.contains(&0) && outcomes.contains(&1));
        assert!(
            execute(&format!("{gate} q[0]; {gate} q[0];"))
                .iter()
                .all(|&bit| bit == 1)
        );
    }
}
