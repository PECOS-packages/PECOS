use pecos_engines::noise::GeneralNoiseModel;
use pecos_engines::{ClassicalEngine, Engine, GateType, QuantumSystem, StateVecEngine};
use pecos_phir_json::PhirJsonEngine;
use serde_json::json;

#[test]
fn phir_measure_reset_measure_keeps_pre_reset_leakage_in_one_batch() {
    let program = json!({
        "format":"PHIR/JSON", "version":"0.1.0", "ops":[
            {"data":"qvar_define","data_type":"qubits","variable":"q","size":1},
            {"data":"cvar_define","data_type":"u32","variable":"m","size":2},
            {"qop":"Measure","args":[["q",0]],"returns":[["m",0]]},
            {"qop":"Init","args":[["q",0]]},
            {"qop":"Measure","args":[["q",0]],"returns":[["m",1]]}
        ]
    });
    let mut classical = PhirJsonEngine::from_json(&program.to_string()).unwrap();
    let commands = classical.generate_commands().unwrap();
    assert_eq!(
        commands
            .quantum_ops()
            .unwrap()
            .iter()
            .map(|g| g.gate_type)
            .collect::<Vec<_>>(),
        vec![GateType::MZ, GateType::PZ, GateType::MZ]
    );
    let mut noise = GeneralNoiseModel::builder().build();
    noise.mark_as_leaked(0);
    let mut quantum = QuantumSystem::new(Box::new(noise), Box::new(StateVecEngine::new(1)));
    let results = quantum.process(commands).unwrap();
    assert_eq!(results.outcomes().unwrap(), vec![1, 0]);
    classical.handle_measurements(results).unwrap();
}
