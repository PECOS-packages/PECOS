//! Fixed-seed readout rates against conditional probabilities.
use pecos_engines::noise::{GeneralNoiseModel, ReadoutMatrix};
use pecos_engines::quantum::StateVecEngine;
use pecos_engines::{ByteMessage, ControlEngine, Engine, EngineStage, Gate, GateType};
use pecos_num::jeffreys_interval;
use std::collections::BTreeMap;

#[test]
fn present_zero_can_report_lost() {
    assert_rate(
        GateType::MeasureZLeakedLost,
        "0",
        r#"{"0":{"0":0.8,"3":0.2},"1":{"1":1.0},"leaked":{"2":1.0}}"#,
        3,
        0.2,
    );
}

#[test]
fn leaked_readout_uses_leaked_row() {
    assert_rate(
        GateType::MeasureZLeaked,
        "leaked",
        r#"{"0":{"0":1.0},"1":{"1":1.0},"leaked":{"2":0.7,"1":0.3}}"#,
        1,
        0.3,
    );
}

#[test]
fn present_one_uses_one_row() {
    assert_rate(
        GateType::MZ,
        "1",
        r#"{"0":{"0":1.0},"1":{"1":0.9,"0":0.1},"leaked":{"1":1.0}}"#,
        0,
        0.1,
    );
}

fn assert_rate(flavor: GateType, state: &str, readout: &str, counted: u32, probability: f64) {
    const SHOTS: u64 = 20_000;
    const ALPHA: f64 = 1e-6;
    let rows: BTreeMap<String, BTreeMap<String, f64>> = serde_json::from_str(readout).unwrap();
    let mut noise = GeneralNoiseModel::builder()
        .with_seed(104_729)
        .with_readout(ReadoutMatrix::new(flavor, &rows).unwrap())
        .build();
    if state == "leaked" {
        noise.mark_as_leaked(0);
    }
    let mut count = 0;
    for _ in 0..SHOTS {
        let mut sim = StateVecEngine::with_seed(1, 65537);
        let mut commands = ByteMessage::quantum_operations_builder();
        if state == "1" {
            commands.x(&[0]);
        }
        commands.add_gate_command(&Gate::simple(flavor, vec![0.into()]));
        let mut stage = noise.start(commands.build()).unwrap();
        loop {
            match stage {
                EngineStage::NeedsProcessing(commands) => {
                    stage = noise
                        .continue_processing(sim.process(commands).unwrap())
                        .unwrap();
                }
                EngineStage::Complete(result) => {
                    count += u64::from(result.outcomes().unwrap() == [counted]);
                    break;
                }
            }
        }
    }
    let interval = jeffreys_interval(count, SHOTS, ALPHA).unwrap();
    assert!(
        interval.lo <= probability && probability <= interval.hi,
        "{flavor}, true {state}: {count}/{SHOTS}, [{}, {}] excludes {probability}",
        interval.lo,
        interval.hi
    );
}
