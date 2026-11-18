//! Fixed-seed Jeffreys checks for the loss contract (20,000 independent shots).
use pecos_engines::noise::{GeneralNoiseModel, LossSite, QubitLoss, ReadoutMatrix};
use pecos_engines::quantum::StateVecEngine;
use pecos_engines::{ByteMessage, ControlEngine, Engine, EngineStage, Gate, GateType};
use pecos_num::jeffreys_interval;
use std::collections::BTreeMap;

const N: u64 = 20_000;
fn weights(entries: &[(&str, f64)]) -> BTreeMap<String, f64> {
    entries.iter().map(|(k, v)| ((*k).into(), *v)).collect()
}
fn lost(q: usize) -> Gate {
    Gate::simple(GateType::MeasureZLost, vec![q.into()])
}
fn idle() -> Gate {
    Gate::idle(1.0, vec![0.into()])
}
fn assert_rate(k: u64, p: f64) {
    let ci = jeffreys_interval(k, N, 1e-6).unwrap();
    assert!(
        ci.lo <= p && p <= ci.hi,
        "{k}/{N}, [{},{}] excludes {p}",
        ci.lo,
        ci.hi
    );
}
fn shots(mut noise: GeneralNoiseModel, gates: &[Gate], mut count: impl FnMut(&[u32], &[Gate])) {
    let mut sim = StateVecEngine::with_seed(2, 65537);
    for _ in 0..N {
        ControlEngine::reset(&mut noise).unwrap();
        Engine::reset(&mut sim).unwrap();
        let mut b = ByteMessage::quantum_operations_builder();
        b.add_gate_commands(gates);
        let mut stage = noise.start(b.build()).unwrap();
        let mut stream = Vec::new();
        loop {
            match stage {
                EngineStage::NeedsProcessing(commands) => {
                    stream.extend(commands.quantum_ops().unwrap());
                    stage = noise
                        .continue_processing(sim.process(commands).unwrap())
                        .unwrap();
                }
                EngineStage::Complete(result) => {
                    count(&result.outcomes().unwrap(), &stream);
                    break;
                }
            }
        }
    }
}

#[test]
fn t05_t07_recovery_maximally_mixed() {
    for site in [LossSite::BeforeTwoQubit, LossSite::BeforeSingleQubit] {
        let loss = QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(1.0)
            .add_recovery_channel(site, 0.0, 0.0)
            .build()
            .unwrap();
        let mut ones = 0;
        let gate = if site == LossSite::BeforeTwoQubit {
            Gate::cx(&[(1, 0)])
        } else {
            Gate::x(&[0])
        };
        shots(
            GeneralNoiseModel::builder()
                .with_seed(104_729)
                .with_loss(loss)
                .build(),
            &[idle(), gate.clone(), lost(0), Gate::mz(&[0])],
            |out, stream| {
                assert_ne!(out[0], 3);
                assert!(stream.contains(&gate));
                ones += u64::from(out[1] == 1);
            },
        );
        assert_rate(ones, 0.5);
    }
}

#[test]
fn t06_false_alarm_breaks_coherence() {
    // Both inputs are |+>: without reset, CX preserves them and the final H
    // returns the control to |0>, even if a recovery coin emits X.
    let loss = QubitLoss::builder()
        .with_default_recovery(false)
        .add_recovery_channel(LossSite::BeforeTwoQubit, 0.0, 1.0)
        .build()
        .unwrap();
    let gates = [
        Gate::h(&[0]),
        Gate::h(&[1]),
        Gate::cx(&[(0, 1)]),
        Gate::h(&[0]),
        Gate::mz(&[0]),
    ];
    let mut ones = 0;
    shots(
        GeneralNoiseModel::builder()
            .with_seed(104_729)
            .with_loss(loss)
            .build(),
        &gates,
        |out, stream| {
            ones += u64::from(out[0] == 1);
            for q in [0, 1] {
                assert!(stream.contains(&Gate::pz(&[q])));
                if let Some(x) = stream.iter().position(|g| *g == Gate::x(&[q])) {
                    assert_eq!(stream[x - 1], Gate::pz(&[q]));
                }
            }
        },
    );
    assert_rate(ones, 0.5);
}

#[test]
fn t10_perrin_joint_event_fractions() {
    let pc = 0.2;
    let weights = weights(&[
        ("LL", pc),
        ("LI", (1.0 - pc) * 3.0 / 16.0),
        ("IL", (1.0 - pc) * 3.0 / 16.0),
        ("LZ", (1.0 - pc) * 3.0 / 16.0),
        ("ZL", (1.0 - pc) * 3.0 / 16.0),
        ("LX", (1.0 - pc) / 16.0),
        ("XL", (1.0 - pc) / 16.0),
        ("LY", (1.0 - pc) / 16.0),
        ("YL", (1.0 - pc) / 16.0),
    ]);
    let loss = QubitLoss::builder()
        .with_default_recovery(false)
        .add_events_channel(LossSite::BeforeTwoQubit, 1.0, &weights)
        .build()
        .unwrap();
    let mut counts = [0; 4];
    shots(
        GeneralNoiseModel::builder()
            .with_seed(104_729)
            .with_loss(loss)
            .build(),
        &[Gate::cx(&[(0, 1)]), lost(0), lost(1)],
        |out, stream| {
            counts[0] += u64::from(out == [3, 3]);
            for (i, kind) in [GateType::X, GateType::Y, GateType::Z]
                .into_iter()
                .enumerate()
            {
                counts[i + 1] += u64::from(stream.iter().any(|g| g.gate_type == kind));
            }
        },
    );
    for (k, p) in counts.into_iter().zip([0.2, 0.1, 0.1, 0.3]) {
        assert_rate(k, p);
    }
}

#[test]
fn t21_scaled_measurement_loss() {
    let loss = QubitLoss::builder()
        .with_default_recovery(false)
        .with_measurement(1.0)
        .build()
        .unwrap();
    let mut count = 0;
    shots(
        GeneralNoiseModel::builder()
            .with_seed(104_729)
            .with_scale(0.5)
            .with_loss(loss)
            .build(),
        &[Gate::mz(&[0]), lost(0)],
        |out, _| {
            count += u64::from(out[1] == 3);
        },
    );
    assert_rate(count, 0.5);
}

#[test]
fn t22_configured_lost_readout() {
    let loss = QubitLoss::builder()
        .with_default_recovery(false)
        .with_idle_rate(1.0)
        .build()
        .unwrap();
    let rows = BTreeMap::from([
        ("lost".into(), weights(&[("3", 0.9), ("0", 0.1)])),
        ("leaked".into(), weights(&[("1", 1.0)])),
    ]);
    let model = GeneralNoiseModel::builder()
        .with_seed(104_729)
        .with_loss(loss)
        .with_readout(ReadoutMatrix::new(GateType::MeasureZLost, &rows).unwrap())
        .build();
    let mut zeros = 0;
    shots(model, &[idle(), lost(0)], |out, _| {
        zeros += u64::from(out[0] == 0);
    });
    assert_rate(zeros, 0.1);
}
