//! Loss streams, outcomes, continuations and exact RNG tails.
use pecos_core::RngManageable;
use pecos_engines::noise::{
    GeneralNoiseConfig, GeneralNoiseModel, GeneralNoiseModelBuilder, LossSite, NoiseRng,
    P2PauliLeakageStep, P2TransitionStep, PauliLeakageChannel, QubitLoss, QubitLossBuilder,
    ReadoutMatrix, StateTransitionChannel, TwoQubitPauliLeakageChannel,
};
use pecos_engines::quantum::StateVecEngine;
use pecos_engines::runtime_frame::{FrameLimits, RuntimeNoise};
use pecos_engines::{ByteMessage, ControlEngine, Engine, EngineStage, Gate, GateType};
use serde_json::json;
use std::collections::BTreeMap;

fn weights(entries: &[(&str, f64)]) -> BTreeMap<String, f64> {
    entries.iter().map(|(k, v)| ((*k).into(), *v)).collect()
}
fn loss() -> QubitLossBuilder {
    QubitLoss::builder().with_default_recovery(false)
}
fn builder(l: QubitLossBuilder) -> GeneralNoiseModelBuilder {
    GeneralNoiseModel::builder()
        .with_seed(7919)
        .with_loss(l.build().unwrap())
}
fn lose(q: usize) -> Gate {
    Gate::idle(1.0, vec![q.into()])
}
fn m(kind: GateType, q: usize) -> Gate {
    Gate::simple(kind, vec![q.into()])
}
fn herald(q: usize) -> Gate {
    m(GateType::MeasureZLost, q)
}
fn tail(model: &GeneralNoiseModel) -> [u64; 4] {
    let mut rng = model.rng().clone();
    std::array::from_fn(|_| rng.next_u64())
}
fn run(model: &mut GeneralNoiseModel, gates: &[Gate]) -> (Vec<u32>, Vec<Gate>, [u64; 4]) {
    let mut sim = StateVecEngine::with_seed(6, 65537);
    let mut b = ByteMessage::quantum_operations_builder();
    b.add_gate_commands(gates);
    let mut stage = model.start(b.build()).unwrap();
    let mut emitted = Vec::new();
    loop {
        match stage {
            EngineStage::NeedsProcessing(commands) => {
                emitted.extend(commands.quantum_ops().unwrap());
                stage = model
                    .continue_processing(sim.process(commands).unwrap())
                    .unwrap();
            }
            EngineStage::Complete(results) => {
                return (results.outcomes().unwrap(), emitted, tail(model));
            }
        }
    }
}
fn count(stream: &[Gate], kind: GateType, q: usize) -> usize {
    stream
        .iter()
        .filter(|g| g.gate_type == kind && g.qubits.contains(&q.into()))
        .count()
}
fn transition() -> StateTransitionChannel {
    StateTransitionChannel::from_transitions(1.0, &weights(&[("0->0", 1.0), ("1->1", 1.0)]))
        .unwrap()
}

#[test]
fn t01_readout_table() {
    for kind in [
        GateType::MZ,
        GateType::MPZ,
        GateType::MeasureZLeaked,
        GateType::MeasureZLost,
        GateType::MeasureZLeakedLost,
    ] {
        for (p0, p1) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)] {
            let mut model = builder(loss().with_idle_rate(1.0))
                .with_p_prep(1.0)
                .with_prep_leak_ratio(1.0)
                .with_p_meas_0(p0)
                .with_p_meas_1(p1)
                .build();
            let result = run(
                &mut model,
                &[
                    lose(0),
                    Gate::pz(&[1]),
                    Gate::x(&[3]),
                    m(kind, 0),
                    m(kind, 1),
                    m(kind, 2),
                    m(kind, 3),
                ],
            );
            let lost = if matches!(kind, GateType::MeasureZLost | GateType::MeasureZLeakedLost) {
                3
            } else {
                u32::from(p0 > 0.5)
            };
            let leaked = if matches!(
                kind,
                GateType::MeasureZLeaked | GateType::MeasureZLeakedLost
            ) {
                2
            } else {
                u32::from(p1 == 0.0)
            };
            assert_eq!(
                result.0,
                [lost, leaked, u32::from(p0 > 0.5), u32::from(p1 == 0.0)],
                "{kind}, {p0}, {p1}"
            );
        }
    }
}

#[test]
fn t02_measurement_loss_timing() {
    let mut model = builder(loss().with_measurement(1.0)).build();
    let result = run(&mut model, &[Gate::x(&[0]), Gate::mz(&[0]), Gate::mz(&[0])]);
    assert_eq!(result.0, [1, 0]);
    assert_eq!(
        result.1,
        [
            Gate::x(&[0]),
            Gate::mz(&[0]),
            Gate::pz(&[0]),
            Gate::mz(&[0])
        ]
    );
    for reload in [false, true] {
        let mut model = builder(loss().with_measurement(1.0).with_reload_on_prep(reload)).build();
        assert_eq!(
            run(&mut model, &[Gate::mpz(&[0]), herald(0)]).0,
            [0, if reload { 0 } else { 3 }]
        );
    }
    let mut model = builder(loss().with_idle_rate(1.0)).build();
    assert_eq!(
        run(&mut model, &[lose(0), Gate::mpz(&[0]), herald(0)]).0,
        [0, 3]
    );
    for kind in [
        GateType::MZ,
        GateType::MPZ,
        GateType::MeasureZLeaked,
        GateType::MeasureZLost,
        GateType::MeasureZLeakedLost,
    ] {
        for already_lost in [false, true] {
            let config = |p| {
                let b = builder(loss().with_idle_rate(1.0).with_measurement(p));
                if already_lost {
                    b
                } else {
                    b.with_noiseless_gate(kind)
                }
            };
            let gates = if already_lost {
                vec![lose(0), m(kind, 0)]
            } else {
                vec![m(kind, 0)]
            };
            assert_eq!(
                run(&mut config(1.0).build(), &gates),
                run(&mut config(0.0).build(), &gates),
                "baseline measurement=0, {kind}"
            );
        }
    }
}

#[test]
fn t03_prep_loss_and_reload() {
    for reload in [false, true] {
        let result = run(
            &mut builder(loss().with_idle_rate(1.0).with_reload_on_prep(reload)).build(),
            &[lose(0), Gate::pz(&[0]), herald(0)],
        );
        assert_eq!(result.0, [if reload { 0 } else { 3 }]);
        assert_eq!(
            count(&result.1, GateType::PZ, 0),
            if reload { 2 } else { 1 }
        );
    }
    for initially_lost in [false, true] {
        let mut gates = if initially_lost {
            vec![lose(0)]
        } else {
            Vec::new()
        };
        gates.extend([Gate::pz(&[0, 1]), herald(0)]);
        let result = run(
            &mut builder(
                loss()
                    .with_idle_rate(1.0)
                    .with_prep(1.0)
                    .with_reload_on_prep(true),
            )
            .build(),
            &gates,
        );
        assert_eq!(result.0, [3]);
        assert_eq!(
            count(&result.1, GateType::PZ, 0),
            if initially_lost { 3 } else { 2 }
        );
        let pz: Vec<_> = result
            .1
            .iter()
            .filter(|g| g.gate_type == GateType::PZ)
            .collect();
        let user = pz
            .iter()
            .position(|gate| **gate == Gate::pz(&[0, 1]))
            .unwrap();
        assert_eq!(*pz[user + 1], Gate::pz(&[0]));
    }
    let b = builder(loss().with_idle_rate(1.0).with_prep(0.5));
    assert_eq!(
        run(&mut b.clone().build(), &[lose(0), Gate::pz(&[0])]),
        run(&mut b.build(), &[lose(0)]),
        "baseline without dropped PZ"
    );
    let make = |p| {
        builder(loss().with_prep(p))
            .with_noiseless_gate(GateType::PZ)
            .build()
    };
    assert_eq!(
        run(&mut make(0.5), &[Gate::pz(&[0])]),
        run(&mut make(0.0), &[Gate::pz(&[0])]),
        "baseline prep=0"
    );
    assert_eq!(
        run(
            &mut builder(loss().with_prep(1.0).with_reload_on_prep(true)).build(),
            &[Gate::mpz(&[0]), herald(0)]
        )
        .0,
        [0, 3]
    );
}

#[test]
fn t04_spreading_and_survivor_pauli() {
    for (p, pauli, outcome) in [
        (1.0, false, 3),
        (0.0, false, 0),
        (0.0, true, 1),
        (1.0, true, 3),
    ] {
        let mut l = loss()
            .with_idle_rate(1.0)
            .with_lost_partner_loss_probability(p);
        if pauli {
            l = l.with_lost_partner_pauli_model(&weights(&[("X", 1.0)]));
        }
        let result = run(
            &mut builder(l).build(),
            &[lose(0), Gate::cx(&[(1, 0)]), herald(1)],
        );
        assert_eq!(result.0, [outcome]);
        assert_eq!(count(&result.1, GateType::CX, 0), 0);
        assert_eq!(
            count(&result.1, GateType::X, 1),
            usize::from(pauli && p == 0.0)
        );
    }
    let b = builder(
        loss()
            .with_idle_rate(1.0)
            .with_lost_partner_loss_probability(0.0)
            .with_lost_partner_pauli_model(&weights(&[("X", 1.0)])),
    );
    let mut leaked = b.clone().build();
    leaked.mark_as_leaked(1);
    let a = run(&mut leaked, &[lose(0), Gate::cx(&[(1, 0)])]);
    let b = run(&mut b.build(), &[lose(0), Gate::cx(&[(1, 0)])]);
    assert_eq!(a.2, b.2, "baseline present survivor");
    assert_eq!(count(&a.1, GateType::X, 1), 0);
    assert_eq!(count(&b.1, GateType::X, 1), 1);
}

#[test]
fn t05_recovery_detection_and_coin() {
    let l = loss()
        .with_idle_rate(1.0)
        .add_recovery_channel(LossSite::BeforeTwoQubit, 0.0, 0.0);
    let mut model = builder(l).build();
    run(&mut model, &[lose(0)]);
    let coin = NoiseRng::new(model.rng().clone()).occurs(0.5);
    let result = run(&mut model, &[Gate::cx(&[(1, 0)]), herald(0)]);
    assert_ne!(result.0, [3]);
    assert_eq!(count(&result.1, GateType::CX, 0), 1);
    assert_eq!(result.1[0], Gate::pz(&[0]));
    assert_eq!(count(&result.1, GateType::X, 0), usize::from(coin));
    if coin {
        assert_eq!(result.1[1], Gate::x(&[0]));
    }
    let result = run(
        &mut builder(loss().with_idle_rate(1.0).add_recovery_channel(
            LossSite::BeforeTwoQubit,
            1.0,
            0.0,
        ))
        .build(),
        &[lose(0), Gate::cx(&[(1, 0)]), herald(0), herald(1)],
    );
    assert_eq!(result.0, [3, 3]);
    assert_eq!(count(&result.1, GateType::CX, 0), 0);
    assert_eq!(
        run(
            &mut builder(loss().add_recovery_channel(LossSite::BeforeTwoQubit, 0.0, 0.0)).build(),
            &[Gate::cx(&[(0, 1)])]
        ),
        run(&mut builder(loss()).build(), &[Gate::cx(&[(0, 1)])]),
        "baseline no recovery channel"
    );
}

#[test]
fn t06_false_alarm_clears_leak_and_resets() {
    let mut model =
        builder(loss().add_recovery_channel(LossSite::BeforeTwoQubit, 0.0, 1.0)).build();
    model.mark_as_leaked(0);
    let result = run(
        &mut model,
        &[Gate::cx(&[(0, 1)]), m(GateType::MeasureZLeaked, 0)],
    );
    assert_ne!(result.0, [2]);
    for q in [0, 1] {
        let pz = result.1.iter().position(|g| *g == Gate::pz(&[q])).unwrap();
        if let Some(x) = result.1.iter().position(|g| *g == Gate::x(&[q])) {
            assert_eq!(x, pz + 1);
        }
    }
}

#[test]
fn t07_all_sites_and_event_pauli_on_leaked() {
    for site in [
        LossSite::BeforeSingleQubit,
        LossSite::AfterSingleQubit,
        LossSite::AfterTwoQubit,
    ] {
        let l = loss()
            .with_idle_rate(1.0)
            .add_recovery_channel(site, 0.0, 0.0);
        let gate = if site == LossSite::AfterTwoQubit {
            Gate::cx(&[(1, 0)])
        } else {
            Gate::h(&[0])
        };
        let result = run(
            &mut builder(l).build(),
            &[lose(0), gate.clone(), herald(0), herald(1)],
        );
        assert!(result.0.iter().all(|v| *v != 3));
        assert_eq!(
            result.1.contains(&gate),
            site == LossSite::BeforeSingleQubit
        );
    }
    for site in [
        LossSite::BeforeSingleQubit,
        LossSite::AfterSingleQubit,
        LossSite::BeforeTwoQubit,
        LossSite::AfterTwoQubit,
    ] {
        let single = matches!(
            site,
            LossSite::BeforeSingleQubit | LossSite::AfterSingleQubit
        );
        let gate = if single {
            Gate::h(&[0])
        } else {
            Gate::cx(&[(0, 1)])
        };
        let l = loss().add_events_channel(
            site,
            1.0,
            &weights(&[(if single { "L" } else { "LI" }, 1.0)]),
        );
        let result = run(
            &mut builder(l).build(),
            &[gate.clone(), herald(0), herald(1)],
        );
        assert_eq!(result.0, [3, 0]);
        assert_eq!(
            result.1.contains(&gate),
            matches!(site, LossSite::AfterSingleQubit | LossSite::AfterTwoQubit)
        );
    }
    let make = || {
        builder(loss().add_events_channel(LossSite::BeforeTwoQubit, 1.0, &weights(&[("XI", 1.0)])))
            .build()
    };
    let mut leaked = make();
    leaked.mark_as_leaked(0);
    let a = run(&mut leaked, &[Gate::cx(&[(0, 1)])]);
    let b = run(&mut make(), &[Gate::cx(&[(0, 1)])]);
    assert_eq!(a.2, b.2, "baseline nonleaked events leg");
    assert_eq!(count(&a.1, GateType::X, 0), 0);
    assert_eq!(count(&b.1, GateType::X, 0), 1);
}

#[test]
fn t09_joint_events_do_not_spread() {
    for label in ["LL", "LZ"] {
        let l = loss().add_events_channel(LossSite::BeforeTwoQubit, 1.0, &weights(&[(label, 1.0)]));
        let result = run(
            &mut builder(l).build(),
            &[
                Gate::h(&[1]),
                Gate::cx(&[(0, 1)]),
                Gate::h(&[1]),
                herald(0),
                herald(1),
            ],
        );
        assert_eq!(result.0, [3, if label == "LL" { 3 } else { 1 }]);
        assert_eq!(count(&result.1, GateType::CX, 0), 0);
    }
}

#[test]
fn t11_already_lost_vs_own_event_batched() {
    let mut model = builder(loss().with_idle_rate(1.0).add_events_channel(
        LossSite::BeforeTwoQubit,
        1.0,
        &weights(&[("LI", 1.0)]),
    ))
    .build();
    run(&mut model, &[lose(0)]);
    let mut expected = NoiseRng::new(model.rng().clone());
    expected.occurs(1.0); // event outer for B
    // categorical on B, then spreading on A; a one-label WeightedSampler still draws.
    let _ =
        pecos_engines::noise::WeightedSampler::new(&weights(&[("LI", 1.0)])).sample(&mut expected);
    expected.occurs(1.0);
    let result = run(&mut model, &[Gate::cx(&[(0, 1), (2, 3)])]);
    assert_eq!(
        result.2,
        std::array::from_fn(|_| expected.inner_mut().next_u64()),
        "baseline one outer + categorical + spread"
    );
    assert_eq!(
        run(&mut model, &[herald(0), herald(1), herald(2), herald(3)]).0,
        [3, 3, 3, 0]
    );
}

#[test]
fn t12_hook_order_and_suppression() {
    let hook = TwoQubitPauliLeakageChannel::new(1.0, &weights(&[("ZZ", 1.0)])).unwrap();
    for before in [false, true] {
        let l = loss().with_idle_rate(1.0).add_recovery_channel(
            if before {
                LossSite::BeforeTwoQubit
            } else {
                LossSite::AfterTwoQubit
            },
            0.0,
            0.0,
        );
        let b = builder(l);
        let b = if before {
            b.add_p2_pauli_leakage_step_before_gate(&P2PauliLeakageStep::joint(hook.clone()))
        } else {
            b.add_p2_pauli_leakage_step_after_gate(&P2PauliLeakageStep::joint(hook.clone()))
        };
        let result = run(&mut b.build(), &[lose(0), Gate::cx(&[(1, 0)])]);
        assert_eq!(count(&result.1, GateType::Z, 0), usize::from(before));
        assert_eq!(count(&result.1, GateType::CX, 0), usize::from(before));
    }
}

#[test]
fn t12b_suppression_survives_after_resume() {
    let hook = TwoQubitPauliLeakageChannel::new(1.0, &weights(&[("ZZ", 1.0)])).unwrap();
    let b = builder(loss().with_idle_rate(1.0).add_recovery_channel(
        LossSite::AfterTwoQubit,
        0.0,
        0.0,
    ))
    .add_p2_transition_step_after_gate(&P2TransitionStep::independent(transition(), transition()))
    .add_p2_pauli_leakage_step_after_gate(&P2PauliLeakageStep::joint(hook.clone()));
    let result = run(&mut b.build(), &[lose(0), Gate::cx(&[(0, 1), (2, 3)])]);
    for q in [0, 1] {
        assert_eq!(count(&result.1, GateType::Z, q), 0);
        assert_eq!(count(&result.1, GateType::MZ, q), 0);
    }
    for q in [2, 3] {
        assert_eq!(count(&result.1, GateType::Z, q), 1);
        assert_eq!(count(&result.1, GateType::MZ, q), 1);
    }
}

#[test]
fn t12c_new_after_loss_skips_nonloss_hooks() {
    let transition =
        StateTransitionChannel::from_transitions(1.0, &weights(&[("0->L", 1.0), ("1->L", 1.0)]))
            .unwrap();
    for single in [false, true] {
        let b = builder(loss().add_events_channel(
            if single {
                LossSite::AfterSingleQubit
            } else {
                LossSite::AfterTwoQubit
            },
            1.0,
            &weights(&[(if single { "L" } else { "LI" }, 1.0)]),
        ));
        let b = if single {
            b.add_p1_pauli_leakage_channel_after_gate(
                &PauliLeakageChannel::new(1.0, &weights(&[("L", 1.0)])).unwrap(),
            )
            .add_p1_transition_channel_after_gate(&transition)
        } else {
            b.add_p2_pauli_leakage_step_after_gate(&P2PauliLeakageStep::joint(
                TwoQubitPauliLeakageChannel::new(1.0, &weights(&[("LL", 1.0)])).unwrap(),
            ))
            .add_p2_transition_step_after_gate(&P2TransitionStep::independent(
                transition.clone(),
                transition.clone(),
            ))
        };
        let result = run(
            &mut b.build(),
            &[
                if single {
                    Gate::h(&[0])
                } else {
                    Gate::cx(&[(0, 1)])
                },
                herald(0),
                m(GateType::MeasureZLeaked, 1),
            ],
        );
        assert_eq!(result.0, [3, 0]);
        assert_eq!(count(&result.1, GateType::MZ, 0), 0);
        assert_eq!(count(&result.1, GateType::PZ, 0), 1);
        assert_eq!(count(&result.1, GateType::PZ, 1), 0);
    }
}

#[test]
fn t13_idle_order_sites_and_zero() {
    let result = run(
        &mut builder(loss().with_idle_rate(1.0))
            .with_idle_after_2q(1.0)
            .build(),
        &[Gate::cx(&[(0, 1)]), herald(0), herald(1)],
    );
    assert_eq!(result.0, [3, 3]);
    let result = run(
        &mut builder(loss().with_idle_rate(1.0))
            .with_p_idle_linear(1.0, &weights(&[("X", 1.0)]))
            .build(),
        &[lose(0), herald(0)],
    );
    assert_eq!(result.0, [3]);
    assert_eq!(count(&result.1, GateType::X, 0), 0);
    assert_eq!(
        run(&mut builder(loss()).build(), &[lose(0)]),
        run(
            &mut GeneralNoiseModel::builder().with_seed(7919).build(),
            &[lose(0)]
        ),
        "baseline loss absent"
    );
}

#[test]
fn t14_noiseless_filter_reload_and_readout() {
    let b = builder(loss().with_idle_rate(1.0))
        .with_noiseless_gate(GateType::CX)
        .with_noiseless_gate(GateType::MZ);
    assert_eq!(
        run(
            &mut b.clone().build(),
            &[lose(0), Gate::cx(&[(1, 0)]), herald(1), Gate::mz(&[0])]
        ),
        run(&mut b.build(), &[lose(0), herald(1), Gate::mz(&[0])]),
        "baseline without noiseless lost CX"
    );
    let nominal = run(
        &mut builder(loss().with_idle_rate(1.0))
            .with_p_meas_0(1.0)
            .with_noiseless_gate(GateType::MZ)
            .build(),
        &[lose(0), Gate::mz(&[0])],
    );
    assert_eq!(nominal.0, [0]);
    for kind in [GateType::PZ, GateType::MPZ] {
        for reload in [false, true] {
            let b = builder(
                loss()
                    .with_idle_rate(1.0)
                    .with_reload_on_prep(reload)
                    .with_prep(0.5)
                    .with_measurement(0.5),
            )
            .with_noiseless_gate(kind);
            let result = run(&mut b.build(), &[lose(0), m(kind, 0), herald(0)]);
            assert_eq!(*result.0.last().unwrap(), if reload { 0 } else { 3 });
        }
    }
}

#[test]
fn t15_lost_operands_skip_all_base_draws_and_allocation() {
    for gate in [
        Gate::x(&[0]),
        Gate::cx(&[(1, 0)]),
        m(GateType::QFree, 0),
        m(GateType::QAlloc, 0),
        m(GateType::PX, 0),
        m(GateType::MX, 0),
        m(GateType::MeasureFree, 0),
        m(GateType::TrackedPauliMeta, 0),
    ] {
        let b = builder(
            loss()
                .with_idle_rate(1.0)
                .with_lost_partner_loss_probability(0.0),
        )
        .with_p1(0.5)
        .with_p2(0.5);
        assert_eq!(
            run(&mut b.clone().build(), &[lose(0), gate, herald(0)]),
            run(&mut b.build(), &[lose(0), herald(0)]),
            "baseline omitted lost gate"
        );
    }
    assert_eq!(
        run(
            &mut builder(loss().with_idle_rate(1.0)).build(),
            &[
                lose(0),
                m(GateType::QFree, 0),
                m(GateType::QAlloc, 0),
                herald(0)
            ]
        )
        .0,
        [3]
    );
}

#[test]
fn custom_is_one_lost_operand_group() {
    let custom = Gate::simple(GateType::Custom, vec![0.into(), 1.into()]);
    let hook = PauliLeakageChannel::new(1.0, &weights(&[("Z", 1.0)])).unwrap();
    let b = builder(loss().with_idle_rate(1.0))
        .with_p1(0.5)
        .add_p1_pauli_leakage_channel_before_gate(&hook);
    assert_eq!(
        run(&mut b.clone().build(), &[lose(0), custom.clone()]),
        run(&mut b.build(), &[lose(0)]),
        "baseline no Custom"
    );
    let b = builder(loss().add_events_channel(
        LossSite::BeforeSingleQubit,
        1.0,
        &weights(&[("L", 1.0)]),
    ))
    .with_p1(1.0);
    let r = run(&mut b.build(), &[custom.clone(), herald(0), herald(1)]);
    assert_eq!(r.0, [3, 3]);
    assert!(!r.1.contains(&custom));
}

#[test]
fn custom_recovery_rechecks_the_whole_operand_group() {
    let custom = Gate::simple(GateType::Custom, vec![0.into(), 1.into()]);
    let hook = PauliLeakageChannel::new(1.0, &weights(&[("Z", 1.0)])).unwrap();
    for site in [
        Some(LossSite::BeforeSingleQubit),
        None,
        Some(LossSite::AfterSingleQubit),
    ] {
        let mut config = loss().with_idle_rate(1.0);
        if let Some(site) = site {
            config = config.add_recovery_channel(site, 0.0, 0.0);
        }
        let b = builder(config)
            .add_p1_pauli_leakage_channel_before_gate(&hook)
            .add_p1_pauli_leakage_channel_after_gate(&hook);
        let result = run(&mut b.clone().build(), &[lose(0), custom.clone()]);
        if site == Some(LossSite::BeforeSingleQubit) {
            let position = result
                .1
                .iter()
                .position(|gate| *gate == custom)
                .expect("before-site recovery must permit the whole Custom command");
            for qubit in [0, 1] {
                assert_eq!(count(&result.1[..position], GateType::Z, qubit), 1);
                assert_eq!(count(&result.1[position + 1..], GateType::Z, qubit), 1);
            }
        } else {
            assert!(!result.1.contains(&custom));
            for qubit in [0, 1] {
                assert_eq!(count(&result.1, GateType::Z, qubit), 0);
            }
            if site.is_none() {
                assert_eq!(
                    result,
                    run(&mut b.build(), &[lose(0)]),
                    "baseline without Custom: no emission, hook or p1 draws"
                );
            }
        }
    }
}

#[test]
fn t17_loss_entry_runs_after_before_resume() {
    let b =
        builder(loss().add_events_channel(LossSite::BeforeTwoQubit, 1.0, &weights(&[("LI", 1.0)])))
            .add_p2_transition_step_before_gate(&P2TransitionStep::independent(
                transition(),
                transition(),
            ));
    let result = run(&mut b.build(), &[Gate::cx(&[(0, 1)]), herald(0), herald(1)]);
    assert_eq!(result.0, [3, 0]);
    assert_eq!(count(&result.1, GateType::MZ, 0), 1);
    assert_eq!(count(&result.1, GateType::CX, 0), 0);
}

#[test]
fn t18_lost_crosstalk_victims() {
    let b = builder(loss().with_idle_rate(1.0))
        .with_p_meas_crosstalk_global(1.0)
        .with_p_meas_crosstalk_local(1.0)
        .with_p_prep_crosstalk(1.0);
    for gate in [
        m(GateType::MeasCrosstalkGlobalPayload, 0),
        m(GateType::MeasCrosstalkLocalPayload, 2),
        Gate::pz(&[0]),
    ] {
        let result = run(&mut b.clone().build(), &[Gate::pz(&[2]), lose(2), gate]);
        assert_eq!(count(&result.1, GateType::MZ, 2), 0);
    }
}

#[test]
fn t19_off_and_apply_to() {
    for (key, value) in [
        ("measurement", json!(0)),
        ("prep", json!(0)),
        ("idle_rate", json!(0)),
        ("lost_partner", json!({})),
        ("reload_on_prep", json!(false)),
        ("default_recovery", json!(false)),
        ("channels", json!([])),
    ] {
        let config = json!({"loss":{"enabled":false,key:value}});
        assert!(
            GeneralNoiseConfig::from_json(&config.to_string())
                .unwrap()
                .build()
                .unwrap_err()
                .to_string()
                .contains(&format!("loss.{key} is set"))
        );
    }
    let apply = |text: &str, b| {
        GeneralNoiseConfig::from_json(text)
            .unwrap()
            .apply_to(b)
            .unwrap()
    };
    let b = builder(loss().with_measurement(1.0));
    assert!(apply("{}", b.clone()).build().has_loss());
    assert!(
        !apply(r#"{"loss":{"enabled":false}}"#, b.clone())
            .build()
            .has_loss()
    );
    let gates = [Gate::x(&[0]), Gate::mz(&[0]), Gate::mz(&[0])];
    assert_eq!(
        run(
            &mut apply(r#"{"loss":{"enabled":true,"default_recovery":false}}"#, b).build(),
            &gates
        )
        .0,
        [1, 1]
    );
    let off = apply(
        r#"{"loss":{"enabled":false}}"#,
        GeneralNoiseModel::builder().with_seed(7919),
    );
    assert_eq!(
        run(&mut off.build(), &gates),
        run(
            &mut GeneralNoiseModel::builder().with_seed(7919).build(),
            &gates
        ),
        "baseline absent loss section"
    );
    assert!(
        QubitLoss::from_json_value(&json!({}))
            .unwrap_err()
            .to_string()
            .contains("loss.enabled is required")
    );
}

#[test]
fn t19b_zero_loss_is_stream_invariant() {
    let hook = PauliLeakageChannel::new(0.2, &weights(&[("X", 0.5), ("L", 0.5)])).unwrap();
    let b = GeneralNoiseModel::builder()
        .with_seed(7919)
        .with_p_meas_crosstalk_local(1.0)
        .add_p1_pauli_leakage_channel_after_gate(&hook)
        .add_p2_transition_step_before_gate(&P2TransitionStep::independent(
            transition(),
            transition(),
        ));
    let gates = [
        Gate::pz(&[0, 1, 2, 3]),
        Gate::h(&[0, 1]),
        Gate::cx(&[(0, 1), (2, 3)]),
        lose(0),
        Gate::mz(&[0]),
        Gate::mpz(&[1]),
        herald(2),
        m(GateType::MeasureZLeakedLost, 3),
        m(GateType::MeasCrosstalkLocalPayload, 2),
    ];
    assert_eq!(
        run(
            &mut b.clone().with_loss(loss().build().unwrap()).build(),
            &gates
        ),
        run(&mut b.build(), &gates),
        "baseline identical configuration with loss absent"
    );
}

#[test]
fn t20_runtime_and_simple_subset_reject() {
    let b = builder(loss());
    assert!(b.simple_probabilities().is_none());
    let error = RuntimeNoise::new(b, 2, FrameLimits::default())
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("qubit loss") && error.contains("remove the loss section"));
}

#[test]
fn t21_idle_overflow_and_scaled_probability_validation() {
    let error = builder(loss().with_measurement(0.6))
        .with_scale(2.0)
        .try_build()
        .unwrap_err()
        .to_string();
    assert_eq!(
        error,
        "Input error: loss.measurement after scaling = 1.2 (0.6 x scale 2): reduce the probability or scale factors to keep the scaled probability finite in [0, 1]"
    );
    for (config, field) in [
        (loss().with_prep(0.75), "prep"),
        (
            loss().add_events_channel(LossSite::BeforeSingleQubit, 0.75, &weights(&[("L", 1.0)])),
            "channels[0].probability",
        ),
        (
            loss().add_events_channel(LossSite::AfterTwoQubit, 0.75, &weights(&[("LI", 1.0)])),
            "channels[0].probability",
        ),
    ] {
        let error = builder(config)
            .with_scale(4.0)
            .with_prep_scale(0.5)
            .with_p1_scale(0.5)
            .with_p2_scale(0.5)
            .try_build()
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(&format!(
                "loss.{field} after scaling = 1.5 (0.75 x scale 2)"
            )),
            "{error}"
        );
    }
    let error = builder(loss().with_idle_rate(1e308))
        .with_scale(2.0)
        .try_build()
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("loss.idle_rate after scaling = inf")
            && error.contains(&format!("({} x scale 2)", 1e308)),
        "{error}"
    );
    let raw_error = loss()
        .with_measurement(1.2)
        .build()
        .unwrap_err()
        .to_string();
    assert!(
        raw_error.contains("loss.measurement = 1.2: use a finite probability in [0, 1]"),
        "{raw_error}"
    );
    assert!(!raw_error.contains("after scaling"));
    let b = builder(loss().with_idle_rate(1e200));
    assert!(
        b.clone()
            .with_idle_after_2q(1e200)
            .try_build()
            .unwrap_err()
            .to_string()
            .contains("loss.idle_rate")
    );
    let mut input = ByteMessage::quantum_operations_builder();
    input.add_gate_command(&Gate::idle(1e200, vec![0.into()]));
    assert!(
        b.build()
            .start(input.build())
            .err()
            .unwrap()
            .to_string()
            .contains("loss.idle_rate")
    );
    run(
        &mut GeneralNoiseModel::default(),
        &[Gate::idle(1e200, vec![0.into()])],
    );
    let result = run(
        &mut builder(loss().add_recovery_channel(LossSite::BeforeTwoQubit, 0.0, 1.0))
            .with_scale(0.0)
            .build(),
        &[Gate::cx(&[(0, 1)])],
    );
    assert_eq!(count(&result.1, GateType::PZ, 0), 1);
    assert_eq!(count(&result.1, GateType::PZ, 1), 1);
}

#[test]
fn t22_lost_readout_noiseless_and_rows() {
    let rows = BTreeMap::from([
        ("lost".into(), weights(&[("0", 1.0)])),
        ("leaked".into(), weights(&[("1", 1.0)])),
    ]);
    let b = builder(loss().with_idle_rate(1.0))
        .with_readout(ReadoutMatrix::new(GateType::MeasureZLost, &rows).unwrap());
    assert_eq!(run(&mut b.clone().build(), &[lose(0), herald(0)]).0, [0]);
    let mut model = b.with_noiseless_gate(GateType::MeasureZLost).build();
    run(&mut model, &[lose(0)]);
    let before = tail(&model);
    let result = run(&mut model, &[herald(0)]);
    assert_eq!(result.0, [3]);
    assert_eq!(
        result.2, before,
        "baseline RNG just before noiseless herald"
    );
}

#[test]
fn t23_shot_reset_clears_loss() {
    let mut model = builder(loss().with_idle_rate(1.0)).build();
    run(&mut model, &[lose(0)]);
    ControlEngine::reset(&mut model).unwrap();
    assert_eq!(run(&mut model, &[herald(0)]).0, [0]);
}

#[test]
fn t24_validation_and_lost_row() {
    let errors = [
        (
            json!({"enabled":true,"measurement":true}),
            "measurement",
            "true",
        ),
        (json!({"enabled":1}), "enabled", "1"),
        (json!({"enabled":true,"prep":-0.1}), "prep", "-0.1"),
        (json!({"enabled":true,"idle_rate":-1}), "idle_rate", "-1"),
        (
            json!({"enabled":true,"reload_on_prep":0}),
            "reload_on_prep",
            "0",
        ),
        (
            json!({"enabled":true,"default_recovery":"false"}),
            "default_recovery",
            "false",
        ),
        (json!({"enabled":true,"typo":0}), "typo", "0"),
        (
            json!({"enabled":true,"lost_partner":{"typo":0}}),
            "lost_partner.typo",
            "0",
        ),
        (
            json!({"enabled":true,"lost_partner":{"loss_probability":2}}),
            "lost_partner.loss_probability",
            "2",
        ),
        (
            json!({"enabled":true,"lost_partner":{"pauli_model":{"L":1}}}),
            "lost_partner.pauli_model",
            "L",
        ),
        (
            json!({"enabled":true,"lost_partner":{"pauli_model":{"I":0.1}}}),
            "lost_partner.pauli_model",
            "0.1",
        ),
    ];
    for (value, field, bad) in errors {
        let err = QubitLoss::from_json_value(&value).unwrap_err().to_string();
        assert!(
            err.contains(&format!("loss.{field}")) && err.contains(bad),
            "{err}"
        );
    }
    let base = json!({"site":"before_two_qubit","kind":"events","probability":1,"events":{"LI":1}});
    for (field, bad) in [
        ("site", json!("wrong")),
        ("kind", json!("wrong")),
        ("probability", json!(2)),
        ("probability", json!("1")),
        ("events", json!({})),
        ("events", json!({"II":1})),
        ("events", json!({"L":1})),
        ("events", json!({"AB":1})),
        ("events", json!({"LI":0.2})),
        ("typo", json!(7)),
    ] {
        let mut channel = base.clone();
        channel[field] = bad.clone();
        let err=QubitLoss::from_json_value(&json!({"enabled":true,"channels":[{"site":"after_single_qubit","kind":"recovery","false_negative":0,"false_positive":0},channel]})).unwrap_err().to_string();
        assert!(err.contains(&format!("loss.channels[1].{field}")), "{err}");
        let bad_text = if let Some(map) = bad.as_object() {
            let weights: BTreeMap<_, _> = map
                .iter()
                .map(|(key, value)| (key.clone(), value.as_f64().unwrap()))
                .collect();
            if matches!(
                map.keys().next().map(String::as_str),
                Some("II" | "L" | "AB")
            ) {
                let (label, weight) = weights.first_key_value().unwrap();
                format!("{label:?}: {weight}")
            } else {
                format!("{weights:?}")
            }
        } else {
            bad.to_string()
        };
        assert!(
            err.contains(&bad_text),
            "missing bad value {bad_text} in {err}"
        );
    }
    for (site, label) in [
        (LossSite::BeforeSingleQubit, "LI"),
        (LossSite::BeforeSingleQubit, "I"),
        (LossSite::BeforeTwoQubit, "X"),
    ] {
        assert!(
            loss()
                .add_events_channel(site, 1.0, &weights(&[(label, 1.0)]))
                .build()
                .is_err()
        );
    }
    for invalid in [f64::NAN, f64::INFINITY, -1.0, 1.1] {
        assert!(loss().with_measurement(invalid).build().is_err());
        assert!(loss().with_prep(invalid).build().is_err());
        assert!(
            loss()
                .with_lost_partner_loss_probability(invalid)
                .build()
                .is_err()
        );
        assert!(
            loss()
                .add_recovery_channel(LossSite::BeforeTwoQubit, invalid, 0.0)
                .build()
                .unwrap_err()
                .to_string()
                .contains("channels[0].false_negative")
        );
        assert!(
            loss()
                .add_recovery_channel(LossSite::BeforeTwoQubit, 0.0, invalid)
                .build()
                .unwrap_err()
                .to_string()
                .contains("channels[0].false_positive")
        );
        assert!(
            loss()
                .add_events_channel(
                    LossSite::BeforeSingleQubit,
                    invalid,
                    &weights(&[("L", 1.0)])
                )
                .build()
                .is_err()
        );
    }
    let matrix = ReadoutMatrix::new(
        GateType::MeasureZLost,
        &BTreeMap::from([("lost".into(), weights(&[("3", 1.0)]))]),
    )
    .unwrap();
    assert!(
        builder(loss())
            .with_readout(matrix.clone())
            .try_build()
            .is_ok()
    );
    assert!(
        GeneralNoiseModel::builder()
            .with_readout(matrix)
            .try_build()
            .unwrap_err()
            .to_string()
            .contains("configure loss or remove the lost row")
    );
}

#[test]
fn t25_json_rust_parity_and_order_independent_composition() {
    let config = json!({"loss":{"enabled":true,"default_recovery":false,"measurement":0.3,"prep":0.2,"idle_rate":0.4,"reload_on_prep":true,"lost_partner":{"loss_probability":0.6,"pauli_model":{"I":0.5,"X":0.5}},"channels":[{"site":"before_two_qubit","kind":"events","probability":0.4,"events":{"LI":0.5,"IL":0.5}},{"site":"after_two_qubit","kind":"recovery","false_negative":0.2,"false_positive":0.1}]}});
    let l = loss()
        .with_measurement(0.3)
        .with_prep(0.2)
        .with_idle_rate(0.4)
        .with_reload_on_prep(true)
        .with_lost_partner_loss_probability(0.6)
        .with_lost_partner_pauli_model(&weights(&[("I", 0.5), ("X", 0.5)]))
        .add_events_channel(
            LossSite::BeforeTwoQubit,
            0.4,
            &weights(&[("LI", 0.5), ("IL", 0.5)]),
        )
        .add_recovery_channel(LossSite::AfterTwoQubit, 0.2, 0.1)
        .build()
        .unwrap();
    let hook = TwoQubitPauliLeakageChannel::new(0.5, &weights(&[("ZZ", 1.0)])).unwrap();
    let common = GeneralNoiseModel::builder().with_seed(7919);
    let a = common
        .clone()
        .with_loss(l.clone())
        .add_p2_pauli_leakage_step_before_gate(&P2PauliLeakageStep::joint(hook.clone()));
    let b = common
        .clone()
        .add_p2_pauli_leakage_step_before_gate(&P2PauliLeakageStep::joint(hook.clone()))
        .with_loss(l);
    let c = GeneralNoiseConfig::from_json(&config.to_string())
        .unwrap()
        .apply_to(common)
        .unwrap()
        .add_p2_pauli_leakage_step_before_gate(&P2PauliLeakageStep::joint(hook.clone()));
    let gates = [
        Gate::pz(&[0, 1]),
        Gate::h(&[0]),
        lose(1),
        Gate::cx(&[(0, 1)]),
        Gate::mz(&[0]),
        Gate::mpz(&[1]),
        herald(0),
        herald(1),
    ];
    let expected = run(&mut a.build(), &gates);
    assert_eq!(run(&mut b.build(), &gates), expected);
    assert_eq!(run(&mut c.build(), &gates), expected);
}

#[test]
fn known_limitation_untracked_mx_outcomes_fail_count_check() {
    // Invert this witness when MX / MeasureFree outcomes are tracked by the model.
    for kind in [GateType::MX, GateType::MeasureFree] {
        let mut b = ByteMessage::quantum_operations_builder();
        b.add_gate_command(&m(kind, 0));
        let mut noise = GeneralNoiseModel::default();
        let EngineStage::NeedsProcessing(commands) = noise.start(b.build()).unwrap() else {
            panic!("measurement emitted")
        };
        let reply = StateVecEngine::with_seed(1, 65537)
            .process(commands)
            .unwrap();
        assert_eq!(reply.outcomes().unwrap().len(), 1);
        assert!(
            noise
                .continue_processing(reply)
                .err()
                .unwrap()
                .to_string()
                .contains("Mismatch in number of measurement outcomes")
        );
    }
}

#[test]
fn all_lost_selection_and_idle_families_skip_before_draws() {
    let event = PauliLeakageChannel::new(1.0, &weights(&[("L", 1.0)])).unwrap();
    let b = builder(
        loss()
            .with_idle_rate(1.0)
            .with_lost_partner_loss_probability(0.0),
    )
    .with_p1(1.0)
    .with_p2(1.0)
    .with_p1_emission_ratio(1.0)
    .with_p2_emission_ratio(1.0)
    .add_p1_transition_channel_before_gate(&transition())
    .add_p1_transition_channel_after_gate(&transition())
    .add_p1_pauli_leakage_channel_before_gate(&event)
    .add_p1_pauli_leakage_channel_after_gate(&event)
    .add_p2_transition_step_before_gate(&P2TransitionStep::independent(transition(), transition()))
    .add_p2_transition_step_after_gate(&P2TransitionStep::independent(transition(), transition()))
    .add_p2_pauli_leakage_channel_before_gate(&event)
    .add_p2_pauli_leakage_channel_after_gate(&event)
    .with_p_idle_linear(1.0, &weights(&[("X", 1.0)]))
    .with_p_idle_sin_squared(1.0, &weights(&[("Z", 1.0)]))
    .with_p_idle_coherent(1.0, &weights(&[("RX", 1.0)]));
    let gates = [Gate::x(&[0]), Gate::cx(&[(0, 1)]), lose(0)];
    for noiseless in [false, true] {
        let mut model = b.clone().build();
        run(&mut model, &[lose(0), lose(1)]);
        if noiseless {
            model.add_noiseless_gate(GateType::X);
            model.add_noiseless_gate(GateType::CX);
        }
        let before = tail(&model);
        let result = run(&mut model, &gates);
        assert_eq!(result.1, Vec::<Gate>::new());
        assert_eq!(
            result.2, before,
            "baseline after both loss events, before suppressed operations"
        );
    }
}

#[test]
fn t24_structural_loss_errors_share_parser() {
    assert!(
        QubitLoss::from_json_value(&json!({"typo":0}))
            .unwrap_err()
            .to_string()
            .contains("loss.enabled is required")
    );

    for value in [json!(null), json!([]), json!(true), json!("loss")] {
        let config = GeneralNoiseConfig::from_json(&json!({"loss":value}).to_string()).unwrap();
        assert_eq!(
            config.build().unwrap_err().to_string(),
            format!("Input error: loss = {value}: use a JSON object")
        );
    }
    for field in ["enabled", "reload_on_prep", "default_recovery"] {
        for value in [json!(null), json!(1), json!("true")] {
            let mut loss = json!({"enabled":true});
            loss[field] = value;
            assert!(
                QubitLoss::from_json_value(&loss)
                    .unwrap_err()
                    .to_string()
                    .contains(&format!("loss.{field}"))
            );
        }
    }
    for (kind, fields) in [
        ("events", vec!["site", "probability", "events"]),
        ("recovery", vec!["site", "false_negative", "false_positive"]),
    ] {
        let base = if kind == "events" {
            json!({"kind":kind,"site":"before_single_qubit","probability":1,"events":{"L":1}})
        } else {
            json!({"kind":kind,"site":"before_single_qubit","false_negative":0,"false_positive":0})
        };
        for field in fields {
            let mut missing = base.clone();
            missing.as_object_mut().unwrap().remove(field);
            let err = QubitLoss::from_json_value(&json!({"enabled":true,"channels":[missing]}))
                .unwrap_err()
                .to_string();
            assert!(
                err.contains(&format!("loss.channels[0].{field} is required")),
                "{err}"
            );
            let mut wrong = base.clone();
            wrong[field] = json!(false);
            let err = QubitLoss::from_json_value(&json!({"enabled":true,"channels":[wrong]}))
                .unwrap_err()
                .to_string();
            assert!(
                err.contains(&format!("loss.channels[0].{field}")) && err.contains("false"),
                "{err}"
            );
        }
    }
    for value in [f64::NAN, f64::INFINITY] {
        assert!(loss().with_idle_rate(value).build().is_err());
    }
    assert!(loss().with_idle_rate(2.0).build().is_ok());
}

#[test]
fn command_flags_do_not_leak_into_next_gate() {
    let mut model =
        builder(loss().add_events_channel(LossSite::BeforeTwoQubit, 1.0, &weights(&[("LI", 1.0)])))
            .build();
    let result = run(
        &mut model,
        &[
            Gate::cx(&[(0, 1)]),
            Gate::cx(&[(0, 1)]),
            herald(0),
            herald(1),
        ],
    );
    assert_eq!(
        result.0,
        [3, 3],
        "the second gate sees a previously lost leg and must spread"
    );
    for single in [false, true] {
        let l = loss()
            .with_idle_rate(1.0)
            .with_lost_partner_loss_probability(0.0)
            .add_recovery_channel(
                if single {
                    LossSite::AfterSingleQubit
                } else {
                    LossSite::AfterTwoQubit
                },
                0.0,
                0.0,
            );
        let b = builder(l);
        let b = if single {
            b.add_p1_transition_channel_after_gate(&transition())
                .add_p1_pauli_leakage_channel_after_gate(
                    &PauliLeakageChannel::new(1.0, &weights(&[("Z", 1.0)])).unwrap(),
                )
        } else {
            b.add_p2_transition_step_after_gate(&P2TransitionStep::independent(
                transition(),
                transition(),
            ))
            .add_p2_pauli_leakage_step_after_gate(&P2PauliLeakageStep::joint(
                TwoQubitPauliLeakageChannel::new(1.0, &weights(&[("ZZ", 1.0)])).unwrap(),
            ))
        };
        let batched = if single {
            Gate::h(&[0, 2])
        } else {
            Gate::cx(&[(0, 1), (2, 3)])
        };
        let next = if single {
            Gate::h(&[0])
        } else {
            Gate::cx(&[(0, 1)])
        };
        let result = run(&mut b.build(), &[lose(0), batched, next]);
        assert_eq!(
            count(&result.1, GateType::Z, 0),
            1,
            "suppression survives the first resume but clears before the next command"
        );
    }
}

#[test]
fn t01_t22_lost_readout_exact_draw_counts() {
    for kind in [
        GateType::MZ,
        GateType::MPZ,
        GateType::MeasureZLeaked,
        GateType::MeasureZLost,
        GateType::MeasureZLeakedLost,
    ] {
        for noiseless in [false, true] {
            let mut b = builder(loss().with_idle_rate(1.0));
            if noiseless {
                b = b.with_noiseless_gate(if kind == GateType::MPZ {
                    GateType::MZ
                } else {
                    kind
                });
            }
            let mut model = b.build();
            run(&mut model, &[lose(0)]);
            let mut expected = NoiseRng::new(model.rng().clone());
            if !noiseless
                && matches!(
                    kind,
                    GateType::MZ | GateType::MPZ | GateType::MeasureZLeaked
                )
            {
                expected.occurs(0.0);
            }
            assert_eq!(
                run(&mut model, &[m(kind, 0)]).2,
                std::array::from_fn(|_| expected.inner_mut().next_u64()),
                "baseline immediately before {kind}: one zero-probability flip for a noisy dark readout, otherwise no draw"
            );
        }
    }
    let matrix = ReadoutMatrix::new(
        GateType::MeasureZLost,
        &BTreeMap::from([("lost".into(), weights(&[("3", 1.0)]))]),
    )
    .unwrap();
    let mut model = builder(loss().with_idle_rate(1.0))
        .with_readout(matrix)
        .build();
    run(&mut model, &[lose(0)]);
    let mut expected = NoiseRng::new(model.rng().clone());
    let _ = pecos_engines::noise::WeightedSampler::new(&BTreeMap::from([(3_u8, 1.0)]))
        .sample(&mut expected);
    let result = run(&mut model, &[herald(0)]);
    assert_eq!(result.0, [3]);
    assert_eq!(
        result.2,
        std::array::from_fn(|_| expected.inner_mut().next_u64()),
        "baseline immediately before configured lost readout, plus one categorical draw"
    );
}

#[test]
fn scaled_loss_json_error_has_one_input_prefix() {
    let error = GeneralNoiseConfig::from_json(
        r#"{"loss":{"enabled":true,"measurement":0.6},"scaling":{"overall":2}}"#,
    )
    .unwrap()
    .build()
    .unwrap_err()
    .to_string();
    assert_eq!(
        error,
        "Input error: loss.measurement after scaling = 1.2 (0.6 x scale 2): reduce the probability or scale factors to keep the scaled probability finite in [0, 1]"
    );
}

#[test]
fn loss_object_errors_name_exact_paths() {
    for (value, expected) in [
        (json!(null), "Input error: loss = null: use a JSON object"),
        (
            json!({"enabled":true,"lost_partner":1}),
            "Input error: loss.lost_partner = 1: use a JSON object",
        ),
        (
            json!({"enabled":true,"channels":[1]}),
            "Input error: loss.channels[0] = 1: use a channel object",
        ),
    ] {
        assert_eq!(
            QubitLoss::from_json_value(&value).unwrap_err().to_string(),
            expected
        );
    }
}

#[test]
fn zero_events_do_not_draw() {
    check_zero_events(0.0, 1.0);
}

#[test]
fn scaled_zero_events_do_not_draw() {
    check_zero_events(0.5, 0.0);
}

fn check_zero_events(probability: f64, scale: f64) {
    for site in [
        LossSite::BeforeSingleQubit,
        LossSite::AfterSingleQubit,
        LossSite::BeforeTwoQubit,
        LossSite::AfterTwoQubit,
    ] {
        let single = matches!(
            site,
            LossSite::BeforeSingleQubit | LossSite::AfterSingleQubit
        );
        let gate = if single {
            Gate::h(&[0, 1])
        } else {
            Gate::cx(&[(0, 1), (2, 3)])
        };
        let config = loss().add_events_channel(
            site,
            probability,
            &weights(&[(if single { "L" } else { "LI" }, 1.0)]),
        );
        let make = |config| builder(config).with_scale(scale).build();
        assert_eq!(
            run(&mut make(config), std::slice::from_ref(&gate)),
            run(&mut make(loss()), &[gate]),
            "baseline without events channel: {site:?}, {probability}, scale {scale}"
        );
    }
}

#[test]
fn lost_mpz_without_reload_skips_prep_loss_draw() {
    let make = |prep| builder(loss().with_idle_rate(1.0).with_prep(prep)).build();
    let gates = [lose(0), Gate::mpz(&[0])];
    assert_eq!(
        run(&mut make(0.5), &gates),
        run(&mut make(0.0), &gates),
        "baseline prep = 0 on the same lost MPZ"
    );
}

#[test]
fn noiseless_untracked_measurements_and_custom_drop_lost_operands() {
    for gate in [
        m(GateType::MX, 0),
        m(GateType::MeasureFree, 0),
        Gate::simple(GateType::Custom, vec![0.into(), 1.into()]),
    ] {
        let b = builder(loss().with_idle_rate(1.0)).with_noiseless_gate(gate.gate_type);
        assert_eq!(
            run(&mut b.clone().build(), &[lose(0), gate]),
            run(&mut b.build(), &[lose(0)]),
            "baseline without the noiseless lost operation"
        );
    }
}

#[test]
fn measurement_loss_draws_in_operand_order() {
    // Several seeds distinguish missing, duplicated and reversed operand draws.
    for seed in 0..16 {
        let mut model = builder(loss().with_measurement(0.5))
            .with_seed(seed)
            .build();
        let mut expected_rng = NoiseRng::new(model.rng().clone());
        let lost: Vec<_> = [0, 1]
            .into_iter()
            .filter(|_| expected_rng.occurs(0.5))
            .collect();
        let mut expected_stream = vec![Gate::mz(&[0, 1])];
        expected_stream.extend(lost.iter().map(|q| Gate::pz(&[*q])));
        // Present readout makes one default flip draw for each operand.
        expected_rng.occurs(0.0);
        expected_rng.occurs(0.0);
        let expected_tail: [u64; 4] = std::array::from_fn(|_| expected_rng.inner_mut().next_u64());
        let result = run(&mut model, &[Gate::mz(&[0, 1])]);
        assert_eq!(result.1, expected_stream, "seed {seed}");
        assert_eq!(
            result.2, expected_tail,
            "baseline clone with two loss draws then two readout draws"
        );
    }
}

#[test]
fn loss_unknown_keys_preserve_literal_names() {
    for (value, expected) in [
        (
            json!({"enabled":true,"":0}),
            r#"Input error: loss."" = 0: remove this unknown key"#,
        ),
        (
            json!({"enabled":true,"lost_partner":{"":0}}),
            r#"Input error: loss.lost_partner."" = 0: remove this unknown key"#,
        ),
        (
            json!({"enabled":true,"typo.":0}),
            "Input error: loss.typo. = 0: remove this unknown key",
        ),
        (
            json!({"enabled":false,"":0}),
            r#"Input error: loss."" is set but loss.enabled is false"#,
        ),
    ] {
        assert_eq!(
            QubitLoss::from_json_value(&value).unwrap_err().to_string(),
            expected
        );
    }
}
