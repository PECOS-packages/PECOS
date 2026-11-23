//! Detect gates are transparent to DEMs and fault mechanisms, including idle filling.
use pecos_core::Gate;
use pecos_core::gate_type::GateType;
use pecos_qec::fault_tolerance::dem_builder::{DemBuilder, NoiseConfig, PerGateTypeNoise};
use pecos_qec::fault_tolerance::fault_sampler::{StochasticNoiseParams, build_fault_table};
use pecos_qec::fault_tolerance::propagator::{DagFaultAnalyzer, GateNoiseKind, gate_noise_kind};
use pecos_quantum::{Attribute, DagCircuit, TickCircuit};

const FLAVORS: [GateType; 3] = [
    GateType::DetectLeaked,
    GateType::DetectLost,
    GateType::DetectLeakedLost,
];
fn memory(detects: bool, mixed: bool, detect_qubits: &[usize]) -> TickCircuit {
    let mut tc = TickCircuit::new();
    tc.tick().pz(&[0, 1, 2]);
    tc.tick().h(&[0]);
    for flavor in FLAVORS {
        if mixed {
            let mut tick = tc.tick();
            tick.x(&[2]);
            if detects {
                tick.try_add_gate(Gate::simple(
                    flavor,
                    detect_qubits
                        .iter()
                        .copied()
                        .map(Into::into)
                        .collect::<Vec<_>>(),
                ))
                .unwrap();
            }
        } else {
            tc.tick().x(&[2]);
            if detects {
                tc.tick()
                    .try_add_gate(Gate::simple(
                        flavor,
                        detect_qubits
                            .iter()
                            .copied()
                            .map(Into::into)
                            .collect::<Vec<_>>(),
                    ))
                    .unwrap();
            }
        }
    }
    tc.tick().cx(&[(0, 1)]);
    tc.tick().cx(&[(0, 1)]);
    tc.tick().h(&[0]);
    tc.tick().mz(&[0, 1]);
    tc.set_meta(
        "detectors",
        Attribute::String(r#"[{"id":0,"records":[-2]}]"#.into()),
    );
    tc.set_meta(
        "observables",
        Attribute::String(r#"[{"id":0,"records":[-1]}]"#.into()),
    );
    tc
}
fn fault_effects(tc: &TickCircuit) -> Vec<(Vec<usize>, u64)> {
    // Each final measurement maps directly to one detector or observable here.
    let noise = StochasticNoiseParams {
        p1: 0.03,
        p2: 0.04,
        p_meas: 0.05,
        p_prep: 0.06,
    };
    let mut effects = vec![];
    for mechanism in build_fault_table(tc, &noise).unwrap() {
        if mechanism.probability == 0.0 {
            continue;
        }
        let p =
            mechanism.probability / f64::from(u32::try_from(mechanism.alternatives.len()).unwrap());
        for effect in mechanism.alternatives {
            effects.push((effect, p.to_bits()));
        }
    }
    effects.sort();
    effects
}
fn inline_p1_dem(tc: &TickCircuit) -> String {
    // Exercise the classification used by gate-triggered noise compilation,
    // independently of the influence-map builder's supported-no-op predicate.
    let noisy = tc
        .try_with_noise(&|gate: &Gate| -> Vec<pecos_core::ChannelExpr> {
            if gate_noise_kind(gate.gate_type) == GateNoiseKind::Single {
                gate.qubits
                    .iter()
                    .map(|qubit| pecos_core::channel::Depolarizing(0.03, qubit.index()))
                    .collect()
            } else {
                vec![]
            }
        })
        .unwrap();
    DemBuilder::try_from_tick_circuit_with_noise_config(
        &noisy,
        NoiseConfig::new(0.0, 0.0, 0.0, 0.0),
    )
    .unwrap()
    .to_string()
}
#[test]
fn detection_dem_and_fault_sampler_identity_oracle() {
    for mixed in [false, true] {
        let base = memory(false, mixed, &[1, 0]);
        let with_detect = memory(true, mixed, &[1, 0]);
        let noise = NoiseConfig::new(0.03, 0.04, 0.05, 0.06);
        let expected = DemBuilder::try_from_tick_circuit_with_noise_config(&base, noise.clone())
            .unwrap()
            .to_string();
        assert!(expected.contains("error("));
        assert_eq!(
            DemBuilder::try_from_tick_circuit_with_noise_config(&with_detect, noise.clone())
                .unwrap()
                .to_string(),
            expected
        );
        let baseline_dag = DagCircuit::try_from(&base).unwrap();
        let detect_dag = DagCircuit::try_from(&with_detect).unwrap();
        assert_eq!(
            DemBuilder::try_from_circuit_with_noise_config(&detect_dag, noise.clone())
                .unwrap()
                .to_string(),
            DemBuilder::try_from_circuit_with_noise_config(&baseline_dag, noise)
                .unwrap()
                .to_string()
        );
        assert_eq!(fault_effects(&with_detect), fault_effects(&base));
        assert_eq!(inline_p1_dem(&with_detect), inline_p1_dem(&base));
        assert_eq!(
            fault_effects(&TickCircuit::from(&detect_dag)),
            fault_effects(&TickCircuit::from(&baseline_dag))
        );
    }
}
#[test]
fn detection_dem_identity_after_idle_filling() {
    let mut base = memory(false, false, &[1]);
    let mut with_detect = memory(true, false, &[1]);
    // Detect only qubit 1: a spurious idle on measured qubit 0 changes the detector.
    base.fill_idle_gates();
    with_detect.fill_idle_gates();
    let noise = NoiseConfig::with_idle(0.03, 0.04, 0.05, 0.06, 0.07);
    assert_eq!(
        DemBuilder::try_from_tick_circuit_with_noise_config(&with_detect, noise.clone())
            .unwrap()
            .to_string(),
        DemBuilder::try_from_tick_circuit_with_noise_config(&base, noise)
            .unwrap()
            .to_string()
    );
}
#[test]
fn detection_nonzero_per_gate_rates_rejected_but_zero_rates_pass() {
    let dag = DagCircuit::try_from(&memory(true, false, &[1, 0])).unwrap();
    let influence = DagFaultAnalyzer::new(&dag).build_influence_map().unwrap();
    for flavor in FLAVORS {
        for table in 0..3 {
            for rate in [0.0, 0.1] {
                let mut noise = PerGateTypeNoise::from_base_noise(NoiseConfig::default());
                match table {
                    0 => {
                        noise.base.p1_gate_rates.insert(flavor, rate);
                    }
                    1 => {
                        noise.rates_1q.insert(flavor, [rate, 0.0, 0.0]);
                    }
                    _ => {
                        noise
                            .rates_1q_per_qubit
                            .insert((flavor, 0.into()), [rate, 0.0, 0.0]);
                    }
                }
                let result = DemBuilder::new(&influence)
                    .with_per_gate_noise(noise)
                    .try_build();
                if rate == 0.0 {
                    result.unwrap();
                } else {
                    let error = result.unwrap_err().to_string();
                    assert!(
                        error.contains(&format!("{flavor} has no noise; remove the per-gate rate")),
                        "{error}"
                    );
                }
            }
        }
    }
}
