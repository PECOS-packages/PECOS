use super::*;
use crate::fault_tolerance::fault_sampler::symbolic_measurement_history;
use crate::{DemBuilder, DetectorErrorModel, NoiseConfig};
use pecos_core::{BitSet, Gate, Pauli, gate_type::GateType};
use pecos_engines::noise::GeneralNoiseModel;
use pecos_engines::noise::general::{HookDescription, HookProbabilities, LossSite, QubitLoss};
use pecos_engines::quantum::SparseStabEngine;
use pecos_engines::{ByteMessage, Engine, EngineSystem, QuantumSystem};
use pecos_quantum::TickCircuit;
use std::collections::BTreeMap;

fn gate(circuit: &mut TickCircuit, kind: GateType, qubits: &[usize]) {
    circuit
        .tick()
        .try_add_gate(Gate::simple(
            kind,
            qubits.iter().copied().map(Into::into).collect::<Vec<_>>(),
        ))
        .unwrap();
}
fn model(loss: QubitLoss) -> GeneralNoiseModel {
    GeneralNoiseModel::builder().with_loss(loss).build()
}
fn location(tick: usize, side: Side) -> Location {
    Location {
        tick,
        batch: 0,
        operand: 0,
        side,
    }
}
fn channel(tick: usize, side: Side, label: &str, p: f64) -> LocationChannel {
    LocationChannel {
        location: location(tick, side),
        probabilities: BTreeMap::from([(label.into(), p)]),
    }
}
fn measured(circuit: &mut TickCircuit, qs: &[usize]) {
    let records = circuit.tick().mz(qs);
    for record in records {
        circuit.detector(&[record]).unwrap();
    }
}
fn bell() -> TickCircuit {
    let mut c = TickCircuit::new();
    c.tick().pz(&[0, 1]);
    c.tick().h(&[0, 1]);
    c.tick().cz(&[(0, 1)]);
    c.tick().h(&[1]);
    c.tick().cx(&[(0, 1)]);
    c.tick().h(&[0]);
    measured(&mut c, &[0, 1]);
    c
}
fn distribution(dem: &DetectorErrorModel) -> Vec<f64> {
    let mut cells = vec![0.0; 1 << dem.num_detectors()];
    cells[0] = 1.0;
    for (p, detectors, _) in dem.to_mechanisms().0 {
        let mask = detectors.into_iter().fold(0, |mask, bit| mask ^ (1 << bit));
        let previous = cells.clone();
        for (cell, value) in cells.iter_mut().enumerate() {
            *value = previous[cell] * (1.0 - p) + previous[cell ^ mask] * p;
        }
    }
    cells
}
fn assert_engine_distribution(
    circuit: &TickCircuit,
    noise: GeneralNoiseModel,
    dem: &DetectorErrorModel,
) {
    assert_engine_cells(circuit, noise, &distribution(dem));
}

fn assert_engine_cells(circuit: &TickCircuit, noise: GeneralNoiseModel, analytic: &[f64]) {
    let mut counts = vec![0_u64; analytic.len()];
    let mut runner = TickRunner::new(circuit, noise, 824_109).unwrap();
    let n = 100_000;
    for _ in 0..n {
        let shot = runner.shot().unwrap();
        let cell = shot
            .detectors
            .into_iter()
            .enumerate()
            .fold(0, |cell, (bit, value)| cell | (usize::from(value) << bit));
        counts[cell] += 1;
    }
    let alpha = 0.001 / f64::from(u32::try_from(analytic.len()).unwrap());
    for (cell, (&count, &p)) in counts.iter().zip(analytic).enumerate() {
        let interval = pecos_num::stats::jeffreys_interval(count, n, alpha).unwrap();
        assert!(
            p >= interval.lo && p <= interval.hi,
            "cell {cell}: analytic {p}, count {count}, interval {interval:?}"
        );
    }
}

#[test]
fn probe_alignment_and_heralds() {
    let mut c = TickCircuit::new();
    c.tick().pz(&[0, 1, 2, 3, 4, 5, 6]);
    c.tick().x(&[3, 4, 5]);
    c.tick().idle(1, &[1]);
    let mut tick = c.tick();
    for (kind, q) in [
        (GateType::DetectLost, 0),
        (GateType::DetectLeakedLost, 1),
        (GateType::DetectLost, 2),
    ] {
        tick.try_add_gate(Gate::simple(kind, vec![q.into()]))
            .unwrap();
    }
    let mut tick = c.tick();
    tick.try_add_gate(Gate::simple(GateType::MZ, vec![3.into()]))
        .unwrap();
    tick.try_add_gate(Gate::simple(GateType::MPZ, vec![4.into()]))
        .unwrap();
    tick.try_add_gate(Gate::simple(GateType::MeasureZLost, vec![5.into()]))
        .unwrap();
    tick.try_add_gate(Gate::simple(GateType::DetectLeaked, vec![6.into()]))
        .unwrap();
    gate(&mut c, GateType::DetectLost, &[2, 0]);
    gate(&mut c, GateType::DetectLost, &[1]);
    c.tick().mz(&[1, 0, 2]);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(1.0)
            .build()
            .unwrap(),
    );
    let mut runner = TickRunner::new(&c, noise, 9).unwrap();
    assert_eq!(
        runner
            .table()
            .probes
            .iter()
            .map(|p| p.qubit.index())
            .collect::<Vec<_>>(),
        [0, 2, 1, 2, 0, 1]
    );
    assert_eq!(
        runner
            .table()
            .probes
            .iter()
            .map(|p| p.next_record)
            .collect::<Vec<_>>(),
        [Some(3), Some(4), Some(2), Some(4), Some(3), Some(2)]
    );
    let shot = runner.shot().unwrap();
    assert_eq!(shot.records, [1, 1, 0, 0, 0]);
    assert_eq!(shot.flags, [false, false, true, false, false, true]);
}

#[test]
fn runner_matches_stream_and_resets() {
    let mut c = TickCircuit::new();
    gate(&mut c, GateType::DetectLost, &[0]);
    c.tick().idle(1, &[0]);
    gate(&mut c, GateType::DetectLost, &[0]);
    c.tick().pz(&[1]);
    c.tick().x(&[1]);
    let mut tick = c.tick();
    tick.try_add_gate(Gate::simple(GateType::MZ, vec![1.into()]))
        .unwrap();
    tick.try_add_gate(Gate::simple(GateType::DetectLost, vec![2.into()]))
        .unwrap();
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(1.0)
            .build()
            .unwrap(),
    );
    let mut runner = TickRunner::new(&c, noise.clone(), 82).unwrap();
    let mut direct = QuantumSystem::new(
        Box::new(noise),
        Box::new(SparseStabEngine::with_seed(3, 82)),
    );
    direct.set_seed(82);
    let mut message = ByteMessage::quantum_operations_builder();
    for g in c.iter_gate_batches() {
        message.add_gate_command(g.as_gate());
    }
    let message = message.build();
    for _ in 0..3 {
        direct.reset().unwrap();
        let result = direct.process_as_system(message.clone()).unwrap();
        let values = result.outcomes().unwrap();
        assert_eq!(values, [0, 3, 1, 0]);
        let shot = runner.shot().unwrap();
        assert_eq!(shot.flags, [false, true, false]);
        assert_eq!(shot.records, [1]);
    }
}

#[test]
fn bridge_conversion_and_rejections() {
    let mut c = TickCircuit::new();
    c.tick().pz(&[0, 1]);
    c.tick().h(&[0]);
    c.tick().cx(&[(0, 1)]);
    c.tick().idle(2, &[0, 1]);
    measured(&mut c, &[0, 1]);
    let axes = BTreeMap::from([("X".into(), 1.0)]);
    let noise = GeneralNoiseModel::builder()
        .with_p1(0.03)
        .with_p2(0.04)
        .with_p_prep(0.05)
        .with_p_meas_0(0.06)
        .with_p_meas_1(0.06)
        .with_p_idle_linear(0.02, &axes)
        .build();
    let config = pauli_noise_config(&c, &noise.description()).unwrap();
    let mut expected = NoiseConfig::new(0.03, 0.04, 0.06, 0.05);
    expected.p_idle_linear = crate::IdleNoiseFamily::new(0.02, axes);
    assert_eq!(
        DemBuilder::try_from_tick_circuit_with_noise_config(&c, config)
            .unwrap()
            .to_string(),
        DemBuilder::try_from_tick_circuit_with_noise_config(&c, expected)
            .unwrap()
            .to_string()
    );
    let base = noise.description();
    macro_rules! rejected {
        ($field:ident, $value:expr) => {{
            let mut d = base.clone();
            d.$field = $value;
            let error = pauli_noise_config(&c, &d).unwrap_err().to_string();
            assert!(error.contains(stringify!($field)), "{error}");
            assert!(error.contains("not supported by the loss-conditioned DEM"));
        }};
    }
    rejected!(p_meas_1, 0.07);
    rejected!(leakage_scale, 0.5);
    rejected!(p_prep_leak_ratio, 0.1);
    rejected!(p_prep_crosstalk, 0.1);
    rejected!(p1_emission_ratio, 0.1);
    rejected!(p2_emission_ratio, 0.1);
    rejected!(p1_seepage_prob, 0.1);
    rejected!(p2_seepage_prob, 0.1);
    rejected!(p_idle_coherent_rate, 0.1);
    rejected!(p_idle_sin_squared_rate, 0.1);
    rejected!(p_meas_crosstalk_local, 0.1);
    rejected!(p_meas_crosstalk_global, 0.1);
    rejected!(p2_angle_a, 0.1);
    rejected!(p2_angle_b, 0.1);
    rejected!(p2_angle_c, 0.1);
    rejected!(p2_angle_d, 0.1);
    rejected!(p2_angle_power, 0.1);
    rejected!(p1_transition_channels_before_gate, vec![0.0]);
    rejected!(p1_pauli_leakage_channels_after_gate, vec![0.0]);
    rejected!(
        p_idle_sin_squared_model,
        BTreeMap::from([("X".into(), 1.0)])
    );
    rejected!(p_idle_coherent_model, BTreeMap::new());
    rejected!(p1_emission_model, BTreeMap::from([("Z".into(), 1.0)]));
    rejected!(p2_emission_model, BTreeMap::from([("ZZ".into(), 1.0)]));
    rejected!(p_meas_crosstalk_model, [BTreeMap::new(), BTreeMap::new()]);
    rejected!(p1_transition_channels_after_gate, vec![0.0]);
    rejected!(p1_pauli_leakage_channels_before_gate, vec![0.0]);
    rejected!(
        p2_transition_steps_before_gate,
        vec![HookProbabilities::Joint(0.0)]
    );
    rejected!(
        p2_transition_steps_after_gate,
        vec![HookProbabilities::Joint(0.0)]
    );
    rejected!(
        p2_pauli_leakage_steps_before_gate,
        vec![HookProbabilities::Joint(0.0)]
    );
    rejected!(
        p2_pauli_leakage_steps_after_gate,
        vec![HookProbabilities::Joint(0.0)]
    );
    rejected!(
        p1_hook_order_before_gate,
        vec![HookDescription::Transition(0)]
    );
    rejected!(
        p1_hook_order_after_gate,
        vec![HookDescription::Transition(0)]
    );
    rejected!(
        p2_hook_order_before_gate,
        vec![HookDescription::Transition(0)]
    );
    rejected!(
        p2_hook_order_after_gate,
        vec![HookDescription::Transition(0)]
    );
    rejected!(idle_after_2q, 1.0);
    let mut d = base.clone();
    d.readout.insert(GateType::MZ, BTreeMap::new());
    assert!(
        pauli_noise_config(&c, &d)
            .unwrap_err()
            .to_string()
            .contains("readout.MZ")
    );
    for kind in [GateType::MZ, GateType::PZ, GateType::MPZ, GateType::Idle] {
        let mut d = base.clone();
        d.noiseless_gates.insert(kind);
        assert!(
            pauli_noise_config(&c, &d)
                .unwrap_err()
                .to_string()
                .contains("noiseless_gates")
        );
    }
}

#[test]
fn measurement_noise_repeated_record_reproduction() {
    let mut c = TickCircuit::new();
    c.tick().pz(&[0]);
    measured(&mut c, &[0]);
    measured(&mut c, &[0]);
    let noise = GeneralNoiseModel::builder()
        .with_p_meas_0(1.0)
        .with_p_meas_1(1.0)
        .build();
    let mut runner = TickRunner::new(&c, noise.clone(), 9).unwrap();
    assert_eq!(runner.shot().unwrap().records, [1, 1]);
    let dem = DemBuilder::try_from_tick_circuit(&c, 0.0, 0.0, 1.0, 0.0).unwrap();
    assert_eq!(distribution(&dem), [0.0, 1.0, 0.0, 0.0]);
    let error = pauli_noise_config(&c, &noise.description())
        .unwrap_err()
        .to_string();
    assert!(error.contains("records 0 and 1"), "{error}");
}

#[test]
fn no_loss_and_zero_loss_are_byte_identical() {
    let mut single = TickCircuit::new();
    single.tick().pz(&[0]);
    single.tick().sz(&[0]);
    measured(&mut single, &[0]);
    let mut pair = TickCircuit::new();
    pair.tick().pz(&[0, 1]);
    pair.tick().cx(&[(0, 1)]);
    measured(&mut pair, &[0]);
    for c in [bell(), single, pair] {
        // Both divide-by-3 and divide-by-15 rounding must match the handwritten defaults.
        for rate in [0.01, 0.011, 0.017, 0.031] {
            let expected = DemBuilder::try_from_tick_circuit(&c, rate, rate, 0.0, 0.0).unwrap();
            for loss in [
                None,
                Some(
                    QubitLoss::builder()
                        .with_default_recovery(false)
                        .build()
                        .unwrap(),
                ),
            ] {
                let mut builder = GeneralNoiseModel::builder().with_p1(rate).with_p2(rate);
                if let Some(loss) = loss {
                    builder = builder.with_loss(loss);
                }
                let noise = builder.build();
                let actual = background_dem(&c, &noise).unwrap();
                assert_eq!(actual.to_string(), expected.to_string());
                assert_eq!(actual.to_mechanisms(), expected.to_mechanisms());
                assert_eq!(
                    actual.idle_noise_residuals(),
                    expected.idle_noise_residuals()
                );
            }
        }
    }
}

#[test]
fn recovery_before_two_qubit_engine_oracle() {
    let c = bell();
    let noise = GeneralNoiseModel::builder()
        .with_noiseless_gate(GateType::CZ)
        .with_loss(
            QubitLoss::builder()
                .with_default_recovery(false)
                .add_recovery_channel(LossSite::BeforeTwoQubit, 0.0, 0.2)
                .build()
                .unwrap(),
        )
        .build();
    let dem = background_dem(&c, &noise).unwrap();
    let cells = distribution(&dem);
    for (&actual, expected) in cells.iter().zip([0.73, 0.09, 0.09, 0.09]) {
        assert!((actual - expected).abs() < 1e-12);
    }
    assert_engine_distribution(&c, noise, &dem);
}

#[test]
fn events_before_two_qubit_engine_oracle() {
    let c = bell();
    let noise = GeneralNoiseModel::builder()
        .with_noiseless_gate(GateType::CZ)
        .with_loss(
            QubitLoss::builder()
                .with_default_recovery(false)
                .add_events_channel(
                    LossSite::BeforeTwoQubit,
                    0.2,
                    &BTreeMap::from([("XI".into(), 0.5), ("ZZ".into(), 0.5)]),
                )
                .build()
                .unwrap(),
        )
        .build();
    let dem = background_dem(&c, &noise).unwrap();
    assert_engine_distribution(&c, noise, &dem);
}

#[test]
fn single_qubit_events_conjugate_and_recovery_sites() {
    let mut c = TickCircuit::new();
    c.tick().pz(&[0]);
    c.tick().sy(&[0]);
    c.tick().h(&[0]);
    measured(&mut c, &[0]);
    let noise = GeneralNoiseModel::builder()
        .with_noiseless_gate(GateType::SY)
        .with_loss(
            QubitLoss::builder()
                .with_default_recovery(false)
                .add_events_channel(
                    LossSite::BeforeSingleQubit,
                    0.2,
                    &BTreeMap::from([("X".into(), 1.0)]),
                )
                .build()
                .unwrap(),
        )
        .build();
    let dem = background_dem(&c, &noise).unwrap();
    assert_eq!(distribution(&dem), [1.0, 0.0]);
    let mut runner = TickRunner::new(&c, noise, 10).unwrap();
    for _ in 0..1000 {
        assert_eq!(runner.shot().unwrap().records, [0]);
    }
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .add_recovery_channel(LossSite::AfterSingleQubit, 0.0, 0.2)
            .build()
            .unwrap(),
    );
    let dem = background_dem(&c, &noise).unwrap();
    let expected = DemBuilder::try_from_tick_circuit_with_location_channels(
        &c,
        NoiseConfig::new(0.0, 0.0, 0.0, 0.0),
        &[
            LocationChannel {
                location: location(1, Side::After),
                probabilities: BTreeMap::from([
                    ("X".into(), 0.05),
                    ("Y".into(), 0.05),
                    ("Z".into(), 0.05),
                ]),
            },
            LocationChannel {
                location: location(2, Side::After),
                probabilities: BTreeMap::from([
                    ("X".into(), 0.05),
                    ("Y".into(), 0.05),
                    ("Z".into(), 0.05),
                ]),
            },
        ],
    )
    .unwrap();
    assert_eq!(distribution(&dem), distribution(&expected));
    assert_engine_distribution(&c, noise, &dem);
}

#[test]
fn affine_random_cancellation() {
    let mut c = TickCircuit::new();
    c.tick().h(&[0]);
    c.tick().mz(&[0]);
    c.tick().x(&[0]);
    c.tick().mz(&[0]);
    let history = symbolic_measurement_history(&c).unwrap();
    let parity = affine_parity(&history, &[0, 1].into_iter().collect()).unwrap();
    assert!(parity.constant);
    assert!(parity.variables.is_empty());
    let single = affine_parity(&history, &BitSet::single(0)).unwrap();
    assert!(!single.constant);
    assert_eq!(single.variables, BitSet::single(0));
}

fn vectors(span: &Span) -> Vec<BitSet> {
    (0..1 << span.rank())
        .map(|mask| {
            let mut v = BitSet::new();
            for (i, row) in span.basis().iter().enumerate() {
                if mask & (1 << i) != 0 {
                    v ^= row;
                }
            }
            v.iter().collect()
        })
        .collect()
}
#[test]
fn gf2_against_bruteforce() {
    for width in 0..=6 {
        let all: Vec<BitSet> = (0..1 << width)
            .map(|mask| (0..width).filter(|bit| mask & (1 << bit) != 0).collect())
            .collect();
        for (a_index, a) in all.iter().enumerate() {
            for (b_index, b) in all.iter().enumerate() {
                let left = Span::new(width, &[a.clone(), b.clone()]).unwrap();
                let c = &all[(a_index * 13 + b_index * 7) % all.len()];
                let right = Span::new(width, &[b.clone(), c.clone()]).unwrap();
                let lv = vectors(&left);
                let rv = vectors(&right);
                let mut ab = a.clone();
                ab ^= b;
                let mut brute: Vec<BitSet> = [BitSet::new(), a.clone(), b.clone(), ab]
                    .into_iter()
                    .map(|v| v.iter().collect())
                    .collect();
                brute.sort_by_key(|v| v.iter().collect::<Vec<_>>());
                brute.dedup();
                assert_eq!(brute.len(), 1 << left.rank());
                for vector in &all {
                    assert_eq!(lv.contains(vector), brute.contains(vector));
                }
                let intersection = left.intersection(&right).unwrap();
                let iv = vectors(&intersection);
                for vector in &all {
                    assert_eq!(
                        iv.contains(vector),
                        lv.contains(vector) && rv.contains(vector)
                    );
                }
                let quotient = left.quotient(&right).unwrap();
                assert_eq!(quotient.width(), width);
                assert_eq!(quotient.rank() + intersection.rank(), left.rank());
                let representatives = vectors(&quotient);
                for vector in &lv {
                    assert!(representatives.iter().any(|rep| rv.iter().any(|r| {
                        let mut candidate = rep.clone();
                        candidate ^= r;
                        candidate.iter().eq(vector.iter())
                    })));
                }
            }
        }
    }
}

#[test]
fn located_injection_before_and_after() {
    let mut c = TickCircuit::new();
    c.tick().pz(&[0]);
    c.tick().h(&[0]);
    c.tick().h(&[0]);
    c.tick().mz(&[0]);
    for (side, expected) in [
        (Side::Before, BitSet::single(0)),
        (Side::After, BitSet::new()),
    ] {
        assert_eq!(
            located_pauli_flips(&c, Pauli::X, location(1, side)).unwrap(),
            expected
        );
        let mut explicit = TickCircuit::new();
        explicit.tick().pz(&[0]);
        if side == Side::Before {
            explicit.tick().x(&[0]);
        }
        explicit.tick().h(&[0]);
        if side == Side::After {
            explicit.tick().x(&[0]);
        }
        explicit.tick().h(&[0]);
        explicit.tick().mz(&[0]);
        let records = TickRunner::new(&explicit, GeneralNoiseModel::default(), 5)
            .unwrap()
            .shot()
            .unwrap()
            .records;
        assert_eq!(records[0] != 0, expected.contains(0));
    }
}

fn assert_mechanisms_close(actual: &DetectorErrorModel, expected: &DetectorErrorModel) {
    let signatures = |dem: &DetectorErrorModel| {
        dem.to_mechanisms()
            .0
            .into_iter()
            .map(|(probability, detectors, observables)| ((detectors, observables), probability))
            .collect::<BTreeMap<_, _>>()
    };
    let actual = signatures(actual);
    let expected = signatures(expected);
    assert_eq!(
        actual.keys().collect::<Vec<_>>(),
        expected.keys().collect::<Vec<_>>()
    );
    for (signature, probability) in actual {
        assert!(
            (probability - expected[&signature]).abs() <= 1e-12,
            "signature {signature:?}: {probability} versus {}",
            expected[&signature]
        );
    }
}

#[test]
fn location_channels_match_inline_and_existing_influence() {
    let mut c = TickCircuit::new();
    c.tick().pz(&[0, 1]);
    c.tick().cx(&[(0, 1)]);
    measured(&mut c, &[0, 1]);
    let before = DemBuilder::try_from_tick_circuit_with_location_channels(
        &c,
        NoiseConfig::new(0.0, 0.0, 0.0, 0.0),
        &[channel(1, Side::Before, "X", 0.2)],
    )
    .unwrap();
    let after = DemBuilder::try_from_tick_circuit_with_location_channels(
        &c,
        NoiseConfig::new(0.0, 0.0, 0.0, 0.0),
        &[channel(1, Side::After, "X", 0.2)],
    )
    .unwrap();
    assert_eq!(distribution(&before), [0.8, 0.0, 0.0, 0.2]);
    assert_eq!(distribution(&after), [0.8, 0.2, 0.0, 0.0]);
    let mut inline = TickCircuit::new();
    inline.tick().pz(&[0, 1]);
    inline
        .tick()
        .try_add_gate(Gate::channel(pecos_core::channel::BitFlip(0.2, 0)))
        .unwrap();
    inline.tick().cx(&[(0, 1)]);
    measured(&mut inline, &[0, 1]);
    let inline_dem = DemBuilder::try_from_tick_circuit(&inline, 0.0, 0.0, 0.0, 0.0).unwrap();
    assert_mechanisms_close(&before, &inline_dem);
    let mut config = NoiseConfig::new(0.0, 0.0, 0.0, 0.0);
    config.p2 = 0.2;
    config.p2_weights = Some(crate::fault_tolerance::dem_builder::PauliWeights::new([(
        "XI".parse().unwrap(),
        1.0,
    )]));
    let ordinary = DemBuilder::try_from_tick_circuit_with_noise_config(&c, config).unwrap();
    assert_mechanisms_close(&after, &ordinary);
}

#[test]
fn ordinary_locations_accept_random_detectors_and_unstamped_records() {
    let mut c = TickCircuit::new();
    c.tick().h(&[0]);
    gate(&mut c, GateType::MZ, &[0]);
    c.set_meta(
        "detectors",
        pecos_quantum::Attribute::String("[{\"id\":0,\"records\":[-1]}]".into()),
    );
    DemBuilder::try_from_tick_circuit_with_location_channels(
        &c,
        NoiseConfig::new(0.0, 0.0, 0.0, 0.0),
        &[channel(0, Side::After, "X", 0.1)],
    )
    .unwrap();
}

#[test]
fn categorical_channels_report_residuals() {
    let mut c = TickCircuit::new();
    c.tick().pz(&[0, 1]);
    c.tick().cx(&[(0, 1)]);
    measured(&mut c, &[0, 1]);
    let source = LocationChannel {
        location: location(1, Side::After),
        probabilities: BTreeMap::from([("XI".into(), 0.1), ("IX".into(), 0.1)]),
    };
    let actual = DemBuilder::try_from_tick_circuit_with_location_channels(
        &c,
        NoiseConfig::new(0.0, 0.0, 0.0, 0.0),
        &[source],
    )
    .unwrap();
    let mut config = NoiseConfig::new(0.0, 0.2, 0.0, 0.0);
    config.p2_weights = Some(crate::fault_tolerance::dem_builder::PauliWeights::new([
        ("XI".parse().unwrap(), 0.5),
        ("IX".parse().unwrap(), 0.5),
    ]));
    let expected = DemBuilder::try_from_tick_circuit_with_noise_config(&c, config).unwrap();
    assert_ne!(actual.idle_noise_residuals(), []);
    assert_eq!(actual.to_mechanisms(), expected.to_mechanisms());
    assert_eq!(
        actual.idle_noise_residuals(),
        expected.idle_noise_residuals()
    );
    // X/Z on one leg of the Bell pair have independent parity signatures too.
    let c = bell();
    let source = LocationChannel {
        location: location(3, Side::After),
        probabilities: BTreeMap::from([("X".into(), 0.1), ("Z".into(), 0.1)]),
    };
    let actual = DemBuilder::try_from_tick_circuit_with_location_channels(
        &c,
        NoiseConfig::new(0.0, 0.0, 0.0, 0.0),
        &[source],
    )
    .unwrap();
    assert_eq!(actual.idle_noise_residuals().len(), 1);
    let target = [0.8, 0.1, 0.1, 0.0];
    let tv = distribution(&actual)
        .iter()
        .zip(target)
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / 2.0;
    assert!((actual.idle_noise_residuals()[0].magnitude - tv).abs() < 1e-12);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .add_events_channel(
                LossSite::AfterSingleQubit,
                0.2,
                &BTreeMap::from([("X".into(), 0.5), ("Z".into(), 0.5)]),
            )
            .build()
            .unwrap(),
    );
    assert_ne!(
        background_dem(&c, &noise).unwrap().idle_noise_residuals(),
        []
    );
    // Above the full-channel uniform endpoint, a one-signature projection can
    // still be accepted by the ordinary builder and must remain accepted here.
    let mut c = TickCircuit::new();
    c.tick().pz(&[0]);
    c.tick().sz(&[0]);
    measured(&mut c, &[0]);
    let source = LocationChannel {
        location: location(1, Side::After),
        probabilities: BTreeMap::from([("X".into(), 0.3), ("Y".into(), 0.3), ("Z".into(), 0.3)]),
    };
    let actual = DemBuilder::try_from_tick_circuit_with_location_channels(
        &c,
        NoiseConfig::new(0.0, 0.0, 0.0, 0.0),
        &[source],
    )
    .unwrap();
    let expected = DemBuilder::try_from_tick_circuit(&c, 0.9, 0.0, 0.0, 0.0).unwrap();
    assert_mechanisms_close(&actual, &expected);
}

#[test]
fn bridge_noiseless_models_and_readout_exemptions() {
    let c = bell();
    let model = GeneralNoiseModel::builder()
        .with_p1(0.02)
        .with_p2(0.03)
        .with_noiseless_gate(GateType::H)
        .with_noiseless_gate(GateType::CZ)
        .with_p1_pauli_model(&BTreeMap::from([("X".into(), 1.0)]))
        .with_p2_pauli_model(&BTreeMap::from([("XI".into(), 1.0)]))
        .build();
    let description = model.description();
    let config = pauli_noise_config(&c, &description).unwrap();
    let mut expected = NoiseConfig::new(0.02, 0.03, 0.0, 0.0);
    expected.p1_gate_rates.insert(GateType::H, 0.0);
    expected.p2_gate_rates.insert(GateType::CZ, 0.0);
    expected.p1_weights = Some(crate::fault_tolerance::dem_builder::PauliWeights::new([(
        "X".parse().unwrap(),
        1.0,
    )]));
    expected.p2_weights = Some(crate::fault_tolerance::dem_builder::PauliWeights::new([(
        "XI".parse().unwrap(),
        1.0,
    )]));
    assert_eq!(
        DemBuilder::try_from_tick_circuit_with_noise_config(&c, config)
            .unwrap()
            .to_string(),
        DemBuilder::try_from_tick_circuit_with_noise_config(&c, expected)
            .unwrap()
            .to_string()
    );
    let mut description = description;
    description.readout.insert(
        GateType::DetectLost,
        BTreeMap::from([("present".into(), BTreeMap::from([(3, 1.0)]))]),
    );
    description.readout.insert(
        GateType::MeasureZLost,
        BTreeMap::from([("0".into(), BTreeMap::from([(1, 1.0)]))]),
    );
    description.idle_after_2q = 2.0;
    pauli_noise_config(&c, &description).unwrap();
    let mut c = TickCircuit::new();
    c.tick().pz(&[0]);
    c.tick().u(
        pecos_core::Angle64::ZERO,
        pecos_core::Angle64::ZERO,
        pecos_core::Angle64::QUARTER_TURN,
        &[0],
    );
    measured(&mut c, &[0]);
    let noise = GeneralNoiseModel::builder()
        .with_p1(0.2)
        .with_noiseless_gate(GateType::RZ)
        .build();
    let config = pauli_noise_config(&c, &noise.description()).unwrap();
    assert_eq!(config.p1_gate_rates.get(&GateType::U), Some(&0.0));
    assert_eq!(
        background_dem(&c, &noise).unwrap().to_mechanisms().0.len(),
        0
    );
}

#[test]
fn runner_gate_errors_name_coordinates() {
    for kind in [
        GateType::MX,
        GateType::MeasureFree,
        GateType::PX,
        GateType::I,
        GateType::QAlloc,
        GateType::QFree,
        GateType::T,
        GateType::Custom,
    ] {
        let mut c = TickCircuit::new();
        let qs = if kind.is_two_qubit() {
            vec![0, 1]
        } else {
            vec![0]
        };
        gate(&mut c, kind, &qs);
        let error = OutputTable::new(&c).unwrap_err().to_string();
        assert!(error.contains("tick 0, batch 0"), "{error}");
        assert!(error.contains(&format!("{kind:?}")), "{error}");
    }
}

#[test]
fn runner_payload_and_rotation_errors() {
    for gate in [
        Gate::channel(pecos_core::channel::BitFlip(0.1, 0)),
        Gate::rz(
            pecos_core::Angle64::from_str_radians("0.123").unwrap(),
            &[0],
        ),
    ] {
        let mut c = TickCircuit::new();
        c.tick().pz(&[0]);
        c.tick().try_add_gate(gate.clone()).unwrap();
        let error = OutputTable::new(&c).unwrap_err();
        assert!(matches!(error, LossDemError::Circuit { .. }));
        let message = error.to_string();
        assert!(message.contains("tick 1, batch 0"), "{message}");
        assert!(
            message.contains(&format!("{:?}", gate.gate_type)),
            "{message}"
        );
        assert!(message.contains(&format!("{:?}", gate.angles)), "{message}");
        assert!(
            message.contains("use supported Clifford gates"),
            "{message}"
        );
    }
}

#[test]
fn bridge_approximate_pauli_models_match_ordinary() {
    let c = bell();
    let one = BTreeMap::from([("X".into(), 0.5), ("Z".into(), 0.5)]);
    let two = BTreeMap::from([("XI".into(), 0.5), ("IX".into(), 0.5)]);
    let noise = GeneralNoiseModel::builder()
        .with_p1(0.2)
        .with_p2(0.2)
        .with_p1_pauli_model(&one)
        .with_p2_pauli_model(&two)
        .build();
    let mut config = NoiseConfig::new(0.2, 0.2, 0.0, 0.0);
    config.p1_weights = Some(crate::fault_tolerance::dem_builder::PauliWeights::new([
        ("X".parse().unwrap(), 0.5),
        ("Z".parse().unwrap(), 0.5),
    ]));
    config.p2_weights = Some(crate::fault_tolerance::dem_builder::PauliWeights::new([
        ("XI".parse().unwrap(), 0.5),
        ("IX".parse().unwrap(), 0.5),
    ]));
    let expected = DemBuilder::try_from_tick_circuit_with_noise_config(&c, config).unwrap();
    assert_ne!(expected.idle_noise_residuals(), []);
    let actual = background_dem(&c, &noise).unwrap();
    assert_eq!(actual.to_mechanisms(), expected.to_mechanisms());
    assert_eq!(
        actual.idle_noise_residuals(),
        expected.idle_noise_residuals()
    );
}

#[test]
fn events_two_qubit_independent_labels() {
    // On |00>, XI before CX flips both records and IX flips only record 1.
    // Unequal weights ensure exchanging the label legs changes the distribution.
    let mut c = TickCircuit::new();
    c.tick().pz(&[0, 1]);
    c.tick().cx(&[(0, 1)]);
    measured(&mut c, &[0, 1]);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .add_events_channel(
                LossSite::BeforeTwoQubit,
                0.2,
                &BTreeMap::from([("XI".into(), 0.25), ("IX".into(), 0.75)]),
            )
            .build()
            .unwrap(),
    );
    let actual = background_dem(&c, &noise).unwrap();
    let mut config = NoiseConfig::new(0.0, 0.2, 0.0, 0.0);
    config.p2_weights = Some(crate::fault_tolerance::dem_builder::PauliWeights::new([
        ("XX".parse().unwrap(), 0.25),
        ("IX".parse().unwrap(), 0.75),
    ]));
    let expected = DemBuilder::try_from_tick_circuit_with_noise_config(&c, config).unwrap();
    assert_eq!(actual.to_mechanisms(), expected.to_mechanisms());
    assert_eq!(
        actual.idle_noise_residuals(),
        expected.idle_noise_residuals()
    );
    let target = [0.8, 0.0, 0.15, 0.05];
    let tv = distribution(&actual)
        .iter()
        .zip(target)
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / 2.0;
    assert!(tv > 0.0);
    assert!((actual.idle_noise_residuals()[0].magnitude - tv).abs() < 1e-12);
    // The engine samples the categorical target, not the approximate DEM.
    assert_engine_cells(&c, noise, &target);
}

#[test]
fn measurement_noise_cross_qubit_guard() {
    let mut c = TickCircuit::new();
    c.tick().pz(&[0, 1]);
    measured(&mut c, &[0]);
    c.tick().cx(&[(0, 1)]);
    measured(&mut c, &[1]);
    let noise = GeneralNoiseModel::builder()
        .with_p_meas_0(0.1)
        .with_p_meas_1(0.1)
        .build();
    let error = pauli_noise_config(&c, &noise.description())
        .unwrap_err()
        .to_string();
    assert!(error.contains("record 0 propagates to record 1"), "{error}");
    assert!(error.contains("prepare the measured qubit"), "{error}");
    pauli_noise_config(&c, &GeneralNoiseModel::default().description()).unwrap();
}

#[test]
fn noiseless_mpz_does_not_have_prep_noise() {
    let mut c = TickCircuit::new();
    gate(&mut c, GateType::MPZ, &[0]);
    let noise = GeneralNoiseModel::builder()
        .with_p_prep(0.2)
        .with_noiseless_gate(GateType::MPZ)
        .build();
    let actual = background_dem(&c, &noise).unwrap();
    let expected = DemBuilder::try_from_tick_circuit(&c, 0.0, 0.0, 0.0, 0.2).unwrap();
    assert_eq!(actual.to_mechanisms(), expected.to_mechanisms());
    let mut description = noise.description();
    description.p_meas_0 = 0.1;
    description.p_meas_1 = 0.1;
    assert!(
        pauli_noise_config(&c, &description)
            .unwrap_err()
            .to_string()
            .contains("noiseless_gates.MPZ")
    );
}

#[test]
fn stochastic_runner_matches_independent_message() {
    let mut c = TickCircuit::new();
    c.tick().pz(&[0, 1]);
    c.tick().h(&[0]);
    c.tick().cx(&[(0, 1)]);
    c.tick().mz(&[1, 0]);
    let noise = GeneralNoiseModel::builder()
        .with_p1(0.17)
        .with_p2(0.23)
        .with_p_prep(0.07)
        .with_p_meas_0(0.11)
        .with_p_meas_1(0.11)
        .build();
    let mut runner = TickRunner::new(&c, noise.clone(), 8192).unwrap();
    let mut direct = QuantumSystem::new(
        Box::new(noise),
        Box::new(SparseStabEngine::with_seed(2, 8192)),
    );
    direct.set_seed(8192);
    // Explicit commands; this reference never traverses TickCircuit batches.
    let mut message = ByteMessage::quantum_operations_builder();
    message.add_gate_command(&Gate::simple(GateType::PZ, vec![0.into(), 1.into()]));
    message.add_gate_command(&Gate::simple(GateType::H, vec![0.into()]));
    message.add_gate_command(&Gate::simple(GateType::CX, vec![0.into(), 1.into()]));
    message.add_gate_command(&Gate::simple(GateType::MZ, vec![1.into(), 0.into()]));
    let message = message.build();
    let mut seen = std::collections::BTreeSet::new();
    for _ in 0..256 {
        direct.reset().unwrap();
        let values = direct
            .process_as_system(message.clone())
            .unwrap()
            .outcomes()
            .unwrap();
        let shot = runner.shot().unwrap();
        assert_eq!(
            shot.records
                .iter()
                .copied()
                .map(u32::from)
                .collect::<Vec<_>>(),
            values
        );
        seen.insert(values);
    }
    assert!(seen.len() > 1, "fixture must exercise stochastic outcomes");
}

#[test]
fn later_and_multi_instance_batch_locations() {
    for operand in 0..2 {
        let qubit = [2, 1][operand];
        for side in [Side::Before, Side::After] {
            let make = |inject| {
                let mut c = TickCircuit::new();
                c.tick().pz(&[0, 1, 2]);
                if inject && side == Side::Before {
                    c.tick().x(&[qubit]);
                }
                c.tick().z(&[0]).h(&[2, 1]);
                if inject && side == Side::After {
                    c.tick().x(&[qubit]);
                }
                c.tick().h(&[1, 2]);
                measured(&mut c, &[2, 1]);
                c
            };
            let c = make(false);
            let loc = Location {
                tick: 1,
                batch: 1,
                operand,
                side,
            };
            let flips = located_pauli_flips(&c, Pauli::X, loc).unwrap();
            let expected = TickRunner::new(&make(true), GeneralNoiseModel::default(), 7)
                .unwrap()
                .shot()
                .unwrap()
                .records;
            assert_eq!(
                flips.iter().collect::<Vec<_>>(),
                expected
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| (v != 0).then_some(i))
                    .collect::<Vec<_>>()
            );
            let dem = DemBuilder::try_from_tick_circuit_with_location_channels(
                &c,
                NoiseConfig::new(0.0, 0.0, 0.0, 0.0),
                &[LocationChannel {
                    location: loc,
                    probabilities: BTreeMap::from([("X".into(), 1.0)]),
                }],
            )
            .unwrap();
            let expected_cell = expected
                .iter()
                .enumerate()
                .fold(0, |mask, (i, &v)| mask | (usize::from(v) << i));
            assert!((distribution(&dem)[expected_cell] - 1.0).abs() < 1e-12);
        }
    }
}

#[test]
fn typed_definition_and_circuit_sources() {
    use std::error::Error;
    let mut c = TickCircuit::new();
    c.tick().mz(&[0]);
    c.set_meta(
        "detectors",
        pecos_quantum::Attribute::String("not JSON".into()),
    );
    let Err(error) = TickRunner::new(&c, GeneralNoiseModel::default(), 1) else {
        panic!("malformed definitions must fail");
    };
    assert!(matches!(error, LossDemError::Definition(_)));
    assert!(error.source().is_some());
    assert!(
        error
            .to_string()
            .contains("correct the named record or definition")
    );
    let mut c = TickCircuit::new();
    c.tick().rz(
        pecos_core::Angle64::from_str_radians("0.123").unwrap(),
        &[0],
    );
    let error = DemBuilder::try_from_tick_circuit_with_location_channels(
        &c,
        NoiseConfig::new(0.0, 0.0, 0.0, 0.0),
        &[channel(0, Side::After, "X", 0.1)],
    )
    .unwrap_err();
    assert!(matches!(error, LossDemError::Circuit { .. }));
    assert!(error.source().is_some());
    assert!(error.to_string().contains("correct the named circuit gate"));
}

#[test]
fn invalid_records_keep_values_and_coordinates() {
    let mut c = TickCircuit::new();
    c.tick().pz(&[0]);
    c.tick().mz(&[0]);
    let table = OutputTable::new(&c).unwrap();
    for (outcomes, expected) in [
        (&[3][..], "nonbinary value 3"),
        (&[][..], "returned 0 outcomes, expected 1"),
    ] {
        let error = table.decode(outcomes, "tick 1, batch 0").unwrap_err();
        assert!(matches!(error, LossDemError::Record { .. }));
        let message = error.to_string();
        assert!(message.contains(expected), "{message}");
        assert!(message.contains("tick"), "{message}");
        assert!(message.contains("batch"), "{message}");
    }
    assert_eq!(table.decode(&[1], "tick 1, batch 0").unwrap().0, [1]);
}

#[test]
fn prepared_propagation_checks_locations() {
    use crate::fault_tolerance::fault_sampler::flatten_tick_circuit;
    let mut c = TickCircuit::new();
    c.tick().pz(&[0, 1, 2]);
    c.tick().z(&[2]).cx(&[(1, 0)]);
    measured(&mut c, &[0, 1]);
    let (gates, measurements) = flatten_tick_circuit(&c);
    for (operand, expected) in [(0, vec![0, 1]), (1, vec![0])] {
        let loc = Location {
            tick: 1,
            batch: 1,
            operand,
            side: Side::Before,
        };
        let flips =
            super::symbolic::located_pauli_flips_prepared(&gates, &measurements, Pauli::X, loc)
                .unwrap();
        assert_eq!(flips.iter().collect::<Vec<_>>(), expected);
        assert_eq!(located_pauli_flips(&c, Pauli::X, loc).unwrap(), flips);
    }
    for loc in [
        Location {
            tick: 10,
            batch: 0,
            operand: 0,
            side: Side::Before,
        },
        Location {
            tick: 1,
            batch: 10,
            operand: 0,
            side: Side::After,
        },
        Location {
            tick: 1,
            batch: 1,
            operand: usize::MAX,
            side: Side::Before,
        },
    ] {
        assert!(
            super::symbolic::located_pauli_flips_prepared(&gates, &measurements, Pauli::X, loc)
                .is_err()
        );
        assert!(located_pauli_flips(&c, Pauli::X, loc).is_err());
    }
    gate(&mut c, GateType::T, &[0]);
    assert!(located_pauli_flips(&c, Pauli::X, location(0, Side::Before)).is_err());
}

#[test]
fn probe_waiters_stop_at_first_record() {
    let mut c = TickCircuit::new();
    gate(&mut c, GateType::DetectLost, &[0]);
    gate(&mut c, GateType::DetectLeakedLost, &[1]);
    gate(&mut c, GateType::DetectLost, &[0]);
    c.tick().mz(&[0]);
    gate(&mut c, GateType::DetectLost, &[0]);
    c.tick().mz(&[1]);
    c.tick().mz(&[0]);
    c.tick().mz(&[0]);
    gate(&mut c, GateType::DetectLost, &[1]);
    let table = OutputTable::new(&c).unwrap();
    assert_eq!(
        table
            .probes
            .iter()
            .map(|p| p.next_record)
            .collect::<Vec<_>>(),
        [Some(0), Some(1), Some(0), Some(2), None]
    );
}

#[test]
fn location_sources_preserve_dem_output() {
    use crate::fault_tolerance::dem_builder::DirectSourceFamily;
    use crate::fault_tolerance::propagator::DagFaultAnalyzer;
    let mut c = TickCircuit::new();
    c.tick().pz(&[0, 1]);
    c.tick().cx(&[(0, 1)]);
    measured(&mut c, &[0, 1]);
    let dag = pecos_quantum::DagCircuit::try_from(&c).unwrap();
    let map = DagFaultAnalyzer::new(&dag).build_influence_map().unwrap();
    for (source, qubits, fitted) in [
        (channel(1, Side::Before, "X", 0.2), vec![0, 1], false),
        (
            LocationChannel {
                location: location(1, Side::After),
                probabilities: BTreeMap::from([
                    ("X".into(), 0.05),
                    ("Y".into(), 0.05),
                    ("Z".into(), 0.05),
                ]),
            },
            vec![0],
            false,
        ),
        (
            LocationChannel {
                location: location(1, Side::After),
                probabilities: BTreeMap::from([("XI".into(), 0.1), ("IX".into(), 0.1)]),
            },
            vec![0, 1],
            true,
        ),
    ] {
        let dem = DemBuilder::try_from_tick_circuit_with_location_channels(
            &c,
            NoiseConfig::new(0.0, 0.0, 0.0, 0.0),
            &[source],
        )
        .unwrap();
        let mut without_metadata =
            DemBuilder::try_from_tick_circuit(&c, 0.0, 0.0, 0.0, 0.0).unwrap();
        assert!(dem.num_contributions() > 0);
        for contribution in dem.contributions() {
            let indices = &contribution.location_indices;
            assert_eq!(
                indices
                    .iter()
                    .map(|&i| map.locations[i as usize].qubits[0].index())
                    .collect::<Vec<_>>(),
                qubits
            );
            assert_eq!(
                contribution.source_gate_types.as_slice(),
                vec![GateType::CX; qubits.len()]
            );
            assert_eq!(
                contribution.source_before_flags.as_slice(),
                vec![false; qubits.len()]
            );
            if fitted {
                assert_eq!(contribution.paulis.as_slice(), []);
                assert_eq!(
                    contribution.direct_source_family,
                    Some(DirectSourceFamily::ExclusiveSignature)
                );
            } else {
                assert_eq!(contribution.paulis.len(), qubits.len());
            }
            // Reconstruct the prior untracked contribution path, preserving its inputs.
            without_metadata
                .add_direct_contribution(contribution.effect.clone(), contribution.probability);
        }
        assert_eq!(dem.to_string(), without_metadata.to_string());
        assert_eq!(dem.to_mechanisms(), without_metadata.to_mechanisms());
    }
}

#[test]
fn location_totals_match_ordinary_strict_validation() {
    use crate::fault_tolerance::dem_builder::PauliWeights;
    let mut c = TickCircuit::new();
    c.tick().pz(&[0]);
    c.tick().sz(&[0]);
    measured(&mut c, &[0]);
    for excess in [0.0, f64::EPSILON] {
        let labels = BTreeMap::from([("X".into(), 0.5), ("Z".into(), 0.5 + excess)]);
        let located = DemBuilder::try_from_tick_circuit_with_location_channels(
            &c,
            NoiseConfig::new(0.0, 0.0, 0.0, 0.0),
            &[LocationChannel {
                location: location(1, Side::After),
                probabilities: labels,
            }],
        );
        let mut config = NoiseConfig::new(1.0, 0.0, 0.0, 0.0);
        config.p1_weights = Some(PauliWeights::new([
            ("X".parse().unwrap(), 0.5),
            ("Z".parse().unwrap(), 0.5 + excess),
        ]));
        let ordinary = DemBuilder::try_from_tick_circuit_with_noise_config(&c, config);
        assert_eq!(located.is_ok(), excess == 0.0);
        assert_eq!(located.is_ok(), ordinary.is_ok());
        if let (Ok(located), Ok(ordinary)) = (located, ordinary) {
            assert_eq!(located.to_mechanisms(), ordinary.to_mechanisms());
        }
    }
}

#[test]
fn idle_cutoff_matches_engine_before_duration_scaling() {
    let mut c = TickCircuit::new();
    c.tick().pz(&[0]);
    c.tick().idle(1_u64 << 51, &[0]);
    measured(&mut c, &[0]);
    for (rate, expected) in [
        (0.0, 0),
        (f64::EPSILON / 2.0, 0),
        (f64::EPSILON, 0),
        (f64::EPSILON * 2.0, 1),
    ] {
        let noise = GeneralNoiseModel::builder()
            .with_p_idle_linear(rate, &BTreeMap::from([("X".into(), 1.0)]))
            .build();
        let description = noise.description();
        let config = pauli_noise_config(&c, &description).unwrap();
        assert_eq!(
            config.p_idle_linear.rate,
            if expected == 0 { 0.0 } else { rate }
        );
        let dem = background_dem(&c, &noise).unwrap();
        assert_eq!(
            distribution(&dem),
            if expected == 0 {
                vec![1.0, 0.0]
            } else {
                vec![0.0, 1.0]
            }
        );
        let mut runner = TickRunner::new(&c, noise, 84).unwrap();
        for _ in 0..8 {
            assert_eq!(runner.shot().unwrap().records, [expected]);
        }
        let mut d = description.clone();
        d.noiseless_gates.insert(GateType::Idle);
        assert_eq!(pauli_noise_config(&c, &d).is_ok(), expected == 0);
        let mut d = description;
        d.idle_after_2q = 1.0;
        assert_eq!(pauli_noise_config(&c, &d).is_ok(), expected == 0);
    }
}

#[test]
fn runtime_record_validation_is_shared() {
    let mut c = TickCircuit::new();
    c.tick().pz(&[0, 1]);
    measured(&mut c, &[0, 1]);
    let build = DetectorErrorModel::builder()
        .with_circuit(c)
        .build()
        .unwrap();
    assert_eq!(build.evaluate_runtime_record(&[1, 0]).unwrap().0, [1, 0]);
    for invalid in [&[0, 0, 0][..], &[0][..], &[2, 0][..]] {
        assert!(build.evaluate_runtime_record(invalid).is_err());
    }
}
