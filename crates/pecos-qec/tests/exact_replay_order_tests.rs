use pecos_core::PauliString;
use pecos_qec::fault_tolerance::dem_builder::{
    DemBuilder, DirectSourceFamily, NoiseConfig, PauliWeights, ReplacementBranchApproximation,
};
use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;
use pecos_quantum::{Attribute, DagCircuit};

fn identity_replacement_noise() -> NoiseConfig {
    NoiseConfig::new(0.0, 0.125, 0.0, 0.0)
        .set_p2_weights(PauliWeights::with_replacement(
            [],
            [(PauliString::identity(), 1.0)],
        ))
        .set_p2_replacement_approximation(ReplacementBranchApproximation::ExactBranchReplay)
}

#[test]
fn exact_replay_rejects_missing_measurement_nodes() {
    use pecos_qec::fault_tolerance::dem_builder::DemBuilderError;

    let dag = circuit(false, false);
    for node in [0, usize::MAX] {
        let mut map = DagFaultAnalyzer::new(&dag)
            .build_influence_map()
            .expect("supported circuit");
        // Node 0 is a preparation, and MAX is beyond the replay's node vector.
        map.measurements[0].0 = node;
        let error = DemBuilder::new(&map)
            .with_exact_branch_replay_context(&dag)
            .with_noise_config(identity_replacement_noise())
            .with_detectors_json(r#"[{"id":0,"records":[0]}]"#)
            .unwrap()
            .try_build()
            .unwrap_err();
        assert!(matches!(error, DemBuilderError::ConfigurationError(_)));
        assert!(error.to_string().contains("has no history entry"));
    }
}

// exercises the deprecated measurement-order API
#[allow(deprecated)]
#[test]
fn exact_crosstalk_translates_non_identity_measurement_order() {
    use pecos_core::Gate;
    use pecos_qec::fault_tolerance::dem_builder::{
        MeasurementCrosstalkDemMode, MeasurementCrosstalkTransitionModel,
    };

    let mut dag = DagCircuit::new();
    dag.pz(&[0, 1]);
    // Both preparations precede the payload, and both measurements follow it,
    // including in the unchanged DFS walk used for global victim enumeration.
    dag.add_gate_auto_wire(Gate::i(&[0usize, 1usize]));
    dag.add_gate_auto_wire(Gate::meas_crosstalk_global_payload(&[1usize]));
    dag.add_gate_auto_wire(Gate::i(&[0usize, 1usize]));
    dag.mz(&[0, 1]);
    let map = DagFaultAnalyzer::new(&dag)
        .build_influence_map()
        .expect("supported circuit");
    let noise = NoiseConfig::new(0.0, 0.0, 0.0, 0.0)
        .set_measurement_crosstalk_global_rate(0.25)
        .set_measurement_crosstalk_transition_model(MeasurementCrosstalkTransitionModel::bit_flip(
            0.4, 0.0,
        ))
        .set_measurement_crosstalk_dem_mode(MeasurementCrosstalkDemMode::ExactDeterministic);
    let builder = DemBuilder::new(&map)
        .with_exact_branch_replay_context(&dag)
        .with_measurement_order(vec![1, 0])
        .with_noise_config(noise)
        .with_detectors_json(r#"[{"id":0,"records":[1]},{"id":1,"records":[0]}]"#)
        .unwrap()
        .with_observables_json(r#"[{"id":0,"records":[1]}]"#)
        .unwrap();
    let dem = builder.try_build().unwrap();
    // The hidden q0 measurement is zero; its 0->1 transition has probability
    // 0.25 * 0.4 = 0.1. Only q0's detector D0 and observable L0 flip.
    assert_eq!(
        dem.to_string()
            .lines()
            .filter(|line| line.starts_with("error("))
            .collect::<Vec<_>>(),
        ["error(0.1) D0 L0"]
    );
    // The target's record has no matching occurrence in this invalid order.
    // The public build must return a configuration error, not panic downstream.
    let error = builder
        .with_measurement_order(vec![1, 1])
        .try_build()
        .unwrap_err();
    assert!(matches!(
        error,
        pecos_qec::fault_tolerance::dem_builder::DemBuilderError::ConfigurationError(_)
    ));
}

#[test]
fn exact_replay_with_out_of_order_stamps() {
    use pecos_core::{Gate, MeasId};

    let mut dag = DagCircuit::new();
    dag.pz(&[0, 1, 2]);
    dag.x(&[0]);
    dag.cx(&[(0, 1)]);
    for (qubit, id) in [(1, 9), (2, 4)] {
        let mut gate = Gate::mz(&[qubit]);
        gate.meas_ids = smallvec::smallvec![MeasId::from_raw(id)];
        dag.add_gate_auto_wire(gate);
    }
    // Emission order is target(id 9), spectator(id 4), regardless of id rank.
    // Omitting CX changes q1 from 1 to 0 and leaves q2 at 0.
    dag.set_attr(
        "detectors",
        Attribute::String(r#"[{"id":0,"meas_ids":[9]},{"id":1,"meas_ids":[4]}]"#.into()),
    );
    let dem =
        DemBuilder::try_from_circuit_with_noise_config(&dag, identity_replacement_noise()).unwrap();
    assert_eq!(
        dem.to_string()
            .lines()
            .filter(|line| line.starts_with("error("))
            .collect::<Vec<_>>(),
        ["error(0.125) D0"]
    );
}

// exercises the deprecated measurement-order API
#[allow(deprecated)]
#[test]
fn exact_replay_translates_non_identity_measurement_order() {
    let dag = circuit(false, false);
    let map = DagFaultAnalyzer::new(&dag)
        .build_influence_map()
        .expect("supported circuit");
    for stamped in [false, true] {
        let (detectors, observables) = if stamped {
            // Stamps already resolve into map positions and must not be translated.
            (
                r#"[{"id":0,"meas_ids":[0]},{"id":1,"meas_ids":[1]}]"#,
                r#"[{"id":0,"meas_ids":[0]}]"#,
            )
        } else {
            // Caller record order is spectator(q2), target(q1), the reverse of
            // the map. Record 1 therefore names the target for both D0 and L0.
            (
                r#"[{"id":0,"records":[1]},{"id":1,"records":[0]}]"#,
                r#"[{"id":0,"records":[1]}]"#,
            )
        };
        let dem = DemBuilder::new(&map)
            .with_measurement_order(vec![2, 1])
            .with_exact_branch_replay_context(&dag)
            .with_noise_config(identity_replacement_noise())
            .with_detectors_json(detectors)
            .unwrap()
            .with_observables_json(observables)
            .unwrap()
            .try_build()
            .unwrap();
        // The omitted CX flips only q1: both its detector and its observable.
        assert_eq!(
            dem.to_string()
                .lines()
                .filter(|line| line.starts_with("error("))
                .collect::<Vec<_>>(),
            ["error(0.125) D0 L0"]
        );
    }
}

fn circuit(spectator_first: bool, use_ids: bool) -> DagCircuit {
    let mut circuit = DagCircuit::new();
    // Only the insertion order of independent preparations changes.
    if spectator_first {
        circuit.pz(&[2]);
    }
    circuit.pz(&[0]);
    circuit.pz(&[1]);
    if !spectator_first {
        circuit.pz(&[2]);
    }
    circuit.x(&[0]);
    circuit.cx(&[(0, 1)]);
    let target = circuit.mz(&[1])[0];
    let spectator = circuit.mz(&[2])[0];
    let ids = [target.meas_id.index(), spectator.meas_id.index()];
    let detectors = if use_ids {
        format!(
            r#"[{{"id":0,"meas_ids":[{}]}},{{"id":1,"meas_ids":[{}]}}]"#,
            ids[0], ids[1]
        )
    } else {
        // Absolute insertion-record positions: target=0, spectator=1.
        r#"[{"id":0,"records":[0]},{"id":1,"records":[1]}]"#.to_string()
    };
    circuit.set_attr("num_measurements", Attribute::String("2".into()));
    circuit.set_attr("detectors", Attribute::String(detectors.clone()));

    let order = circuit.topological_order();
    println!("spectator_first={spectator_first}, use_ids={use_ids}");
    println!("  topological_order={order:?}");
    for &node in &order {
        let gate = circuit.gate(node).unwrap();
        println!(
            "  node={node} gate={:?} qubits={:?} meas_ids={:?}",
            gate.gate_type, gate.qubits, gate.meas_ids
        );
    }
    println!(
        "  insertion measurements: q1 node={} id={}, q2 node={} id={}",
        target.node, ids[0], spectator.node, ids[1]
    );
    let analyzer = DagFaultAnalyzer::new(&circuit);
    let (measurements, _) = analyzer.extract_measurements();
    let map = analyzer.build_influence_map().expect("supported circuit");
    println!(
        "  influence measurements={measurements:?}, meas_ids={:?}",
        map.meas_ids
    );
    println!("  detectors={detectors}");
    circuit
}

#[test]
fn exact_replay_must_preserve_physical_measurement_identity() {
    // Hand oracle: PZ q0,q1,q2 gives |000>; X q0 and CX(0,1)
    // give q1=1, q2=0. With probability 1/8, replace the sole CX by
    // identity, giving q1=0, q2=0. Relative to the ideal measurements,
    // only D0 (q1) flips. Thus the sole mechanism must be error(0.125) D0,
    // regardless of preparation insertion order or metadata reference kind.
    let noise = NoiseConfig::new(0.0, 0.125, 0.0, 0.0)
        .set_p2_weights(PauliWeights::with_replacement(
            [],
            [(PauliString::identity(), 1.0)],
        ))
        .set_p2_replacement_approximation(ReplacementBranchApproximation::ExactBranchReplay);
    println!("noise={noise:?}");
    let mut all_errors = Vec::new();
    let mut all_match_oracle = true;
    for use_ids in [true, false] {
        let mut pair = Vec::new();
        for spectator_first in [false, true] {
            let circuit = circuit(spectator_first, use_ids);
            let dem = DemBuilder::try_from_circuit_with_noise_config(&circuit, noise.clone())
                .expect("identity replacement must reach exact replay and be representable");
            let text = dem.to_string();
            println!("DEM spectator_first={spectator_first}, use_ids={use_ids}:\n{text}");
            let errors: Vec<String> = text
                .lines()
                .filter(|line| line.starts_with("error("))
                .map(str::to_owned)
                .collect();
            let target = dem.contributions_for_effect(&[0], &[]);
            let spectator = dem.contributions_for_effect(&[1], &[]);
            let matches_oracle = target.len() == 1
                && (target[0].probability - 0.125).abs() < 1e-12
                && spectator.is_empty()
                && errors.len() == 1;
            println!("  matches hand oracle={matches_oracle}");
            assert!(
                target
                    .iter()
                    .chain(spectator.iter())
                    .any(|c| c.replacement_branch),
                "must have an actual replacement-branch contribution"
            );
            for contribution in target.iter().chain(spectator.iter()) {
                println!("  source family={:?}", contribution.direct_source_family);
                assert_eq!(
                    contribution.direct_source_family,
                    Some(DirectSourceFamily::TwoLocationExactReplacementBranch)
                );
            }
            all_match_oracle &= matches_oracle;
            pair.push(errors);
        }
        println!(
            "use_ids={use_ids}: insertion-order invariant={}",
            pair[0] == pair[1]
        );
        all_errors.push(pair);
    }
    // Delay the oracle assertion until both reference modes and circuits print.
    assert!(
        all_match_oracle && all_errors.iter().all(|pair| pair[0] == pair[1]),
        "exact replay violated physical measurement identity; see both DEM pairs above"
    );
}

#[test]
fn exact_pauli_replacement_uses_emission_positions_with_influence_builder_map() {
    use pecos_core::{Gate, MeasId};
    use pecos_qec::fault_tolerance::influence_builder::InfluenceBuilder;

    let mut dag = DagCircuit::new();
    dag.pz(&[0, 1]);
    dag.cx(&[(0, 1)]);
    for (qubit, id) in [(1, 9), (0, 4)] {
        let mut gate = Gate::mz(&[qubit]);
        gate.meas_ids = smallvec::smallvec![MeasId::from_raw(id)];
        dag.add_gate_auto_wire(gate);
    }
    let map = InfluenceBuilder::new(&dag)
        .expect("supported circuit")
        .build()
        .unwrap();
    let noise = NoiseConfig::new(0.0, 0.125, 0.0, 0.0)
        .set_p2_weights(PauliWeights::with_replacement(
            [],
            [(PauliString::xs(&[1]), 1.0)],
        ))
        .set_p2_replacement_approximation(ReplacementBranchApproximation::ExactBranchReplay);
    let dem = DemBuilder::new(&map)
        .with_exact_branch_replay_context(&dag)
        .with_noise_config(noise)
        .with_detectors_json(r#"[{"id":0,"meas_ids":[9]},{"id":1,"meas_ids":[4]}]"#)
        .unwrap()
        .try_build()
        .unwrap();
    // CX leaves |00> unchanged. Replacing it by IX gives |01>, so only
    // q1's detector D0 flips, with probability 1/8. Its id 9 has rank 1
    // but emission position 0. No records or record translation are involved.
    assert_eq!(
        dem.to_string()
            .lines()
            .filter(|line| line.starts_with("error("))
            .collect::<Vec<_>>(),
        ["error(0.125) D0"]
    );
}
