//! Core identities, byte builders, plain simulator dispatch, and wire rejection.
use pecos_engines::noise::{BiasedDepolarizingNoiseModel, DepolarizingNoiseModel};
use pecos_engines::quantum::{
    CoinTossEngine, DensityMatrixEngine, SparseStabEngine, StabVecEngine, StabilizerEngine,
    StateVecEngine,
};
use pecos_engines::runtime_frame::{FrameRecord, encode_frame};
use pecos_engines::scheduled_frame::{TimedBatch, encode_timed_batches};
use pecos_engines::{ByteMessage, Engine, Gate, GateType, QuantumSystem};

const HERALDS: [GateType; 2] = [GateType::MeasureZLost, GateType::MeasureZLeakedLost];

#[test]
fn herald_identity_metadata_and_builders() {
    for (gate, wire, name) in [
        (HERALDS[0], 108, "MeasureZLost"),
        (HERALDS[1], 109, "MeasureZLeakedLost"),
    ] {
        assert_eq!(gate as u8, wire);
        assert_eq!(GateType::try_from(wire), Ok(gate));
        assert_eq!(gate.to_string(), name);
        assert_eq!(name.to_lowercase().parse::<GateType>().unwrap(), gate);
        assert_eq!(gate.quantum_arity(), 1);
        assert_eq!(gate.classical_arity(), 0);
        assert_eq!(gate.measured_pauli(), None);
        assert!(!gate.consumes_measurement_record());
        assert!(gate.is_single_qubit());
        let constructed = if gate == HERALDS[0] {
            Gate::measure_z_lost(&[0, 1])
        } else {
            Gate::measure_z_leaked_lost(&[0, 1])
        };
        assert_eq!(constructed.gate_type, gate);
        constructed.validate().unwrap();
        let mut bytes = ByteMessage::quantum_operations_builder();
        if gate == HERALDS[0] {
            bytes.measure_z_lost(&[0, 1]);
        } else {
            bytes.measure_z_leaked_lost(&[0, 1]);
        }
        assert_eq!(
            bytes.build().quantum_ops().unwrap(),
            [
                Gate::simple(gate, vec![0.into()]),
                Gate::simple(gate, vec![1.into()])
            ]
        );
    }
}

#[test]
fn all_engine_backends_execute_plain_z_heralds() {
    for gate in HERALDS {
        let engines: Vec<Box<dyn Engine<Input = ByteMessage, Output = ByteMessage>>> = vec![
            Box::new(StateVecEngine::with_seed(2, 42)),
            Box::new(SparseStabEngine::with_seed(2, 42)),
            Box::new(StabilizerEngine::with_seed(2, 42)),
            Box::new(StabVecEngine::with_seed(2, 42)),
            Box::new(DensityMatrixEngine::with_seed(2, 42)),
        ];
        for mut engine in engines {
            let mut b = ByteMessage::quantum_operations_builder();
            b.x(&[1]);
            // Consecutive different flavors exercise batched dispatch as well.
            b.add_gate_command(&Gate::simple(gate, vec![0.into()]));
            b.mz(&[1]);
            b.add_gate_command(&Gate::simple(gate, vec![1.into()]));
            assert_eq!(
                engine.process(b.build()).unwrap().outcomes().unwrap(),
                [0, 1, 1]
            );
        }
        let mut coin = CoinTossEngine::with_prob_and_seed(1, 1.0, 42);
        let mut b = ByteMessage::quantum_operations_builder();
        b.add_gate_command(&Gate::simple(gate, vec![0.into()]));
        assert_eq!(coin.process(b.build()).unwrap().outcomes().unwrap(), [1]);
    }
}

#[test]
fn depolarizing_models_classify_heralds_as_measurements() {
    for gate in HERALDS {
        let models: Vec<Box<dyn pecos_engines::noise::NoiseModel>> = vec![
            Box::new(
                DepolarizingNoiseModel::builder()
                    .with_uniform_probability(0.0)
                    .with_p_meas(1.0)
                    .with_p1(0.0)
                    .build(),
            ),
            Box::new(
                BiasedDepolarizingNoiseModel::builder()
                    .with_uniform_probability(0.0)
                    .with_p_meas_0(1.0)
                    .with_p1(0.0)
                    .build(),
            ),
        ];
        for noise in models {
            let mut system = QuantumSystem::new(noise, Box::new(StateVecEngine::with_seed(1, 42)));
            let mut b = ByteMessage::quantum_operations_builder();
            b.add_gate_command(&Gate::simple(gate, vec![0.into()]));
            assert_eq!(system.process(b.build()).unwrap().outcomes().unwrap(), [1]);
        }
    }
}

#[test]
fn versioned_wire_tables_reject_each_herald_by_name() {
    for gate in HERALDS {
        let command = Gate::simple(gate, vec![0.into()]);
        let error = encode_frame(&[FrameRecord::gate(command.clone())])
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains(&format!("unsupported frame gate: {gate}")),
            "{error}"
        );
        let batch = TimedBatch {
            runtime_shot_id: 0,
            batch_index: 0,
            start_nanos: 0,
            duration_nanos: 1,
            gates: vec![command],
        };
        let error = encode_timed_batches(&[batch]).err().unwrap().to_string();
        assert!(
            error.contains(&format!("unsupported scheduled gate: {gate}")),
            "{error}"
        );
    }
}
