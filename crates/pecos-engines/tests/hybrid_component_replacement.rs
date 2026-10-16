// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

use pecos_engines::hybrid::{HybridEngine, HybridEngineBuilder};
use pecos_engines::monte_carlo::engine::ExternalClassicalEngine;
use pecos_engines::noise::{DepolarizingNoiseModel, PassThroughNoiseModel};
use pecos_engines::quantum::{StabVecEngine, StateVecEngine};
use pecos_engines::quantum_system::QuantumSystem;
use pecos_engines::{ByteMessage, Engine, EngineSystem, MonteCarloEngine};
use pecos_random::RngExt;

fn hybrid() -> HybridEngine {
    // The classical controller reports 2; the actual quantum capacity is 4.
    let mut engine = HybridEngineBuilder::new()
        .with_classical_engine(Box::new(ExternalClassicalEngine::new()))
        .with_quantum_engine(Box::new(StabVecEngine::new(4)))
        .with_depolarizing_noise(0.23)
        .unwrap()
        .with_seed(913)
        .build();
    engine
        .quantum_system
        .engine_mut()
        .process(ByteMessage::quantum_operations_builder().x(&[3]).build())
        .unwrap();
    engine
}

fn addresses(engine: &HybridEngine) -> [*const (); 3] {
    [
        std::ptr::from_ref(engine.classical_engine.as_ref()).cast(),
        std::ptr::from_ref(engine.quantum_system.quantum_engine()).cast(),
        std::ptr::from_ref(engine.quantum_system.noise_model()).cast(),
    ]
}

macro_rules! component_cases {
    ($($name:ident => $setter:literal),* $(,)?) => {
        $(#[test] fn $name() { assert_preserved_components($setter); })*
    };
}
component_cases! {
    seeded_hybrid_classical => "classical",
    seeded_hybrid_quantum => "quantum",
    seeded_hybrid_noise => "noise",
    seeded_hybrid_depolarizing => "depolarizing",
    seeded_hybrid_qubits => "qubits",
    seeded_hybrid_system => "system",
    seeded_hybrid_seed => "seed",
    seeded_hybrid_workers => "workers",
}

fn assert_preserved_components(setter: &str) {
    let original = hybrid();
    let original_addresses = addresses(&original);
    let mut noise_rng = original.quantum_system.noise_model().rng().clone();
    let expected_noise_draws: [u64; 8] = std::array::from_fn(|_| noise_rng.random());
    let mut expected_quantum = original.quantum_system.engine().clone();
    let builder = MonteCarloEngine::builder().with_hybrid_engine(original);
    let (builder, preserved) = match setter {
        "classical" => (
            builder.with_classical_engine(Box::new(ExternalClassicalEngine::new())),
            [false, true, true],
        ),
        "quantum" => (
            builder.with_quantum_engine(Box::new(StateVecEngine::new(4))),
            [true, false, true],
        ),
        "noise" => (
            builder.with_noise_model(Box::new(PassThroughNoiseModel::new())),
            [true, true, false],
        ),
        "depolarizing" => (
            builder.with_depolarizing_noise(0.1).unwrap(),
            [true, true, false],
        ),
        "qubits" => (builder.with_num_qubits(4), [true, false, true]),
        "system" => (
            builder.with_quantum_system(QuantumSystem::new_without_noise(Box::new(
                StateVecEngine::new(4),
            ))),
            [true, false, false],
        ),
        "seed" => (builder.with_seed(81), [true, true, true]),
        "workers" => (builder.with_default_workers(3), [true, true, true]),
        _ => unreachable!(),
    };
    let mut result = builder.build().hybrid_engine_template;
    for (i, keep) in preserved.into_iter().enumerate() {
        if keep {
            assert_eq!(
                addresses(&result)[i],
                original_addresses[i],
                "{setter}: component {i} was replaced or cloned"
            );
        }
    }
    if preserved[2] {
        let noise = result.quantum_system.noise_model();
        assert_eq!(
            noise
                .as_any()
                .downcast_ref::<DepolarizingNoiseModel>()
                .unwrap()
                .probabilities(),
            (0.23, 0.23, 0.23, 0.23)
        );
        let mut rng = noise.rng().clone();
        let draws: [u64; 8] = std::array::from_fn(|_| rng.random());
        assert_eq!(draws, expected_noise_draws, "{setter}: noise RNG changed");
    }
    if preserved[1] {
        assert!(
            result
                .quantum_system
                .quantum_engine()
                .as_any()
                .is::<StabVecEngine>()
        );
        assert_quantum_state_and_rng(&mut result, expected_quantum.as_mut(), setter);
    }
    if setter == "qubits" {
        assert_eq!(
            result
                .quantum_system
                .quantum_engine()
                .as_any()
                .downcast_ref::<StateVecEngine>()
                .unwrap()
                .simulator()
                .num_qubits(),
            4
        );
    }
}

fn assert_quantum_state_and_rng(
    result: &mut HybridEngine,
    expected_quantum: &mut dyn pecos_engines::QuantumEngine,
    setter: &str,
) {
    // Check both the existing state on qubit 3 and future measurement RNG.
    for _ in 0..32 {
        let command = ByteMessage::quantum_operations_builder()
            .h(&[0])
            .mz(&[0, 3])
            .build();
        let expected = expected_quantum
            .process(command.clone())
            .unwrap()
            .outcomes()
            .unwrap();
        let actual = result
            .quantum_system
            .engine_mut()
            .process(command)
            .unwrap()
            .outcomes()
            .unwrap();
        assert_eq!(actual, expected, "{setter}: quantum state/RNG changed");
        assert_eq!(actual[1], 1, "{setter}: existing qubit state was reset");
    }
}

#[test]
fn hybrid_builder_replaces_only_requested_quantum_system_component() {
    for noise in [true, false] {
        let original = hybrid();
        let pointers = addresses(&original);
        let builder = HybridEngineBuilder::from(original);
        let result = if noise {
            builder
                .with_noise_model(Box::new(PassThroughNoiseModel::new()))
                .build()
        } else {
            builder
                .with_quantum_engine(Box::new(StateVecEngine::new(4)))
                .build()
        };
        assert_eq!(addresses(&result)[0], pointers[0]);
        let retained = if noise { 1 } else { 2 };
        assert_eq!(addresses(&result)[retained], pointers[retained]);
    }
}

#[test]
fn hybrid_depolarizing_probability_is_fallible() {
    for p in [-0.1, 1.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let error = HybridEngineBuilder::from(hybrid())
            .with_depolarizing_noise(p)
            .err()
            .expect("invalid caller probability must return an error")
            .to_string();
        assert!(error.contains(&p.to_string()), "{error}");
        assert!(
            error.contains("supply a finite probability in 0.0..=1.0"),
            "{error}"
        );
    }
    for p in [0.0, 1.0] {
        let original = hybrid();
        let pointers = addresses(&original);
        let engine = HybridEngineBuilder::from(original)
            .with_depolarizing_noise(p)
            .unwrap()
            .build();
        assert_eq!(addresses(&engine)[0], pointers[0]);
        assert_eq!(addresses(&engine)[1], pointers[1]);
        assert_eq!(
            engine
                .quantum_system
                .noise_model()
                .as_any()
                .downcast_ref::<DepolarizingNoiseModel>()
                .unwrap()
                .probabilities(),
            (p, p, p, p)
        );
    }
}
