// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

use pecos_core::{Gate, QubitId};
use pecos_engines::quantum::SparseStabEngine;
use pecos_engines::{ByteMessage, DenseStateVecEngine, DensityMatrixEngine, Engine};

/// MX measures and leaves the qubit in the measured X eigenstate, so a second
/// MX without a reset repeats the first outcome. Each engine below dispatches
/// through a different gate match.
fn check_engine_mx<E: Engine<Input = ByteMessage, Output = ByteMessage>>(
    mut engine: E,
    minus: bool,
) {
    let mut message = ByteMessage::quantum_operations_builder();
    message.h(&[0]);
    if minus {
        message.z(&[0]);
    }
    message.add_gate_command(&Gate::mx(&[QubitId(0)]));
    message.add_gate_command(&Gate::mx(&[QubitId(0)]));
    let outcomes = engine.process(message.build()).unwrap().outcomes().unwrap();
    assert_eq!(outcomes, [u32::from(minus), u32::from(minus)]);
}

#[test]
fn mx_clifford_engine_restores_x_eigenstate() {
    for seed in 0..16 {
        for minus in [false, true] {
            check_engine_mx(SparseStabEngine::with_seed(1, seed), minus);
        }
    }
}

#[test]
fn mx_general_engine_restores_x_eigenstate() {
    for seed in 0..16 {
        for minus in [false, true] {
            check_engine_mx(DensityMatrixEngine::with_seed(1, seed), minus);
        }
    }
}

#[test]
fn mx_state_vector_engine_restores_x_eigenstate() {
    for seed in 0..16 {
        for minus in [false, true] {
            check_engine_mx(DenseStateVecEngine::with_seed(1, seed), minus);
        }
    }
}
