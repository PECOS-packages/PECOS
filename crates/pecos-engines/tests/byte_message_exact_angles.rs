// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
use pecos_core::gate_type::GateType;
use pecos_core::{Angle64, Gate};
use pecos_engines::ByteMessage;

#[test]
fn every_fraction_round_trips_in_every_builder_path() {
    let mut fractions = vec![
        0,
        1,
        u64::MAX,
        Angle64::QUARTER_TURN.fraction(),
        Angle64::HALF_TURN.fraction(),
        Angle64::THREE_QUARTERS_TURN.fraction(),
    ];
    let mut state = 0x1234_5678_9abc_def0_u64;
    for _ in 0..1024 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        fractions.push(state);
    }
    for fraction in fractions {
        let a = Angle64::new(fraction);
        let b = -a;
        let custom = Gate::new(GateType::Custom, vec![a, b], vec![], vec![0.into()]);
        let mut builder = ByteMessage::quantum_operations_builder();
        builder
            .rz(a, &[0])
            .rxy1q(a, b, &[1])
            .rxyxy2q(a, b, &[(0, 1)])
            .rz(a, &[0, 1])
            .add_gate_command(&custom);
        let gates = builder.build().quantum_ops().unwrap();
        assert_eq!(
            gates,
            [
                Gate::rz(a, &[0]),
                Gate::rxy1q(a, b, &[1]),
                Gate::rxyxy2q(a, b, &[(0, 1)]),
                Gate::rz(a, &[0, 1]),
                custom
            ]
        );
    }
}

#[test]
fn classical_parameters_remain_f64() {
    for duration in [0.0_f64, -0.0, 1.25, f64::MIN_POSITIVE, f64::MAX] {
        let mut builder = ByteMessage::quantum_operations_builder();
        builder.idle(duration, &[0]);
        let gates = builder.build().quantum_ops().unwrap();
        assert_eq!(gates[0].params[0].to_bits(), duration.to_bits());
        assert!(gates[0].angles.is_empty());
    }
}

#[test]
fn legacy_f64_angle_batches_are_rejected() {
    let mut bytes = ByteMessage::quantum_operations_builder()
        .rz(Angle64::QUARTER_TURN, &[0])
        .build()
        .into_bytes();
    bytes[4] = 1; // Legacy version with f64 angle words.
    bytes[32..40].copy_from_slice(&std::f64::consts::FRAC_PI_2.to_le_bytes());
    let error = ByteMessage::new(&bytes)
        .quantum_ops()
        .unwrap_err()
        .to_string();
    assert!(error.contains("version 5"), "{error}");
    assert!(error.contains("rebuild"), "{error}");
}
