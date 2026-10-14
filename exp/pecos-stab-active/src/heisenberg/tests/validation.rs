// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

use super::*;

fn noisy_program() -> HeisenbergProgram {
    let mut circuit = TickCircuit::new();
    circuit.tick().channel(channel::BitFlip(0.2, 0));
    circuit.tick().mz(&[0]);
    HeisenbergProgram::compile(&circuit).unwrap()
}

#[test]
fn noise_symbol_range() {
    let mut program = noisy_program();
    let HeisenbergOp::Measurement { sign, .. } = &mut program.operations[0] else {
        panic!("expected measurement");
    };
    sign.noise = vec![2];
    assert_eq!(
        program.validate(),
        Err(ProgramError::NoiseSymbolOutOfRange {
            operation: 0,
            symbol: 2,
            limit: 2,
        })
    );
}

#[test]
fn channel_span_range() {
    let mut program = noisy_program();
    for first_symbol in [1, usize::MAX] {
        program.noise_channels[0].first_symbol = first_symbol;
        assert_eq!(
            program.validate(),
            Err(ProgramError::NoiseChannelSpanOutOfRange {
                channel: 0,
                first_symbol,
                num_qubits: 1,
                limit: 2,
            })
        );
    }
}

fn two_noise_channels() -> HeisenbergProgram {
    let mut circuit = TickCircuit::new();
    circuit.tick().channel(channel::BitFlip(0.2, 0));
    circuit.tick().channel(channel::BitFlip(0.3, 0));
    circuit.tick().mz(&[0]);
    HeisenbergProgram::compile(&circuit).unwrap()
}

#[test]
fn empty_noise_alternatives() {
    let mut program = two_noise_channels();
    program.noise_channels[1].alternatives.clear();
    assert_eq!(
        program.validate(),
        Err(ProgramError::EmptyNoiseAlternatives { channel: 1 })
    );
}

#[test]
fn noise_probability_total() {
    let mut program = two_noise_channels();
    for probability in [0.0, f64::MAX] {
        for (weight, _) in &mut program.noise_channels[1].alternatives {
            *weight = probability;
        }
        assert_eq!(
            program.validate(),
            Err(ProgramError::InvalidNoiseProbabilityTotal { channel: 1 })
        );
    }
}

#[test]
fn noise_alternative_probability() {
    let mut program = two_noise_channels();
    for probability in [-0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        program.noise_channels[1].alternatives[1].0 = probability;
        assert_eq!(
            program.validate(),
            Err(ProgramError::InvalidNoiseProbability {
                channel: 1,
                alternative: 1,
            })
        );
    }
}

#[test]
fn noise_alternative_length() {
    let mut program = two_noise_channels();
    // Lengths 3 and 4 fit the total noise vector but overwrite the next channel.
    for actual in [0, 1, 3, 4, 5] {
        program.noise_channels[0].alternatives[1]
            .1
            .resize(actual, false);
        assert_eq!(
            program.validate(),
            Err(ProgramError::NoiseAlternativeLength {
                channel: 0,
                alternative: 1,
                expected: 2,
                actual,
            })
        );
    }
    // Zero-probability alternatives still have to describe the correct span.
    program.noise_channels[0].alternatives[1].0 = 0.0;
    assert_eq!(
        program.validate(),
        Err(ProgramError::NoiseAlternativeLength {
            channel: 0,
            alternative: 1,
            expected: 2,
            actual: 5,
        })
    );
}

#[test]
fn valid_noise_tables_run_and_match_oracle() {
    use super::joint_distribution::{Distribution, assert_distribution, with_noise};
    for probability in [f64::from_bits(1), 3.0, f64::MAX] {
        let mut program = two_noise_channels();
        program.noise_channels[0].alternatives[0].0 = 0.0;
        program.noise_channels[0].alternatives[1].0 = probability;
        program.noise_channels[1].alternatives[0].0 = 2.0;
        program.noise_channels[1].alternatives[1].0 = 0.0;
        let program = program
            .clone()
            .with_operations(program.operations.clone())
            .unwrap();
        for seed in 0..64 {
            assert_eq!(program.run(seed).records, [true]);
        }
        assert_distribution(
            &with_noise(&program).records,
            &Distribution::from([(vec![true], 1.0)]),
        );
    }
    let mut program = two_noise_channels();
    for (channel, weights) in program
        .noise_channels
        .iter_mut()
        .zip([[1.0, 3.0], [3.0, 1.0]])
    {
        for ((probability, _), weight) in channel.alternatives.iter_mut().zip(weights) {
            *probability = weight;
        }
    }
    let program = program
        .clone()
        .with_operations(program.operations.clone())
        .unwrap();
    for seed in 0..64 {
        let _ = program.run(seed);
    }
    // XOR of independent faults with probabilities 3/4 and 1/4.
    assert_distribution(
        &with_noise(&program).records,
        &Distribution::from([(vec![false], 0.375), (vec![true], 0.625)]),
    );
}

#[test]
fn detector_ordinal_range() {
    let mut program = noisy_program();
    program.detectors = vec![vec![1]];
    assert_eq!(
        program.validate(),
        Err(ProgramError::DetectorOrdinalOutOfRange {
            detector: 0,
            ordinal: 1,
            limit: 1,
        })
    );
}

#[test]
fn observable_ordinal_range() {
    let mut program = noisy_program();
    program.observables = vec![vec![1]];
    assert_eq!(
        program.validate(),
        Err(ProgramError::ObservableOrdinalOutOfRange {
            observable: 0,
            ordinal: 1,
            limit: 1,
        })
    );
}

#[test]
fn factor_qubit_range() {
    let mut program = noisy_program();
    let HeisenbergOp::Measurement { pauli, .. } = &mut program.operations[0] else {
        panic!("expected measurement");
    };
    pauli.factors[0].0 = 1;
    assert_eq!(
        program.validate(),
        Err(ProgramError::FactorQubitOutOfRange {
            operation: 0,
            qubit: 1,
            limit: 1,
        })
    );
}

#[test]
fn produced_symbol_range() {
    let mut program = noisy_program();
    let HeisenbergOp::Measurement { symbol, .. } = &mut program.operations[0] else {
        panic!("expected measurement");
    };
    *symbol = 1;
    assert_eq!(
        program.validate(),
        Err(ProgramError::MeasurementSymbolOutOfRange {
            operation: 0,
            symbol: 1,
            limit: 1,
        })
    );
}

#[test]
fn produced_ordinal_range() {
    let mut program = noisy_program();
    let HeisenbergOp::Measurement { record, .. } = &mut program.operations[0] else {
        panic!("expected measurement");
    };
    *record = Some(1);
    assert_eq!(
        program.validate(),
        Err(ProgramError::RecordOrdinalOutOfRange {
            operation: 0,
            ordinal: 1,
            limit: 1,
        })
    );
}

#[test]
fn rotations_obey_causality_and_ranges() {
    let mut circuit = TickCircuit::new();
    circuit.tick().pz(&[0]);
    circuit.tick().ry(Angle64::from_radians(0.31), &[0]);
    let program = HeisenbergProgram::compile(&circuit).unwrap();
    let mut operations = program.operations.clone();
    operations.swap(0, 1);
    assert_eq!(
        program.clone().with_operations(operations).unwrap_err(),
        ProgramError::Causality {
            operation: 0,
            symbol: 0
        }
    );
    let mut operations = program.operations.clone();
    let HeisenbergOp::Rotation { sign, .. } = &mut operations[1] else {
        panic!("expected rotation");
    };
    sign.noise = vec![0];
    assert_eq!(
        program.clone().with_operations(operations).unwrap_err(),
        ProgramError::NoiseSymbolOutOfRange {
            operation: 1,
            symbol: 0,
            limit: 0
        }
    );
}

#[test]
fn measurement_cannot_reference_itself() {
    let program = noisy_program();
    let mut operations = program.operations.clone();
    let HeisenbergOp::Measurement { sign, .. } = &mut operations[0] else {
        panic!("expected measurement");
    };
    sign.measurements = vec![0];
    assert_eq!(
        program.clone().with_operations(operations).unwrap_err(),
        ProgramError::Causality {
            operation: 0,
            symbol: 0
        }
    );
}

#[test]
fn error_messages_identify_distinct_faults() {
    let cases = [
        (
            ProgramError::Causality {
                operation: 2,
                symbol: 3,
            },
            "operation 2 reads measurement symbol 3 before it is produced",
        ),
        (
            ProgramError::DuplicateSymbol {
                operation: 2,
                symbol: 3,
            },
            "operation 2 duplicates measurement symbol 3",
        ),
        (
            ProgramError::MissingSymbol { symbol: 3 },
            "missing measurement symbol 3",
        ),
        (
            ProgramError::DuplicateOrdinal {
                operation: 2,
                ordinal: 3,
            },
            "operation 2 duplicates record ordinal 3",
        ),
        (
            ProgramError::MissingOrdinal { ordinal: 3 },
            "missing record ordinal 3",
        ),
        (
            ProgramError::MeasurementReferenceOutOfRange {
                operation: 2,
                symbol: 3,
                limit: 1,
            },
            "operation 2 references measurement symbol 3 outside 0..1",
        ),
        (
            ProgramError::MeasurementSymbolOutOfRange {
                operation: 2,
                symbol: 3,
                limit: 1,
            },
            "operation 2 produces measurement symbol 3 outside 0..1",
        ),
        (
            ProgramError::RecordOrdinalOutOfRange {
                operation: 2,
                ordinal: 3,
                limit: 1,
            },
            "operation 2 produces record ordinal 3 outside 0..1",
        ),
        (
            ProgramError::NoiseSymbolOutOfRange {
                operation: 2,
                symbol: 3,
                limit: 1,
            },
            "operation 2 references noise symbol 3 outside 0..1",
        ),
        (
            ProgramError::NoiseChannelSpanOutOfRange {
                channel: 2,
                first_symbol: 3,
                num_qubits: 4,
                limit: 1,
            },
            "noise channel 2 span starting at 3 for 4 qubits exceeds 1 noise symbols",
        ),
        (
            ProgramError::EmptyNoiseAlternatives { channel: 2 },
            "noise channel 2 has no alternatives",
        ),
        (
            ProgramError::InvalidNoiseProbabilityTotal { channel: 2 },
            "noise channel 2 probability total must be finite and positive",
        ),
        (
            ProgramError::InvalidNoiseProbability {
                channel: 2,
                alternative: 3,
            },
            "noise channel 2 alternative 3 probability must be finite and nonnegative",
        ),
        (
            ProgramError::NoiseAlternativeLength {
                channel: 2,
                alternative: 3,
                expected: 4,
                actual: 1,
            },
            "noise channel 2 alternative 3 has 1 component bits, expected 4",
        ),
        (
            ProgramError::DetectorOrdinalOutOfRange {
                detector: 2,
                ordinal: 3,
                limit: 1,
            },
            "detector 2 references record ordinal 3 outside 0..1",
        ),
        (
            ProgramError::ObservableOrdinalOutOfRange {
                observable: 2,
                ordinal: 3,
                limit: 1,
            },
            "observable 2 references record ordinal 3 outside 0..1",
        ),
        (
            ProgramError::FactorQubitOutOfRange {
                operation: 2,
                qubit: 3,
                limit: 1,
            },
            "operation 2 has Pauli factor qubit 3 outside 0..1",
        ),
    ];
    let mut messages = std::collections::BTreeSet::new();
    for (error, expected) in cases {
        let message = error.to_string();
        assert_eq!(message, expected);
        assert!(messages.insert(message));
    }
}
