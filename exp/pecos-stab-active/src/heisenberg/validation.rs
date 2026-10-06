// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

use super::{HeisenbergOp, HeisenbergProgram};
use std::fmt;

/// An invalid program produced by a builder or rewrite.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ProgramError {
    /// An operation reads a measurement symbol before its producer.
    Causality {
        /// Zero-based operation index.
        operation: usize,
        /// Measurement symbol read too early.
        symbol: usize,
    },
    /// A measurement reuses a symbol already produced by another measurement.
    DuplicateSymbol {
        /// Zero-based index of the second producing operation.
        operation: usize,
        /// Duplicated measurement symbol.
        symbol: usize,
    },
    /// No measurement produces a declared symbol.
    MissingSymbol {
        /// Missing measurement symbol.
        symbol: usize,
    },
    /// A visible measurement reuses an occupied record ordinal.
    DuplicateOrdinal {
        /// Zero-based index of the second producing operation.
        operation: usize,
        /// Duplicated record ordinal.
        ordinal: usize,
    },
    /// No visible measurement produces a declared record.
    MissingOrdinal {
        /// Missing record ordinal.
        ordinal: usize,
    },
    /// An operation's sign references an undeclared measurement symbol.
    MeasurementReferenceOutOfRange {
        /// Zero-based operation index.
        operation: usize,
        /// Referenced measurement symbol.
        symbol: usize,
        /// Exclusive upper bound on measurement symbols.
        limit: usize,
    },
    /// A measurement produces an undeclared symbol.
    MeasurementSymbolOutOfRange {
        /// Zero-based operation index.
        operation: usize,
        /// Produced measurement symbol.
        symbol: usize,
        /// Exclusive upper bound on measurement symbols.
        limit: usize,
    },
    /// A visible measurement produces an undeclared record ordinal.
    RecordOrdinalOutOfRange {
        /// Zero-based operation index.
        operation: usize,
        /// Produced record ordinal.
        ordinal: usize,
        /// Exclusive upper bound on record ordinals.
        limit: usize,
    },
    /// An operation's sign references an undeclared noise symbol.
    NoiseSymbolOutOfRange {
        /// Zero-based operation index.
        operation: usize,
        /// Referenced noise symbol.
        symbol: usize,
        /// Exclusive upper bound on noise symbols.
        limit: usize,
    },
    /// A channel's component bits extend past the declared noise symbols.
    NoiseChannelSpanOutOfRange {
        /// Zero-based noise channel index.
        channel: usize,
        /// First noise symbol in this channel's span.
        first_symbol: usize,
        /// Number of support qubits, each requiring two component bits.
        num_qubits: usize,
        /// Exclusive upper bound on noise symbols.
        limit: usize,
    },
    /// A noise channel has no alternatives to sample.
    EmptyNoiseAlternatives {
        /// Zero-based noise channel index.
        channel: usize,
    },
    /// A channel's probability total is not finite and positive.
    InvalidNoiseProbabilityTotal {
        /// Zero-based noise channel index.
        channel: usize,
    },
    /// An alternative's probability is not finite and nonnegative.
    InvalidNoiseProbability {
        /// Zero-based noise channel index.
        channel: usize,
        /// Zero-based alternative index within the channel.
        alternative: usize,
    },
    /// An alternative's component bits do not fill exactly its channel's span.
    NoiseAlternativeLength {
        /// Zero-based noise channel index.
        channel: usize,
        /// Zero-based alternative index within the channel.
        alternative: usize,
        /// Required number of component bits, twice the support size.
        expected: usize,
        /// Stored number of component bits.
        actual: usize,
    },
    /// A detector references an undeclared record ordinal.
    DetectorOrdinalOutOfRange {
        /// Zero-based detector annotation index.
        detector: usize,
        /// Referenced record ordinal.
        ordinal: usize,
        /// Exclusive upper bound on record ordinals.
        limit: usize,
    },
    /// An observable references an undeclared record ordinal.
    ObservableOrdinalOutOfRange {
        /// Zero-based observable annotation index.
        observable: usize,
        /// Referenced record ordinal.
        ordinal: usize,
        /// Exclusive upper bound on record ordinals.
        limit: usize,
    },
    /// A virtual Pauli factor names a qubit outside the register.
    FactorQubitOutOfRange {
        /// Zero-based operation index.
        operation: usize,
        /// Referenced factor qubit.
        qubit: usize,
        /// Exclusive upper bound on register qubits.
        limit: usize,
    },
}

impl fmt::Display for ProgramError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Causality { operation, symbol } => write!(
                f,
                "operation {operation} reads measurement symbol {symbol} before it is produced"
            ),
            Self::DuplicateSymbol { operation, symbol } => write!(
                f,
                "operation {operation} duplicates measurement symbol {symbol}"
            ),
            Self::MissingSymbol { symbol } => write!(f, "missing measurement symbol {symbol}"),
            Self::DuplicateOrdinal { operation, ordinal } => write!(
                f,
                "operation {operation} duplicates record ordinal {ordinal}"
            ),
            Self::MissingOrdinal { ordinal } => write!(f, "missing record ordinal {ordinal}"),
            Self::MeasurementReferenceOutOfRange {
                operation,
                symbol,
                limit,
            } => write!(
                f,
                "operation {operation} references measurement symbol {symbol} outside 0..{limit}"
            ),
            Self::MeasurementSymbolOutOfRange {
                operation,
                symbol,
                limit,
            } => write!(
                f,
                "operation {operation} produces measurement symbol {symbol} outside 0..{limit}"
            ),
            Self::RecordOrdinalOutOfRange {
                operation,
                ordinal,
                limit,
            } => write!(
                f,
                "operation {operation} produces record ordinal {ordinal} outside 0..{limit}"
            ),
            Self::NoiseSymbolOutOfRange {
                operation,
                symbol,
                limit,
            } => write!(
                f,
                "operation {operation} references noise symbol {symbol} outside 0..{limit}"
            ),
            Self::NoiseChannelSpanOutOfRange {
                channel,
                first_symbol,
                num_qubits,
                limit,
            } => write!(
                f,
                "noise channel {channel} span starting at {first_symbol} for {num_qubits} qubits exceeds {limit} noise symbols"
            ),
            Self::EmptyNoiseAlternatives { channel } => {
                write!(f, "noise channel {channel} has no alternatives")
            }
            Self::InvalidNoiseProbabilityTotal { channel } => write!(
                f,
                "noise channel {channel} probability total must be finite and positive"
            ),
            Self::InvalidNoiseProbability {
                channel,
                alternative,
            } => write!(
                f,
                "noise channel {channel} alternative {alternative} probability must be finite and nonnegative"
            ),
            Self::NoiseAlternativeLength {
                channel,
                alternative,
                expected,
                actual,
            } => write!(
                f,
                "noise channel {channel} alternative {alternative} has {actual} component bits, expected {expected}"
            ),
            Self::DetectorOrdinalOutOfRange {
                detector,
                ordinal,
                limit,
            } => write!(
                f,
                "detector {detector} references record ordinal {ordinal} outside 0..{limit}"
            ),
            Self::ObservableOrdinalOutOfRange {
                observable,
                ordinal,
                limit,
            } => write!(
                f,
                "observable {observable} references record ordinal {ordinal} outside 0..{limit}"
            ),
            Self::FactorQubitOutOfRange {
                operation,
                qubit,
                limit,
            } => write!(
                f,
                "operation {operation} has Pauli factor qubit {qubit} outside 0..{limit}"
            ),
        }
    }
}

impl std::error::Error for ProgramError {}

impl HeisenbergProgram {
    pub(super) fn validate(&self) -> Result<(), ProgramError> {
        let mut symbols = vec![false; self.num_measurements];
        let mut ordinals = vec![false; self.num_records];
        for (operation, op) in self.operations.iter().enumerate() {
            let (pauli, sign) = match op {
                HeisenbergOp::Rotation { pauli, sign, .. }
                | HeisenbergOp::Measurement { pauli, sign, .. } => (pauli, sign),
            };
            for &symbol in &sign.measurements {
                if symbol >= self.num_measurements {
                    return Err(ProgramError::MeasurementReferenceOutOfRange {
                        operation,
                        symbol,
                        limit: self.num_measurements,
                    });
                }
                if !symbols[symbol] {
                    return Err(ProgramError::Causality { operation, symbol });
                }
            }
            for &symbol in &sign.noise {
                if symbol >= self.num_noise_symbols {
                    return Err(ProgramError::NoiseSymbolOutOfRange {
                        operation,
                        symbol,
                        limit: self.num_noise_symbols,
                    });
                }
            }
            for &(qubit, _) in pauli.factors() {
                if qubit >= self.num_qubits {
                    return Err(ProgramError::FactorQubitOutOfRange {
                        operation,
                        qubit,
                        limit: self.num_qubits,
                    });
                }
            }
            if let HeisenbergOp::Measurement { symbol, record, .. } = op {
                if *symbol >= self.num_measurements {
                    return Err(ProgramError::MeasurementSymbolOutOfRange {
                        operation,
                        symbol: *symbol,
                        limit: self.num_measurements,
                    });
                }
                if symbols[*symbol] {
                    return Err(ProgramError::DuplicateSymbol {
                        operation,
                        symbol: *symbol,
                    });
                }
                symbols[*symbol] = true;
                if let Some(ordinal) = record {
                    if *ordinal >= self.num_records {
                        return Err(ProgramError::RecordOrdinalOutOfRange {
                            operation,
                            ordinal: *ordinal,
                            limit: self.num_records,
                        });
                    }
                    if ordinals[*ordinal] {
                        return Err(ProgramError::DuplicateOrdinal {
                            operation,
                            ordinal: *ordinal,
                        });
                    }
                    ordinals[*ordinal] = true;
                }
            }
        }
        if let Some(symbol) = symbols.iter().position(|&seen| !seen) {
            return Err(ProgramError::MissingSymbol { symbol });
        }
        if let Some(ordinal) = ordinals.iter().position(|&seen| !seen) {
            return Err(ProgramError::MissingOrdinal { ordinal });
        }
        for (channel, noise) in self.noise_channels.iter().enumerate() {
            let end = noise
                .qubits
                .len()
                .checked_mul(2)
                .and_then(|width| noise.first_symbol.checked_add(width));
            if end.is_none_or(|end| end > self.num_noise_symbols) {
                return Err(ProgramError::NoiseChannelSpanOutOfRange {
                    channel,
                    first_symbol: noise.first_symbol,
                    num_qubits: noise.qubits.len(),
                    limit: self.num_noise_symbols,
                });
            }
            if noise.alternatives.is_empty() {
                return Err(ProgramError::EmptyNoiseAlternatives { channel });
            }
            let width = 2 * noise.qubits.len();
            for (alternative, (probability, values)) in noise.alternatives.iter().enumerate() {
                if !probability.is_finite() || *probability < 0.0 {
                    return Err(ProgramError::InvalidNoiseProbability {
                        channel,
                        alternative,
                    });
                }
                if values.len() != width {
                    return Err(ProgramError::NoiseAlternativeLength {
                        channel,
                        alternative,
                        expected: width,
                        actual: values.len(),
                    });
                }
            }
            let total: f64 = noise.alternatives.iter().map(|(p, _)| p).sum();
            if !total.is_finite() || total <= 0.0 {
                return Err(ProgramError::InvalidNoiseProbabilityTotal { channel });
            }
        }
        for (detector, ordinals) in self.detectors.iter().enumerate() {
            for &ordinal in ordinals {
                if ordinal >= self.num_records {
                    return Err(ProgramError::DetectorOrdinalOutOfRange {
                        detector,
                        ordinal,
                        limit: self.num_records,
                    });
                }
            }
        }
        for (observable, ordinals) in self.observables.iter().enumerate() {
            for &ordinal in ordinals {
                if ordinal >= self.num_records {
                    return Err(ProgramError::ObservableOrdinalOutOfRange {
                        observable,
                        ordinal,
                        limit: self.num_records,
                    });
                }
            }
        }
        Ok(())
    }
}
