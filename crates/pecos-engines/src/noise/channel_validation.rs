//! Validation shared by the public gate-hook channel constructors.
//!
//! Legacy weighted-noise configuration continues to use its existing validation paths.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Invalid gate-hook channel configuration, with the offending field retained.
#[derive(Clone, Debug, PartialEq)]
pub enum ChannelValidationError {
    /// An outer application probability is not finite or outside `[0, 1]`.
    InvalidProbability(f64),
    /// A distribution or conditional row is empty.
    EmptyDistribution,
    /// A state/event/transition label has invalid symbols or syntax.
    InvalidLabel(String),
    /// A label or dictionary has the wrong number of operands.
    InvalidArity { expected: usize, actual: usize },
    /// A conditional/event weight is not finite or outside `[0, 1]`.
    InvalidWeight { label: String, value: f64 },
    /// A conditional row or event distribution does not sum to one.
    NotNormalized { source: Option<String>, total: f64 },
    /// Missing source rows by leakage pattern, where C denotes a computational leg.
    IncompleteComputationalRows(BTreeMap<String, Vec<String>>),
    /// Wildcard rules move the identity wire or mix with concrete rows.
    ConflictingWildcard(String),
}

impl fmt::Display for ChannelValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProbability(p) => write!(
                f,
                "channel probability must be finite and between 0 and 1, got {p}"
            ),
            Self::EmptyDistribution => write!(
                f,
                "channel distribution and supplied source rows cannot be empty"
            ),
            Self::InvalidLabel(label) => write!(
                f,
                "invalid channel label {label:?}; transitions use 'i->j' over 0, 1, L (or a preserved two-qubit '*'); Pauli events use I, X, Y, Z, L"
            ),
            Self::InvalidArity { expected, actual } => {
                write!(f, "channel requires arity {expected}, got {actual}")
            }
            Self::InvalidWeight { label, value } => write!(
                f,
                "weight for {label:?} must be finite and between 0 and 1, got {value}"
            ),
            Self::NotNormalized { source, total } => write!(
                f,
                "channel probabilities for {} must sum to one; total {total} deviates from 1.0",
                source.as_deref().unwrap_or("all events")
            ),
            Self::IncompleteComputationalRows(patterns) => {
                write!(f, "incomplete computational rows (C denotes 0 or 1): ")?;
                for (pattern, missing) in patterns {
                    write!(
                        f,
                        "pattern {pattern}: missing source rows {}; ",
                        missing.join(", ")
                    )?;
                }
                write!(
                    f,
                    "supply every computational completion of each leakage pattern. The channel performs MZ when it fires on a matching computational source; explicit identity rows follow measurement and are not the identity quantum map. Add the missing rows explicitly, for example: "
                )?;
                for source in patterns.values().flatten() {
                    write!(f, "'{source}->{source}': 1.0; ")?;
                }
                Ok(())
            }
            Self::ConflictingWildcard(label) => write!(
                f,
                "conflicting wildcard rule {label:?}: each source and destination must contain exactly one '*' in the same position; concrete and wildcard groups cannot overlap"
            ),
        }
    }
}

impl std::error::Error for ChannelValidationError {}

pub(crate) fn probability(p: f64) -> Result<(), ChannelValidationError> {
    if p.is_finite() && (0.0..=1.0).contains(&p) {
        Ok(())
    } else {
        Err(ChannelValidationError::InvalidProbability(p))
    }
}

pub(crate) fn arity(expected: usize, actual: usize) -> Result<(), ChannelValidationError> {
    if expected == actual {
        Ok(())
    } else {
        Err(ChannelValidationError::InvalidArity { expected, actual })
    }
}

fn distribution(
    row: &BTreeMap<String, f64>,
    source: Option<&str>,
) -> Result<(), ChannelValidationError> {
    if row.is_empty() {
        return Err(ChannelValidationError::EmptyDistribution);
    }
    for (label, &value) in row {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(ChannelValidationError::InvalidWeight {
                label: label.clone(),
                value,
            });
        }
    }
    let total: f64 = row.values().sum();
    if (total - 1.0).abs() > super::weighted_sampler::NORMALIZATION_TOLERANCE {
        return Err(ChannelValidationError::NotNormalized {
            source: source.map(str::to_string),
            total,
        });
    }
    Ok(())
}

pub(crate) fn events(events: &BTreeMap<String, f64>) -> Result<(), ChannelValidationError> {
    distribution(events, None)?;
    let count = events
        .keys()
        .next()
        .expect("validated nonempty distribution")
        .chars()
        .count();
    for label in events.keys() {
        arity(count, label.chars().count())?;
        if label.is_empty()
            || !label
                .chars()
                .all(|c| matches!(c, 'I' | 'X' | 'Y' | 'Z' | 'L'))
        {
            return Err(ChannelValidationError::InvalidLabel(label.clone()));
        }
    }
    Ok(())
}

pub(crate) fn transitions(
    rows: &BTreeMap<String, BTreeMap<String, f64>>,
) -> Result<(), ChannelValidationError> {
    let Some(first) = rows.keys().next() else {
        return Err(ChannelValidationError::EmptyDistribution);
    };
    let count = first.chars().count();
    let wildcard = rows
        .iter()
        .any(|(source, row)| source.contains('*') || row.keys().any(|dest| dest.contains('*')));
    let mut factors: [BTreeMap<String, BTreeMap<String, f64>>; 2] =
        std::array::from_fn(|_| BTreeMap::new());
    for (source, row) in rows {
        distribution(row, Some(source))?;
        for label in std::iter::once(source).chain(row.keys()) {
            arity(count, label.chars().count())?;
            if label.is_empty() || !label.chars().all(|c| matches!(c, '0' | '1' | 'L' | '*')) {
                return Err(ChannelValidationError::InvalidLabel(label.clone()));
            }
            if wildcard {
                arity(2, count)?;
                if label.chars().filter(|&c| c == '*').count() != 1
                    || label.find('*') != source.find('*')
                {
                    return Err(ChannelValidationError::ConflictingWildcard(label.clone()));
                }
            }
        }
        if wildcard {
            let leg = usize::from(source.starts_with('*'));
            factors[leg].insert(
                source.replace('*', ""),
                row.iter()
                    .map(|(dest, &p)| (dest.replace('*', ""), p))
                    .collect(),
            );
        }
    }
    if matches!(count, 1 | 2) && !wildcard {
        let mut missing = BTreeSet::new();
        for source in rows.keys() {
            let mut completions = vec![String::new()];
            for symbol in source.chars() {
                let symbols: &[char] = if symbol == 'L' { &['L'] } else { &['0', '1'] };
                completions = completions
                    .into_iter()
                    .flat_map(|prefix| {
                        symbols
                            .iter()
                            .map(move |symbol| format!("{prefix}{symbol}"))
                    })
                    .collect();
            }
            missing.extend(
                completions
                    .into_iter()
                    .filter(|row| !rows.contains_key(row)),
            );
        }
        if !missing.is_empty() {
            let mut patterns: BTreeMap<String, Vec<String>> = BTreeMap::new();
            for source in missing {
                let pattern = source
                    .chars()
                    .map(|symbol| if symbol == 'L' { 'L' } else { 'C' })
                    .collect();
                patterns.entry(pattern).or_default().push(source);
            }
            return Err(ChannelValidationError::IncompleteComputationalRows(
                patterns,
            ));
        }
    }
    if wildcard {
        for factor in factors.iter().filter(|factor| !factor.is_empty()) {
            transitions(factor)?;
        }
    }
    Ok(())
}
