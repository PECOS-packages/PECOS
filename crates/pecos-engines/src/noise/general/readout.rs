//! Validated, per-flavor conditional readout distributions.

use super::GeneralNoiseModel;
use crate::noise::weighted_sampler::{NORMALIZATION_TOLERANCE, WeightedSampler};
use pecos_core::{errors::PecosError, gate_type::GateType};
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
#[repr(usize)]
enum TrueState {
    Zero,
    One,
    Leaked,
    Lost,
}

/// Conditional readout probabilities for one Z-measurement flavor.
///
/// Rows name true states (`"0"`, `"1"`, `"leaked"`, or `"lost"`); columns
/// name outcomes in the flavor's alphabet. Missing rows retain legacy readout.
/// A well-formed `lost` row can be constructed, but model building rejects it
/// until qubit loss is configured.
#[derive(Debug, Clone)]
pub struct ReadoutMatrix {
    pub(super) flavor: GateType,
    rows: [Option<WeightedSampler<u8>>; 4],
}

impl ReadoutMatrix {
    /// Validate and compile the supplied rows. Missing outcomes have weight zero.
    ///
    /// # Errors
    /// Returns an input error naming the flavor, row, and invalid value, with a
    /// suggested correction. `MPZ` shares `MZ` readout and cannot own a matrix.
    pub fn new(
        flavor: GateType,
        rows: &BTreeMap<String, BTreeMap<String, f64>>,
    ) -> Result<Self, PecosError> {
        let invalid = |row: &str, value: String, fix: &str| {
            PecosError::Input(format!(
                "readout {flavor} row {row:?}: invalid value {value}; {fix}"
            ))
        };
        let alphabet: &[u8] = match flavor {
            GateType::MZ => &[0, 1],
            GateType::MeasureZLeaked => &[0, 1, 2],
            GateType::MeasureZLost => &[0, 1, 3],
            GateType::MeasureZLeakedLost => &[0, 1, 2, 3],
            GateType::MPZ => {
                return Err(invalid(
                    "<flavor>",
                    flavor.to_string(),
                    "MPZ uses MZ's readout matrix; configure MZ",
                ));
            }
            _ => {
                return Err(invalid(
                    "<flavor>",
                    flavor.to_string(),
                    &format!(
                        "{flavor} has no readout matrix; flavors are MZ, MeasureZLeaked, MeasureZLost, MeasureZLeakedLost"
                    ),
                ));
            }
        };
        if rows.is_empty() {
            return Err(invalid(
                "<flavor>",
                "{}".into(),
                "configure at least one row",
            ));
        }
        let mut compiled = std::array::from_fn(|_| None);
        for (row, weights) in rows {
            let state = match row.as_str() {
                "0" => TrueState::Zero,
                "1" => TrueState::One,
                "leaked" => TrueState::Leaked,
                "lost" => TrueState::Lost,
                _ => {
                    return Err(invalid(
                        row,
                        format!("{row:?}"),
                        "use row 0, 1, leaked, or lost",
                    ));
                }
            };
            if weights.is_empty() {
                return Err(invalid(row, "{}".into(), "configure at least one outcome"));
            }
            let mut outcomes = BTreeMap::new();
            for (column, &probability) in weights {
                let outcome = match column.as_str() {
                    "0" => 0,
                    "1" => 1,
                    "2" => 2,
                    "3" => 3,
                    _ => u8::MAX,
                };
                if !alphabet.contains(&outcome) {
                    return Err(invalid(
                        row,
                        format!("column {column:?} = {probability}"),
                        &format!("use outcomes in the alphabet {alphabet:?}"),
                    ));
                }
                if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                    return Err(invalid(
                        row,
                        format!("column {column:?} = {probability}"),
                        "use a finite probability in [0, 1]",
                    ));
                }
                outcomes.insert(outcome, probability);
            }
            let sum: f64 = outcomes.values().sum();
            if (sum - 1.0).abs() > NORMALIZATION_TOLERANCE {
                return Err(invalid(
                    row,
                    format!("sum {sum}"),
                    &format!("make the row sum to 1 within {NORMALIZATION_TOLERANCE}"),
                ));
            }
            let sampler = WeightedSampler::try_new(&outcomes)
                .map_err(|message| invalid(row, format!("{weights:?}"), &message))?;
            compiled[state as usize] = Some(sampler);
        }
        Ok(Self {
            flavor,
            rows: compiled,
        })
    }

    pub(super) fn validate_without_loss(&self) -> Result<(), String> {
        if let Some(row) = &self.rows[TrueState::Lost as usize] {
            return Err(format!(
                "readout {} row \"lost\": invalid value {:?}; the lost row needs qubit loss, which is not configured; remove the lost row",
                self.flavor,
                row.get_weighted_map()
            ));
        }
        Ok(())
    }
}

impl GeneralNoiseModel {
    /// Whether any measurement flavor has explicitly configured readout rows.
    #[must_use]
    pub fn has_configured_readout(&self) -> bool {
        !self.readout.is_empty()
    }

    // Called only for noisy user readouts, after injected measurements are handled.
    pub(super) fn configured_readout(
        &mut self,
        flavor: GateType,
        outcome: u32,
        leaked: bool,
    ) -> Option<usize> {
        let row = if leaked {
            TrueState::Leaked
        } else if outcome == 0 {
            TrueState::Zero
        } else {
            TrueState::One
        };
        self.readout.get(&flavor)?.rows[row as usize]
            .as_ref()
            .map(|sampler| usize::from(sampler.sample(&mut self.rng)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiled_readout_weights_renormalize_within_tolerance() {
        for (weight, normalized) in [
            (0.699_995, true),
            (0.700_005, true),
            (0.699_999_999_99, false),
        ] {
            let weights = BTreeMap::from([(0, 0.3), (1, weight)]);
            let input = BTreeMap::from([(
                "0".into(),
                BTreeMap::from([("0".into(), 0.3), ("1".into(), weight)]),
            )]);
            let matrix = ReadoutMatrix::new(GateType::MZ, &input).unwrap();
            let compiled = matrix.rows[TrueState::Zero as usize].as_ref().unwrap();
            if normalized {
                assert_ne!(compiled.get_weighted_map(), &weights);
                let total = 0.3 + weight;
                assert_eq!(
                    compiled.get_weighted_map(),
                    &BTreeMap::from([(0, 0.3 / total), (1, weight / total)])
                );
            } else {
                assert_eq!(compiled.get_weighted_map(), &weights);
            }
        }
        let outside = BTreeMap::from([(
            "0".into(),
            BTreeMap::from([("0".into(), 0.3), ("1".into(), 0.699_98)]),
        )]);
        assert!(ReadoutMatrix::new(GateType::MZ, &outside).is_err());
    }
}
