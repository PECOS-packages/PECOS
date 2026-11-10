// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

use super::{DemBuildError, PreparedDem};
use pecos_decoder_core::obs_mask::ObsMask;
use std::collections::BTreeMap;

type Evaluation = (Vec<u8>, ObsMask);

fn bit(value: u8, context: &str) -> Result<u8, DemBuildError> {
    if value > 1 {
        return Err(DemBuildError::Configuration(format!(
            "{context} must be bool, 0, or 1; got {value}"
        )));
    }
    Ok(value)
}

impl PreparedDem {
    /// Detection events and observable flips relative to exact noiseless references.
    /// # Errors
    /// Rejects incorrect runtime record widths and invalid bits.
    pub fn evaluate_runtime_record(&self, values: &[u8]) -> Result<Evaluation, DemBuildError> {
        self.runtime_record_parities(values)
            .map(|raw| self.relative(raw))
    }

    /// Raw parities of a record in runtime measurement emission order.
    /// # Errors
    /// Rejects incorrect record widths and invalid bits, including unused positions.
    pub fn runtime_record_parities(&self, values: &[u8]) -> Result<Evaluation, DemBuildError> {
        if values.len() != self.schema.measurement_ids.len() {
            return Err(DemBuildError::Configuration(format!(
                "runtime measurement record has {} values; expected {}",
                values.len(),
                self.schema.measurement_ids.len()
            )));
        }
        let values = self
            .schema
            .measurement_ids
            .iter()
            .zip(values)
            .enumerate()
            .map(|(position, (&id, &value))| {
                bit(value, &format!("runtime measurement {position}")).map(|value| (id, value))
            })
            .collect::<Result<_, _>>()?;
        self.measurement_parities(&values)
    }

    /// Evaluate only the measurement IDs referenced by some definition.
    /// # Errors
    /// Rejects missing referenced IDs and invalid referenced bits; extra IDs are ignored.
    pub fn evaluate_measurements(
        &self,
        values: &BTreeMap<usize, u8>,
    ) -> Result<Evaluation, DemBuildError> {
        self.measurement_parities(values)
            .map(|raw| self.relative(raw))
    }

    /// Raw definition parities keyed by stable measurement ID.
    /// # Errors
    /// Rejects missing referenced IDs and invalid referenced bits.
    pub fn measurement_parities(
        &self,
        values: &BTreeMap<usize, u8>,
    ) -> Result<Evaluation, DemBuildError> {
        let parity = |positions: &[usize]| {
            let ids: Vec<_> = positions
                .iter()
                .map(|&position| self.schema.measurement_ids[position])
                .collect();
            let missing: Vec<_> = ids
                .iter()
                .filter(|id| !values.contains_key(id))
                .take(8)
                .collect();
            if !missing.is_empty() {
                return Err(DemBuildError::Configuration(format!(
                    "measurement values are missing referenced MeasId(s): {missing:?}"
                )));
            }
            ids.into_iter().try_fold(0, |parity, id| {
                bit(values[&id], &format!("measurement MeasId {id}")).map(|value| parity ^ value)
            })
        };
        let mut detectors = vec![0; self.schema.definitions.detectors.len()];
        for d in &self.schema.definitions.detectors {
            detectors[d.id as usize] = parity(&d.measurements)?;
        }
        let mut observables = ObsMask::new();
        for d in &self.schema.definitions.observables {
            if parity(&d.measurements)? != 0 {
                observables.set(d.id as usize);
            }
        }
        Ok((detectors, observables))
    }

    /// Evaluate one shot's named result values, skipping absent tags and hole slots.
    /// # Errors
    /// Rejects incorrect widths, invalid bound bits, contradictory aliases and missing supports.
    pub fn evaluate_results(
        &self,
        values: &BTreeMap<String, Vec<u8>>,
    ) -> Result<Evaluation, DemBuildError> {
        self.result_parities(values).map(|raw| self.relative(raw))
    }

    /// Raw parities from one shot's named result values.
    /// # Errors
    /// Rejects incorrect widths, invalid bound bits, contradictory aliases and missing supports.
    pub fn result_parities(
        &self,
        values: &BTreeMap<String, Vec<u8>>,
    ) -> Result<Evaluation, DemBuildError> {
        let mut by_id = BTreeMap::new();
        for (tag, ids) in &self.schema.named_result_measurements {
            let Some(values) = values.get(tag) else {
                continue;
            };
            if values.len() != ids.len() {
                return Err(DemBuildError::Configuration(format!(
                    "result {tag:?} has {} values; trace provenance expects {}",
                    values.len(),
                    ids.len()
                )));
            }
            for (id, &value) in ids.iter().zip(values) {
                let Some(id) = id else {
                    continue;
                };
                let value = bit(value, &format!("result {tag:?}"))?;
                if by_id
                    .insert(*id, value)
                    .is_some_and(|previous| previous != value)
                {
                    return Err(DemBuildError::Configuration(format!(
                        "result outputs disagree about measurement MeasId {id}"
                    )));
                }
            }
        }
        self.measurement_parities(&by_id)
    }

    /// Evaluate bound result columns. Row i of each column must belong to the same shot.
    /// # Errors
    /// Requires at least one bound tag, equal shot counts and valid per-shot values.
    pub fn evaluate_result_columns(
        &self,
        columns: &BTreeMap<String, Vec<Vec<u8>>>,
    ) -> Result<Vec<Evaluation>, DemBuildError> {
        self.result_column_parities(columns)
            .map(|rows| rows.into_iter().map(|raw| self.relative(raw)).collect())
    }

    /// Raw parities from bound result columns; zero shots returns an empty vector.
    /// # Errors
    /// Requires at least one bound tag, equal shot counts and valid per-shot values.
    pub fn result_column_parities(
        &self,
        columns: &BTreeMap<String, Vec<Vec<u8>>>,
    ) -> Result<Vec<Evaluation>, DemBuildError> {
        let available: BTreeMap<_, _> = self
            .schema
            .named_result_measurements
            .keys()
            .filter_map(|tag| columns.get(tag).map(|rows| (tag, rows)))
            .collect();
        let Some(rows) = available.values().next() else {
            return Err(DemBuildError::Configuration(
                "result columns do not contain any bound measurement tags".into(),
            ));
        };
        let shots = rows.len();
        let mismatched: BTreeMap<_, _> = available
            .iter()
            .filter(|(_, rows)| rows.len() != shots)
            .map(|(tag, rows)| (*tag, rows.len()))
            .collect();
        if !mismatched.is_empty() {
            return Err(DemBuildError::Configuration(format!(
                "result columns have inconsistent shot counts: {mismatched:?}"
            )));
        }
        (0..shots)
            .map(|shot| {
                self.result_parities(
                    &available
                        .iter()
                        .map(|(tag, rows)| ((*tag).clone(), rows[shot].clone()))
                        .collect(),
                )
            })
            .collect()
    }

    fn relative(&self, (mut detectors, mut observables): Evaluation) -> Evaluation {
        for definition in &self.schema.definitions.detectors {
            if definition.reference == Some(1) {
                detectors[definition.id as usize] ^= 1;
            }
        }
        for definition in &self.schema.definitions.observables {
            if definition.reference == Some(1) {
                observables.toggle(definition.id as usize);
            }
        }
        (detectors, observables)
    }
}
