// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Empirical inference of measurement parities from independent coin-toss probes.
//!
//! This module has no compiler or runtime dependency. Callers supply probe rows
//! and bind their columns to stable measurement identities. A successful fit is
//! evidence about the sampled classical outputs, not a proof of static quantum
//! control flow or noiseless detector determinism.

use std::collections::{BTreeMap, BTreeSet};

/// Settings shared by the Rust and Python DEM annotation-inference APIs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DemInferenceSettings {
    pub raw_tag: String,
    pub detector_tag: String,
    pub observable_tags: Vec<String>,
    /// `None` derives the shot count from measurement width and a rank margin.
    pub probe_shots: Option<usize>,
    pub provenance_shots: usize,
    /// Extra consistency equations, not a held-out validation set.
    pub validation_rows: usize,
    pub require_raw_provenance: bool,
}

impl Default for DemInferenceSettings {
    fn default() -> Self {
        Self {
            raw_tag: "raw measurements".into(),
            detector_tag: "DETECTOR".into(),
            observable_tags: vec!["obs".into()],
            probe_shots: None,
            provenance_shots: 32,
            validation_rows: 32,
            require_raw_provenance: true,
        }
    }
}

impl DemInferenceSettings {
    #[must_use]
    pub fn with_raw_tag(mut self, tag: impl Into<String>) -> Self {
        self.raw_tag = tag.into();
        self
    }
    #[must_use]
    pub fn with_detector_tag(mut self, tag: impl Into<String>) -> Self {
        self.detector_tag = tag.into();
        self
    }
    #[must_use]
    pub fn with_observable_tags(
        mut self,
        tags: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.observable_tags = tags.into_iter().map(Into::into).collect();
        self
    }
    #[must_use]
    pub fn with_probe_shots(mut self, shots: usize) -> Self {
        self.probe_shots = Some(shots);
        self
    }
    #[must_use]
    pub fn with_provenance_shots(mut self, shots: usize) -> Self {
        self.provenance_shots = shots;
        self
    }
    #[must_use]
    pub fn with_validation_rows(mut self, rows: usize) -> Self {
        self.validation_rows = rows;
        self
    }
    #[must_use]
    pub fn with_require_raw_provenance(mut self, required: bool) -> Self {
        self.require_raw_provenance = required;
        self
    }

    /// Validate settings before executing a program.
    ///
    /// # Errors
    /// Returns an error for empty/overlapping tags or invalid shot counts.
    pub fn validate(&self) -> Result<(), DemInferenceError> {
        if self.probe_shots == Some(0) || self.provenance_shots < 2 || self.validation_rows < 1 {
            return Err(DemInferenceError::InvalidSettings(
                "probe_shots must be positive, provenance_shots at least 2, and validation_rows at least 1".into(),
            ));
        }
        let tags: Vec<_> = std::iter::once(&self.raw_tag)
            .chain(std::iter::once(&self.detector_tag))
            .chain(&self.observable_tags)
            .collect();
        if self.observable_tags.is_empty() || tags.iter().any(|tag| tag.is_empty()) {
            return Err(DemInferenceError::InvalidSettings(
                "result tags must be nonempty, including observable_tags".into(),
            ));
        }
        if tags.iter().collect::<BTreeSet<_>>().len() != tags.len() {
            return Err(DemInferenceError::InvalidSettings(
                "raw, detector, and observable tags must be distinct".into(),
            ));
        }
        Ok(())
    }

    /// Resolve the automatic probe count, including 32 rows of rank margin.
    ///
    /// # Errors
    /// Returns an error for invalid settings or an overflowing row count.
    pub fn resolved_probe_shots(&self, raw_width: usize) -> Result<usize, DemInferenceError> {
        self.validate()?;
        self.probe_shots.map_or_else(
            || {
                raw_width
                    .checked_add(1)
                    .and_then(|n| n.checked_add(self.validation_rows))
                    .and_then(|n| n.checked_add(32))
                    .ok_or_else(|| {
                        DemInferenceError::InvalidSettings("probe row count overflow".into())
                    })
            },
            Ok,
        )
    }
}

/// A failed empirical fit or invalid measurement binding.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum DemInferenceError {
    #[error("{0}")]
    InvalidSettings(String),
    #[error("{0}")]
    InvalidData(String),
    #[error("affine inference needs at least {required} shots; got {actual}")]
    InsufficientRows { required: usize, actual: usize },
    #[error(
        "coin-toss probes have GF(2) rank {rank}; need {required}. Increase probe_shots or change the seed."
    )]
    RankDeficient { rank: usize, required: usize },
    #[error("derived Guppy outputs are not affine parities of the raw measurement record")]
    NonAffine,
    #[error("DEM annotations cannot represent affine constant-one outputs")]
    ConstantOne,
    #[error("detector and observable outputs must depend on at least one physical measurement")]
    EmptySupport,
    #[error("{0}")]
    Provenance(String),
}

fn validate_rows(rows: &[Vec<u8>], name: &str) -> Result<usize, DemInferenceError> {
    let width = rows
        .first()
        .ok_or_else(|| DemInferenceError::InvalidData(format!("{name} contains no shots")))?
        .len();
    if width == 0
        || rows
            .iter()
            .any(|row| row.len() != width || row.iter().any(|&bit| bit > 1))
    {
        return Err(DemInferenceError::InvalidData(format!(
            "{name} must contain nonempty, equally sized rows of bits"
        )));
    }
    Ok(width)
}

/// Infer `(constant, input-column support)` for each output column over GF(2).
/// All rows participate in the solve; surplus rows constrain consistency.
///
/// # Errors
/// Rejects malformed data, insufficient rank/rows, and non-affine outputs.
pub fn infer_affine_columns(
    inputs: &[Vec<u8>],
    outputs: &[Vec<u8>],
    validation_rows: usize,
) -> Result<Vec<(u8, Vec<usize>)>, DemInferenceError> {
    let input_width = validate_rows(inputs, "raw measurement records")?;
    let output_width = validate_rows(outputs, "derived output records")?;
    if inputs.len() != outputs.len() {
        return Err(DemInferenceError::InvalidData(
            "raw and derived output records contain different shot counts".into(),
        ));
    }
    if validation_rows == 0 {
        return Err(DemInferenceError::InvalidSettings(
            "validation_rows must be at least 1".into(),
        ));
    }
    let variables = input_width + 1;
    let required = variables
        .checked_add(validation_rows)
        .ok_or_else(|| DemInferenceError::InvalidSettings("probe row count overflow".into()))?;
    if inputs.len() < required {
        return Err(DemInferenceError::InsufficientRows {
            required,
            actual: inputs.len(),
        });
    }
    let mut matrix: Vec<Vec<u8>> = inputs
        .iter()
        .zip(outputs)
        .map(|(raw, derived)| {
            std::iter::once(1)
                .chain(raw.iter().copied())
                .chain(derived.iter().copied())
                .collect()
        })
        .collect();
    let mut pivots = Vec::new();
    for column in 0..variables {
        let next = pivots.len();
        if let Some(pivot) = (next..matrix.len()).find(|&row| matrix[row][column] != 0) {
            matrix.swap(next, pivot);
            let pivot_row = matrix[next].clone();
            for (row_index, row) in matrix.iter_mut().enumerate() {
                if row_index != next && row[column] != 0 {
                    for (bit, pivot_bit) in row.iter_mut().zip(&pivot_row) {
                        *bit ^= pivot_bit;
                    }
                }
            }
            pivots.push(next);
        }
    }
    if pivots.len() != variables {
        return Err(DemInferenceError::RankDeficient {
            rank: pivots.len(),
            required: variables,
        });
    }
    if matrix.iter().any(|row| {
        row[..variables].iter().all(|&b| b == 0) && row[variables..].iter().any(|&b| b != 0)
    }) {
        return Err(DemInferenceError::NonAffine);
    }
    Ok((0..output_width)
        .map(|output| {
            let support = (1..variables)
                .filter(|&column| matrix[pivots[column]][variables + output] != 0)
                .map(|column| column - 1)
                .collect();
            (matrix[pivots[0]][variables + output], support)
        })
        .collect())
}

/// Bind raw-output columns to physical IDs using unique probe signatures.
/// Callers must separately verify an unchanged quantum schedule across probes.
///
/// # Errors
/// Rejects malformed rows, signature collisions, and incomplete/non-bijective bindings.
pub fn correlate_measurement_ids(
    raw_rows: &[Vec<u8>],
    physical_rows: &[Vec<u8>],
    source_ids: &[usize],
) -> Result<Vec<usize>, DemInferenceError> {
    let width = validate_rows(raw_rows, "raw provenance records")?;
    let physical_width = validate_rows(physical_rows, "physical provenance records")?;
    if raw_rows.len() < 2
        || raw_rows.len() != physical_rows.len()
        || width != source_ids.len()
        || physical_width != width
        || source_ids.iter().collect::<BTreeSet<_>>().len() != width
    {
        return Err(DemInferenceError::Provenance("provenance records must cover every physical measurement exactly once across at least two shots".into()));
    }
    let mut signatures = BTreeMap::new();
    for (column, &id) in source_ids.iter().enumerate() {
        let signature: Vec<_> = physical_rows.iter().map(|row| row[column]).collect();
        if signatures.insert(signature, id).is_some() {
            return Err(DemInferenceError::Provenance("physical measurement signatures are ambiguous during provenance probing; increase provenance_shots".into()));
        }
    }
    let ids: Vec<_> = (0..width).map(|column| {
        let signature: Vec<_> = raw_rows.iter().map(|row| row[column]).collect();
        signatures.get(&signature).copied().ok_or_else(|| DemInferenceError::Provenance(format!("raw output element {column} is not a direct physical measurement across provenance probes")))
    }).collect::<Result<_, _>>()?;
    if ids.iter().collect::<BTreeSet<_>>().len() != width {
        return Err(DemInferenceError::Provenance(
            "raw output does not expose every physical measurement exactly once".into(),
        ));
    }
    Ok(ids)
}

/// Runtime-independent probe data. The IDs are in raw-output column order.
/// The caller is responsible for supplying independently established provenance.
#[derive(Clone, Debug)]
pub struct DemInferenceData {
    pub raw_measurement_ids: Vec<usize>,
    pub raw_rows: Vec<Vec<u8>>,
    pub detector_rows: Vec<Vec<u8>>,
    pub observable_rows: Vec<Vec<u8>>,
}

/// Inferred supports use stable measurement IDs, never runtime record positions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DemInferenceReport {
    pub raw_measurement_ids: Vec<usize>,
    pub detector_supports: Vec<Vec<usize>>,
    pub observable_supports: Vec<Vec<usize>>,
    pub probe_shots: usize,
    pub validation_rows: usize,
}

impl DemInferenceSettings {
    /// Infer detector and observable supports together, reusing one elimination.
    ///
    /// # Errors
    /// Rejects invalid settings/data, deficient rank, nonlinear outputs, offsets,
    /// and outputs that are independent of all physical measurements.
    pub fn infer(&self, data: &DemInferenceData) -> Result<DemInferenceReport, DemInferenceError> {
        self.validate()?;
        let width = validate_rows(&data.raw_rows, "raw measurement records")?;
        // An unrequested role is represented by one empty row per shot.
        let validate_outputs = |rows: &[Vec<u8>], name| {
            if !rows.is_empty() && rows.iter().all(Vec::is_empty) {
                Ok(0)
            } else {
                validate_rows(rows, name)
            }
        };
        let detectors = validate_outputs(&data.detector_rows, "detector records")?;
        validate_outputs(&data.observable_rows, "observable records")?;
        if data.raw_measurement_ids.len() != width
            || data
                .raw_measurement_ids
                .iter()
                .collect::<BTreeSet<_>>()
                .len()
                != width
        {
            return Err(DemInferenceError::InvalidData(
                "raw measurement IDs must be unique and match the raw row width".into(),
            ));
        }
        if data.detector_rows.len() != data.raw_rows.len()
            || data.observable_rows.len() != data.raw_rows.len()
        {
            return Err(DemInferenceError::InvalidData(
                "raw and derived output records contain different shot counts".into(),
            ));
        }
        if let Some(shots) = self.probe_shots
            && shots != data.raw_rows.len()
        {
            return Err(DemInferenceError::InvalidData(
                "probe_shots does not match supplied rows".into(),
            ));
        }
        let outputs: Vec<_> = data
            .detector_rows
            .iter()
            .zip(&data.observable_rows)
            .map(|(det, obs)| det.iter().chain(obs).copied().collect())
            .collect();
        let affine = infer_affine_columns(&data.raw_rows, &outputs, self.validation_rows)?;
        let supports: Vec<Vec<usize>> = affine
            .into_iter()
            .map(|(constant, support)| {
                if constant != 0 {
                    return Err(DemInferenceError::ConstantOne);
                }
                if support.is_empty() {
                    return Err(DemInferenceError::EmptySupport);
                }
                Ok(support
                    .into_iter()
                    .map(|index| data.raw_measurement_ids[index])
                    .collect())
            })
            .collect::<Result<_, _>>()?;
        Ok(DemInferenceReport {
            raw_measurement_ids: data.raw_measurement_ids.clone(),
            detector_supports: supports[..detectors].to_vec(),
            observable_supports: supports[detectors..].to_vec(),
            probe_shots: data.raw_rows.len(),
            validation_rows: self.validation_rows,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> DemInferenceData {
        let raw_rows: Vec<Vec<u8>> = (0..8)
            .map(|bits| (0..3).map(|i| (bits >> i) & 1).collect())
            .collect();
        DemInferenceData {
            raw_measurement_ids: vec![8, 2, 5],
            detector_rows: raw_rows
                .iter()
                .map(|row| vec![row[0] ^ row[2], row[1]])
                .collect(),
            observable_rows: raw_rows.iter().map(|row| vec![row[2]]).collect(),
            raw_rows,
        }
    }

    #[test]
    fn infers_reordered_ids_and_all_outputs_together() {
        let settings = DemInferenceSettings::default().with_validation_rows(2);
        let report = settings.infer(&data()).unwrap();
        assert_eq!(report.detector_supports, vec![vec![8, 5], vec![2]]);
        assert_eq!(report.observable_supports, vec![vec![5]]);
        assert_eq!(settings.resolved_probe_shots(3).unwrap(), 38);
    }

    #[test]
    fn permits_one_unrequested_output_role_and_rejects_empty_combined_output() {
        let settings = DemInferenceSettings::default().with_validation_rows(2);
        let mut inputs = data();
        inputs.detector_rows = vec![vec![]; inputs.raw_rows.len()];
        let report = settings.infer(&inputs).unwrap();
        assert_eq!(report.detector_supports, Vec::<Vec<usize>>::new());
        assert_eq!(report.observable_supports, vec![vec![5]]);
        inputs.observable_rows = vec![vec![]; inputs.raw_rows.len()];
        assert!(matches!(
            settings.infer(&inputs),
            Err(DemInferenceError::InvalidData(_))
        ));
    }

    #[test]
    fn rejects_nonlinear_constant_and_rank_deficient_data() {
        let settings = DemInferenceSettings::default().with_validation_rows(2);
        let mut nonlinear = data();
        nonlinear.detector_rows = nonlinear
            .raw_rows
            .iter()
            .map(|row| vec![row[0] & row[1]])
            .collect();
        assert_eq!(
            settings.infer(&nonlinear),
            Err(DemInferenceError::NonAffine)
        );
        let mut offset = data();
        for row in &mut offset.observable_rows {
            row[0] ^= 1;
        }
        assert_eq!(settings.infer(&offset), Err(DemInferenceError::ConstantOne));
        for row in &mut offset.observable_rows {
            row[0] = 0;
        }
        assert_eq!(
            settings.infer(&offset),
            Err(DemInferenceError::EmptySupport)
        );
        let mut rank = data();
        rank.raw_rows.fill(vec![0, 0, 0]);
        assert!(matches!(
            settings.infer(&rank),
            Err(DemInferenceError::RankDeficient { .. })
        ));
    }

    #[test]
    fn validates_settings_and_malformed_rows() {
        assert!(
            DemInferenceSettings::default()
                .with_observable_tags(["DETECTOR"])
                .validate()
                .is_err()
        );
        assert!(
            DemInferenceSettings::default()
                .with_probe_shots(0)
                .validate()
                .is_err()
        );
        assert!(
            DemInferenceSettings::default()
                .with_validation_rows(0)
                .validate()
                .is_err()
        );
        assert!(infer_affine_columns(&[], &[], 1).is_err());
        let mut malformed = data();
        malformed.raw_rows[0][0] = 2;
        assert!(DemInferenceSettings::default().infer(&malformed).is_err());
    }

    #[test]
    fn correlates_permutations_and_rejects_ambiguous_or_duplicate_columns() {
        let physical = data().raw_rows;
        let raw: Vec<_> = physical
            .iter()
            .map(|row| vec![row[2], row[0], row[1]])
            .collect();
        assert_eq!(
            correlate_measurement_ids(&raw, &physical, &[0, 1, 2]).unwrap(),
            vec![2, 0, 1]
        );
        // Results are the caller's measurement IDs, not column indices.
        assert_eq!(
            correlate_measurement_ids(&raw, &physical, &[4, 7, 9]).unwrap(),
            vec![9, 4, 7]
        );
        assert!(correlate_measurement_ids(&vec![vec![0; 3]; 8], &physical, &[0, 1, 2]).is_err());
        assert!(correlate_measurement_ids(&raw, &vec![vec![0; 3]; 8], &[0, 1, 2]).is_err());
    }
}
