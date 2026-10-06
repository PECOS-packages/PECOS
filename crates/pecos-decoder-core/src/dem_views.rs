// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Sparse detector-error-model views for XZ memory experiments.
//!
//! The initialization-detector view restricts a model to one detector family.
//! The GARI view implements the sparse change of variables from Eq. (5) of
//! Maan, Garcia-Herrero, Paler, and Savin, arXiv:2510.14060. Both transforms
//! preserve [`SparseDem`] as their interchange type.
//!
//! Inputs are canonicalized over GF(2) before grouping. Equal columns are
//! merged in first-occurrence order with
//! `prev * (1.0 - p) + p * (1.0 - prev)`. The initialization view performs
//! this merge once before projection and again afterward, matching the
//! reference's floating-point evaluation order. GARI construction finishes
//! with a deterministic, complete check of every physical unit vector; by
//! GF(2) linearity, those unit vectors establish the two model equivalences
//! for every error vector.

use std::collections::BTreeMap;
use std::ops::Range;

use crate::dem::SparseDem;
use crate::errors::DecoderError;

type Mechanism = (f64, Vec<u32>, Vec<u32>);
type ColumnKey = (Vec<u32>, Vec<u32>);

/// Basis family associated with a detector in a correlated XZ DEM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectorBasis {
    /// X-basis detector, excited by a Z error component.
    X,
    /// Z-basis detector, excited by an X error component.
    Z,
}

/// Column blocks in a GARI-transformed detector error model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GariColumnBlock {
    /// Physical Z-component columns.
    EZ,
    /// Physical X-component columns.
    EX,
    /// Physical Y-component columns.
    EY,
    /// Auxiliary Z-answer columns.
    EbarZ,
    /// Auxiliary X-answer columns.
    EbarX,
}

/// A DEM restricted to the detector family used for initialization.
#[derive(Debug, Clone)]
pub struct InitDetsView {
    /// Restricted and merged detector error model.
    pub dem: SparseDem,
    /// Original detector id for each row of `dem`.
    pub detector_index: Vec<u32>,
    original_num_detectors: usize,
}

impl InitDetsView {
    /// Gather the initialization-family bits from a full detector syndrome.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError::InvalidDimensions`] when `syndrome` does not
    /// have the detector width of the original DEM.
    pub fn project_syndrome(&self, syndrome: &[u8]) -> Result<Vec<u8>, DecoderError> {
        if syndrome.len() != self.original_num_detectors {
            return Err(DecoderError::InvalidDimensions {
                expected: self.original_num_detectors,
                actual: syndrome.len(),
            });
        }
        Ok(self
            .detector_index
            .iter()
            .map(|&detector| syndrome[detector as usize])
            .collect())
    }
}

/// Sparse GARI model with its physical-to-auxiliary column maps.
#[derive(Debug, Clone)]
pub struct GariModel {
    /// GARI-transformed DEM, with columns `[eZ | eX | eY | ebarZ | ebarX]`.
    pub dem: SparseDem,
    /// Number of detector rows in the original DEM.
    pub num_detectors: usize,
    /// Initialization basis whose logical observable is tracked.
    pub init_basis: DetectorBasis,
    /// Auxiliary block carrying the decoded answer.
    pub answer_block: GariColumnBlock,
    /// Original detector rows belonging to `init_basis`.
    pub relevant_rows: Vec<u32>,
    /// Effective priors for the columns in `answer_block`.
    pub relevant_priors: Vec<f64>,
    /// For each eY column, the index of its eZ partner.
    pub u_map: Vec<u32>,
    /// For each eY column, the index of its eX partner.
    pub v_map: Vec<u32>,
    physical_dem: SparseDem,
    num_ez: usize,
    num_ex: usize,
    num_ey: usize,
}

impl GariModel {
    /// Return the half-open column range for a GARI block.
    #[must_use]
    pub fn columns(&self, block: GariColumnBlock) -> Range<usize> {
        let ex_start = self.num_ez;
        let ey_start = ex_start + self.num_ex;
        let ebar_z_start = ey_start + self.num_ey;
        let ebar_x_start = ebar_z_start + self.num_ez;
        match block {
            GariColumnBlock::EZ => 0..ex_start,
            GariColumnBlock::EX => ex_start..ey_start,
            GariColumnBlock::EY => ey_start..ebar_z_start,
            GariColumnBlock::EbarZ => ebar_z_start..ebar_x_start,
            GariColumnBlock::EbarX => ebar_x_start..ebar_x_start + self.num_ex,
        }
    }

    /// Return the original-detector row range.
    #[must_use]
    pub fn detector_rows(&self) -> Range<usize> {
        0..self.num_detectors
    }

    /// Return the U consistency-row range, one row per eZ column.
    #[must_use]
    pub fn u_rows(&self) -> Range<usize> {
        self.num_detectors..self.num_detectors + self.num_ez
    }

    /// Return the V consistency-row range, one row per eX column.
    #[must_use]
    pub fn v_rows(&self) -> Range<usize> {
        let start = self.num_detectors + self.num_ez;
        start..start + self.num_ex
    }

    /// Zero-pad a physical syndrome with the GARI consistency syndromes.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError::InvalidDimensions`] when `syndrome` does not
    /// have the detector width of the original DEM.
    pub fn extend_syndrome(&self, syndrome: &[u8]) -> Result<Vec<u8>, DecoderError> {
        if syndrome.len() != self.num_detectors {
            return Err(DecoderError::InvalidDimensions {
                expected: self.num_detectors,
                actual: syndrome.len(),
            });
        }
        let mut extended = vec![0; self.dem.num_detectors];
        extended[..self.num_detectors].copy_from_slice(syndrome);
        Ok(extended)
    }

    /// Return the merged original columns in GARI order `[eZ | eX | eY]`.
    #[must_use]
    pub fn physical_columns(&self) -> &SparseDem {
        &self.physical_dem
    }

    /// Return `(nnz(H), nnz(Hbar))` for the merged physical and GARI models.
    #[must_use]
    pub fn edge_counts(&self) -> (usize, usize) {
        let physical = self
            .physical_dem
            .mechanisms
            .iter()
            .map(|(_, detectors, _)| detectors.len())
            .sum();
        let transformed = self
            .dem
            .mechanisms
            .iter()
            .map(|(_, detectors, _)| detectors.len())
            .sum();
        (physical, transformed)
    }
}

/// Build the canonical detector-basis mask for an XZ memory experiment.
///
/// Round one and final readout emit only the initialization-basis family;
/// intermediate rounds emit X detectors followed by Z detectors.
///
/// # Panics
///
/// Panics when `rounds` is zero.
#[must_use]
pub fn xz_memory_detector_bases(
    n_x: usize,
    n_z: usize,
    rounds: usize,
    init_basis: DetectorBasis,
) -> Vec<DetectorBasis> {
    assert!(
        rounds >= 1,
        "an XZ memory experiment needs at least one round"
    );

    let mut bases = Vec::new();
    let init_count = match init_basis {
        DetectorBasis::X => n_x,
        DetectorBasis::Z => n_z,
    };
    bases.extend(std::iter::repeat_n(init_basis, init_count));
    for _ in 1..rounds {
        bases.extend(std::iter::repeat_n(DetectorBasis::X, n_x));
        bases.extend(std::iter::repeat_n(DetectorBasis::Z, n_z));
    }
    bases.extend(std::iter::repeat_n(init_basis, init_count));
    bases
}

/// Restrict a DEM to the detector family matching `init_basis`.
///
/// Columns made indistinguishable by the projection are merged in their first
/// occurrence order.
///
/// # Errors
///
/// Returns [`DecoderError::InvalidConfiguration`] for a basis-width mismatch,
/// an invalid sparse index, or a detector-logical collision in the input DEM.
pub fn init_dets_view(
    dem: &SparseDem,
    bases: &[DetectorBasis],
    init_basis: DetectorBasis,
) -> Result<InitDetsView, DecoderError> {
    let canonical = validate_input(dem, bases)?;
    let merged = merge_columns(&canonical)?;

    let mut detector_index = Vec::new();
    let mut projected_index = vec![None; dem.num_detectors];
    for (detector, &basis) in bases.iter().enumerate() {
        if basis == init_basis {
            let row = u32_index(detector_index.len(), "projected detector row")?;
            projected_index[detector] = Some(row);
            detector_index.push(u32_index(detector, "detector")?);
        }
    }

    let mut projected = Vec::with_capacity(merged.len());
    for (probability, detectors, observables) in merged {
        let detectors: Vec<u32> = detectors
            .iter()
            .filter_map(|&detector| projected_index[detector as usize])
            .collect();
        if !detectors.is_empty() || !observables.is_empty() {
            projected.push((probability, detectors, observables));
        }
    }
    let mechanisms = merge_by_key(projected);

    let detector_coords = detector_index
        .iter()
        .enumerate()
        .filter_map(|(row, &original)| {
            dem.detector_coords
                .get(&(original as usize))
                .map(|coords| (row, coords.clone()))
        })
        .collect();

    Ok(InitDetsView {
        dem: SparseDem {
            mechanisms,
            detector_coords,
            num_detectors: detector_index.len(),
            num_observables: dem.num_observables,
        },
        detector_index,
        original_num_detectors: dem.num_detectors,
    })
}

/// Apply the sparse GARI transform to a correlated XYZ detector error model.
///
/// The returned model is verified against every physical unit vector before
/// this function succeeds. This is a complete equivalence check because both
/// sides of the checked identities are GF(2)-linear.
///
/// Columns that are empty after GF(2) cancellation, with no detectors and no
/// observables, change neither the syndrome nor the logical outcome and are
/// dropped. The reference implementation rejects them instead.
///
/// # Errors
///
/// Returns [`DecoderError::InvalidConfiguration`] when the DEM cannot satisfy
/// the GARI structural assumptions, and [`DecoderError::InternalError`] if the
/// built-in equivalence verification fails.
pub fn gari_transform(
    dem: &SparseDem,
    bases: &[DetectorBasis],
    init_basis: DetectorBasis,
) -> Result<GariModel, DecoderError> {
    let canonical = validate_input(dem, bases)?;
    let merged = merge_columns(&canonical)?;

    let mut ez = Vec::new();
    let mut ex = Vec::new();
    let mut ey = Vec::new();
    for mechanism in merged {
        let mut has_x = false;
        let mut has_z = false;
        for &detector in &mechanism.1 {
            match bases[detector as usize] {
                DetectorBasis::X => has_x = true,
                DetectorBasis::Z => has_z = true,
            }
        }
        match (has_x, has_z) {
            (true, false) => ez.push(mechanism),
            (false, true) => ex.push(mechanism),
            (true, true) => ey.push(mechanism),
            (false, false) if mechanism.2.is_empty() => {}
            (false, false) => {
                return Err(DecoderError::InvalidConfiguration(
                    "detector-silent error mechanisms with observable support are not supported by the GARI transform".into(),
                ));
            }
        }
    }

    let off_side_count = match init_basis {
        DetectorBasis::X => ex
            .iter()
            .filter(|(_, _, observables)| !observables.is_empty())
            .count(),
        DetectorBasis::Z => ez
            .iter()
            .filter(|(_, _, observables)| !observables.is_empty())
            .count(),
    };
    if off_side_count != 0 {
        let side = match init_basis {
            DetectorBasis::X => "eX",
            DetectorBasis::Z => "eZ",
        };
        return Err(DecoderError::InvalidConfiguration(format!(
            "GARI transform found {off_side_count} pure {side} columns carrying observables on the off side"
        )));
    }

    let ez_lookup = partner_lookup(&ez, init_basis == DetectorBasis::X, "eZ")?;
    let ex_lookup = partner_lookup(&ex, init_basis == DetectorBasis::Z, "eX")?;
    let mut u_map = Vec::with_capacity(ey.len());
    let mut v_map = Vec::with_capacity(ey.len());
    let mut missing_ez = 0;
    let mut missing_ex = 0;
    for (_, detectors, observables) in &ey {
        let u_key = (
            detector_restriction(detectors, bases, DetectorBasis::X),
            if init_basis == DetectorBasis::X {
                observables.clone()
            } else {
                Vec::new()
            },
        );
        if let Some(&partner) = ez_lookup.get(&u_key) {
            u_map.push(partner);
        } else {
            missing_ez += 1;
            u_map.push(0);
        }

        let v_key = (
            detector_restriction(detectors, bases, DetectorBasis::Z),
            if init_basis == DetectorBasis::Z {
                observables.clone()
            } else {
                Vec::new()
            },
        );
        if let Some(&partner) = ex_lookup.get(&v_key) {
            v_map.push(partner);
        } else {
            missing_ex += 1;
            v_map.push(0);
        }
    }
    if missing_ez != 0 || missing_ex != 0 {
        return Err(DecoderError::InvalidConfiguration(format!(
            "{missing_ez} of {} eY columns lacked an eZ partner and {missing_ex} of {} lacked an eX partner",
            ey.len(),
            ey.len()
        )));
    }

    let num_ez = ez.len();
    let num_ex = ex.len();
    let num_ey = ey.len();
    let ey_start = checked_add(num_ez, num_ex, "eY column offset")?;
    let physical_count = checked_add(ey_start, num_ey, "physical column count")?;
    let ebar_z_start = physical_count;
    let ebar_x_start = checked_add(ebar_z_start, num_ez, "ebarX column offset")?;
    let transformed_column_count = checked_add(ebar_x_start, num_ex, "transformed column count")?;
    u32_index(transformed_column_count, "transformed column count")?;

    let u_start = dem.num_detectors;
    let v_start = checked_add(u_start, num_ez, "V row offset")?;
    let transformed_row_count = checked_add(v_start, num_ex, "transformed row count")?;
    u32_index(transformed_row_count, "transformed row count")?;

    let mut physical_mechanisms = Vec::with_capacity(physical_count);
    physical_mechanisms.extend(ez.iter().chain(&ex).chain(&ey).cloned());
    let physical_dem = SparseDem {
        mechanisms: physical_mechanisms,
        detector_coords: dem.detector_coords.clone(),
        num_detectors: dem.num_detectors,
        num_observables: dem.num_observables,
    };

    let mut relevant_priors: Vec<f64> = match init_basis {
        DetectorBasis::X => ez.iter().map(|(probability, _, _)| *probability).collect(),
        DetectorBasis::Z => ex.iter().map(|(probability, _, _)| *probability).collect(),
    };
    for (column, (probability, _, _)) in ey.iter().enumerate() {
        let partner = match init_basis {
            DetectorBasis::X => u_map[column] as usize,
            DetectorBasis::Z => v_map[column] as usize,
        };
        let previous = relevant_priors[partner];
        relevant_priors[partner] = previous + probability - 2.0 * previous * probability;
    }

    let mut mechanisms = Vec::with_capacity(transformed_column_count);
    for (column, (probability, _, _)) in ez.iter().enumerate() {
        let row = checked_add(u_start, column, "U row")?;
        mechanisms.push((*probability, vec![u32_index(row, "U row")?], Vec::new()));
    }
    for (column, (probability, _, _)) in ex.iter().enumerate() {
        let row = checked_add(v_start, column, "V row")?;
        mechanisms.push((*probability, vec![u32_index(row, "V row")?], Vec::new()));
    }
    for (column, (probability, _, _)) in ey.iter().enumerate() {
        let u_row = checked_add(u_start, u_map[column] as usize, "U row")?;
        let v_row = checked_add(v_start, v_map[column] as usize, "V row")?;
        mechanisms.push((
            *probability,
            vec![u32_index(u_row, "U row")?, u32_index(v_row, "V row")?],
            Vec::new(),
        ));
    }
    for (column, (_, detectors, observables)) in ez.iter().enumerate() {
        let mut rows = detectors.clone();
        let row = checked_add(u_start, column, "U row")?;
        rows.push(u32_index(row, "U row")?);
        mechanisms.push((0.5, rows, observables.clone()));
    }
    for (column, (_, detectors, observables)) in ex.iter().enumerate() {
        let mut rows = detectors.clone();
        let row = checked_add(v_start, column, "V row")?;
        rows.push(u32_index(row, "V row")?);
        mechanisms.push((0.5, rows, observables.clone()));
    }

    let mut relevant_rows = Vec::new();
    for (row, &basis) in bases.iter().enumerate() {
        if basis == init_basis {
            relevant_rows.push(u32_index(row, "relevant detector row")?);
        }
    }
    let detector_coords = dem
        .detector_coords
        .iter()
        .filter(|(row, _)| **row < dem.num_detectors)
        .map(|(&row, coords)| (row, coords.clone()))
        .collect();
    let model = GariModel {
        dem: SparseDem {
            mechanisms,
            detector_coords,
            num_detectors: transformed_row_count,
            num_observables: dem.num_observables,
        },
        num_detectors: dem.num_detectors,
        init_basis,
        answer_block: match init_basis {
            DetectorBasis::X => GariColumnBlock::EbarZ,
            DetectorBasis::Z => GariColumnBlock::EbarX,
        },
        relevant_rows,
        relevant_priors,
        u_map,
        v_map,
        physical_dem,
        num_ez,
        num_ex,
        num_ey,
    };
    verify_gari_equivalence(&model)?;
    Ok(model)
}

fn validate_input(
    dem: &SparseDem,
    bases: &[DetectorBasis],
) -> Result<Vec<Mechanism>, DecoderError> {
    if bases.len() != dem.num_detectors {
        return Err(DecoderError::InvalidConfiguration(format!(
            "detector-basis mask length must be {}, got {}",
            dem.num_detectors,
            bases.len()
        )));
    }

    let mut canonical = Vec::with_capacity(dem.mechanisms.len());
    for (column, (probability, detectors, observables)) in dem.mechanisms.iter().enumerate() {
        if !probability.is_finite() || !(0.0..=1.0).contains(probability) {
            return Err(DecoderError::InvalidConfiguration(format!(
                "mechanism {column} probability must satisfy 0 <= p <= 1, got {probability}"
            )));
        }
        for &detector in detectors {
            if detector as usize >= dem.num_detectors {
                return Err(DecoderError::InvalidConfiguration(format!(
                    "mechanism {column} detector index {detector} is out of range 0..{}",
                    dem.num_detectors
                )));
            }
        }
        for &observable in observables {
            if observable as usize >= dem.num_observables {
                return Err(DecoderError::InvalidConfiguration(format!(
                    "mechanism {column} observable index {observable} is out of range 0..{}",
                    dem.num_observables
                )));
            }
        }
        canonical.push((
            *probability,
            canonicalize_support(detectors),
            canonicalize_support(observables),
        ));
    }
    Ok(canonical)
}

fn canonicalize_support(support: &[u32]) -> Vec<u32> {
    let mut sorted = support.to_vec();
    sorted.sort_unstable();
    let mut canonical = Vec::with_capacity(sorted.len());
    let mut start = 0;
    while start < sorted.len() {
        let value = sorted[start];
        let mut end = start + 1;
        while end < sorted.len() && sorted[end] == value {
            end += 1;
        }
        if (end - start) % 2 != 0 {
            canonical.push(value);
        }
        start = end;
    }
    canonical
}

fn validate_detector_logical_collisions(mechanisms: &[Mechanism]) -> Result<(), DecoderError> {
    let mut observables_by_detectors: BTreeMap<Vec<u32>, Vec<u32>> = BTreeMap::new();
    for (_, detectors, observables) in mechanisms {
        if let Some(previous) = observables_by_detectors.get(detectors) {
            if previous != observables {
                return Err(DecoderError::InvalidConfiguration(
                    "two columns have identical detector support but different observable support; \
                     XORing them is a detector-silent, logically nontrivial error, so the model's \
                     distance is at most 2"
                        .into(),
                ));
            }
        } else {
            observables_by_detectors.insert(detectors.clone(), observables.clone());
        }
    }
    Ok(())
}

fn merge_columns(mechanisms: &[Mechanism]) -> Result<Vec<Mechanism>, DecoderError> {
    validate_detector_logical_collisions(mechanisms)?;
    Ok(merge_by_key(mechanisms.to_vec()))
}

fn merge_by_key(mechanisms: Vec<Mechanism>) -> Vec<Mechanism> {
    let mut positions: BTreeMap<ColumnKey, usize> = BTreeMap::new();
    let mut merged: Vec<Mechanism> = Vec::new();
    for (probability, detectors, observables) in mechanisms {
        let key = (detectors.clone(), observables.clone());
        if let Some(&position) = positions.get(&key) {
            let previous = merged[position].0;
            merged[position].0 = merge_probability(previous, probability);
        } else {
            let position = merged.len();
            positions.insert(key, position);
            let previous = 0.0;
            let probability = merge_probability(previous, probability);
            merged.push((probability, detectors, observables));
        }
    }
    merged
}

fn merge_probability(previous: f64, probability: f64) -> f64 {
    previous * (1.0 - probability) + probability * (1.0 - previous)
}

fn partner_lookup(
    mechanisms: &[Mechanism],
    keep_observables: bool,
    block: &str,
) -> Result<BTreeMap<ColumnKey, u32>, DecoderError> {
    mechanisms
        .iter()
        .enumerate()
        .map(|(column, (_, detectors, observables))| {
            let key = (
                detectors.clone(),
                if keep_observables {
                    observables.clone()
                } else {
                    Vec::new()
                },
            );
            Ok((key, u32_index(column, block)?))
        })
        .collect()
}

fn detector_restriction(
    detectors: &[u32],
    bases: &[DetectorBasis],
    basis: DetectorBasis,
) -> Vec<u32> {
    detectors
        .iter()
        .copied()
        .filter(|&detector| bases[detector as usize] == basis)
        .collect()
}

fn u32_index(index: usize, kind: &str) -> Result<u32, DecoderError> {
    u32::try_from(index).map_err(|_| {
        DecoderError::InvalidConfiguration(format!(
            "{kind} index {index} cannot be represented in SparseDem"
        ))
    })
}

fn checked_add(left: usize, right: usize, kind: &str) -> Result<usize, DecoderError> {
    left.checked_add(right).ok_or_else(|| {
        DecoderError::InvalidConfiguration(format!("{kind} exceeds the platform index space"))
    })
}

fn verify_gari_equivalence(model: &GariModel) -> Result<(), DecoderError> {
    let physical_count = model.physical_dem.mechanisms.len();
    let ez = model.columns(GariColumnBlock::EZ);
    let ex = model.columns(GariColumnBlock::EX);
    let ey = model.columns(GariColumnBlock::EY);
    let ebar_z = model.columns(GariColumnBlock::EbarZ);
    let ebar_x = model.columns(GariColumnBlock::EbarX);

    let mut rows = Vec::new();
    let mut row_scratch = Vec::new();
    let mut observables = Vec::new();
    let mut observable_scratch = Vec::new();
    for column in 0..physical_count {
        let mut auxiliary_columns = [0; 2];
        let auxiliary_count = if ez.contains(&column) {
            auxiliary_columns[0] = ebar_z
                .start
                .checked_add(column)
                .ok_or_else(|| gari_verification_error(column))?;
            1
        } else if ex.contains(&column) {
            let block_column = column - ex.start;
            auxiliary_columns[0] = ebar_x
                .start
                .checked_add(block_column)
                .ok_or_else(|| gari_verification_error(column))?;
            1
        } else if ey.contains(&column) {
            let block_column = column - ey.start;
            let Some(&u_partner) = model.u_map.get(block_column) else {
                return Err(gari_verification_error(column));
            };
            let Some(&v_partner) = model.v_map.get(block_column) else {
                return Err(gari_verification_error(column));
            };
            if u_partner as usize >= model.num_ez || v_partner as usize >= model.num_ex {
                return Err(gari_verification_error(column));
            }
            auxiliary_columns[0] = ebar_z
                .start
                .checked_add(u_partner as usize)
                .ok_or_else(|| gari_verification_error(column))?;
            auxiliary_columns[1] = ebar_x
                .start
                .checked_add(v_partner as usize)
                .ok_or_else(|| gari_verification_error(column))?;
            2
        } else {
            return Err(gari_verification_error(column));
        };

        rows.clear();
        observables.clear();
        for transformed_column in
            std::iter::once(column).chain(auxiliary_columns[..auxiliary_count].iter().copied())
        {
            let Some((_, column_rows, column_observables)) =
                model.dem.mechanisms.get(transformed_column)
            else {
                return Err(gari_verification_error(column));
            };
            xor_sorted(&mut rows, column_rows, &mut row_scratch);
            xor_sorted(
                &mut observables,
                column_observables,
                &mut observable_scratch,
            );
        }

        let Some((_, expected_rows, expected_observables)) =
            model.physical_dem.mechanisms.get(column)
        else {
            return Err(gari_verification_error(column));
        };
        if rows.as_slice() != expected_rows || observables.as_slice() != expected_observables {
            return Err(gari_verification_error(column));
        }
    }
    Ok(())
}

fn gari_verification_error(column: usize) -> DecoderError {
    DecoderError::InternalError(format!(
        "GARI equivalence verification failed for physical column {column}"
    ))
}

fn xor_sorted(accumulator: &mut Vec<u32>, values: &[u32], scratch: &mut Vec<u32>) {
    scratch.clear();
    let mut left = 0;
    let mut right = 0;
    while left < accumulator.len() && right < values.len() {
        match accumulator[left].cmp(&values[right]) {
            std::cmp::Ordering::Less => {
                scratch.push(accumulator[left]);
                left += 1;
            }
            std::cmp::Ordering::Greater => {
                scratch.push(values[right]);
                right += 1;
            }
            std::cmp::Ordering::Equal => {
                left += 1;
                right += 1;
            }
        }
    }
    scratch.extend_from_slice(&accumulator[left..]);
    scratch.extend_from_slice(&values[right..]);
    std::mem::swap(accumulator, scratch);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dem(mechanisms: Vec<Mechanism>, num_detectors: usize, num_observables: usize) -> SparseDem {
        SparseDem {
            mechanisms,
            detector_coords: BTreeMap::new(),
            num_detectors,
            num_observables,
        }
    }

    #[test]
    fn one_round_memory_mask_has_only_boundary_families() {
        assert_eq!(
            xz_memory_detector_bases(3, 5, 1, DetectorBasis::X),
            vec![DetectorBasis::X; 6]
        );
        assert_eq!(
            xz_memory_detector_bases(3, 5, 1, DetectorBasis::Z),
            vec![DetectorBasis::Z; 10]
        );
    }

    #[test]
    fn three_identical_columns_use_reference_merge_expression() {
        let model = dem(
            vec![
                (0.1, vec![0], vec![]),
                (0.2, vec![0], vec![]),
                (0.3, vec![0], vec![]),
            ],
            1,
            0,
        );
        let merged = merge_columns(&model.mechanisms).unwrap();
        let first = 0.0_f64 * (1.0 - 0.1) + 0.1 * (1.0 - 0.0);
        let second = first * (1.0 - 0.2) + 0.2 * (1.0 - first);
        let expected = second * (1.0 - 0.3) + 0.3 * (1.0 - second);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].0.to_bits(), expected.to_bits());
    }

    #[test]
    fn input_supports_are_sorted_and_xor_canonicalized() {
        let model = dem(
            vec![(0.2, vec![2, 0, 2, 1, 0, 2], vec![1, 0, 1, 0, 1])],
            3,
            2,
        );
        let canonical = validate_input(
            &model,
            &[DetectorBasis::X, DetectorBasis::X, DetectorBasis::X],
        )
        .unwrap();
        assert_eq!(canonical[0].1, vec![1, 2]);
        assert_eq!(canonical[0].2, vec![1]);
        assert_eq!(model.mechanisms[0].1, vec![2, 0, 2, 1, 0, 2]);
        assert_eq!(model.mechanisms[0].2, vec![1, 0, 1, 0, 1]);
    }

    #[test]
    fn input_probabilities_must_be_finite_and_in_range() {
        for probability in [f64::NAN, -0.1, 1.01] {
            let model = dem(vec![(probability, vec![0], vec![])], 1, 0);
            assert!(matches!(
                validate_input(&model, &[DetectorBasis::X]),
                Err(DecoderError::InvalidConfiguration(_))
            ));
        }
        for probability in [0.0, 0.5] {
            let model = dem(vec![(probability, vec![0], vec![])], 1, 0);
            assert!(validate_input(&model, &[DetectorBasis::X]).is_ok());
        }
    }

    #[test]
    fn init_view_merges_before_and_after_projection() {
        let model = dem(
            vec![
                (0.1, vec![0, 1], vec![]),
                (0.2, vec![0, 2], vec![]),
                (0.3, vec![0, 1], vec![]),
            ],
            3,
            0,
        );
        let view = init_dets_view(
            &model,
            &[DetectorBasis::X, DetectorBasis::Z, DetectorBasis::Z],
            DetectorBasis::X,
        )
        .unwrap();

        let merged_a = merge_probability(merge_probability(0.0, 0.1), 0.3);
        let merged_b = merge_probability(0.0, 0.2);
        let expected = merge_probability(merge_probability(0.0, merged_a), merged_b);
        assert_eq!(view.dem.mechanisms.len(), 1);
        assert_eq!(view.dem.mechanisms[0].0.to_bits(), expected.to_bits());
        assert_eq!(view.dem.mechanisms[0].1, vec![0]);
    }

    #[test]
    fn off_side_observables_report_the_column_count() {
        let model = dem(vec![(0.1, vec![0], vec![0])], 1, 1);
        let error = gari_transform(&model, &[DetectorBasis::Z], DetectorBasis::X).unwrap_err();
        assert!(matches!(
            error,
            DecoderError::InvalidConfiguration(message)
                if message == "GARI transform found 1 pure eX columns carrying observables on the off side"
        ));
    }

    #[test]
    fn edge_counts_report_physical_and_transformed_nonzeros() {
        let model = dem(
            vec![
                (0.1, vec![0], vec![0]),
                (0.2, vec![1], vec![]),
                (0.3, vec![0, 1], vec![0]),
            ],
            2,
            1,
        );
        let gari = gari_transform(
            &model,
            &[DetectorBasis::X, DetectorBasis::Z],
            DetectorBasis::X,
        )
        .unwrap();
        assert_eq!(gari.edge_counts(), (4, 8));
    }

    #[test]
    fn deterministic_verification_rejects_a_corrupted_partner_map() {
        let model = dem(
            vec![
                (0.1, vec![0], vec![0]),
                (0.2, vec![1], vec![]),
                (0.3, vec![0, 1], vec![0]),
            ],
            2,
            1,
        );
        let mut gari = gari_transform(
            &model,
            &[DetectorBasis::X, DetectorBasis::Z],
            DetectorBasis::X,
        )
        .unwrap();
        gari.u_map[0] = u32::MAX;

        let error = verify_gari_equivalence(&gari).unwrap_err();
        assert!(matches!(
            error,
            DecoderError::InternalError(message)
                if message.contains("physical column 2")
        ));
    }

    #[test]
    fn deterministic_verification_compares_rows_and_observables() {
        let model = dem(
            vec![
                (0.1, vec![0], vec![0]),
                (0.2, vec![1], vec![]),
                (0.3, vec![0, 1], vec![0]),
            ],
            2,
            1,
        );
        let gari = gari_transform(
            &model,
            &[DetectorBasis::X, DetectorBasis::Z],
            DetectorBasis::X,
        )
        .unwrap();
        let ebar_z = gari.columns(GariColumnBlock::EbarZ).start;

        let mut wrong_detector_row = gari.clone();
        wrong_detector_row.dem.mechanisms[ebar_z]
            .1
            .retain(|&row| row != 0);
        // Without its U entry, eZ leaves the ebarZ U entry uncancelled.
        let mut consistency_residue = gari.clone();
        consistency_residue.dem.mechanisms[0].1.clear();
        let mut wrong_observables = gari.clone();
        wrong_observables.dem.mechanisms[ebar_z].2.clear();

        for corrupted in [wrong_detector_row, consistency_residue, wrong_observables] {
            assert!(matches!(
                verify_gari_equivalence(&corrupted),
                Err(DecoderError::InternalError(message))
                    if message.contains("physical column 0")
            ));
        }
    }

    #[test]
    fn unmatched_mixed_columns_report_each_missing_partner_side() {
        let bases = [DetectorBasis::X, DetectorBasis::Z, DetectorBasis::Z];
        let cases = [
            (
                (0.3, vec![0, 1], vec![]),
                "1 of 1 eY columns lacked an eZ partner and 0 of 1 lacked an eX partner",
            ),
            (
                (0.3, vec![0, 2], vec![0]),
                "0 of 1 eY columns lacked an eZ partner and 1 of 1 lacked an eX partner",
            ),
        ];
        for (mixed, expected) in cases {
            let model = dem(
                vec![(0.1, vec![0], vec![0]), (0.2, vec![1], vec![]), mixed],
                3,
                1,
            );
            let error = gari_transform(&model, &bases, DetectorBasis::X).unwrap_err();
            assert!(matches!(
                error,
                DecoderError::InvalidConfiguration(message) if message == expected
            ));
        }
    }

    #[test]
    fn gari_drops_columns_that_cancel_to_nothing() {
        let bases = [DetectorBasis::X, DetectorBasis::Z];
        let mechanisms = vec![
            (0.1, vec![0], vec![0]),
            (0.2, vec![1], vec![]),
            (0.3, vec![0, 1], vec![0]),
        ];
        let reference =
            gari_transform(&dem(mechanisms.clone(), 2, 1), &bases, DetectorBasis::X).unwrap();

        let mut with_empty = mechanisms;
        with_empty.insert(1, (0.4, vec![1, 1], vec![0, 0]));
        let gari =
            gari_transform(&dem(with_empty.clone(), 2, 1), &bases, DetectorBasis::X).unwrap();
        assert_eq!(gari.dem.mechanisms, reference.dem.mechanisms);
        assert_eq!(
            gari.physical_columns().mechanisms,
            reference.physical_columns().mechanisms
        );

        // An empty detector support that still flips an observable is a
        // logical error no syndrome can see, so it is still rejected.
        with_empty[1].2 = vec![0];
        assert!(matches!(
            gari_transform(&dem(with_empty, 2, 1), &bases, DetectorBasis::X),
            Err(DecoderError::InvalidConfiguration(message))
                if message.starts_with("detector-silent")
        ));
    }

    #[test]
    fn init_view_rejects_detector_logical_collisions() {
        let model = dem(vec![(0.1, vec![0], vec![0]), (0.2, vec![0], vec![])], 1, 1);
        assert!(matches!(
            init_dets_view(&model, &[DetectorBasis::X], DetectorBasis::X),
            Err(DecoderError::InvalidConfiguration(message))
                if message.contains("identical detector support")
        ));
    }

    #[test]
    fn init_view_reindexes_detector_coordinates() {
        let mut model = dem(vec![(0.1, vec![0, 1, 2], vec![])], 3, 0);
        model.detector_coords = BTreeMap::from([(0, vec![0.0]), (1, vec![1.0]), (2, vec![2.0])]);
        let view = init_dets_view(
            &model,
            &[DetectorBasis::Z, DetectorBasis::X, DetectorBasis::X],
            DetectorBasis::X,
        )
        .unwrap();
        assert_eq!(
            view.dem.detector_coords,
            BTreeMap::from([(0, vec![1.0]), (1, vec![2.0])])
        );
    }

    #[test]
    fn input_observable_indices_must_be_in_range() {
        let model = dem(vec![(0.1, vec![0], vec![1])], 1, 1);
        assert!(matches!(
            validate_input(&model, &[DetectorBasis::X]),
            Err(DecoderError::InvalidConfiguration(message))
                if message.contains("observable index 1 is out of range")
        ));
    }

    #[test]
    #[should_panic(expected = "at least one round")]
    fn memory_mask_requires_a_round() {
        let _ = xz_memory_detector_bases(1, 1, 0, DetectorBasis::X);
    }
}
