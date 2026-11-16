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

//! Prepared feasibility and deterministic fallback explanations for independent binary DEMs.
//!
//! This is not a maximum-likelihood decoder. It selects a probability-ordered
//! independent column basis, then solves in that basis. An explanation certifies
//! only consistency with the supplied DEM, not with the underlying circuit.

use crate::{DecoderError, dem::SparseDem, obs_mask::ObsMask};

#[derive(Clone, Debug)]
struct Column {
    detectors: Vec<u64>,
    mechanisms: Vec<u64>,
}

/// A syndrome-consistent DEM fault explanation, not a data-qubit Pauli correction.
#[derive(Clone, Debug, PartialEq)]
pub struct FeasibleExplanation {
    /// Original DEM mechanism indices whose effects XOR to the syndrome.
    pub mechanisms: Vec<usize>,
    /// Observable effects of exactly those mechanisms.
    pub observables: ObsMask,
}

/// Cached GF(2) basis reusable across decoders and worker threads.
///
/// Start from the independent modal fault assignment (p > 1/2 is on), then
/// prefer toggle columns with larger min(p, 1-p), breaking ties by original
/// index. p=0 and p=1 are fixed, not available basis columns. Correlated `^`
/// components in one DEM mechanism remain one joint column. Categorical
/// outcomes cannot be treated as independent columns and are not accepted here.
#[derive(Clone, Debug)]
pub struct DemFeasibility {
    dem: SparseDem,
    basis: Vec<Option<Column>>,
    baseline: Vec<u64>,
    baseline_syndrome: Vec<u64>,
}

fn toggle(words: &mut [u64], index: usize) {
    words[index / 64] ^= 1 << (index % 64);
}
fn xor(left: &mut [u64], right: &[u64]) {
    for (left, right) in left.iter_mut().zip(right) {
        *left ^= right;
    }
}
fn pivot(words: &[u64]) -> Option<usize> {
    words
        .iter()
        .enumerate()
        .rev()
        .find_map(|(i, &word)| (word != 0).then(|| i * 64 + 63 - word.leading_zeros() as usize))
}

impl DemFeasibility {
    /// Prepare and validate a binary DEM, retaining original mechanism indices.
    ///
    /// # Errors
    /// Rejects non-finite/out-of-range probabilities or out-of-range targets.
    pub fn new(dem: &SparseDem) -> Result<Self, DecoderError> {
        let detector_words = dem.num_detectors.div_ceil(64);
        let mechanism_words = dem.mechanisms.len().div_ceil(64);
        let mut result = Self {
            dem: dem.clone(),
            basis: vec![None; dem.num_detectors],
            baseline: vec![0; mechanism_words],
            baseline_syndrome: vec![0; detector_words],
        };
        let mut order = Vec::new();
        for (i, (p, detectors, observables)) in dem.mechanisms.iter().enumerate() {
            if !p.is_finite() || !(0.0..=1.0).contains(p) {
                return Err(DecoderError::InvalidConfiguration(format!(
                    "invalid probability for mechanism {i}"
                )));
            }
            if detectors.iter().any(|&d| d as usize >= dem.num_detectors)
                || observables
                    .iter()
                    .any(|&l| l as usize >= dem.num_observables)
            {
                return Err(DecoderError::InvalidConfiguration(format!(
                    "out-of-range target in mechanism {i}"
                )));
            }
            if *p > 0.5 {
                toggle(&mut result.baseline, i);
                for &d in detectors {
                    toggle(&mut result.baseline_syndrome, d as usize);
                }
            }
            if *p > 0.0 && *p < 1.0 {
                order.push((i, p.min(1.0 - p)));
            }
        }
        order.sort_by(|(i, p), (j, q)| q.total_cmp(p).then_with(|| i.cmp(j)));
        for (i, _) in order {
            let mut column = Column {
                detectors: vec![0; detector_words],
                mechanisms: vec![0; mechanism_words],
            };
            for &d in &dem.mechanisms[i].1 {
                toggle(&mut column.detectors, d as usize);
            }
            toggle(&mut column.mechanisms, i);
            while let Some(p) = pivot(&column.detectors) {
                if let Some(basis) = &result.basis[p] {
                    xor(&mut column.detectors, &basis.detectors);
                    xor(&mut column.mechanisms, &basis.mechanisms);
                } else {
                    result.basis[p] = Some(column);
                    break;
                }
            }
        }
        Ok(result)
    }

    /// Number of independent probabilistic detector columns.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.basis.iter().filter(|b| b.is_some()).count()
    }

    /// Solve for a verified explanation; `None` proves no explanation exists
    /// under this independent binary DEM, including its deterministic offsets.
    /// Like the trellis engine, nonzero syndrome bytes mean fired detectors.
    ///
    /// # Errors
    /// Rejects dimension mismatches; a failed internal witness check is an error.
    pub fn solve(&self, syndrome: &[u8]) -> Result<Option<FeasibleExplanation>, DecoderError> {
        if syndrome.len() != self.dem.num_detectors {
            return Err(DecoderError::InvalidDimensions {
                expected: self.dem.num_detectors,
                actual: syndrome.len(),
            });
        }
        let mut residual = self.baseline_syndrome.clone();
        for (d, &bit) in syndrome.iter().enumerate() {
            if bit != 0 {
                toggle(&mut residual, d);
            }
        }
        let mut witness = self.baseline.clone();
        while let Some(p) = pivot(&residual) {
            let Some(basis) = &self.basis[p] else {
                return Ok(None);
            };
            xor(&mut residual, &basis.detectors);
            xor(&mut witness, &basis.mechanisms);
        }
        let mut mechanisms = Vec::new();
        let mut observables = ObsMask::new();
        let mut check = vec![0u8; syndrome.len()];
        for (i, (_, detectors, logicals)) in self.dem.mechanisms.iter().enumerate() {
            if witness[i / 64] & (1 << (i % 64)) != 0 {
                mechanisms.push(i);
                for &d in detectors {
                    check[d as usize] ^= 1;
                }
                for &l in logicals {
                    observables.toggle(l as usize);
                }
            }
        }
        if check
            .iter()
            .zip(syndrome)
            .any(|(&a, &b)| a != u8::from(b != 0))
        {
            return Err(DecoderError::InternalError(
                "DEM feasibility witness failed verification".into(),
            ));
        }
        Ok(Some(FeasibleExplanation {
            mechanisms,
            observables,
        }))
    }
}
