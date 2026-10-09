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

//! Validated packed-column shot storage.
use pecos_decoder_core::obs_mask::ObsMask;
use std::fmt;

/// Invalid packed columns or shot-major rows.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SampleBatchError(String);

impl fmt::Display for SampleBatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for SampleBatchError {}

/// Detector events and true observable flips, packed by column.
///
/// Shot `s` occupies bit `s % 64` of word `s / 64` in each column.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SampleBatch {
    det_columns: Vec<Vec<u64>>,
    obs_columns: Vec<Vec<u64>>,
    num_shots: usize,
}

/// One materialized shot from [`SampleBatch::shots`].
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct Shot {
    /// One byte per detector, either zero or one.
    pub syndrome: Vec<u8>,
    /// True observable flips for this shot, without a 64-bit width limit.
    pub observable_flips: ObsMask,
}

impl SampleBatch {
    /// Construct from packed detector and observable columns.
    ///
    /// # Errors
    /// Every column must contain exactly `num_shots.div_ceil(64)` words,
    /// with zero padding bits above the last shot in the final word.
    pub fn from_columnar(
        det_columns: Vec<Vec<u64>>,
        obs_columns: Vec<Vec<u64>>,
        num_shots: usize,
    ) -> Result<Self, SampleBatchError> {
        let expected = num_shots.div_ceil(64);
        for (kind, columns) in [("detector", &det_columns), ("observable", &obs_columns)] {
            for (index, column) in columns.iter().enumerate() {
                if column.len() != expected {
                    return Err(SampleBatchError(format!(
                        "{kind} column {index} has {} words but expected {expected}",
                        column.len()
                    )));
                }
                let used_bits = num_shots % 64;
                if used_bits != 0 && column.last().is_some_and(|word| word >> used_bits != 0) {
                    return Err(SampleBatchError(format!(
                        "{kind} column {index} has nonzero padding bits above num_shots={num_shots}"
                    )));
                }
            }
        }
        Ok(Self {
            det_columns,
            obs_columns,
            num_shots,
        })
    }

    /// Construct from shot-major detector bytes and wide observable masks.
    /// Every nonzero detector byte means a set detector event.
    /// Empty rows infer zero detectors; packed columns can retain an empty batch's width.
    ///
    /// # Errors
    /// Rejects unequal shot counts, unequal row widths, and observable bits outside the width.
    pub fn from_row_major<R: AsRef<[u8]>>(
        detection_events: &[R],
        observable_masks: &[ObsMask],
        num_observables: usize,
    ) -> Result<Self, SampleBatchError> {
        if detection_events.len() != observable_masks.len() {
            return Err(SampleBatchError(format!(
                "detection_events ({}) and observable_masks ({}) must have same length",
                detection_events.len(),
                observable_masks.len()
            )));
        }
        let num_detectors = detection_events.first().map_or(0, |row| row.as_ref().len());
        for (i, row) in detection_events.iter().enumerate() {
            let row = row.as_ref();
            if row.len() != num_detectors {
                return Err(SampleBatchError(format!(
                    "detection_events row {i} has length {} but expected {num_detectors} (matching row 0)",
                    row.len()
                )));
            }
        }
        if let Some(bit) = observable_masks
            .iter()
            .flat_map(ObsMask::iter_set_bits)
            .find(|&bit| bit >= num_observables)
        {
            return Err(SampleBatchError(format!(
                "observable mask bit {bit} is outside num_observables={num_observables}"
            )));
        }
        let num_shots = detection_events.len();
        let num_words = num_shots.div_ceil(64);
        let mut det_columns = vec![vec![0u64; num_words]; num_detectors];
        for (shot, row) in detection_events.iter().enumerate() {
            let word_idx = shot / 64;
            let bit_mask = 1u64 << (shot % 64);
            for (det_idx, &val) in row.as_ref().iter().enumerate() {
                if val != 0 {
                    det_columns[det_idx][word_idx] |= bit_mask;
                }
            }
        }
        let mut obs_columns = vec![vec![0u64; num_words]; num_observables];
        for (shot, mask) in observable_masks.iter().enumerate() {
            let word_idx = shot / 64;
            let bit_mask = 1u64 << (shot % 64);
            for obs_idx in mask.iter_set_bits() {
                obs_columns[obs_idx][word_idx] |= bit_mask;
            }
        }
        Ok(Self {
            det_columns,
            obs_columns,
            num_shots,
        })
    }

    /// Number of stored shots.
    #[must_use]
    pub const fn num_shots(&self) -> usize {
        self.num_shots
    }
    /// Number of detector columns.
    #[must_use]
    pub fn num_detectors(&self) -> usize {
        self.det_columns.len()
    }
    /// Number of observable columns, including all-zero columns.
    #[must_use]
    pub fn num_observables(&self) -> usize {
        self.obs_columns.len()
    }
    /// Read the canonical packed detector columns; unused final-word bits are zero.
    #[must_use]
    pub fn det_columns(&self) -> &[Vec<u64>] {
        &self.det_columns
    }
    /// Read the canonical packed observable columns; unused final-word bits are zero.
    #[must_use]
    pub fn obs_columns(&self) -> &[Vec<u64>] {
        &self.obs_columns
    }

    /// Extract a syndrome, first clearing the whole destination buffer.
    /// The caller supplies space for every detector.
    ///
    /// # Panics
    /// Panics if `shot >= self.num_shots()` or `buf` is shorter than `self.num_detectors()`.
    pub fn syndrome_into(&self, shot: usize, buf: &mut [u8]) {
        assert!(
            shot < self.num_shots,
            "shot index {shot} out of range (num_shots={})",
            self.num_shots
        );
        assert!(
            buf.len() >= self.num_detectors(),
            "syndrome buffer has {} bytes but the batch has {} detectors",
            buf.len(),
            self.num_detectors()
        );
        buf.fill(0);
        let word_idx = shot / 64;
        let bit_mask = 1u64 << (shot % 64);
        for (det_idx, col) in self.det_columns.iter().enumerate() {
            if col[word_idx] & bit_mask != 0 {
                buf[det_idx] = 1;
            }
        }
    }

    /// Extract the wide observable mask at a valid shot index.
    ///
    /// # Panics
    /// Panics if `shot >= self.num_shots()`.
    #[must_use]
    pub fn observable_flips(&self, shot: usize) -> ObsMask {
        assert!(
            shot < self.num_shots,
            "shot index {shot} out of range (num_shots={})",
            self.num_shots
        );
        let word_idx = shot / 64;
        let bit_mask = 1u64 << (shot % 64);
        let mut mask = ObsMask::new();
        for (obs_idx, col) in self.obs_columns.iter().enumerate() {
            if col[word_idx] & bit_mask != 0 {
                mask.set(obs_idx);
            }
        }
        mask
    }

    /// Materialize exactly `num_shots()` shots in original order.
    /// The iterator never exposes padding and returns `None` after the last shot.
    pub fn shots(&self) -> impl ExactSizeIterator<Item = Shot> + '_ {
        (0..self.num_shots).map(|shot| {
            let mut syndrome = vec![0; self.num_detectors()];
            self.syndrome_into(shot, &mut syndrome);
            Shot {
                syndrome,
                observable_flips: self.observable_flips(shot),
            }
        })
    }
}
