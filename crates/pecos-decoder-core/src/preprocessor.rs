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

//! Syndrome preprocessor for real-system QEC decoding.
//!
//! Sits between the hardware readout and the decoder. Responsibilities:
//! - Convert leakage flags (from PECOS's `MeasureLeaked` simulation or
//!   neutral atom loss detection) into erasure edge indices for the decoder
//! - Detect syndrome anomalies (excessive weight, all-ones, etc.)
//! - Validate syndrome dimensions
//!
//! # Leakage-to-Erasure Pipeline
//!
//! Neutral atoms: atom loss is detected per-qubit. Each lost qubit
//! affects specific error mechanisms (edges in the matching graph).
//! The preprocessor maps qubit loss → affected DEM edges → erasure
//! indices for `ObservableErasureDecoder`.
//!
//! The mapping is built at construction from the DEM's detector
//! coordinates or a user-provided qubit-to-edge mapping.

/// Anomaly detected in a syndrome.
#[derive(Debug, Clone)]
pub enum SyndromeAnomaly {
    /// More defects than expected (possible burst error or readout failure).
    ExcessiveWeight { weight: usize, threshold: usize },
    /// All detectors fired (likely readout failure, not a real syndrome).
    AllOnes,
    /// Wrong syndrome length.
    WrongLength { expected: usize, actual: usize },
}

/// Preprocessed syndrome ready for the decoder.
#[derive(Debug, Clone)]
pub struct ProcessedSyndrome {
    /// The syndrome (possibly with leakage-affected bits cleared/modified).
    pub syndrome: Vec<u8>,
    /// DEM edge indices known to be erased (from leakage detection).
    pub erasure_edges: Vec<usize>,
    /// Anomaly if detected, None if syndrome looks normal.
    pub anomaly: Option<SyndromeAnomaly>,
}

/// Syndrome preprocessor.
pub struct SyndromePreprocessor {
    num_detectors: usize,
    /// Maximum expected syndrome weight before flagging anomaly.
    /// Set to 0 to disable (default).
    weight_threshold: usize,
    /// Qubit index → list of DEM edge indices affected by that qubit's loss.
    /// Built from the DEM structure at construction time.
    qubit_to_erasure_edges: Vec<Vec<usize>>,
}

impl SyndromePreprocessor {
    /// Create with basic syndrome validation.
    #[must_use]
    pub fn new(num_detectors: usize) -> Self {
        Self {
            num_detectors,
            weight_threshold: 0,
            qubit_to_erasure_edges: Vec::new(),
        }
    }

    /// Set the maximum expected syndrome weight for anomaly detection.
    pub fn set_weight_threshold(&mut self, threshold: usize) {
        self.weight_threshold = threshold;
    }

    /// Set the qubit-to-erasure-edge mapping for leakage conversion.
    ///
    /// `mapping[qubit_idx]` = list of DEM edge indices affected by that qubit.
    /// Built from the DEM: for each edge, find which data qubits it involves.
    pub fn set_erasure_mapping(&mut self, mapping: Vec<Vec<usize>>) {
        self.qubit_to_erasure_edges = mapping;
    }

    /// Preprocess a raw syndrome with optional leakage flags.
    ///
    /// `leakage_flags`: one byte per qubit, nonzero = qubit leaked/lost.
    /// Converts leaked qubits to erasure edge indices via the mapping.
    #[must_use]
    pub fn preprocess(
        &self,
        raw_syndrome: &[u8],
        leakage_flags: Option<&[u8]>,
    ) -> ProcessedSyndrome {
        let mut anomaly = None;

        // Validate length.
        if raw_syndrome.len() != self.num_detectors {
            return ProcessedSyndrome {
                syndrome: raw_syndrome.to_vec(),
                erasure_edges: Vec::new(),
                anomaly: Some(SyndromeAnomaly::WrongLength {
                    expected: self.num_detectors,
                    actual: raw_syndrome.len(),
                }),
            };
        }

        // Check weight.
        let weight = raw_syndrome.iter().filter(|&&v| v != 0).count();
        if weight == self.num_detectors && self.num_detectors > 0 {
            anomaly = Some(SyndromeAnomaly::AllOnes);
        } else if self.weight_threshold > 0 && weight > self.weight_threshold {
            anomaly = Some(SyndromeAnomaly::ExcessiveWeight {
                weight,
                threshold: self.weight_threshold,
            });
        }

        // Convert leakage flags to erasure edges.
        let mut erasure_edges = Vec::new();
        if let Some(flags) = leakage_flags {
            for (qubit, &leaked) in flags.iter().enumerate() {
                if leaked != 0 && qubit < self.qubit_to_erasure_edges.len() {
                    erasure_edges.extend_from_slice(&self.qubit_to_erasure_edges[qubit]);
                }
            }
            erasure_edges.sort_unstable();
            erasure_edges.dedup();
        }

        ProcessedSyndrome {
            syndrome: raw_syndrome.to_vec(),
            erasure_edges,
            anomaly,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_preprocessing() {
        let pp = SyndromePreprocessor::new(4);
        let result = pp.preprocess(&[0, 1, 0, 1], None);
        assert!(result.anomaly.is_none());
        assert!(result.erasure_edges.is_empty());
    }

    #[test]
    fn test_wrong_length() {
        let pp = SyndromePreprocessor::new(4);
        let result = pp.preprocess(&[0, 1], None);
        assert!(matches!(
            result.anomaly,
            Some(SyndromeAnomaly::WrongLength { .. })
        ));
    }

    #[test]
    fn test_excessive_weight() {
        let mut pp = SyndromePreprocessor::new(4);
        pp.set_weight_threshold(2);
        let result = pp.preprocess(&[1, 1, 1, 0], None);
        assert!(matches!(
            result.anomaly,
            Some(SyndromeAnomaly::ExcessiveWeight { weight: 3, .. })
        ));
    }

    #[test]
    fn test_all_ones() {
        let pp = SyndromePreprocessor::new(3);
        let result = pp.preprocess(&[1, 1, 1], None);
        assert!(matches!(result.anomaly, Some(SyndromeAnomaly::AllOnes)));
    }

    #[test]
    fn test_leakage_to_erasure() {
        let mut pp = SyndromePreprocessor::new(4);
        // Qubit 0 affects edges [0, 2], qubit 1 affects edge [1]
        pp.set_erasure_mapping(vec![vec![0, 2], vec![1]]);
        let result = pp.preprocess(&[0, 1, 0, 0], Some(&[1, 0])); // qubit 0 leaked
        assert_eq!(result.erasure_edges, vec![0, 2]);
    }

    #[test]
    fn test_multiple_leakage() {
        let mut pp = SyndromePreprocessor::new(4);
        pp.set_erasure_mapping(vec![vec![0, 2], vec![1, 2]]); // edge 2 shared
        let result = pp.preprocess(&[0, 0, 0, 0], Some(&[1, 1])); // both leaked
        assert_eq!(result.erasure_edges, vec![0, 1, 2]); // deduped
    }
}
