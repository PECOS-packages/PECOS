// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Detector Error Model (DEM) extraction from Pauli webs and a noise model.
//!
//! A DEM describes how physical errors map to detector syndromes and logical
//! observable flips. It is the input consumed by decoders such as PyMatching.
//!
//! # Algorithm
//!
//! A Pauli error `E` on edge `(u, v)` fires detector web `D` iff `D` has a
//! Pauli operator `P` on that edge and `P != E` (distinct single-qubit Paulis
//! always anticommute).
//!
//! # Assumptions
//!
//! The ZX graph passed to [`compute_pauli_webs`](crate::pauli_web::compute_pauli_webs)
//! should be unsimplified (circuit-derived), so that `make_bipartite()` is a
//! no-op and PauliWeb edge keys match the original graph vertex IDs.

use std::collections::HashMap;

use quizx::detection_webs::Pauli;

use crate::noise::NoiseModel;
use crate::pauli_web::{PauliWebResult, WebClassification, classify_webs};

/// A single error mechanism in the DEM.
#[derive(Debug, Clone)]
pub struct DemError {
    /// Probability of this error mechanism.
    pub probability: f64,
    /// Indices into [`Dem::detectors`] that this error fires.
    pub detectors: Vec<usize>,
    /// Indices into [`Dem::observables`] that this error flips.
    pub observables: Vec<usize>,
}

/// A Detector Error Model.
#[derive(Debug, Clone)]
pub struct Dem {
    /// Web indices (into the original `PauliWebResult::webs`) classified as detectors.
    pub detectors: Vec<usize>,
    /// Web indices classified as propagated observables.
    pub observables: Vec<usize>,
    /// The error mechanisms.
    pub errors: Vec<DemError>,
}

/// Combine two independent error probabilities via XOR:
/// `p_combined = p1 + p2 - 2 * p1 * p2`.
fn combine_probs(p1: f64, p2: f64) -> f64 {
    p1 + p2 - 2.0 * p1 * p2
}

impl Dem {
    /// Build a DEM from Pauli webs and a noise model.
    ///
    /// The algorithm:
    /// 1. Classify webs as detectors or propagated observables.
    /// 2. For each noisy edge and each Pauli error (X, Y, Z) with nonzero
    ///    probability, scan all webs. A web fires if it has a **different**
    ///    Pauli on that edge.
    /// 3. Errors with the same detector/observable signature are combined via
    ///    independent XOR probability combination.
    #[must_use]
    pub fn from_webs(result: &PauliWebResult, noise: &NoiseModel) -> Self {
        let classifications = classify_webs(result);

        // Collect detector and observable web indices
        let mut detectors = Vec::new();
        let mut observables = Vec::new();
        for (i, class) in classifications.iter().enumerate() {
            match class {
                WebClassification::Detector => detectors.push(i),
                WebClassification::Propagated => observables.push(i),
                _ => {}
            }
        }

        // Reverse maps: web_index -> position in detectors/observables vec
        let det_pos: HashMap<usize, usize> = detectors
            .iter()
            .enumerate()
            .map(|(pos, &idx)| (idx, pos))
            .collect();
        let obs_pos: HashMap<usize, usize> = observables
            .iter()
            .enumerate()
            .map(|(pos, &idx)| (idx, pos))
            .collect();

        // Effect map: (sorted detector positions, sorted observable positions) -> combined probability
        let mut effect_map: HashMap<(Vec<usize>, Vec<usize>), f64> = HashMap::new();

        let error_paulis = [Pauli::X, Pauli::Y, Pauli::Z];

        for (&(u, v), rates) in &noise.edge_errors {
            for &error_pauli in &error_paulis {
                let p = match error_pauli {
                    Pauli::X => rates.px,
                    Pauli::Y => rates.py,
                    Pauli::Z => rates.pz,
                };
                if p == 0.0 {
                    continue;
                }

                // Find which webs this error fires
                let mut fired_dets = Vec::new();
                let mut fired_obs = Vec::new();

                for (web_idx, web) in result.webs.iter().enumerate() {
                    if let Some(web_pauli) = web.edge(u, v) {
                        // Fires if the web's Pauli differs from the error Pauli
                        if web_pauli != error_pauli {
                            if let Some(&pos) = det_pos.get(&web_idx) {
                                fired_dets.push(pos);
                            }
                            if let Some(&pos) = obs_pos.get(&web_idx) {
                                fired_obs.push(pos);
                            }
                        }
                    }
                }

                if fired_dets.is_empty() && fired_obs.is_empty() {
                    continue;
                }

                fired_dets.sort();
                fired_obs.sort();
                let key = (fired_dets, fired_obs);

                let entry = effect_map.entry(key).or_insert(0.0);
                *entry = combine_probs(*entry, p);
            }
        }

        // Convert effect map to DemError structs
        let mut errors: Vec<DemError> = effect_map
            .into_iter()
            .map(|((dets, obs), prob)| DemError {
                probability: prob,
                detectors: dets,
                observables: obs,
            })
            .collect();

        // Sort for deterministic output: by detectors first, then observables
        errors.sort_by(|a, b| {
            a.detectors
                .cmp(&b.detectors)
                .then(a.observables.cmp(&b.observables))
        });

        Dem {
            detectors,
            observables,
            errors,
        }
    }

    /// Render the DEM in Stim's detector error model format.
    #[must_use]
    pub fn to_stim_string(&self) -> String {
        let mut lines = Vec::new();

        for (i, _) in self.detectors.iter().enumerate() {
            lines.push(format!("detector D{i}"));
        }
        for (i, _) in self.observables.iter().enumerate() {
            lines.push(format!("logical_observable L{i}"));
        }

        for error in &self.errors {
            let mut targets = Vec::new();
            for &d in &error.detectors {
                targets.push(format!("D{d}"));
            }
            for &o in &error.observables {
                targets.push(format!("L{o}"));
            }
            let target_str = targets.join(" ");
            lines.push(format!("error({:.6}) {target_str}", error.probability));
        }

        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quizx::detection_webs::PauliWeb;

    use crate::noise::ErrorRates;
    use pecos_decoder_core::dem::utils::{parse_dem_metadata, validate_dem};

    #[test]
    fn test_anticommutation_logic() {
        // A web with X on edge (0,1) should be fired by Y and Z errors, but not X
        let mut web = PauliWeb::new();
        web.set_edge(0, 1, Pauli::X);

        // X error on same edge: same Pauli, should NOT fire
        assert_eq!(web.edge(0, 1), Some(Pauli::X));
        assert!(Pauli::X == Pauli::X); // same -> no fire

        // Y error: different -> fires
        assert!(Pauli::X != Pauli::Y);

        // Z error: different -> fires
        assert!(Pauli::X != Pauli::Z);
    }

    #[test]
    fn test_probability_combination() {
        // Two independent errors with same signature should combine via XOR
        let p1 = 0.01;
        let p2 = 0.02;
        let combined = combine_probs(p1, p2);
        let expected = p1 + p2 - 2.0 * p1 * p2;
        assert!((combined - expected).abs() < 1e-12);

        // Combining with 0 should give the other
        assert!((combine_probs(0.0, 0.05) - 0.05).abs() < 1e-12);
        assert!((combine_probs(0.05, 0.0) - 0.05).abs() < 1e-12);
    }

    #[test]
    fn test_stim_format() {
        let dem = Dem {
            detectors: vec![0, 1],
            observables: vec![2],
            errors: vec![
                DemError {
                    probability: 0.001,
                    detectors: vec![0],
                    observables: vec![],
                },
                DemError {
                    probability: 0.002,
                    detectors: vec![0, 1],
                    observables: vec![0],
                },
            ],
        };

        let s = dem.to_stim_string();
        assert!(s.contains("detector D0"));
        assert!(s.contains("detector D1"));
        assert!(s.contains("logical_observable L0"));
        assert!(s.contains("error(0.001000) D0"));
        assert!(s.contains("error(0.002000) D0 D1 L0"));
    }

    #[test]
    fn test_from_webs_basic() {
        // Build a minimal scenario: two webs (one detector, one propagated)
        // and one noisy edge.

        // Web 0: detector with X on edge (2, 3) -- no boundary legs
        let mut web0 = PauliWeb::new();
        web0.set_edge(2, 3, Pauli::X);

        // Web 1: propagated with Z on edge (2, 3) and boundary edges touching
        // both input (vertex 0) and output (vertex 1)
        let mut web1 = PauliWeb::new();
        web1.set_edge(2, 3, Pauli::Z);
        web1.set_edge(0, 2, Pauli::Z); // touches input boundary vertex 0
        web1.set_edge(1, 3, Pauli::Z); // touches output boundary vertex 1

        let result = PauliWebResult {
            webs: vec![web0, web1],
            input_ids: vec![0],
            output_ids: vec![1],
        };

        let mut noise = NoiseModel::new();
        noise.set_edge(2, 3, ErrorRates::new(0.01, 0.0, 0.0)); // X error only

        let dem = Dem::from_webs(&result, &noise);

        // Web 0 is detector (no boundary), web 1 is propagated (both boundaries)
        assert_eq!(dem.detectors.len(), 1);
        assert_eq!(dem.observables.len(), 1);

        // X error on edge (2,3): web0 has X there (same -> no fire),
        // web1 has Z there (different -> fires observable)
        assert_eq!(dem.errors.len(), 1);
        assert!(dem.errors[0].detectors.is_empty());
        assert_eq!(dem.errors[0].observables, vec![0]);
        assert!((dem.errors[0].probability - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_multiple_pauli_errors_same_edge() {
        // All three Pauli errors on the same edge, each firing different webs
        let mut web0 = PauliWeb::new();
        web0.set_edge(2, 3, Pauli::X); // detector

        let mut web1 = PauliWeb::new();
        web1.set_edge(2, 3, Pauli::Z); // another detector

        let result = PauliWebResult {
            webs: vec![web0, web1],
            input_ids: vec![],
            output_ids: vec![],
        };

        let mut noise = NoiseModel::new();
        // Depolarizing: all three Pauli errors have nonzero probability
        noise.set_edge(2, 3, ErrorRates::depolarizing(0.03));

        let dem = Dem::from_webs(&result, &noise);
        assert_eq!(dem.detectors.len(), 2);

        // X error: fires web1 (Z != X) but not web0 (X == X) -> signature [1]
        // Y error: fires web0 (X != Y) AND web1 (Z != Y) -> signature [0, 1]
        // Z error: fires web0 (X != Z) but not web1 (Z == Z) -> signature [0]
        // Three distinct signatures -> 3 errors
        assert_eq!(dem.errors.len(), 3);
    }

    #[test]
    fn test_error_probability_combination() {
        // Two different Pauli errors on the same edge that fire the same detector
        // should have their probabilities combined via XOR
        let mut web0 = PauliWeb::new();
        web0.set_edge(2, 3, Pauli::X); // detector

        let result = PauliWebResult {
            webs: vec![web0],
            input_ids: vec![],
            output_ids: vec![],
        };

        let mut noise = NoiseModel::new();
        // Y and Z errors both fire web0 (since web0 has X, and X != Y, X != Z)
        // They share the same signature [detector 0], so they combine
        noise.set_edge(2, 3, ErrorRates::new(0.0, 0.01, 0.02)); // py=0.01, pz=0.02

        let dem = Dem::from_webs(&result, &noise);

        assert_eq!(dem.errors.len(), 1);
        let expected = combine_probs(0.01, 0.02);
        assert!((dem.errors[0].probability - expected).abs() < 1e-12);
        assert_eq!(dem.errors[0].detectors, vec![0]);
    }

    #[test]
    fn test_empty_noise_model() {
        let web = PauliWeb::new();
        let result = PauliWebResult {
            webs: vec![web],
            input_ids: vec![],
            output_ids: vec![],
        };
        let noise = NoiseModel::new();
        let dem = Dem::from_webs(&result, &noise);

        assert_eq!(dem.detectors.len(), 1);
        assert!(dem.errors.is_empty());
    }

    #[test]
    fn test_empty_webs() {
        let result = PauliWebResult {
            webs: vec![],
            input_ids: vec![],
            output_ids: vec![],
        };
        let mut noise = NoiseModel::new();
        noise.set_edge(0, 1, ErrorRates::depolarizing(0.01));

        let dem = Dem::from_webs(&result, &noise);

        assert!(dem.detectors.is_empty());
        assert!(dem.observables.is_empty());
        assert!(dem.errors.is_empty());
    }

    #[test]
    fn test_stim_format_empty_dem() {
        let dem = Dem {
            detectors: vec![],
            observables: vec![],
            errors: vec![],
        };
        let s = dem.to_stim_string();
        assert!(s.is_empty());
    }

    #[test]
    fn test_from_webs_detectors_only() {
        // Errors that fire only detectors (no observables)
        let mut web0 = PauliWeb::new();
        web0.set_edge(2, 3, Pauli::X);

        let mut web1 = PauliWeb::new();
        web1.set_edge(2, 3, Pauli::Z);

        // Both webs are detectors (no boundary legs)
        let result = PauliWebResult {
            webs: vec![web0, web1],
            input_ids: vec![],
            output_ids: vec![],
        };

        let mut noise = NoiseModel::new();
        noise.set_edge(2, 3, ErrorRates::new(0.0, 0.01, 0.0)); // Y error only

        let dem = Dem::from_webs(&result, &noise);

        assert_eq!(dem.detectors.len(), 2);
        assert!(dem.observables.is_empty());

        // Y error fires both detectors (X != Y and Z != Y)
        assert_eq!(dem.errors.len(), 1);
        assert_eq!(dem.errors[0].detectors, vec![0, 1]);
        assert!(dem.errors[0].observables.is_empty());
    }

    #[test]
    fn test_stabilizers_excluded_from_dem() {
        // Input/output stabilizers should not appear as detectors or observables
        let mut web_det = PauliWeb::new();
        web_det.set_edge(2, 3, Pauli::X); // detector (no boundary)

        let mut web_stab = PauliWeb::new();
        web_stab.set_edge(2, 3, Pauli::Z);
        web_stab.set_edge(0, 2, Pauli::Z); // touches only input -> InputStabilizer

        let result = PauliWebResult {
            webs: vec![web_det, web_stab],
            input_ids: vec![0],
            output_ids: vec![1],
        };

        let mut noise = NoiseModel::new();
        noise.set_edge(2, 3, ErrorRates::new(0.0, 0.01, 0.0));

        let dem = Dem::from_webs(&result, &noise);

        // Only detector, no observables (stabilizer is excluded)
        assert_eq!(dem.detectors.len(), 1);
        assert_eq!(dem.observables.len(), 0);
    }

    #[test]
    fn test_stim_format_validates() {
        // Build a DEM with detectors and observables, validate with pecos-decoder-core
        let dem = Dem {
            detectors: vec![0, 1],
            observables: vec![2],
            errors: vec![
                DemError {
                    probability: 0.001,
                    detectors: vec![0],
                    observables: vec![],
                },
                DemError {
                    probability: 0.002,
                    detectors: vec![0, 1],
                    observables: vec![0],
                },
            ],
        };

        let stim = dem.to_stim_string();
        validate_dem(&stim).expect("should be valid Stim format");

        let (det_count, obs_count) = parse_dem_metadata(&stim).unwrap();
        assert_eq!(det_count, dem.detectors.len());
        assert_eq!(obs_count, dem.observables.len());
    }

    #[test]
    fn test_repetition_code_dem_structure() {
        use crate::convert::dag_to_zx;
        use crate::pauli_web::{classify_webs, compute_pauli_webs};
        use pecos_quantum::DagCircuit;

        // Build a 2-round repetition code syndrome extraction circuit
        //   data qubits: 0, 1, 2
        //   round 1 ancillas: 3, 4
        //   round 2 ancillas: 5, 6
        let mut dag = DagCircuit::new();

        // Round 1
        dag.pz(3);
        dag.pz(4);
        dag.cx(0, 3);
        dag.cx(1, 3);
        dag.cx(1, 4);
        dag.cx(2, 4);
        dag.mz(3);
        dag.mz(4);

        // Round 2 (fresh ancilla indices)
        dag.pz(5);
        dag.pz(6);
        dag.cx(0, 5);
        dag.cx(1, 5);
        dag.cx(1, 6);
        dag.cx(2, 6);
        dag.mz(5);
        dag.mz(6);

        let graph = dag_to_zx(&dag).expect("conversion failed");
        let result = compute_pauli_webs(&graph);
        let classifications = classify_webs(&result);

        // Count by classification
        let n_det = classifications
            .iter()
            .filter(|c| **c == WebClassification::Detector)
            .count();
        let n_prop = classifications
            .iter()
            .filter(|c| **c == WebClassification::Propagated)
            .count();

        assert_eq!(n_det, 2, "expected 2 detectors for 2-round rep code");
        assert!(n_prop >= 1, "expected at least 1 propagated observable");

        // Build DEM
        let noise = NoiseModel::uniform_depolarizing(&graph, 0.001);
        let dem = Dem::from_webs(&result, &noise);

        assert_eq!(dem.detectors.len(), 2);
        assert!(!dem.observables.is_empty());
        assert!(!dem.errors.is_empty());

        // Validate Stim format
        let stim = dem.to_stim_string();
        validate_dem(&stim).expect("valid Stim format");

        // Validate metadata parsing agrees with DEM structure
        let (det_count, obs_count) = parse_dem_metadata(&stim).unwrap();
        assert_eq!(det_count, dem.detectors.len());
        assert_eq!(obs_count, dem.observables.len());
    }
}
