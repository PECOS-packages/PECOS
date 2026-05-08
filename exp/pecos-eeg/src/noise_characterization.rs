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

//! Unified noise characterization: correlations + mechanisms + DEM.
//!
//! Outputs use string labels ("D0", "D1", "L0") consistent with Stim format.
//! Includes detector/observable definitions mapping to MeasIds.

use crate::coherent_dem::build_coherent_dem_exact;
use crate::correlation_table::{CorrelationTableInput, compute_correlation_table};
use crate::dem_mapping::{DemEntry, DemEvent, Detector, Observable, format_dem};
use crate::noise::NoiseSpec;
use crate::stabilizer::StabilizerGroup;
use pecos_core::Gate;
use std::fmt::Write as _;

/// A correlation entry with string labels.
#[derive(Debug, Clone)]
pub struct LabeledCorrelation {
    /// Node labels: "D0", "D1", "L0", etc.
    pub labels: Vec<String>,
    /// Joint probability.
    pub probability: f64,
}

/// A mechanism in the DEM with string labels.
#[derive(Debug, Clone)]
pub struct LabeledMechanism {
    /// Detectors this mechanism flips: "D0", "D3", etc.
    pub detectors: Vec<String>,
    /// Observables this mechanism flips: "L0", etc.
    pub observables: Vec<String>,
    /// Fitted probability.
    pub probability: f64,
}

/// Definition of a detector or observable in terms of MeasIds.
#[derive(Debug, Clone)]
pub struct NodeDefinition {
    /// Label: "D0", "L0", etc.
    pub label: String,
    /// MeasIds that XOR together to produce this node's value.
    pub meas_ids: Vec<usize>,
    /// Record offsets (negative, relative to end of measurement record).
    pub records: Vec<i32>,
}

/// Complete noise characterization.
#[derive(Debug, Clone)]
pub struct NoiseCharacterization {
    /// Detector and observable definitions (label -> MeasIds).
    pub definitions: Vec<NodeDefinition>,
    /// Exact k-body correlations with string labels.
    pub correlations: Vec<LabeledCorrelation>,
    /// DEM mechanisms with string labels and fitted probabilities.
    pub mechanisms: Vec<LabeledMechanism>,
    /// Decomposable DEM entries with X/Z component info for MWPM decoders.
    pub decomposable_entries: Vec<crate::dem_mapping::DecomposableDemEntry>,
    /// Maximum correlation order computed.
    pub max_order: usize,
    /// Number of Heisenberg walks performed.
    pub num_walks: usize,
}

/// Inputs for building a complete EEG noise characterization.
#[derive(Clone, Copy)]
pub struct NoiseCharacterizationInput<'a> {
    /// Circuit gates.
    pub gates: &'a [Gate],
    /// Noise model used for exact Heisenberg correlation targets.
    pub noise: &'a dyn NoiseSpec,
    /// Optional alternate noise model used for DEM mechanism structure.
    pub structure_noise: Option<&'a dyn NoiseSpec>,
    /// Detector definitions.
    pub detectors: &'a [Detector],
    /// Observable definitions.
    pub observables: &'a [Observable],
    /// Initial stabilizer group.
    pub initial_stab: &'a StabilizerGroup,
    /// Number of circuit qubits.
    pub num_qubits: usize,
    /// Maximum correlation order to compute.
    pub max_order: usize,
    /// Drop probabilities below this threshold.
    pub prune_threshold: f64,
    /// Detector measurement-record definitions.
    pub detector_meas_ids: &'a [(usize, Vec<usize>, Vec<i32>)],
    /// Observable measurement-record definitions.
    pub observable_meas_ids: &'a [(usize, Vec<usize>, Vec<i32>)],
}

impl NoiseCharacterization {
    /// Build from circuit + noise model.
    ///
    /// `noise` is used for exact Heisenberg correlation targets.
    /// `structure_noise` (if provided) is used for DEM mechanism extraction —
    /// useful when passing compressed noise for structure while keeping
    /// original noise for exact targets. If `None`, uses `noise` for both.
    #[must_use]
    pub fn build(input: NoiseCharacterizationInput<'_>) -> Self {
        let NoiseCharacterizationInput {
            gates,
            noise,
            structure_noise,
            detectors,
            observables,
            initial_stab,
            num_qubits,
            max_order,
            prune_threshold,
            detector_meas_ids,
            observable_meas_ids,
        } = input;
        let mechanism_noise = structure_noise.unwrap_or(noise);

        // Correlation table (always uses exact noise)
        let table = compute_correlation_table(CorrelationTableInput {
            gates,
            noise,
            detectors,
            observables,
            initial_stab,
            num_qubits,
            max_order,
            prune_threshold,
        });

        // DEM with fitted probabilities (uses mechanism noise for structure)
        let num_dets = detectors.len();
        let mut marginals = vec![0.0_f64; num_dets];
        for det in detectors {
            if let Some(&p) = table.rates.get(&vec![det.id])
                && det.id < num_dets
            {
                marginals[det.id] = p;
            }
        }
        let pairwise: Vec<((usize, usize), f64)> = table
            .rates
            .iter()
            .filter(|(k, _)| k.len() == 2)
            .map(|(k, &v)| ((k[0], k[1]), v))
            .collect();
        let gate_index = crate::expand::GateIndex::build(gates, num_qubits);
        let dem_entries = build_coherent_dem_exact(
            gates,
            mechanism_noise,
            detectors,
            observables,
            &gate_index.expansion_gates,
            &marginals,
            Some(&pairwise),
        );
        let decomposable_entries = crate::coherent_dem::build_coherent_dem_exact_decomposable(
            gates,
            mechanism_noise,
            detectors,
            observables,
            &gate_index.expansion_gates,
            &marginals,
            Some(&pairwise),
        );

        // Build definitions
        let mut definitions = Vec::new();
        for &(id, ref mids, ref recs) in detector_meas_ids {
            definitions.push(NodeDefinition {
                label: format!("D{id}"),
                meas_ids: mids.clone(),
                records: recs.clone(),
            });
        }
        for &(id, ref mids, ref recs) in observable_meas_ids {
            definitions.push(NodeDefinition {
                label: format!("L{id}"),
                meas_ids: mids.clone(),
                records: recs.clone(),
            });
        }

        // Build labeled correlations from detector rates
        let mut correlations = Vec::new();
        for (key, &prob) in &table.rates {
            if prob > 1e-15 {
                let labels: Vec<String> = key.iter().map(|&d| format!("D{d}")).collect();
                correlations.push(LabeledCorrelation {
                    labels,
                    probability: prob,
                });
            }
        }
        // Add observable correlations
        for ((det_ids, obs_id), &prob) in &table.observable_rates {
            if prob > 1e-15 {
                let mut labels: Vec<String> = det_ids.iter().map(|&d| format!("D{d}")).collect();
                labels.push(format!("L{obs_id}"));
                correlations.push(LabeledCorrelation {
                    labels,
                    probability: prob,
                });
            }
        }

        // Build labeled mechanisms
        let mechanisms: Vec<LabeledMechanism> = dem_entries
            .iter()
            .filter(|e| e.probability > 1e-15)
            .map(|e| LabeledMechanism {
                detectors: e.event.detectors.iter().map(|&d| format!("D{d}")).collect(),
                observables: e
                    .event
                    .observables
                    .iter()
                    .map(|&o| format!("L{o}"))
                    .collect(),
                probability: e.probability,
            })
            .collect();

        NoiseCharacterization {
            definitions,
            correlations,
            mechanisms,
            decomposable_entries,
            max_order: table.max_order,
            num_walks: table.num_walks,
        }
    }

    /// Output as Stim DEM string.
    #[must_use]
    pub fn to_dem_string(&self) -> String {
        let entries: Vec<DemEntry> = self
            .mechanisms
            .iter()
            .map(|m| {
                let dets: Vec<usize> = m
                    .detectors
                    .iter()
                    .map(|s| s[1..].parse().unwrap_or(0))
                    .collect();
                let obs: Vec<usize> = m
                    .observables
                    .iter()
                    .map(|s| s[1..].parse().unwrap_or(0))
                    .collect();
                DemEntry {
                    event: DemEvent {
                        detectors: dets.into_iter().collect(),
                        observables: obs.into_iter().collect(),
                    },
                    probability: m.probability,
                }
            })
            .collect();
        format_dem(&entries)
    }

    /// Output as decomposed (graphlike) DEM string for MWPM decoders.
    ///
    /// Uses X/Z Pauli-aware decomposition from the backward mechanism
    /// extraction. Hyperedges are split into X ^ Z components.
    #[must_use]
    pub fn to_dem_string_decomposed(&self) -> String {
        crate::dem_mapping::format_dem_decomposed(&self.decomposable_entries)
    }

    /// Serialize to JSON.
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut j = String::from("{\n");

        let _ = writeln!(j, "  \"max_order\": {},", self.max_order);
        let _ = writeln!(j, "  \"num_walks\": {},", self.num_walks);

        // Definitions
        j.push_str("  \"definitions\": [\n");
        for (i, def) in self.definitions.iter().enumerate() {
            let _ = write!(
                j,
                "    {{\"label\": \"{}\", \"meas_ids\": {:?}, \"records\": {:?}}}",
                def.label, def.meas_ids, def.records
            );
            if i + 1 < self.definitions.len() {
                j.push(',');
            }
            j.push('\n');
        }
        j.push_str("  ],\n");

        // Correlations
        j.push_str("  \"correlations\": [\n");
        for (i, c) in self.correlations.iter().enumerate() {
            let _ = write!(
                j,
                "    {{\"nodes\": {:?}, \"probability\": {:.10e}}}",
                c.labels, c.probability
            );
            if i + 1 < self.correlations.len() {
                j.push(',');
            }
            j.push('\n');
        }
        j.push_str("  ],\n");

        // Mechanisms
        j.push_str("  \"mechanisms\": [\n");
        for (i, m) in self.mechanisms.iter().enumerate() {
            let mut nodes: Vec<&str> = m
                .detectors
                .iter()
                .map(std::string::String::as_str)
                .collect();
            nodes.extend(m.observables.iter().map(std::string::String::as_str));
            let _ = write!(
                j,
                "    {{\"nodes\": {:?}, \"probability\": {:.10e}}}",
                nodes, m.probability
            );
            if i + 1 < self.mechanisms.len() {
                j.push(',');
            }
            j.push('\n');
        }
        j.push_str("  ]\n");

        j.push('}');
        j
    }
}
