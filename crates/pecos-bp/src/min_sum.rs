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

//! Native soft-inference primitives for detector error models.
//!
//! This module runs min-sum belief propagation (BP) on a Tanner graph and
//! returns per-mechanism posterior log-likelihood ratios. It returns beliefs,
//! never corrections: deciding how to consume the soft information belongs to
//! a decoder.

use pecos_decoder_core::dem::{DemCheckMatrix, SparseDem};
use pecos_decoder_core::errors::DecoderError;

/// Finite LLR magnitude used to represent certainty throughout native BP.
pub const LLR_SATURATION: f64 = 30.0;

/// Precomputed sparse Tanner graph for BP message passing.
///
/// The graph uses CSR-style flat arrays and is intended to be constructed once
/// and reused across shots.
#[derive(Clone, Debug)]
pub struct BpGraph {
    num_checks: usize,
    num_vars: usize,
    prior_llr: Vec<f64>,
    /// CSR for checks: flat data of (`var_idx`, `message_idx`).
    check_data: Vec<(u32, u32)>,
    /// CSR offsets for checks: `check_offset[c]..check_offset[c + 1]`.
    check_offset: Vec<u32>,
    /// CSR for variables: flat data of (`check_idx`, `message_idx`).
    var_data: Vec<(u32, u32)>,
    /// CSR offsets for variables: `var_offset[v]..var_offset[v + 1]`.
    var_offset: Vec<u32>,
    total_edges: usize,
}

impl BpGraph {
    /// Build a Tanner graph from a dense DEM check matrix.
    #[must_use]
    pub fn from_dcm(dcm: &DemCheckMatrix) -> Self {
        Self::from_connections(dcm.num_detectors, &dcm.error_priors, |check, mechanism| {
            dcm.check_matrix[[check, mechanism]] != 0
        })
    }

    /// Build a Tanner graph from a sparse detector error model.
    ///
    /// Both CSR orientations are built in `O(checks + mechanisms + edges)`
    /// while retaining check-major message numbering.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError::InvalidConfiguration`] if a probability is not
    /// in `[0, 1]`, or if a mechanism contains an out-of-range or duplicate
    /// detector index. Check, mechanism, and edge counts must fit in `u32`.
    pub fn from_sparse_dem(dem: &SparseDem) -> Result<Self, DecoderError> {
        checked_graph_index(dem.num_detectors, "check count")?;
        checked_graph_index(dem.mechanisms.len(), "mechanism count")?;
        let mut total_edges = 0usize;
        let mut last_seen = vec![usize::MAX; dem.num_detectors];
        let mut check_degrees = vec![0usize; dem.num_detectors];
        for (mechanism, (probability, detectors, _)) in dem.mechanisms.iter().enumerate() {
            if !probability.is_finite() || !(0.0..=1.0).contains(probability) {
                return Err(DecoderError::InvalidConfiguration(format!(
                    "mechanism {mechanism} probability must satisfy 0 <= p <= 1, got {probability}"
                )));
            }

            for &detector in detectors {
                let check = detector as usize;
                if check >= dem.num_detectors {
                    return Err(DecoderError::InvalidConfiguration(format!(
                        "mechanism {mechanism} detector index {detector} is out of range 0..{}",
                        dem.num_detectors
                    )));
                }
                if last_seen[check] == mechanism {
                    return Err(DecoderError::InvalidConfiguration(format!(
                        "mechanism {mechanism} repeats detector index {detector}"
                    )));
                }
                last_seen[check] = mechanism;
                check_degrees[check] = check_degrees[check].checked_add(1).ok_or_else(|| {
                    DecoderError::InvalidConfiguration(
                        "check degree exceeds the platform index space".into(),
                    )
                })?;
            }
            total_edges = total_edges.checked_add(detectors.len()).ok_or_else(|| {
                DecoderError::InvalidConfiguration(
                    "edge count exceeds the platform index space".into(),
                )
            })?;
        }
        checked_graph_index(total_edges, "edge count")?;

        Self::from_sparse_connections(dem, total_edges, &check_degrees)
    }

    /// Number of parity checks in the Tanner graph.
    #[must_use]
    pub const fn check_count(&self) -> usize {
        self.num_checks
    }

    /// Number of mechanism variables in the Tanner graph.
    #[must_use]
    pub const fn mechanism_count(&self) -> usize {
        self.num_vars
    }

    /// Number of check-to-mechanism incidences in the Tanner graph.
    #[must_use]
    pub const fn edge_count(&self) -> usize {
        self.total_edges
    }

    /// Prior per-mechanism log-likelihood ratios, in graph mechanism order.
    ///
    /// Saturated to `[-30.0, 30.0]`: probabilities below about `9.36e-14` (or
    /// above one minus that) produce the bound rather than the exact ratio,
    /// matching the values used for out-of-range probabilities. Everything
    /// inside that regime is the exact `ln((1 - p) / p)`.
    #[must_use]
    pub fn prior_llrs(&self) -> &[f64] {
        &self.prior_llr
    }

    fn from_connections(
        num_checks: usize,
        probabilities: &[f64],
        mut connected: impl FnMut(usize, usize) -> bool,
    ) -> Self {
        let num_vars = probabilities.len();
        let prior_llr = probabilities.iter().copied().map(prior_llr).collect();

        let mut temp_check: Vec<Vec<(u32, u32)>> = vec![Vec::new(); num_checks];
        let mut temp_var: Vec<Vec<(u32, u32)>> = vec![Vec::new(); num_vars];
        let mut message_index: u32 = 0;

        for (check, check_entries) in temp_check.iter_mut().enumerate() {
            for (mechanism, variable_entries) in temp_var.iter_mut().enumerate() {
                if connected(check, mechanism) {
                    check_entries.push((index(mechanism), message_index));
                    variable_entries.push((index(check), message_index));
                    message_index += 1;
                }
            }
        }

        let mut check_data = Vec::new();
        let mut check_offset = Vec::with_capacity(num_checks + 1);
        for entries in &temp_check {
            check_offset.push(index(check_data.len()));
            check_data.extend_from_slice(entries);
        }
        check_offset.push(index(check_data.len()));

        let mut var_data = Vec::new();
        let mut var_offset = Vec::with_capacity(num_vars + 1);
        for entries in &temp_var {
            var_offset.push(index(var_data.len()));
            var_data.extend_from_slice(entries);
        }
        var_offset.push(index(var_data.len()));

        Self {
            num_checks,
            num_vars,
            prior_llr,
            check_data,
            check_offset,
            var_data,
            var_offset,
            total_edges: message_index as usize,
        }
    }

    fn from_sparse_connections(
        dem: &SparseDem,
        total_edges: usize,
        check_degrees: &[usize],
    ) -> Result<Self, DecoderError> {
        let num_checks = dem.num_detectors;
        let num_vars = dem.mechanisms.len();
        let prior_llr = dem
            .mechanisms
            .iter()
            .map(|(probability, _, _)| prior_llr(*probability))
            .collect();

        let variable_degrees: Vec<usize> = dem
            .mechanisms
            .iter()
            .map(|(_, detectors, _)| detectors.len())
            .collect();
        let check_offset = csr_offsets(check_degrees, "check CSR offset")?;
        let var_offset = csr_offsets(&variable_degrees, "variable CSR offset")?;

        let mut check_data = vec![(0, 0); total_edges];
        let mut check_write: Vec<usize> = check_offset[..num_checks]
            .iter()
            .map(|&offset| offset as usize)
            .collect();
        for (mechanism, (_, detectors, _)) in dem.mechanisms.iter().enumerate() {
            let mechanism_index = checked_graph_index(mechanism, "mechanism index")?;
            for &check in detectors {
                let position = check_write[check as usize];
                let message_index = checked_graph_index(position, "message index")?;
                check_data[position] = (mechanism_index, message_index);
                check_write[check as usize] += 1;
            }
        }

        let mut var_data = vec![(0, 0); total_edges];
        let mut var_write: Vec<usize> = var_offset[..num_vars]
            .iter()
            .map(|&offset| offset as usize)
            .collect();
        for check in 0..num_checks {
            let check_index = checked_graph_index(check, "check index")?;
            let start = check_offset[check] as usize;
            let end = check_offset[check + 1] as usize;
            for &(mechanism, message_index) in &check_data[start..end] {
                let position = var_write[mechanism as usize];
                var_data[position] = (check_index, message_index);
                var_write[mechanism as usize] += 1;
            }
        }

        Ok(Self {
            num_checks,
            num_vars,
            prior_llr,
            check_data,
            check_offset,
            var_data,
            var_offset,
            total_edges,
        })
    }

    #[inline]
    pub(crate) fn check_entries(&self, check: usize) -> &[(u32, u32)] {
        let start = self.check_offset[check] as usize;
        let end = self.check_offset[check + 1] as usize;
        &self.check_data[start..end]
    }

    #[inline]
    pub(crate) fn var_entries(&self, variable: usize) -> &[(u32, u32)] {
        let start = self.var_offset[variable] as usize;
        let end = self.var_offset[variable + 1] as usize;
        &self.var_data[start..end]
    }
}

/// Narrow a graph index to the `u32` the CSR arrays store.
///
/// Indices are `u32` to keep the graph compact. A Tanner graph with more
/// than `u32::MAX` checks, mechanisms, or edges is outside the supported
/// size, so the narrowing fails loudly instead of wrapping.
fn index(value: usize) -> u32 {
    u32::try_from(value).expect("Tanner graph index exceeds the u32 index space")
}

fn checked_graph_index(value: usize, kind: &str) -> Result<u32, DecoderError> {
    u32::try_from(value).map_err(|_| {
        DecoderError::InvalidConfiguration(format!("{kind} {value} exceeds the u32 index space"))
    })
}

fn csr_offsets(degrees: &[usize], kind: &str) -> Result<Vec<u32>, DecoderError> {
    let capacity = degrees.len().checked_add(1).ok_or_else(|| {
        DecoderError::InvalidConfiguration(format!("{kind} count exceeds the platform index space"))
    })?;
    let mut offsets = Vec::with_capacity(capacity);
    let mut offset = 0usize;
    for &degree in degrees {
        offsets.push(checked_graph_index(offset, kind)?);
        offset = offset.checked_add(degree).ok_or_else(|| {
            DecoderError::InvalidConfiguration(format!("{kind} exceeds the platform index space"))
        })?;
    }
    offsets.push(checked_graph_index(offset, kind)?);
    Ok(offsets)
}

/// Reusable work buffers for [`min_sum_bp_into`].
///
/// Construct this once for a [`BpGraph`] and reuse it across shots. The BP
/// entry point resets every buffer before each run.
#[derive(Clone, Debug)]
pub struct BpScratch {
    syn_sign: Vec<f64>,
    ewa_posterior: Vec<f64>,
    c_to_v: Vec<f64>,
    v_to_c: Vec<f64>,
}

impl BpScratch {
    /// Allocate work buffers sized for `graph`.
    #[must_use]
    pub fn new(graph: &BpGraph) -> Self {
        Self {
            syn_sign: vec![1.0; graph.num_checks],
            ewa_posterior: vec![0.0; graph.num_vars],
            c_to_v: vec![0.0; graph.total_edges],
            v_to_c: vec![0.0; graph.total_edges],
        }
    }

    fn matches(&self, graph: &BpGraph) -> bool {
        self.syn_sign.len() == graph.num_checks
            && self.ewa_posterior.len() == graph.num_vars
            && self.c_to_v.len() == graph.total_edges
            && self.v_to_c.len() == graph.total_edges
    }
}

/// Run normalized min-sum BP and write posterior LLR beliefs per mechanism.
///
/// The update path preserves the established serial/flooding schedules,
/// damping, and exponentially weighted posterior accumulator. `posterior` and
/// `scratch` must be sized for `graph`; reusing both avoids allocation inside
/// this function.
///
/// # Errors
///
/// Returns [`DecoderError::InvalidDimensions`] unless `syndrome` has exactly
/// one entry per check or `posterior` has exactly one entry per mechanism.
/// Returns [`DecoderError::InvalidConfiguration`] when `scratch` was created
/// for a differently sized graph.
pub fn min_sum_bp_into(
    graph: &BpGraph,
    syndrome: &[u8],
    num_iterations: usize,
    min_sum_scale: f64,
    serial: bool,
    scratch: &mut BpScratch,
    posterior: &mut [f64],
) -> Result<(), DecoderError> {
    if syndrome.len() != graph.num_checks {
        return Err(DecoderError::InvalidDimensions {
            expected: graph.num_checks,
            actual: syndrome.len(),
        });
    }
    if posterior.len() != graph.num_vars {
        return Err(DecoderError::InvalidDimensions {
            expected: graph.num_vars,
            actual: posterior.len(),
        });
    }
    if !scratch.matches(graph) {
        return Err(DecoderError::InvalidConfiguration(
            "BpScratch dimensions do not match BpGraph".into(),
        ));
    }

    scratch.c_to_v.fill(0.0);
    scratch.v_to_c.fill(0.0);
    scratch.syn_sign.fill(1.0);
    scratch.ewa_posterior.fill(0.0);

    for variable in 0..graph.num_vars {
        for &(_, index) in graph.var_entries(variable) {
            scratch.v_to_c[index as usize] = graph.prior_llr[variable];
        }
    }

    for (check, sign) in scratch.syn_sign.iter_mut().enumerate() {
        if syndrome[check] != 0 {
            *sign = -1.0;
        }
    }

    let damp = 0.25;
    let ewa_weight = 0.3;
    scratch.ewa_posterior.copy_from_slice(&graph.prior_llr);

    let outer_iterations = if num_iterations >= 6 { 2 } else { 1 };
    let inner_iterations = if outer_iterations > 1 {
        num_iterations / outer_iterations
    } else {
        num_iterations
    };

    for outer in 0..outer_iterations {
        if outer > 0 {
            for (variable, &prior) in scratch.ewa_posterior.iter().enumerate() {
                for &(_, index) in graph.var_entries(variable) {
                    scratch.v_to_c[index as usize] = prior;
                }
            }
            scratch.c_to_v.fill(0.0);
        }

        for iteration in 0..inner_iterations {
            for (check, &syndrome_sign) in scratch.syn_sign.iter().enumerate() {
                let entries = graph.check_entries(check);
                if entries.len() < 2 {
                    continue;
                }

                let mut total_sign = syndrome_sign;
                let mut min1 = f64::INFINITY;
                let mut min2 = f64::INFINITY;
                let mut min1_position = usize::MAX;

                for (position, &(_, index)) in entries.iter().enumerate() {
                    let message = scratch.v_to_c[index as usize];
                    if message < 0.0 {
                        total_sign = -total_sign;
                    }
                    let absolute_message = message.abs();
                    if absolute_message < min1 {
                        min2 = min1;
                        min1 = absolute_message;
                        min1_position = position;
                    } else if absolute_message < min2 {
                        min2 = absolute_message;
                    }
                }

                for (position, &(_, index)) in entries.iter().enumerate() {
                    let variable_message = scratch.v_to_c[index as usize];
                    let sign_without_variable = total_sign.copysign(total_sign * variable_message);
                    let min_without_variable = if position == min1_position {
                        min2
                    } else {
                        min1
                    };
                    scratch.c_to_v[index as usize] =
                        sign_without_variable * min_without_variable * min_sum_scale;
                }

                if serial {
                    for &(variable_index, _) in entries {
                        let variable = variable_index as usize;
                        let entries = graph.var_entries(variable);
                        let total: f64 = entries
                            .iter()
                            .map(|&(_, index)| scratch.c_to_v[index as usize])
                            .sum();
                        for &(_, index) in entries {
                            let new_message =
                                graph.prior_llr[variable] + total - scratch.c_to_v[index as usize];
                            scratch.v_to_c[index as usize] =
                                (1.0 - damp) * new_message + damp * scratch.v_to_c[index as usize];
                        }
                    }
                }
            }

            if !serial {
                for (variable, &prior) in graph.prior_llr.iter().enumerate() {
                    let entries = graph.var_entries(variable);
                    let total: f64 = entries
                        .iter()
                        .map(|&(_, index)| scratch.c_to_v[index as usize])
                        .sum();
                    for &(_, index) in entries {
                        let new_message = prior + total - scratch.c_to_v[index as usize];
                        scratch.v_to_c[index as usize] =
                            (1.0 - damp) * new_message + damp * scratch.v_to_c[index as usize];
                    }
                }
            }

            let weight = if iteration == 0 && outer == 0 {
                1.0
            } else {
                ewa_weight
            };
            for (variable, ewa) in scratch.ewa_posterior.iter_mut().enumerate() {
                let current_posterior = graph.prior_llr[variable]
                    + graph
                        .var_entries(variable)
                        .iter()
                        .map(|&(_, index)| scratch.c_to_v[index as usize])
                        .sum::<f64>();
                *ewa = (1.0 - weight) * *ewa + weight * current_posterior;
            }
        }
    }

    posterior.copy_from_slice(&scratch.ewa_posterior);
    for (variable, belief) in posterior.iter_mut().enumerate() {
        let raw = graph.prior_llr[variable]
            + graph
                .var_entries(variable)
                .iter()
                .map(|&(_, index)| scratch.c_to_v[index as usize])
                .sum::<f64>();
        if (*belief > 0.0) == (raw > 0.0) && raw.abs() > belief.abs() {
            *belief = raw;
        }
    }

    // Finite priors do not guarantee finite arithmetic: message sums over
    // high-degree variables can still overflow, and `prior + total - c_to_v`
    // then evaluates infinity minus infinity. Garbage beliefs must not steer a
    // consumer silently, so non-finite posteriors are a loud failure here, at
    // the layer that produced them.
    if let Some(variable) = posterior.iter().position(|belief| !belief.is_finite()) {
        return Err(DecoderError::InternalError(format!(
            "belief propagation produced a non-finite posterior for mechanism {variable};              the model's degree and prior regime exceeds what min-sum message              accumulation can represent"
        )));
    }

    Ok(())
}

fn prior_llr(probability: f64) -> f64 {
    if probability <= 0.0 {
        LLR_SATURATION
    } else if probability >= 1.0 {
        -LLR_SATURATION
    } else {
        // The clamp keeps the computed branch inside the same +-30 saturation
        // the boundary branches already use, which changes the result exactly
        // for probabilities below ~9.36e-14 or above one minus that. Without
        // it, a subnormal probability overflows the ratio to infinity before
        // `ln`, and one infinite prior turns downstream exponentially-weighted
        // updates into NaN. Clamp returns the input bit-identically whenever
        // it is in range.
        ((1.0 - probability) / probability)
            .ln()
            .clamp(-LLR_SATURATION, LLR_SATURATION)
    }
}

#[cfg(test)]
mod tests {
    use super::{BpGraph, BpScratch, LLR_SATURATION, min_sum_bp_into, prior_llr};
    use pecos_decoder_core::dem::{DemCheckMatrix, SparseDem};
    use pecos_random::PecosRng;
    use std::collections::BTreeMap;

    #[test]
    fn sparse_csr_builder_matches_dense_connection_ordering() {
        let num_checks = 13;
        let mut rng = PecosRng::seed_from_u64(0x5eed);
        let mechanisms = (0_u32..29)
            .map(|mechanism| {
                let detectors: Vec<u32> = (0..num_checks)
                    .filter(|_| rng.next_u64() % 5 < 2)
                    .map(|check| u32::try_from(check).unwrap())
                    .collect();
                (0.01 + f64::from(mechanism) / 1_000.0, detectors, Vec::new())
            })
            .collect::<Vec<_>>();
        let probabilities = mechanisms
            .iter()
            .map(|(probability, _, _)| *probability)
            .collect::<Vec<_>>();
        let dense = BpGraph::from_connections(num_checks, &probabilities, |check, mechanism| {
            mechanisms[mechanism]
                .1
                .contains(&u32::try_from(check).unwrap())
        });
        let sparse = BpGraph::from_sparse_dem(&SparseDem {
            mechanisms,
            detector_coords: BTreeMap::new(),
            num_detectors: num_checks,
            num_observables: 0,
        })
        .unwrap();

        assert_eq!(sparse.check_data, dense.check_data);
        assert_eq!(sparse.check_offset, dense.check_offset);
        assert_eq!(sparse.var_data, dense.var_data);
        assert_eq!(sparse.var_offset, dense.var_offset);
        assert_eq!(sparse.total_edges, dense.total_edges);
    }

    /// A subnormal probability overflows `(1 - p) / p` to infinity before the
    /// logarithm; the prior must saturate at the same +-30 the boundary
    /// branches use, or one infinite prior poisons every downstream
    /// exponentially-weighted update with NaN.
    #[test]
    fn prior_llr_is_finite_and_saturated_for_subnormal_probabilities() {
        assert_eq!(prior_llr(5e-324).to_bits(), LLR_SATURATION.to_bits());
        // The mirrored extreme saturates at the negative bound.
        assert_eq!(
            prior_llr(1.0 - f64::EPSILON).to_bits(),
            (-LLR_SATURATION).to_bits()
        );
        // Ordinary probabilities are untouched bit-for-bit.
        let ordinary = 0.03_f64;
        assert_eq!(
            prior_llr(ordinary).to_bits(),
            ((1.0 - ordinary) / ordinary).ln().to_bits()
        );
    }

    #[test]
    fn scratch_is_reset_between_calls() {
        let dcm =
            DemCheckMatrix::from_dem_str("error(0.1) D0 D1 L0\nerror(0.1) D1\nerror(0.05) D0\n")
                .unwrap();
        let graph = BpGraph::from_dcm(&dcm);
        let mut reused_scratch = BpScratch::new(&graph);
        let mut reused = vec![0.0; graph.mechanism_count()];
        min_sum_bp_into(
            &graph,
            &[1, 1],
            5,
            0.625,
            true,
            &mut reused_scratch,
            &mut reused,
        )
        .unwrap();
        min_sum_bp_into(
            &graph,
            &[0, 1],
            5,
            0.625,
            true,
            &mut reused_scratch,
            &mut reused,
        )
        .unwrap();

        let mut fresh_scratch = BpScratch::new(&graph);
        let mut fresh = vec![0.0; graph.mechanism_count()];
        min_sum_bp_into(
            &graph,
            &[0, 1],
            5,
            0.625,
            true,
            &mut fresh_scratch,
            &mut fresh,
        )
        .unwrap();

        assert_eq!(
            reused
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            fresh
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
}
