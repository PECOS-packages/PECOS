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

use crate::dem::{DemCheckMatrix, SparseDem};
use crate::errors::DecoderError;

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
    /// # Errors
    ///
    /// Returns [`DecoderError::InvalidConfiguration`] if a probability is not
    /// in `[0, 1]`, or if a mechanism contains an out-of-range or duplicate
    /// detector index.
    pub fn from_sparse_dem(dem: &SparseDem) -> Result<Self, DecoderError> {
        for (mechanism, (probability, detectors, _)) in dem.mechanisms.iter().enumerate() {
            if !probability.is_finite() || !(0.0..=1.0).contains(probability) {
                return Err(DecoderError::InvalidConfiguration(format!(
                    "mechanism {mechanism} probability must satisfy 0 <= p <= 1, got {probability}"
                )));
            }

            let mut seen = std::collections::BTreeSet::new();
            for &detector in detectors {
                if detector as usize >= dem.num_detectors {
                    return Err(DecoderError::InvalidConfiguration(format!(
                        "mechanism {mechanism} detector index {detector} is out of range 0..{}",
                        dem.num_detectors
                    )));
                }
                if !seen.insert(detector) {
                    return Err(DecoderError::InvalidConfiguration(format!(
                        "mechanism {mechanism} repeats detector index {detector}"
                    )));
                }
            }
        }

        let probabilities: Vec<f64> = dem
            .mechanisms
            .iter()
            .map(|(probability, _, _)| *probability)
            .collect();
        Ok(Self::from_connections(
            dem.num_detectors,
            &probabilities,
            |check, mechanism| dem.mechanisms[mechanism].1.contains(&(check as u32)),
        ))
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
                    check_entries.push((mechanism as u32, message_index));
                    variable_entries.push((check as u32, message_index));
                    message_index += 1;
                }
            }
        }

        let mut check_data = Vec::new();
        let mut check_offset = Vec::with_capacity(num_checks + 1);
        for entries in &temp_check {
            check_offset.push(check_data.len() as u32);
            check_data.extend_from_slice(entries);
        }
        check_offset.push(check_data.len() as u32);

        let mut var_data = Vec::new();
        let mut var_offset = Vec::with_capacity(num_vars + 1);
        for entries in &temp_var {
            var_offset.push(var_data.len() as u32);
            var_data.extend_from_slice(entries);
        }
        var_offset.push(var_data.len() as u32);

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

    #[inline]
    fn check_entries(&self, check: usize) -> &[(u32, u32)] {
        let start = self.check_offset[check] as usize;
        let end = self.check_offset[check + 1] as usize;
        &self.check_data[start..end]
    }

    #[inline]
    fn var_entries(&self, variable: usize) -> &[(u32, u32)] {
        let start = self.var_offset[variable] as usize;
        let end = self.var_offset[variable + 1] as usize;
        &self.var_data[start..end]
    }
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

    Ok(())
}

fn prior_llr(probability: f64) -> f64 {
    if probability <= 0.0 {
        30.0
    } else if probability >= 1.0 {
        -30.0
    } else {
        ((1.0 - probability) / probability).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::{BpGraph, BpScratch, min_sum_bp_into};
    use crate::dem::DemCheckMatrix;

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
