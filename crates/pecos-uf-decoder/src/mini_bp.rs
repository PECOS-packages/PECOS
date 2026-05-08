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

//! Minimal min-sum belief propagation for BP+UF hybrid decoding.
//!
//! Runs a few iterations of min-sum BP on the check matrix (Tanner graph)
//! to produce per-mechanism soft reliability scores. These scores are then
//! used to adjust UF edge weights for better accuracy.
//!
//! This is intentionally minimal -- no normalization, no scheduling tricks.
//! Just enough BP to extract useful soft information for UF.

use pecos_decoder_core::dem::DemCheckMatrix;

/// Pre-computed sparse structure for BP message passing.
/// Build once at construction time, reuse across shots.
/// Uses CSR-style flat arrays for cache-friendly iteration.
pub struct BpGraph {
    pub num_checks: usize,
    pub num_vars: usize,
    pub prior_llr: Vec<f64>,
    /// CSR for checks: flat data of (`var_idx`, `msg_array_idx`).
    check_data: Vec<(u32, u32)>,
    /// CSR offsets for checks: `check_offset`[c]..`check_offset`[c+1].
    check_offset: Vec<u32>,
    /// CSR for vars: flat data of (`check_idx`, `msg_array_idx`).
    var_data: Vec<(u32, u32)>,
    /// CSR offsets for vars: `var_offset`[v]..`var_offset`[v+1].
    var_offset: Vec<u32>,
    /// Total number of edges in the Tanner graph.
    pub total_edges: usize,
}

impl BpGraph {
    /// Get check entries (CSR slice).
    #[inline]
    #[must_use]
    pub fn check_entries(&self, c: usize) -> &[(u32, u32)] {
        let s = self.check_offset[c] as usize;
        let e = self.check_offset[c + 1] as usize;
        &self.check_data[s..e]
    }

    /// Get var entries (CSR slice).
    #[inline]
    fn var_entries(&self, v: usize) -> &[(u32, u32)] {
        let s = self.var_offset[v] as usize;
        let e = self.var_offset[v + 1] as usize;
        &self.var_data[s..e]
    }

    /// Build from a `DemCheckMatrix`.
    #[must_use]
    pub fn from_dcm(dcm: &DemCheckMatrix) -> Self {
        let num_checks = dcm.num_detectors;
        let num_vars = dcm.num_mechanisms;

        let prior_llr: Vec<f64> = dcm
            .error_priors
            .iter()
            .map(|&p| {
                if p <= 0.0 {
                    30.0
                } else if p >= 1.0 {
                    -30.0
                } else {
                    ((1.0 - p) / p).ln()
                }
            })
            .collect();

        // Build temporary adjacency then flatten to CSR.
        let mut temp_check: Vec<Vec<(u32, u32)>> = vec![Vec::new(); num_checks];
        let mut temp_var: Vec<Vec<(u32, u32)>> = vec![Vec::new(); num_vars];
        let mut msg_idx: u32 = 0;

        for (c, check_entries) in temp_check.iter_mut().enumerate().take(num_checks) {
            for (v, var_entries) in temp_var.iter_mut().enumerate().take(num_vars) {
                if dcm.check_matrix[[c, v]] != 0 {
                    check_entries.push((v as u32, msg_idx));
                    var_entries.push((c as u32, msg_idx));
                    msg_idx += 1;
                }
            }
        }

        // Flatten check entries.
        let mut check_data = Vec::new();
        let mut check_offset = Vec::with_capacity(num_checks + 1);
        for entries in &temp_check {
            check_offset.push(check_data.len() as u32);
            check_data.extend_from_slice(entries);
        }
        check_offset.push(check_data.len() as u32);

        // Flatten var entries.
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
            total_edges: msg_idx as usize,
        }
    }
}

/// Run min-sum BP on a pre-computed graph and return posterior LLRs per mechanism.
///
/// - `graph`: pre-computed sparse BP graph
/// - `syndrome`: detection events (1 = triggered)
/// - `num_iterations`: number of BP iterations
/// - `min_sum_scale`: scaling factor for min-sum messages (0.625 is standard)
/// - `serial`: if true, use serial schedule (better convergence, slower)
/// - `c_to_v`, `v_to_c`: reusable message buffers (must be `graph.total_edges` long)
/// - `posterior`: output buffer (must be `graph.num_vars` long)
#[allow(clippy::too_many_arguments)] // Hot-path helper takes reusable buffers explicitly.
pub fn min_sum_bp_into(
    graph: &BpGraph,
    syndrome: &[u8],
    num_iterations: usize,
    min_sum_scale: f64,
    serial: bool,
    c_to_v: &mut [f64],
    v_to_c: &mut [f64],
    posterior: &mut Vec<f64>,
) {
    let num_checks = graph.num_checks;
    let num_vars = graph.num_vars;

    // Initialize v→c with priors.
    for v in 0..num_vars {
        for &(_c, idx) in graph.var_entries(v) {
            v_to_c[idx as usize] = graph.prior_llr[v];
        }
    }

    // Pre-compute syndrome signs (avoid branch in inner loop).
    let mut syn_sign = vec![1.0f64; num_checks];
    for (c, sign) in syn_sign.iter_mut().enumerate() {
        if c < syndrome.len() && syndrome[c] != 0 {
            *sign = -1.0;
        }
    }

    let damp = 0.25;

    // EWA posterior accumulator.
    let ewa_weight = 0.3;
    let mut ewa_posterior = vec![0.0f64; num_vars];
    ewa_posterior.copy_from_slice(&graph.prior_llr);

    // EWAInit: run BP multiple times, using EWA of previous posteriors
    // as the prior for the next run. This finds better fixed points.
    let outer_iterations = if num_iterations >= 6 { 2 } else { 1 };
    let inner_iterations = if outer_iterations > 1 {
        num_iterations / outer_iterations
    } else {
        num_iterations
    };

    for outer in 0..outer_iterations {
        // Re-initialize v→c with current EWA posteriors as priors.
        if outer > 0 {
            for (v, &prior) in ewa_posterior.iter().enumerate().take(num_vars) {
                for &(_c, idx) in graph.var_entries(v) {
                    v_to_c[idx as usize] = prior;
                }
            }
            c_to_v.fill(0.0);
        }

        for iter in 0..inner_iterations {
            for (c, &syndrome_sign) in syn_sign.iter().enumerate().take(num_checks) {
                let entries = graph.check_entries(c);
                if entries.len() < 2 {
                    continue;
                }

                // Check-to-variable (normalized min-sum).
                let mut total_sign = syndrome_sign;
                let mut min1 = f64::INFINITY;
                let mut min2 = f64::INFINITY;
                let mut min1_pos = usize::MAX;

                for (pos, &(_v, idx)) in entries.iter().enumerate() {
                    let msg = v_to_c[idx as usize];
                    if msg < 0.0 {
                        total_sign = -total_sign;
                    }
                    let abs_msg = msg.abs();
                    if abs_msg < min1 {
                        min2 = min1;
                        min1 = abs_msg;
                        min1_pos = pos;
                    } else if abs_msg < min2 {
                        min2 = abs_msg;
                    }
                }

                for (pos, &(_v, idx)) in entries.iter().enumerate() {
                    let msg_v = v_to_c[idx as usize];
                    let sign_without_v = total_sign.copysign(total_sign * msg_v);
                    let min_without_v = if pos == min1_pos { min2 } else { min1 };
                    c_to_v[idx as usize] = sign_without_v * min_without_v * min_sum_scale;
                }

                if serial {
                    // Serial: immediately update v→c for connected variables.
                    for &(v_idx, _) in entries {
                        let v = v_idx as usize;
                        let gamma = damp;
                        let v_entries = graph.var_entries(v);
                        let total: f64 = v_entries
                            .iter()
                            .map(|&(_c2, idx2)| c_to_v[idx2 as usize])
                            .sum();
                        for &(_c2, idx2) in v_entries {
                            let new_msg = graph.prior_llr[v] + total - c_to_v[idx2 as usize];
                            v_to_c[idx2 as usize] =
                                (1.0 - gamma) * new_msg + gamma * v_to_c[idx2 as usize];
                        }
                    }
                }
            }

            if !serial {
                // Flooding: batch update all variables after all checks.
                for (v, &prior) in graph.prior_llr.iter().enumerate().take(num_vars) {
                    let gamma = damp;
                    let entries = graph.var_entries(v);
                    let total: f64 = entries.iter().map(|&(_c, idx)| c_to_v[idx as usize]).sum();
                    for &(_c, idx) in entries {
                        let new_msg = prior + total - c_to_v[idx as usize];
                        v_to_c[idx as usize] =
                            (1.0 - gamma) * new_msg + gamma * v_to_c[idx as usize];
                    }
                }
            }

            // EWA: blend current iteration's posterior into the running average.
            let w = if iter == 0 && outer == 0 {
                1.0
            } else {
                ewa_weight
            };
            for (v, ewa) in ewa_posterior.iter_mut().enumerate().take(num_vars) {
                let cur_posterior = graph.prior_llr[v]
                    + graph
                        .var_entries(v)
                        .iter()
                        .map(|&(_c, idx)| c_to_v[idx as usize])
                        .sum::<f64>();
                *ewa = (1.0 - w) * *ewa + w * cur_posterior;
            }
        } // end inner iteration loop
    } // end outer EWAInit loop

    // Use EWA-averaged posteriors (smoothed across all iterations).
    posterior.clear();
    posterior.extend_from_slice(&ewa_posterior);
    // Also include final iteration's raw posterior for variables where
    // EWA and raw agree in sign (reinforcement).
    for (v, post) in posterior.iter_mut().enumerate().take(num_vars) {
        let raw = graph.prior_llr[v]
            + graph
                .var_entries(v)
                .iter()
                .map(|&(_c, idx)| c_to_v[idx as usize])
                .sum::<f64>();
        // If EWA and raw agree, use the one with larger magnitude (more confident).
        if (*post > 0.0) == (raw > 0.0) && raw.abs() > post.abs() {
            *post = raw;
        }
        // If they disagree, keep EWA (it's more stable).
    }
}

/// BP on the matching graph (Hack et al. 2026 style).
///
/// Instead of running BP on the Tanner graph (check matrix), run on the
/// matching graph where:
/// - Variables = matching graph edges (is this edge in the correction?)
/// - Factors = detector nodes (parity constraint from syndrome)
///
/// The matching graph has simpler topology (no hyperedges, more tree-like),
/// so BP converges better. Returns per-edge posterior LLRs.
#[must_use]
pub fn matching_graph_bp(
    graph: &pecos_decoder_core::dem::DemMatchingGraph,
    syndrome: &[u8],
    num_iterations: usize,
    min_sum_scale: f64,
) -> Vec<f64> {
    let num_nodes = graph.num_detectors + 1; // +1 for boundary
    let num_edges = graph.edges.len();
    let boundary = graph.num_detectors;

    // Prior LLRs for each edge.
    let prior_llr: Vec<f64> = graph.edges.iter().map(|e| e.weight).collect();

    // Build adjacency: for each node, list of incident edges.
    let mut node_edges: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
    for (idx, edge) in graph.edges.iter().enumerate() {
        node_edges[edge.node1 as usize].push(idx);
        if let Some(n2) = edge.node2 {
            node_edges[n2 as usize].push(idx);
        } else {
            node_edges[boundary].push(idx);
        }
    }

    // Messages: node-to-edge and edge-to-node.
    // For each (node, edge) pair, store the message index.
    let mut msg_idx = 0usize;
    let mut node_msg: Vec<Vec<(usize, usize)>> = vec![Vec::new(); num_nodes]; // node -> [(edge_idx, msg_idx)]
    let mut edge_msg: Vec<Vec<(usize, usize)>> = vec![Vec::new(); num_edges]; // edge -> [(node_idx, msg_idx)]
    for (node, edges) in node_edges.iter().enumerate() {
        for &edge_idx in edges {
            node_msg[node].push((edge_idx, msg_idx));
            edge_msg[edge_idx].push((node, msg_idx));
            msg_idx += 1;
        }
    }

    let total_msgs = msg_idx;
    let mut n_to_e = vec![0.0f64; total_msgs]; // node→edge messages
    let mut e_to_n = vec![0.0f64; total_msgs]; // edge→node messages

    // Initialize edge→node with prior LLRs.
    for (edge_idx, entries) in edge_msg.iter().enumerate() {
        for &(_, midx) in entries {
            e_to_n[midx] = prior_llr[edge_idx];
        }
    }

    // Syndrome sign.
    let syn_sign: Vec<f64> = (0..num_nodes)
        .map(|n| {
            if n < syndrome.len() && syndrome[n] != 0 {
                -1.0
            } else {
                1.0
            }
        })
        .collect();

    let damp = 0.25;

    for _iter in 0..num_iterations {
        // Node-to-edge (check-to-variable): min-sum update.
        // Same as Tanner graph BP but on matching graph nodes.
        for node in 0..num_nodes {
            let entries = &node_msg[node];
            if entries.len() < 2 {
                continue;
            }

            let mut total_sign = syn_sign[node];
            let mut min1 = f64::INFINITY;
            let mut min2 = f64::INFINITY;
            let mut min1_pos = usize::MAX;

            for (pos, &(_, midx)) in entries.iter().enumerate() {
                let msg = e_to_n[midx];
                if msg < 0.0 {
                    total_sign = -total_sign;
                }
                let abs_msg = msg.abs();
                if abs_msg < min1 {
                    min2 = min1;
                    min1 = abs_msg;
                    min1_pos = pos;
                } else if abs_msg < min2 {
                    min2 = abs_msg;
                }
            }

            for (pos, &(_, midx)) in entries.iter().enumerate() {
                let msg_v = e_to_n[midx];
                let sign_without = total_sign.copysign(total_sign * msg_v);
                let min_without = if pos == min1_pos { min2 } else { min1 };
                n_to_e[midx] = sign_without * min_without * min_sum_scale;
            }
        }

        // Edge-to-node (variable-to-check): sum incoming + prior.
        for (edge_idx, entries) in edge_msg.iter().enumerate() {
            let total: f64 = entries.iter().map(|&(_, midx)| n_to_e[midx]).sum();
            for &(_, midx) in entries {
                let new_msg = prior_llr[edge_idx] + total - n_to_e[midx];
                e_to_n[midx] = (1.0 - damp) * new_msg + damp * e_to_n[midx];
            }
        }
    }

    // Posterior: prior + sum of all node→edge messages.
    let mut posterior = prior_llr;
    for (edge_idx, entries) in edge_msg.iter().enumerate() {
        for &(_, midx) in entries {
            posterior[edge_idx] += n_to_e[midx];
        }
    }

    posterior
}

/// Convenience wrapper: build graph, run BP, return posteriors.
#[must_use]
pub fn min_sum_bp(
    dcm: &DemCheckMatrix,
    syndrome: &[u8],
    num_iterations: usize,
    min_sum_scale: f64,
) -> Vec<f64> {
    let graph = BpGraph::from_dcm(dcm);
    let mut c_to_v = vec![0.0f64; graph.total_edges];
    let mut v_to_c = vec![0.0f64; graph.total_edges];
    let mut posterior = Vec::with_capacity(graph.num_vars);
    min_sum_bp_into(
        &graph,
        syndrome,
        num_iterations,
        min_sum_scale,
        false,
        &mut c_to_v,
        &mut v_to_c,
        &mut posterior,
    );
    posterior
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mini_bp_no_syndrome() {
        // Simple 2-check, 3-mechanism DEM.
        let dem_str = "\
error(0.1) D0 D1 L0
error(0.1) D1
error(0.05) D0
";
        let dcm = DemCheckMatrix::from_dem_str(dem_str).unwrap();
        let syndrome = vec![0u8; dcm.num_detectors];

        let posterior = min_sum_bp(&dcm, &syndrome, 5, 0.625);
        assert_eq!(posterior.len(), dcm.num_mechanisms);

        // With no syndrome, all posteriors should be positive (no error likely).
        for &llr in &posterior {
            assert!(llr > 0.0, "Expected positive LLR for no-syndrome case");
        }
    }

    #[test]
    fn test_mini_bp_with_syndrome() {
        let dem_str = "\
error(0.1) D0 D1 L0
error(0.1) D1
error(0.05) D0
";
        let dcm = DemCheckMatrix::from_dem_str(dem_str).unwrap();
        // D0 and D1 both triggered -> mechanism 0 (D0-D1) is likely.
        let syndrome = vec![1, 1];

        let posterior = min_sum_bp(&dcm, &syndrome, 5, 0.625);
        assert_eq!(posterior.len(), dcm.num_mechanisms);

        // Mechanism 0 (D0 D1) should have lower (more negative) LLR
        // since both its checks are triggered.
        assert!(
            posterior[0] < posterior[2],
            "Mechanism touching both triggered checks should be more likely"
        );
    }
}
