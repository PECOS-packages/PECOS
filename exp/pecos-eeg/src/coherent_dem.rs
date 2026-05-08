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

//! Coherent DEM builder via backward Heisenberg mechanism extraction.
//!
//! Walks each noise source backward through the circuit to determine its
//! effective Pauli label at each detector. Groups noise sources by their
//! effective label (= same DEM mechanism), accumulates coherent amplitudes,
//! and computes mechanism probabilities.
//!
//! For H-type (coherent) noise: amplitudes add, probability = sin²(total).
//! For S-type (stochastic) noise: rates add, probability = (1-exp(2·total))/2.

use crate::Bm;
use crate::dem_mapping::{DecomposableDemEntry, DemEntry, DemEvent, Detector, Observable};
use crate::eeg::EegType;
use crate::heisenberg::{SparsePauli, sparse_conjugate};
use crate::noise::NoiseSpec;
use pecos_core::Gate;
use pecos_core::pauli::pauli_bitmask::BitmaskStorage;
use smallvec::SmallVec;
use std::collections::BTreeMap;

type FittedEventKey = (SmallVec<[usize; 4]>, SmallVec<[usize; 2]>);

/// A noise contribution at a specific gate.
struct NoiseContribution {
    /// Effective Pauli label after backward propagation.
    label: Bm,
    /// EEG type (H or S).
    eeg_type: EegType,
    /// Amplitude or rate.
    value: f64,
}

/// Build a coherent DEM by extracting mechanisms from backward propagation.
///
/// For each noise injection point in the expanded circuit, propagates
/// its Pauli label backward to the detector measurement point. Noise
/// sources that produce the same effective Pauli label are grouped into
/// a single DEM mechanism with coherently accumulated amplitude.
///
/// This gives both correct mechanism structure AND correct coherent
/// probabilities from a single framework.
pub fn build_coherent_dem(
    gates: &[Gate],
    noise: &dyn NoiseSpec,
    detectors: &[Detector],
    observables: &[Observable],
    expansion_gates: &[bool],
) -> Vec<DemEntry> {
    // Step 1: Collect all noise sources and their Pauli labels
    let mut noise_sources: Vec<(usize, NoiseContribution)> = Vec::new();

    for (gate_idx, gate) in gates.iter().enumerate() {
        if gate_idx < expansion_gates.len() && expansion_gates[gate_idx] {
            continue;
        }
        let qubits: SmallVec<[usize; 4]> =
            gate.qubits.iter().map(pecos_core::QubitId::index).collect();
        let injections = noise.noise_after_gate(gate_idx, gate.gate_type, &qubits);

        for inj in injections {
            noise_sources.push((
                gate_idx,
                NoiseContribution {
                    label: inj.label.clone(),
                    eeg_type: inj.eeg_type,
                    value: inj.rate,
                },
            ));
        }
    }

    // Step 2: For each noise source, determine which detectors it affects
    // by checking if its Pauli label, after backward propagation through
    // the circuit, anticommutes with each detector's stabilizer.
    //
    // Rather than propagating each noise label forward (expensive), we use
    // the detectors' stabilizers propagated backward to each noise location.
    // For each detector, we run the backward Heisenberg walk and at each
    // noise source check anticommutation. If the backward-propagated
    // stabilizer anticommutes with the noise label at that gate, the noise
    // source affects that detector.
    //
    // We compute this per-detector, then group noise sources by which
    // detectors they affect.

    // For each noise source: which detectors and observables it flips
    let num_noise = noise_sources.len();
    let mut noise_det_sets: Vec<SmallVec<[usize; 4]>> = vec![SmallVec::new(); num_noise];
    let mut noise_obs_sets: Vec<SmallVec<[usize; 2]>> = vec![SmallVec::new(); num_noise];

    // Build gate_index -> noise_source_indices map (shared across all walks)
    let mut gate_to_noise: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (ns_idx, (gate_idx, _)) in noise_sources.iter().enumerate() {
        gate_to_noise.entry(*gate_idx).or_default().push(ns_idx);
    }

    // Helper: propagate a stabilizer/observable backward and record
    // which noise sources anticommute with it.
    let backward_classify = |stabilizer: &Bm| -> Vec<bool> {
        let mut prop = stabilizer.clone();
        let mut hits = vec![false; num_noise];

        for gate_idx in (0..gates.len()).rev() {
            // Check noise sources BEFORE undoing the gate
            if let Some(ns_indices) = gate_to_noise.get(&gate_idx) {
                for &ns_idx in ns_indices {
                    if !prop.commutes_with(&noise_sources[ns_idx].1.label) {
                        hits[ns_idx] = true;
                    }
                }
            }
            backward_conjugate_bm(&mut prop, &gates[gate_idx]);
        }

        hits
    };

    // Classify: which detectors does each noise source flip?
    for det in detectors {
        let hits = backward_classify(&det.stabilizer);
        for (ns_idx, &hit) in hits.iter().enumerate() {
            if hit {
                noise_det_sets[ns_idx].push(det.id);
            }
        }
    }

    // Classify: which observables does each noise source flip?
    for obs in observables {
        let hits = backward_classify(&obs.pauli);
        for (ns_idx, &hit) in hits.iter().enumerate() {
            if hit {
                noise_obs_sets[ns_idx].push(obs.id);
            }
        }
    }

    // Step 3: Group noise sources by (detector_set, observable_set, eeg_type, label).
    //
    // For H-type: only noise sources with the SAME Pauli label accumulate
    // coherently. Different labels (e.g., Z on qubit 1 vs Z on qubit 2)
    // are separate mechanisms even if they flip the same detectors.
    //
    // For S-type: same grouping — different Pauli types at the same location
    // are independent mechanisms.
    //
    // After coherent accumulation per label, mechanisms with the same
    // detector set are combined independently (product formula).
    let mut h_groups: BTreeMap<(DemEvent, Bm), f64> = BTreeMap::new();
    let mut s_groups: BTreeMap<(DemEvent, Bm), f64> = BTreeMap::new();

    for (ns_idx, (_, contrib)) in noise_sources.iter().enumerate() {
        let dets = &noise_det_sets[ns_idx];
        let obs = &noise_obs_sets[ns_idx];
        if dets.is_empty() && obs.is_empty() {
            continue;
        }

        let event = DemEvent {
            detectors: dets.clone(),
            observables: obs.clone(),
        };

        let key = (event, contrib.label.clone());
        match contrib.eeg_type {
            EegType::H => {
                *h_groups.entry(key).or_insert(0.0) += contrib.value;
            }
            EegType::S => {
                *s_groups.entry(key).or_insert(0.0) += contrib.value;
            }
            _ => {}
        }
    }

    // Step 4: Compute approximate probabilities per mechanism
    let mut entries = Vec::new();

    for ((event, _label), total_h) in &h_groups {
        let prob = total_h.sin().powi(2);
        if prob > 1e-15 {
            entries.push(DemEntry {
                event: event.clone(),
                probability: prob,
            });
        }
    }

    for ((event, _label), total_s) in &s_groups {
        let prob = (1.0 - (2.0 * total_s).exp()) / 2.0;
        if prob.abs() > 1e-15 {
            entries.push(DemEntry {
                event: event.clone(),
                probability: prob.abs(),
            });
        }
    }

    merge_dem_entries(entries)
}

/// Build a coherent DEM with X/Z decomposition info for MWPM decoders.
///
/// Same mechanism extraction as `build_coherent_dem`, but additionally
/// splits each Pauli label into X-only and Z-only components and checks
/// anticommutation separately. This produces `DecomposableDemEntry`s
/// that know which detectors each component flips, enabling proper
/// graphlike decomposition for pymatching.
pub fn build_coherent_dem_decomposable(
    gates: &[Gate],
    noise: &dyn NoiseSpec,
    detectors: &[Detector],
    observables: &[Observable],
    expansion_gates: &[bool],
) -> Vec<DecomposableDemEntry> {
    // Step 1: Collect noise sources (same as build_coherent_dem)
    let mut noise_sources: Vec<(usize, NoiseContribution)> = Vec::new();

    for (gate_idx, gate) in gates.iter().enumerate() {
        if gate_idx < expansion_gates.len() && expansion_gates[gate_idx] {
            continue;
        }
        let qubits: SmallVec<[usize; 4]> =
            gate.qubits.iter().map(pecos_core::QubitId::index).collect();
        let injections = noise.noise_after_gate(gate_idx, gate.gate_type, &qubits);

        for inj in injections {
            noise_sources.push((
                gate_idx,
                NoiseContribution {
                    label: inj.label.clone(),
                    eeg_type: inj.eeg_type,
                    value: inj.rate,
                },
            ));
        }
    }

    let num_noise = noise_sources.len();

    // For each noise source: full, X-only, and Z-only detector/observable sets
    let mut noise_det_sets: Vec<SmallVec<[usize; 4]>> = vec![SmallVec::new(); num_noise];
    let mut noise_obs_sets: Vec<SmallVec<[usize; 2]>> = vec![SmallVec::new(); num_noise];
    let mut noise_x_det_sets: Vec<SmallVec<[usize; 4]>> = vec![SmallVec::new(); num_noise];
    let mut noise_x_obs_sets: Vec<SmallVec<[usize; 2]>> = vec![SmallVec::new(); num_noise];
    let mut noise_z_det_sets: Vec<SmallVec<[usize; 4]>> = vec![SmallVec::new(); num_noise];
    let mut noise_z_obs_sets: Vec<SmallVec<[usize; 2]>> = vec![SmallVec::new(); num_noise];

    // Precompute X-only and Z-only labels for each noise source
    let noise_x_labels: Vec<Bm> = noise_sources
        .iter()
        .map(|(_, c)| Bm {
            x_bits: c.label.x_bits.clone(),
            ..Default::default()
        })
        .collect();
    let noise_z_labels: Vec<Bm> = noise_sources
        .iter()
        .map(|(_, c)| Bm {
            z_bits: c.label.z_bits.clone(),
            ..Default::default()
        })
        .collect();

    // Build gate -> noise source index map
    let mut gate_to_noise: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (ns_idx, (gate_idx, _)) in noise_sources.iter().enumerate() {
        gate_to_noise.entry(*gate_idx).or_default().push(ns_idx);
    }

    // Step 2: Backward walk — check anticommutation with full, X-only, Z-only labels
    let backward_classify_xz = |stabilizer: &Bm| -> (Vec<bool>, Vec<bool>, Vec<bool>) {
        let mut prop = stabilizer.clone();
        let mut hits_full = vec![false; num_noise];
        let mut hits_x = vec![false; num_noise];
        let mut hits_z = vec![false; num_noise];

        for gate_idx in (0..gates.len()).rev() {
            if let Some(ns_indices) = gate_to_noise.get(&gate_idx) {
                for &ns_idx in ns_indices {
                    // Full anticommutation
                    if !prop.commutes_with(&noise_sources[ns_idx].1.label) {
                        hits_full[ns_idx] = true;
                    }
                    // X-only: ⟨S, P_X⟩ = S_Z · P_X
                    // P_X has no Z bits, so anticommutation only from S_Z * P_X
                    if !noise_x_labels[ns_idx].x_bits.is_zero()
                        && !prop.commutes_with(&noise_x_labels[ns_idx])
                    {
                        hits_x[ns_idx] = true;
                    }
                    // Z-only: ⟨S, P_Z⟩ = S_X · P_Z
                    // P_Z has no X bits, so anticommutation only from S_X * P_Z
                    if !noise_z_labels[ns_idx].z_bits.is_zero()
                        && !prop.commutes_with(&noise_z_labels[ns_idx])
                    {
                        hits_z[ns_idx] = true;
                    }
                }
            }
            backward_conjugate_bm(&mut prop, &gates[gate_idx]);
        }

        (hits_full, hits_x, hits_z)
    };

    // Classify detectors
    for det in detectors {
        let (hits_full, hits_x, hits_z) = backward_classify_xz(&det.stabilizer);
        for ns_idx in 0..num_noise {
            if hits_full[ns_idx] {
                noise_det_sets[ns_idx].push(det.id);
            }
            if hits_x[ns_idx] {
                noise_x_det_sets[ns_idx].push(det.id);
            }
            if hits_z[ns_idx] {
                noise_z_det_sets[ns_idx].push(det.id);
            }
        }
    }

    // Classify observables
    for obs in observables {
        let (hits_full, hits_x, hits_z) = backward_classify_xz(&obs.pauli);
        for ns_idx in 0..num_noise {
            if hits_full[ns_idx] {
                noise_obs_sets[ns_idx].push(obs.id);
            }
            if hits_x[ns_idx] {
                noise_x_obs_sets[ns_idx].push(obs.id);
            }
            if hits_z[ns_idx] {
                noise_z_obs_sets[ns_idx].push(obs.id);
            }
        }
    }

    // Step 3: Group by (event, label, eeg_type) — same as build_coherent_dem
    // but also track X/Z component events
    let mut h_groups: BTreeMap<(DemEvent, Bm), (f64, DemEvent, DemEvent)> = BTreeMap::new();
    let mut s_groups: BTreeMap<(DemEvent, Bm), (f64, DemEvent, DemEvent)> = BTreeMap::new();

    for (ns_idx, (_, contrib)) in noise_sources.iter().enumerate() {
        let dets = &noise_det_sets[ns_idx];
        let obs = &noise_obs_sets[ns_idx];
        if dets.is_empty() && obs.is_empty() {
            continue;
        }

        let event = DemEvent {
            detectors: dets.clone(),
            observables: obs.clone(),
        };
        let x_event = DemEvent {
            detectors: noise_x_det_sets[ns_idx].clone(),
            observables: noise_x_obs_sets[ns_idx].clone(),
        };
        let z_event = DemEvent {
            detectors: noise_z_det_sets[ns_idx].clone(),
            observables: noise_z_obs_sets[ns_idx].clone(),
        };

        let key = (event.clone(), contrib.label.clone());
        let groups = match contrib.eeg_type {
            EegType::H => &mut h_groups,
            EegType::S => &mut s_groups,
            _ => continue,
        };
        groups
            .entry(key)
            .and_modify(|(val, _, _)| *val += contrib.value)
            .or_insert((contrib.value, x_event, z_event));
    }

    // Step 4: Compute probabilities with decomposition info
    let mut entries = Vec::new();

    for ((event, _label), (total_h, x_ev, z_ev)) in &h_groups {
        let prob = total_h.sin().powi(2);
        if prob > 1e-15 {
            let has_x = !x_ev.detectors.is_empty() || !x_ev.observables.is_empty();
            let has_z = !z_ev.detectors.is_empty() || !z_ev.observables.is_empty();
            entries.push(DecomposableDemEntry {
                event: event.clone(),
                probability: prob,
                x_component: if has_x { Some(x_ev.clone()) } else { None },
                z_component: if has_z { Some(z_ev.clone()) } else { None },
            });
        }
    }

    for ((event, _label), (total_s, x_ev, z_ev)) in &s_groups {
        let prob = (1.0 - (2.0 * total_s).exp()) / 2.0;
        if prob.abs() > 1e-15 {
            let has_x = !x_ev.detectors.is_empty() || !x_ev.observables.is_empty();
            let has_z = !z_ev.detectors.is_empty() || !z_ev.observables.is_empty();
            entries.push(DecomposableDemEntry {
                event: event.clone(),
                probability: prob.abs(),
                x_component: if has_x { Some(x_ev.clone()) } else { None },
                z_component: if has_z { Some(z_ev.clone()) } else { None },
            });
        }
    }

    // Merge entries with identical combined events
    merge_decomposable_dem_entries(entries)
}

fn merge_decomposable_dem_entries(
    mut entries: Vec<DecomposableDemEntry>,
) -> Vec<DecomposableDemEntry> {
    if entries.len() <= 1 {
        return entries;
    }

    // Sort by combined event
    entries.sort_by(|a, b| {
        a.event
            .detectors
            .cmp(&b.event.detectors)
            .then(a.event.observables.cmp(&b.event.observables))
    });

    let mut merged = Vec::new();
    let mut i = 0;
    while i < entries.len() {
        let mut entry = entries[i].clone();
        let mut j = i + 1;
        while j < entries.len()
            && entries[j].event.detectors == entry.event.detectors
            && entries[j].event.observables == entry.event.observables
        {
            // Independent combination: p = p1 + p2 - 2*p1*p2
            entry.probability = entry.probability + entries[j].probability
                - 2.0 * entry.probability * entries[j].probability;
            // Keep X/Z components from first entry (they should be consistent
            // for same-event mechanisms, or we take the first as representative)
            j += 1;
        }
        merged.push(entry);
        i = j;
    }

    merged
}

/// Build a coherent DEM with Heisenberg-exact marginals.
///
/// Uses the backward mechanism extraction for structure (which detectors
/// each noise source flips) and fits mechanism probabilities to match
/// Heisenberg-exact per-detector marginal rates.
///
/// This combines:
/// - Correct mechanism structure from backward propagation
/// - Exact marginals from the Heisenberg walk
/// - Best independent approximation via iterative fitting
///
/// The `heisenberg_marginals` parameter should be a slice where
/// `heisenberg_marginals[det_id] = exact_detection_probability`.
///
/// The optional `heisenberg_pairwise` parameter gives exact joint rates
/// P(Di AND Dj) for detector pairs. When provided, the fit also matches
/// pairwise correlations, significantly improving 2-body and 3-body accuracy.
/// Each entry is `((det_i, det_j), joint_probability)`.
pub fn build_coherent_dem_exact(
    gates: &[Gate],
    noise: &dyn NoiseSpec,
    detectors: &[Detector],
    observables: &[Observable],
    expansion_gates: &[bool],
    heisenberg_marginals: &[f64],
    heisenberg_pairwise: Option<&[((usize, usize), f64)]>,
) -> Vec<DemEntry> {
    // Step 1-3: Get mechanism structure (same as approximate version)
    let approx = build_coherent_dem(gates, noise, detectors, observables, expansion_gates);

    if approx.is_empty() {
        return approx;
    }

    let num_dets = heisenberg_marginals.len();

    // Build incidence: for each detector, which mechanisms affect it
    let mut det_to_mechs: Vec<Vec<usize>> = vec![Vec::new(); num_dets];
    for (m, entry) in approx.iter().enumerate() {
        for &d in &entry.event.detectors {
            if d < num_dets {
                det_to_mechs[d].push(m);
            }
        }
    }

    // Extract initial probabilities
    let mut q: Vec<f64> = approx.iter().map(|e| e.probability).collect();
    let n_mech = q.len();

    // Precompute mechanism sets for pairwise computation
    let det_mech_sets: Vec<std::collections::BTreeSet<usize>> = (0..num_dets)
        .map(|d| det_to_mechs[d].iter().copied().collect())
        .collect();

    // Compute DEM marginal for detector d
    let compute_marginal = |q: &[f64], d: usize| -> f64 {
        let mut prod = 1.0;
        for &m in &det_to_mechs[d] {
            prod *= 1.0 - 2.0 * q[m];
        }
        (1.0 - prod) / 2.0
    };

    // L-BFGS optimization in sigmoid-parameterized space.
    //
    // Parameterize q_m = 0.499 * sigmoid(x_m) so x_m is unconstrained.
    // This gives a smooth loss landscape that L-BFGS can navigate efficiently.
    //
    // Loss = sum_d (marginal_d - target_d)^2
    //      + sum_pairs (pairwise_ij - target_ij)^2
    let pairs: Vec<((usize, usize), f64)> = heisenberg_pairwise
        .map(<[((usize, usize), f64)]>::to_vec)
        .unwrap_or_default();
    let has_pairwise = !pairs.is_empty();

    // Initialize x from q: x = logit(q / 0.499)
    let mut x: Vec<f64> = q
        .iter()
        .map(|&qi| {
            let s = (qi / 0.499).clamp(1e-10, 1.0 - 1e-10);
            (s / (1.0 - s)).ln()
        })
        .collect();

    let sigmoid = |xi: f64| -> f64 { 0.499 / (1.0 + (-xi).exp()) };
    let sigmoid_deriv = |xi: f64| -> f64 {
        let s = 1.0 / (1.0 + (-xi).exp());
        0.499 * s * (1.0 - s)
    };

    // Compute loss and gradient in x-space
    let compute_loss_grad = |x: &[f64]| -> (f64, Vec<f64>) {
        let q_local: Vec<f64> = x.iter().map(|&xi| sigmoid(xi)).collect();
        let dq_dx: Vec<f64> = x.iter().map(|&xi| sigmoid_deriv(xi)).collect();

        let mut grad_q = vec![0.0_f64; n_mech];
        let mut loss = 0.0_f64;

        // Marginal terms
        for d in 0..num_dets {
            let current_d = compute_marginal(&q_local, d);
            let residual = current_d - heisenberg_marginals[d];
            loss += residual * residual;

            let mut full_prod = 1.0;
            for &m in &det_to_mechs[d] {
                full_prod *= 1.0 - 2.0 * q_local[m];
            }
            for &m in &det_to_mechs[d] {
                let factor = 1.0 - 2.0 * q_local[m];
                if factor.abs() > 1e-30 {
                    grad_q[m] += 2.0 * residual * full_prod / factor;
                }
            }
        }

        // Pairwise terms
        if has_pairwise {
            let full_prods: Vec<f64> = (0..num_dets)
                .map(|d| {
                    let mut p = 1.0;
                    for &m in &det_to_mechs[d] {
                        p *= 1.0 - 2.0 * q_local[m];
                    }
                    p
                })
                .collect();

            for &((di, dj), target_p) in &pairs {
                if di >= num_dets || dj >= num_dets || target_p < 1e-10 {
                    continue;
                }

                let prod_i = full_prods[di];
                let prod_j = full_prods[dj];
                let mut prod_both = 1.0;
                for &m in det_mech_sets[di].intersection(&det_mech_sets[dj]) {
                    prod_both *= 1.0 - 2.0 * q_local[m];
                }
                let prod_xor = if prod_both.abs() > 1e-30 {
                    prod_i * prod_j / (prod_both * prod_both)
                } else {
                    0.0
                };

                let current_p = (1.0 - prod_i - prod_j + prod_xor) / 4.0;
                let residual = current_p - target_p;
                loss += residual * residual;

                for &m in det_mech_sets[di].intersection(&det_mech_sets[dj]) {
                    let factor = 1.0 - 2.0 * q_local[m];
                    if factor.abs() > 1e-30 {
                        grad_q[m] += 2.0 * residual * (prod_i + prod_j) / (2.0 * factor);
                    }
                }
                for &m in &det_to_mechs[di] {
                    if !det_mech_sets[dj].contains(&m) {
                        let factor = 1.0 - 2.0 * q_local[m];
                        if factor.abs() > 1e-30 {
                            grad_q[m] += 2.0 * residual * (prod_i - prod_xor) / (2.0 * factor);
                        }
                    }
                }
                for &m in &det_to_mechs[dj] {
                    if !det_mech_sets[di].contains(&m) {
                        let factor = 1.0 - 2.0 * q_local[m];
                        if factor.abs() > 1e-30 {
                            grad_q[m] += 2.0 * residual * (prod_j - prod_xor) / (2.0 * factor);
                        }
                    }
                }
            }
        }

        // Chain rule: grad_x = grad_q * dq/dx
        let grad_x: Vec<f64> = grad_q
            .iter()
            .zip(dq_dx.iter())
            .map(|(&gq, &dx)| gq * dx)
            .collect();

        (loss, grad_x)
    };

    // L-BFGS two-loop recursion
    let m_lbfgs = 10; // history size
    let mut s_hist: Vec<Vec<f64>> = Vec::new(); // x differences
    let mut y_hist: Vec<Vec<f64>> = Vec::new(); // gradient differences
    let mut rho_hist: Vec<f64> = Vec::new();

    let (mut loss, mut grad) = compute_loss_grad(&x);

    for _iter in 0..500 {
        if loss < 1e-14 {
            break;
        }

        // L-BFGS direction: H_k * grad
        let mut direction = grad.clone();

        // Two-loop recursion
        let hist_len = s_hist.len();
        let mut alpha = vec![0.0; hist_len];
        for i in (0..hist_len).rev() {
            alpha[i] = rho_hist[i] * dot(&s_hist[i], &direction);
            for j in 0..n_mech {
                direction[j] -= alpha[i] * y_hist[i][j];
            }
        }
        // Scale by gamma = s'y / y'y from most recent pair
        if let (Some(s), Some(y)) = (s_hist.last(), y_hist.last()) {
            let yy = dot(y, y);
            if yy > 1e-30 {
                let gamma = dot(s, y) / yy;
                for d in &mut direction {
                    *d *= gamma;
                }
            }
        }
        for i in 0..hist_len {
            let beta = rho_hist[i] * dot(&y_hist[i], &direction);
            for j in 0..n_mech {
                direction[j] += (alpha[i] - beta) * s_hist[i][j];
            }
        }

        // Negate for descent direction
        for d in &mut direction {
            *d = -*d;
        }

        // Backtracking line search (Armijo condition)
        let dg = dot(&grad, &direction);
        if dg >= 0.0 {
            break;
        } // not a descent direction

        let mut step = 1.0;
        let c1 = 1e-4;
        let mut x_new: Vec<f64> = x
            .iter()
            .zip(direction.iter())
            .map(|(&xi, &di)| xi + step * di)
            .collect();
        let (mut loss_new, mut grad_new) = compute_loss_grad(&x_new);

        for _ in 0..20 {
            if loss_new <= loss + c1 * step * dg {
                break;
            }
            step *= 0.5;
            x_new = x
                .iter()
                .zip(direction.iter())
                .map(|(&xi, &di)| xi + step * di)
                .collect();
            let (ln, gn) = compute_loss_grad(&x_new);
            loss_new = ln;
            grad_new = gn;
        }

        // Update L-BFGS history
        let s_k: Vec<f64> = x_new.iter().zip(x.iter()).map(|(&a, &b)| a - b).collect();
        let y_k: Vec<f64> = grad_new
            .iter()
            .zip(grad.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        let sy = dot(&s_k, &y_k);
        if sy > 1e-30 {
            if s_hist.len() >= m_lbfgs {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s_k);
            y_hist.push(y_k);
            rho_hist.push(1.0 / sy);
        }

        x = x_new;
        loss = loss_new;
        grad = grad_new;
    }

    // Convert back to q
    for (m, &xi) in x.iter().enumerate() {
        q[m] = sigmoid(xi);
    }

    // Build fitted DEM entries
    let fitted: Vec<DemEntry> = approx
        .iter()
        .zip(q.iter())
        .filter(|(_, p)| **p > 1e-15)
        .map(|(entry, p)| DemEntry {
            event: entry.event.clone(),
            probability: *p,
        })
        .collect();

    merge_dem_entries(fitted)
}

/// Build a coherent DEM with Heisenberg-exact marginals AND X/Z decomposition.
///
/// Combines the exact probability fitting from `build_coherent_dem_exact`
/// with the X/Z component tracking from `build_coherent_dem_decomposable`.
pub fn build_coherent_dem_exact_decomposable(
    gates: &[Gate],
    noise: &dyn NoiseSpec,
    detectors: &[Detector],
    observables: &[Observable],
    expansion_gates: &[bool],
    heisenberg_marginals: &[f64],
    heisenberg_pairwise: Option<&[((usize, usize), f64)]>,
) -> Vec<DecomposableDemEntry> {
    // Get X/Z component structure from decomposable builder
    let decomposable =
        build_coherent_dem_decomposable(gates, noise, detectors, observables, expansion_gates);

    if decomposable.is_empty() {
        return decomposable;
    }

    // Get fitted probabilities from exact builder
    let fitted = build_coherent_dem_exact(
        gates,
        noise,
        detectors,
        observables,
        expansion_gates,
        heisenberg_marginals,
        heisenberg_pairwise,
    );

    // Build lookup: event → fitted probability
    let mut prob_lookup: BTreeMap<FittedEventKey, f64> = BTreeMap::new();
    for entry in &fitted {
        prob_lookup.insert(
            (
                entry.event.detectors.clone(),
                entry.event.observables.clone(),
            ),
            entry.probability,
        );
    }

    // Combine: X/Z structure from decomposable + fitted probabilities from exact
    decomposable
        .into_iter()
        .filter_map(|mut entry| {
            let key = (
                entry.event.detectors.clone(),
                entry.event.observables.clone(),
            );
            if let Some(&fitted_prob) = prob_lookup.get(&key) {
                entry.probability = fitted_prob;
                Some(entry)
            } else if entry.probability > 1e-15 {
                // Keep original probability if no fitted version (edge case)
                Some(entry)
            } else {
                None
            }
        })
        .collect()
}

/// Merge DEM entries with the same event via independent combination.
fn merge_dem_entries(mut entries: Vec<DemEntry>) -> Vec<DemEntry> {
    entries.sort_by(|a, b| a.event.cmp(&b.event));
    let mut merged = Vec::new();
    for entry in entries {
        if let Some(last) = merged.last_mut() {
            let last: &mut DemEntry = last;
            if last.event == entry.event {
                let p1 = last.probability;
                let p2 = entry.probability;
                last.probability = p1 + p2 - 2.0 * p1 * p2;
                continue;
            }
        }
        merged.push(entry);
    }
    merged
}

/// Dot product of two slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Backward-conjugate a Bm stabilizer through a gate (Heisenberg picture).
///
/// Converts to SparsePauli, uses the tested sparse_conjugate function
/// (which already handles adjoint swapping for backward direction),
/// then converts back. Panics on unsupported gates.
fn backward_conjugate_bm(prop: &mut Bm, gate: &Gate) {
    use pecos_core::gate_type::GateType;
    match gate.gate_type {
        // Prep/alloc: kill the Pauli on the prepared qubit.
        // Z-basis prep projects onto |0>, destroying X coherences.
        // Backward propagation stops here — errors before prep
        // don't affect measurements after it.
        GateType::PZ | GateType::QAlloc => {
            for q in &gate.qubits {
                let qi = q.index();
                // Clear both X and Z on this qubit
                if prop.has_x(qi) {
                    let mut sp = SparsePauli::from_bm(prop);
                    sp.clear_x(qi as u16);
                    *prop = sp.to_bm();
                }
                if prop.has_z(qi) {
                    let mut sp = SparsePauli::from_bm(prop);
                    sp.clear_z(qi as u16);
                    *prop = sp.to_bm();
                }
            }
            return;
        }
        // Measurement: kill X on measured qubit (Z-basis measurement
        // is insensitive to Z errors, but X errors flip the result).
        // For backward propagation, we don't propagate X past MZ.
        GateType::MZ | GateType::MeasureFree | GateType::MeasureLeaked => {
            for q in &gate.qubits {
                let qi = q.index();
                if prop.has_x(qi) {
                    let mut sp = SparsePauli::from_bm(prop);
                    sp.clear_x(qi as u16);
                    *prop = sp.to_bm();
                }
            }
            return;
        }
        GateType::QFree | GateType::I | GateType::Idle => return,
        _ => {}
    }

    let mut sp = SparsePauli::from_bm(prop);
    // sparse_conjugate already applies adjoint swap for backward walk
    let _sign = sparse_conjugate(&mut sp, gate);
    // Sign is tracked in the Heisenberg walk's coefficients;
    // for the Bm-level classification we only need the Pauli structure.
    *prop = sp.to_bm();
}
