// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! Python bindings for EEG DEM builder.

use pecos_core::pauli::pauli_bitmask::BitmaskStorage;
use pecos_core::{Angle64, Gate, GateAngles, GateMeasIds, GateParams, QubitId};
use pecos_eeg::Bm;
use pecos_eeg::circuit::{self, NoiseModel};
use pecos_eeg::correlation_table::CorrelationTableInput;
use pecos_eeg::dem_mapping::{DemEntry, Detector, Observable};
use pecos_eeg::noise_characterization::NoiseCharacterizationInput;
use pyo3::prelude::*;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::collections::BTreeMap;

type PyDemEvent = (f64, Vec<usize>, Vec<usize>);
type PyEegEventDiagnostic = (Vec<usize>, usize, usize, Vec<f64>, f64);
type MeasurementRecordDefinition = (usize, Vec<usize>, Vec<i32>);

/// Build a DEM using forward EEG analysis (perturbative, fast).
///
/// Returns (raw_dem, decomposed_dem) where the decomposed version uses
/// X/Z Pauli-aware decomposition for MWPM decoders.
///
/// Fast (milliseconds) but approximate (~50% error for coherent noise).
/// For exact probabilities, use `coherent_dem_decomposed`.
///
/// h_formula: "taylor" (default), "sin_squared", "exact_commuting", or "exact_subset"
#[pyfunction]
#[pyo3(signature = (tick_circuit, idle_rz=0.0, p1=0.0, p2=0.0, p_meas=0.0, p_prep=0.0, h_formula="taylor", bch_order=1))]
pub fn perturbative_dem(
    tick_circuit: &Bound<'_, PyAny>,
    idle_rz: f64,
    p1: f64,
    p2: f64,
    p_meas: f64,
    p_prep: f64,
    h_formula: &str,
    bch_order: u32,
) -> PyResult<(String, String)> {
    let (raw, decomposable) = run_eeg_decomposable(
        tick_circuit,
        idle_rz,
        p1,
        p2,
        p_meas,
        p_prep,
        h_formula,
        bch_order,
    )?;
    Ok((
        pecos_eeg::dem_mapping::format_dem(&raw),
        pecos_eeg::dem_mapping::format_dem_decomposed(&decomposable),
    ))
}

/// Build perturbative DEM and return structured events: list of (prob, [det_ids], [obs_ids]).
#[pyfunction]
#[pyo3(signature = (tick_circuit, idle_rz=0.0, p1=0.0, p2=0.0, p_meas=0.0, p_prep=0.0, h_formula="taylor", bch_order=1))]
pub fn perturbative_dem_events(
    tick_circuit: &Bound<'_, PyAny>,
    idle_rz: f64,
    p1: f64,
    p2: f64,
    p_meas: f64,
    p_prep: f64,
    h_formula: &str,
    bch_order: u32,
) -> PyResult<Vec<PyDemEvent>> {
    let entries = run_eeg(
        tick_circuit,
        idle_rz,
        p1,
        p2,
        p_meas,
        p_prep,
        h_formula,
        bch_order,
    )?;
    Ok(entries
        .into_iter()
        .map(|e| {
            (
                e.probability,
                e.event.detectors.to_vec(),
                e.event.observables.to_vec(),
            )
        })
        .collect())
}

/// Return (num_h_generators, num_s_generators, num_detectors, numobservables).
#[pyfunction]
#[pyo3(signature = (tick_circuit, idle_rz=0.0, p1=0.0, p2=0.0, p_meas=0.0, p_prep=0.0))]
pub fn eeg_summary(
    tick_circuit: &Bound<'_, PyAny>,
    idle_rz: f64,
    p1: f64,
    p2: f64,
    p_meas: f64,
    p_prep: f64,
) -> PyResult<(usize, usize, usize, usize)> {
    let noise = NoiseModel {
        idle_rz,
        p1,
        p2,
        p_meas,
        p_prep,
    };
    let gates = extract_gates(tick_circuit)?;
    let expanded = pecos_eeg::expand::expand_circuit(&gates);
    let result = circuit::analyze_expanded(&expanded.gates, &noise);
    let (detectors, observables) = extract_detectors_expanded(tick_circuit, &expanded)?;
    let h = result
        .generators
        .iter()
        .filter(|g| g.eeg_type == pecos_eeg::eeg::EegType::H)
        .count();
    let s = result
        .generators
        .iter()
        .filter(|g| g.eeg_type == pecos_eeg::eeg::EegType::S)
        .count();
    Ok((h, s, detectors.len(), observables.len()))
}

/// Diagnostic: for each DEM event, return generator details.
///
/// Returns list of (det_ids, num_labels, num_same_label_groups, rates_by_label, max_combined_rate)
/// This helps understand why perturbative formulas are inaccurate.
#[pyfunction]
#[pyo3(signature = (tick_circuit, idle_rz=0.0, p1=0.0, p2=0.0, p_meas=0.0, p_prep=0.0))]
pub fn eeg_event_diagnostics(
    tick_circuit: &Bound<'_, PyAny>,
    idle_rz: f64,
    p1: f64,
    p2: f64,
    p_meas: f64,
    p_prep: f64,
) -> PyResult<Vec<PyEegEventDiagnostic>> {
    let noise = NoiseModel {
        idle_rz,
        p1,
        p2,
        p_meas,
        p_prep,
    };
    let gates = extract_gates(tick_circuit)?;
    let expanded = pecos_eeg::expand::expand_circuit(&gates);
    let result = circuit::analyze_expanded(&expanded.gates, &noise);
    let (detectors, _observables) = extract_detectors_expanded(tick_circuit, &expanded)?;

    // Group H generators by DEM event, tracking labels
    let mut h_events: BTreeMap<Vec<usize>, BTreeMap<Bm, f64>> = BTreeMap::new();

    for g in &result.generators {
        if g.eeg_type != pecos_eeg::eeg::EegType::H {
            continue;
        }
        // Classify manually
        let mut dets = Vec::new();
        for det in &detectors {
            if !g.label.commutes_with(&det.stabilizer) {
                dets.push(det.id);
            }
        }
        if dets.is_empty() {
            continue;
        }
        *h_events
            .entry(dets)
            .or_default()
            .entry(g.label.clone())
            .or_insert(0.0) += g.coeff;
    }

    let mut out = Vec::new();
    for (det_ids, labels) in &h_events {
        let num_labels = labels.len();
        let rates: Vec<f64> = labels.values().copied().collect();
        let num_groups = rates.iter().filter(|&&r| r.abs() > 1e-15).count();
        let max_combined = rates.iter().map(|r| r.abs()).fold(0.0_f64, f64::max);
        out.push((det_ids.clone(), num_labels, num_groups, rates, max_combined));
    }
    Ok(out)
}

/// Compute per-detector marginals using ALL generators that flip each detector.
///
/// Unlike the DEM-based approach (which groups by event then sums), this pools
/// all H generators for each detector into a single quadratic form with beta
/// cross-terms. This captures cross-event interference that the per-event
/// computation misses.
///
/// Returns list of (detector_id, probability).
#[pyfunction]
#[pyo3(signature = (tick_circuit, idle_rz=0.0, p1=0.0, p2=0.0, p_meas=0.0, p_prep=0.0, h_formula="taylor"))]
pub fn eeg_per_detector(
    tick_circuit: &Bound<'_, PyAny>,
    idle_rz: f64,
    p1: f64,
    p2: f64,
    p_meas: f64,
    p_prep: f64,
    h_formula: &str,
) -> PyResult<Vec<(usize, f64)>> {
    let noise = NoiseModel {
        idle_rz,
        p1,
        p2,
        p_meas,
        p_prep,
    };
    let gates = extract_gates(tick_circuit)?;
    let expanded = pecos_eeg::expand::expand_circuit(&gates);
    let result = circuit::analyze_expanded(&expanded.gates, &noise);
    let (detectors, _observables) = extract_detectors_expanded(tick_circuit, &expanded)?;

    let expanded_pre_readout = exclude_final_mz(&expanded.gates);
    let stab_group = pecos_eeg::stabilizer::StabilizerGroup::from_circuit(
        &expanded_pre_readout,
        expanded.num_qubits,
    );

    let formula = parse_h_formula(h_formula)?;

    // Collect all H generators with their BCH-combined rates per label
    let mut h_by_label: std::collections::BTreeMap<Bm, f64> = std::collections::BTreeMap::new();
    for g in &result.generators {
        if g.eeg_type == pecos_eeg::eeg::EegType::H {
            *h_by_label.entry(g.label.clone()).or_insert(0.0) += g.coeff;
        }
    }
    let h_labels: Vec<(Bm, f64)> = h_by_label
        .into_iter()
        .filter(|(_, c)| c.abs() > 1e-20)
        .collect();

    // Also collect S generators
    let mut s_by_label: std::collections::BTreeMap<Bm, f64> = std::collections::BTreeMap::new();
    for g in &result.generators {
        if g.eeg_type == pecos_eeg::eeg::EegType::S {
            *s_by_label.entry(g.label.clone()).or_insert(0.0) += g.coeff;
        }
    }

    let mut results = Vec::new();

    for det in &detectors {
        // Find all H generators that anticommute with this detector
        let det_h: Vec<(usize, f64)> = h_labels
            .iter()
            .enumerate()
            .filter(|(_, (label, _))| !label.commutes_with(&det.stabilizer))
            .map(|(i, (_, c))| (i, *c))
            .collect();

        // Find S generators that anticommute
        let s_sum: f64 = s_by_label
            .iter()
            .filter(|(label, _)| !label.commutes_with(&det.stabilizer))
            .map(|(_, c)| c)
            .sum();

        // S contribution
        let p_s = if s_sum.abs() > 1e-20 {
            (1.0 - (2.0 * s_sum).exp()) / 2.0
        } else {
            0.0
        };

        // H contribution: quadratic form with beta
        let h_prob = match formula {
            pecos_eeg::dem_mapping::HFormula::Taylor
            | pecos_eeg::dem_mapping::HFormula::SinSquared
            | pecos_eeg::dem_mapping::HFormula::ExactSubset => {
                let mut total = 0.0_f64;
                for (j, &(idx_j, h_j)) in det_h.iter().enumerate() {
                    // Diagonal
                    total += h_j * h_j;
                    // Off-diagonal with beta
                    for &(idx_k, h_k) in det_h.iter().skip(j + 1) {
                        let q_j = &h_labels[idx_j].0;
                        let q_k = &h_labels[idx_k].0;

                        if !q_j.commutes_with(q_k) {
                            continue;
                        }
                        let product = q_j.multiply(q_k);
                        if product.is_identity() {
                            total += 2.0 * h_j * h_k;
                            continue;
                        }
                        match stab_group.is_stabilizer(&product) {
                            Some(true) => {
                                total += 2.0 * h_j * h_k;
                            }
                            Some(false) => {
                                total -= 2.0 * h_j * h_k;
                            }
                            None => {}
                        }
                    }
                }
                let total = total.max(0.0);
                match formula {
                    pecos_eeg::dem_mapping::HFormula::SinSquared => total.sqrt().sin().powi(2),
                    _ => total,
                }
            }
            pecos_eeg::dem_mapping::HFormula::ExactCommuting => {
                // Product formula over all generators for this detector
                let mut prod_re = 1.0_f64;
                let mut prod_im = 0.0_f64;
                for &(idx_j, h_j) in &det_h {
                    let label = &h_labels[idx_j].0;
                    let p_stab = if label.is_identity() {
                        Some(true)
                    } else {
                        stab_group.is_stabilizer(label)
                    };

                    let (f_re, f_im) = if let Some(sign) = p_stab {
                        let s = if sign { 1.0 } else { -1.0 };
                        let angle = 2.0 * s * h_j;
                        (angle.cos(), angle.sin())
                    } else {
                        let dp = det.stabilizer.multiply(label);
                        let dp_stab = if dp.is_identity() {
                            Some(true)
                        } else {
                            stab_group.is_stabilizer(&dp)
                        };
                        if let Some(sign) = dp_stab {
                            let s = if sign { 1.0 } else { -1.0 };
                            let angle = -2.0 * s * h_j;
                            (angle.cos(), angle.sin())
                        } else {
                            ((2.0 * h_j).cos(), 0.0)
                        }
                    };
                    let new_re = prod_re * f_re - prod_im * f_im;
                    let new_im = prod_re * f_im + prod_im * f_re;
                    prod_re = new_re;
                    prod_im = new_im;
                }
                (0.5 * (1.0 - prod_re)).max(0.0)
            }
        };

        // Combine S and H (independent)
        let p = if p_s.abs() > 1e-15 && h_prob > 1e-15 {
            p_s + h_prob - 2.0 * p_s * h_prob
        } else {
            p_s.abs() + h_prob
        };
        results.push((det.id, p));
    }

    Ok(results)
}

/// Compute exact per-detector detection probabilities.
///
/// Uses backward Heisenberg propagation: walks the detector observable
/// backward through the circuit, splitting at each noise source. Exact
/// for both coherent (idle_rz) and stochastic (depolarizing) noise.
///
/// This is the most accurate DEM generation method in PECOS. Use it when:
/// - You need exact detection rates under coherent noise
/// - You want to validate the non-EEG DEM builder
/// - You need per-detector probabilities (not a full DEM)
///
/// For a full DEM (event structure + probabilities), use `eeg_heisenberg_dem`.
/// For fast approximate rates under coherent noise, use `eeg_dem_events`.
/// For depolarizing-only noise, `DemSampler.from_circuit` is faster and exact.
///
/// Returns: list of (detector_id, probability) for each detector.
///
/// Example:
///     probs = exact_detection_rates(tc, idle_rz=0.05)
///     probs = exact_detection_rates(tc, idle_rz=0.05, p2=0.01)
#[pyfunction]
#[pyo3(signature = (tick_circuit, idle_rz=0.0, p1=0.0, p2=0.0, p_meas=0.0, p_prep=0.0, prune=1e-12, window=None))]
pub fn exact_detection_rates(
    tick_circuit: &Bound<'_, PyAny>,
    idle_rz: f64,
    p1: f64,
    p2: f64,
    p_meas: f64,
    p_prep: f64,
    prune: f64,
    #[allow(unused_variables)] window: Option<usize>,
) -> PyResult<Vec<(usize, f64)>> {
    let noise = pecos_eeg::noise::UniformNoise {
        idle_rz,
        p1,
        p2,
        p_meas,
        p_prep,
    };
    let gates = extract_gates(tick_circuit)?;
    let expanded = pecos_eeg::expand::expand_circuit(&gates);
    let (detectors, _observables) = extract_detectors_expanded(tick_circuit, &expanded)?;

    // Build initial stabilizer group: Z on each original qubit
    let init_gates: Vec<Gate> = (0..expanded.num_original_qubits)
        .map(|q| pecos_eeg::expand::make_gate(pecos_core::gate_type::GateType::PZ, &[q]))
        .collect();
    let stab =
        pecos_eeg::stabilizer::StabilizerGroup::from_circuit(&init_gates, expanded.num_qubits);

    // Build qubit-to-gate index once, shared across all detector walks.
    let gate_index = pecos_eeg::expand::GateIndex::build(&expanded.gates, expanded.num_qubits);

    // Use noise map (with batched S-type) when stochastic noise is present.
    // For coherent-only (idle_rz), the bitmap-enhanced linear scan is faster.
    let has_stochastic = p1 > 0.0 || p2 > 0.0 || p_meas > 0.0 || p_prep > 0.0;

    let noise_map = if has_stochastic {
        Some(pecos_eeg::heisenberg::build_noise_map(
            &expanded.gates,
            &noise,
            &gate_index.expansion_gates,
        ))
    } else {
        None
    };

    // Parallelize across detectors — each walk is independent.
    // Uses sparse traversal (heap + gate index) for O(active_gates) instead of O(all_gates).
    let results: Vec<(usize, f64)> = detectors
        .par_iter()
        .map(|det| {
            let p = pecos_eeg::heisenberg::heisenberg_sparse(
                &expanded.gates,
                &det.stabilizer,
                &noise,
                &stab,
                prune,
                &gate_index,
                noise_map.as_deref(),
            );
            (det.id, p)
        })
        .collect();

    Ok(results)
}

/// Compute exact pairwise detection rates via backward Heisenberg walk.
///
/// For each pair of detectors (i, j), computes P(Di AND Dj both fire)
/// using the identity:
///   P(Di=1, Dj=1) = (P(Di) + P(Dj) - P_walk(Si*Sj)) / 2
/// where P_walk(Si*Sj) is a Heisenberg walk with the product stabilizer.
///
/// Returns a list of ((det_i, det_j), joint_probability) tuples.
#[pyfunction]
#[pyo3(signature = (tick_circuit, idle_rz=0.0, p1=0.0, p2=0.0, p_meas=0.0, p_prep=0.0, prune=1e-12))]
pub fn exact_pairwise_rates(
    tick_circuit: &Bound<'_, PyAny>,
    idle_rz: f64,
    p1: f64,
    p2: f64,
    p_meas: f64,
    p_prep: f64,
    prune: f64,
) -> PyResult<Vec<((usize, usize), f64)>> {
    let noise = pecos_eeg::noise::UniformNoise {
        idle_rz,
        p1,
        p2,
        p_meas,
        p_prep,
    };
    let gates = extract_gates(tick_circuit)?;
    let expanded = pecos_eeg::expand::expand_circuit(&gates);
    let (detectors, _observables) = extract_detectors_expanded(tick_circuit, &expanded)?;

    let init_gates: Vec<Gate> = (0..expanded.num_original_qubits)
        .map(|q| pecos_eeg::expand::make_gate(pecos_core::gate_type::GateType::PZ, &[q]))
        .collect();
    let stab =
        pecos_eeg::stabilizer::StabilizerGroup::from_circuit(&init_gates, expanded.num_qubits);

    let gate_index = pecos_eeg::expand::GateIndex::build(&expanded.gates, expanded.num_qubits);
    let has_stochastic = p1 > 0.0 || p2 > 0.0 || p_meas > 0.0 || p_prep > 0.0;
    let noise_map = if has_stochastic {
        Some(pecos_eeg::heisenberg::build_noise_map(
            &expanded.gates,
            &noise,
            &gate_index.expansion_gates,
        ))
    } else {
        None
    };

    let walk = |stab_bm: &Bm| -> f64 {
        pecos_eeg::heisenberg::heisenberg_sparse(
            &expanded.gates,
            stab_bm,
            &noise,
            &stab,
            prune,
            &gate_index,
            noise_map.as_deref(),
        )
    };

    // Marginals
    let marginals: Vec<f64> = detectors.iter().map(|d| walk(&d.stabilizer)).collect();

    // Pairwise: P(Di AND Dj) = (P(Di) + P(Dj) - P_walk(Si*Sj)) / 2
    let pairs: Vec<(usize, usize)> = (0..detectors.len())
        .flat_map(|i| ((i + 1)..detectors.len()).map(move |j| (i, j)))
        .collect();

    let results: Vec<((usize, usize), f64)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let product = detectors[i].stabilizer.multiply(&detectors[j].stabilizer);
            let p_product = walk(&product);
            let p_joint = (marginals[i] + marginals[j] - p_product) / 2.0;
            ((detectors[i].id, detectors[j].id), p_joint.max(0.0))
        })
        .collect();

    Ok(results)
}

/// Build a coherent DEM with exact Heisenberg marginals.
///
/// Combines backward mechanism extraction (correct structure) with
/// Heisenberg-exact per-detector rates (correct probabilities).
/// Fits mechanism probabilities to match the exact marginals.
///
/// Returns the DEM as a Stim-format string.
#[pyfunction]
#[pyo3(signature = (tick_circuit, idle_rz=0.0, p1=0.0, p2=0.0, p_meas=0.0, p_prep=0.0, prune=1e-12))]
pub fn coherent_dem_exact(
    tick_circuit: &Bound<'_, PyAny>,
    idle_rz: f64,
    p1: f64,
    p2: f64,
    p_meas: f64,
    p_prep: f64,
    prune: f64,
) -> PyResult<String> {
    let noise = pecos_eeg::noise::UniformNoise {
        idle_rz,
        p1,
        p2,
        p_meas,
        p_prep,
    };
    let gates = extract_gates(tick_circuit)?;
    let expanded = pecos_eeg::expand::expand_circuit(&gates);
    let (detectors, observables) = extract_detectors_expanded(tick_circuit, &expanded)?;
    let gate_index = pecos_eeg::expand::GateIndex::build(&expanded.gates, expanded.num_qubits);

    // Compute Heisenberg exact marginals
    let init_gates: Vec<Gate> = (0..expanded.num_original_qubits)
        .map(|q| pecos_eeg::expand::make_gate(pecos_core::gate_type::GateType::PZ, &[q]))
        .collect();
    let stab =
        pecos_eeg::stabilizer::StabilizerGroup::from_circuit(&init_gates, expanded.num_qubits);
    let has_stochastic = p1 > 0.0 || p2 > 0.0 || p_meas > 0.0 || p_prep > 0.0;
    let noise_map = if has_stochastic {
        Some(pecos_eeg::heisenberg::build_noise_map(
            &expanded.gates,
            &noise,
            &gate_index.expansion_gates,
        ))
    } else {
        None
    };

    let walk = |stab_bm: &Bm| -> f64 {
        pecos_eeg::heisenberg::heisenberg_sparse(
            &expanded.gates,
            stab_bm,
            &noise,
            &stab,
            prune,
            &gate_index,
            noise_map.as_deref(),
        )
    };

    let mut marginals = vec![0.0_f64; detectors.iter().map(|d| d.id + 1).max().unwrap_or(0)];
    for det in &detectors {
        let p = walk(&det.stabilizer);
        if det.id < marginals.len() {
            marginals[det.id] = p;
        }
    }

    // Compute pairwise rates via product stabilizer walks
    let mut pairwise: Vec<((usize, usize), f64)> = Vec::new();
    for i in 0..detectors.len() {
        for j in (i + 1)..detectors.len() {
            let product = detectors[i].stabilizer.multiply(&detectors[j].stabilizer);
            let p_product = walk(&product);
            let p_joint =
                (marginals[detectors[i].id] + marginals[detectors[j].id] - p_product) / 2.0;
            if p_joint > 1e-10 {
                pairwise.push(((detectors[i].id, detectors[j].id), p_joint.max(0.0)));
            }
        }
    }

    // Build DEM with exact marginals + pairwise
    let entries = pecos_eeg::coherent_dem::build_coherent_dem_exact(
        &expanded.gates,
        &noise,
        &detectors,
        &observables,
        &gate_index.expansion_gates,
        &marginals,
        Some(&pairwise),
    );

    Ok(pecos_eeg::dem_mapping::format_dem(&entries))
}

/// Build coherent DEM with proper X/Z decomposition for MWPM decoders.
///
/// Returns (raw_dem, decomposed_dem) where the decomposed version uses
/// Pauli provenance to split hyperedges into X ^ Z components.
/// Probabilities are fitted to Heisenberg-exact marginals via L-BFGS.
#[pyfunction]
#[pyo3(signature = (tick_circuit, idle_rz=0.0, p1=0.0, p2=0.0, p_meas=0.0, p_prep=0.0, prune=1e-12))]
pub fn coherent_dem_decomposed(
    tick_circuit: &Bound<'_, PyAny>,
    idle_rz: f64,
    p1: f64,
    p2: f64,
    p_meas: f64,
    p_prep: f64,
    prune: f64,
) -> PyResult<(String, String)> {
    let noise = pecos_eeg::noise::UniformNoise {
        idle_rz,
        p1,
        p2,
        p_meas,
        p_prep,
    };
    let gates = extract_gates(tick_circuit)?;
    let expanded = pecos_eeg::expand::expand_circuit(&gates);
    let (detectors, observables) = extract_detectors_expanded(tick_circuit, &expanded)?;
    let gate_index = pecos_eeg::expand::GateIndex::build(&expanded.gates, expanded.num_qubits);

    // Compute Heisenberg-exact marginals for probability fitting
    let init_gates: Vec<Gate> = (0..expanded.num_original_qubits)
        .map(|q| pecos_eeg::expand::make_gate(pecos_core::gate_type::GateType::PZ, &[q]))
        .collect();
    let stab =
        pecos_eeg::stabilizer::StabilizerGroup::from_circuit(&init_gates, expanded.num_qubits);
    let has_stochastic = p1 > 0.0 || p2 > 0.0 || p_meas > 0.0 || p_prep > 0.0;
    let noise_map = if has_stochastic {
        Some(pecos_eeg::heisenberg::build_noise_map(
            &expanded.gates,
            &noise,
            &gate_index.expansion_gates,
        ))
    } else {
        None
    };

    let walk = |stab_bm: &Bm| -> f64 {
        pecos_eeg::heisenberg::heisenberg_sparse(
            &expanded.gates,
            stab_bm,
            &noise,
            &stab,
            prune,
            &gate_index,
            noise_map.as_deref(),
        )
    };

    let mut marginals = vec![0.0_f64; detectors.iter().map(|d| d.id + 1).max().unwrap_or(0)];
    for det in &detectors {
        let p = walk(&det.stabilizer);
        if det.id < marginals.len() {
            marginals[det.id] = p;
        }
    }

    // Pairwise rates for better fitting
    let mut pairwise: Vec<((usize, usize), f64)> = Vec::new();
    for i in 0..detectors.len() {
        for j in (i + 1)..detectors.len() {
            let product = detectors[i].stabilizer.multiply(&detectors[j].stabilizer);
            let p_product = walk(&product);
            let p_joint =
                (marginals[detectors[i].id] + marginals[detectors[j].id] - p_product) / 2.0;
            if p_joint > 1e-10 {
                pairwise.push(((detectors[i].id, detectors[j].id), p_joint.max(0.0)));
            }
        }
    }

    // Build decomposable entries with exact-fitted probabilities
    let entries = pecos_eeg::coherent_dem::build_coherent_dem_exact_decomposable(
        &expanded.gates,
        &noise,
        &detectors,
        &observables,
        &gate_index.expansion_gates,
        &marginals,
        Some(&pairwise),
    );

    let raw = pecos_eeg::dem_mapping::format_dem(
        &entries
            .iter()
            .map(|e| pecos_eeg::dem_mapping::DemEntry {
                event: e.event.clone(),
                probability: e.probability,
            })
            .collect::<Vec<_>>(),
    );
    let decomposed = pecos_eeg::dem_mapping::format_dem_decomposed(&entries);

    Ok((raw, decomposed))
}

/// Compute exact k-body detector correlation table from Heisenberg walks.
///
/// Returns exact joint detection probabilities for all detector subsets
/// up to `max_order`. No DEM approximation — captures all coherent
/// interference. Useful for decoders that can consume raw correlation data.
///
/// Returns a list of (detector_indices, probability) pairs.
#[pyfunction]
#[pyo3(signature = (tick_circuit, idle_rz=0.0, p1=0.0, p2=0.0, p_meas=0.0, p_prep=0.0, max_order=2, prune=1e-12))]
pub fn exact_correlation_table(
    tick_circuit: &Bound<'_, PyAny>,
    idle_rz: f64,
    p1: f64,
    p2: f64,
    p_meas: f64,
    p_prep: f64,
    max_order: usize,
    prune: f64,
) -> PyResult<Vec<(Vec<String>, f64)>> {
    let noise = pecos_eeg::noise::UniformNoise {
        idle_rz,
        p1,
        p2,
        p_meas,
        p_prep,
    };
    let gates = extract_gates(tick_circuit)?;
    let expanded = pecos_eeg::expand::expand_circuit(&gates);
    let (detectors, observables) = extract_detectors_expanded(tick_circuit, &expanded)?;

    let init_gates: Vec<Gate> = (0..expanded.num_original_qubits)
        .map(|q| pecos_eeg::expand::make_gate(pecos_core::gate_type::GateType::PZ, &[q]))
        .collect();
    let stab =
        pecos_eeg::stabilizer::StabilizerGroup::from_circuit(&init_gates, expanded.num_qubits);

    let table = pecos_eeg::correlation_table::compute_correlation_table(CorrelationTableInput {
        gates: &expanded.gates,
        noise: &noise,
        detectors: &detectors,
        observables: &observables,
        initial_stab: &stab,
        num_qubits: expanded.num_qubits,
        max_order,
        prune_threshold: prune,
    });

    // String labels: "D0", "D1", "L0" — consistent with Stim DEM format.
    let mut result: Vec<(Vec<String>, f64)> = table
        .rates
        .into_iter()
        .map(|(k, v)| (k.into_iter().map(|d| format!("D{d}")).collect(), v))
        .collect();

    // Observable correlations with string labels
    for ((det_ids, obs_id), prob) in table.observable_rates {
        let mut labels: Vec<String> = det_ids.into_iter().map(|d| format!("D{d}")).collect();
        labels.push(format!("L{obs_id}"));
        result.push((labels, prob));
    }

    Ok(result)
}

/// Build a graphlike DEM from exact Heisenberg correlation tables.
///
/// Bypasses the DEM independent error model entirely. Edge weights come
/// directly from exact pairwise correlations (including all coherent
/// interference effects). For MWPM decoders.
///
/// Returns a DEM string suitable for pymatching/fusion_blossom.
#[pyfunction]
#[pyo3(signature = (tick_circuit, idle_rz=0.0, p1=0.0, p2=0.0, p_meas=0.0, p_prep=0.0, max_order=2, prune=1e-12))]
pub fn correlation_matching_dem(
    tick_circuit: &Bound<'_, PyAny>,
    idle_rz: f64,
    p1: f64,
    p2: f64,
    p_meas: f64,
    p_prep: f64,
    max_order: usize,
    prune: f64,
) -> PyResult<String> {
    let noise = pecos_eeg::noise::UniformNoise {
        idle_rz,
        p1,
        p2,
        p_meas,
        p_prep,
    };
    let gates = extract_gates(tick_circuit)?;
    let expanded = pecos_eeg::expand::expand_circuit(&gates);
    let (detectors, observables) = extract_detectors_expanded(tick_circuit, &expanded)?;

    let init_gates: Vec<Gate> = (0..expanded.num_original_qubits)
        .map(|q| pecos_eeg::expand::make_gate(pecos_core::gate_type::GateType::PZ, &[q]))
        .collect();
    let stab =
        pecos_eeg::stabilizer::StabilizerGroup::from_circuit(&init_gates, expanded.num_qubits);

    let table = pecos_eeg::correlation_table::compute_correlation_table(CorrelationTableInput {
        gates: &expanded.gates,
        noise: &noise,
        detectors: &detectors,
        observables: &observables,
        initial_stab: &stab,
        num_qubits: expanded.num_qubits,
        max_order,
        prune_threshold: prune,
    });

    Ok(table.to_matching_dem())
}

/// Compress mid-round noise to round boundaries (optional optimization).
///
/// Propagates gate noise forward to round boundaries, accumulating
/// faults with the same effective Pauli label. Measurement and prep
/// noise kept at original positions. Returns compression statistics.
///
/// For stochastic Pauli noise: exact. For coherent: within-round exact.
///
/// Returns (original_count, compressed_count, boundary_noise_labels).
#[pyfunction]
#[pyo3(signature = (tick_circuit, idle_rz=0.0, p1=0.0, p2=0.0, p_meas=0.0, p_prep=0.0))]
pub fn compress_noise(
    tick_circuit: &Bound<'_, PyAny>,
    idle_rz: f64,
    p1: f64,
    p2: f64,
    p_meas: f64,
    p_prep: f64,
) -> PyResult<(usize, usize)> {
    let noise = pecos_eeg::noise::UniformNoise {
        idle_rz,
        p1,
        p2,
        p_meas,
        p_prep,
    };
    let gates = extract_gates(tick_circuit)?;
    let expanded = pecos_eeg::expand::expand_circuit(&gates);
    let gate_index = pecos_eeg::expand::GateIndex::build(&expanded.gates, expanded.num_qubits);

    let result = pecos_eeg::noise_compression::compress_noise_to_boundaries(
        &expanded.gates,
        &noise,
        &gate_index.expansion_gates,
    );

    Ok((result.original_count, result.compressed_count))
}

/// Complete noise characterization: correlations + mechanisms + DEM.
///
/// Returns a JSON string containing:
/// - Exact k-body detector correlations (from Heisenberg walks)
/// - Detector-observable cross-correlations
/// - Mechanism catalog with fitted probabilities
/// - DEM string for standard decoders
///
/// This is the unified output that captures everything a decoder needs.
#[pyfunction]
#[pyo3(signature = (tick_circuit, idle_rz=0.0, p1=0.0, p2=0.0, p_meas=0.0, p_prep=0.0, max_order=2, prune=1e-12, compress=false))]
pub fn noise_characterization(
    tick_circuit: &Bound<'_, PyAny>,
    idle_rz: f64,
    p1: f64,
    p2: f64,
    p_meas: f64,
    p_prep: f64,
    max_order: usize,
    prune: f64,
    compress: bool,
) -> PyResult<(String, String, String)> {
    let base_noise = pecos_eeg::noise::UniformNoise {
        idle_rz,
        p1,
        p2,
        p_meas,
        p_prep,
    };
    let gates = extract_gates(tick_circuit)?;
    let expanded = pecos_eeg::expand::expand_circuit(&gates);
    let (detectors, observables) = extract_detectors_expanded(tick_circuit, &expanded)?;

    let init_gates: Vec<Gate> = (0..expanded.num_original_qubits)
        .map(|q| pecos_eeg::expand::make_gate(pecos_core::gate_type::GateType::PZ, &[q]))
        .collect();
    let stab =
        pecos_eeg::stabilizer::StabilizerGroup::from_circuit(&init_gates, expanded.num_qubits);

    // For compressed mode: use original noise for Heisenberg targets (exact),
    // compressed noise for mechanism structure (fast).
    let structure_noise: Option<Box<dyn pecos_eeg::noise::NoiseSpec>> = if compress {
        let gate_index = pecos_eeg::expand::GateIndex::build(&expanded.gates, expanded.num_qubits);
        let compressed = pecos_eeg::noise_compression::compress_noise_to_boundaries(
            &expanded.gates,
            &base_noise,
            &gate_index.expansion_gates,
        );
        Some(Box::new(
            pecos_eeg::noise_compression::CompressedNoiseSpec::from_compressed(&compressed),
        ))
    } else {
        None
    };

    let det_meas_ids = extract_meas_id_defs(tick_circuit, "detectors")?;
    let obs_meas_ids = extract_meas_id_defs(tick_circuit, "observables")?;

    let nc = pecos_eeg::noise_characterization::NoiseCharacterization::build(
        NoiseCharacterizationInput {
            gates: &expanded.gates,
            noise: &base_noise,
            structure_noise: structure_noise.as_deref(),
            detectors: &detectors,
            observables: &observables,
            initial_stab: &stab,
            num_qubits: expanded.num_qubits,
            max_order,
            prune_threshold: prune,
            detector_meas_ids: &det_meas_ids,
            observable_meas_ids: &obs_meas_ids,
        },
    );

    Ok((
        nc.to_json(),
        nc.to_dem_string(),
        nc.to_dem_string_decomposed(),
    ))
}

// -- Internal --

fn parse_h_formula(s: &str) -> PyResult<pecos_eeg::dem_mapping::HFormula> {
    match s {
        "taylor" => Ok(pecos_eeg::dem_mapping::HFormula::Taylor),
        "sin_squared" => Ok(pecos_eeg::dem_mapping::HFormula::SinSquared),
        "exact_commuting" => Ok(pecos_eeg::dem_mapping::HFormula::ExactCommuting),
        "exact_subset" => Ok(pecos_eeg::dem_mapping::HFormula::ExactSubset),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown h_formula '{s}'. Use 'taylor', 'sin_squared', 'exact_commuting', or 'exact_subset'."
        ))),
    }
}

fn run_eeg(
    py_tc: &Bound<'_, PyAny>,
    idle_rz: f64,
    p1: f64,
    p2: f64,
    p_meas: f64,
    p_prep: f64,
    h_formula: &str,
    bch_order: u32,
) -> PyResult<Vec<DemEntry>> {
    let noise = NoiseModel {
        idle_rz,
        p1,
        p2,
        p_meas,
        p_prep,
    };
    let gates = extract_gates(py_tc)?;

    // Step 1: Expand circuit (defer measurements)
    let expanded = pecos_eeg::expand::expand_circuit(&gates);

    // Step 2: Propagate through expanded circuit
    let result = pecos_eeg::circuit::analyze_expanded(&expanded.gates, &noise);

    // Step 3: Build detectors using expanded circuit mapping
    let (detectors, observables) = extract_detectors_expanded(py_tc, &expanded)?;

    // Step 4: Compute stabilizer group from EXPANDED circuit (pre-readout).
    // Use expanded frame directly — no lossy original-frame mapping.
    // Strip trailing deferred MZ(aux) from the expanded circuit.
    let expanded_pre_readout = exclude_final_mz(&expanded.gates);
    let stab_group = pecos_eeg::stabilizer::StabilizerGroup::from_circuit(
        &expanded_pre_readout,
        expanded.num_qubits,
    );

    // Step 5: Build DEM (stabilizer check in expanded frame)
    let config = pecos_eeg::dem_mapping::EegConfig {
        h_formula: parse_h_formula(h_formula)?,
        bch_order: if bch_order >= 2 {
            pecos_eeg::dem_mapping::BchOrder::Second
        } else {
            pecos_eeg::dem_mapping::BchOrder::First
        },
    };
    Ok(pecos_eeg::dem_mapping::build_dem_configured(
        &result.generators,
        &detectors,
        &observables,
        Some(&stab_group),
        &config,
    ))
}

fn run_eeg_decomposable(
    py_tc: &Bound<'_, PyAny>,
    idle_rz: f64,
    p1: f64,
    p2: f64,
    p_meas: f64,
    p_prep: f64,
    h_formula: &str,
    bch_order: u32,
) -> PyResult<(
    Vec<DemEntry>,
    Vec<pecos_eeg::dem_mapping::DecomposableDemEntry>,
)> {
    let noise = NoiseModel {
        idle_rz,
        p1,
        p2,
        p_meas,
        p_prep,
    };
    let gates = extract_gates(py_tc)?;
    let expanded = pecos_eeg::expand::expand_circuit(&gates);
    let result = pecos_eeg::circuit::analyze_expanded(&expanded.gates, &noise);
    let (detectors, observables) = extract_detectors_expanded(py_tc, &expanded)?;
    let expanded_pre_readout = exclude_final_mz(&expanded.gates);
    let stab_group = pecos_eeg::stabilizer::StabilizerGroup::from_circuit(
        &expanded_pre_readout,
        expanded.num_qubits,
    );
    let config = pecos_eeg::dem_mapping::EegConfig {
        h_formula: parse_h_formula(h_formula)?,
        bch_order: if bch_order >= 2 {
            pecos_eeg::dem_mapping::BchOrder::Second
        } else {
            pecos_eeg::dem_mapping::BchOrder::First
        },
    };
    let raw = pecos_eeg::dem_mapping::build_dem_configured(
        &result.generators,
        &detectors,
        &observables,
        Some(&stab_group),
        &config,
    );
    let decomposable = pecos_eeg::dem_mapping::build_dem_decomposable(
        &result.generators,
        &detectors,
        &observables,
        Some(&stab_group),
        &config,
    );
    Ok((raw, decomposable))
}

/// Strip all trailing MZ from expanded circuit (deferred measurements).
fn exclude_final_mz(gates: &[Gate]) -> Vec<Gate> {
    let last_non_mz = gates
        .iter()
        .rposition(|g| g.gate_type != pecos_core::gate_type::GateType::MZ);
    match last_non_mz {
        Some(idx) => gates[..=idx].to_vec(),
        None => Vec::new(),
    }
}

fn extract_gates(py_tc: &Bound<'_, PyAny>) -> PyResult<Vec<Gate>> {
    let num_ticks: usize = py_tc.call_method0("num_ticks")?.extract()?;
    let mut gates = Vec::new();

    for tick_idx in 0..num_ticks {
        let py_tick = py_tc.call_method1("get_tick", (tick_idx,))?;
        let py_gates = py_tick.call_method0("gate_batches")?;
        let gate_list: Vec<Bound<'_, PyAny>> = py_gates.extract()?;

        // Collect all gates in this tick first, then emit them.
        // This preserves the simultaneity of gates within a tick:
        // all Clifford gates in the tick execute "at once", and noise
        // is injected after the entire tick (not between gates).
        let mut tick_gates = Vec::new();

        for gate in &gate_list {
            let name: String = gate.getattr("gate_type")?.getattr("name")?.extract()?;
            let qubits: Vec<usize> = gate.getattr("qubits")?.extract()?;

            match name.as_str() {
                "CX" | "CY" | "CZ" | "SWAP" | "SZZ" | "SZZdg" | "SXX" | "SXXdg" | "SYY"
                | "SYYdg" => {
                    // Split multi-pair 2q gates into individual pairs
                    let gt = match name.as_str() {
                        "CX" => pecos_core::gate_type::GateType::CX,
                        "CY" => pecos_core::gate_type::GateType::CY,
                        "CZ" => pecos_core::gate_type::GateType::CZ,
                        "SWAP" => pecos_core::gate_type::GateType::SWAP,
                        "SZZ" => pecos_core::gate_type::GateType::SZZ,
                        "SZZdg" => pecos_core::gate_type::GateType::SZZdg,
                        "SXX" => pecos_core::gate_type::GateType::SXX,
                        "SXXdg" => pecos_core::gate_type::GateType::SXXdg,
                        "SYY" => pecos_core::gate_type::GateType::SYY,
                        _ => pecos_core::gate_type::GateType::SYYdg,
                    };
                    for pair in qubits.chunks(2) {
                        if pair.len() == 2 {
                            tick_gates.push(Gate {
                                gate_type: gt,
                                qubits: pair.iter().map(|&q| QubitId(q)).collect(),
                                angles: GateAngles::new(),
                                params: GateParams::new(),
                                meas_ids: GateMeasIds::new(),
                                channel: None,
                            });
                        }
                    }
                }
                // Single-qubit gates: split multi-qubit into individual per-qubit gates
                "H" | "X" | "Y" | "Z" | "SZ" | "SZdg" | "SX" | "SXdg" | "SY" | "SYdg" | "F"
                | "Fdg" => {
                    let gt = match name.as_str() {
                        "H" => pecos_core::gate_type::GateType::H,
                        "X" => pecos_core::gate_type::GateType::X,
                        "Y" => pecos_core::gate_type::GateType::Y,
                        "Z" => pecos_core::gate_type::GateType::Z,
                        "SZ" => pecos_core::gate_type::GateType::SZ,
                        "SZdg" => pecos_core::gate_type::GateType::SZdg,
                        "SX" => pecos_core::gate_type::GateType::SX,
                        "SXdg" => pecos_core::gate_type::GateType::SXdg,
                        "SY" => pecos_core::gate_type::GateType::SY,
                        "F" => pecos_core::gate_type::GateType::F,
                        "Fdg" => pecos_core::gate_type::GateType::Fdg,
                        _ => pecos_core::gate_type::GateType::SYdg,
                    };
                    for &q in &qubits {
                        tick_gates.push(Gate {
                            gate_type: gt,
                            qubits: std::iter::once(QubitId(q)).collect(),
                            angles: GateAngles::new(),
                            params: GateParams::new(),
                            meas_ids: GateMeasIds::new(),
                            channel: None,
                        });
                    }
                }
                // PZ/QAlloc: split multi-qubit into per-qubit
                "QAlloc" | "PZ" => {
                    for &q in &qubits {
                        tick_gates.push(Gate {
                            gate_type: pecos_core::gate_type::GateType::PZ,
                            qubits: std::iter::once(QubitId(q)).collect(),
                            angles: GateAngles::new(),
                            params: GateParams::new(),
                            meas_ids: GateMeasIds::new(),
                            channel: None,
                        });
                    }
                }
                // MZ: keep multi-qubit (expansion handles per-qubit)
                _ => {
                    let gt = match name.as_str() {
                        "MZ" | "MeasureFree" => pecos_core::gate_type::GateType::MZ,
                        "RZ" => pecos_core::gate_type::GateType::RZ,
                        "Idle" | "I" => pecos_core::gate_type::GateType::Idle,
                        other => {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "EEG extract_gates: unsupported gate type {other:?}"
                            )));
                        }
                    };
                    let mut g = Gate {
                        gate_type: gt,
                        qubits: qubits.iter().map(|&q| QubitId(q)).collect(),
                        angles: GateAngles::new(),
                        params: GateParams::new(),
                        meas_ids: GateMeasIds::new(),
                        channel: None,
                    };
                    if gt == pecos_core::gate_type::GateType::RZ
                        && let Ok(angles) = gate.getattr("angles")?.extract::<Vec<f64>>()
                        && let Some(&a) = angles.first()
                    {
                        g.angles.push(Angle64::from_radians(a));
                    }
                    tick_gates.push(g);
                }
            }
        }

        // Emit all gates from this tick. Gates within a tick are simultaneous,
        // so noise injected after the last gate correctly sees all tick gates
        // as already applied.
        gates.extend(tick_gates);
    }

    Ok(gates)
}

fn measurement_record_index(record: i32, num_measurements: usize) -> Option<usize> {
    let idx = if record < 0 {
        i32::try_from(num_measurements).ok()?.checked_add(record)?
    } else {
        record
    };
    usize::try_from(idx)
        .ok()
        .filter(|&idx| idx < num_measurements)
}

fn extract_detectors_expanded(
    py_tc: &Bound<'_, PyAny>,
    expanded: &pecos_eeg::expand::ExpandedCircuit,
) -> PyResult<(Vec<Detector>, Vec<Observable>)> {
    // In the expanded circuit, each measurement record k maps to a
    // Z-measurement on auxiliary qubit expanded.measurement_qubit[k].
    //
    // A detector defined by records {r1, r2, ...} has stabilizer
    // Z_{aux_r1} * Z_{aux_r2} * ... in the expanded circuit.
    let num_meas = expanded.measurement_qubit.len();

    let mut detectors = Vec::new();
    let mut observables = Vec::new();

    // Parse detector JSON from metadata
    if let Ok(det_json_str) = py_tc.call_method1("get_meta", ("detectors",))
        && let Ok(det_json) = det_json_str.extract::<String>()
        && let Ok(det_list) = serde_json_parse_detectors(&det_json)
    {
        for (id, records) in det_list {
            let mut bm = Bm::default();
            for &rec in &records {
                if let Some(abs_idx) = measurement_record_index(rec, num_meas) {
                    // Map to AUXILIARY qubit in expanded circuit
                    let aux_qubit = expanded.measurement_qubit[abs_idx];
                    bm.z_bits.xor_bit(aux_qubit);
                }
            }
            detectors.push(Detector { id, stabilizer: bm });
        }
    }

    // Parse observable JSON from metadata
    if let Ok(obs_json_str) = py_tc.call_method1("get_meta", ("observables",))
        && let Ok(obs_json) = obs_json_str.extract::<String>()
        && let Ok(obs_list) = serde_json_parseobservables(&obs_json)
    {
        for (id, records) in obs_list {
            let mut bm = Bm::default();
            for &rec in &records {
                if let Some(abs_idx) = measurement_record_index(rec, num_meas) {
                    let aux_qubit = expanded.measurement_qubit[abs_idx];
                    bm.z_bits.xor_bit(aux_qubit);
                }
            }
            observables.push(Observable { id, pauli: bm });
        }
    }

    Ok((detectors, observables))
}

/// Minimal JSON parser for detector definitions (avoids serde dependency).
/// Parses [{"id": N, "records": [R1, R2, ...], ...}, ...]
fn serde_json_parse_detectors(json: &str) -> Result<Vec<(usize, Vec<i32>)>, String> {
    // Simple approach: find "id" and "records" fields via string scanning
    let mut result = Vec::new();
    let mut pos = 0;
    while let Some(start) = json[pos..].find('{') {
        let start = pos + start;
        let end = json[start..]
            .find('}')
            .map(|e| start + e + 1)
            .ok_or_else(|| "Unmatched brace".to_string())?;
        let entry = &json[start..end];

        let id = extract_json_int(entry, "\"id\"")
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(result.len());
        let records = extract_json_int_array(entry, "\"records\"").unwrap_or_default();

        result.push((id, records));
        pos = end;
    }
    Ok(result)
}

fn serde_json_parseobservables(json: &str) -> Result<Vec<(usize, Vec<i32>)>, String> {
    serde_json_parse_detectors(json) // Same format
}

/// Extract MeasId definitions from circuit metadata JSON.
fn extract_meas_id_defs(
    py_tc: &Bound<'_, pyo3::PyAny>,
    key: &str, // "detectors" or "observables"
) -> PyResult<Vec<MeasurementRecordDefinition>> {
    let mut result = Vec::new();
    if let Ok(json_str) = py_tc.call_method1("get_meta", (key,))
        && let Ok(s) = json_str.extract::<String>()
    {
        // Parse JSON: each item has id, meas_ids (optional), records
        let items = parse_json_items(&s);
        for (idx, (records, meas_ids)) in items.iter().enumerate() {
            result.push((idx, meas_ids.clone(), records.clone()));
        }
    }
    Ok(result)
}

/// Parse JSON array items, extracting records and meas_ids fields.
fn parse_json_items(json: &str) -> Vec<(Vec<i32>, Vec<usize>)> {
    let mut result = Vec::new();
    // Split by "records" occurrences
    let trimmed = json.trim();
    if !trimmed.starts_with('[') {
        return result;
    }

    // Simple state machine: find each {...} block and extract fields
    let mut depth = 0;
    let mut block_start = None;
    for (i, ch) in trimmed.char_indices() {
        match ch {
            '{' => {
                if depth == 1 {
                    block_start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 1
                    && let Some(start) = block_start
                {
                    let block = &trimmed[start..=i];
                    let records = extract_json_int_array(block, "records").unwrap_or_default();
                    let meas_ids = extract_json_int_array(block, "meas_ids")
                        .map(|v| {
                            v.into_iter()
                                .filter_map(|x| usize::try_from(x).ok())
                                .collect()
                        })
                        .unwrap_or_default();
                    result.push((records, meas_ids));
                }
            }
            '[' if depth == 0 => {
                depth = 1;
            }
            ']' if depth == 1 => {
                break;
            }
            _ => {}
        }
    }
    result
}

fn extract_json_int(s: &str, key: &str) -> Option<i64> {
    let key_pos = s.find(key)?;
    let after_key = &s[key_pos + key.len()..];
    let colon = after_key.find(':')?;
    let value_str = after_key[colon + 1..].trim();
    // Read digits (possibly with minus)
    let end = value_str
        .find(|c: char| !c.is_ascii_digit() && c != '-')
        .unwrap_or(value_str.len());
    value_str[..end].trim().parse().ok()
}

fn extract_json_int_array(s: &str, key: &str) -> Option<Vec<i32>> {
    let key_pos = s.find(key)?;
    let after_key = &s[key_pos + key.len()..];
    let bracket_start = after_key.find('[')?;
    let bracket_end = after_key[bracket_start..].find(']')? + bracket_start;
    let array_str = &after_key[bracket_start + 1..bracket_end];
    let values: Vec<i32> = array_str
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect();
    Some(values)
}
