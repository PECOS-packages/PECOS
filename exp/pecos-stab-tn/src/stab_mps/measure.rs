// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Shared measurement logic for STN and MAST simulators.
//!
//! Measures a qubit in the Z basis using the stabilizer tableau for structure
//! and the MPS for probability computation and projection.
//!
//! The measurement protocol decomposes `Z_q` in the stabilizer basis, computes
//! the expectation value from the MPS, samples an outcome, and projects the
//! MPS using the (I + sign * `Z_q)/2` projector. After projection, the measured
//! site collapses to sigma=0 (the stabilizer eigenstate).
//!
//! Reference: Masot-Llima, Garcia-Saez. arXiv:2403.08724, Section III.

use super::pauli_decomp::{ZDecomposition, decompose_z};
use crate::errors::MpsError;
use crate::mps::{BondCompressionTelemetry, Mps, MpsPreReductionTiming};
use nalgebra::DMatrix;
use num_complex::Complex64;
use pecos_core::BitSet;
use pecos_random::PecosRng;
use pecos_simulators::{CliffordGateable, MeasurementResult, SparseStabY};
use std::time::Instant;

fn profile_query_phase<T>(
    mps: &mut Mps,
    telemetry: &mut Option<&mut super::QueryDepthTelemetry>,
    phase: super::QueryPhase,
    operation: impl FnOnce(&mut Mps) -> T,
) -> T {
    profile_query_phase_with_elapsed(mps, telemetry, phase, operation).0
}

fn profile_query_phase_with_elapsed<T>(
    mps: &mut Mps,
    telemetry: &mut Option<&mut super::QueryDepthTelemetry>,
    phase: super::QueryPhase,
    operation: impl FnOnce(&mut Mps) -> T,
) -> (T, f64) {
    let Some(telemetry) = telemetry.as_deref_mut() else {
        return (operation(mps), 0.0);
    };
    telemetry.begin_phase();
    mps.reset_phase_svd_operations();
    let started = Instant::now();
    let result = operation(mps);
    let elapsed = started.elapsed().as_secs_f64();
    let (svd_operations, capped_svd_operations) = mps.take_phase_svd_operations();
    telemetry.record(phase, svd_operations, capped_svd_operations, elapsed);
    (result, elapsed)
}

fn mix_pre_reduction_fingerprint(hash: &mut u64, value: usize) {
    *hash ^= value as u64;
    *hash = hash.wrapping_mul(0x100_0000_01b3);
}

fn begin_pre_reduction_diagnostics(
    mps: &Mps,
    accumulated_projector_count: usize,
    sibling_pair_id: Option<u64>,
) -> super::PreReductionTelemetry {
    let started = Instant::now();
    let input_bond_profile = mps
        .bond_dims()
        .iter()
        .copied()
        .skip(1)
        .take(mps.num_sites().saturating_sub(1))
        .collect::<Vec<_>>();
    let bond_cap = mps.config().max_bond_dim;
    let input_max_bond = input_bond_profile.iter().copied().max().unwrap_or(1);
    let input_bond_sum = input_bond_profile.iter().sum();
    let input_cap_saturated_bonds = input_bond_profile
        .iter()
        .filter(|&&bond| bond == bond_cap)
        .count();
    let mut input_fingerprint = 0xcbf2_9ce4_8422_2325;
    mix_pre_reduction_fingerprint(&mut input_fingerprint, mps.num_sites());
    mix_pre_reduction_fingerprint(&mut input_fingerprint, accumulated_projector_count);
    mix_pre_reduction_fingerprint(&mut input_fingerprint, input_bond_profile.len());
    for &bond in &input_bond_profile {
        mix_pre_reduction_fingerprint(&mut input_fingerprint, bond);
    }
    let input_profile_scan_seconds = started.elapsed().as_secs_f64();
    super::PreReductionTelemetry {
        chain_length: mps.num_sites(),
        accumulated_projector_count,
        sibling_pair_id,
        measured_qubit: accumulated_projector_count,
        bond_cap,
        input_bond_profile,
        output_bond_profile: Vec::new(),
        input_max_bond,
        input_bond_sum,
        input_cap_saturated_bonds,
        input_fingerprint,
        input_profile_scan_seconds,
        output_profile_scan_seconds: 0.0,
        chosen_stabilizer: None,
        chosen_stabilizer_weight: None,
        replacement_candidate_count: 0,
        minimum_weight_candidate_count: 0,
        chosen_compensation_cost: 0,
        minimum_weight_optimal_cost: 0,
        weight_plus_one_candidate_count: 0,
        weight_plus_one_optimal_cost: None,
        target_min: None,
        target_max: None,
        target_span: 0,
        compensation_support_span: 0,
        wall_time_seconds: 0.0,
        svd_compute_seconds: 0.0,
        qr_gauge_seconds: 0.0,
        tensor_contraction_seconds: 0.0,
        bookkeeping_seconds: 0.0,
        svd_operations: 0,
        capped_svd_operations: 0,
        compensating_cnot_count: 0,
        structural_identity_cnot_count: 0,
        unconditional_x_cnot_count: 0,
        svd_steps: Vec::new(),
        cnot_steps: Vec::new(),
        output_profile_unchanged: false,
    }
}

fn profile_pre_reduction<T>(
    mps: &mut Mps,
    telemetry: &mut Option<&mut super::QueryDepthTelemetry>,
    accumulated_projector_count: usize,
    sibling_pair_id: Option<u64>,
    operation: impl FnOnce(&mut Mps, Option<&mut super::PreReductionTelemetry>) -> T,
) -> T {
    let diagnostics_active = telemetry
        .as_deref()
        .is_some_and(super::QueryDepthTelemetry::projection_locality_active);
    if !diagnostics_active {
        return profile_query_phase(mps, telemetry, super::QueryPhase::PreReduction, |mps| {
            operation(mps, None)
        });
    }

    let mut event =
        begin_pre_reduction_diagnostics(mps, accumulated_projector_count, sibling_pair_id);
    let telemetry = telemetry
        .as_deref_mut()
        .expect("active pre-reduction diagnostics require query telemetry");
    telemetry.begin_phase();
    mps.reset_phase_svd_operations();
    let started = Instant::now();
    let result = operation(mps, Some(&mut event));
    event.wall_time_seconds = started.elapsed().as_secs_f64();
    let (svd_operations, capped_svd_operations) = mps.take_phase_svd_operations();
    event.svd_operations = svd_operations;
    event.capped_svd_operations = capped_svd_operations;
    telemetry.record(
        super::QueryPhase::PreReduction,
        svd_operations,
        capped_svd_operations,
        event.wall_time_seconds,
    );

    let output_started = Instant::now();
    event.output_bond_profile = mps
        .bond_dims()
        .iter()
        .copied()
        .skip(1)
        .take(mps.num_sites().saturating_sub(1))
        .collect();
    event.output_profile_unchanged = event.output_bond_profile == event.input_bond_profile;
    event.output_profile_scan_seconds = output_started.elapsed().as_secs_f64();
    let attributed =
        event.svd_compute_seconds + event.qr_gauge_seconds + event.tensor_contraction_seconds;
    assert!(
        attributed <= event.wall_time_seconds,
        "nested pre-reduction timers exceeded the owning phase"
    );
    event.bookkeeping_seconds = event.wall_time_seconds - attributed;
    telemetry.record_pre_reduction_diagnostics(event);
    result
}

#[derive(Default)]
struct PreReductionWork {
    mps: MpsPreReductionTiming,
    compensating_cnot_count: u64,
    structural_identity_cnot_count: u64,
    unconditional_x_cnot_count: u64,
    chosen_stabilizer: Option<usize>,
    chosen_stabilizer_weight: Option<usize>,
    replacement_candidate_count: usize,
    minimum_weight_candidate_count: usize,
    chosen_compensation_cost: usize,
    minimum_weight_optimal_cost: usize,
    weight_plus_one_candidate_count: usize,
    weight_plus_one_optimal_cost: Option<usize>,
    target_min: Option<usize>,
    target_max: Option<usize>,
    target_span: usize,
    compensation_support_span: usize,
    cnot_steps: Vec<super::PreReductionCnotTelemetry>,
}

fn finish_pre_reduction_work(event: &mut super::PreReductionTelemetry, work: &PreReductionWork) {
    event.svd_compute_seconds += work.mps.svd_compute;
    event.qr_gauge_seconds += work.mps.qr_gauge;
    event.tensor_contraction_seconds += work.mps.tensor_contraction;
    event.compensating_cnot_count += work.compensating_cnot_count;
    event.structural_identity_cnot_count += work.structural_identity_cnot_count;
    event.unconditional_x_cnot_count += work.unconditional_x_cnot_count;
    event.chosen_stabilizer = work.chosen_stabilizer;
    event.chosen_stabilizer_weight = work.chosen_stabilizer_weight;
    event.replacement_candidate_count = work.replacement_candidate_count;
    event.minimum_weight_candidate_count = work.minimum_weight_candidate_count;
    event.chosen_compensation_cost = work.chosen_compensation_cost;
    event.minimum_weight_optimal_cost = work.minimum_weight_optimal_cost;
    event.weight_plus_one_candidate_count = work.weight_plus_one_candidate_count;
    event.weight_plus_one_optimal_cost = work.weight_plus_one_optimal_cost;
    event.target_min = work.target_min;
    event.target_max = work.target_max;
    event.target_span = work.target_span;
    event.compensation_support_span = work.compensation_support_span;
    event.svd_steps.clone_from(&work.mps.svd_steps);
    event.cnot_steps.clone_from(&work.cnot_steps);
}

fn time_pre_reduction_tensor_work<const DIAGNOSTICS: bool, T>(
    timing: &mut MpsPreReductionTiming,
    operation: impl FnOnce() -> T,
) -> T {
    if DIAGNOSTICS {
        let started = Instant::now();
        let result = operation();
        timing.tensor_contraction += started.elapsed().as_secs_f64();
        result
    } else {
        operation()
    }
}

#[cfg(test)]
std::thread_local! {
    static INJECTED_PROJECTION_VANISHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(all(test, not(debug_assertions)))]
std::thread_local! {
    static INJECTED_ZERO_PROBABILITIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(super) fn inject_projection_vanishes(count: usize) {
    INJECTED_PROJECTION_VANISHES.with(|remaining| remaining.set(count));
}

#[cfg(all(test, not(debug_assertions)))]
pub(super) fn inject_zero_projection_probabilities(count: usize) {
    // Model a branch whose amplitude was erased by earlier configured
    // truncation: projection itself leaves a healthy normalized candidate,
    // but reports an exactly zero Born probability to the MAST owner.
    INJECTED_ZERO_PROBABILITIES.with(|remaining| remaining.set(count));
}

#[cfg(test)]
fn inject_projection_vanish_if_requested(mps: &mut Mps) {
    INJECTED_PROJECTION_VANISHES.with(|remaining| {
        let count = remaining.get();
        if count > 0 {
            remaining.set(count - 1);
            mps.scale(Complex64::new(0.0, 0.0));
        }
    });
}

#[cfg(all(test, not(debug_assertions)))]
fn inject_zero_projection_probability_if_requested(probability: f64) -> f64 {
    INJECTED_ZERO_PROBABILITIES.with(|remaining| {
        let count = remaining.get();
        if count > 0 {
            remaining.set(count - 1);
            0.0
        } else {
            probability
        }
    })
}

#[cfg(not(test))]
fn inject_projection_vanish_if_requested(_mps: &mut Mps) {}

#[cfg(any(not(test), debug_assertions))]
fn inject_zero_projection_probability_if_requested(probability: f64) -> f64 {
    probability
}

/// Virtual coefficient-MPS sites whose disentangling proofs must be rechecked
/// after a physical measurement projection.
#[derive(Debug, Default)]
pub(super) struct ProjectionUpdate {
    /// Site intended to be placed in `|0>` by the measurement basis rotation.
    pub(super) collapsed_site: Option<usize>,
    /// Other sites touched by projection or compensating virtual gates.
    pub(super) modified_sites: Vec<usize>,
}

/// Probability and virtual-site metadata returned by a forced projection.
pub(super) struct ForcedProjectionResult {
    pub(super) snapped_probability: f64,
    pub(super) survival_ratio: f64,
    pub(super) update: ProjectionUpdate,
}

/// Sampled measurement and virtual-site metadata for a live simulator update.
pub(super) struct LiveMeasurementResult {
    pub(super) measurement: MeasurementResult,
    pub(super) update: ProjectionUpdate,
}

/// Block-norm tolerance for the coefficient-MPS computational-basis fast path.
///
/// A bond-one site with either physical block below this threshold is treated
/// as a basis state. Its physical measurement probability is consequently
/// quantized to the stabilizer values `{0, 1/2, 1}` before both sampling and
/// projection; this tolerance is intentionally distinct from the Pauli
/// endpoint snap below.
pub(super) const TRIVIAL_MPS_BLOCK_NORM_TOLERANCE: f64 = 1e-12;

/// Check if the MPS is trivial (all sites in a computational basis state).
fn is_mps_trivial(mps: &Mps) -> bool {
    mps.max_bond_dim() == 1
        && mps.tensors().iter().all(|t| {
            let chi_r = t.ncols() / 2;
            let b0_norm: f64 = (0..t.nrows())
                .flat_map(|i| (0..chi_r).map(move |j| t[(i, j)].norm_sqr()))
                .sum();
            let b1_norm: f64 = (0..t.nrows())
                .flat_map(|i| (0..chi_r).map(move |j| t[(i, chi_r + j)].norm_sqr()))
                .sum();
            b0_norm < TRIVIAL_MPS_BLOCK_NORM_TOLERANCE || b1_norm < TRIVIAL_MPS_BLOCK_NORM_TOLERANCE
        })
}

/// Put a computational-basis product coefficient MPS into `|0...0>` while
/// preserving the represented physical state by absorbing its basis Xs into
/// the Clifford tableau.
///
/// The tableau-only measurement fast path is valid for `C|0...0>`, not for a
/// general `C|b>`. Non-Clifford evolution and earlier exact projections can
/// produce the latter even though every MPS bond is one.
fn canonicalize_trivial_mps_basis(
    tableau: &mut SparseStabY,
    mps: &mut Mps,
    phase_accumulator: Option<&mut crate::stab_mps::canonical_ket::CanonicalPhaseTracker>,
) -> Vec<usize> {
    canonicalize_trivial_mps_basis_impl::<false>(
        tableau,
        mps,
        phase_accumulator,
        &mut PreReductionWork::default(),
    )
}

fn canonicalize_trivial_mps_basis_profiled(
    tableau: &mut SparseStabY,
    mps: &mut Mps,
    phase_accumulator: Option<&mut crate::stab_mps::canonical_ket::CanonicalPhaseTracker>,
    event: &mut super::PreReductionTelemetry,
) -> Vec<usize> {
    let mut work = PreReductionWork::default();
    let modified_sites =
        canonicalize_trivial_mps_basis_impl::<true>(tableau, mps, phase_accumulator, &mut work);
    finish_pre_reduction_work(event, &work);
    modified_sites
}

fn canonicalize_trivial_mps_basis_impl<const DIAGNOSTICS: bool>(
    tableau: &mut SparseStabY,
    mps: &mut Mps,
    mut phase_accumulator: Option<&mut crate::stab_mps::canonical_ket::CanonicalPhaseTracker>,
    work: &mut PreReductionWork,
) -> Vec<usize> {
    let norm_squared =
        time_pre_reduction_tensor_work::<DIAGNOSTICS, _>(&mut work.mps, || mps.norm_squared());
    debug_assert!(
        (norm_squared - 1.0).abs() < 1e-8,
        "trivial-basis canonicalization requires a normalized MPS, got norm²={norm_squared}"
    );
    let x_gate = time_pre_reduction_tensor_work::<DIAGNOSTICS, _>(&mut work.mps, || {
        DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
    });
    let mut modified_sites = Vec::new();
    for site in 0..mps.num_sites() {
        let chi_r = mps.bond_dim(site + 1);
        let block_0_norm = time_pre_reduction_tensor_work::<DIAGNOSTICS, _>(&mut work.mps, || {
            let block_0 = crate::mps::tensor::phys_block(&mps.tensors()[site], 0, chi_r);
            block_0
                .iter()
                .map(num_complex::Complex::norm_sqr)
                .sum::<f64>()
        });
        // Every project_forced_z entry path normalizes before this helper can
        // be reached again, so the block weight has unit-state scale and this
        // absolute zero threshold is intentional.
        if block_0_norm < TRIVIAL_MPS_BLOCK_NORM_TOLERANCE {
            // C|...1...> = (C X_site)(X_site|...1...>).
            let before = phase_accumulator.as_ref().map(|_| tableau.clone());
            crate::stab_mps::tableau_compose::right_compose_x(tableau, site);
            if let (Some(accumulator), Some(before)) =
                (phase_accumulator.as_deref_mut(), before.as_ref())
            {
                accumulator.right_compose_x(before, tableau, site);
            }
            time_pre_reduction_tensor_work::<DIAGNOSTICS, _>(&mut work.mps, || {
                mps.apply_one_site_gate(site, &x_gate)
                    .expect("MPS op on valid site");
            });
            modified_sites.push(site);
        }
    }
    modified_sites
}

/// Measure a computational-basis product coefficient state through the tableau.
///
/// The tableau-only read is valid only after the coefficient basis word has
/// been absorbed into the tableau, leaving the stored MPS in `|0...0>`.
fn measure_trivial_mps_with_update(
    tableau: &mut SparseStabY,
    mps: &mut Mps,
    q_idx: usize,
) -> LiveMeasurementResult {
    debug_assert!(is_mps_trivial(mps));
    let modified_sites = canonicalize_trivial_mps_basis(tableau, mps, None);
    let measurement = tableau
        .mz(&[pecos_core::QubitId(q_idx)])
        .into_iter()
        .next()
        .expect("MPS op on valid site");
    LiveMeasurementResult {
        measurement,
        update: ProjectionUpdate {
            collapsed_site: None,
            modified_sites,
        },
    }
}

/// Compute `<mps| phase · X_flip · Z_sign |mps>` via clone + inner product.
///
/// Returns the expectation value of the Pauli string. Z applied first, then
/// X (matches the measurement projection convention in this module).
///
/// # Panics
///
/// Panics if any MPS gate application fails on a valid site (should not happen
/// for in-range sites).
#[must_use]
pub fn pauli_expectation(
    mps: &Mps,
    flip_sites: &[usize],
    sign_sites: &[usize],
    phase: Complex64,
) -> Complex64 {
    if flip_sites.is_empty() && sign_sites.is_empty() {
        return phase;
    }
    let x_gate = DMatrix::from_row_slice(
        2,
        2,
        &[
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    );
    let z_diag = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
    let mut mps_op = mps.clone();
    for &k in sign_sites {
        mps_op
            .apply_diagonal_one_site(k, &z_diag)
            .expect("MPS op on valid site");
    }
    for &j in flip_sites {
        mps_op
            .apply_one_site_gate(j, &x_gate)
            .expect("MPS op on valid site");
    }
    let raw = mps_inner_product(mps, &mps_op);
    phase * raw
}

/// Compute `<mps|Z_q|mps>` by applying the decomposition to a clone and taking the inner product.
///
/// Returns the raw expectation value (before multiplying by the decomposition phase).
/// The full expectation is: `phase * apply_z_to_clone_and_overlap(...)`.
#[must_use]
pub fn z_expectation_value(tableau: &SparseStabY, mps: &Mps, q: usize) -> Complex64 {
    let decomp = decompose_z(tableau.stabs(), tableau.destabs(), q);
    match decomp {
        ZDecomposition::Stabilizer { phase, sign_sites } => {
            pauli_expectation(mps, &[], &sign_sites, phase)
        }
        ZDecomposition::DestabilizerFlip {
            flip_sites,
            phase,
            sign_sites,
        } => pauli_expectation(mps, &flip_sites, &sign_sites, phase),
    }
}

/// Endpoint tolerance shared by sampling and forced projection.
///
/// A cancellation residue this close to a Pauli endpoint would be amplified
/// by the projector's division by `sqrt(probability)`, so it is classified as
/// the indistinguishable exact endpoint before any RNG draw.
pub(super) const EXPECTATION_ENDPOINT_TOLERANCE: f64 = 1e-14;

/// A normalized forced projector divides by `sqrt(probability)`, so a valid
/// post-projection state has a survival ratio near one regardless of its Born
/// probability. `1e-12` is therefore twelve orders below a healthy branch; it
/// is not a lower bound on the smallest admissible probability.
pub(super) const BRANCH_VANISH_SURVIVAL_THRESHOLD: f64 = 1e-12;

pub(super) fn forced_outcome_probability(expectation: f64, outcome: bool) -> f64 {
    // This endpoint guard is deliberately looser than the 1e-15 product-site
    // installer and its 5e-16 proof assertion. Those tolerances classify a
    // normalized physical marginal after environment contraction. Here the
    // expectation has accumulated a full Pauli-string contraction, and the
    // resulting probability is about to divide a cancellation residue by
    // sqrt(2p). At p ~ 1e-15 that would amplify roundoff by O(1e7) before
    // normalize() turns it into a spurious state. Values within 1e-14 of a
    // Pauli endpoint are therefore treated as the exact eigenvalue they are
    // indistinguishable from at this contraction's accuracy.
    let expectation = if 1.0 - expectation.abs() <= EXPECTATION_ENDPOINT_TOLERANCE {
        expectation.signum()
    } else {
        expectation
    };
    let probability_zero = f64::midpoint(1.0, expectation).clamp(0.0, 1.0);
    if outcome {
        1.0 - probability_zero
    } else {
        probability_zero
    }
}

fn quantize_trivial_probability(probability: f64) -> f64 {
    [0.0_f64, 0.5, 1.0]
        .into_iter()
        .min_by(|left, right| {
            (probability - *left)
                .abs()
                .total_cmp(&(probability - *right).abs())
        })
        .expect("three trivial probabilities")
}

/// Return the probability used by both exact sampling and forced projection.
///
/// The trivial coefficient-MPS path is a pure stabilizer state even when
/// contraction roundoff reports a nearby non-stabilizer value, so its result
/// is quantized to `{0, 1/2, 1}` under the same predicate used by the
/// projector's tableau fast path.
pub(super) fn z_outcome_probability(
    tableau: &SparseStabY,
    mps: &Mps,
    q_idx: usize,
    outcome: bool,
    operation: &str,
) -> f64 {
    let norm_squared = mps.norm_squared();
    assert!(
        norm_squared.is_finite() && norm_squared > 0.0,
        "{operation}: cannot measure an MPS with non-finite or zero norm"
    );
    let expectation = (z_expectation_value(tableau, mps, q_idx).re / norm_squared).clamp(-1.0, 1.0);
    let probability = forced_outcome_probability(expectation, outcome);
    if is_mps_trivial(mps) {
        quantize_trivial_probability(probability)
    } else {
        probability
    }
}

fn projection_survival_ratio(mps: &Mps, pre_projection_norm_squared: f64) -> f64 {
    let post_projection_norm_squared = mps.norm_squared();
    assert!(
        pre_projection_norm_squared.is_finite() && post_projection_norm_squared.is_finite(),
        "forced Z projection produced a non-finite pre/post norm"
    );
    assert!(
        pre_projection_norm_squared > 0.0,
        "cannot project a zero-norm MPS"
    );
    let ratio = post_projection_norm_squared / pre_projection_norm_squared;
    assert!(
        ratio.is_finite(),
        "forced Z projection produced a non-finite survival ratio"
    );
    ratio
}

/// Compute the inner product <`mps_a|mps_b`> by contracting from left to right.
fn mps_inner_product(mps_a: &Mps, mps_b: &Mps) -> Complex64 {
    let d = mps_a.phys_dim();
    let mut transfer = DMatrix::from_element(1, 1, Complex64::new(1.0, 0.0));

    for q in 0..mps_a.num_sites() {
        let chi_r_a = mps_a.bond_dim(q + 1);
        let chi_r_b = mps_b.bond_dim(q + 1);
        let t_a = &mps_a.tensors()[q];
        let t_b = &mps_b.tensors()[q];

        let mut new_transfer = DMatrix::zeros(chi_r_a, chi_r_b);
        for sigma in 0..d {
            let block_a = crate::mps::tensor::phys_block(t_a, sigma, chi_r_a);
            let block_b = crate::mps::tensor::phys_block(t_b, sigma, chi_r_b);
            let conj_a_t = block_a.conjugate().transpose();
            let tmp = &conj_a_t * &transfer * &block_b;
            new_transfer += tmp;
        }
        transfer = new_transfer;
    }

    transfer[(0, 0)]
}

/// Find the stabilizer index that `mz_forced` will select for replacement.
///
/// This is the minimum-weight stabilizer that anticommutes with `Z_q`,
/// matching the logic in `SparseStabY::nondeterministic_meas`.
fn find_replaced_stabilizer(tableau: &SparseStabY, q_idx: usize) -> usize {
    let stabs = tableau.stabs();
    let col_x = &stabs.col_x[q_idx];

    let mut best_id = None;
    let mut best_weight = usize::MAX;
    for stab_id in col_x {
        let weight = stabs.row_x[stab_id].len() + stabs.row_z[stab_id].len();
        if weight < best_weight {
            best_weight = weight;
            best_id = Some(stab_id);
            if weight == 1 {
                break;
            }
        }
    }
    best_id.expect("col_x should be non-empty for DestabilizerFlip case")
}

fn stabilizer_weight(tableau: &SparseStabY, stabilizer: usize) -> usize {
    let stabs = tableau.stabs();
    stabs.row_x[stabilizer].len() + stabs.row_z[stabilizer].len()
}

fn replacement_compensation_cost(candidates: &[usize], replacement: usize) -> usize {
    candidates
        .iter()
        .copied()
        .filter(|&candidate| candidate != replacement)
        .map(|target| 2 * replacement.abs_diff(target) - 1)
        .sum()
}

fn record_replacement_choice_diagnostics(
    tableau: &SparseStabY,
    q_idx: usize,
    chosen: usize,
    work: &mut PreReductionWork,
) {
    let candidates = tableau.stabs().col_x[q_idx].iter().collect::<Vec<_>>();
    let chosen_weight = stabilizer_weight(tableau, chosen);
    let minimum_weight_candidates = candidates
        .iter()
        .copied()
        .filter(|&candidate| stabilizer_weight(tableau, candidate) == chosen_weight)
        .collect::<Vec<_>>();
    let weight_plus_one_candidates = chosen_weight
        .checked_add(1)
        .map_or_else(Vec::new, |weight| {
            candidates
                .iter()
                .copied()
                .filter(|&candidate| stabilizer_weight(tableau, candidate) == weight)
                .collect()
        });
    let targets = candidates
        .iter()
        .copied()
        .filter(|&candidate| candidate != chosen)
        .collect::<Vec<_>>();
    let target_min = targets.iter().copied().min();
    let target_max = targets.iter().copied().max();
    let support_min = candidates.iter().copied().min().unwrap_or(chosen);
    let support_max = candidates.iter().copied().max().unwrap_or(chosen);

    work.chosen_stabilizer = Some(chosen);
    work.chosen_stabilizer_weight = Some(chosen_weight);
    work.replacement_candidate_count = candidates.len();
    work.minimum_weight_candidate_count = minimum_weight_candidates.len();
    work.chosen_compensation_cost = replacement_compensation_cost(&candidates, chosen);
    work.minimum_weight_optimal_cost = minimum_weight_candidates
        .iter()
        .copied()
        .map(|candidate| replacement_compensation_cost(&candidates, candidate))
        .min()
        .expect("the chosen stabilizer is a minimum-weight candidate");
    work.weight_plus_one_candidate_count = weight_plus_one_candidates.len();
    work.weight_plus_one_optimal_cost = weight_plus_one_candidates
        .iter()
        .copied()
        .map(|candidate| replacement_compensation_cost(&candidates, candidate))
        .min();
    work.target_min = target_min;
    work.target_max = target_max;
    work.target_span = target_min
        .zip(target_max)
        .map_or(0, |(minimum, maximum)| maximum - minimum);
    work.compensation_support_span = support_max - support_min;
}

/// Test hook for `pre_reduce_for_measurement`.
///
/// # Errors
///
/// Returns an [`MpsError`] if an exact compensating MPS gate fails.
pub fn pre_reduce_for_measurement_pub(
    tableau: &mut SparseStabY,
    mps: &mut Mps,
    q_idx: usize,
) -> Result<(), MpsError> {
    pre_reduce_for_measurement(tableau, mps, q_idx, true).map(drop)
}

/// Pre-reduce the stabilizer tableau so that `Z_q` anticommutes with at most
/// one stabilizer. For each other anti-commuting stab:
///   - Tableau: `S[other] *= S[replaced]`, `D[replaced] *= D[other]` (via
///     full Y-convention `multiply_row`, including sign/phase tracking).
///   - MPS (when `apply_mps_compensation=true`): apply virtual-frame
///     `CNOT(c=replaced, t=other)` for CAMPS state preservation. The tableau
///     change transforms the Clifford as `C → C · CNOT` — applying the
///     same CNOT to the MPS (self-inverse) compensates so
///     `C'·MPS_new = C·MPS_old`. Non-adjacent CNOTs use
///     `apply_long_range_two_site_gate`.
///
/// `apply_mps_compensation` is `true` for the exact-state caller
/// `project_forced_z`, used by `prob_bitstring` / `amplitude_iterative`. It is `false` for random
/// measurement (`measure_qubit_stab_mps_pragmatic`): the state representation becomes
/// inconsistent with the tableau after row ops, but measurement
/// statistics stay correct and subsequent measurements remain
/// self-consistent. Skipping compensation avoids SWAP-chain bond growth
/// during measurement-heavy circuits (MAST magic-state injection).
///
/// Proper long-term fix: lazy virtual-frame tracking — accumulate a
/// deferred Clifford V such that effective MPS = V·stored MPS, conjugate
/// Pauli strings by V before applying to stored MPS, flush only when MPS
/// must be read directly.
fn pre_reduce_for_measurement(
    tableau: &mut SparseStabY,
    mps: &mut Mps,
    q_idx: usize,
    apply_mps_compensation: bool,
) -> Result<Vec<usize>, MpsError> {
    pre_reduce_for_measurement_impl::<false>(
        tableau,
        mps,
        q_idx,
        apply_mps_compensation,
        &mut PreReductionWork::default(),
    )
}

fn pre_reduce_for_measurement_profiled(
    tableau: &mut SparseStabY,
    mps: &mut Mps,
    q_idx: usize,
    event: &mut super::PreReductionTelemetry,
) -> Result<Vec<usize>, MpsError> {
    let mut work = PreReductionWork::default();
    let result = pre_reduce_for_measurement_impl::<true>(tableau, mps, q_idx, true, &mut work);
    finish_pre_reduction_work(event, &work);
    result
}

fn pre_reduce_for_measurement_impl<const DIAGNOSTICS: bool>(
    tableau: &mut SparseStabY,
    mps: &mut Mps,
    q_idx: usize,
    apply_mps_compensation: bool,
    work: &mut PreReductionWork,
) -> Result<Vec<usize>, MpsError> {
    let col_x = &tableau.stabs().col_x[q_idx];
    if col_x.len() <= 1 {
        return Ok(Vec::new());
    }

    let replaced_idx = find_replaced_stabilizer(tableau, q_idx);
    if DIAGNOSTICS {
        record_replacement_choice_diagnostics(tableau, q_idx, replaced_idx, work);
    }
    let anticom: Vec<usize> = tableau.stabs().col_x[q_idx]
        .iter()
        .filter(|&id| id != replaced_idx)
        .collect();

    if apply_mps_compensation {
        // Exact callers need the phase-complete Clifford composition and the
        // matching inverse action on the coefficient MPS. Treat the sequence
        // transactionally so an SVD failure cannot leave a partially changed
        // tableau/MPS pair.
        let original_tableau = tableau.clone();
        let original_mps = mps.clone();
        let mut modified_sites = Vec::with_capacity(1 + anticom.len());
        modified_sites.push(replaced_idx);
        for other_id in anticom {
            if let Err(error) =
                apply_cnot_to_mps_impl::<DIAGNOSTICS>(mps, replaced_idx, other_id, work)
            {
                *tableau = original_tableau;
                *mps = original_mps;
                return Err(error);
            }
            crate::stab_mps::tableau_compose::right_compose_cx(tableau, replaced_idx, other_id);
            modified_sites.push(other_id);
        }
        modified_sites.sort_unstable();
        modified_sites.dedup();
        Ok(modified_sites)
    } else {
        // Preserve the established pragmatic measurement path byte-for-byte:
        // it intentionally updates only the structural row frame and does not
        // compensate the MPS. Its tracked drift is outside the exact route.
        let n = tableau.num_qubits();
        let stabs_snapshot = tableau.stabs().clone();
        let destabs_snapshot = tableau.destabs().clone();
        for other_id in anticom {
            crate::stab_mps::tableau_compose::multiply_row(
                tableau.stabs_mut(),
                other_id,
                &stabs_snapshot,
                replaced_idx,
                n,
            );
            crate::stab_mps::tableau_compose::multiply_row(
                tableau.destabs_mut(),
                replaced_idx,
                &destabs_snapshot,
                other_id,
                n,
            );
        }
        Ok(Vec::new())
    }
}

fn apply_cnot_to_mps(mps: &mut Mps, control: usize, target: usize) -> Result<(), MpsError> {
    apply_cnot_to_mps_impl::<false>(mps, control, target, &mut PreReductionWork::default())
}

fn internal_bond_profile(mps: &Mps) -> Vec<usize> {
    mps.bond_dims()
        .iter()
        .copied()
        .skip(1)
        .take(mps.num_sites().saturating_sub(1))
        .collect()
}

fn record_cnot_diagnostics(
    mps: &Mps,
    control: usize,
    target: usize,
    input_bond_profile: Vec<usize>,
    svd_start: usize,
    route: (bool, bool),
    work: &mut PreReductionWork,
) {
    let (structural_identity, unconditional_x) = route;
    let output_bond_profile = internal_bond_profile(mps);
    let svd_count = work.mps.svd_steps.len() - svd_start;
    let peak_bond_rank = input_bond_profile
        .iter()
        .copied()
        .chain(
            work.mps.svd_steps[svd_start..]
                .iter()
                .map(|step| step.output_rank),
        )
        .max()
        .unwrap_or(1);
    work.cnot_steps.push(super::PreReductionCnotTelemetry {
        control,
        target,
        distance: control.abs_diff(target),
        input_bond_profile,
        output_bond_profile,
        peak_bond_rank,
        svd_start,
        svd_count,
        structural_identity,
        unconditional_x,
    });
}

fn apply_cnot_to_mps_impl<const DIAGNOSTICS: bool>(
    mps: &mut Mps,
    control: usize,
    target: usize,
    work: &mut PreReductionWork,
) -> Result<(), MpsError> {
    work.compensating_cnot_count += 1;
    let input_bond_profile = DIAGNOSTICS.then(|| internal_bond_profile(mps));
    let svd_start = work.mps.svd_steps.len();
    // Optimization: if the control site has no |1⟩_virt amplitude, CNOT is
    // identity on this MPS — skip the unnecessary SWAP/SVD work.
    // Mirror: if control has no |0⟩_virt amp, CNOT reduces to X on target.
    if mps_site_block_is_structurally_zero(mps, control, 1) {
        work.structural_identity_cnot_count += 1;
        if DIAGNOSTICS {
            record_cnot_diagnostics(
                mps,
                control,
                target,
                input_bond_profile.expect("diagnostic CNOT has an input profile"),
                svd_start,
                (true, false),
                work,
            );
        }
        return Ok(());
    }
    if mps_site_block_is_structurally_zero(mps, control, 0) {
        work.unconditional_x_cnot_count += 1;
        // Control is |1⟩ → CNOT unconditionally flips target = X on target.
        let result = time_pre_reduction_tensor_work::<DIAGNOSTICS, _>(&mut work.mps, || {
            let x_gate = DMatrix::from_row_slice(
                2,
                2,
                &[
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            );
            mps.apply_one_site_gate(target, &x_gate)
        });
        result?;
        if DIAGNOSTICS {
            record_cnot_diagnostics(
                mps,
                control,
                target,
                input_bond_profile.expect("diagnostic CNOT has an input profile"),
                svd_start,
                (false, true),
                work,
            );
        }
        return Ok(());
    }

    let (cnot_control_low, cnot_control_high) =
        time_pre_reduction_tensor_work::<DIAGNOSTICS, _>(&mut work.mps, || {
            let zero = Complex64::new(0.0, 0.0);
            let one = Complex64::new(1.0, 0.0);
            let cnot_control_low = DMatrix::from_row_slice(
                4,
                4,
                &[
                    one, zero, zero, zero, zero, one, zero, zero, zero, zero, zero, one, zero,
                    zero, one, zero,
                ],
            );
            let cnot_control_high = DMatrix::from_row_slice(
                4,
                4,
                &[
                    one, zero, zero, zero, zero, zero, zero, one, zero, zero, one, zero, zero, one,
                    zero, zero,
                ],
            );
            (cnot_control_low, cnot_control_high)
        });
    let (first, second, gate) = if control < target {
        (control, target, cnot_control_low)
    } else {
        (target, control, cnot_control_high)
    };
    let result = if DIAGNOSTICS {
        mps.apply_long_range_two_site_gate_profiled(first, second, &gate, &mut work.mps)
    } else {
        mps.apply_long_range_two_site_gate(first, second, &gate)
    };
    if DIAGNOSTICS && result.is_ok() {
        record_cnot_diagnostics(
            mps,
            control,
            target,
            input_bond_profile.expect("diagnostic CNOT has an input profile"),
            svd_start,
            (false, false),
            work,
        );
    }
    result
}

/// A deferred Clifford primitive in the virtual-frame queue.
///
/// The queue represents a Clifford `V = ops[last] · ... · ops[0]` where
/// index 0 is the first pushed (earliest applied if flushed). Each primitive
/// has a cheap Heisenberg conjugation rule (bit XOR on flip/sign sets) and
/// a cheap MPS application (single-site for H, diagonal for CZ, SWAP-chain
/// for CNOT).
#[derive(Clone, Copy, Debug)]
pub enum DeferredOp {
    /// CNOT(control, target).
    Cnot(usize, usize),
    /// Hadamard on qubit.
    H(usize),
    /// CZ(a, b) — symmetric.
    Cz(usize, usize),
    /// Pauli Z on qubit. Used for outcome-dependent W basis rotation:
    /// for outcome=1, W includes a `Z_id` factor to flip `Z_id` → -`X_id`.
    Z(usize),
    /// Phase gate adjoint (SZ†) — needed for outcome-dependent W when
    /// flip and sign overlap at id (Y-like Pauli).
    SZdg(usize),
    /// Phase gate SZ — needed for outcome-dependent W when the
    /// decomposition phase `sp` is purely imaginary. Conjugation rule:
    /// SZ†·P·SZ — if X at q, toggle Z at q and multiply phase by -i.
    SZ(usize),
}

fn toggle(v: &mut Vec<usize>, x: usize) {
    if let Some(pos) = v.iter().position(|&y| y == x) {
        v.swap_remove(pos);
    } else {
        v.push(x);
    }
}

/// Conjugate a Pauli `P = X_flip · Z_sign` by `V†` where
/// `V = ops[last] · ops[last-1] · ... · ops[0]`. Updates `flip_sites` and
/// `sign_sites` in place to represent `V† · P · V`. The scalar phase is
/// updated whenever the X-block-before-Z-block representation picks up a
/// sign or a factor of i.
///
/// Heisenberg rules:
/// - CNOT(c, t): `X_c -> X_c · X_t`; `Z_t -> Z_c · Z_t`.
/// - H(q): swap `X_q` and `Z_q` (swap q between flip and sign).
/// - CZ(a, b): `X_a -> X_a · Z_b`; `X_b -> X_b · Z_a`.
///
/// Order: `V† P V = op_0·...·op_last·P·op_last·...·op_0`, so iterate `ops`
/// in REVERSE (innermost conjugation by `op_last` first).
pub fn conjugate_pauli_by_deferred_ops(
    flip_sites: &mut Vec<usize>,
    sign_sites: &mut Vec<usize>,
    phase: &mut Complex64,
    ops: &[DeferredOp],
) {
    for op in ops.iter().rev() {
        match *op {
            DeferredOp::Cnot(c, t) => {
                let has_x_c = flip_sites.contains(&c);
                let has_z_t = sign_sites.contains(&t);
                if has_x_c {
                    toggle(flip_sites, t);
                }
                if has_z_t {
                    toggle(sign_sites, c);
                }
            }
            DeferredOp::H(q) => {
                let has_x = flip_sites.contains(&q);
                let has_z = sign_sites.contains(&q);
                // Swap membership of q between flip and sign.
                if has_x != has_z {
                    if has_x {
                        toggle(flip_sites, q);
                        toggle(sign_sites, q);
                    } else {
                        toggle(sign_sites, q);
                        toggle(flip_sites, q);
                    }
                }
                // If both: Y → -Y (H·Y·H = -Y). Membership stays. Phase flips.
                if has_x && has_z {
                    *phase = -*phase;
                }
            }
            DeferredOp::Cz(a, b) => {
                let has_x_a = flip_sites.contains(&a);
                let has_x_b = flip_sites.contains(&b);
                // CZ·(X_a X_b)·CZ = Y_a Y_b = -(XZ)_a (XZ)_b in the
                // X-block-before-Z-block convention.
                if has_x_a && has_x_b {
                    *phase = -*phase;
                }
                if has_x_a {
                    toggle(sign_sites, b);
                }
                if has_x_b {
                    toggle(sign_sites, a);
                }
            }
            DeferredOp::Z(q) => {
                // Z·X_q·Z = -X_q. If X present at q (and Z not at q), phase flips.
                // Z·Y_q·Z = -Y_q (Y has X factor). So if X present regardless of Z, phase flips.
                // Z·Z_q·Z = Z_q. No flip if only Z at q.
                if flip_sites.contains(&q) {
                    *phase = -*phase;
                }
            }
            DeferredOp::SZdg(q) => {
                // SZdg conjugation: SZdg†·P·SZdg = SZ·P·SZdg.
                // SZ·X·SZdg = Y = iXZ; SZ·Z·SZdg = Z.
                // If X at q and Z not at q: add q to sign, phase *= i.
                // If X at q and Z at q: SZ·Y·SZdg = i·(SZ·X·SZdg)·(SZ·Z·SZdg) = i·Y·Z = i·(iXZ)·Z = -X.
                //   So XZ → X only (toggle z), phase *= i (aggregate: p · iXZ · Z = ip·X).
                // Matrix sanity-check: SZ = [[1,0],[0,i]], SZdg = [[1,0],[0,-i]],
                //   Y = [[0,-i],[i,0]].
                //   SZ·Y·SZdg = [[1,0],[0,i]]·[[0,-i],[i,0]]·[[1,0],[0,-i]]
                //             = [[0,-i],[-1,0]]·[[1,0],[0,-i]]
                //             = [[0, -1],[-1, 0]] = -X. ✓
                let has_x = flip_sites.contains(&q);
                let has_z = sign_sites.contains(&q);
                if has_x && !has_z {
                    // X only → XZ (add Z), phase *= i.
                    toggle(sign_sites, q);
                    *phase *= Complex64::new(0.0, 1.0);
                } else if has_x && has_z {
                    // XZ → X only (remove Z), phase *= i.
                    toggle(sign_sites, q);
                    *phase *= Complex64::new(0.0, 1.0);
                }
                // Z only or none: unchanged.
            }
            DeferredOp::SZ(q) => {
                // SZ conjugation: SZdg·P·SZ.
                // SZdg·X·SZ = -Y = -i·X·Z; SZdg·Z·SZ = Z; SZdg·Y·SZ = X.
                // X only → X·Z, phase *= -i.
                // X·Z → X only, phase *= -i.
                // Z only or none: unchanged.
                let has_x = flip_sites.contains(&q);
                if has_x {
                    toggle(sign_sites, q);
                    *phase *= Complex64::new(0.0, -1.0);
                }
            }
        }
    }
}

/// Return the MPS-site support of the current conjugated observable `C† Z_q C`.
///
/// The support is derived from the same `decompose_z` result used by the
/// measurement protocol. When lazy measurement has accumulated a virtual
/// Clifford frame, the decomposition is conjugated through that frame before
/// its X and Z sites are combined.
#[must_use]
pub(crate) fn conjugated_z_support(
    tableau: &SparseStabY,
    q_idx: usize,
    deferred: &[DeferredOp],
) -> Vec<usize> {
    let (mut flip_sites, mut sign_sites, mut phase) =
        match decompose_z(tableau.stabs(), tableau.destabs(), q_idx) {
            ZDecomposition::Stabilizer { phase, sign_sites } => (Vec::new(), sign_sites, phase),
            ZDecomposition::DestabilizerFlip {
                flip_sites,
                phase,
                sign_sites,
            } => (flip_sites, sign_sites, phase),
        };
    conjugate_pauli_by_deferred_ops(&mut flip_sites, &mut sign_sites, &mut phase, deferred);
    flip_sites.extend(sign_sites);
    flip_sites.sort_unstable();
    flip_sites.dedup();
    flip_sites
}

/// Backwards-compatible CNOT-only conjugation wrapper. CNOT conjugation
/// doesn't touch phase, so this discards the phase output.
pub fn conjugate_pauli_by_deferred(
    flip_sites: &mut Vec<usize>,
    sign_sites: &mut Vec<usize>,
    cnots: &[(usize, usize)],
) {
    let ops: Vec<DeferredOp> = cnots.iter().map(|&(c, t)| DeferredOp::Cnot(c, t)).collect();
    let mut phase = Complex64::new(1.0, 0.0);
    conjugate_pauli_by_deferred_ops(flip_sites, sign_sites, &mut phase, &ops);
}

/// Apply the deferred op queue `V = ops[last]·...·ops[0]` to `mps` and clear.
///
/// # Errors
///
/// Returns an [`MpsError`] if an MPS operation fails. The queue and original
/// MPS remain unchanged so a caller never observes a half-materialized frame.
pub fn flush_deferred_ops(mps: &mut Mps, ops: &mut Vec<DeferredOp>) -> Result<(), MpsError> {
    let mut working = mps.clone();
    let h_gate = DMatrix::from_row_slice(
        2,
        2,
        &[
            Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
            Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
            Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
            Complex64::new(-std::f64::consts::FRAC_1_SQRT_2, 0.0),
        ],
    );
    let cz_diag = [
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(-1.0, 0.0),
    ];
    for op in ops.iter() {
        match *op {
            DeferredOp::Cnot(c, t) => apply_cnot_to_mps(&mut working, c, t)?,
            DeferredOp::H(q) => {
                working.apply_one_site_gate(q, &h_gate)?;
            }
            DeferredOp::Cz(a, b) => {
                // CZ is diagonal; use apply_two_site_gate (adjacent) or
                // long-range two-site (non-adjacent). Either preserves bond
                // dim since it's diagonal in the product basis.
                let (q0, q1) = if a < b { (a, b) } else { (b, a) };
                let o = Complex64::new(0.0, 0.0);
                let cz = DMatrix::from_row_slice(
                    4,
                    4,
                    &[
                        cz_diag[0], o, o, o, o, cz_diag[1], o, o, o, o, cz_diag[2], o, o, o, o,
                        cz_diag[3],
                    ],
                );
                if q1 == q0 + 1 {
                    working.apply_two_site_gate(q0, &cz)?;
                } else {
                    working.apply_long_range_two_site_gate(q0, q1, &cz)?;
                }
            }
            DeferredOp::Z(q) => {
                let z_diag = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
                working.apply_diagonal_one_site(q, &z_diag)?;
            }
            DeferredOp::SZdg(q) => {
                let sdg_diag = [Complex64::new(1.0, 0.0), Complex64::new(0.0, -1.0)];
                working.apply_diagonal_one_site(q, &sdg_diag)?;
            }
            DeferredOp::SZ(q) => {
                let s_diag = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
                working.apply_diagonal_one_site(q, &s_diag)?;
            }
        }
    }
    *mps = working;
    ops.clear();
    Ok(())
}

/// Backwards-compatible CNOT-only flush wrapper.
///
/// # Errors
///
/// Returns an [`MpsError`] if a deferred CNOT cannot be applied. Both the MPS
/// and the queue remain unchanged on failure.
pub fn flush_deferred(mps: &mut Mps, cnots: &mut Vec<(usize, usize)>) -> Result<(), MpsError> {
    let mut ops: Vec<DeferredOp> = cnots.iter().map(|&(c, t)| DeferredOp::Cnot(c, t)).collect();
    flush_deferred_ops(mps, &mut ops)?;
    cnots.clear();
    Ok(())
}

/// Returns true if the MPS tensor at `site` has a structurally zero
/// σ=`block`. An approximate local-zero test is not gauge invariant.
fn mps_site_block_is_structurally_zero(mps: &Mps, site: usize, block: usize) -> bool {
    let chi_r = mps.bond_dim(site + 1);
    let t = &mps.tensors()[site];
    let start_col = block * chi_r;
    for i in 0..t.nrows() {
        for j in 0..chi_r {
            if t[(i, start_col + j)] != Complex64::new(0.0, 0.0) {
                return false;
            }
        }
    }
    true
}

/// Recover the unique MPS-frame computational-basis index selected after all
/// physical Z outcomes have been projected.
///
/// A forced projection through the stabilizer branch can leave the selected
/// coefficient at a nonzero virtual index. The final tableau supplies the
/// invertible GF(2) system relating physical Z eigenvalues to virtual
/// stabilizer-sign bits.
///
/// # Panics
///
/// Panics if the outcome length differs from the tableau size or the fully
/// projected tableau does not define an invertible real-valued Z constraint
/// system. Those cases indicate an internal projection inconsistency.
#[must_use]
pub(super) fn projected_mps_basis_index(tableau: &SparseStabY, outcomes: &[bool]) -> Vec<u8> {
    let n = tableau.num_qubits();
    assert_eq!(outcomes.len(), n, "projected outcome length mismatch");

    let mut matrix = vec![vec![false; n + 1]; n];
    for (q, &outcome) in outcomes.iter().enumerate() {
        let ZDecomposition::Stabilizer { phase, sign_sites } =
            decompose_z(tableau.stabs(), tableau.destabs(), q)
        else {
            panic!("projected tableau still has a destabilizer component at qubit {q}");
        };
        assert!(
            phase.im.abs() < 1e-10,
            "projected Z decomposition has non-real phase at qubit {q}"
        );
        for site in sign_sites {
            matrix[q][site] = true;
        }
        matrix[q][n] = outcome ^ (phase.re < 0.0);
    }

    let mut pivot_row = 0;
    let mut pivot_columns = vec![usize::MAX; n];
    for col in 0..n {
        let found = matrix[pivot_row..]
            .iter()
            .position(|row| row[col])
            .map(|offset| pivot_row + offset);
        if let Some(found_row) = found {
            matrix.swap(pivot_row, found_row);
            let pivot = matrix[pivot_row].clone();
            for (row_index, row) in matrix.iter_mut().enumerate() {
                if row_index != pivot_row && row[col] {
                    for (cell, &pivot_cell) in row.iter_mut().zip(pivot.iter()) {
                        *cell ^= pivot_cell;
                    }
                }
            }
            pivot_columns[pivot_row] = col;
            pivot_row += 1;
        }
    }
    assert_eq!(
        pivot_row, n,
        "projected Z constraints are not an invertible MPS-frame basis"
    );

    let mut index = vec![0; n];
    for row in 0..n {
        index[pivot_columns[row]] = u8::from(matrix[row][n]);
    }
    index
}

/// Right-compose the phase-complete measurement basis rotation `W`.
///
/// Keeping an independently predicted tableau lets the exact projection path
/// recover the Pauli gauge omitted by `SparseStabY::nondeterministic_meas`
/// when it structurally XORs destabilizer rows without propagating their
/// phases.
fn right_compose_measurement_basis_rotation(
    tableau: &mut SparseStabY,
    id: usize,
    phase: Complex64,
    sign_sites: &[usize],
    outcome: bool,
    phase_accumulator: Option<&mut crate::stab_mps::canonical_ket::CanonicalPhaseTracker>,
) {
    let signed_phase = Complex64::new(if outcome { -1.0 } else { 1.0 }, 0.0) * phase;
    let id_in_sign = sign_sites.contains(&id);

    if signed_phase.im.abs() < 1e-9 {
        assert!(
            !id_in_sign,
            "real measurement phase {signed_phase:?} must not overlap the flip site"
        );
        if signed_phase.re < 0.0 {
            crate::stab_mps::tableau_compose::right_compose_z(tableau, id);
        }
        for &site in sign_sites {
            if site != id {
                crate::stab_mps::tableau_compose::right_compose_cz(tableau, id, site);
            }
        }
    } else {
        assert!(
            id_in_sign,
            "imaginary measurement phase {signed_phase:?} must overlap the flip site"
        );
        assert!(
            signed_phase.re.abs() < 1e-9,
            "measurement phase must be a fourth root of unity: {signed_phase:?}"
        );
        for &site in sign_sites {
            if site != id {
                crate::stab_mps::tableau_compose::right_compose_cz(tableau, id, site);
            }
        }
        if signed_phase.im > 0.0 {
            crate::stab_mps::tableau_compose::right_compose_sz(tableau, id);
        } else {
            crate::stab_mps::tableau_compose::right_compose_szdg(tableau, id);
        }
    }
    let before_h = phase_accumulator.as_ref().map(|_| tableau.clone());
    crate::stab_mps::tableau_compose::right_compose_h(tableau, id);
    if let (Some(accumulator), Some(before_h)) = (phase_accumulator, before_h.as_ref()) {
        accumulator.right_compose_h(before_h, tableau, id);
    }
}

/// Return the virtual Z gauge between two structurally identical tableaux.
///
/// `SparseStabY::nondeterministic_meas` structurally XORs destabilizer rows
/// without propagating their phases. The forced outcome fixes stabilizer-row
/// signs, so the only possible difference from a phase-complete predicted
/// tableau is a right-composed Z on each returned virtual site:
/// `measured * Z_gauge = predicted`.
fn measurement_pauli_gauge_sites(predicted: &SparseStabY, measured: &SparseStabY) -> Vec<usize> {
    assert_eq!(predicted.num_qubits(), measured.num_qubits());
    let rows_match = |predicted_rows: &[BitSet], measured_rows: &[BitSet]| {
        predicted_rows
            .iter()
            .zip(measured_rows)
            .all(|(predicted_row, measured_row)| predicted_row.iter().eq(measured_row.iter()))
    };
    // Compare semantic set contents rather than BitSet's backing allocation:
    // operations above bit 63 can leave different numbers of trailing zero
    // words in otherwise identical sparse rows.
    assert!(rows_match(
        &predicted.stabs().row_x,
        &measured.stabs().row_x
    ));
    assert!(rows_match(
        &predicted.stabs().row_z,
        &measured.stabs().row_z
    ));
    assert!(rows_match(
        &predicted.destabs().row_x,
        &measured.destabs().row_x
    ));
    assert!(rows_match(
        &predicted.destabs().row_z,
        &measured.destabs().row_z
    ));
    assert!(
        predicted
            .stabs()
            .signs_i
            .iter()
            .eq(measured.stabs().signs_i.iter())
    );
    assert!(
        predicted
            .destabs()
            .signs_i
            .iter()
            .eq(measured.destabs().signs_i.iter())
    );

    // `SparseStabY::apply_outcome`, called by the nondeterministic
    // `mz_forced` path, forces the replacement stabilizer row's sign to the
    // requested outcome. The predicted basis rotation encodes that same sign,
    // so an X-gauge difference is structurally unreachable here.
    assert!(
        predicted
            .stabs()
            .signs_minus
            .iter()
            .eq(measured.stabs().signs_minus.iter()),
        "apply_outcome must leave no stabilizer-sign (X-gauge) difference"
    );
    (0..predicted.num_qubits())
        .filter(|&site| {
            predicted.destabs().signs_minus.contains(site)
                != measured.destabs().signs_minus.contains(site)
        })
        .collect()
}

/// Apply the inverse of the virtual Pauli gauge between two structurally
/// identical Clifford tableaux.
fn compensate_measurement_pauli_gauge(
    mps: &mut Mps,
    predicted: &SparseStabY,
    measured: &SparseStabY,
) -> Vec<usize> {
    let gauge_sites = measurement_pauli_gauge_sites(predicted, measured);
    let z_diag = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
    for &site in &gauge_sites {
        mps.apply_diagonal_one_site(site, &z_diag)
            .expect("MPS op on valid site");
    }
    gauge_sites
}

/// Apply the forced tableau update and enqueue its phase-complete inverse.
///
/// If the phase-complete basis rotation predicts `C_predicted = C_before * W`,
/// the structural forced-measurement implementation can return a tableau with
/// a virtual Pauli gauge, `C_measured * G = C_predicted`. The deferred frame
/// must therefore be left-multiplied by both factors: `G * W^-1`.
fn enqueue_lazy_measurement_basis_update(
    tableau: &mut SparseStabY,
    deferred: &mut Vec<DeferredOp>,
    q_idx: usize,
    id: usize,
    phase: Complex64,
    sign_sites: &[usize],
    outcome: bool,
) -> Vec<usize> {
    assert!(
        tableau.tracks_destab_signs(),
        "lazy measurement requires destabilizer-sign tracking"
    );
    let mut predicted_tableau = tableau.clone();
    right_compose_measurement_basis_rotation(
        &mut predicted_tableau,
        id,
        phase,
        sign_sites,
        outcome,
        None,
    );

    let signed_phase = Complex64::new(if outcome { -1.0 } else { 1.0 }, 0.0) * phase;
    let id_in_sign = sign_sites.contains(&id);
    if signed_phase.im.abs() < 1e-9 {
        debug_assert!(
            !id_in_sign,
            "lazy measure: real signed phase={signed_phase:?} but id in sign"
        );
        if signed_phase.re < 0.0 {
            deferred.push(DeferredOp::Z(id));
        }
        for &site in sign_sites {
            if site != id {
                deferred.push(DeferredOp::Cz(id, site));
            }
        }
    } else {
        debug_assert!(
            id_in_sign,
            "lazy measure: imaginary signed phase={signed_phase:?} but id not in sign"
        );
        debug_assert!(
            signed_phase.re.abs() < 1e-9,
            "lazy measure: signed phase={signed_phase:?} not pure imaginary"
        );
        for &site in sign_sites {
            if site != id {
                deferred.push(DeferredOp::Cz(id, site));
            }
        }
        // W contains SZ for +i and SZdg for -i, so W^-1 contains the adjoint.
        if signed_phase.im > 0.0 {
            deferred.push(DeferredOp::SZdg(id));
        } else {
            deferred.push(DeferredOp::SZ(id));
        }
    }
    deferred.push(DeferredOp::H(id));

    let result = tableau.mz_forced(q_idx, outcome);
    debug_assert_eq!(result.outcome, outcome);
    let gauge_sites = measurement_pauli_gauge_sites(&predicted_tableau, tableau);
    deferred.extend(gauge_sites.iter().copied().map(DeferredOp::Z));
    gauge_sites
}

/// Convert a normalized single-flip Pauli eigenstate into the coefficient
/// basis selected by the corresponding tableau measurement.
fn collapse_projected_flip_site(mps: &mut Mps, id: usize) {
    let chi_r = mps.bond_dim(id + 1);
    let block_0 = crate::mps::tensor::phys_block(&mps.tensors()[id], 0, chi_r)
        * Complex64::new(std::f64::consts::SQRT_2, 0.0);
    let zero = DMatrix::zeros(mps.tensors()[id].nrows(), chi_r);
    mps.set_physical_block(id, 0, &block_0);
    mps.set_physical_block(id, 1, &zero);
}

/// Project a real single-site X branch and absorb its measurement-basis H
/// without constructing a rank-doubled MPS sum.
fn project_single_flip_without_sign(
    mps: &mut Mps,
    id: usize,
    signed_phase: Complex64,
    probability: f64,
) {
    assert!(
        signed_phase.im.abs() < 1e-9,
        "single-flip branch without Z support must have real phase"
    );
    let chi_r = mps.bond_dim(id + 1);
    let block_0 = crate::mps::tensor::phys_block(&mps.tensors()[id], 0, chi_r);
    let block_1 = crate::mps::tensor::phys_block(&mps.tensors()[id], 1, chi_r);
    let denominator = Complex64::new((2.0 * probability).max(1e-20).sqrt(), 0.0);
    let projected = (block_0 + block_1 * signed_phase) / denominator;
    let zero = DMatrix::zeros(mps.tensors()[id].nrows(), chi_r);
    mps.set_physical_block(id, 0, &projected);
    mps.set_physical_block(id, 1, &zero);
}

/// Remove the direct-sum projector's structurally oversized virtual bonds.
///
/// A right-canonical QR sweep is an exact factorization and bounds every bond
/// by the Hilbert-space dimension to its right. The following configured SVD
/// sweep reuses that gauge and runs unconditionally: it removes exact rank
/// redundancy and numerical dust, while any nonzero truncation remains the
/// caller-approved cutoff/cap behavior and is recorded by MPS telemetry.
fn reduce_exact_projection_bonds(mps: &mut Mps) -> Result<(), MpsError> {
    mps.right_canonicalize();
    mps.compress_from_right_canonical()
}

struct ProjectionLocalityContext {
    pre_write_tensors: Vec<DMatrix<Complex64>>,
    pre_write_bond_dims: Vec<usize>,
    center_before_positioning: Option<usize>,
    center_before_positioning_is_valid: bool,
    center_before_projection_write: Option<usize>,
    center_before_projection_write_is_valid: bool,
    construction: Option<super::ProjectionConstruction>,
    projector_flip_sites: Vec<usize>,
    projector_sign_sites: Vec<usize>,
    projector_sites: Vec<usize>,
    gauge_compensation_sites: Vec<usize>,
    touched_sites: Vec<usize>,
}

impl ProjectionLocalityContext {
    fn new(mps: &Mps) -> Self {
        let (center, center_is_valid) = mps.projection_diagnostic_center();
        Self {
            pre_write_tensors: mps.tensors().to_vec(),
            pre_write_bond_dims: mps.bond_dims().to_vec(),
            center_before_positioning: center,
            center_before_positioning_is_valid: center_is_valid,
            center_before_projection_write: None,
            center_before_projection_write_is_valid: false,
            construction: None,
            projector_flip_sites: Vec::new(),
            projector_sign_sites: Vec::new(),
            projector_sites: Vec::new(),
            gauge_compensation_sites: Vec::new(),
            touched_sites: Vec::new(),
        }
    }

    fn record_projection_write(
        &mut self,
        mps: &Mps,
        construction: super::ProjectionConstruction,
        flip_sites: &[usize],
        sign_sites: &[usize],
    ) {
        assert!(
            self.construction.is_none(),
            "projection write recorded twice"
        );
        let (center, center_is_valid) = mps.projection_diagnostic_center();
        self.center_before_projection_write = center;
        self.center_before_projection_write_is_valid = center_is_valid;
        self.construction = Some(construction);
        self.projector_flip_sites = flip_sites.to_vec();
        self.projector_flip_sites.sort_unstable();
        self.projector_flip_sites.dedup();
        self.projector_sign_sites = sign_sites.to_vec();
        self.projector_sign_sites.sort_unstable();
        self.projector_sign_sites.dedup();
        self.projector_sites = self.projector_flip_sites.clone();
        self.projector_sites
            .extend(self.projector_sign_sites.iter().copied());
        self.projector_sites.sort_unstable();
        self.projector_sites.dedup();
        match construction {
            super::ProjectionConstruction::ScalarScale => {
                assert!(self.projector_sites.is_empty());
            }
            super::ProjectionConstruction::LocalBlockWrite => {
                assert_eq!(self.projector_flip_sites.len(), 1);
                assert!(self.projector_sign_sites.is_empty());
            }
            super::ProjectionConstruction::DirectSum => {
                assert!(!self.projector_sites.is_empty());
            }
        }
    }

    fn record_gauge_and_touched_sites(
        &mut self,
        gauge_compensation_sites: &[usize],
        mut touched_sites: Vec<usize>,
    ) {
        self.gauge_compensation_sites = gauge_compensation_sites.to_vec();
        self.gauge_compensation_sites.sort_unstable();
        self.gauge_compensation_sites.dedup();
        touched_sites.sort_unstable();
        touched_sites.dedup();
        self.touched_sites = touched_sites;
    }

    fn into_event(self, mps: &Mps) -> super::ProjectionQrLocalityTelemetry {
        fn span(indices: &[usize]) -> (Option<usize>, Option<usize>) {
            (indices.first().copied(), indices.last().copied())
        }

        let changed_tensors = self
            .pre_write_tensors
            .iter()
            .zip(mps.tensors())
            .enumerate()
            .filter_map(|(site, (before, after))| (before != after).then_some(site))
            .collect::<Vec<_>>();
        let changed_bonds = self
            .pre_write_bond_dims
            .iter()
            .zip(mps.bond_dims())
            .enumerate()
            .skip(1)
            .take(mps.num_sites().saturating_sub(1))
            .filter_map(|(bond, (before, after))| (before != after).then_some(bond))
            .collect::<Vec<_>>();
        let (center_before_qr, center_before_qr_is_valid) = mps.projection_diagnostic_center();
        let qr_sites = center_before_qr.unwrap_or_else(|| mps.num_sites().saturating_sub(1));
        let (changed_tensor_min, changed_tensor_max) = span(&changed_tensors);
        let (touched_site_min, touched_site_max) = span(&self.touched_sites);
        let (changed_bond_min, changed_bond_max) = span(&changed_bonds);
        let (projector_site_min, projector_site_max) = span(&self.projector_sites);
        let projector_span = projector_site_min
            .zip(projector_site_max)
            .map_or(0, |(minimum, maximum)| maximum - minimum);
        let construction = self
            .construction
            .expect("projection construction must precede QR telemetry");
        // `Mps::add` changes every tensor's shape, so its bitwise footprint is
        // uninformative and cannot validate the direct-sum support report. For
        // the local construction paths, every changed tensor must be covered
        // by the conservative touched-site frontier.
        if construction != super::ProjectionConstruction::DirectSum {
            assert!(
                changed_tensor_max <= touched_site_max,
                "non-direct-sum projection changed tensor {changed_tensor_max:?} beyond reported touched-site frontier {touched_site_max:?}"
            );
        }
        let qr_sites_skippable_by_center_ceiling =
            if center_before_qr_is_valid || !self.center_before_projection_write_is_valid {
                0
            } else {
                let center = self
                    .center_before_projection_write
                    .expect("a valid pre-write center must have a site");
                mps.num_sites().saturating_sub(1).saturating_sub(center)
            };
        let qr_sites_skippable_by_locality =
            if center_before_qr_is_valid || !self.center_before_projection_write_is_valid {
                0
            } else {
                let center = self
                    .center_before_projection_write
                    .expect("a valid pre-write center must have a site");
                let locality_frontier = touched_site_max.map_or(center, |site| center.max(site));
                mps.num_sites()
                    .saturating_sub(1)
                    .saturating_sub(locality_frontier)
            };
        debug_assert!(qr_sites_skippable_by_locality <= qr_sites);
        debug_assert!(qr_sites_skippable_by_locality <= qr_sites_skippable_by_center_ceiling);
        debug_assert!(qr_sites_skippable_by_center_ceiling <= qr_sites);

        super::ProjectionQrLocalityTelemetry {
            chain_length: mps.num_sites(),
            center_before_positioning: self.center_before_positioning,
            center_before_positioning_is_valid: self.center_before_positioning_is_valid,
            center_before_projection_write: self.center_before_projection_write,
            center_before_projection_write_is_valid: self.center_before_projection_write_is_valid,
            projection_entry_bond_profile: self
                .pre_write_bond_dims
                .iter()
                .copied()
                .skip(1)
                .take(mps.num_sites().saturating_sub(1))
                .collect(),
            center_before_qr,
            center_before_qr_is_valid,
            construction,
            projector_flip_sites: self.projector_flip_sites,
            projector_sign_sites: self.projector_sign_sites,
            projector_sites: self.projector_sites,
            projector_site_min,
            projector_site_max,
            projector_span,
            gauge_compensation_sites: self.gauge_compensation_sites,
            touched_site_min,
            touched_site_max,
            touched_sites: self.touched_sites.len(),
            changed_tensor_min,
            changed_tensor_max,
            changed_tensors: changed_tensors.len(),
            changed_bond_min,
            changed_bond_max,
            changed_bonds: changed_bonds.len(),
            qr_sites,
            qr_sites_skippable_by_locality,
            qr_sites_skippable_by_center_ceiling,
            post_projection_qr_wall_time_seconds: 0.0,
            compression_bonds_observed: 0,
            external_bonds: Vec::new(),
            external_discarded_weight: 0.0,
            normalization_preserved_center: None,
        }
    }
}

fn finish_projection_compression_telemetry(
    event: &mut super::ProjectionQrLocalityTelemetry,
    compression: &[BondCompressionTelemetry],
) {
    assert_eq!(
        compression.len(),
        event.chain_length.saturating_sub(1),
        "post-projection compression must report every internal bond"
    );
    event.compression_bonds_observed = compression.len();
    if event.construction != super::ProjectionConstruction::DirectSum {
        return;
    }

    let support_min = event
        .projector_site_min
        .expect("direct-sum projector must have support");
    let support_max = event
        .projector_site_max
        .expect("direct-sum projector must have support");
    for observed in compression {
        let pre_projection_rank = event.projection_entry_bond_profile[observed.bond - 1];
        let is_window_internal = support_min < observed.bond && observed.bond <= support_max;
        if !is_window_internal {
            event
                .external_bonds
                .push(super::ExternalProjectionBondTelemetry {
                    bond: observed.bond,
                    pre_projection_rank,
                    compression_input_rank: observed.input_rank,
                    post_compression_rank: observed.output_rank,
                    retained_rank_changed: observed.output_rank != pre_projection_rank,
                    discarded_weight: observed.discarded_weight,
                });
            event.external_discarded_weight += observed.discarded_weight;
        }
    }
    // Preserve the public telemetry convention that an empty/numerically-zero
    // discarded-weight sum is positive zero.
    event.external_discarded_weight += 0.0;
}

fn reduce_exact_projection_bonds_profiled(
    mps: &mut Mps,
    telemetry: &mut Option<&mut super::QueryDepthTelemetry>,
    locality: Option<ProjectionLocalityContext>,
) -> Result<(), MpsError> {
    if telemetry.is_none() {
        return reduce_exact_projection_bonds(mps);
    }
    let mut event = locality.map(|locality| locality.into_event(mps));
    let ((), qr_wall_time_seconds) = profile_query_phase_with_elapsed(
        mps,
        telemetry,
        super::QueryPhase::PostProjectionQr,
        Mps::right_canonicalize,
    );
    let mut compression = Vec::new();
    let result = if event.is_some() {
        profile_query_phase(
            mps,
            telemetry,
            super::QueryPhase::PostProjectionSvd,
            |mps| mps.compress_from_right_canonical_profiled(&mut compression),
        )
    } else {
        profile_query_phase(
            mps,
            telemetry,
            super::QueryPhase::PostProjectionSvd,
            Mps::compress_from_right_canonical,
        )
    };
    result?;
    if let Some(mut event) = event.take() {
        event.post_projection_qr_wall_time_seconds = qr_wall_time_seconds;
        finish_projection_compression_telemetry(&mut event, &compression);
        telemetry
            .as_deref_mut()
            .expect("profiled reduction requires query telemetry")
            .record_projection_qr_locality(event);
    }
    Ok(())
}

/// Shared implementation for tracked and phase-insensitive forced projection.
fn project_forced_z_with_update_impl(
    tableau: &mut SparseStabY,
    mps: &mut Mps,
    q_idx: usize,
    outcome: bool,
    mut phase_accumulator: Option<&mut crate::stab_mps::canonical_ket::CanonicalPhaseTracker>,
    mut telemetry: Option<&mut super::QueryDepthTelemetry>,
    sibling_pair_id: Option<u64>,
) -> Result<ForcedProjectionResult, MpsError> {
    let projection_locality_active = telemetry
        .as_deref()
        .is_some_and(super::QueryDepthTelemetry::projection_locality_active);
    let (pre_projection_norm_squared, probability) =
        profile_query_phase(mps, &mut telemetry, super::QueryPhase::Expectation, |mps| {
            let norm_squared = mps.norm_squared();
            assert!(
                norm_squared.is_finite(),
                "forced Z projection received a non-finite pre-projection norm"
            );
            assert!(norm_squared > 0.0, "cannot project a zero-norm MPS");
            let probability =
                z_outcome_probability(tableau, mps, q_idx, outcome, "forced Z projection");
            (norm_squared, probability)
        });
    let is_trivial =
        profile_query_phase(mps, &mut telemetry, super::QueryPhase::Bookkeeping, |mps| {
            is_mps_trivial(mps)
        });
    if is_trivial {
        // A trivial coefficient MPS represents a pure stabilizer state. First
        // absorb a possible nonzero virtual basis word into the tableau; only
        // then can its forced update supply the exact probability and state.
        let modified_sites = profile_pre_reduction(
            mps,
            &mut telemetry,
            q_idx,
            sibling_pair_id,
            |mps, diagnostics| match diagnostics {
                Some(diagnostics) => canonicalize_trivial_mps_basis_profiled(
                    tableau,
                    mps,
                    phase_accumulator.as_deref_mut(),
                    diagnostics,
                ),
                None => {
                    canonicalize_trivial_mps_basis(tableau, mps, phase_accumulator.as_deref_mut())
                }
            },
        );
        let decomp = profile_query_phase(
            mps,
            &mut telemetry,
            super::QueryPhase::Decomposition,
            |_| decompose_z(tableau.stabs(), tableau.destabs(), q_idx),
        );
        profile_query_phase(mps, &mut telemetry, super::QueryPhase::Projection, |_| {
            let tableau_probability: f64 = match decomp {
                ZDecomposition::Stabilizer { phase, .. } => {
                    if (phase.re < 0.0) == outcome {
                        1.0
                    } else {
                        0.0
                    }
                }
                ZDecomposition::DestabilizerFlip { .. } => 0.5,
            };
            debug_assert_eq!(probability.to_bits(), tableau_probability.to_bits());
            if probability > 0.0 {
                let before_measurement = phase_accumulator.as_ref().map(|_| tableau.clone());
                tableau.mz_forced(q_idx, outcome);
                if let (Some(accumulator), Some(before_measurement)) = (
                    phase_accumulator.as_deref_mut(),
                    before_measurement.as_ref(),
                ) {
                    accumulator.forced_measurement(
                        before_measurement,
                        tableau,
                        q_idx,
                        outcome,
                        probability,
                    );
                }
            }
        });
        let survival_ratio =
            profile_query_phase(mps, &mut telemetry, super::QueryPhase::Survival, |mps| {
                inject_projection_vanish_if_requested(mps);
                projection_survival_ratio(mps, pre_projection_norm_squared)
            });
        profile_query_phase(
            mps,
            &mut telemetry,
            super::QueryPhase::Normalization,
            |mps| {
                if probability > 0.0 && survival_ratio >= BRANCH_VANISH_SURVIVAL_THRESHOLD {
                    mps.normalize();
                }
            },
        );
        return profile_query_phase(mps, &mut telemetry, super::QueryPhase::Bookkeeping, |_| {
            Ok(ForcedProjectionResult {
                snapped_probability: inject_zero_projection_probability_if_requested(probability),
                survival_ratio,
                update: ProjectionUpdate {
                    collapsed_site: None,
                    modified_sites,
                },
            })
        });
    }

    // Reduce to one virtual X site while compensating every generator-basis
    // CNOT on the MPS. This preserves C·MPS exactly.
    let mut modified_sites = profile_pre_reduction(
        mps,
        &mut telemetry,
        q_idx,
        sibling_pair_id,
        |mps, diagnostics| match diagnostics {
            Some(diagnostics) => {
                pre_reduce_for_measurement_profiled(tableau, mps, q_idx, diagnostics)
            }
            None => pre_reduce_for_measurement(tableau, mps, q_idx, true),
        },
    )?;
    // Snapshot only after expectation and compensated pre-reduction. Bitwise
    // differences in the resulting event therefore belong to the projection
    // and its gauge compensation, rather than to earlier query phases.
    let mut projection_locality =
        projection_locality_active.then(|| ProjectionLocalityContext::new(mps));
    let decomposition = profile_query_phase(
        mps,
        &mut telemetry,
        super::QueryPhase::Decomposition,
        |_| decompose_z(tableau.stabs(), tableau.destabs(), q_idx),
    );

    match decomposition {
        ZDecomposition::Stabilizer { phase, sign_sites } => {
            if probability == 0.0 {
                let survival_ratio =
                    profile_query_phase(mps, &mut telemetry, super::QueryPhase::Survival, |mps| {
                        projection_survival_ratio(mps, pre_projection_norm_squared)
                    });
                profile_query_phase(
                    mps,
                    &mut telemetry,
                    super::QueryPhase::Normalization,
                    |_| {},
                );
                return profile_query_phase(
                    mps,
                    &mut telemetry,
                    super::QueryPhase::Bookkeeping,
                    |_| {
                        Ok(ForcedProjectionResult {
                            snapped_probability: 0.0,
                            survival_ratio,
                            update: ProjectionUpdate {
                                collapsed_site: None,
                                modified_sites,
                            },
                        })
                    },
                );
            }
            let construction = if sign_sites.is_empty() {
                super::ProjectionConstruction::ScalarScale
            } else {
                super::ProjectionConstruction::DirectSum
            };
            if let Some(locality) = projection_locality.as_mut() {
                locality.record_projection_write(mps, construction, &[], &sign_sites);
            }
            profile_query_phase(mps, &mut telemetry, super::QueryPhase::Projection, |mps| {
                apply_pauli_projection(
                    mps,
                    &[],
                    &sign_sites,
                    phase,
                    if outcome { -1.0 } else { 1.0 },
                    probability,
                );
            });
            if let Some(locality) = projection_locality.as_mut() {
                let mut touched_sites = modified_sites.clone();
                touched_sites.extend(sign_sites.iter().copied());
                locality.record_gauge_and_touched_sites(&[], touched_sites);
            }
            // The physical observable was already in the stabilizer span, so
            // mz_forced performs no Clifford-basis change. The projected
            // stabilizer-sign superposition remains encoded in the MPS.
            reduce_exact_projection_bonds_profiled(
                mps,
                &mut telemetry,
                projection_locality.take(),
            )?;
            let survival_ratio =
                profile_query_phase(mps, &mut telemetry, super::QueryPhase::Survival, |mps| {
                    inject_projection_vanish_if_requested(mps);
                    projection_survival_ratio(mps, pre_projection_norm_squared)
                });
            let normalization_center_before =
                projection_locality_active.then(|| mps.projection_diagnostic_center());
            profile_query_phase(
                mps,
                &mut telemetry,
                super::QueryPhase::Normalization,
                |mps| {
                    if survival_ratio >= BRANCH_VANISH_SURVIVAL_THRESHOLD {
                        mps.normalize();
                    }
                },
            );
            if let Some((center_before, valid_before)) = normalization_center_before {
                let (center_after, valid_after) = mps.projection_diagnostic_center();
                telemetry
                    .as_deref_mut()
                    .expect("locality diagnostics require query telemetry")
                    .record_projection_normalization(
                        valid_before && valid_after && center_before == center_after,
                    );
            }
            profile_query_phase(mps, &mut telemetry, super::QueryPhase::Bookkeeping, |_| {
                modified_sites.extend(sign_sites);
                modified_sites.sort_unstable();
                modified_sites.dedup();
                Ok(ForcedProjectionResult {
                    snapped_probability: inject_zero_projection_probability_if_requested(
                        probability,
                    ),
                    survival_ratio,
                    update: ProjectionUpdate {
                        collapsed_site: None,
                        modified_sites,
                    },
                })
            })
        }
        ZDecomposition::DestabilizerFlip {
            flip_sites,
            phase,
            sign_sites,
        } => {
            debug_assert_eq!(
                flip_sites.len(),
                1,
                "forced projection must have one flip after pre-reduction"
            );
            if probability == 0.0 {
                let survival_ratio =
                    profile_query_phase(mps, &mut telemetry, super::QueryPhase::Survival, |mps| {
                        projection_survival_ratio(mps, pre_projection_norm_squared)
                    });
                profile_query_phase(
                    mps,
                    &mut telemetry,
                    super::QueryPhase::Normalization,
                    |_| {},
                );
                return profile_query_phase(
                    mps,
                    &mut telemetry,
                    super::QueryPhase::Bookkeeping,
                    |_| {
                        Ok(ForcedProjectionResult {
                            snapped_probability: 0.0,
                            survival_ratio,
                            update: ProjectionUpdate {
                                collapsed_site: None,
                                modified_sites,
                            },
                        })
                    },
                );
            }
            let is_local_projection = sign_sites.is_empty();
            let construction = if is_local_projection {
                super::ProjectionConstruction::LocalBlockWrite
            } else {
                super::ProjectionConstruction::DirectSum
            };
            if let Some(locality) = projection_locality.as_mut() {
                locality.record_projection_write(mps, construction, &flip_sites, &sign_sites);
            }
            let gauge_sites =
                profile_query_phase(mps, &mut telemetry, super::QueryPhase::Projection, |mps| {
                    let sign_f = if outcome { -1.0 } else { 1.0 };
                    if is_local_projection {
                        project_single_flip_without_sign(
                            mps,
                            flip_sites[0],
                            Complex64::new(sign_f, 0.0) * phase,
                            probability,
                        );
                    } else {
                        apply_pauli_projection(
                            mps,
                            &flip_sites,
                            &sign_sites,
                            phase,
                            sign_f,
                            probability,
                        );
                    }
                    if !is_local_projection {
                        collapse_projected_flip_site(mps, flip_sites[0]);
                    }

                    assert!(
                        tableau.tracks_destab_signs(),
                        "exact forced projection requires destabilizer-sign tracking"
                    );
                    let mut predicted_tableau = tableau.clone();
                    right_compose_measurement_basis_rotation(
                        &mut predicted_tableau,
                        flip_sites[0],
                        phase,
                        &sign_sites,
                        outcome,
                        phase_accumulator,
                    );

                    let result = tableau.mz_forced(q_idx, outcome);
                    debug_assert_eq!(result.outcome, outcome);
                    compensate_measurement_pauli_gauge(mps, &predicted_tableau, tableau)
                });
            if let Some(locality) = projection_locality.as_mut() {
                let mut touched_sites = modified_sites.clone();
                touched_sites.extend(flip_sites.iter().copied());
                touched_sites.extend(sign_sites.iter().copied());
                touched_sites.extend(gauge_sites.iter().copied());
                locality.record_gauge_and_touched_sites(&gauge_sites, touched_sites);
            }
            // `mz_forced` produces the same projected stabilizer group as the
            // predicted basis rotation; compensation changes only the
            // destabilizer gauge. Therefore the post-H canonical ket cached
            // by the phase tracker remains valid for the measured tableau and
            // can be the before-ket of the next scalar site.
            reduce_exact_projection_bonds_profiled(
                mps,
                &mut telemetry,
                projection_locality.take(),
            )?;
            let survival_ratio =
                profile_query_phase(mps, &mut telemetry, super::QueryPhase::Survival, |mps| {
                    inject_projection_vanish_if_requested(mps);
                    projection_survival_ratio(mps, pre_projection_norm_squared)
                });
            let normalization_center_before =
                projection_locality_active.then(|| mps.projection_diagnostic_center());
            profile_query_phase(
                mps,
                &mut telemetry,
                super::QueryPhase::Normalization,
                |mps| {
                    if survival_ratio >= BRANCH_VANISH_SURVIVAL_THRESHOLD {
                        mps.normalize();
                    }
                },
            );
            if let Some((center_before, valid_before)) = normalization_center_before {
                let (center_after, valid_after) = mps.projection_diagnostic_center();
                telemetry
                    .as_deref_mut()
                    .expect("locality diagnostics require query telemetry")
                    .record_projection_normalization(
                        valid_before && valid_after && center_before == center_after,
                    );
            }
            profile_query_phase(mps, &mut telemetry, super::QueryPhase::Bookkeeping, |_| {
                modified_sites.extend(flip_sites.iter().copied());
                modified_sites.extend(sign_sites);
                modified_sites.extend(gauge_sites);
                modified_sites.sort_unstable();
                modified_sites.dedup();
                Ok(ForcedProjectionResult {
                    snapped_probability: inject_zero_projection_probability_if_requested(
                        probability,
                    ),
                    survival_ratio,
                    update: ProjectionUpdate {
                        collapsed_site: Some(flip_sites[0]),
                        modified_sites,
                    },
                })
            })
        }
    }
}

/// Project qubit `q_idx` onto a forced Z-basis outcome and return its
/// probability together with the affected virtual coefficient-MPS sites.
///
/// Mirrors `measure_qubit_stab_mps_pragmatic` but is deterministic: the caller supplies
/// the outcome. This is the phase-insensitive Liu-Clark 2412.17209 Algorithm 3
/// / VI.A path used by probability and measurement callers.
///
/// # Errors
///
/// Returns an [`MpsError`] if compensation or compression fails.
pub(super) fn project_forced_z_with_update(
    tableau: &mut SparseStabY,
    mps: &mut Mps,
    q_idx: usize,
    outcome: bool,
) -> Result<ForcedProjectionResult, MpsError> {
    project_forced_z_with_update_impl(tableau, mps, q_idx, outcome, None, None, None)
}

/// Forced projection with canonical-ket scalar tracking for phase-sensitive
/// amplitude reconstruction. Measurement and probability paths deliberately
/// use the untracked sibling above.
pub(super) fn project_forced_z_with_phase(
    tableau: &mut SparseStabY,
    mps: &mut Mps,
    q_idx: usize,
    outcome: bool,
    phase_accumulator: &mut crate::stab_mps::canonical_ket::CanonicalPhaseTracker,
) -> Result<f64, MpsError> {
    Ok(project_forced_z_with_update_impl(
        tableau,
        mps,
        q_idx,
        outcome,
        Some(phase_accumulator),
        None,
        None,
    )?
    .snapped_probability)
}

/// Project qubit `q_idx` onto a forced Z-basis outcome and return the
/// probability of that outcome given the current state.
///
/// # Errors
///
/// Returns an [`MpsError`] if an MPS operation fails.
pub fn project_forced_z(
    tableau: &mut SparseStabY,
    mps: &mut Mps,
    q_idx: usize,
    outcome: bool,
) -> Result<f64, MpsError> {
    Ok(project_forced_z_with_update(tableau, mps, q_idx, outcome)?.snapped_probability)
}

/// Profiled sibling of [`project_forced_z`] used by batched query telemetry.
pub(super) fn project_forced_z_profiled(
    tableau: &mut SparseStabY,
    mps: &mut Mps,
    q_idx: usize,
    outcome: bool,
    telemetry: &mut super::QueryDepthTelemetry,
    sibling_pair_id: Option<u64>,
) -> Result<f64, MpsError> {
    Ok(project_forced_z_with_update_impl(
        tableau,
        mps,
        q_idx,
        outcome,
        None,
        Some(telemetry),
        sibling_pair_id,
    )?
    .snapped_probability)
}

/// Measure qubit `q_idx` in the Z basis using the STN protocol.
///
/// Uses the tableau for structure (stabilizer/destabilizer decomposition)
/// and the MPS for probability computation and projection.
/// Lazy-compensation measurement (V2): accumulates `pre_reduce` CNOTs AND
/// the post-projection `W⁻¹` (single-qubit H + diagonal CZs) into a
/// `DeferredOp` queue. Uses `V†`-conjugated Pauli for projection. State
/// invariant: `effective = C_tableau · V_deferred · stored_mps`.
///
/// Derivation:
/// - After `pre_reduce` row ops, tableau's C -> C*A (A = product of CNOTs).
///   Push each CNOT to V: `V_new = A * V_old` (left-multiply).
/// - After projection `(I + sp*P)/2` in effective frame, stored MPS is
///   projected via conjugated `Q = V^dag * P * V`: `stored' = (I+sp*Q)/2*stored`.
/// - `mz_forced` updates tableau: C*A -> C*A*W where `W*Z_id*W^dag = P`.
///   To preserve `effective = C_tableau * V * stored`, absorb `W^-1` into
///   V: `V_new = W^-1 * V` (append `W^-1`'s primitives at end of queue).
/// - For single-flip `P = X_id * Z_{sign}`, `W = CZ(id, s_1)*...*CZ(id, s_k)*H_id`
///   and `W^-1 = H_id * CZ(id, s_1)*...*CZ(id, s_k)`. All cheap primitives
///   (single-site H, diagonal CZ).
///
/// # Panics
///
/// Panics if the tableau measurement iterator is empty (should not happen).
pub(super) fn measure_qubit_stab_mps_lazy_with_update(
    tableau: &mut SparseStabY,
    mps: &mut Mps,
    rng: &mut PecosRng,
    q_idx: usize,
    deferred: &mut Vec<DeferredOp>,
) -> Result<LiveMeasurementResult, MpsError> {
    if is_mps_trivial(mps) {
        // A tableau-only read requires both a zero coefficient-basis word and
        // an identity virtual frame. Materialize a pending frame first: it can
        // turn a computational product MPS into a nontrivial coefficient
        // state, in which case the general Lazy projection must handle it.
        if !deferred.is_empty() {
            flush_deferred_ops(mps, deferred)?;
        }
        if is_mps_trivial(mps) {
            return Ok(measure_trivial_mps_with_update(tableau, mps, q_idx));
        }
    }

    // Push pre_reduce CNOTs to deferred instead of applying eagerly.
    let mut modified_sites = Vec::new();
    {
        let col_x = &tableau.stabs().col_x[q_idx];
        if col_x.len() > 1 {
            let replaced_idx = find_replaced_stabilizer(tableau, q_idx);
            let n = tableau.num_qubits();
            let anticom: Vec<usize> = tableau.stabs().col_x[q_idx]
                .iter()
                .filter(|&id| id != replaced_idx)
                .collect();
            let stabs_snapshot = tableau.stabs().clone();
            let destabs_snapshot = tableau.destabs().clone();
            modified_sites.push(replaced_idx);
            for other_id in anticom {
                crate::stab_mps::tableau_compose::multiply_row(
                    tableau.stabs_mut(),
                    other_id,
                    &stabs_snapshot,
                    replaced_idx,
                    n,
                );
                crate::stab_mps::tableau_compose::multiply_row(
                    tableau.destabs_mut(),
                    replaced_idx,
                    &destabs_snapshot,
                    other_id,
                    n,
                );
                deferred.push(DeferredOp::Cnot(replaced_idx, other_id));
                modified_sites.push(other_id);
            }
        }
    }
    modified_sites.sort_unstable();
    modified_sites.dedup();

    let decomp = decompose_z(tableau.stabs(), tableau.destabs(), q_idx);
    match decomp {
        ZDecomposition::Stabilizer { phase, sign_sites } => {
            let mut flip_conj: Vec<usize> = Vec::new();
            let mut sign_conj: Vec<usize> = sign_sites;
            let mut phase_conj = phase;
            conjugate_pauli_by_deferred_ops(
                &mut flip_conj,
                &mut sign_conj,
                &mut phase_conj,
                deferred,
            );

            let ev = pauli_expectation(mps, &flip_conj, &sign_conj, phase_conj).re;

            if sign_conj.is_empty() && flip_conj.is_empty() {
                let outcome = phase_conj.re < 0.0;
                tableau.mz_forced(q_idx, outcome);
                return Ok(LiveMeasurementResult {
                    measurement: MeasurementResult {
                        outcome,
                        is_deterministic: true,
                    },
                    update: ProjectionUpdate {
                        collapsed_site: None,
                        modified_sites,
                    },
                });
            }
            let prob_plus = f64::midpoint(1.0, ev).clamp(0.0, 1.0);
            let is_determ = (ev.abs() - 1.0).abs() < 1e-6;
            let outcome = if is_determ {
                ev < 0.0
            } else {
                rng.random_bool(1.0 - prob_plus)
            };
            let sign_f = if outcome { -1.0 } else { 1.0 };
            let prob = if outcome { 1.0 - prob_plus } else { prob_plus };
            apply_pauli_projection(mps, &flip_conj, &sign_conj, phase_conj, sign_f, prob);
            mps.compress()?;
            modified_sites.extend(flip_conj);
            modified_sites.extend(sign_conj);
            modified_sites.sort_unstable();
            modified_sites.dedup();
            Ok(LiveMeasurementResult {
                measurement: MeasurementResult {
                    outcome,
                    is_deterministic: is_determ,
                },
                update: ProjectionUpdate {
                    collapsed_site: None,
                    modified_sites,
                },
            })
        }
        ZDecomposition::DestabilizerFlip {
            flip_sites,
            phase,
            sign_sites,
        } => {
            // Pre_reduce ensures flip_sites.len() == 1. Let id = flip_sites[0].
            // Mz_forced will transform tableau as C → C·W where
            //   W · Z_id · W† = X_id · Z_{sign_sites}
            //   (the decomposition's Pauli content, phase absorbed in sp).
            // Valid W: CZ(id, s_1)·...·CZ(id, s_k) · H_id.
            // To preserve invariant, V_new = W⁻¹ · V_old. W⁻¹ = H_id · CZ_chain
            // (reversed product with self-adjoint primitives).
            let id = if flip_sites.len() == 1 {
                flip_sites[0]
            } else {
                // Shouldn't happen after pre_reduce; use first as fallback.
                debug_assert!(
                    !flip_sites.is_empty(),
                    "lazy measure: flip_sites empty in DestabilizerFlip"
                );
                flip_sites[0]
            };

            // Conjugate the PRE-basis-rotation Pauli by existing V†.
            let mut flip_conj: Vec<usize> = flip_sites.clone();
            let mut sign_conj: Vec<usize> = sign_sites.clone();
            let mut phase_conj = phase;
            conjugate_pauli_by_deferred_ops(
                &mut flip_conj,
                &mut sign_conj,
                &mut phase_conj,
                deferred,
            );

            let ev = pauli_expectation(mps, &flip_conj, &sign_conj, phase_conj).re;
            let prob_plus = f64::midpoint(1.0, ev).clamp(0.0, 1.0);
            let outcome = rng.random_bool(1.0 - prob_plus);
            let sign_f = if outcome { -1.0 } else { 1.0 };
            let prob = if outcome { 1.0 - prob_plus } else { prob_plus };

            // Project stored MPS via conjugated Pauli.
            apply_pauli_projection(mps, &flip_conj, &sign_conj, phase_conj, sign_f, prob);
            mps.compress()?;
            // Use the original decomposition phase here: deferred conjugation
            // changes the stored-MPS projector, but the tableau update acts in
            // the current post-pre-reduction generator basis. The helper also
            // queues the virtual Z gauge omitted by `mz_forced`'s structural
            // destabilizer-row XORs.
            let gauge_sites = enqueue_lazy_measurement_basis_update(
                tableau,
                deferred,
                q_idx,
                id,
                phase,
                &sign_sites,
                outcome,
            );
            modified_sites.extend(flip_conj);
            modified_sites.extend(sign_conj);
            modified_sites.extend(gauge_sites);
            modified_sites.push(id);
            modified_sites.sort_unstable();
            modified_sites.dedup();
            Ok(LiveMeasurementResult {
                measurement: MeasurementResult {
                    outcome,
                    is_deterministic: false,
                },
                update: ProjectionUpdate {
                    // The queued W⁻¹ makes `id` equal |0> only in the
                    // effective `V * stored_mps` frame. Disentangling flags
                    // describe the stored tensors, where the conjugated
                    // projection modified `id`, so no stored-frame collapse
                    // proof may be installed here.
                    collapsed_site: None,
                    modified_sites,
                },
            })
        }
    }
}

/// Measure `q_idx` while accumulating virtual Clifford compensation in
/// `deferred`.
///
/// # Errors
///
/// Returns an [`MpsError`] if an MPS operation fails.
pub fn measure_qubit_stab_mps_lazy(
    tableau: &mut SparseStabY,
    mps: &mut Mps,
    rng: &mut PecosRng,
    q_idx: usize,
    deferred: &mut Vec<DeferredOp>,
) -> Result<MeasurementResult, MpsError> {
    Ok(measure_qubit_stab_mps_lazy_with_update(tableau, mps, rng, q_idx, deferred)?.measurement)
}

/// Apply projection `(I + sign_f · phase · X_flip · Z_sign) / 2` to `mps`,
/// normalized by `1/√prob`. Uses MPS addition; no site-collapse step
/// (caller is responsible for collapse if exact state needed).
fn apply_pauli_projection(
    mps: &mut Mps,
    flip_sites: &[usize],
    sign_sites: &[usize],
    phase: Complex64,
    sign_f: f64,
    prob: f64,
) {
    let x_gate = DMatrix::from_row_slice(
        2,
        2,
        &[
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    );
    let z_diag = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
    let denom = Complex64::new(2.0 * prob.max(1e-20).sqrt(), 0.0);
    if flip_sites.is_empty() && sign_sites.is_empty() {
        mps.scale(Complex64::new(1.0, 0.0) + Complex64::new(sign_f, 0.0) * phase);
        mps.scale(Complex64::new(1.0, 0.0) / denom);
        return;
    }
    let mut mps_z = mps.clone();
    for &k in sign_sites {
        mps_z
            .apply_diagonal_one_site(k, &z_diag)
            .expect("MPS op on valid site");
    }
    for &j in flip_sites {
        mps_z
            .apply_one_site_gate(j, &x_gate)
            .expect("MPS op on valid site");
    }
    mps_z.scale(Complex64::new(sign_f, 0.0) * phase / denom);
    mps.scale(Complex64::new(1.0, 0.0) / denom);
    *mps = mps.add(&mps_z);
}

/// Measure qubit `q_idx` in the Z basis using the STN protocol.
///
/// # Panics
///
/// Panics if the tableau measurement iterator is empty (should not happen).
pub(super) fn measure_qubit_stab_mps_with_update(
    tableau: &mut SparseStabY,
    mps: &mut Mps,
    rng: &mut PecosRng,
    q_idx: usize,
) -> Result<LiveMeasurementResult, MpsError> {
    if is_mps_trivial(mps) {
        return Ok(measure_trivial_mps_with_update(tableau, mps, q_idx));
    }

    // Pre-reduce the tableau so that Z_q has at most one anticommuting stabilizer.
    // This avoids the problematic multi-flip projection path.
    //
    // MPS compensation is intentionally SKIPPED here (`false`). Random
    // measurement doesn't require exact (tableau, mps) consistency — the
    // sampled outcome statistics and subsequent measurement stats remain
    // self-consistent (same row ops happen in both forward and reverse
    // comparisons). Compensation would trigger O(N) long-range CNOTs per
    // measurement (SWAP chain -> exponential bond growth in MAST's
    // measurement-heavy workload). Exact-state paths
    // `project_forced_z` passes `true`.
    let _modified_sites = pre_reduce_for_measurement(tableau, mps, q_idx, false)?;

    // Compute the expectation value <Z_q>
    let ev = z_expectation_value(tableau, mps, q_idx).re;

    let decomp = decompose_z(tableau.stabs(), tableau.destabs(), q_idx);

    match decomp {
        ZDecomposition::Stabilizer { phase, sign_sites } => {
            // Z_q is in the stabilizer group: measurement is deterministic.
            if sign_sites.is_empty() {
                let outcome = phase.re < 0.0;
                tableau.mz_forced(q_idx, outcome);
                return Ok(LiveMeasurementResult {
                    measurement: MeasurementResult {
                        outcome,
                        is_deterministic: true,
                    },
                    update: ProjectionUpdate::default(),
                });
            }
            let prob_plus = f64::midpoint(1.0, ev).clamp(0.0, 1.0);

            // Check if measurement is deterministic (ev ≈ ±1)
            let is_determ = (ev.abs() - 1.0).abs() < 1e-6;
            let outcome = if is_determ {
                ev < 0.0
            } else {
                rng.random_bool(1.0 - prob_plus)
            };

            let sign_f = if outcome { -1.0 } else { 1.0 };
            let prob = if outcome { 1.0 - prob_plus } else { prob_plus };

            let z_diag = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
            let mut mps_z = mps.clone();
            for &k in &sign_sites {
                mps_z.apply_diagonal_one_site(k, &z_diag)?;
            }
            mps_z.scale(
                Complex64::new(sign_f, 0.0) * phase
                    / Complex64::new(2.0 * prob.max(1e-20).sqrt(), 0.0),
            );
            mps.scale(Complex64::new(1.0 / (2.0 * prob.max(1e-20).sqrt()), 0.0));
            *mps = mps.add(&mps_z);
            mps.compress()?;

            tableau.mz_forced(q_idx, outcome);
            let mut modified_sites = sign_sites;
            modified_sites.sort_unstable();
            modified_sites.dedup();
            Ok(LiveMeasurementResult {
                measurement: MeasurementResult {
                    outcome,
                    is_deterministic: is_determ,
                },
                update: ProjectionUpdate {
                    collapsed_site: None,
                    modified_sites,
                },
            })
        }

        ZDecomposition::DestabilizerFlip {
            flip_sites,
            phase,
            sign_sites,
        } => {
            let prob_plus = f64::midpoint(1.0, ev).clamp(0.0, 1.0);
            let outcome = rng.random_bool(1.0 - prob_plus);
            let prob = if outcome { 1.0 - prob_plus } else { prob_plus };

            if flip_sites.len() == 1 && sign_sites.is_empty() {
                // Single flip at site k. Project to eigenstate of phase*X_k.
                // After mz_forced: the projected state always goes to σ=0,
                // because mz_forced encodes the outcome in the stabilizer sign.
                let k = flip_sites[0];
                let chi_r = mps.bond_dim(k + 1);
                let sign_f = if outcome { -1.0 } else { 1.0 };
                let sp = Complex64::new(sign_f, 0.0) * phase;

                let block_0 = crate::mps::tensor::phys_block(&mps.tensors()[k], 0, chi_r);
                let block_1 = crate::mps::tensor::phys_block(&mps.tensors()[k], 1, chi_r);
                let projected = (&block_0 + &block_1 * sp)
                    / Complex64::new((2.0 * prob).max(1e-20).sqrt(), 0.0);

                let zero = DMatrix::zeros(mps.tensors()[k].nrows(), chi_r);
                mps.set_physical_block(k, 0, &projected);
                mps.set_physical_block(k, 1, &zero);
                mps.normalize();
            } else {
                // Multi-site case with sign_sites: use MPS addition then collapse flip site.
                let sign_f = if outcome { -1.0 } else { 1.0 };
                let x_gate = DMatrix::from_row_slice(
                    2,
                    2,
                    &[
                        Complex64::new(0.0, 0.0),
                        Complex64::new(1.0, 0.0),
                        Complex64::new(1.0, 0.0),
                        Complex64::new(0.0, 0.0),
                    ],
                );
                let z_diag = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];

                let mut mps_z = mps.clone();
                // Apply Z first, then X (order must match z_expectation_value).
                for &k in &sign_sites {
                    mps_z.apply_diagonal_one_site(k, &z_diag)?;
                }
                for &j in &flip_sites {
                    mps_z.apply_one_site_gate(j, &x_gate)?;
                }
                mps_z.scale(
                    Complex64::new(sign_f, 0.0) * phase
                        / Complex64::new(2.0 * prob.max(1e-20).sqrt(), 0.0),
                );
                mps.scale(Complex64::new(1.0 / (2.0 * prob.max(1e-20).sqrt()), 0.0));
                *mps = mps.add(&mps_z);
                mps.compress()?;

                // Collapse the flip site to σ=0. After the MPS addition projector,
                // block_1 = sp * block_0 (eigenstate condition). After mz_forced,
                // σ=0 is the stabilizer eigenstate. Just zero out σ=1 and renormalize.
                if flip_sites.len() == 1 {
                    let k = flip_sites[0];
                    let chi_r = mps.bond_dim(k + 1);
                    let zero = DMatrix::zeros(mps.tensors()[k].nrows(), chi_r);
                    mps.set_physical_block(k, 1, &zero);
                }

                mps.normalize();
            }

            tableau.mz_forced(q_idx, outcome);
            let collapsed_site = (flip_sites.len() == 1).then_some(flip_sites[0]);
            let mut modified_sites = flip_sites;
            modified_sites.extend(sign_sites);
            modified_sites.sort_unstable();
            modified_sites.dedup();
            Ok(LiveMeasurementResult {
                measurement: MeasurementResult {
                    outcome,
                    is_deterministic: false,
                },
                update: ProjectionUpdate {
                    collapsed_site,
                    modified_sites,
                },
            })
        }
    }
}

/// Measure qubit `q_idx` with the pragmatic eager STN protocol.
///
/// This path deliberately skips coefficient-MPS compensation when tableau
/// generator pre-reduction is needed. Its outcome stream remains internally
/// useful for throughput-oriented workloads, but the stored conditional state
/// is biased and cannot support exact continuation or amplitude reads.
///
/// # Errors
///
/// Returns an [`MpsError`] if an MPS operation fails.
pub fn measure_qubit_stab_mps_pragmatic(
    tableau: &mut SparseStabY,
    mps: &mut Mps,
    rng: &mut PecosRng,
    q_idx: usize,
) -> Result<MeasurementResult, MpsError> {
    Ok(measure_qubit_stab_mps_with_update(tableau, mps, rng, q_idx)?.measurement)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mps::MpsConfig;

    fn sort_dedup(v: &mut Vec<usize>) {
        v.sort_unstable();
        v.dedup();
    }

    fn dense_pauli(flip_mask: usize, sign_mask: usize, phase: Complex64) -> DMatrix<Complex64> {
        let identity = DMatrix::identity(2, 2);
        let x = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        );
        let z = DMatrix::from_diagonal(&nalgebra::DVector::from_row_slice(&[
            Complex64::new(1.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ]));
        let factors: Vec<_> = (0..2)
            .map(
                |q| match (flip_mask & (1 << q) != 0, sign_mask & (1 << q) != 0) {
                    (false, false) => identity.clone(),
                    (true, false) => x.clone(),
                    (false, true) => z.clone(),
                    (true, true) => &x * &z,
                },
            )
            .collect();
        factors[0].kronecker(&factors[1]).map(|entry| phase * entry)
    }

    fn dense_single_qubit_gate(gate: &DMatrix<Complex64>, q: usize) -> DMatrix<Complex64> {
        let identity = DMatrix::identity(2, 2);
        if q == 0 {
            gate.kronecker(&identity)
        } else {
            identity.kronecker(gate)
        }
    }

    fn identify_dense_pauli(matrix: &DMatrix<Complex64>) -> (usize, usize, Complex64) {
        let phases = [
            Complex64::new(1.0, 0.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, -1.0),
        ];
        let mut matches = Vec::new();
        for flip_mask in 0..4 {
            for sign_mask in 0..4 {
                for phase in phases {
                    if (dense_pauli(flip_mask, sign_mask, phase) - matrix).norm() < 1e-12 {
                        matches.push((flip_mask, sign_mask, phase));
                    }
                }
            }
        }
        assert_eq!(matches.len(), 1, "dense oracle must identify one Pauli");
        matches[0]
    }

    #[test]
    fn conjugate_pauli_by_deferred_ops_matches_dense_oracle_for_every_op_kind() {
        let r = std::f64::consts::FRAC_1_SQRT_2;
        let h = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(r, 0.0),
                Complex64::new(r, 0.0),
                Complex64::new(r, 0.0),
                Complex64::new(-r, 0.0),
            ],
        );
        let z = DMatrix::from_diagonal(&nalgebra::DVector::from_row_slice(&[
            Complex64::new(1.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ]));
        let s = DMatrix::from_diagonal(&nalgebra::DVector::from_row_slice(&[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
        ]));
        let sdg = DMatrix::from_diagonal(&nalgebra::DVector::from_row_slice(&[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, -1.0),
        ]));
        let cnot_01 = DMatrix::from_row_slice(
            4,
            4,
            &[
                1.0, 0.0, 0.0, 0.0, // |00> -> |00>
                0.0, 1.0, 0.0, 0.0, // |01> -> |01>
                0.0, 0.0, 0.0, 1.0, // |11> -> |10>
                0.0, 0.0, 1.0, 0.0, // |10> -> |11>
            ],
        )
        .map(|entry| Complex64::new(entry, 0.0));
        let cnot_10 = DMatrix::from_row_slice(
            4,
            4,
            &[
                1.0, 0.0, 0.0, 0.0, // |00> -> |00>
                0.0, 0.0, 0.0, 1.0, // |11> -> |01>
                0.0, 0.0, 1.0, 0.0, // |10> -> |10>
                0.0, 1.0, 0.0, 0.0, // |01> -> |11>
            ],
        )
        .map(|entry| Complex64::new(entry, 0.0));
        let cz = DMatrix::from_diagonal(&nalgebra::DVector::from_row_slice(&[
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ]));
        let cases = vec![
            ("Cnot(0,1)", DeferredOp::Cnot(0, 1), cnot_01),
            ("Cnot(1,0)", DeferredOp::Cnot(1, 0), cnot_10),
            ("H(0)", DeferredOp::H(0), dense_single_qubit_gate(&h, 0)),
            ("H(1)", DeferredOp::H(1), dense_single_qubit_gate(&h, 1)),
            ("Z(0)", DeferredOp::Z(0), dense_single_qubit_gate(&z, 0)),
            ("Z(1)", DeferredOp::Z(1), dense_single_qubit_gate(&z, 1)),
            (
                "SZdg(0)",
                DeferredOp::SZdg(0),
                dense_single_qubit_gate(&sdg, 0),
            ),
            (
                "SZdg(1)",
                DeferredOp::SZdg(1),
                dense_single_qubit_gate(&sdg, 1),
            ),
            ("SZ(0)", DeferredOp::SZ(0), dense_single_qubit_gate(&s, 0)),
            ("SZ(1)", DeferredOp::SZ(1), dense_single_qubit_gate(&s, 1)),
            ("Cz(0,1)", DeferredOp::Cz(0, 1), cz),
        ];
        let input_phases = [
            Complex64::new(1.0, 0.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, -1.0),
        ];

        for (name, op, gate) in cases {
            let mut mismatches = Vec::new();
            for input_flip_mask in 0..4 {
                for input_sign_mask in 0..4 {
                    for input_phase in input_phases {
                        let input = dense_pauli(input_flip_mask, input_sign_mask, input_phase);
                        let oracle = gate.adjoint() * input * &gate;
                        let expected = identify_dense_pauli(&oracle);
                        let mut flip_sites = (0..2)
                            .filter(|q| input_flip_mask & (1 << q) != 0)
                            .collect::<Vec<_>>();
                        let mut sign_sites = (0..2)
                            .filter(|q| input_sign_mask & (1 << q) != 0)
                            .collect::<Vec<_>>();
                        let mut actual_phase = input_phase;
                        conjugate_pauli_by_deferred_ops(
                            &mut flip_sites,
                            &mut sign_sites,
                            &mut actual_phase,
                            std::slice::from_ref(&op),
                        );
                        sort_dedup(&mut flip_sites);
                        sort_dedup(&mut sign_sites);
                        let actual = (
                            flip_sites.iter().fold(0, |mask, q| mask | (1 << q)),
                            sign_sites.iter().fold(0, |mask, q| mask | (1 << q)),
                            actual_phase,
                        );
                        if actual != expected {
                            mismatches.push(format!(
                                "input=(X={input_flip_mask:02b}, Z={input_sign_mask:02b}, phase={input_phase}); expected={expected:?}, actual={actual:?}"
                            ));
                        }
                    }
                }
            }
            assert!(
                mismatches.is_empty(),
                "{name} had {} dense-oracle mismatches; first: {}",
                mismatches.len(),
                mismatches.first().map_or("none", String::as_str)
            );
        }
    }

    #[test]
    fn lazy_measurement_basis_update_matches_dense_oracle_for_all_two_qubit_cliffords() {
        use crate::stab_mps::StabMps;
        use pecos_core::QubitId;
        use std::collections::{HashSet, VecDeque};

        fn tableau_key(tableau: &SparseStabY) -> Vec<u8> {
            let mut key = Vec::new();
            for generators in [tableau.stabs(), tableau.destabs()] {
                for row in 0..tableau.num_qubits() {
                    for qubit in 0..tableau.num_qubits() {
                        key.push(u8::from(generators.row_x[row].contains(qubit)));
                        key.push(u8::from(generators.row_z[row].contains(qubit)));
                    }
                    key.push(u8::from(generators.signs_minus.contains(row)));
                    key.push(u8::from(generators.signs_i.contains(row)));
                }
            }
            key
        }

        fn phase_bucket(phase: Complex64) -> usize {
            if phase.re > 0.5 {
                0
            } else if phase.im > 0.5 {
                1
            } else if phase.re < -0.5 {
                2
            } else if phase.im < -0.5 {
                3
            } else {
                panic!("phase is not a fourth root of unity: {phase}");
            }
        }

        fn generic_coefficient_mps() -> Mps {
            let mut mps = Mps::new(2, MpsConfig::default());
            for (qubit, probability_one, argument) in [(0, 0.37_f64, 0.29_f64), (1, 0.61, -0.47)] {
                let zero = (1.0 - probability_one).sqrt();
                let one = Complex64::from_polar(probability_one.sqrt(), argument);
                let gate = DMatrix::from_row_slice(
                    2,
                    2,
                    &[
                        Complex64::new(zero, 0.0),
                        -one.conj(),
                        one,
                        Complex64::new(zero, 0.0),
                    ],
                );
                mps.apply_one_site_gate(qubit, &gate).unwrap();
            }
            mps
        }

        fn represented_state(
            template: &StabMps,
            tableau: &SparseStabY,
            mps: &Mps,
        ) -> Vec<Complex64> {
            let mut snapshot = template.clone();
            snapshot.tableau = tableau.clone();
            snapshot.mps = mps.clone();
            snapshot.deferred_ops.clear();
            snapshot.state_vector()
        }

        let coefficient_mps = generic_coefficient_mps();
        let template = StabMps::new(2);
        let initial = SparseStabY::with_seed(2, 0).with_destab_sign_tracking();
        let mut seen = HashSet::from([tableau_key(&initial)]);
        let mut pending = VecDeque::from([initial]);
        let mut cliffords = 0_usize;
        let mut cases = 0_usize;
        let mut signed_phase_counts = [0_usize; 4];

        while let Some(tableau) = pending.pop_front() {
            cliffords += 1;
            let before = represented_state(&template, &tableau, &coefficient_mps);
            for measured_qubit in 0..2 {
                // This is exactly the input domain of the post-pre-reduction
                // lazy basis update: one virtual X site remains.
                if tableau.stabs().col_x[measured_qubit].len() != 1 {
                    continue;
                }
                let ZDecomposition::DestabilizerFlip {
                    flip_sites,
                    phase,
                    sign_sites,
                } = decompose_z(tableau.stabs(), tableau.destabs(), measured_qubit)
                else {
                    continue;
                };
                if flip_sites.len() != 1 {
                    continue;
                }

                for outcome in [false, true] {
                    signed_phase_counts[phase_bucket(
                        Complex64::new(if outcome { -1.0 } else { 1.0 }, 0.0) * phase,
                    )] += 1;
                    let mut measured = tableau.clone();
                    let mut deferred = Vec::new();
                    enqueue_lazy_measurement_basis_update(
                        &mut measured,
                        &mut deferred,
                        measured_qubit,
                        flip_sites[0],
                        phase,
                        &sign_sites,
                        outcome,
                    );
                    let queued = deferred.clone();
                    let mut transformed_mps = coefficient_mps.clone();
                    flush_deferred_ops(&mut transformed_mps, &mut deferred).unwrap();
                    let after = represented_state(&template, &measured, &transformed_mps);
                    let overlap = before
                        .iter()
                        .zip(&after)
                        .map(|(left, right)| left.conj() * right)
                        .sum::<Complex64>();
                    let fidelity = overlap.norm_sqr();
                    assert!(
                        fidelity >= 1.0 - 2e-12,
                        "dense lazy-basis oracle failed: measured={measured_qubit} outcome={outcome} \
                         flip={flip_sites:?} sign={sign_sites:?} phase={phase} \
                         queue={queued:?} fidelity={fidelity:.16}"
                    );
                    cases += 1;
                }
            }

            for gate in 0..6 {
                let mut next = tableau.clone();
                match gate {
                    0 => next.h(&[QubitId(0)]),
                    1 => next.h(&[QubitId(1)]),
                    2 => next.sz(&[QubitId(0)]),
                    3 => next.sz(&[QubitId(1)]),
                    4 => next.cx(&[(QubitId(0), QubitId(1))]),
                    _ => next.cx(&[(QubitId(1), QubitId(0))]),
                };
                if seen.insert(tableau_key(&next)) {
                    pending.push_back(next);
                }
            }
        }

        assert_eq!(
            cliffords, 11_520,
            "must enumerate the full two-qubit Clifford group"
        );
        assert!(cases > 0);
        assert!(
            signed_phase_counts.iter().all(|&count| count > 0),
            "the exhaustive oracle must cover signed phases +1,+i,-1,-i: {signed_phase_counts:?}"
        );
        eprintln!(
            "lazy measurement basis dense oracle: cliffords={cliffords} cases={cases} signed_phase_counts={signed_phase_counts:?}"
        );
    }

    #[test]
    fn endpoint_probability_and_survival_loss_are_distinct_classifications() {
        let probability_one: f64 = 4.0e-15;
        let mut mps = Mps::new(1, MpsConfig::default());
        let preparation = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new((1.0 - probability_one).sqrt(), 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(probability_one.sqrt(), 0.0),
                Complex64::new(1.0, 0.0),
            ],
        );
        mps.apply_one_site_gate(0, &preparation).unwrap();
        let mut tableau = SparseStabY::new(1).with_destab_sign_tracking();
        let endpoint = project_forced_z_with_update(&mut tableau, &mut mps, 0, true).unwrap();
        assert_eq!(endpoint.snapped_probability.to_bits(), 0.0_f64.to_bits());
        assert!(endpoint.survival_ratio > 1.0 - 1e-12);

        let mut mps = Mps::new(1, MpsConfig::default());
        let hadamard = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
                Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
                Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
                Complex64::new(-std::f64::consts::FRAC_1_SQRT_2, 0.0),
            ],
        );
        mps.apply_one_site_gate(0, &hadamard).unwrap();
        let mut tableau = SparseStabY::new(1).with_destab_sign_tracking();
        inject_projection_vanishes(1);
        let vanished = project_forced_z_with_update(&mut tableau, &mut mps, 0, false).unwrap();
        assert!((vanished.snapped_probability - 0.5).abs() < 1e-14);
        assert!(vanished.survival_ratio < BRANCH_VANISH_SURVIVAL_THRESHOLD);
    }

    #[test]
    fn trivial_probability_quantization_is_shared_by_sampling_and_projection() {
        assert_eq!(
            quantize_trivial_probability(0.499_999_68).to_bits(),
            0.5_f64.to_bits()
        );

        let probability_one: f64 = 1.0e-13;
        let mut mps = Mps::new(1, MpsConfig::default());
        let preparation = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new((1.0 - probability_one).sqrt(), 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(probability_one.sqrt(), 0.0),
                Complex64::new(1.0, 0.0),
            ],
        );
        mps.apply_one_site_gate(0, &preparation).unwrap();
        let mut tableau = SparseStabY::new(1).with_destab_sign_tracking();
        let sampled_probability =
            z_outcome_probability(&tableau, &mps, 0, true, "trivial probability test");
        let projected = project_forced_z_with_update(&mut tableau, &mut mps, 0, true).unwrap();
        assert_eq!(sampled_probability.to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            projected.snapped_probability.to_bits(),
            sampled_probability.to_bits()
        );
    }

    #[test]
    fn exact_projection_compresses_clean_rank_redundancy_without_discarded_weight() {
        let product = Mps::new(4, MpsConfig::default());
        let mut mps = product.add(&product);
        mps.normalize();
        mps.reset_truncation_stats();
        assert_eq!(mps.max_bond_dim(), 2);
        let mut tableau = SparseStabY::new(4).with_destab_sign_tracking();

        let projection = project_forced_z_with_update(&mut tableau, &mut mps, 0, false).unwrap();

        assert_eq!(projection.snapped_probability.to_bits(), 1.0_f64.to_bits());
        assert_eq!(mps.max_bond_dim(), 1);
        assert_eq!(mps.summed_discarded_weight().to_bits(), 0.0_f64.to_bits());
        assert_eq!(mps.truncation_error().to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn renamed_pragmatic_low_level_entry_is_callable() {
        let mut tableau = SparseStabY::new(1).with_destab_sign_tracking();
        let mut mps = Mps::new(1, MpsConfig::default());
        let mut rng = PecosRng::seed_from_u64(11);
        let result = measure_qubit_stab_mps_pragmatic(&mut tableau, &mut mps, &mut rng, 0).unwrap();
        assert!(!result.outcome);
    }

    #[test]
    fn projection_block_replacement_invalidation_guards_canonical_routing() {
        let product = Mps::new(4, MpsConfig::default());
        let mut mps = product.add(&product);
        mps.left_canonicalize();

        project_single_flip_without_sign(&mut mps, 0, Complex64::new(1.0, 0.0), 0.5);

        mps.right_canonicalize();
        assert_eq!(mps.tracked_center_for_test(), Some(0));
        assert_eq!(mps.full_canonical_sweep_count(), 1);
        assert_eq!(mps.center_reuse_count(), 0);
    }

    #[test]
    fn projection_block_replacement_at_center_reuses_canonical_routing() {
        let product = Mps::new(4, MpsConfig::default());
        let mut mps = product.add(&product);
        mps.left_canonicalize();
        assert_eq!(mps.tracked_center_for_test(), Some(3));

        project_single_flip_without_sign(&mut mps, 3, Complex64::new(1.0, 0.0), 0.5);

        assert_eq!(mps.tracked_center_for_test(), Some(3));
        // The production exact-projection compression route invokes the debug
        // center validator here before reusing the preserved center.
        mps.right_canonicalize();
        assert_eq!(mps.tracked_center_for_test(), Some(0));
        assert_eq!(mps.full_canonical_sweep_count(), 0);
        assert_eq!(mps.center_reuse_count(), 1);
    }

    #[test]
    fn conjugate_single_cnot_x_on_control() {
        // V = CNOT(0,1). V†·X_0·V = X_0·X_1.
        let mut flip = vec![0];
        let mut sign: Vec<usize> = vec![];
        conjugate_pauli_by_deferred(&mut flip, &mut sign, &[(0, 1)]);
        sort_dedup(&mut flip);
        assert_eq!(flip, vec![0, 1]);
        assert!(sign.is_empty());
    }

    #[test]
    fn conjugate_single_cnot_z_on_target() {
        // V = CNOT(0,1). V†·Z_1·V = Z_0·Z_1.
        let mut flip: Vec<usize> = vec![];
        let mut sign = vec![1];
        conjugate_pauli_by_deferred(&mut flip, &mut sign, &[(0, 1)]);
        sort_dedup(&mut sign);
        assert!(flip.is_empty());
        assert_eq!(sign, vec![0, 1]);
    }

    #[test]
    fn conjugate_cnot_x_on_target_unchanged() {
        // V = CNOT(0,1). V†·X_1·V = X_1 (target X unchanged).
        let mut flip = vec![1];
        let mut sign: Vec<usize> = vec![];
        conjugate_pauli_by_deferred(&mut flip, &mut sign, &[(0, 1)]);
        assert_eq!(flip, vec![1]);
        assert!(sign.is_empty());
    }

    #[test]
    fn conjugate_cnot_z_on_control_unchanged() {
        // V = CNOT(0,1). V†·Z_0·V = Z_0 (control Z unchanged).
        let mut flip: Vec<usize> = vec![];
        let mut sign = vec![0];
        conjugate_pauli_by_deferred(&mut flip, &mut sign, &[(0, 1)]);
        assert!(flip.is_empty());
        assert_eq!(sign, vec![0]);
    }

    #[test]
    fn conjugate_two_cnots_cancels() {
        // V = CNOT(0,1)·CNOT(0,1) = I. V†·X_0·V = X_0.
        let mut flip = vec![0];
        let mut sign: Vec<usize> = vec![];
        conjugate_pauli_by_deferred(&mut flip, &mut sign, &[(0, 1), (0, 1)]);
        sort_dedup(&mut flip);
        assert_eq!(flip, vec![0]);
    }

    #[test]
    fn conjugate_cnot_chain_fanout() {
        // V = CNOT(0,3)·CNOT(0,2)·CNOT(0,1) — fan-out from qubit 0.
        // V†·X_0·V = ? Chain conjugation: innermost first.
        // Step 1 (CNOT(0,3)): X_0 -> X_0·X_3. flip={0,3}.
        // Step 2 (CNOT(0,2)): X_0 -> X_0·X_2. flip={0,2,3}.
        // Step 3 (CNOT(0,1)): X_0 -> X_0·X_1. flip={0,1,2,3}.
        let mut flip = vec![0];
        let mut sign: Vec<usize> = vec![];
        // Pushed in chronological order: first pushed = CNOT(0,1).
        // V = last·...·first = CNOT(0,3)·CNOT(0,2)·CNOT(0,1).
        conjugate_pauli_by_deferred(&mut flip, &mut sign, &[(0, 1), (0, 2), (0, 3)]);
        sort_dedup(&mut flip);
        assert_eq!(flip, vec![0, 1, 2, 3]);
        assert!(sign.is_empty());
    }

    #[test]
    fn flush_deferred_matches_eager() {
        // Two MPS: one where we apply CNOTs eagerly, one where we flush
        // the queue at the end. Final states should agree.
        let config = MpsConfig::default();
        let num_qubits = 4;

        let mut mps_eager = Mps::new(num_qubits, config.clone());
        // Put into a non-trivial state first: apply H on site 0 via
        // single-site gate (to avoid bond-dim 1 trivial case).
        let h = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(0.5_f64.sqrt(), 0.0),
                Complex64::new(0.5_f64.sqrt(), 0.0),
                Complex64::new(0.5_f64.sqrt(), 0.0),
                Complex64::new(-0.5_f64.sqrt(), 0.0),
            ],
        );
        mps_eager
            .apply_one_site_gate(0, &h)
            .expect("MPS op on valid site");
        let mut mps_lazy = mps_eager.clone();

        // Apply CNOT(0,1), CNOT(0,2), CNOT(1,3) eagerly.
        let cnots = vec![(0usize, 1usize), (0, 2), (1, 3)];
        for &(c, t) in &cnots {
            apply_cnot_to_mps(&mut mps_eager, c, t).unwrap();
        }

        // Flush the same CNOTs.
        let mut queue = cnots;
        flush_deferred(&mut mps_lazy, &mut queue).unwrap();
        assert!(queue.is_empty());

        // Compare state vectors.
        let sv_e = mps_eager.state_vector();
        let sv_l = mps_lazy.state_vector();
        assert_eq!(sv_e.len(), sv_l.len());
        for (a, b) in sv_e.iter().zip(sv_l.iter()) {
            assert!((a - b).norm() < 1e-10, "eager vs lazy differ: {a} vs {b}");
        }
    }
}

// Issue #562 diagnostics intentionally live outside the shipped measurement
// path.  The module is kept as a separate file because it reconstructs dense
// intermediate states and is far too expensive for production use.
#[cfg(test)]
mod phase_diagnostic;
