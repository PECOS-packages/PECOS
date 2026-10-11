// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! DEM event classification and probability computation.
//!
//! Classifies propagated EEG generators by DEM event class (which
//! detectors each Pauli anticommutes with), then computes event
//! probabilities using the correct formulas from Hines et al.

use crate::Bm;
use crate::circuit::PropagatedEeg;
use crate::eeg::EegType;
use crate::stabilizer::StabilizerGroup;
use pecos_core::pauli::pauli_bitmask::BitmaskStorage;
use pecos_core::{Pauli, PauliString, Phase};
use smallvec::SmallVec;
use std::collections::BTreeMap;
use std::fmt::Write as _;

type DetectorSet = SmallVec<[usize; 4]>;
type ObservableSet = SmallVec<[usize; 2]>;
type EventKey = (DetectorSet, ObservableSet);
type XzComponents = (Option<DemEvent>, Option<DemEvent>);
type GraphlikePieces = Vec<DetectorSet>;
type DecompMemo = BTreeMap<DetectorSet, Option<GraphlikePieces>>;

/// Controls the H-type probability formula.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HFormula {
    /// p = sum h_j h_k beta (leading-order Taylor). Fast, accurate for small angles.
    #[default]
    Taylor,
    /// p = sin^2(h_eff) applied to the Taylor quadratic form.
    SinSquared,
    /// Independent-rotation product: p = (1 - prod_j cos(2 h_j))/2.
    /// Ignores interference between generators. For commuting generators it
    /// is exact only when no product of two or more event labels is (up to
    /// sign) a stabilizer. Use `ExactSubset` for all-orders interference.
    ExactCommuting,
    /// Exact subset sum formula for commuting generators:
    ///   p = (1/2)(1 - Re(Σ_S i^|S| Π sin · Π cos · ε_S))
    /// Enumerates all even-size subsets of generators, checks if the
    /// product (Π_{j∈S} P_j)·D is a stabilizer. Captures all orders of
    /// multi-body interference. Exact for commuting generators. Cost: O(2^N)
    /// where N is generators per event. Practical for N ≤ ~25.
    ExactSubset,
}

/// BCH order for generator accumulation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BchOrder {
    /// First-order: G_c = sum G_i. Same-label rates add.
    #[default]
    First,
    /// Second-order: G_c = sum G_i + (1/2) sum [G_i, G_j].
    /// Adds new H generators from [H_P, H_Q] = -2i H_{PQ} for anticommuting P,Q.
    /// Also adds Zassenhaus W_2 cross-event [H,S] → C corrections.
    Second,
}

/// Configuration for the EEG DEM builder.
///
/// Controls all three approximation levels described in the paper:
/// 1. BCH order (k): how generators from different layers are combined
/// 2. Zassenhaus order: how the combined generator is split into single-event channels
///    (coupled to BCH order — Second enables W_2 cross-event terms)
/// 3. H-type formula: how detection probabilities are estimated from generators
#[derive(Clone, Copy, Debug)]
pub struct EegConfig {
    /// BCH expansion order for combining layer errors (default: First).
    pub bch_order: BchOrder,
    /// Formula for H-type detection probability (default: Taylor).
    pub h_formula: HFormula,
}

impl Default for EegConfig {
    fn default() -> Self {
        Self {
            bch_order: BchOrder::First,
            h_formula: HFormula::Taylor,
        }
    }
}

impl EegConfig {
    /// Create config with default settings (first-order BCH, Taylor formula).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set BCH order to first (default). Same-label rate summation only.
    #[must_use]
    pub fn bch_first(mut self) -> Self {
        self.bch_order = BchOrder::First;
        self
    }

    /// Set BCH order to second. Adds [H,H] and [H,S] commutator corrections.
    #[must_use]
    pub fn bch_second(mut self) -> Self {
        self.bch_order = BchOrder::Second;
        self
    }

    /// Use leading-order Taylor (h²) for H-type probabilities (default).
    #[must_use]
    pub fn taylor(mut self) -> Self {
        self.h_formula = HFormula::Taylor;
        self
    }

    /// Use sin²(h_eff) for H-type probabilities.
    #[must_use]
    pub fn sin_squared(mut self) -> Self {
        self.h_formula = HFormula::SinSquared;
        self
    }

    /// Use the independent-rotation product; see [`HFormula::ExactCommuting`].
    #[must_use]
    pub fn exact_commuting(mut self) -> Self {
        self.h_formula = HFormula::ExactCommuting;
        self
    }

    /// Use exact subset-sum formula for commuting H-type generators.
    /// Captures all orders of multi-body interference. O(2^N) per event.
    #[must_use]
    pub fn exact_subset(mut self) -> Self {
        self.h_formula = HFormula::ExactSubset;
        self
    }
}

/// A detector definition for EEG classification.
#[derive(Clone, Debug)]
pub struct Detector {
    pub id: usize,
    /// Pauli stabilizer. A Pauli P flips this detector iff P anticommutes with it.
    pub stabilizer: Bm,
}

/// A logical observable for EEG classification.
#[derive(Clone, Debug)]
pub struct Observable {
    pub id: usize,
    pub pauli: Bm,
}

/// A DEM event: the set of detectors and observables flipped.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DemEvent {
    pub detectors: SmallVec<[usize; 4]>,
    pub observables: SmallVec<[usize; 2]>,
}

/// A DEM entry: event + probability.
#[derive(Clone, Debug)]
pub struct DemEntry {
    pub event: DemEvent,
    pub probability: f64,
}

/// A DEM entry with X/Z decomposition info for MWPM decoders.
///
/// When both `x_component` and `z_component` are `Some`, the mechanism
/// can be output in decomposed form: `error(p) x_targets ^ z_targets`.
/// When only one is present (pure X or Z error), output directly.
#[derive(Clone, Debug)]
pub struct DecomposableDemEntry {
    /// Combined detector/observable flips (XOR of X and Z components).
    pub event: DemEvent,
    /// Mechanism probability.
    pub probability: f64,
    /// Detector/observable flips from the X-only component of the Pauli label.
    /// None if the label has no X part.
    pub x_component: Option<DemEvent>,
    /// Detector/observable flips from the Z-only component of the Pauli label.
    /// None if the label has no Z part.
    pub z_component: Option<DemEvent>,
}

/// Convert PauliString to Bm.
#[must_use]
pub fn pauli_string_to_bitmask(ps: &PauliString) -> Bm {
    let mut bm = Bm::default();
    for &(pauli, qubit) in ps.paulis() {
        let q = qubit.index();
        match pauli {
            Pauli::X => bm.x_bits.set_bit(q),
            Pauli::Z => bm.z_bits.set_bit(q),
            Pauli::Y => {
                bm.x_bits.set_bit(q);
                bm.z_bits.set_bit(q);
            }
            Pauli::I => {}
        }
    }
    bm
}

/// Classify which detectors and observables a Pauli label anticommutes with.
fn classify(label: &Bm, detectors: &[Detector], observables: &[Observable]) -> DemEvent {
    let mut dets = SmallVec::new();
    for det in detectors {
        if !label.commutes_with(&det.stabilizer) {
            dets.push(det.id);
        }
    }
    let mut obs = SmallVec::new();
    for o in observables {
        if !label.commutes_with(&o.pauli) {
            obs.push(o.id);
        }
    }
    DemEvent {
        detectors: dets,
        observables: obs,
    }
}

/// Classify with X/Z component decomposition.
///
/// Returns (combined_event, x_component, z_component) where x_component
/// is the detectors/observables flipped by the X-only part of the label,
/// and z_component by the Z-only part.
fn classify_xz(
    label: &Bm,
    detectors: &[Detector],
    observables: &[Observable],
) -> (DemEvent, Option<DemEvent>, Option<DemEvent>) {
    use pecos_core::pauli::pauli_bitmask::BitmaskStorage;

    let has_x = !label.x_bits.is_zero();
    let has_z = !label.z_bits.is_zero();

    // Build X-only and Z-only labels
    let x_only = if has_x {
        Some(Bm {
            x_bits: label.x_bits.clone(),
            ..Default::default()
        })
    } else {
        None
    };
    let z_only = if has_z {
        Some(Bm {
            z_bits: label.z_bits.clone(),
            ..Default::default()
        })
    } else {
        None
    };

    let combined = classify(label, detectors, observables);

    let x_event = x_only.map(|x_label| classify(&x_label, detectors, observables));
    let z_event = z_only.map(|z_label| classify(&z_label, detectors, observables));

    (combined, x_event, z_event)
}

/// Build a DEM from propagated EEG generators.
///
/// Groups generators by DEM event class, then computes probabilities:
/// - **S-only event**: p = (1/2)(1 - exp(2 * sum_coefficients)) [exact]
/// - **H-only event**: p = sum_j h_j^2 [leading order, diagonal terms]
///   (Off-diagonal coherent interference captured at second order via beta)
/// - **Mixed**: S at O(epsilon) + H^2 at O(epsilon^2)
///
/// For first-order BCH with leading-order Taylor, the H-only formula uses
/// only diagonal terms: p = sum_j h_j^2. This is the Pauli-twirled
/// equivalent. To capture coherent accumulation (off-diagonal terms),
/// we need to check if pairs of H generators anticommute with the same
/// detectors AND their product Q_j*Q_k is a stabilizer of |psi>.
///
/// Without a stabilizer group H uses the diagonal approximation. Use
/// [`build_dem_with_stabilizers`] to include coherent interference and C/A.
///
/// # Panics
/// Panics for C/A input without `label2` or without a stabilizer group,
/// including zero coefficients, or an event probability below -1e-12.
#[must_use]
pub fn build_dem(
    generators: &[PropagatedEeg],
    detectors: &[Detector],
    observables: &[Observable],
) -> Vec<DemEntry> {
    build_dem_with_stabilizers(generators, detectors, observables, None)
}

/// Build DEM with stabilizer group for coherent interference and C/A.
///
/// Equal-event commuting C labels contribute `2 c <PQ>`; anticommuting A
/// labels contribute `-2 i a <PQ>`. These are folded into the S coefficient
/// sum before exponentiation. Thus `C_{P,P}(c) = S_P(-2c)` exactly.
/// The C fold is exact alone and with S labels commuting with PQ; the A fold
/// agrees to first order (its exact pattern probability is linear in a).
///
/// # Panics
/// Panics for C/A input without `label2` or without a stabilizer group,
/// including zero coefficients, or an event probability below -1e-12.
#[must_use]
pub fn build_dem_with_stabilizers(
    generators: &[PropagatedEeg],
    detectors: &[Detector],
    observables: &[Observable],
    stabilizer_group: Option<&StabilizerGroup>,
) -> Vec<DemEntry> {
    build_dem_inner(
        generators,
        detectors,
        observables,
        stabilizer_group,
        HFormula::Taylor,
        BchOrder::First,
    )
}

/// Build DEM with all options via config struct.
///
/// # Panics
/// Panics for C/A input without `label2` or without a stabilizer group,
/// including zero coefficients, or an event probability below -1e-12.
/// `ExactSubset` panics when an event has more than 25 H labels.
#[must_use]
pub fn build_dem_configured(
    generators: &[PropagatedEeg],
    detectors: &[Detector],
    observables: &[Observable],
    stabilizer_group: Option<&StabilizerGroup>,
    config: &EegConfig,
) -> Vec<DemEntry> {
    build_dem_inner(
        generators,
        detectors,
        observables,
        stabilizer_group,
        config.h_formula,
        config.bch_order,
    )
}

/// Build DEM with individual options (convenience).
///
/// # Panics
/// Panics for C/A input without `label2` or without a stabilizer group,
/// including zero coefficients, or an event probability below -1e-12.
/// `ExactSubset` panics when an event has more than 25 H labels.
#[must_use]
pub fn build_dem_with_options(
    generators: &[PropagatedEeg],
    detectors: &[Detector],
    observables: &[Observable],
    stabilizer_group: Option<&StabilizerGroup>,
    h_formula: HFormula,
    bch_order: BchOrder,
) -> Vec<DemEntry> {
    build_dem_inner(
        generators,
        detectors,
        observables,
        stabilizer_group,
        h_formula,
        bch_order,
    )
}

/// Return the ExactSubset event-size violation for the inputs a DEM build uses.
///
/// Returns `None` for other formulas. Counts distinct, non-negligible H labels
/// after same-label accumulation and the configured BCH corrections, using
/// the same event preparation and error message as the builder. Other input
/// validity conditions are not checked here.
#[must_use]
pub fn exact_subset_violation(
    generators: &[PropagatedEeg],
    detectors: &[Detector],
    observables: &[Observable],
    config: &EegConfig,
) -> Option<String> {
    if config.h_formula != HFormula::ExactSubset {
        return None;
    }
    prepare_h_events(
        collect_h_coefficients(generators),
        detectors,
        observables,
        config.bch_order,
    )
    .exact_subset_violation(config.h_formula)
}

struct HEvents {
    coefficients: BTreeMap<DemEvent, Vec<f64>>,
    labels: BTreeMap<DemEvent, Vec<Bm>>,
}

impl HEvents {
    fn exact_subset_violation(&self, formula: HFormula) -> Option<String> {
        if formula != HFormula::ExactSubset {
            return None;
        }
        self.coefficients.iter().find_map(|(event, coefficients)| {
            (coefficients.len() > 25).then(|| {
                format!(
                    "ExactSubset limit is 25 H labels per event; got {} for event {event:?}",
                    coefficients.len()
                )
            })
        })
    }
}

fn collect_h_coefficients(generators: &[PropagatedEeg]) -> BTreeMap<Bm, f64> {
    let mut coefficients = BTreeMap::new();
    for generator in generators {
        if generator.eeg_type == EegType::H {
            *coefficients.entry(generator.label.clone()).or_insert(0.0) += generator.coeff;
        }
    }
    coefficients
}

fn prepare_h_events(
    mut h_by_label: BTreeMap<Bm, f64>,
    detectors: &[Detector],
    observables: &[Observable],
    bch_order: BchOrder,
) -> HEvents {
    // Second-order BCH: [H_P, H_Q] = -2i H_{PQ} for anticommuting P, Q.
    // PQ = i^k * R (from multiply_with_phase), so H_{PQ} = i^k * H_R.
    // BCH coefficient: (1/2) * (-2i) * i^k * h_i * h_j = -i^{k+1} * h_i * h_j.
    // Anticommuting Hermitian Paulis have odd k, so this is always real.
    let mut h_bch2_re_by_label: BTreeMap<Bm, f64> = BTreeMap::new();

    if bch_order == BchOrder::Second {
        let h_entries: Vec<(Bm, f64)> = h_by_label.iter().map(|(l, &c)| (l.clone(), c)).collect();

        for i in 0..h_entries.len() {
            for j in (i + 1)..h_entries.len() {
                let (p_i, h_i) = &h_entries[i];
                let (p_j, h_j) = &h_entries[j];

                if p_i.commutes_with(p_j) {
                    continue;
                }

                // PQ = i^k * R. Coefficient: -i^{k+1} * h_i * h_j = i^{k+3} * h_i * h_j.
                let (product, phase_k) = p_i.multiply_with_phase(p_j);
                let mag = h_i * h_j;
                let phase = (phase_k + 3) % 4; // -i^{k+1} = i^{k+3}
                let coeff = match phase {
                    0 => mag,
                    2 => -mag,
                    _ => unreachable!("anticommuting Pauli product has odd phase"),
                };
                *h_bch2_re_by_label.entry(product).or_insert(0.0) += coeff;
            }
        }
    }

    // Merge the real BCH2 correction into the signed H coefficients.
    for (label, &coeff) in &h_bch2_re_by_label {
        *h_by_label.entry(label.clone()).or_insert(0.0) += coeff;
    }

    // Group BCH-combined generators by DEM event class.
    let mut h_events: BTreeMap<DemEvent, Vec<f64>> = BTreeMap::new();
    let mut event_pauli_labels: BTreeMap<DemEvent, Vec<Bm>> = BTreeMap::new();

    for (label, &coeff) in &h_by_label {
        if coeff.abs() < 1e-20 {
            continue;
        }
        let event = classify(label, detectors, observables);
        if event.detectors.is_empty() && event.observables.is_empty() {
            continue;
        }
        h_events.entry(event.clone()).or_default().push(coeff);
        event_pauli_labels
            .entry(event)
            .or_default()
            .push(label.clone());
    }

    HEvents {
        coefficients: h_events,
        labels: event_pauli_labels,
    }
}

fn build_dem_inner(
    generators: &[PropagatedEeg],
    detectors: &[Detector],
    observables: &[Observable],
    stabilizer_group: Option<&StabilizerGroup>,
    h_formula: HFormula,
    bch_order: BchOrder,
) -> Vec<DemEntry> {
    // First-order BCH: combine generators with the same Pauli label.
    // This is where coherent accumulation happens: h1 + h2 for same label.
    let h_by_label = collect_h_coefficients(generators);
    let mut s_by_label: BTreeMap<Bm, f64> = BTreeMap::new();

    // C and A type generators: two labels, first-order contribution.
    // Key = (label1, label2), value = coefficient.
    let mut c_generators: Vec<(Bm, Bm, f64)> = Vec::new();
    let mut a_generators: Vec<(Bm, Bm, f64)> = Vec::new();

    for g in generators {
        match g.eeg_type {
            EegType::H => {}
            EegType::S => {
                *s_by_label.entry(g.label.clone()).or_insert(0.0) += g.coeff;
            }
            EegType::C | EegType::A => {
                let l2 = g.label2.clone().expect("C/A generator requires label2");
                assert!(
                    stabilizer_group.is_some(),
                    "C/A generator requires a stabilizer group"
                );
                let target = if g.eeg_type == EegType::C {
                    &mut c_generators
                } else {
                    &mut a_generators
                };
                target.push((g.label.clone(), l2, g.coeff));
            }
        }
    }

    // Zassenhaus W_2 [H,S] placeholders. For anticommuting P,Q,
    // QP = i^k R has odd k, so the C coefficient i^{k+1} h s is real.
    // Q and R anticommute: their C pattern contribution is zero under the
    // equal-event rule below. Retain the internal zero entries; they are not
    // user input and need neither a stabilizer group nor validation.
    if bch_order == BchOrder::Second {
        let h_entries: Vec<(Bm, f64)> = h_by_label.iter().map(|(l, &c)| (l.clone(), c)).collect();
        let s_entries: Vec<(Bm, f64)> = s_by_label.iter().map(|(l, &c)| (l.clone(), c)).collect();

        for (p, _h_coeff) in &h_entries {
            for (q, _s_coeff) in &s_entries {
                if p.commutes_with(q) {
                    continue;
                }

                let (qp, _phase) = q.multiply_with_phase(p);
                c_generators.push((q.clone(), qp, 0.0));
            }
        }
    }

    let h_events = prepare_h_events(h_by_label, detectors, observables, bch_order);
    if let Some(message) = h_events.exact_subset_violation(h_formula) {
        panic!("{message}");
    }
    let mut s_events: BTreeMap<DemEvent, Vec<f64>> = BTreeMap::new();

    for (label, &coeff) in &s_by_label {
        let event = classify(label, detectors, observables);
        if event.detectors.is_empty() && event.observables.is_empty() {
            continue;
        }
        s_events.entry(event).or_default().push(coeff);
    }

    // For the pattern projector Pi_E = prod_i (1 - d_i D_i)/2,
    // equal-event labels give delta p = 2 c <PQ> (C) or -2 i a <PQ> (A),
    // independently of |E|: Pi_E annihilates |psi> and preserves P|psi>,
    // Q|psi>. Thus Tr(Pi_E P rho Q)=<QP>, Tr(Pi_E Q rho P)=<PQ>,
    // and the anticommutator terms vanish. Their sum (C) or i times their
    // difference (A) gives the expressions above. No detector-product sign
    // enters this rule.
    if let Some(stab_group) = stabilizer_group {
        for (is_c, gens) in [(true, &c_generators), (false, &a_generators)] {
            for (p, q, coeff) in gens {
                let event = classify(p, detectors, observables);
                if event != classify(q, detectors, observables)
                    || (event.detectors.is_empty() && event.observables.is_empty())
                    || p.commutes_with(q) != is_c
                {
                    continue;
                }
                let expectation = stab_group.product_expectation([p, q]);
                let delta_p = expectation.map_or(0.0, |phase| {
                    let value = phase.to_complex();
                    2.0 * coeff * if is_c { value.re } else { value.im }
                });
                s_events.entry(event).or_default().push(-delta_p);
            }
        }
    }

    let mut entries = Vec::new();
    // S and effective C/A rates combine BEFORE the exact S exponential.
    for (event, rates) in &s_events {
        let sum_rate: f64 = rates.iter().sum();
        let prob = checked_event_probability((1.0 - (2.0 * sum_rate).exp()) / 2.0, event);
        if prob > 1e-15 {
            entries.push(DemEntry {
                event: event.clone(),
                probability: prob,
            });
        }
    }

    // H-only events: full leading-order formula with coherent interference.
    //
    // p_D = -(1/4) * sum_{j,k} h_j * h_k * beta(psi, C_{Qj,Qk}, P)
    //
    // beta(psi, C_{Q,Q'}, P) = -4 if [Q,Q']=0 and Q*Q'|psi>=+|psi> (stabilizer)
    //                        = +4 if [Q,Q']=0 and Q*Q'|psi>=-|psi> (anti-stabilizer)
    //                        = 0 otherwise
    //
    // Diagonal terms (j=k): Q*Q=I, always +1 stabilizer → beta=-4 → contributes +h_j^2
    // Off-diagonal with stabilizer product: contributes +h_j*h_k (constructive)
    // Off-diagonal with anti-stabilizer: contributes -h_j*h_k (destructive)
    //
    // If stabilizer_group is None, fall back to diagonal approximation.
    for (event, coeffs) in &h_events.coefficients {
        // Collect the detector stabilizers for ExactSubset.
        let event_det_stab = if h_formula == HFormula::ExactSubset {
            // Producers use Z-only measurement parities, so this product has no phase.
            let mut stab = Bm::default();
            for &d_id in &event.detectors {
                if let Some(det) = detectors.iter().find(|d| d.id == d_id) {
                    stab = stab.multiply(&det.stabilizer);
                }
            }
            Some(stab)
        } else {
            None
        };

        let prob = if let Some(stab_group) = stabilizer_group {
            compute_h_probability_full(
                coeffs,
                &h_events.labels,
                event,
                stab_group,
                h_formula,
                event_det_stab.as_ref(),
            )
        } else {
            checked_event_probability(coeffs.iter().map(|h| h * h).sum(), event)
        };
        if prob > 1e-15 {
            if let Some(existing) = entries.iter_mut().find(|e| e.event == *event) {
                let p_s = existing.probability;
                existing.probability =
                    checked_event_probability(p_s + prob - 2.0 * p_s * prob, event);
            } else {
                entries.push(DemEntry {
                    event: event.clone(),
                    probability: prob,
                });
            }
        }
    }

    entries
}

/// Compute H-type event probability using the full beta function.
///
/// p_D = -(1/4) * sum_{j,k} h_j * h_k * beta(psi, C_{Qj,Qk}, P)
///
/// For each pair (j,k):
/// - If [Q_j, Q_k] ≠ 0: beta = 0
/// - If [Q_j, Q_k] = 0 and Q_j*Q_k is +1 stabilizer: beta = -4 → +h_j*h_k
/// - If [Q_j, Q_k] = 0 and Q_j*Q_k is -1 stabilizer: beta = +4 → -h_j*h_k
/// - Otherwise: beta = 0
fn compute_h_probability_full(
    coeffs: &[f64],
    event_labels: &BTreeMap<DemEvent, Vec<Bm>>,
    event: &DemEvent,
    stab_group: &StabilizerGroup,
    h_formula: HFormula,
    det_stabilizer: Option<&Bm>,
) -> f64 {
    let Some(labels) = event_labels.get(event) else {
        return 0.0;
    };

    let n = coeffs.len();

    if h_formula == HFormula::ExactCommuting {
        return compute_exact_commuting(coeffs, event);
    }
    if h_formula == HFormula::ExactSubset
        && let Some(det_stab) = det_stabilizer
    {
        return compute_exact_subset(coeffs, labels, stab_group, det_stab, event);
    }

    // --- Taylor or SinSquared: quadratic form with beta ---
    let mut total = 0.0_f64;

    for j in 0..n {
        for k in 0..n {
            if j == k {
                total += coeffs[j] * coeffs[j];
            } else if labels[j].commutes_with(&labels[k]) {
                let beta = stab_group
                    .product_expectation([&labels[j], &labels[k]])
                    .map_or(0.0, |phase| phase.to_complex().re);
                total += coeffs[j] * coeffs[k] * beta;
            }
        }
    }

    let total = checked_event_probability(total, event);

    match h_formula {
        HFormula::SinSquared => {
            let h_eff = total.sqrt();
            h_eff.sin().powi(2)
        }
        HFormula::Taylor | HFormula::ExactCommuting | HFormula::ExactSubset => total,
    }
}

/// Reject materially negative event probabilities; tolerate roundoff only.
fn checked_event_probability(probability: f64, event: &DemEvent) -> f64 {
    assert!(
        probability >= -1e-12,
        "negative probability {probability} for event {event:?}"
    );
    probability.max(0.0)
}

/// Independent rotations, ignoring interference (see HFormula::ExactCommuting).
fn compute_exact_commuting(coeffs: &[f64], event: &DemEvent) -> f64 {
    let product: f64 = coeffs.iter().map(|h| (2.0 * h).cos()).product();
    checked_event_probability(0.5 * (1.0 - product), event)
}

/// Exact subset-sum formula for commuting H-type generators.
///
/// Enumerates ALL even-size subsets S of generators. For each:
///   coefficient = i^|S| · Π_{j∈S} sin(2h_j) · Π_{j∉S} cos(2h_j)
///   eigenvalue  = ε_S = ⟨ψ|(Π_{j∈S} P_j)·D|ψ⟩
///
/// p_D = (1/2)(1 - Re(Σ_S coefficient · ε_S))
///
/// Only even-|S| subsets contribute to Re (odd powers of i are imaginary).
/// This captures all orders of multi-body interference. Exact when all
/// generators commute. Cost: O(2^N) where N = number of generators.
fn compute_exact_subset(
    coeffs: &[f64],
    labels: &[Bm],
    stab_group: &StabilizerGroup,
    det_stab: &Bm,
    event: &DemEvent,
) -> f64 {
    let n = coeffs.len();

    // Precompute sin(2h_j) and cos(2h_j) for each generator
    let mut sin2h = Vec::with_capacity(n);
    let mut cos2h = Vec::with_capacity(n);
    for &h in coeffs.iter().take(n) {
        sin2h.push((2.0 * h).sin());
        cos2h.push((2.0 * h).cos());
    }

    // The empty set (S = {}) contributes: Π cos(2h_j) · ε_{D}
    // D itself should be a stabilizer with eigenvalue +1 (no error → ⟨D⟩=1).
    let all_cos: f64 = cos2h.iter().product();

    let mut sum_re = all_cos; // |S|=0 contribution

    // Enumerate all non-empty even-size subsets via bitmask
    // For |S| even: i^|S| has Re = (-1)^{|S|/2}
    let total_subsets = 1u64 << n;
    for mask in 1..total_subsets {
        let size = mask.count_ones() as usize;
        if !size.is_multiple_of(2) {
            continue; // odd-size subsets have Im(i^|S|) only → Re = 0
        }

        let factors = std::iter::once(det_stab).chain(
            labels
                .iter()
                .enumerate()
                .filter(|(j, _)| mask & (1u64 << j) != 0)
                .map(|(_, label)| label),
        );
        let epsilon = stab_group
            .product_expectation(factors)
            .map_or(0.0, |phase| phase.to_complex().re);

        // Coefficient: (-1)^{|S|/2} · Π_{j∈S} sin(2h_j) · Π_{j∉S} cos(2h_j)
        let sign = if (size / 2).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };

        let mut coeff = sign;
        for (j, (&sin, &cos)) in sin2h.iter().zip(&cos2h).enumerate().take(n) {
            if mask & (1u64 << j) != 0 {
                coeff *= sin;
            } else {
                coeff *= cos;
            }
        }

        sum_re += coeff * epsilon;
    }

    let prob = 0.5 * (1.0 - sum_re);
    checked_event_probability(prob, event)
}

/// Sensitivity matrix M_E for a DEM event E (Hines Eq. 21).
///
/// M_E encodes how physical-level Hamiltonian error parameters affect the
/// probability of event E: p_E = -(1/2) θ^T M_E θ.
///
/// The matrix is indexed by `NoiseSource` pairs. Each entry M[i,j] = b_{P,Q_i,Q_j}
/// where b is the beta coefficient for the generator pair (Q_i, Q_j) and P is
/// the detector for event E.
///
/// Returns: Vec of (source_i, source_j, value) triplets for non-zero entries.
#[must_use]
pub fn sensitivity_matrix(
    generators: &[crate::circuit::PropagatedEeg],
    detectors: &[Detector],
    observables: &[Observable],
    stabilizer_group: Option<&StabilizerGroup>,
) -> BTreeMap<
    DemEvent,
    Vec<(
        crate::circuit::NoiseSource,
        crate::circuit::NoiseSource,
        f64,
    )>,
> {
    use crate::circuit::NoiseSource;
    use crate::eeg::EegType;

    let mut result = BTreeMap::new();

    // Collect H generators with their sources
    let h_gens: Vec<_> = generators
        .iter()
        .filter(|g| g.eeg_type == EegType::H && g.source.is_some())
        .collect();

    // Group by event class
    let mut event_gens: BTreeMap<DemEvent, Vec<(Bm, f64, NoiseSource)>> = BTreeMap::new();
    for g in &h_gens {
        let event = classify(&g.label, detectors, observables);
        if event.detectors.is_empty() && event.observables.is_empty() {
            continue;
        }
        event_gens.entry(event).or_default().push((
            g.label.clone(),
            g.coeff,
            g.source.clone().unwrap(),
        ));
    }

    // For each event class, build the sensitivity matrix
    for (event, gens) in &event_gens {
        let mut entries = Vec::new();

        for i in 0..gens.len() {
            for j in 0..gens.len() {
                let (ref label_i, _coeff_i, ref src_i) = gens[i];
                let (ref label_j, _coeff_j, ref src_j) = gens[j];

                // Beta coefficient for pair (i,j)
                let beta_val: f64 = if i == j {
                    1.0 // diagonal: beta = -4, but -(1/4)*(-4) = 1
                } else if label_i.commutes_with(label_j) {
                    if let Some(stab) = stabilizer_group {
                        stab.product_expectation([label_i, label_j])
                            .map_or(0.0, |phase| phase.to_complex().re)
                    } else if label_i == label_j {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                if beta_val.abs() > 1e-15_f64 {
                    entries.push((src_i.clone(), src_j.clone(), beta_val));
                }
            }
        }

        if !entries.is_empty() {
            result.insert(event.clone(), entries);
        }
    }

    result
}

/// Build a decomposable DEM from propagated EEG generators.
///
/// Same as `build_dem_configured` but includes X/Z component decomposition
/// for each mechanism, enabling proper graphlike decomposition for MWPM decoders.
///
/// # Panics
/// Panics for C/A input without `label2` or without a stabilizer group,
/// including zero coefficients, or an event probability below -1e-12.
/// `ExactSubset` panics when an event has more than 25 H labels.
#[must_use]
pub fn build_dem_decomposable(
    generators: &[PropagatedEeg],
    detectors: &[Detector],
    observables: &[Observable],
    stabilizer_group: Option<&StabilizerGroup>,
    config: &EegConfig,
) -> Vec<DecomposableDemEntry> {
    // First, build the standard DEM entries with the requested config
    let standard_entries =
        build_dem_configured(generators, detectors, observables, stabilizer_group, config);

    // Build a map from combined event → (x_component, z_component)
    // by classifying each unique label's X/Z components.
    // Multiple generators may contribute to the same event, but their
    // X/Z classification should be consistent (same combined effect = same decomposition).
    let mut event_xz: BTreeMap<EventKey, XzComponents> = BTreeMap::new();

    for g in generators {
        let (combined, x_ev, z_ev) = classify_xz(&g.label, detectors, observables);
        let key = (combined.detectors.clone(), combined.observables.clone());
        // First generator for this event wins (they should all agree)
        event_xz.entry(key).or_insert((x_ev, z_ev));
    }

    // Convert standard entries to decomposable entries
    standard_entries
        .into_iter()
        .map(|entry| {
            let key = (
                entry.event.detectors.clone(),
                entry.event.observables.clone(),
            );
            let (x_comp, z_comp) = event_xz.get(&key).cloned().unwrap_or((None, None));
            DecomposableDemEntry {
                event: entry.event,
                probability: entry.probability,
                x_component: x_comp,
                z_component: z_comp,
            }
        })
        .collect()
}

/// Format DEM entries as a Stim-compatible string.
#[must_use]
pub fn format_dem(entries: &[DemEntry]) -> String {
    let mut lines = Vec::new();
    for entry in entries {
        let mut parts = Vec::new();
        for &d in &entry.event.detectors {
            parts.push(format!("D{d}"));
        }
        for &o in &entry.event.observables {
            parts.push(format!("L{o}"));
        }
        if !parts.is_empty() {
            lines.push(format!(
                "error({:.6e}) {}",
                entry.probability,
                parts.join(" ")
            ));
        }
    }
    lines.join("\n")
}

/// Format a list of decomposable DEM entries into a graphlike DEM string.
///
/// Mechanisms with both X and Z components are output as `error(p) x_targets ^ z_targets`.
/// Single-component mechanisms with ≤ 2 detectors are output directly.
/// Single-component hyperedges (3+ detectors) are decomposed via a graphlike
/// index: expressed as XOR of existing graphlike mechanisms. If no decomposition
/// exists, the mechanism is dropped (cannot be used by MWPM decoders).
#[must_use]
pub fn format_dem_decomposed(entries: &[DecomposableDemEntry]) -> String {
    use std::collections::{BTreeMap, BTreeSet};

    fn format_event(ev: &DemEvent) -> String {
        let mut parts = Vec::new();
        for &d in &ev.detectors {
            parts.push(format!("D{d}"));
        }
        for &o in &ev.observables {
            parts.push(format!("L{o}"));
        }
        parts.join(" ")
    }

    fn is_graphlike(ev: &DemEvent) -> bool {
        ev.detectors.len() <= 2
    }

    // Search for decomposition of a hyperedge into XOR of graphlike pieces
    fn search_decomp(
        remaining: &DetectorSet,
        by_det: &[Vec<DetectorSet>],
        graphlike_set: &BTreeSet<DetectorSet>,
        memo: &mut DecompMemo,
    ) -> Option<GraphlikePieces> {
        if let Some(cached) = memo.get(remaining) {
            return cached.clone();
        }
        if remaining.is_empty() {
            let r = Some(Vec::new());
            memo.insert(remaining.clone(), r.clone());
            return r;
        }
        if remaining.len() <= 2 && graphlike_set.contains(remaining) {
            let r = Some(vec![remaining.clone()]);
            memo.insert(remaining.clone(), r.clone());
            return r;
        }

        let pivot = remaining[0];
        if pivot >= by_det.len() {
            memo.insert(remaining.clone(), None);
            return None;
        }

        for candidate in &by_det[pivot] {
            // Candidate detectors must be subset of remaining
            if !candidate.iter().all(|d| remaining.contains(d)) {
                continue;
            }
            // XOR: remove shared detectors, keep symmetric difference
            let mut next: SmallVec<[usize; 4]> = SmallVec::new();
            let mut i = 0;
            let mut j = 0;
            let r = remaining;
            let c = candidate;
            while i < r.len() && j < c.len() {
                match r[i].cmp(&c[j]) {
                    std::cmp::Ordering::Less => {
                        next.push(r[i]);
                        i += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        next.push(c[j]);
                        j += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        i += 1;
                        j += 1;
                    }
                }
            }
            while i < r.len() {
                next.push(r[i]);
                i += 1;
            }
            while j < c.len() {
                next.push(c[j]);
                j += 1;
            }

            if next.len() >= remaining.len() {
                continue;
            } // must make progress

            if let Some(suffix) = search_decomp(&next, by_det, graphlike_set, memo) {
                let mut result = vec![candidate.clone()];
                result.extend(suffix);
                result.sort();
                let r = Some(result);
                memo.insert(remaining.clone(), r.clone());
                return r;
            }
        }

        memo.insert(remaining.clone(), None);
        None
    }

    // Step 1: Collect all graphlike mechanisms (building blocks for decomposition)
    let mut graphlike_set: BTreeSet<SmallVec<[usize; 4]>> = BTreeSet::new();
    for entry in entries {
        if entry.probability <= 0.0 {
            continue;
        }
        // Collect graphlike from X/Z components
        if let Some(ref x) = entry.x_component
            && is_graphlike(x)
            && !x.detectors.is_empty()
        {
            graphlike_set.insert(x.detectors.clone());
        }
        if let Some(ref z) = entry.z_component
            && is_graphlike(z)
            && !z.detectors.is_empty()
        {
            graphlike_set.insert(z.detectors.clone());
        }
        // Also from combined event
        if is_graphlike(&entry.event) && !entry.event.detectors.is_empty() {
            graphlike_set.insert(entry.event.detectors.clone());
        }
    }

    // Step 2: Build index for graphlike decomposition search
    let max_det = graphlike_set
        .iter()
        .flat_map(|d| d.iter().copied())
        .max()
        .unwrap_or(0);
    let mut by_det: Vec<Vec<SmallVec<[usize; 4]>>> = vec![Vec::new(); max_det + 1];
    for g in &graphlike_set {
        for &d in g {
            by_det[d].push(g.clone());
        }
    }

    // Step 3: Format entries
    let mut by_targets: BTreeMap<String, f64> = BTreeMap::new();
    let mut memo: DecompMemo = BTreeMap::new();

    for entry in entries {
        if entry.probability <= 0.0 {
            continue;
        }

        let targets = match (&entry.x_component, &entry.z_component) {
            (Some(x), Some(z)) if !x.detectors.is_empty() && !z.detectors.is_empty() => {
                // Both components non-empty: decompose as X ^ Z
                let x_str = format_event(x);
                let z_str = format_event(z);
                if x_str == z_str {
                    x_str
                } else {
                    format!("{x_str} ^ {z_str}")
                }
            }
            (Some(x), _) if !x.detectors.is_empty() || !x.observables.is_empty() => {
                if is_graphlike(x) {
                    format_event(x)
                } else {
                    // Hyperedge X-only: try graphlike decomposition
                    match search_decomp(&x.detectors, &by_det, &graphlike_set, &mut memo) {
                        Some(pieces) => {
                            let mut parts: Vec<String> = pieces
                                .iter()
                                .map(|p| {
                                    p.iter()
                                        .map(|d| format!("D{d}"))
                                        .collect::<Vec<_>>()
                                        .join(" ")
                                })
                                .collect();
                            // Attach observables to first piece
                            if !x.observables.is_empty() && !parts.is_empty() {
                                for &o in &x.observables {
                                    let _ = write!(&mut parts[0], " L{o}");
                                }
                            }
                            parts.join(" ^ ")
                        }
                        None => continue, // drop undecomposable mechanisms
                    }
                }
            }
            (_, Some(z)) if !z.detectors.is_empty() || !z.observables.is_empty() => {
                if is_graphlike(z) {
                    format_event(z)
                } else {
                    // Hyperedge Z-only: try graphlike decomposition
                    match search_decomp(&z.detectors, &by_det, &graphlike_set, &mut memo) {
                        Some(pieces) => {
                            let mut parts: Vec<String> = pieces
                                .iter()
                                .map(|p| {
                                    p.iter()
                                        .map(|d| format!("D{d}"))
                                        .collect::<Vec<_>>()
                                        .join(" ")
                                })
                                .collect();
                            if !z.observables.is_empty() && !parts.is_empty() {
                                for &o in &z.observables {
                                    let _ = write!(&mut parts[0], " L{o}");
                                }
                            }
                            parts.join(" ^ ")
                        }
                        None => continue, // drop undecomposable mechanisms
                    }
                }
            }
            _ => {
                if is_graphlike(&entry.event) {
                    format_event(&entry.event)
                } else {
                    // Combined hyperedge without components: try graphlike decomposition
                    match search_decomp(&entry.event.detectors, &by_det, &graphlike_set, &mut memo)
                    {
                        Some(pieces) => {
                            let mut parts: Vec<String> = pieces
                                .iter()
                                .map(|p| {
                                    p.iter()
                                        .map(|d| format!("D{d}"))
                                        .collect::<Vec<_>>()
                                        .join(" ")
                                })
                                .collect();
                            if !entry.event.observables.is_empty() && !parts.is_empty() {
                                for &o in &entry.event.observables {
                                    let _ = write!(&mut parts[0], " L{o}");
                                }
                            }
                            parts.join(" ^ ")
                        }
                        None => continue,
                    }
                }
            }
        };

        if targets.is_empty() {
            continue;
        }

        by_targets
            .entry(targets)
            .and_modify(|p| {
                *p = *p + entry.probability - 2.0 * *p * entry.probability;
            })
            .or_insert(entry.probability);
    }

    let mut lines = Vec::new();
    for (targets, prob) in &by_targets {
        if *prob > 0.0 {
            lines.push(format!("error({prob:.6e}) {targets}"));
        }
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn z_det(id: usize, qubits: &[usize]) -> Detector {
        let mut bm = Bm::default();
        for &q in qubits {
            bm.z_bits.set_bit(q);
        }
        Detector { id, stabilizer: bm }
    }

    #[test]
    fn test_s_only_probability() {
        // Single S_X generator coefficient -0.01 → (1 - e^{-0.02})/2 ≈ 0.0099007
        let gens = vec![PropagatedEeg {
            eeg_type: EegType::S,
            label: Bm::x(0),
            label2: None,
            coeff: -0.01,
            source: None,
        }];
        let dets = vec![z_det(0, &[0])]; // Z0 anticommutes with X0
        let entries = build_dem(&gens, &dets, &[]);

        assert_eq!(entries.len(), 1);
        let expected = (1.0 - (-0.02_f64).exp()) / 2.0;
        assert!((entries[0].probability - expected).abs() < 1e-6);
    }

    #[test]
    fn test_h_diagonal_probability() {
        // H_X with rate 0.1 → p = 0.1^2 = 0.01 (diagonal approx)
        let gens = vec![PropagatedEeg {
            eeg_type: EegType::H,
            label: Bm::x(0),
            label2: None,
            coeff: 0.1,
            source: None,
        }];
        let dets = vec![z_det(0, &[0])];
        let entries = build_dem(&gens, &dets, &[]);

        assert_eq!(entries.len(), 1);
        assert!((entries[0].probability - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_h_multiple_same_event() {
        // Two H generators in same event class: rates don't add (diagonal approx)
        // p = h1^2 + h2^2 (NOT (h1+h2)^2 — that would be coherent accumulation)
        let gens = vec![
            PropagatedEeg {
                eeg_type: EegType::H,
                label: Bm::x(0),
                label2: None,
                coeff: 0.1,
                source: None,
            },
            PropagatedEeg {
                eeg_type: EegType::H,
                label: Bm::y(0),
                label2: None,
                coeff: 0.05,
                source: None,
            },
        ];
        let dets = vec![z_det(0, &[0])]; // Both X0 and Y0 anticommute with Z0
        let entries = build_dem(&gens, &dets, &[]);

        // Diagonal: p = 0.1^2 + 0.05^2 = 0.01 + 0.0025 = 0.0125
        assert_eq!(entries.len(), 1);
        assert!((entries[0].probability - 0.0125).abs() < 1e-10);
    }

    #[test]
    fn test_z_invisible_to_z_detector() {
        // H_Z should NOT flip a Z-type detector (Z commutes with Z)
        let gens = vec![PropagatedEeg {
            eeg_type: EegType::H,
            label: Bm::z(0),
            label2: None,
            coeff: 0.1,
            source: None,
        }];
        let dets = vec![z_det(0, &[0])];
        let entries = build_dem(&gens, &dets, &[]);

        assert!(entries.is_empty(), "Z should not flip Z detector");
    }

    #[test]
    fn test_bch_same_label_accumulation() {
        // Two H generators with SAME Pauli label: BCH sums coefficients.
        // Two H_X(0) with rates 0.1 and 0.05 → combined rate 0.15 → p = 0.15^2
        let gens = vec![
            PropagatedEeg {
                eeg_type: EegType::H,
                label: Bm::x(0),
                label2: None,
                coeff: 0.1,
                source: None,
            },
            PropagatedEeg {
                eeg_type: EegType::H,
                label: Bm::x(0),
                label2: None,
                coeff: 0.05,
                source: None,
            },
        ];
        let dets = vec![z_det(0, &[0])];
        let entries = build_dem(&gens, &dets, &[]);

        // BCH combines: single generator with rate 0.15, p = 0.15^2 = 0.0225
        assert_eq!(entries.len(), 1);
        assert!(
            (entries[0].probability - 0.0225).abs() < 1e-10,
            "BCH should sum same-label rates: got {}",
            entries[0].probability
        );
    }

    #[test]
    fn test_beta_constructive_interference() {
        // Two H generators Q1=X0, Q2=X1 in same event class (both flip Z0Z1).
        // Q1*Q2 = X0X1. State is |Phi+> Bell state → X0X1 is +1 stabilizer.
        // Constructive: p = (h1+h2)^2
        use crate::stabilizer::StabilizerGroup;
        use pecos_core::gate_type::GateType;
        use pecos_core::{Gate, GateAngles, GateParams, QubitId};

        fn g(gt: GateType, qs: &[usize]) -> Gate {
            Gate {
                gate_type: gt,
                qubits: qs.iter().map(|&q| QubitId(q)).collect(),
                angles: GateAngles::new(),
                params: GateParams::new(),
                meas_ids: pecos_core::GateMeasIds::new(),
                channel: None,
            }
        }

        let gens = vec![
            PropagatedEeg {
                eeg_type: EegType::H,
                label: Bm::x(0),
                label2: None,
                coeff: 0.1,
                source: None,
            },
            PropagatedEeg {
                eeg_type: EegType::H,
                label: Bm::x(1),
                label2: None,
                coeff: 0.05,
                source: None,
            },
        ];
        // Z0Z1 detector: X0 and X1 both anticommute with it
        let dets = vec![Detector {
            id: 0,
            stabilizer: Bm::z(0).multiply(&Bm::z(1)),
        }];

        // |Phi+> = H CX |00> → stabilizers +X0X1, +Z0Z1
        let stab_group =
            StabilizerGroup::from_circuit(&[g(GateType::H, &[0]), g(GateType::CX, &[0, 1])], 2);

        let entries = build_dem_with_stabilizers(&gens, &dets, &[], Some(&stab_group));

        assert_eq!(entries.len(), 1);
        // X0*X1 is +1 stabilizer → constructive: (0.1+0.05)^2 = 0.0225
        assert!(
            (entries[0].probability - 0.0225).abs() < 1e-10,
            "Constructive beta: got {}, expected 0.0225",
            entries[0].probability
        );
    }

    #[test]
    fn test_beta_destructive_interference() {
        // Same event class but |Phi-> state → X0X1 is -1 stabilizer.
        // Destructive: p = (h1-h2)^2
        use crate::stabilizer::StabilizerGroup;
        use pecos_core::gate_type::GateType;
        use pecos_core::{Gate, GateAngles, GateParams, QubitId};

        fn g(gt: GateType, qs: &[usize]) -> Gate {
            Gate {
                gate_type: gt,
                qubits: qs.iter().map(|&q| QubitId(q)).collect(),
                angles: GateAngles::new(),
                params: GateParams::new(),
                meas_ids: pecos_core::GateMeasIds::new(),
                channel: None,
            }
        }

        let gens = vec![
            PropagatedEeg {
                eeg_type: EegType::H,
                label: Bm::x(0),
                label2: None,
                coeff: 0.1,
                source: None,
            },
            PropagatedEeg {
                eeg_type: EegType::H,
                label: Bm::x(1),
                label2: None,
                coeff: 0.05,
                source: None,
            },
        ];
        let dets = vec![Detector {
            id: 0,
            stabilizer: Bm::z(0).multiply(&Bm::z(1)),
        }];

        // |Phi-> = CX H X |00> → stabilizers -X0X1, +Z0Z1
        let stab_group = StabilizerGroup::from_circuit(
            &[
                g(GateType::X, &[0]),
                g(GateType::H, &[0]),
                g(GateType::CX, &[0, 1]),
            ],
            2,
        );

        let entries = build_dem_with_stabilizers(&gens, &dets, &[], Some(&stab_group));

        assert_eq!(entries.len(), 1);
        // X0*X1 is -1 stabilizer → destructive: (0.1-0.05)^2 = 0.0025
        assert!(
            (entries[0].probability - 0.0025).abs() < 1e-10,
            "Destructive beta: got {}, expected 0.0025",
            entries[0].probability
        );
    }

    #[test]
    fn test_beta_equal_rates_destructive_cancels() {
        // h1 = h2 = 0.1, -1 stabilizer product → p = (h1-h2)^2 = 0
        use crate::stabilizer::StabilizerGroup;
        use pecos_core::gate_type::GateType;
        use pecos_core::{Gate, GateAngles, GateParams, QubitId};

        fn g(gt: GateType, qs: &[usize]) -> Gate {
            Gate {
                gate_type: gt,
                qubits: qs.iter().map(|&q| QubitId(q)).collect(),
                angles: GateAngles::new(),
                params: GateParams::new(),
                meas_ids: pecos_core::GateMeasIds::new(),
                channel: None,
            }
        }

        let gens = vec![
            PropagatedEeg {
                eeg_type: EegType::H,
                label: Bm::x(0),
                label2: None,
                coeff: 0.1,
                source: None,
            },
            PropagatedEeg {
                eeg_type: EegType::H,
                label: Bm::x(1),
                label2: None,
                coeff: 0.1,
                source: None,
            },
        ];
        let dets = vec![Detector {
            id: 0,
            stabilizer: Bm::z(0).multiply(&Bm::z(1)),
        }];

        let stab_group = StabilizerGroup::from_circuit(
            &[
                g(GateType::X, &[0]),
                g(GateType::H, &[0]),
                g(GateType::CX, &[0, 1]),
            ],
            2,
        );

        let entries = build_dem_with_stabilizers(&gens, &dets, &[], Some(&stab_group));

        // Complete cancellation: p = 0
        assert!(
            entries.is_empty(),
            "Equal-rate destructive interference should cancel completely"
        );
    }

    #[test]
    fn test_beta_no_stabilizer_diagonal() {
        // When product is NOT in stabilizer group, beta = 0, fall back to diagonal.
        // X0 and X1 both flip Z0Z1 detector. Product X0X1.
        // State |00> has stabilizers Z0, Z1. X0X1 is not a stabilizer.
        use crate::stabilizer::StabilizerGroup;

        let gens = vec![
            PropagatedEeg {
                eeg_type: EegType::H,
                label: Bm::x(0),
                label2: None,
                coeff: 0.1,
                source: None,
            },
            PropagatedEeg {
                eeg_type: EegType::H,
                label: Bm::x(1),
                label2: None,
                coeff: 0.05,
                source: None,
            },
        ];
        let dets = vec![Detector {
            id: 0,
            stabilizer: Bm::z(0).multiply(&Bm::z(1)),
        }];

        let stab_group = StabilizerGroup::from_circuit(&[], 2);

        let entries = build_dem_with_stabilizers(&gens, &dets, &[], Some(&stab_group));

        assert_eq!(entries.len(), 1);
        // p = h1^2 + h2^2 = 0.0125 (diagonal only)
        assert!(
            (entries[0].probability - 0.0125).abs() < 1e-10,
            "Non-stabilizer product → diagonal: got {}",
            entries[0].probability
        );
    }

    #[test]
    fn test_s_exact_formula_multiple() {
        // Multiple S generators in same event class: exact formula
        // p = (1/2)(1 - exp(2 * sum_rates))
        let gens = vec![
            PropagatedEeg {
                eeg_type: EegType::S,
                label: Bm::x(0),
                label2: None,
                coeff: -0.01,
                source: None,
            },
            PropagatedEeg {
                eeg_type: EegType::S,
                label: Bm::y(0),
                label2: None,
                coeff: -0.005,
                source: None,
            },
        ];
        let dets = vec![z_det(0, &[0])];
        let entries = build_dem(&gens, &dets, &[]);

        assert_eq!(entries.len(), 1);
        // sum_rates = -0.01 + -0.005 = -0.015 (same event class, both flip Z0)
        let expected = (1.0 - (2.0 * -0.015_f64).exp()) / 2.0;
        assert!((entries[0].probability - expected).abs() < 1e-10);
    }

    #[test]
    fn test_observable_classification() {
        // Generator that flips an observable but no detectors
        let gens = vec![PropagatedEeg {
            eeg_type: EegType::H,
            label: Bm::x(0),
            label2: None,
            coeff: 0.1,
            source: None,
        }];
        let dets = vec![z_det(0, &[1])]; // Detector on qubit 1
        let obs = vec![Observable {
            id: 0,
            pauli: Bm::z(0),
        }]; // Observable Z0

        let entries = build_dem(&gens, &dets, &obs);

        // X0 anticommutes with Z0 (observable) but not with Z1 (detector)
        assert_eq!(entries.len(), 1);
        assert!(entries[0].event.detectors.is_empty());
        assert_eq!(entries[0].event.observables.as_slice(), &[0]);
    }

    #[test]
    fn test_bch2_anticommuting_h_generators() {
        // [H_X, H_Z] = -2i H_{XZ} = -2 H_Y since XZ = -iY.
        // BCH2 therefore adds a REAL H_Y coefficient -h_x*h_z.
        // X and Y flip Z; their anticommuting cross term vanishes.
        let h_x = 0.1;
        let h_z = 0.05;

        let gens = vec![
            PropagatedEeg {
                eeg_type: EegType::H,
                label: Bm::x(0),
                label2: None,
                coeff: h_x,
                source: None,
            },
            PropagatedEeg {
                eeg_type: EegType::H,
                label: Bm::z(0),
                label2: None,
                coeff: h_z,
                source: None,
            },
        ];
        let dets = vec![z_det(0, &[0])]; // Z detector: X and Y anticommute, Z commutes

        // Without BCH2: only X flips detector. p = h_x² = 0.01
        let entries_k1 =
            build_dem_with_options(&gens, &dets, &[], None, HFormula::Taylor, BchOrder::First);
        let p_k1: f64 = entries_k1.iter().map(|e| e.probability).sum();
        assert!(
            (p_k1 - h_x * h_x).abs() < 1e-10,
            "BCH1: p should be h_x² = {}, got {p_k1}",
            h_x * h_x
        );

        // The added real Y coefficient contributes (h_x*h_z)^2 diagonally.
        let entries_k2 =
            build_dem_with_options(&gens, &dets, &[], None, HFormula::Taylor, BchOrder::Second);
        let p_k2: f64 = entries_k2.iter().map(|e| e.probability).sum();
        let expected_k2 = h_x * h_x + (h_x * h_z).powi(2);
        assert!(
            (p_k2 - expected_k2).abs() < 1e-10,
            "BCH2: p should be h_x² + (h_x·h_z)² = {expected_k2}, got {p_k2}"
        );

        // BCH2 adds a small correction: 0.01 + 0.000025 = 0.010025
        assert!(p_k2 > p_k1, "BCH2 should add to probability");
    }
}
