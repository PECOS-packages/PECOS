//! Phase-exactness diagnostics for PECOS issue #562.
//!
//! This module deliberately calls the shipped forced-projection routine and
//! reconstructs every intermediate `(tableau, MPS)` pair through
//! `StabMps::state_vector`.  It is test-only: none of these dense operations or
//! access paths are compiled into the library.

use super::*;
use crate::stab_mps::StabMps;
use num_complex::Complex64;
use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable};
use std::collections::BTreeMap;

const PHASE_TOLERANCE: f64 = 5e-12;
const VECTOR_TOLERANCE: f64 = 5e-12;

/// Reconstruct an intermediate pair by reusing the production dense
/// reconstruction, including both stabilizer and destabilizer row phases.
fn dense_pair(template: &StabMps, tableau: &SparseStabY, mps: &Mps) -> Vec<Complex64> {
    let mut snapshot = template.clone();
    snapshot.tableau = tableau.clone();
    snapshot.mps = mps.clone();
    snapshot.state_vector()
}

fn normalized_projected_dense(
    before: &[Complex64],
    qubit: usize,
    outcome: bool,
) -> Option<Vec<Complex64>> {
    let mut projected = before.to_vec();
    for (index, amplitude) in projected.iter_mut().enumerate() {
        if ((index >> qubit) & 1 != 0) != outcome {
            *amplitude = Complex64::new(0.0, 0.0);
        }
    }
    let norm_squared: f64 = projected.iter().map(Complex64::norm_sqr).sum();
    if norm_squared <= 1e-20 {
        return None;
    }
    let inverse_norm = Complex64::new(norm_squared.sqrt().recip(), 0.0);
    for amplitude in &mut projected {
        *amplitude *= inverse_norm;
    }
    Some(projected)
}

#[derive(Clone, Copy, Debug)]
struct PhaseComparison {
    power: u8,
    phase_error: f64,
    vector_error: f64,
}

fn compare_up_to_quarter_phase(actual: &[Complex64], expected: &[Complex64]) -> PhaseComparison {
    assert_eq!(actual.len(), expected.len());
    let pivot = expected
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.norm_sqr().total_cmp(&right.norm_sqr()))
        .map(|(index, _)| index)
        .unwrap();
    assert!(expected[pivot].norm() > 1e-12);
    let raw_ratio = actual[pivot] / expected[pivot];
    let phase = raw_ratio / Complex64::new(raw_ratio.norm(), 0.0);
    let roots = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.0, -1.0),
    ];
    let (power, phase_error) = roots
        .iter()
        .enumerate()
        .map(|(power, root)| (power as u8, (phase - root).norm()))
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap();
    let root = roots[usize::from(power)];
    let vector_error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - root * *expected).norm())
        .fold(0.0, f64::max);
    PhaseComparison {
        power,
        phase_error,
        vector_error,
    }
}

fn bits_from_index(index: usize, n: usize) -> Vec<bool> {
    (0..n).map(|qubit| (index >> qubit) & 1 != 0).collect()
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Census-shaped circuit family.  The extended family deliberately retains
/// SZ/CZ/X after non-Clifford gates; the clean family uses only H/RZ/CX.
#[derive(Clone, Copy, Debug, Default)]
#[allow(clippy::struct_excessive_bools)] // Independent census feature flags, not simulator state.
struct CircuitFeatures {
    sz_after_nonclifford: bool,
    x_after_nonclifford: bool,
    cz_after_nonclifford: bool,
    rx: bool,
}

fn random_circuit_with_features(n: usize, seed: u64, extended: bool) -> (StabMps, CircuitFeatures) {
    let mut stn = StabMps::builder(n)
        .seed(seed)
        .merge_rz(false)
        .svd_cutoff(0.0)
        .max_truncation_error(0.0)
        .build();
    let mut rng = seed.wrapping_add(1);
    let mut features = CircuitFeatures::default();
    let mut has_nonclifford = false;
    for _ in 0..32 {
        let gate_type = xorshift(&mut rng) % if extended { 8 } else { 4 };
        let q0 = (xorshift(&mut rng) % n as u64) as usize;
        let q1 = loop {
            let candidate = (xorshift(&mut rng) % n as u64) as usize;
            if candidate != q0 {
                break candidate;
            }
        };
        match gate_type {
            0 => {
                stn.h(&[QubitId(q0)]);
            }
            1 => {
                stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(q0)]);
                has_nonclifford = true;
            }
            2 => {
                let angle_bits = xorshift(&mut rng);
                let angle = Angle64::from_radians(
                    (angle_bits % 1000) as f64 * 0.001 * std::f64::consts::TAU,
                );
                stn.rz(angle, &[QubitId(q0)]);
                has_nonclifford = true;
            }
            3 => {
                stn.cx(&[(QubitId(q0), QubitId(q1))]);
            }
            4 => {
                stn.sz(&[QubitId(q0)]);
                features.sz_after_nonclifford |= has_nonclifford;
            }
            5 => {
                stn.x(&[QubitId(q0)]);
                features.x_after_nonclifford |= has_nonclifford;
            }
            6 => {
                stn.cz(&[(QubitId(q0), QubitId(q1))]);
                features.cz_after_nonclifford |= has_nonclifford;
            }
            _ => {
                let angle_bits = xorshift(&mut rng);
                let angle = Angle64::from_radians(
                    (angle_bits % 1000) as f64 * 0.001 * std::f64::consts::TAU,
                );
                stn.rx(angle, &[QubitId(q0)]);
                features.rx = true;
                has_nonclifford = true;
            }
        }
    }
    stn.flush();
    (stn, features)
}

fn random_circuit(n: usize, seed: u64, extended: bool) -> StabMps {
    random_circuit_with_features(n, seed, extended).0
}

#[derive(Default)]
struct WalkSummary {
    step_phases: BTreeMap<u8, usize>,
    final_phases: BTreeMap<u8, usize>,
    step_features: BTreeMap<String, BTreeMap<u8, usize>>,
    internal_phases: BTreeMap<String, BTreeMap<u8, usize>>,
    internal_non_scalar: BTreeMap<String, usize>,
    first_leak: Option<(usize, usize, Vec<bool>, PhaseComparison)>,
    max_phase_error: f64,
    max_vector_error: f64,
    bitstrings_checked: usize,
    zero_bitstrings: usize,
}

fn row_has_y(gens: &pecos_simulators::Gens, row: usize) -> bool {
    gens.row_x[row]
        .iter()
        .any(|qubit| gens.row_z[row].contains(qubit))
}

fn step_features(
    input_tableau: &SparseStabY,
    input_mps: &Mps,
    qubit: usize,
    outcome: bool,
) -> String {
    if is_mps_trivial(input_mps) {
        return "branch=trivial".to_string();
    }
    let pre_reduce = input_tableau.stabs().col_x[qubit].len().saturating_sub(1);
    let mut tableau = input_tableau.clone();
    let mut mps = input_mps.clone();
    pre_reduce_for_measurement(&mut tableau, &mut mps, qubit, true).unwrap();
    let any_y = (0..tableau.num_qubits())
        .any(|row| row_has_y(tableau.stabs(), row) || row_has_y(tableau.destabs(), row));
    let any_i = !tableau.stabs().signs_i.is_empty() || !tableau.destabs().signs_i.is_empty();
    match decompose_z(tableau.stabs(), tableau.destabs(), qubit) {
        ZDecomposition::Stabilizer { sign_sites, .. } => {
            let relevant_y = sign_sites
                .iter()
                .any(|&row| row_has_y(tableau.stabs(), row));
            let relevant_i = sign_sites.iter().any(|&row| {
                tableau.stabs().signs_i.contains(row) || tableau.destabs().signs_i.contains(row)
            });
            format!(
                "branch=stabilizer pre_reduce={} any_y={any_y} relevant_y={relevant_y} any_i={any_i} relevant_i={relevant_i}",
                pre_reduce > 0
            )
        }
        ZDecomposition::DestabilizerFlip {
            flip_sites,
            phase,
            sign_sites,
        } => {
            let id = flip_sites[0];
            let signed_phase = Complex64::new(if outcome { -1.0 } else { 1.0 }, 0.0) * phase;
            let rotation = if signed_phase.im > 0.5 {
                "imag+"
            } else if signed_phase.im < -0.5 {
                "imag-"
            } else if signed_phase.re < 0.0 {
                "real-"
            } else {
                "real+"
            };
            let relevant_y = row_has_y(tableau.destabs(), id)
                || sign_sites
                    .iter()
                    .any(|&row| row_has_y(tableau.stabs(), row));
            let relevant_i = tableau.destabs().signs_i.contains(id)
                || sign_sites.iter().any(|&row| {
                    tableau.stabs().signs_i.contains(row) || tableau.destabs().signs_i.contains(row)
                });
            format!(
                "branch=flip local={} rotation={rotation} pre_reduce={} any_y={any_y} relevant_y={relevant_y} any_i={any_i} relevant_i={relevant_i}",
                sign_sites.is_empty(),
                pre_reduce > 0
            )
        }
    }
}

fn record_internal(
    records: &mut Vec<(String, PhaseComparison)>,
    label: &str,
    actual: &[Complex64],
    expected: &[Complex64],
) {
    records.push((
        label.to_string(),
        compare_up_to_quarter_phase(actual, expected),
    ));
}

fn diagnostic_pre_reduce(
    template: &StabMps,
    tableau: &mut SparseStabY,
    mps: &mut Mps,
    qubit: usize,
    records: &mut Vec<(String, PhaseComparison)>,
) {
    if tableau.stabs().col_x[qubit].len() <= 1 {
        return;
    }
    let replaced = find_replaced_stabilizer(tableau, qubit);
    let anticommuting: Vec<usize> = tableau.stabs().col_x[qubit]
        .iter()
        .filter(|&row| row != replaced)
        .collect();
    for other in anticommuting {
        let before = dense_pair(template, tableau, mps);
        apply_cnot_to_mps(mps, replaced, other).unwrap();
        crate::stab_mps::tableau_compose::right_compose_cx(tableau, replaced, other);
        record_internal(
            records,
            "pre_reduce right_compose_cx + MPS CNOT",
            &dense_pair(template, tableau, mps),
            &before,
        );
    }
}

fn rotation_operations(
    id: usize,
    phase: Complex64,
    sign_sites: &[usize],
    outcome: bool,
) -> Vec<(&'static str, DeferredOp)> {
    let signed_phase = Complex64::new(if outcome { -1.0 } else { 1.0 }, 0.0) * phase;
    let mut operations = Vec::new();
    if signed_phase.im.abs() < 1e-9 {
        if signed_phase.re < 0.0 {
            operations.push(("right_compose_z + MPS Z", DeferredOp::Z(id)));
        }
        for &site in sign_sites {
            if site != id {
                operations.push(("right_compose_cz + MPS CZ", DeferredOp::Cz(id, site)));
            }
        }
    } else {
        for &site in sign_sites {
            if site != id {
                operations.push(("right_compose_cz + MPS CZ", DeferredOp::Cz(id, site)));
            }
        }
        operations.push(if signed_phase.im > 0.0 {
            ("right_compose_sz + MPS SZdg", DeferredOp::SZ(id))
        } else {
            ("right_compose_szdg + MPS SZ", DeferredOp::SZdg(id))
        });
    }
    operations.push(("right_compose_h + MPS H", DeferredOp::H(id)));
    operations
}

fn right_compose_primitive(tableau: &mut SparseStabY, operation: DeferredOp) {
    match operation {
        DeferredOp::Z(q) => crate::stab_mps::tableau_compose::right_compose_z(tableau, q),
        DeferredOp::Cz(a, b) => crate::stab_mps::tableau_compose::right_compose_cz(tableau, a, b),
        DeferredOp::SZ(q) => crate::stab_mps::tableau_compose::right_compose_sz(tableau, q),
        DeferredOp::SZdg(q) => crate::stab_mps::tableau_compose::right_compose_szdg(tableau, q),
        DeferredOp::H(q) => crate::stab_mps::tableau_compose::right_compose_h(tableau, q),
        DeferredOp::Cnot(c, t) => {
            crate::stab_mps::tableau_compose::right_compose_cx(tableau, c, t);
        }
    }
}

/// Replay one shipped forced-projection step at internal operation boundaries.
/// Every reference vector comes from the same production dense reconstruction
/// used for the outer walk.
fn internal_step_diagnostics(
    template: &StabMps,
    input_tableau: &SparseStabY,
    input_mps: &Mps,
    qubit: usize,
    outcome: bool,
) -> Vec<(String, PhaseComparison)> {
    let mut records = Vec::new();
    let before = dense_pair(template, input_tableau, input_mps);
    let Some(expected) = normalized_projected_dense(&before, qubit, outcome) else {
        return records;
    };
    let mut tableau = input_tableau.clone();
    let mut mps = input_mps.clone();
    if is_mps_trivial(&mps) {
        let previous = dense_pair(template, &tableau, &mps);
        canonicalize_trivial_mps_basis(&mut tableau, &mut mps, None);
        record_internal(
            &mut records,
            "canonicalize_trivial_mps_basis",
            &dense_pair(template, &tableau, &mps),
            &previous,
        );
        tableau.mz_forced(qubit, outcome);
        mps.normalize();
        record_internal(
            &mut records,
            "trivial mz_forced",
            &dense_pair(template, &tableau, &mps),
            &expected,
        );
        return records;
    }

    let norm_squared = mps.norm_squared();
    let expectation =
        (z_expectation_value(&tableau, &mps, qubit).re / norm_squared).clamp(-1.0, 1.0);
    let probability = forced_outcome_probability(expectation, outcome);
    diagnostic_pre_reduce(template, &mut tableau, &mut mps, qubit, &mut records);
    let after_pre_reduce = dense_pair(template, &tableau, &mps);
    record_internal(
        &mut records,
        "pre_reduce_for_measurement total",
        &after_pre_reduce,
        &before,
    );
    match decompose_z(tableau.stabs(), tableau.destabs(), qubit) {
        ZDecomposition::Stabilizer { phase, sign_sites } => {
            apply_pauli_projection(
                &mut mps,
                &[],
                &sign_sites,
                phase,
                if outcome { -1.0 } else { 1.0 },
                probability,
            );
            record_internal(
                &mut records,
                "apply_pauli_projection stabilizer",
                &dense_pair(template, &tableau, &mps),
                &expected,
            );
            if !sign_sites.is_empty() {
                reduce_exact_projection_bonds(&mut mps).unwrap();
            }
            mps.normalize();
            record_internal(
                &mut records,
                "reduce_exact_projection_bonds + normalize",
                &dense_pair(template, &tableau, &mps),
                &expected,
            );
        }
        ZDecomposition::DestabilizerFlip {
            flip_sites,
            phase,
            sign_sites,
        } => {
            let id = flip_sites[0];
            let sign_f = if outcome { -1.0 } else { 1.0 };
            let mut projected_mps = mps.clone();
            apply_pauli_projection(
                &mut projected_mps,
                &flip_sites,
                &sign_sites,
                phase,
                sign_f,
                probability,
            );
            record_internal(
                &mut records,
                "apply_pauli_projection flip reference",
                &dense_pair(template, &tableau, &projected_mps),
                &expected,
            );

            let mut primitive_tableau = tableau.clone();
            let mut primitive_mps = projected_mps;
            for (label, operation) in rotation_operations(id, phase, &sign_sites, outcome) {
                let previous = dense_pair(template, &primitive_tableau, &primitive_mps);
                right_compose_primitive(&mut primitive_tableau, operation);
                apply_inverse_virtual_primitive(&mut primitive_mps, operation);
                record_internal(
                    &mut records,
                    label,
                    &dense_pair(template, &primitive_tableau, &primitive_mps),
                    &previous,
                );
            }

            if sign_sites.is_empty() {
                project_single_flip_without_sign(
                    &mut mps,
                    id,
                    Complex64::new(sign_f, 0.0) * phase,
                    probability,
                );
            } else {
                apply_pauli_projection(
                    &mut mps,
                    &flip_sites,
                    &sign_sites,
                    phase,
                    sign_f,
                    probability,
                );
                collapse_projected_flip_site(&mut mps, id);
            }
            let collapse_label = if sign_sites.is_empty() {
                "project_single_flip_without_sign vs explicit projector + W^-1"
            } else {
                "collapse_projected_flip_site vs explicit W^-1"
            };
            record_internal(
                &mut records,
                collapse_label,
                &mps.state_vector(),
                &primitive_mps.state_vector(),
            );
            let mut predicted = tableau.clone();
            right_compose_measurement_basis_rotation(
                &mut predicted,
                id,
                phase,
                &sign_sites,
                outcome,
                None,
            );
            record_internal(
                &mut records,
                "projection collapse + predicted basis rotation",
                &dense_pair(template, &predicted, &mps),
                &expected,
            );
            let predicted_dense = dense_pair(template, &predicted, &mps);
            tableau.mz_forced(qubit, outcome);
            let gauge_sites = compensate_measurement_pauli_gauge(&mut mps, &predicted, &tableau);
            let gauge_label = if gauge_sites.is_empty() {
                "mz_forced + empty Pauli gauge"
            } else {
                "mz_forced + compensate_measurement_pauli_gauge"
            };
            record_internal(
                &mut records,
                gauge_label,
                &dense_pair(template, &tableau, &mps),
                &predicted_dense,
            );
            let before_reduction = dense_pair(template, &tableau, &mps);
            if !sign_sites.is_empty() {
                reduce_exact_projection_bonds(&mut mps).unwrap();
            }
            mps.normalize();
            record_internal(
                &mut records,
                "forced projection final normalization",
                &dense_pair(template, &tableau, &mps),
                &before_reduction,
            );
        }
    }
    records
}

fn walk_all_bitstrings_with_internal(stn: &StabMps, collect_internal: bool) -> WalkSummary {
    struct Walker<'a> {
        template: &'a StabMps,
        original: Vec<Complex64>,
        summary: WalkSummary,
        collect_internal: bool,
        bits: Vec<bool>,
    }

    impl Walker<'_> {
        fn descend(
            &mut self,
            tableau: &SparseStabY,
            mps: &Mps,
            before: &[Complex64],
            qubit: usize,
            projected_norm: f64,
            accumulated_phase: Complex64,
        ) {
            let n = self.template.num_qubits;
            if qubit == n {
                let index = self
                    .bits
                    .iter()
                    .enumerate()
                    .fold(0usize, |index, (q, bit)| index | (usize::from(*bit) << q));
                let expected = self.original[index];
                if expected.norm() <= 1e-12 {
                    return;
                }
                let mps_index = projected_mps_basis_index(tableau, &self.bits);
                let coefficient = mps.amplitude(&mps_index);
                let terminal_tableau_phase =
                    crate::stab_mps::canonical_ket::terminal_tableau_basis_phase(
                        tableau, &mps_index, &self.bits,
                    );
                let actual = self.template.global_phase
                    * accumulated_phase
                    * terminal_tableau_phase
                    * coefficient
                    / coefficient.norm()
                    * projected_norm;
                let comparison = compare_up_to_quarter_phase(&[actual], &[expected]);
                assert!(comparison.phase_error <= PHASE_TOLERANCE);
                assert!(comparison.vector_error <= VECTOR_TOLERANCE);
                self.summary.max_phase_error =
                    self.summary.max_phase_error.max(comparison.phase_error);
                self.summary.max_vector_error =
                    self.summary.max_vector_error.max(comparison.vector_error);
                *self
                    .summary
                    .final_phases
                    .entry(comparison.power)
                    .or_default() += 1;
                return;
            }

            for outcome in [false, true] {
                let Some(expected_after) = normalized_projected_dense(before, qubit, outcome)
                else {
                    continue;
                };
                let mut next_tableau = tableau.clone();
                let mut next_mps = mps.clone();
                let mut next_phase = accumulated_phase;
                let probability = project_forced_z_with_phase(
                    &mut next_tableau,
                    &mut next_mps,
                    qubit,
                    outcome,
                    &mut next_phase,
                )
                .unwrap();
                if probability <= 1e-20 {
                    continue;
                }
                let mut actual_after = dense_pair(self.template, &next_tableau, &next_mps);
                for amplitude in &mut actual_after {
                    *amplitude *= next_phase;
                }
                let comparison = compare_up_to_quarter_phase(&actual_after, &expected_after);
                assert!(
                    comparison.phase_error <= PHASE_TOLERANCE
                        && comparison.vector_error <= VECTOR_TOLERANCE,
                    "q={qubit} outcome={outcome} phase={comparison:?}"
                );
                self.summary.max_phase_error =
                    self.summary.max_phase_error.max(comparison.phase_error);
                self.summary.max_vector_error =
                    self.summary.max_vector_error.max(comparison.vector_error);
                *self
                    .summary
                    .step_phases
                    .entry(comparison.power)
                    .or_default() += 1;
                let features = step_features(tableau, mps, qubit, outcome);
                *self
                    .summary
                    .step_features
                    .entry(features)
                    .or_default()
                    .entry(comparison.power)
                    .or_default() += 1;
                self.bits.push(outcome);
                if comparison.power != 0 && self.summary.first_leak.is_none() {
                    self.summary.first_leak =
                        Some((qubit, usize::from(outcome), self.bits.clone(), comparison));
                }
                if self.collect_internal {
                    for (label, internal) in
                        internal_step_diagnostics(self.template, tableau, mps, qubit, outcome)
                    {
                        if internal.vector_error <= VECTOR_TOLERANCE {
                            *self
                                .summary
                                .internal_phases
                                .entry(label)
                                .or_default()
                                .entry(internal.power)
                                .or_default() += 1;
                        } else {
                            *self.summary.internal_non_scalar.entry(label).or_default() += 1;
                        }
                    }
                }
                self.descend(
                    &next_tableau,
                    &next_mps,
                    &actual_after,
                    qubit + 1,
                    projected_norm * probability.sqrt(),
                    next_phase,
                );
                self.bits.pop();
            }
        }
    }

    let original = stn.state_vector();
    let mut walker = Walker {
        template: stn,
        original: original.clone(),
        summary: WalkSummary::default(),
        collect_internal,
        bits: Vec::new(),
    };
    walker.descend(
        &stn.tableau,
        &stn.mps,
        &original,
        0,
        1.0,
        Complex64::new(1.0, 0.0),
    );
    for (index, &expected) in original.iter().enumerate() {
        let bits = bits_from_index(index, stn.num_qubits);
        let actual = stn.amplitude_iterative(&bits);
        walker.summary.bitstrings_checked += 1;
        if expected.norm() <= 1e-12 {
            walker.summary.zero_bitstrings += 1;
            assert!(
                actual.norm() <= VECTOR_TOLERANCE,
                "zero dense amplitude had nonzero iterative result: bits={bits:?} actual={actual}"
            );
        } else {
            let comparison = compare_up_to_quarter_phase(&[actual], &[expected]);
            assert!(
                comparison.phase_error <= PHASE_TOLERANCE
                    && comparison.vector_error <= VECTOR_TOLERANCE,
                "end-to-end amplitude was not a fourth-root phase ratio: bits={bits:?} comparison={comparison:?}"
            );
        }
    }
    walker.summary
}

fn walk_all_bitstrings(stn: &StabMps) -> WalkSummary {
    walk_all_bitstrings_with_internal(stn, false)
}

fn print_comparison(label: &str, actual: &[Complex64], expected: &[Complex64]) {
    let comparison = compare_up_to_quarter_phase(actual, expected);
    eprintln!(
        "    {label:34} i^{} phase_err={:.2e} vector_err={:.2e}",
        comparison.power, comparison.phase_error, comparison.vector_error
    );
}

fn apply_inverse_virtual_primitive(mps: &mut Mps, operation: DeferredOp) {
    let inverse = match operation {
        DeferredOp::SZ(q) => DeferredOp::SZdg(q),
        DeferredOp::SZdg(q) => DeferredOp::SZ(q),
        other => other,
    };
    flush_deferred_ops(mps, &mut vec![inverse]).unwrap();
}

fn trace_forced_step(
    template: &StabMps,
    input_tableau: &SparseStabY,
    input_mps: &Mps,
    qubit: usize,
    outcome: bool,
) {
    let before = dense_pair(template, input_tableau, input_mps);
    let expected = normalized_projected_dense(&before, qubit, outcome).unwrap();
    let mut tableau = input_tableau.clone();
    let mut mps = input_mps.clone();
    eprintln!("trace q={qubit} outcome={outcome}");

    if is_mps_trivial(&mps) {
        canonicalize_trivial_mps_basis(&mut tableau, &mut mps, None);
        print_comparison(
            "canonicalize_trivial_mps_basis",
            &dense_pair(template, &tableau, &mps),
            &before,
        );
        tableau.mz_forced(qubit, outcome);
        mps.normalize();
        print_comparison(
            "trivial mz_forced",
            &dense_pair(template, &tableau, &mps),
            &expected,
        );
        return;
    }

    let norm_squared = mps.norm_squared();
    let expectation =
        (z_expectation_value(&tableau, &mps, qubit).re / norm_squared).clamp(-1.0, 1.0);
    let probability = forced_outcome_probability(expectation, outcome);
    pre_reduce_for_measurement(&mut tableau, &mut mps, qubit, true).unwrap();
    let after_pre_reduce = dense_pair(template, &tableau, &mps);
    print_comparison("pre_reduce_for_measurement", &after_pre_reduce, &before);
    let decomposition = decompose_z(tableau.stabs(), tableau.destabs(), qubit);
    eprintln!("    decomposition={decomposition:?} probability={probability:.16e}");

    match decomposition {
        ZDecomposition::Stabilizer { phase, sign_sites } => {
            apply_pauli_projection(
                &mut mps,
                &[],
                &sign_sites,
                phase,
                if outcome { -1.0 } else { 1.0 },
                probability,
            );
            print_comparison(
                "apply_pauli_projection(stab)",
                &dense_pair(template, &tableau, &mps),
                &expected,
            );
            if !sign_sites.is_empty() {
                reduce_exact_projection_bonds(&mut mps).unwrap();
            }
            mps.normalize();
            print_comparison(
                "reduce + normalize",
                &dense_pair(template, &tableau, &mps),
                &expected,
            );
        }
        ZDecomposition::DestabilizerFlip {
            flip_sites,
            phase,
            sign_sites,
        } => {
            let id = flip_sites[0];
            let sign_f = if outcome { -1.0 } else { 1.0 };
            if sign_sites.is_empty() {
                project_single_flip_without_sign(
                    &mut mps,
                    id,
                    Complex64::new(sign_f, 0.0) * phase,
                    probability,
                );
                // This specialized operation also absorbs the basis H, so it
                // is only phase-comparable after the tableau basis rotation.
            } else {
                apply_pauli_projection(
                    &mut mps,
                    &flip_sites,
                    &sign_sites,
                    phase,
                    sign_f,
                    probability,
                );
                print_comparison(
                    "apply_pauli_projection(flip)",
                    &dense_pair(template, &tableau, &mps),
                    &expected,
                );
                collapse_projected_flip_site(&mut mps, id);
            }

            let signed_phase = Complex64::new(sign_f, 0.0) * phase;
            let mut primitive_tableau = tableau.clone();
            let mut primitive_mps = if sign_sites.is_empty() {
                // The local fast path already absorbed H. Reconstruct the
                // pre-collapse projected eigenstate for primitive tests.
                let mut projected = input_mps.clone();
                pre_reduce_for_measurement(&mut input_tableau.clone(), &mut projected, qubit, true)
                    .unwrap();
                apply_pauli_projection(
                    &mut projected,
                    &flip_sites,
                    &sign_sites,
                    phase,
                    sign_f,
                    probability,
                );
                projected
            } else {
                // Undo the collapse solely for independent primitive tests by
                // rebuilding the projected coefficient state.
                let mut projected = input_mps.clone();
                let mut projected_tableau = input_tableau.clone();
                pre_reduce_for_measurement(&mut projected_tableau, &mut projected, qubit, true)
                    .unwrap();
                primitive_tableau = projected_tableau;
                apply_pauli_projection(
                    &mut projected,
                    &flip_sites,
                    &sign_sites,
                    phase,
                    sign_f,
                    probability,
                );
                projected
            };
            let primitive_reference = dense_pair(template, &primitive_tableau, &primitive_mps);
            let id_in_sign = sign_sites.contains(&id);
            let mut operations = Vec::new();
            if signed_phase.im.abs() < 1e-9 {
                if signed_phase.re < 0.0 {
                    operations.push(("right_compose_z", DeferredOp::Z(id)));
                }
                for &site in &sign_sites {
                    if site != id {
                        operations.push(("right_compose_cz", DeferredOp::Cz(id, site)));
                    }
                }
            } else {
                assert!(id_in_sign);
                for &site in &sign_sites {
                    if site != id {
                        operations.push(("right_compose_cz", DeferredOp::Cz(id, site)));
                    }
                }
                operations.push(if signed_phase.im > 0.0 {
                    ("right_compose_sz", DeferredOp::SZ(id))
                } else {
                    ("right_compose_szdg", DeferredOp::SZdg(id))
                });
            }
            operations.push(("right_compose_h", DeferredOp::H(id)));
            for (label, operation) in operations {
                match operation {
                    DeferredOp::Z(q) => {
                        crate::stab_mps::tableau_compose::right_compose_z(
                            &mut primitive_tableau,
                            q,
                        );
                    }
                    DeferredOp::Cz(a, b) => crate::stab_mps::tableau_compose::right_compose_cz(
                        &mut primitive_tableau,
                        a,
                        b,
                    ),
                    DeferredOp::SZ(q) => crate::stab_mps::tableau_compose::right_compose_sz(
                        &mut primitive_tableau,
                        q,
                    ),
                    DeferredOp::SZdg(q) => crate::stab_mps::tableau_compose::right_compose_szdg(
                        &mut primitive_tableau,
                        q,
                    ),
                    DeferredOp::H(q) => {
                        crate::stab_mps::tableau_compose::right_compose_h(
                            &mut primitive_tableau,
                            q,
                        );
                    }
                    DeferredOp::Cnot(_, _) => unreachable!(),
                }
                apply_inverse_virtual_primitive(&mut primitive_mps, operation);
                print_comparison(
                    label,
                    &dense_pair(template, &primitive_tableau, &primitive_mps),
                    &primitive_reference,
                );
            }

            let mut predicted_tableau = tableau.clone();
            right_compose_measurement_basis_rotation(
                &mut predicted_tableau,
                id,
                phase,
                &sign_sites,
                outcome,
                None,
            );
            print_comparison(
                "collapse + predicted rotation",
                &dense_pair(template, &predicted_tableau, &mps),
                &expected,
            );

            tableau.mz_forced(qubit, outcome);
            print_comparison(
                "mz_forced before gauge",
                &dense_pair(template, &tableau, &mps),
                &expected,
            );
            let gauge_sites =
                compensate_measurement_pauli_gauge(&mut mps, &predicted_tableau, &tableau);
            eprintln!("    gauge_sites={gauge_sites:?}");
            print_comparison(
                "compensate_pauli_gauge",
                &dense_pair(template, &tableau, &mps),
                &expected,
            );
            if !sign_sites.is_empty() {
                reduce_exact_projection_bonds(&mut mps).unwrap();
            }
            mps.normalize();
            print_comparison(
                "final reduce + normalize",
                &dense_pair(template, &tableau, &mps),
                &expected,
            );
        }
    }
}

#[test]
#[ignore = "issue #562 diagnostic census"]
fn phase_walk_exploration() {
    for extended in [false, true] {
        for n in 3..=6 {
            for seed in 0..24 {
                let stn = random_circuit(n, seed, extended);
                let summary = walk_all_bitstrings(&stn);
                if summary.step_phases.keys().any(|power| *power != 0)
                    || summary.final_phases.keys().any(|power| *power != 0)
                {
                    eprintln!(
                        "extended={extended} n={n} seed={seed} steps={:?} final={:?} first={:?}",
                        summary.step_phases, summary.final_phases, summary.first_leak
                    );
                }
            }
        }
    }
}

fn merge_counts(target: &mut BTreeMap<u8, usize>, source: &BTreeMap<u8, usize>) {
    for (&power, &count) in source {
        *target.entry(power).or_default() += count;
    }
}

fn merge_nested_counts(
    target: &mut BTreeMap<String, BTreeMap<u8, usize>>,
    source: &BTreeMap<String, BTreeMap<u8, usize>>,
) {
    for (label, counts) in source {
        merge_counts(target.entry(label.clone()).or_default(), counts);
    }
}

fn clean_validation_circuit(n: usize, seed: u64) -> StabMps {
    let mut stn = StabMps::builder(n)
        .seed(seed)
        .merge_rz(false)
        .svd_cutoff(0.0)
        .max_truncation_error(0.0)
        .build();
    stn.h(&(0..n).map(QubitId).collect::<Vec<_>>());
    for q in 0..n {
        let angle = Angle64::from_radians(
            ((seed.wrapping_mul(137) + q as u64 * 251) % 1000) as f64
                * 0.001
                * std::f64::consts::TAU,
        );
        stn.rz(angle, &[QubitId(q)]);
    }
    for q in 0..n - 1 {
        stn.cx(&[(QubitId(q), QubitId(q + 1))]);
    }
    stn.flush();
    stn
}

/// Harness self-check: this real-Clifford-prefix H/RZ/CX family is known not
/// to fire #562. Every outer forced-projection step and every final amplitude
/// must retain phase exactly.
#[test]
fn clean_h_rz_cx_walk_is_phase_exact() {
    let mut step_count = 0usize;
    let mut final_count = 0usize;
    let mut max_phase_error = 0.0f64;
    let mut max_vector_error = 0.0f64;
    let mut bitstrings_checked = 0usize;
    let mut zero_bitstrings = 0usize;
    for n in 3..=6 {
        for seed in 0..4 {
            let summary = walk_all_bitstrings(&clean_validation_circuit(n, seed));
            assert_eq!(
                summary.step_phases.keys().copied().collect::<Vec<_>>(),
                vec![0]
            );
            assert_eq!(
                summary.final_phases.keys().copied().collect::<Vec<_>>(),
                vec![0]
            );
            step_count += summary.step_phases.values().sum::<usize>();
            final_count += summary.final_phases.values().sum::<usize>();
            max_phase_error = max_phase_error.max(summary.max_phase_error);
            max_vector_error = max_vector_error.max(summary.max_vector_error);
            bitstrings_checked += summary.bitstrings_checked;
            zero_bitstrings += summary.zero_bitstrings;
        }
    }
    eprintln!(
        "clean self-check: steps={step_count} finals={final_count} bitstrings={bitstrings_checked} zeros={zero_bitstrings} max_phase_error={max_phase_error:.3e} max_vector_error={max_vector_error:.3e}"
    );
}

/// Required n=3..=6, 24-seed, all-bitstring issue census. Kept ignored because
/// dense reconstruction at every tree edge is intentionally expensive.
#[test]
#[ignore = "release-gated issue #562 full dense phase census"]
fn extended_phase_census_report() {
    let mut step_phases = BTreeMap::new();
    let mut final_phases = BTreeMap::new();
    let mut step_features = BTreeMap::new();
    let mut internal_phases = BTreeMap::new();
    let mut internal_non_scalar: BTreeMap<String, usize> = BTreeMap::new();
    let mut circuit_feature_results: BTreeMap<String, [usize; 2]> = BTreeMap::new();
    let mut leaking_circuits = 0usize;
    let mut max_phase_error = 0.0f64;
    let mut max_vector_error = 0.0f64;
    let mut bitstrings_checked = 0usize;
    let mut zero_bitstrings = 0usize;
    for n in 3..=6 {
        for seed in 0..24 {
            let (stn, features) = random_circuit_with_features(n, seed, true);
            let summary = walk_all_bitstrings_with_internal(&stn, true);
            let leaks = summary.step_phases.keys().any(|power| *power != 0)
                || summary.final_phases.keys().any(|power| *power != 0);
            leaking_circuits += usize::from(leaks);
            max_phase_error = max_phase_error.max(summary.max_phase_error);
            max_vector_error = max_vector_error.max(summary.max_vector_error);
            bitstrings_checked += summary.bitstrings_checked;
            zero_bitstrings += summary.zero_bitstrings;
            let feature_key = format!(
                "sz_after={} x_after={} cz_after={} rx={}",
                features.sz_after_nonclifford,
                features.x_after_nonclifford,
                features.cz_after_nonclifford,
                features.rx
            );
            circuit_feature_results.entry(feature_key).or_default()[usize::from(leaks)] += 1;
            merge_counts(&mut step_phases, &summary.step_phases);
            merge_counts(&mut final_phases, &summary.final_phases);
            merge_nested_counts(&mut step_features, &summary.step_features);
            merge_nested_counts(&mut internal_phases, &summary.internal_phases);
            for (label, count) in summary.internal_non_scalar {
                *internal_non_scalar.entry(label).or_default() += count;
            }
        }
    }
    eprintln!("circuits=96 leaking_circuits={leaking_circuits}");
    eprintln!("bitstrings_checked={bitstrings_checked} zero_bitstrings={zero_bitstrings}");
    eprintln!("max_phase_error={max_phase_error:.3e} max_vector_error={max_vector_error:.3e}");
    eprintln!("step_phases={step_phases:?}");
    eprintln!("final_phases={final_phases:?}");
    eprintln!("circuit_feature_results [clean, leak]={circuit_feature_results:#?}");
    eprintln!("step_features={step_features:#?}");
    eprintln!("internal_phases={internal_phases:#?}");
    eprintln!("internal_non_scalar={internal_non_scalar:#?}");
    assert_eq!(leaking_circuits, 0, "phase-tracked forced walk leaked");
    assert_eq!(step_phases.keys().copied().collect::<Vec<_>>(), vec![0]);
    assert_eq!(final_phases.keys().copied().collect::<Vec<_>>(), vec![0]);
}

#[test]
#[ignore = "issue #562 targeted trace"]
fn trace_first_exploratory_leak() {
    let stn = random_circuit(3, 0, false);
    trace_forced_step(&stn, &stn.tableau, &stn.mps, 0, true);
}

#[test]
#[ignore = "issue #562 minimal trace search"]
fn trace_one_qubit_h_rz() {
    let mut stn = StabMps::builder(1)
        .merge_rz(false)
        .svd_cutoff(0.0)
        .max_truncation_error(0.0)
        .build();
    stn.h(&[QubitId(0)]);
    stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
    stn.flush();
    eprintln!("state={:?}", stn.state_vector());
    eprintln!(
        "iter0={} iter1={}",
        stn.amplitude_iterative(&[false]),
        stn.amplitude_iterative(&[true])
    );
    trace_forced_step(&stn, &stn.tableau, &stn.mps, 0, false);
    trace_forced_step(&stn, &stn.tableau, &stn.mps, 0, true);
}

#[test]
#[ignore = "issue #562 candidate minimal trace"]
fn trace_two_qubit_bell_rz() {
    let mut stn = StabMps::builder(2)
        .merge_rz(false)
        .svd_cutoff(0.0)
        .max_truncation_error(0.0)
        .build();
    stn.h(&[QubitId(0)]);
    stn.cx(&[(QubitId(0), QubitId(1))]);
    stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(1)]);
    stn.flush();
    eprintln!("state={:?}", stn.state_vector());
    for index in 0..4 {
        let bits = bits_from_index(index, 2);
        eprintln!(
            "bits={bits:?} dense={} iter={}",
            stn.amplitude(&bits),
            stn.amplitude_iterative(&bits)
        );
    }
    trace_forced_step(&stn, &stn.tableau, &stn.mps, 0, false);
    trace_forced_step(&stn, &stn.tableau, &stn.mps, 0, true);
    trace_forced_step(&stn, &stn.tableau, &stn.mps, 1, false);
    trace_forced_step(&stn, &stn.tableau, &stn.mps, 1, true);
}

#[test]
fn dense_pair_helper_reuses_state_vector_exactly() {
    let stn = random_circuit(3, 7, true);
    assert_eq!(dense_pair(&stn, &stn.tableau, &stn.mps), stn.state_vector());
    for index in 0..(1usize << stn.num_qubits) {
        let bits = bits_from_index(index, stn.num_qubits);
        assert_eq!(stn.amplitude(&bits), stn.state_vector()[index]);
    }
}

/// Exhaustively validate the polynomial canonical-ket amplitude against the
/// dense projector convention used by `StabMps::state_vector`.
#[test]
fn canonical_ket_amplitude_matches_dense_projector_exhaustively() {
    let mut tableaux_checked = 0usize;
    let mut amplitudes_checked = 0usize;
    let mut saw_y = false;
    let mut saw_minus = false;
    let mut saw_sign_i = false;

    for n in 1..=6 {
        for seed in 0..24_u64 {
            let mut tableau = SparseStabY::with_seed(n, seed).with_destab_sign_tracking();
            let mut random = seed.wrapping_add(1) ^ ((n as u64) << 48);
            for gate_index in 0..48 {
                let gate = xorshift(&mut random) % if n == 1 { 4 } else { 7 };
                let q0 = (xorshift(&mut random) % n as u64) as usize;
                let q1 = if n == 1 {
                    0
                } else {
                    loop {
                        let candidate = (xorshift(&mut random) % n as u64) as usize;
                        if candidate != q0 {
                            break candidate;
                        }
                    }
                };
                match gate {
                    0 => {
                        tableau.h(&[QubitId(q0)]);
                    }
                    1 => {
                        tableau.sz(&[QubitId(q0)]);
                    }
                    2 => {
                        tableau.x(&[QubitId(q0)]);
                    }
                    3 => {
                        tableau.y(&[QubitId(q0)]);
                    }
                    4 => {
                        tableau.cx(&[(QubitId(q0), QubitId(q1))]);
                    }
                    5 => {
                        tableau.cz(&[(QubitId(q0), QubitId(q1))]);
                    }
                    _ => {
                        tableau.sz(&[QubitId(q0)]).h(&[QubitId(q0)]);
                    }
                }

                // Exercise the Y-convention measurement row-reduction path,
                // including its signs_i bookkeeping, on reachable tableaux.
                if gate_index % 7 == 6 {
                    let outcome = xorshift(&mut random) & 1 != 0;
                    tableau.mz_forced(q0, outcome);
                }
            }

            for row in 0..n {
                saw_y |= row_has_y(tableau.stabs(), row) || row_has_y(tableau.destabs(), row);
            }
            saw_minus |= !tableau.stabs().signs_minus.is_empty()
                || !tableau.destabs().signs_minus.is_empty();
            saw_sign_i |=
                !tableau.stabs().signs_i.is_empty() || !tableau.destabs().signs_i.is_empty();

            let mut dense_snapshot = StabMps::new(n);
            dense_snapshot.tableau = tableau.clone();
            let dense = dense_snapshot.state_vector();
            for (index, expected) in dense.into_iter().enumerate() {
                let bits = bits_from_index(index, n);
                let actual =
                    crate::stab_mps::canonical_ket::CanonicalKet::new(&tableau).amplitude(&bits);
                assert!(
                    (actual - expected).norm() <= 1e-12,
                    "n={n} seed={seed} index={index}: canonical={actual:?}, dense={expected:?}"
                );
                amplitudes_checked += 1;
            }
            tableaux_checked += 1;
        }
    }

    assert!(saw_y, "random validation never reached a Y-carrying row");
    assert!(saw_minus, "random validation never reached a negative row");
    // Valid Hermitian SparseStabY generators finish Clifford/measurement
    // updates with real stored signs. The separate signed-Pauli action test
    // exercises signs_i because it matters for destabilizer matrix action.
    assert!(!saw_sign_i);
    eprintln!(
        "canonical ket validation: tableaux={tableaux_checked} amplitudes={amplitudes_checked} Y={saw_y} minus={saw_minus} reachable_signs_i={saw_sign_i}"
    );
}

/// Smallest exhaustive-search reproducer over {H, T, SZ, X, CX, CZ}: one
/// qubit and three gates. The shipped amplitude is now exact while the raw
/// internal trace continues to bind the right-compose-H localization.
#[test]
fn smallest_reproducer_localizes_to_right_compose_h() {
    let mut stn = StabMps::builder(1)
        .merge_rz(false)
        .svd_cutoff(0.0)
        .max_truncation_error(0.0)
        .build();
    stn.h(&[QubitId(0)]);
    stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
    stn.sz(&[QubitId(0)]);
    stn.flush();

    let dense = stn.amplitude(&[true]);
    let iterative = stn.amplitude_iterative(&[true]);
    let ratio = iterative / dense;
    assert!((ratio - Complex64::new(1.0, 0.0)).norm() <= PHASE_TOLERANCE);

    let summary = walk_all_bitstrings_with_internal(&stn, true);
    assert_eq!(summary.step_phases, BTreeMap::from([(0, 2)]));
    assert_eq!(summary.final_phases, BTreeMap::from([(0, 2)]));
    assert_eq!(
        summary.internal_phases["right_compose_h + MPS H"],
        BTreeMap::from([(0, 1), (3, 1)])
    );
    for (label, phases) in &summary.internal_phases {
        if label != "right_compose_h + MPS H"
            && label != "projection collapse + predicted basis rotation"
        {
            assert_eq!(
                phases.keys().copied().collect::<Vec<_>>(),
                vec![0],
                "unexpected scalar introduced by {label}: {phases:?}"
            );
        }
    }
}

/// Fixed-workload timing/evidence lane for the original eight-qubit #562 census.
/// Kept ignored because it exhausts every amplitude of six dense-reference states.
#[test]
#[ignore = "issue #562 eight-qubit amplitude evidence and timing"]
fn issue_562_eight_qubit_amplitude_evidence() {
    let mut cases = Vec::new();
    for seed in 0..6 {
        let mut stn = crate::stab_mps::tests::stability_census_random_circuit(seed);
        stn.flush();
        let expected = stn.state_vector();
        cases.push((seed, stn, expected));
    }

    let started = std::time::Instant::now();
    let mut max_delta = 0.0_f64;
    let mut checksum = Complex64::new(0.0, 0.0);
    for (seed, stn, expected) in &cases {
        for (index, expected) in expected.iter().enumerate() {
            let bits = bits_from_index(index, 8);
            let actual = stn.amplitude_iterative(&bits);
            max_delta = max_delta.max((actual - expected).norm());
            checksum += actual * Complex64::new(1.0 + *seed as f64, index as f64 + 1.0);
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    eprintln!(
        "issue #562 8q: circuits=6 amplitudes=1536 elapsed={elapsed:.6}s max_delta={max_delta:.3e} checksum={checksum:?}"
    );
    assert!(
        max_delta <= 1e-12,
        "eight-qubit #562 census max amplitude delta was {max_delta:.3e}"
    );
}
