// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Statistical falsifier for the default `StabMps` random-measurement route.
//!
//! The default route is exact sample-then-force. Debug builds run a tiny smoke
//! subset; optimized builds run the full fast surface matrix, while larger
//! matrices remain in the explicit ignored release lane. The liveness
//! meta-test selects `Pragmatic` explicitly and proves the known bias remains
//! visible to the harness.

use num_complex::Complex64;
use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, DenseStateVec};
use pecos_stab_tn::mps::MpsConfig;
use pecos_stab_tn::stab_mps::mast::Mast;
use pecos_stab_tn::stab_mps::{MeasurementMode, PauliKind, StabMps};
use rayon::prelude::*;

#[cfg(debug_assertions)]
const DEBUG_SHOTS: usize = 16;
#[cfg(not(debug_assertions))]
const FAST_SHOTS: usize = 2_048;
const RELEASE_SHOTS: usize = 1_024;
#[cfg(debug_assertions)]
const META_SHOTS: usize = 32;
#[cfg(not(debug_assertions))]
const META_SHOTS: usize = 256;
const PROBABILITY_EPSILON: f64 = 1e-14;

#[derive(Clone, Copy, Debug)]
enum Gate {
    H(usize),
    Sz(usize),
    X(usize),
    Cx(usize, usize),
    Cz(usize, usize),
    Rz(usize, Angle64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Surface {
    MzMid,
    MzEnd,
    Reset,
    Pz,
    Px,
    ExtractSyndromes,
    Continuation,
    SampleBitstring,
    SampleBitstrings,
}

impl Surface {
    fn name(self) -> &'static str {
        match self {
            Self::MzMid => "mz_mid",
            Self::MzEnd => "mz_end",
            Self::Reset => "reset_qubit",
            Self::Pz => "pz",
            Self::Px => "px",
            Self::ExtractSyndromes => "extract_syndromes",
            Self::Continuation => "continuation",
            Self::SampleBitstring => "sample_bitstring",
            Self::SampleBitstrings => "sample_bitstrings",
        }
    }

    fn record_width(self, num_qubits: usize) -> usize {
        match self {
            Self::MzMid | Self::ExtractSyndromes => 1,
            Self::MzEnd | Self::Pz | Self::Px | Self::SampleBitstring | Self::SampleBitstrings => {
                num_qubits
            }
            Self::Reset | Self::Continuation => num_qubits + 1,
        }
    }
}

/// Deterministic Clifford+rotation family with non-diagonal continuation.
///
/// Every non-Clifford RZ/T is followed by SZ, CZ, X, H, and CX operations.
/// In particular, this is not a diagonal-tail toy: the H after every rotation
/// changes basis, while CZ/CX spread the resulting frame across the circuit.
fn honest_family(num_qubits: usize, phase: usize) -> Vec<Gate> {
    assert!(num_qubits >= 2);
    let mut gates = vec![Gate::H(phase % num_qubits)];
    if num_qubits > 2 {
        gates.push(Gate::H((phase + 2) % num_qubits));
    }
    for q in 0..num_qubits - 1 {
        gates.push(Gate::Cx(q, q + 1));
    }

    for layer in 0..num_qubits + 2 {
        let target = (layer + phase) % num_qubits;
        let neighbor = (target + 1) % num_qubits;
        let other = (target + 2) % num_qubits;
        let angle = if layer % 2 == 0 {
            Angle64::QUARTER_TURN / 2_u64
        } else {
            Angle64::from_radians(0.37 + 0.11 * layer as f64)
        };
        gates.extend([
            Gate::Rz(target, angle),
            Gate::Sz(neighbor),
            Gate::Cz(target, neighbor),
            Gate::X(other),
            Gate::H(target),
            Gate::Cx(neighbor, other),
        ]);
    }
    gates
}

fn continuation_family(num_qubits: usize) -> Vec<Gate> {
    vec![
        Gate::Rz(0, Angle64::from_radians(0.731)),
        Gate::Sz(1 % num_qubits),
        Gate::Cz(0, 1 % num_qubits),
        Gate::X(2 % num_qubits),
        Gate::H(0),
        Gate::Cx(1 % num_qubits, 2 % num_qubits),
    ]
}

fn replay_gates<S>(simulator: &mut S, gates: &[Gate])
where
    S: CliffordGateable + ArbitraryRotationGateable,
{
    for &gate in gates {
        match gate {
            Gate::H(q) => simulator.h(&[QubitId(q)]),
            Gate::Sz(q) => simulator.sz(&[QubitId(q)]),
            Gate::X(q) => simulator.x(&[QubitId(q)]),
            Gate::Cx(control, target) => simulator.cx(&[(QubitId(control), QubitId(target))]),
            Gate::Cz(first, second) => simulator.cz(&[(QubitId(first), QubitId(second))]),
            Gate::Rz(q, angle) => simulator.rz(angle, &[QubitId(q)]),
        };
    }
}

fn non_clifford_count(gates: &[Gate]) -> usize {
    gates
        .iter()
        .filter(|gate| matches!(gate, Gate::Rz(_, _)))
        .count()
}

fn exact_mps_config() -> MpsConfig {
    MpsConfig {
        // The release matrix has at most 15 MAST sites (six data, eight
        // preparation injections, and one continuation injection), whose
        // exact Schmidt-rank ceiling is 2^floor(15/2) = 128.
        max_bond_dim: 128,
        svd_cutoff: 0.0,
        max_truncation_error: Some(0.0),
        parallel: false,
    }
}

fn build_stn_with_mode(num_qubits: usize, seed: u64, mode: MeasurementMode) -> StabMps {
    StabMps::builder(num_qubits)
        .seed(seed)
        .measurement(mode)
        .merge_rz(false)
        .max_bond_dim(64)
        .svd_cutoff(0.0)
        .max_truncation_error(0.0)
        .build()
}

fn build_stn(num_qubits: usize, seed: u64) -> StabMps {
    StabMps::builder(num_qubits)
        .seed(seed)
        .merge_rz(false)
        .max_bond_dim(64)
        .svd_cutoff(0.0)
        .max_truncation_error(0.0)
        .build()
}

fn build_mast(num_qubits: usize, capacity: usize, seed: u64) -> Mast {
    Mast::with_seed(num_qubits, capacity, seed).with_mps_config(exact_mps_config())
}

fn measure_all<S: CliffordGateable>(simulator: &mut S, num_qubits: usize) -> Vec<bool> {
    simulator
        .mz(&(0..num_qubits).map(QubitId).collect::<Vec<_>>())
        .into_iter()
        .map(|result| result.outcome)
        .collect()
}

fn reset_through_measurement<S: CliffordGateable>(simulator: &mut S, q: usize) -> bool {
    let outcome = simulator.mz(&[QubitId(q)])[0].outcome;
    if outcome {
        simulator.x(&[QubitId(q)]);
    }
    outcome
}

fn pz_through_measurement<S: CliffordGateable>(simulator: &mut S, q: usize) {
    reset_through_measurement(simulator, q);
}

fn px_through_measurement<S: CliffordGateable>(simulator: &mut S, q: usize) {
    reset_through_measurement(simulator, q);
    simulator.h(&[QubitId(q)]);
}

fn syndrome_generator(num_qubits: usize) -> Vec<(usize, PauliKind)> {
    let num_data = num_qubits - 1;
    (0..num_data)
        .map(|q| {
            let kind = match q % 3 {
                0 => PauliKind::X,
                1 => PauliKind::Z,
                _ => PauliKind::Y,
            };
            (q, kind)
        })
        .collect()
}

fn extract_one_syndrome_through_measurement<S: CliffordGateable>(
    simulator: &mut S,
    num_qubits: usize,
) -> bool {
    let ancilla = num_qubits - 1;
    px_through_measurement(simulator, ancilla);
    for (data, kind) in syndrome_generator(num_qubits) {
        let pair = [(QubitId(ancilla), QubitId(data))];
        match kind {
            PauliKind::X => simulator.cx(&pair),
            PauliKind::Y => simulator.cy(&pair),
            PauliKind::Z => simulator.cz(&pair),
        };
    }
    simulator.h(&[QubitId(ancilla)]);
    let syndrome = simulator.mz(&[QubitId(ancilla)])[0].outcome;
    reset_through_measurement(simulator, ancilla);
    syndrome
}

fn encode_bits(bits: &[bool]) -> usize {
    bits.iter()
        .enumerate()
        .fold(0, |value, (bit, &set)| value | (usize::from(set) << bit))
}

#[derive(Clone)]
struct DenseBranch {
    state: DenseStateVec,
    probability: f64,
    record: Vec<bool>,
}

fn projected_dense_state(
    mut state: DenseStateVec,
    qubit: usize,
    outcome: bool,
) -> Option<(f64, DenseStateVec)> {
    let amplitudes = state.state();
    let probability = amplitudes
        .iter()
        .enumerate()
        .filter(|(index, _)| ((*index >> qubit) & 1 != 0) == outcome)
        .map(|(_, amplitude)| amplitude.norm_sqr())
        .sum::<f64>();
    if probability <= PROBABILITY_EPSILON {
        return None;
    }

    let normalization = probability.sqrt();
    for (index, amplitude) in amplitudes.into_iter().enumerate() {
        let projected = if ((index >> qubit) & 1 != 0) == outcome {
            amplitude / normalization
        } else {
            Complex64::new(0.0, 0.0)
        };
        state.set_amplitude(index, projected);
    }
    Some((probability, state))
}

fn dense_measure(branches: Vec<DenseBranch>, qubit: usize, record: bool) -> Vec<DenseBranch> {
    let mut measured = Vec::with_capacity(branches.len() * 2);
    for branch in branches {
        for outcome in [false, true] {
            if let Some((conditional_probability, state)) =
                projected_dense_state(branch.state.clone(), qubit, outcome)
            {
                let mut next = DenseBranch {
                    state,
                    probability: branch.probability * conditional_probability,
                    record: branch.record.clone(),
                };
                if record {
                    next.record.push(outcome);
                }
                measured.push(next);
            }
        }
    }
    measured
}

fn dense_reset(branches: Vec<DenseBranch>, qubit: usize, record: bool) -> Vec<DenseBranch> {
    let mut reset = Vec::with_capacity(branches.len() * 2);
    for branch in branches {
        for outcome in [false, true] {
            if let Some((conditional_probability, mut state)) =
                projected_dense_state(branch.state.clone(), qubit, outcome)
            {
                if outcome {
                    state.x(&[QubitId(qubit)]);
                }
                let mut next = DenseBranch {
                    state,
                    probability: branch.probability * conditional_probability,
                    record: branch.record.clone(),
                };
                if record {
                    next.record.push(outcome);
                }
                reset.push(next);
            }
        }
    }
    reset
}

fn dense_pz(branches: Vec<DenseBranch>, qubit: usize) -> Vec<DenseBranch> {
    dense_reset(branches, qubit, false)
}

fn dense_px(branches: Vec<DenseBranch>, qubit: usize) -> Vec<DenseBranch> {
    let mut branches = dense_pz(branches, qubit);
    for branch in &mut branches {
        branch.state.h(&[QubitId(qubit)]);
    }
    branches
}

fn replay_dense_branches(branches: &mut [DenseBranch], gates: &[Gate]) {
    for branch in branches {
        replay_gates(&mut branch.state, gates);
    }
}

fn dense_measure_all(mut branches: Vec<DenseBranch>, num_qubits: usize) -> Vec<DenseBranch> {
    for q in 0..num_qubits {
        branches = dense_measure(branches, q, true);
    }
    branches
}

fn dense_extract_syndrome(mut branches: Vec<DenseBranch>, num_qubits: usize) -> Vec<DenseBranch> {
    let ancilla = num_qubits - 1;
    branches = dense_px(branches, ancilla);
    for (data, kind) in syndrome_generator(num_qubits) {
        let pair = [(QubitId(ancilla), QubitId(data))];
        for branch in &mut branches {
            match kind {
                PauliKind::X => branch.state.cx(&pair),
                PauliKind::Y => branch.state.cy(&pair),
                PauliKind::Z => branch.state.cz(&pair),
            };
        }
    }
    for branch in &mut branches {
        branch.state.h(&[QubitId(ancilla)]);
    }
    branches = dense_measure(branches, ancilla, true);
    dense_reset(branches, ancilla, false)
}

fn exact_probabilities(surface: Surface, num_qubits: usize) -> Vec<f64> {
    let preparation_qubits = if surface == Surface::ExtractSyndromes {
        num_qubits - 1
    } else {
        num_qubits
    };
    let preparation = honest_family(preparation_qubits, num_qubits);
    let mut dense = DenseStateVec::with_seed(num_qubits, 0xD3E5_E000 + num_qubits as u64);
    replay_gates(&mut dense, &preparation);
    let mut branches = vec![DenseBranch {
        state: dense,
        probability: 1.0,
        record: Vec::new(),
    }];
    let measured = num_qubits / 2;

    branches = match surface {
        Surface::MzMid => {
            let mut measured = dense_measure(branches, measured, true);
            replay_dense_branches(&mut measured, &continuation_family(num_qubits));
            measured
        }
        Surface::MzEnd | Surface::SampleBitstring | Surface::SampleBitstrings => {
            dense_measure_all(branches, num_qubits)
        }
        Surface::Reset => {
            let mut reset = dense_reset(branches, measured, true);
            replay_dense_branches(&mut reset, &continuation_family(num_qubits));
            dense_measure_all(reset, num_qubits)
        }
        Surface::Pz => {
            let mut prepared = dense_pz(branches, measured);
            replay_dense_branches(&mut prepared, &continuation_family(num_qubits));
            dense_measure_all(prepared, num_qubits)
        }
        Surface::Px => {
            let mut prepared = dense_px(branches, measured);
            replay_dense_branches(&mut prepared, &continuation_family(num_qubits));
            dense_measure_all(prepared, num_qubits)
        }
        Surface::ExtractSyndromes => dense_extract_syndrome(branches, num_qubits),
        Surface::Continuation => {
            let mut continued = dense_measure(branches, measured, true);
            replay_dense_branches(&mut continued, &continuation_family(num_qubits));
            dense_measure_all(continued, num_qubits)
        }
    };

    let mut probabilities = vec![0.0; 1 << surface.record_width(num_qubits)];
    for branch in branches {
        probabilities[encode_bits(&branch.record)] += branch.probability;
    }
    let total = probabilities.iter().sum::<f64>();
    assert!(
        (total - 1.0).abs() <= 1e-10,
        "surface={} n={num_qubits}: dense probabilities sum to {total:.16}",
        surface.name()
    );
    for probability in &mut probabilities {
        *probability /= total;
    }
    probabilities
}

fn run_built_stn_shot(surface: Surface, num_qubits: usize, mut simulator: StabMps) -> usize {
    let preparation_qubits = if surface == Surface::ExtractSyndromes {
        num_qubits - 1
    } else {
        num_qubits
    };
    replay_gates(
        &mut simulator,
        &honest_family(preparation_qubits, num_qubits),
    );
    let measured = num_qubits / 2;
    let bits = match surface {
        Surface::MzMid => {
            let outcome = simulator.mz(&[QubitId(measured)])[0].outcome;
            replay_gates(&mut simulator, &continuation_family(num_qubits));
            vec![outcome]
        }
        Surface::MzEnd => measure_all(&mut simulator, num_qubits),
        Surface::Reset => {
            let mut bits = vec![simulator.reset_qubit(QubitId(measured))];
            replay_gates(&mut simulator, &continuation_family(num_qubits));
            bits.extend(measure_all(&mut simulator, num_qubits));
            bits
        }
        Surface::Pz => {
            simulator.pz(QubitId(measured));
            replay_gates(&mut simulator, &continuation_family(num_qubits));
            measure_all(&mut simulator, num_qubits)
        }
        Surface::Px => {
            simulator.px(QubitId(measured));
            replay_gates(&mut simulator, &continuation_family(num_qubits));
            measure_all(&mut simulator, num_qubits)
        }
        Surface::ExtractSyndromes => simulator.extract_syndromes(
            &[syndrome_generator(num_qubits)],
            &[QubitId(num_qubits - 1)],
        ),
        Surface::Continuation => {
            let mut bits = vec![simulator.mz(&[QubitId(measured)])[0].outcome];
            replay_gates(&mut simulator, &continuation_family(num_qubits));
            bits.extend(measure_all(&mut simulator, num_qubits));
            bits
        }
        Surface::SampleBitstring | Surface::SampleBitstrings => {
            unreachable!("sampling surfaces use their public batch APIs")
        }
    };
    assert_eq!(
        simulator.bond_cap_hits(),
        0,
        "surface={} n={num_qubits}: StabMps exact-test cap was binding",
        surface.name()
    );
    encode_bits(&bits)
}

fn run_stn_shot_with_mode(
    surface: Surface,
    num_qubits: usize,
    seed: u64,
    mode: MeasurementMode,
) -> usize {
    run_built_stn_shot(
        surface,
        num_qubits,
        build_stn_with_mode(num_qubits, seed, mode),
    )
}

fn run_stn_shot(surface: Surface, num_qubits: usize, seed: u64) -> usize {
    run_built_stn_shot(surface, num_qubits, build_stn(num_qubits, seed))
}

fn run_mast_shot(surface: Surface, num_qubits: usize, seed: u64) -> usize {
    let preparation_qubits = if surface == Surface::ExtractSyndromes {
        num_qubits - 1
    } else {
        num_qubits
    };
    let preparation = honest_family(preparation_qubits, num_qubits);
    let continuation = continuation_family(num_qubits);
    let capacity = non_clifford_count(&preparation) + non_clifford_count(&continuation);
    let mut simulator = build_mast(num_qubits, capacity, seed);
    replay_gates(&mut simulator, &preparation);
    let measured = num_qubits / 2;
    let bits = match surface {
        Surface::MzMid => {
            let outcome = simulator.mz(&[QubitId(measured)])[0].outcome;
            replay_gates(&mut simulator, &continuation);
            vec![outcome]
        }
        Surface::MzEnd | Surface::SampleBitstring | Surface::SampleBitstrings => {
            measure_all(&mut simulator, num_qubits)
        }
        Surface::Reset => {
            let mut bits = vec![reset_through_measurement(&mut simulator, measured)];
            replay_gates(&mut simulator, &continuation);
            bits.extend(measure_all(&mut simulator, num_qubits));
            bits
        }
        Surface::Pz => {
            pz_through_measurement(&mut simulator, measured);
            replay_gates(&mut simulator, &continuation);
            measure_all(&mut simulator, num_qubits)
        }
        Surface::Px => {
            px_through_measurement(&mut simulator, measured);
            replay_gates(&mut simulator, &continuation);
            measure_all(&mut simulator, num_qubits)
        }
        Surface::ExtractSyndromes => {
            vec![extract_one_syndrome_through_measurement(
                &mut simulator,
                num_qubits,
            )]
        }
        Surface::Continuation => {
            let mut bits = vec![simulator.mz(&[QubitId(measured)])[0].outcome];
            replay_gates(&mut simulator, &continuation);
            bits.extend(measure_all(&mut simulator, num_qubits));
            bits
        }
    };
    assert_eq!(
        simulator.bond_cap_hits(),
        0,
        "surface={} n={num_qubits}: Mast exact-test cap was binding",
        surface.name()
    );
    encode_bits(&bits)
}

fn parallel_counts(
    num_outcomes: usize,
    num_shots: usize,
    sample: impl Fn(usize) -> usize + Sync,
) -> Vec<usize> {
    (0..num_shots)
        .into_par_iter()
        .fold(
            || vec![0; num_outcomes],
            |mut counts, shot| {
                counts[sample(shot)] += 1;
                counts
            },
        )
        .reduce(
            || vec![0; num_outcomes],
            |mut counts, local| {
                for (count, increment) in counts.iter_mut().zip(local) {
                    *count += increment;
                }
                counts
            },
        )
}

fn sampled_stn_counts(surface: Surface, num_qubits: usize, num_shots: usize) -> Vec<usize> {
    let num_outcomes = 1 << surface.record_width(num_qubits);
    if matches!(
        surface,
        Surface::SampleBitstring | Surface::SampleBitstrings
    ) {
        let mut simulator = build_stn(num_qubits, 0x5A00_0000 + num_qubits as u64);
        replay_gates(&mut simulator, &honest_family(num_qubits, num_qubits));
        let samples = if surface == Surface::SampleBitstring {
            simulator.sample_bitstring(num_shots)
        } else {
            simulator.sample_bitstrings(num_shots)
        };
        let mut counts = vec![0; num_outcomes];
        for bits in samples {
            counts[encode_bits(&bits)] += 1;
        }
        assert_eq!(simulator.bond_cap_hits(), 0);
        counts
    } else {
        parallel_counts(num_outcomes, num_shots, |shot| {
            run_stn_shot(
                surface,
                num_qubits,
                0x5100_0000 + (num_qubits as u64) * 100_000 + shot as u64,
            )
        })
    }
}

fn sampled_mast_counts(surface: Surface, num_qubits: usize, num_shots: usize) -> Vec<usize> {
    let num_outcomes = 1 << surface.record_width(num_qubits);
    parallel_counts(num_outcomes, num_shots, |shot| {
        run_mast_shot(
            surface,
            num_qubits,
            0x6A00_0000 + (num_qubits as u64) * 100_000 + shot as u64,
        )
    })
}

#[derive(Debug)]
struct Comparison {
    surface: Surface,
    num_qubits: usize,
    num_shots: usize,
    exact: Vec<f64>,
    default_counts: Vec<usize>,
    control_counts: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
struct WorstDeviation {
    outcome: usize,
    sigma: f64,
    exact: f64,
    sampled: f64,
}

fn binomial_sigma(count: usize, probability: f64, num_shots: usize) -> f64 {
    let sampled = count as f64 / num_shots as f64;
    if probability <= PROBABILITY_EPSILON || 1.0 - probability <= PROBABILITY_EPSILON {
        return if (sampled - probability).abs() <= PROBABILITY_EPSILON {
            0.0
        } else {
            f64::INFINITY
        };
    }
    let standard_error = (probability * (1.0 - probability) / num_shots as f64).sqrt();
    (sampled - probability).abs() / standard_error
}

fn worst_deviation(counts: &[usize], exact: &[f64], num_shots: usize) -> WorstDeviation {
    exact
        .iter()
        .enumerate()
        .map(|(outcome, &probability)| WorstDeviation {
            outcome,
            sigma: binomial_sigma(counts[outcome], probability, num_shots),
            exact: probability,
            sampled: counts[outcome] as f64 / num_shots as f64,
        })
        .max_by(|left, right| left.sigma.total_cmp(&right.sigma))
        .expect("every surface has at least one outcome")
}

/// Bonferroni-style family allowance for a matrix of binomial checks.
///
/// Each bin starts with a two-sided five-sigma guard. For `m` simultaneous
/// default/control bin checks, `sqrt(5^2 + 2 ln(m))` keeps the union-bound
/// Gaussian tail envelope no larger than the single-bin five-sigma envelope.
/// This is deliberately an allowance for multiple comparisons, not a relaxed
/// per-surface empirical tolerance.
fn corrected_five_sigma_limit(num_comparisons: usize) -> f64 {
    (25.0 + 2.0 * (num_comparisons.max(1) as f64).ln()).sqrt()
}

fn compare_surface(surface: Surface, num_qubits: usize, num_shots: usize) -> Comparison {
    let exact = exact_probabilities(surface, num_qubits);
    let default_counts = sampled_stn_counts(surface, num_qubits, num_shots);
    let control_counts = sampled_mast_counts(surface, num_qubits, num_shots);
    assert_eq!(default_counts.iter().sum::<usize>(), num_shots);
    assert_eq!(control_counts.iter().sum::<usize>(), num_shots);
    Comparison {
        surface,
        num_qubits,
        num_shots,
        exact,
        default_counts,
        control_counts,
    }
}

fn assert_surface_matrix(surface: Surface, qubits: &[usize], num_shots: usize) {
    let comparisons = qubits
        .iter()
        .map(|&num_qubits| compare_surface(surface, num_qubits, num_shots))
        .collect::<Vec<_>>();
    let num_binomial_checks = comparisons
        .iter()
        .map(|comparison| comparison.exact.len() * 2)
        .sum();
    let limit = corrected_five_sigma_limit(num_binomial_checks);

    let mut worst_default = None::<(WorstDeviation, usize)>;
    let mut worst_control = None::<(WorstDeviation, usize)>;
    for comparison in &comparisons {
        let default = worst_deviation(
            &comparison.default_counts,
            &comparison.exact,
            comparison.num_shots,
        );
        let control = worst_deviation(
            &comparison.control_counts,
            &comparison.exact,
            comparison.num_shots,
        );
        eprintln!(
            "exact-default-falsifier surface={} n={} shots={} default={:.2}sigma \
             (outcome={}, exact={:.6}, sampled={:.6}) control={:.2}sigma \
             (outcome={}, exact={:.6}, sampled={:.6}) corrected_limit={limit:.2}sigma",
            comparison.surface.name(),
            comparison.num_qubits,
            comparison.num_shots,
            default.sigma,
            default.outcome,
            default.exact,
            default.sampled,
            control.sigma,
            control.outcome,
            control.exact,
            control.sampled,
        );
        if worst_default.is_none_or(|(worst, _)| default.sigma > worst.sigma) {
            worst_default = Some((default, comparison.num_qubits));
        }
        if worst_control.is_none_or(|(worst, _)| control.sigma > worst.sigma) {
            worst_control = Some((control, comparison.num_qubits));
        }
    }

    let (control, control_n) = worst_control.expect("nonempty surface matrix");
    assert!(
        control.sigma <= limit,
        "Mast exact-route control failed: surface={} n={control_n} outcome={} \
         exact={:.8} sampled={:.8}, deviation={:.2}sigma > corrected {limit:.2}sigma",
        surface.name(),
        control.outcome,
        control.exact,
        control.sampled,
        control.sigma,
    );
    let (default, default_n) = worst_default.expect("nonempty surface matrix");
    assert!(
        default.sigma <= limit,
        "default StabMps measurement is biased: surface={} n={default_n} outcome={} \
         exact={:.8} sampled={:.8}, deviation={:.2}sigma > corrected {limit:.2}sigma",
        surface.name(),
        default.outcome,
        default.exact,
        default.sampled,
        default.sigma,
    );
}

fn state_fidelity(first: &[Complex64], second: &[Complex64]) -> f64 {
    first
        .iter()
        .zip(second)
        .map(|(left, right)| left.conj() * right)
        .sum::<Complex64>()
        .norm_sqr()
}

fn lazy_frame_clifford_family(seed: u64, num_qubits: usize) -> Vec<Gate> {
    let mut word = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next = || {
        word ^= word << 13;
        word ^= word >> 7;
        word ^= word << 17;
        word
    };
    let mut gates = Vec::new();
    for layer in 0..2 {
        let target = next() as usize % num_qubits;
        let mut neighbor = next() as usize % num_qubits;
        if neighbor == target {
            neighbor = (neighbor + 1) % num_qubits;
        }
        // Every layer changes basis before spreading it, so this is an
        // honest Clifford family rather than a diagonal tail.
        gates.push(Gate::H(target));
        if layer % 2 == 0 {
            gates.push(Gate::Sz(neighbor));
            gates.push(Gate::Cx(target, neighbor));
        } else {
            gates.push(Gate::Cz(target, neighbor));
            gates.push(Gate::Cx(neighbor, target));
        }
    }
    gates
}

#[test]
fn lazy_measure_clifford_rz_conditional_state_fidelity() {
    const NUM_SEEDS: u64 = 24;
    const NUM_QUBITS: usize = 3;
    let mut fidelities = Vec::with_capacity(NUM_SEEDS as usize);

    for seed in 0..NUM_SEEDS {
        let preparation = vec![
            Gate::H(1),
            Gate::Rz(1, Angle64::QUARTER_TURN / 2_u64),
            Gate::Sz(0),
            Gate::H(0),
            Gate::Cx(0, 1),
            Gate::H(2),
            Gate::Cx(1, 2),
        ];
        let cliffords = lazy_frame_clifford_family(seed, NUM_QUBITS);
        let measured = 0;
        let rotated = 1;
        let angle = Angle64::from_radians(0.65 + 0.01 * seed as f64);

        let mut simulator =
            build_stn_with_mode(NUM_QUBITS, 0x5550_0000 + seed, MeasurementMode::Lazy);
        let mut dense = DenseStateVec::new(NUM_QUBITS);
        replay_gates(&mut simulator, &preparation);
        replay_gates(&mut dense, &preparation);

        let outcome = simulator.mz(&[QubitId(measured)])[0].outcome;
        let (_, mut expected) = projected_dense_state(dense, measured, outcome)
            .expect("the sampled Lazy branch must be present in the dense oracle");
        replay_gates(&mut simulator, &cliffords);
        replay_gates(&mut expected, &cliffords);
        let mut before_rotation = simulator.clone();
        before_rotation.flush();
        let before_fidelity = state_fidelity(&before_rotation.state_vector(), &expected.state());
        assert!(
            before_fidelity >= 1.0 - 1e-10,
            "seed={seed}: the Lazy conditional state must be correct before RZ; fidelity={before_fidelity:.16}"
        );
        simulator.rz(angle, &[QubitId(rotated)]);
        expected.rz(angle, &[QubitId(rotated)]);

        simulator.flush();
        let fidelity = state_fidelity(&simulator.state_vector(), &expected.state());
        eprintln!(
            "lazy-frame-rz-fidelity seed={seed} measured={measured} rotated={rotated} before={before_fidelity:.16} fidelity={fidelity:.16}"
        );
        fidelities.push(fidelity);
    }

    let worst = fidelities.iter().copied().fold(1.0_f64, f64::min);
    let failed = fidelities
        .iter()
        .filter(|&&fidelity| fidelity < 1.0 - 1e-10)
        .count();
    assert_eq!(
        failed, 0,
        "issue #555: {failed}/{NUM_SEEDS} Lazy mz -> Clifford -> RZ conditional states failed; worst fidelity={worst:.16}"
    );
}

#[test]
fn lazy_frame_rz_does_not_consume_stored_disentangling_proof() {
    const FRAME_SEED: u64 = 4;
    let preparation = vec![
        Gate::H(1),
        Gate::Rz(1, Angle64::QUARTER_TURN / 2_u64),
        Gate::H(0),
        Gate::Cx(0, 1),
    ];
    let cliffords = lazy_frame_clifford_family(FRAME_SEED, 4);
    let angle = Angle64::from_radians(0.71);
    let mut simulator = build_stn_with_mode(4, 0x5555, MeasurementMode::Lazy);
    let mut dense = DenseStateVec::new(4);
    replay_gates(&mut simulator, &preparation);
    replay_gates(&mut dense, &preparation);
    let outcome = simulator.mz(&[QubitId(0)])[0].outcome;
    let (_, mut expected) = projected_dense_state(dense, 0, outcome).unwrap();
    replay_gates(&mut simulator, &cliffords);
    replay_gates(&mut expected, &cliffords);

    let bypass_before = simulator.stats.deferred_disent_bypass;
    let disent_before = simulator.stats.multi_disent;
    simulator.rz(angle, &[QubitId(2)]);
    expected.rz(angle, &[QubitId(2)]);
    assert_eq!(
        simulator.stats.deferred_disent_bypass,
        bypass_before + 1,
        "the fixture must expose a stored |0> proof that is unsafe to absorb across V"
    );
    assert_eq!(
        simulator.stats.multi_disent, disent_before,
        "a pending Lazy frame must prevent tableau-right-composing exact disentangling"
    );

    simulator.flush();
    let fidelity = state_fidelity(&simulator.state_vector(), &expected.state());
    assert!(
        fidelity >= 1.0 - 1e-10,
        "bypassed stored-proof path must remain state-correct; fidelity={fidelity:.16}"
    );
}

fn assert_default_state_matches(
    simulator: &mut StabMps,
    expected: &mut DenseStateVec,
    context: &str,
) {
    simulator.flush();
    let fidelity = state_fidelity(&simulator.state_vector(), &expected.state());
    assert!(
        fidelity >= 1.0 - 1e-10,
        "{context}: conditional-state fidelity={fidelity:.16}"
    );
}

#[test]
fn exact_default_conditional_state_fidelity_matrix() {
    for num_qubits in 3..=4 {
        let measured = num_qubits / 2;
        let preparation = honest_family(num_qubits, 0x41 + num_qubits);

        let prepare = |seed| {
            let mut stn = build_stn(num_qubits, seed);
            let mut dense = DenseStateVec::new(num_qubits);
            replay_gates(&mut stn, &preparation);
            replay_gates(&mut dense, &preparation);
            (stn, dense)
        };

        let (mut stn, dense) = prepare(0x7100 + num_qubits as u64);
        let outcome = stn.mz(&[QubitId(measured)])[0].outcome;
        let (_, mut expected) = projected_dense_state(dense, measured, outcome).unwrap();
        assert_default_state_matches(
            &mut stn,
            &mut expected,
            &format!("default mz n={num_qubits}"),
        );

        let (mut stn, dense) = prepare(0x7200 + num_qubits as u64);
        let outcome = stn.reset_qubit(QubitId(measured));
        let (_, mut expected) = projected_dense_state(dense, measured, outcome).unwrap();
        if outcome {
            expected.x(&[QubitId(measured)]);
        }
        assert_default_state_matches(
            &mut stn,
            &mut expected,
            &format!("default reset n={num_qubits}"),
        );

        for px in [false, true] {
            let (mut stn, dense) = prepare(0x7300 + num_qubits as u64 + u64::from(px));
            if px {
                stn.px(QubitId(measured));
            } else {
                stn.pz(QubitId(measured));
            }
            let branch = DenseBranch {
                state: dense,
                probability: 1.0,
                record: Vec::new(),
            };
            let mut branches = if px {
                dense_px(vec![branch], measured)
            } else {
                dense_pz(vec![branch], measured)
            };
            stn.flush();
            let actual = stn.state_vector();
            let fidelity = branches
                .iter_mut()
                .map(|branch| state_fidelity(&actual, &branch.state.state()))
                .fold(0.0_f64, f64::max);
            assert!(
                fidelity >= 1.0 - 1e-10,
                "default {} n={num_qubits}: conditional-state fidelity={fidelity:.16}",
                if px { "px" } else { "pz" }
            );
        }

        let (mut stn, dense) = prepare(0x7400 + num_qubits as u64);
        let outcome = stn.mz(&[QubitId(measured)])[0].outcome;
        let (_, mut expected) = projected_dense_state(dense, measured, outcome).unwrap();
        let continuation = continuation_family(num_qubits);
        replay_gates(&mut stn, &continuation);
        replay_gates(&mut expected, &continuation);
        assert_default_state_matches(
            &mut stn,
            &mut expected,
            &format!("default continuation n={num_qubits}"),
        );

        let data_qubits = num_qubits - 1;
        let preparation = honest_family(data_qubits, 0x51 + num_qubits);
        let mut stn = build_stn(num_qubits, 0x7500 + num_qubits as u64);
        let mut dense = DenseStateVec::new(num_qubits);
        replay_gates(&mut stn, &preparation);
        replay_gates(&mut dense, &preparation);
        let generator = syndrome_generator(num_qubits);
        let syndrome =
            stn.extract_syndromes(std::slice::from_ref(&generator), &[QubitId(data_qubits)])[0];
        dense.h(&[QubitId(data_qubits)]);
        for (data, kind) in generator {
            let pair = [(QubitId(data_qubits), QubitId(data))];
            match kind {
                PauliKind::X => dense.cx(&pair),
                PauliKind::Y => dense.cy(&pair),
                PauliKind::Z => dense.cz(&pair),
            };
        }
        dense.h(&[QubitId(data_qubits)]);
        let (_, mut expected) = projected_dense_state(dense, data_qubits, syndrome).unwrap();
        if syndrome {
            expected.x(&[QubitId(data_qubits)]);
        }
        assert_default_state_matches(
            &mut stn,
            &mut expected,
            &format!("default syndrome extraction n={num_qubits}"),
        );
    }
}

#[test]
fn exact_default_measurement_falsifier_is_live() {
    let mut comparison = compare_surface(Surface::Continuation, 3, META_SHOTS);
    comparison.default_counts = parallel_counts(comparison.exact.len(), META_SHOTS, |shot| {
        run_stn_shot_with_mode(
            Surface::Continuation,
            3,
            0x5100_0000 + 300_000 + shot as u64,
            MeasurementMode::Pragmatic,
        )
    });
    let num_binomial_checks = comparison.exact.len() * 2;
    let limit = corrected_five_sigma_limit(num_binomial_checks);
    let default = worst_deviation(
        &comparison.default_counts,
        &comparison.exact,
        comparison.num_shots,
    );
    let control = worst_deviation(
        &comparison.control_counts,
        &comparison.exact,
        comparison.num_shots,
    );
    eprintln!(
        "exact-default-falsifier meta surface={} n={} shots={} pragmatic={:.2}sigma \
         control={:.2}sigma corrected_limit={limit:.2}sigma",
        comparison.surface.name(),
        comparison.num_qubits,
        comparison.num_shots,
        default.sigma,
        control.sigma,
    );
    assert!(
        control.sigma <= limit,
        "live-harness Mast control deviated by {:.2}sigma (limit {limit:.2}sigma)",
        control.sigma
    );
    assert!(
        default.sigma > 5.0,
        "explicit pragmatic measurement no longer shows the expected bias: \
         worst deviation={:.2}sigma",
        default.sigma
    );
}

macro_rules! surface_gate_tests {
    ($fast_name:ident, $release_name:ident, $surface:expr) => {
        #[cfg(not(debug_assertions))]
        #[test]
        fn $fast_name() {
            assert_surface_matrix($surface, &[3, 4], FAST_SHOTS);
        }

        #[test]
        #[ignore = "larger exact-default statistical release lane"]
        fn $release_name() {
            assert_surface_matrix($surface, &[5, 6], RELEASE_SHOTS);
        }
    };
}

#[cfg(debug_assertions)]
#[test]
fn exact_default_debug_smoke_subset() {
    for surface in [
        Surface::MzMid,
        Surface::ExtractSyndromes,
        Surface::Continuation,
    ] {
        assert_surface_matrix(surface, &[3], DEBUG_SHOTS);
    }
}

surface_gate_tests!(
    exact_default_mz_mid_fast,
    exact_default_mz_mid_release,
    Surface::MzMid
);
surface_gate_tests!(
    exact_default_mz_end_fast,
    exact_default_mz_end_release,
    Surface::MzEnd
);
surface_gate_tests!(
    exact_default_reset_qubit_fast,
    exact_default_reset_qubit_release,
    Surface::Reset
);
surface_gate_tests!(exact_default_pz_fast, exact_default_pz_release, Surface::Pz);
surface_gate_tests!(exact_default_px_fast, exact_default_px_release, Surface::Px);
surface_gate_tests!(
    exact_default_extract_syndromes_fast,
    exact_default_extract_syndromes_release,
    Surface::ExtractSyndromes
);
surface_gate_tests!(
    exact_default_continuation_fast,
    exact_default_continuation_release,
    Surface::Continuation
);
surface_gate_tests!(
    exact_default_sample_bitstring_fast,
    exact_default_sample_bitstring_release,
    Surface::SampleBitstring
);
surface_gate_tests!(
    exact_default_sample_bitstrings_fast,
    exact_default_sample_bitstrings_release,
    Surface::SampleBitstrings
);
