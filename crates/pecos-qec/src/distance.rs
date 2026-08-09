// Copyright 2024 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Code distance calculation and minimum weight logical operator search.
//!
//! This module provides algorithms for computing the distance of a stabilizer code
//! by exhaustively searching for minimum weight logical operators.

use crate::stabilizer_code_spec::CodeIndices;
use crate::{StabilizerCodeSpec, StabilizerCodeSpecError};
use pecos_core::{Pauli, PauliString, QubitId};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Minimum candidate count at a single weight before that weight is searched in parallel.
///
/// Rayon's fixed per-weight overhead is worth paying only once a weight carries enough
/// candidates to amortize it, so the decision is made per weight rather than per code: a
/// large code's low weights are still cheap and stay serial.
///
/// The value sits inside a measured window (see `benches/modules/code_distance.rs`).
/// Below roughly 22k candidates parallelism loses: forcing the toric [[18, 2, 3]] weight-3
/// tier (22,032 candidates) parallel made that search 4.6x slower than serial. Above
/// roughly 193k it stops engaging where it matters: the color [[17, 1, 5]] search spends
/// most of its time in the weight-4 tier (192,780 candidates), and a threshold past that
/// erased its speedup entirely. 65,536 is near the geometric centre of that window, so
/// both bounds keep margin.
const PARALLEL_CANDIDATE_THRESHOLD: usize = 65_536;

/// Result of a distance calculation, including the minimum weight logical operator found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DistanceResult {
    /// The code distance (minimum weight of any logical operator).
    pub distance: usize,
    /// A logical operator achieving the minimum weight.
    pub min_weight_operator: PauliString,
}

/// A logical operator with information about which logical operations it implements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalOperatorInfo {
    /// The Pauli operator.
    pub operator: PauliString,
    /// Weight of the operator.
    pub weight: usize,
    /// Which logical operators this is equivalent to.
    /// Each entry is a (type, index) pair where type is 'X' or 'Z' and index is the logical qubit.
    /// For example, `[('X', 0), ('Z', 1)]` means this operator is equivalent to `X_0` * `Z_1`.
    pub equivalent_logicals: Vec<(char, usize)>,
}

impl LogicalOperatorInfo {
    /// Returns a human-readable string describing the equivalent logical operators.
    ///
    /// For example: "X0", "Z1", "X0*Z1", etc.
    #[must_use]
    pub fn equivalence_string(&self) -> String {
        if self.equivalent_logicals.is_empty() {
            return "I".to_string();
        }
        self.equivalent_logicals
            .iter()
            .map(|(t, i)| format!("{t}{i}"))
            .collect::<Vec<_>>()
            .join("*")
    }
}

/// Configuration for distance search.
#[derive(Clone, Debug, Default)]
pub struct DistanceSearchConfig {
    /// Maximum weight to search up to (None for unlimited).
    pub max_weight: Option<usize>,
    /// Whether to search only CSS-type errors (X-only or Z-only).
    pub css_only: bool,
    /// Whether to be verbose during search.
    pub verbose: bool,
}

impl DistanceSearchConfig {
    /// Create a new config that searches up to the given weight.
    #[must_use]
    pub fn with_max_weight(max_weight: usize) -> Self {
        Self {
            max_weight: Some(max_weight),
            ..Default::default()
        }
    }

    /// Create a config for CSS-only search (faster for CSS codes).
    #[must_use]
    pub fn css() -> Self {
        Self {
            css_only: true,
            ..Default::default()
        }
    }
}

/// Generate all Pauli strings of a given weight on a set of qubits.
///
/// This is a helper iterator that generates all possible Pauli operators
/// of exactly the specified weight.
///
/// In CSS mode (`css_only=true`), only generates pure X errors (XXXX...) and
/// pure Z errors (ZZZZ...), not mixed XZ errors. This matches the Python
/// `gen_errors` behavior and is faster for CSS codes.
pub struct WeightedPauliIterator {
    num_qubits: usize,
    weight: usize,
    /// Current combination of qubit positions
    positions: Vec<usize>,
    /// Current Pauli assignment (0=X, 1=Y, 2=Z for general; 0=X, 1=Z for CSS)
    paulis: Vec<usize>,
    /// Whether we've exhausted all combinations
    done: bool,
    /// Whether to use CSS mode (pure X or pure Z only, no mixed)
    css_only: bool,
    /// In CSS mode: 0 = generating X errors, 1 = generating Z errors
    css_pauli_type: usize,
}

/// Position in the serial candidate enumeration.
///
/// General searches order by support and then Pauli assignment. CSS searches order
/// all X supports before all Z supports, so the fields hold Pauli type and support.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CandidateIndex {
    outer: usize,
    inner: usize,
}

impl WeightedPauliIterator {
    /// Create a new iterator for Pauli operators of the given weight.
    ///
    /// If `css_only` is true, only pure X errors (XXXX...) and pure Z errors (ZZZZ...)
    /// are generated, not mixed XZ errors.
    #[must_use]
    pub fn new(num_qubits: usize, weight: usize, css_only: bool) -> Self {
        if weight == 0 || weight > num_qubits {
            return Self {
                num_qubits,
                weight,
                positions: vec![],
                paulis: vec![],
                done: true,
                css_only,
                css_pauli_type: 0,
            };
        }

        // Initialize with first combination: 0, 1, 2, ..., weight-1
        let positions: Vec<usize> = (0..weight).collect();
        let paulis = vec![0; weight]; // All X initially

        Self {
            num_qubits,
            weight,
            positions,
            paulis,
            done: false,
            css_only,
            css_pauli_type: 0,
        }
    }

    /// Advance to the next Pauli assignment (only used in non-CSS mode).
    fn next_pauli(&mut self) -> bool {
        if self.css_only {
            // In CSS mode, we don't mix Paulis - handled by css_pauli_type
            return false;
        }

        // Try to increment the Pauli assignment
        for i in (0..self.weight).rev() {
            if self.paulis[i] < 2 {
                self.paulis[i] += 1;
                // Reset all following positions
                for j in (i + 1)..self.weight {
                    self.paulis[j] = 0;
                }
                return true;
            }
        }
        false
    }

    /// Advance to the next position combination.
    fn next_combination(&mut self) -> bool {
        // Reset Pauli assignments
        for p in &mut self.paulis {
            *p = 0;
        }

        // Find the rightmost position that can be incremented
        let mut i = self.weight;
        while i > 0 {
            i -= 1;
            if self.positions[i] < self.num_qubits - self.weight + i {
                self.positions[i] += 1;
                // Reset all following positions
                for j in (i + 1)..self.weight {
                    self.positions[j] = self.positions[j - 1] + 1;
                }
                return true;
            }
        }
        false
    }

    /// Convert current state to a `PauliString`.
    fn current_pauli_string(&self) -> PauliString {
        let paulis: Vec<(Pauli, QubitId)> = self
            .positions
            .iter()
            .zip(self.paulis.iter())
            .map(|(&pos, &p)| {
                let pauli = if self.css_only {
                    // In CSS mode, css_pauli_type determines X or Z
                    if self.css_pauli_type == 0 {
                        Pauli::X
                    } else {
                        Pauli::Z
                    }
                } else {
                    match p {
                        0 => Pauli::X,
                        1 => Pauli::Y,
                        _ => Pauli::Z,
                    }
                };
                (pauli, QubitId::new(pos))
            })
            .collect();

        PauliString::with_phase_and_paulis(pecos_core::QuarterPhase::PlusOne, paulis)
    }
}

impl Iterator for WeightedPauliIterator {
    type Item = PauliString;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = self.current_pauli_string();

        // Try to advance to next state
        if self.css_only {
            // In CSS mode: first iterate all X positions, then all Z positions
            if !self.next_combination() {
                if self.css_pauli_type == 0 {
                    // Switch from X to Z
                    self.css_pauli_type = 1;
                    // Reset positions to start
                    self.positions = (0..self.weight).collect();
                } else {
                    // Done with both X and Z
                    self.done = true;
                }
            }
        } else {
            // In general mode: iterate paulis first, then positions
            if !self.next_pauli() && !self.next_combination() {
                self.done = true;
            }
        }

        Some(result)
    }
}

/// Calculate the distance of a stabilizer code.
///
/// The distance is the minimum weight of any logical operator (an operator that
/// commutes with all stabilizers but is not in the stabilizer group).
///
/// # Warning
/// This is an exponential-time algorithm. For codes with many qubits, it may take
/// a very long time to complete. Use `config.max_weight` to limit the search.
///
/// # Errors
///
/// Returns an error if the supplied logical basis is incomplete or the code encodes no logical
/// qubits.
pub fn calculate_distance(
    code: &StabilizerCodeSpec,
    config: &DistanceSearchConfig,
) -> Result<Option<DistanceResult>, StabilizerCodeSpecError> {
    code.verify_logical_completeness()?;
    let max_weight = config.max_weight.unwrap_or(code.num_qubits());

    // Build indices once for O(weight) lookups instead of O(num_stabilizers * weight)
    let indices = code.build_indices();

    for weight in 1..=max_weight {
        if config.verbose {
            eprintln!("Checking weight {weight}...");
        }

        if let Some(pauli) = first_logical_error_at_weight(code, weight, config, &indices) {
            return Ok(Some(DistanceResult {
                distance: weight,
                min_weight_operator: pauli,
            }));
        }
    }

    Ok(None)
}

/// Check whether a code has a logical error at exactly one physical weight.
///
/// Stabilizer and logical column indices are built once for the complete scan.
///
/// # Errors
///
/// Returns an error if the supplied logical basis is incomplete or the code encodes no logical
/// qubits.
pub fn has_logical_error_at_weight(
    code: &StabilizerCodeSpec,
    weight: usize,
    config: &DistanceSearchConfig,
) -> Result<bool, StabilizerCodeSpecError> {
    code.verify_logical_completeness()?;
    if config.verbose {
        eprintln!("Checking weight {weight}...");
    }

    let indices = code.build_indices();
    Ok(first_logical_error_at_weight(code, weight, config, &indices).is_some())
}

/// Find all minimum weight logical operators.
///
/// Unlike `calculate_distance`, this returns all logical operators of the minimum weight,
/// not just one.
///
/// # Errors
///
/// Returns an error if the supplied logical basis is incomplete or the code encodes no logical
/// qubits.
pub fn find_min_weight_logicals(
    code: &StabilizerCodeSpec,
    config: &DistanceSearchConfig,
) -> Result<Vec<PauliString>, StabilizerCodeSpecError> {
    find_shortest_logicals(code, config, 0)
        .map(|logicals| logicals.into_iter().map(|info| info.operator).collect())
}

/// Find all logical operators from the minimum weight through `delta` weights above it.
///
/// The search always starts at weight 1. Once the minimum logical weight is found,
/// collection continues through `minimum_weight + delta`, subject to
/// `config.max_weight`.
///
/// # Errors
///
/// Returns an error if the supplied logical basis is incomplete or the code encodes no logical
/// qubits.
pub fn find_shortest_logicals(
    code: &StabilizerCodeSpec,
    config: &DistanceSearchConfig,
    delta: usize,
) -> Result<Vec<LogicalOperatorInfo>, StabilizerCodeSpecError> {
    code.verify_logical_completeness()?;
    let max_weight = config.max_weight.unwrap_or(code.num_qubits());
    let mut results = Vec::new();
    let mut found_distance: Option<usize> = None;

    // Build indices once for O(weight) lookups instead of O(num_stabilizers * weight)
    let indices = code.build_indices();

    for weight in 1..=max_weight {
        // If we've searched through the requested range above the minimum, stop.
        if let Some(d) = found_distance
            && weight > d.saturating_add(delta)
        {
            break;
        }

        if config.verbose {
            eprintln!("Checking weight {weight}...");
        }

        let weight_matches = logical_errors_at_weight(code, weight, config, &indices);
        if !weight_matches.is_empty() && found_distance.is_none() {
            found_distance = Some(weight);
        }

        for pauli in weight_matches {
            // Determine which logical operators this is equivalent to
            let equivalent_logicals = classify_logical_equivalence_indexed(
                &indices.logical,
                code.num_logical_qubits(),
                &pauli,
            );

            results.push(LogicalOperatorInfo {
                operator: pauli,
                weight,
                equivalent_logicals,
            });
        }
    }

    Ok(results)
}

fn first_logical_error_at_weight(
    code: &StabilizerCodeSpec,
    weight: usize,
    config: &DistanceSearchConfig,
    indices: &CodeIndices,
) -> Option<PauliString> {
    if should_parallelize(code.num_qubits(), weight, config.css_only) {
        first_logical_error_at_weight_parallel(code, weight, config, indices)
    } else {
        first_logical_error_at_weight_serial(code, weight, config, indices)
    }
}

fn first_logical_error_at_weight_serial(
    code: &StabilizerCodeSpec,
    weight: usize,
    config: &DistanceSearchConfig,
    indices: &CodeIndices,
) -> Option<PauliString> {
    WeightedPauliIterator::new(code.num_qubits(), weight, config.css_only)
        .find(|pauli| code.is_logical_error_indexed(pauli, &indices.stabilizer, &indices.logical))
}

fn first_logical_error_at_weight_parallel(
    code: &StabilizerCodeSpec,
    weight: usize,
    config: &DistanceSearchConfig,
    indices: &CodeIndices,
) -> Option<PauliString> {
    let best_support = AtomicUsize::new(usize::MAX);

    support_combinations_at_weight(code.num_qubits(), weight)
        .filter_map(|(support_index, support)| {
            if support_index > best_support.load(Ordering::Relaxed) {
                return None;
            }

            let candidate = first_matching_pauli_for_support(
                code,
                indices,
                support_index,
                &support.qubits(),
                config.css_only,
                &best_support,
            );
            if candidate
                .as_ref()
                .is_some_and(|(index, _)| !config.css_only || index.outer == 0)
            {
                best_support.fetch_min(support_index, Ordering::Relaxed);
            }
            candidate
        })
        .min_by_key(|(index, _)| *index)
        .map(|(_, pauli)| pauli)
}

fn logical_errors_at_weight(
    code: &StabilizerCodeSpec,
    weight: usize,
    config: &DistanceSearchConfig,
    indices: &CodeIndices,
) -> Vec<PauliString> {
    if should_parallelize(code.num_qubits(), weight, config.css_only) {
        logical_errors_at_weight_parallel(code, weight, config.css_only, indices)
    } else {
        logical_errors_at_weight_serial(code, weight, config.css_only, indices)
    }
}

fn logical_errors_at_weight_serial(
    code: &StabilizerCodeSpec,
    weight: usize,
    css_only: bool,
    indices: &CodeIndices,
) -> Vec<PauliString> {
    WeightedPauliIterator::new(code.num_qubits(), weight, css_only)
        .filter(|pauli| code.is_logical_error_indexed(pauli, &indices.stabilizer, &indices.logical))
        .collect()
}

fn logical_errors_at_weight_parallel(
    code: &StabilizerCodeSpec,
    weight: usize,
    css_only: bool,
    indices: &CodeIndices,
) -> Vec<PauliString> {
    let mut matches: Vec<_> = matching_paulis_at_weight(code, weight, css_only, indices).collect();
    matches.sort_unstable_by_key(|(index, _)| *index);
    matches.into_iter().map(|(_, pauli)| pauli).collect()
}

fn should_parallelize(num_qubits: usize, weight: usize, css_only: bool) -> bool {
    candidate_count_at_weight(num_qubits, weight, css_only) > PARALLEL_CANDIDATE_THRESHOLD
}

fn candidate_count_at_weight(num_qubits: usize, weight: usize, css_only: bool) -> usize {
    let support_count = saturating_binomial(num_qubits, weight);
    let assignments_per_support = if css_only {
        2
    } else {
        3usize.saturating_pow(u32::try_from(weight).unwrap_or(u32::MAX))
    };

    support_count.saturating_mul(assignments_per_support)
}

fn saturating_binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }

    let k = k.min(n - k);
    let mut result = 1_u128;
    for i in 1..=k {
        result = result * (n - k + i) as u128 / i as u128;
        if result > usize::MAX as u128 {
            return usize::MAX;
        }
    }

    usize::try_from(result).unwrap_or(usize::MAX)
}

fn matching_paulis_at_weight<'a>(
    code: &'a StabilizerCodeSpec,
    weight: usize,
    css_only: bool,
    indices: &'a CodeIndices,
) -> impl ParallelIterator<Item = (CandidateIndex, PauliString)> + 'a {
    support_combinations_at_weight(code.num_qubits(), weight).flat_map_iter(
        move |(support_index, support)| {
            matching_paulis_for_support(code, indices, support_index, &support.qubits(), css_only)
                .into_iter()
        },
    )
}

fn support_combinations_at_weight(
    num_qubits: usize,
    weight: usize,
) -> impl ParallelIterator<Item = (usize, PauliString)> {
    // Bridge only support combinations into Rayon. Each task enumerates all Pauli
    // assignments for its support serially, amortizing synchronization overhead.
    WeightedPauliIterator::new(num_qubits, weight, true)
        .take_while(|pauli| {
            pauli
                .paulis()
                .first()
                .is_some_and(|(pauli, _)| *pauli == Pauli::X)
        })
        .enumerate()
        .par_bridge()
}

fn first_matching_pauli_for_support(
    code: &StabilizerCodeSpec,
    indices: &CodeIndices,
    support_index: usize,
    positions: &[usize],
    css_only: bool,
    best_support: &AtomicUsize,
) -> Option<(CandidateIndex, PauliString)> {
    if css_only {
        let x = PauliString::xs(positions);
        if code.is_logical_error_indexed(&x, &indices.stabilizer, &indices.logical) {
            return Some((
                CandidateIndex {
                    outer: 0,
                    inner: support_index,
                },
                x,
            ));
        }

        // Any X match sorts before every Z match. Once another task has found an
        // X candidate, a Z candidate cannot improve the reduction.
        if best_support.load(Ordering::Relaxed) != usize::MAX {
            return None;
        }

        let z = PauliString::zs(positions);
        return code
            .is_logical_error_indexed(&z, &indices.stabilizer, &indices.logical)
            .then_some((
                CandidateIndex {
                    outer: 1,
                    inner: support_index,
                },
                z,
            ));
    }

    let mut paulis = vec![0; positions.len()];
    let mut assignment_index = 0;

    loop {
        let pauli = pauli_string_for_assignment(positions, &paulis);
        if code.is_logical_error_indexed(&pauli, &indices.stabilizer, &indices.logical) {
            return Some((
                CandidateIndex {
                    outer: support_index,
                    inner: assignment_index,
                },
                pauli,
            ));
        }

        if !increment_pauli_assignment(&mut paulis) {
            return None;
        }
        assignment_index += 1;
    }
}

fn matching_paulis_for_support(
    code: &StabilizerCodeSpec,
    indices: &CodeIndices,
    support_index: usize,
    positions: &[usize],
    css_only: bool,
) -> Vec<(CandidateIndex, PauliString)> {
    if css_only {
        let candidates = [
            (
                CandidateIndex {
                    outer: 0,
                    inner: support_index,
                },
                PauliString::xs(positions),
            ),
            (
                CandidateIndex {
                    outer: 1,
                    inner: support_index,
                },
                PauliString::zs(positions),
            ),
        ];

        return candidates
            .into_iter()
            .filter(|(_, pauli)| {
                code.is_logical_error_indexed(pauli, &indices.stabilizer, &indices.logical)
            })
            .collect();
    }

    let mut matches = Vec::new();
    let mut paulis = vec![0; positions.len()];
    let mut assignment_index = 0;

    loop {
        let pauli = pauli_string_for_assignment(positions, &paulis);
        if code.is_logical_error_indexed(&pauli, &indices.stabilizer, &indices.logical) {
            matches.push((
                CandidateIndex {
                    outer: support_index,
                    inner: assignment_index,
                },
                pauli,
            ));
        }

        if !increment_pauli_assignment(&mut paulis) {
            break;
        }
        assignment_index += 1;
    }

    matches
}

fn pauli_string_for_assignment(positions: &[usize], paulis: &[usize]) -> PauliString {
    let paulis = positions
        .iter()
        .zip(paulis)
        .map(|(&position, &pauli)| {
            let pauli = match pauli {
                0 => Pauli::X,
                1 => Pauli::Y,
                _ => Pauli::Z,
            };
            (pauli, QubitId::new(position))
        })
        .collect();

    PauliString::with_phase_and_paulis(pecos_core::QuarterPhase::PlusOne, paulis)
}

fn increment_pauli_assignment(paulis: &mut [usize]) -> bool {
    for index in (0..paulis.len()).rev() {
        if paulis[index] < 2 {
            paulis[index] += 1;
            paulis[index + 1..].fill(0);
            return true;
        }
    }
    false
}

/// Classify which logical operators a given Pauli operator is equivalent to.
///
/// Uses precomputed column indices for O(weight) performance instead of
/// O(k * weight) where k is the number of logical qubits.
///
/// Returns a list of (type, index) pairs where type is 'X' or 'Z'.
/// - If the operator anticommutes with logical Z[i], it contains X[i]
/// - If the operator anticommutes with logical X[i], it contains Z[i]
fn classify_logical_equivalence_indexed(
    log_index: &crate::stabilizer_code_spec::LogicalIndex,
    num_logical_qubits: usize,
    pauli: &PauliString,
) -> Vec<(char, usize)> {
    let mut result = Vec::new();

    // The logical index contains [Z_0, Z_1, ..., Z_{k-1}, X_0, X_1, ..., X_{k-1}]
    // So indices 0..k are logical Zs, and k..2k are logical Xs
    let anticommuting = log_index.find_anticommuting(pauli);

    for idx in anticommuting {
        if idx < num_logical_qubits {
            // Anticommutes with logical Z[idx] -> equivalent to X[idx]
            result.push(('X', idx));
        } else {
            // Anticommutes with logical X[idx - k] -> equivalent to Z[idx - k]
            result.push(('Z', idx - num_logical_qubits));
        }
    }

    // Sort for consistent output (X before Z, then by index)
    result.sort_unstable();

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::{Pauli, PauliOperator, Xs, Zs};

    fn pauli_string(paulis: &[(Pauli, usize)]) -> PauliString {
        PauliString::with_phase_and_paulis(
            pecos_core::QuarterPhase::PlusOne,
            paulis.iter().map(|&(p, q)| (p, QubitId::new(q))).collect(),
        )
    }

    fn five_qubit_code() -> StabilizerCodeSpec {
        // The [[5,1,3]] perfect code
        // Stabilizers: XZZXI, IXZZX, XIXZZ, ZXIXZ
        let stab1 = pauli_string(&[(Pauli::X, 0), (Pauli::Z, 1), (Pauli::Z, 2), (Pauli::X, 3)]);
        let stab2 = pauli_string(&[(Pauli::X, 1), (Pauli::Z, 2), (Pauli::Z, 3), (Pauli::X, 4)]);
        let stab3 = pauli_string(&[(Pauli::X, 0), (Pauli::X, 2), (Pauli::Z, 3), (Pauli::Z, 4)]);
        let stab4 = pauli_string(&[(Pauli::Z, 0), (Pauli::X, 1), (Pauli::X, 3), (Pauli::Z, 4)]);

        // Logical operators for [[5,1,3]]: Z = ZZZZZ, X = XXXXX
        let logical_z = pauli_string(&[
            (Pauli::Z, 0),
            (Pauli::Z, 1),
            (Pauli::Z, 2),
            (Pauli::Z, 3),
            (Pauli::Z, 4),
        ]);
        let logical_x = pauli_string(&[
            (Pauli::X, 0),
            (Pauli::X, 1),
            (Pauli::X, 2),
            (Pauli::X, 3),
            (Pauli::X, 4),
        ]);

        StabilizerCodeSpec::new(
            5,
            vec![stab1, stab2, stab3, stab4],
            vec![logical_z],
            vec![logical_x],
        )
        .unwrap()
    }

    fn color_code_17() -> StabilizerCodeSpec {
        const SUPPORTS: [&[usize]; 8] = [
            &[0, 9, 12, 15],
            &[1, 9, 12, 16],
            &[2, 11, 13, 14],
            &[3, 8, 9, 12],
            &[4, 8, 10, 12, 13, 14, 15, 16],
            &[5, 10, 11, 13],
            &[6, 8, 9, 10, 13, 14, 15, 16],
            &[7, 10, 11, 14],
        ];

        let mut builder = StabilizerCodeSpec::builder(17);
        for support in SUPPORTS {
            builder = builder.check(Xs(support));
        }
        for support in SUPPORTS {
            builder = builder.check(Zs(support));
        }

        builder
            .logical_x(Xs(0..17))
            .logical_z(Zs(0..17))
            .build()
            .expect("[[17,1,5]] color code should be valid")
    }

    fn incomplete_five_qubit_spec() -> StabilizerCodeSpec {
        StabilizerCodeSpec::new(
            5,
            vec![Xs(1..=4), Zs(1..=4)],
            vec![Zs([1, 2])],
            vec![Xs([2, 3])],
        )
        .unwrap()
    }

    #[test]
    fn incomplete_logical_basis_is_rejected_by_exhaustive_distance_apis() {
        let code = incomplete_five_qubit_spec();
        let config = DistanceSearchConfig::default();
        let expected = StabilizerCodeSpecError::IncompleteLogicalBasis {
            supplied_logical_pairs: 1,
            num_logical_qubits: 3,
        };

        assert_eq!(calculate_distance(&code, &config), Err(expected.clone()));
        assert_eq!(
            has_logical_error_at_weight(&code, 1, &config),
            Err(expected.clone())
        );
        assert_eq!(
            find_min_weight_logicals(&code, &config),
            Err(expected.clone())
        );
        assert_eq!(find_shortest_logicals(&code, &config, 1), Err(expected));
    }

    #[test]
    fn test_weighted_pauli_iterator_weight_1() {
        let iter = WeightedPauliIterator::new(3, 1, false);
        let paulis: Vec<_> = iter.collect();

        // Should have 3 qubits * 3 Paulis = 9 operators
        assert_eq!(paulis.len(), 9);

        // First few should be X0, Y0, Z0
        assert_eq!(paulis[0].weight(), 1);
    }

    #[test]
    fn test_weighted_pauli_iterator_css() {
        // CSS mode should only generate pure X and pure Z errors, not mixed
        let iter = WeightedPauliIterator::new(3, 1, true);
        let paulis: Vec<_> = iter.collect();

        // Weight 1: C(3,1) positions * 2 types (X, Z) = 6 operators
        assert_eq!(paulis.len(), 6);

        // First 3 should be pure X errors, last 3 should be pure Z errors
        for p in &paulis[0..3] {
            // Pure X: has X positions but no Z positions
            assert!(
                !p.x_positions().is_empty() && p.z_positions().is_empty(),
                "Expected pure X error, got {p:?}"
            );
        }
        for p in &paulis[3..6] {
            // Pure Z: has Z positions but no X positions
            assert!(
                p.x_positions().is_empty() && !p.z_positions().is_empty(),
                "Expected pure Z error, got {p:?}"
            );
        }
    }

    #[test]
    fn test_weighted_pauli_iterator_css_weight2() {
        // CSS mode at weight 2 should generate pure XX and pure ZZ, not XZ
        let iter = WeightedPauliIterator::new(4, 2, true);
        let paulis: Vec<_> = iter.collect();

        // Weight 2 on 4 qubits: C(4,2) = 6 positions * 2 types = 12 operators
        assert_eq!(paulis.len(), 12);

        // First 6 should be pure XX errors, last 6 should be pure ZZ errors
        for p in &paulis[0..6] {
            assert!(
                !p.x_positions().is_empty() && p.z_positions().is_empty(),
                "Expected pure X error, got {p:?}"
            );
        }
        for p in &paulis[6..12] {
            assert!(
                p.x_positions().is_empty() && !p.z_positions().is_empty(),
                "Expected pure Z error, got {p:?}"
            );
        }
    }

    #[test]
    fn test_weighted_pauli_iterator_weight_2() {
        let iter = WeightedPauliIterator::new(4, 2, false);
        let paulis: Vec<_> = iter.collect();

        // Should have C(4,2) * 3^2 = 6 * 9 = 54 operators
        assert_eq!(paulis.len(), 54);
    }

    #[test]
    fn test_three_qubit_bit_flip_distance() {
        // 3-qubit bit flip code should have distance 1 for X errors
        // (single X error is a logical error for this code when viewed as protecting against Z)

        let stab1 = pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 1)]);
        let stab2 = pauli_string(&[(Pauli::Z, 1), (Pauli::Z, 2)]);
        let logical_z = pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 1), (Pauli::Z, 2)]);
        let logical_x = pauli_string(&[(Pauli::X, 0), (Pauli::X, 1), (Pauli::X, 2)]);

        let code = StabilizerCodeSpec::new(3, vec![stab1, stab2], vec![logical_z], vec![logical_x])
            .unwrap();

        let config = DistanceSearchConfig::default();
        let result = calculate_distance(&code, &config).unwrap();

        // The minimum weight logical operator for this code is a single Z
        // (Z on any qubit commutes with ZZ stabilizers and anticommutes with XXX)
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.distance, 1);
    }

    #[test]
    fn test_five_qubit_code_distance() {
        let code = five_qubit_code();

        // Verify the code is valid
        assert!(code.verify().is_ok());

        let config = DistanceSearchConfig::with_max_weight(3);
        let result = calculate_distance(&code, &config).unwrap();

        // The [[5,1,3]] code has distance 3
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.distance, 3);
    }

    #[test]
    fn test_five_qubit_shortest_logicals_respect_logical_weight_spectrum() {
        let code = five_qubit_code();
        let config = DistanceSearchConfig::default();
        let minimum = find_shortest_logicals(&code, &config, 0).unwrap();
        let delta_one = find_shortest_logicals(&code, &config, 1).unwrap();
        let delta_two = find_shortest_logicals(&code, &config, 2).unwrap();

        assert_eq!(minimum.len(), 30);
        assert_eq!(delta_one.len(), 30);
        assert!(
            delta_one
                .iter()
                .map(|info| &info.operator)
                .eq(minimum.iter().map(|info| &info.operator))
        );

        assert_eq!(delta_two.len(), 48);
        assert_eq!(delta_two.iter().filter(|info| info.weight == 3).count(), 30);
        assert_eq!(delta_two.iter().filter(|info| info.weight == 5).count(), 18);
        assert!(delta_two.iter().all(|info| matches!(info.weight, 3 | 5)));
        assert!(
            delta_two
                .iter()
                .take(minimum.len())
                .map(|info| &info.operator)
                .eq(minimum.iter().map(|info| &info.operator))
        );
    }

    #[test]
    fn test_serial_branch_preserves_candidate_order() {
        let code = five_qubit_code();
        let config = DistanceSearchConfig::default();
        let indices = code.build_indices();
        assert!(!(1..=5).any(|weight| should_parallelize(5, weight, false)));
        let expected: Vec<_> = (1..=5)
            .flat_map(|weight| {
                WeightedPauliIterator::new(code.num_qubits(), weight, config.css_only)
                    .filter(|pauli| {
                        code.is_logical_error_indexed(pauli, &indices.stabilizer, &indices.logical)
                    })
                    .map(move |pauli| (weight, pauli))
            })
            .collect();

        let distance = calculate_distance(&code, &config).unwrap().unwrap();
        assert_eq!(distance.min_weight_operator, expected[0].1);

        let actual = find_shortest_logicals(&code, &config, 2).unwrap();
        assert!(
            actual
                .iter()
                .map(|info| (info.weight, &info.operator))
                .eq(expected.iter().map(|(weight, pauli)| (*weight, pauli)))
        );

        let css_config = DistanceSearchConfig::css();
        let expected_css = (1..=code.num_qubits())
            .flat_map(|weight| {
                WeightedPauliIterator::new(code.num_qubits(), weight, true)
                    .filter(|pauli| {
                        code.is_logical_error_indexed(pauli, &indices.stabilizer, &indices.logical)
                    })
                    .map(move |pauli| (weight, pauli))
            })
            .next()
            .unwrap();
        let actual_css = calculate_distance(&code, &css_config).unwrap().unwrap();
        assert_eq!(actual_css.distance, expected_css.0);
        assert_eq!(actual_css.min_weight_operator, expected_css.1);
    }

    #[test]
    fn test_parallel_branch_preserves_serial_candidate_order() {
        let code = color_code_17();
        let config = DistanceSearchConfig::with_max_weight(5);
        let indices = code.build_indices();
        assert!(should_parallelize(17, 5, false));

        let expected: Vec<_> = WeightedPauliIterator::new(17, 5, false)
            .filter(|pauli| {
                code.is_logical_error_indexed(pauli, &indices.stabilizer, &indices.logical)
            })
            .collect();
        let actual = logical_errors_at_weight(&code, 5, &config, &indices);
        assert_eq!(actual, expected);

        let distance = calculate_distance(&code, &config).unwrap().unwrap();
        assert_eq!(distance.distance, 5);
        assert_eq!(distance.min_weight_operator, expected[0]);
    }

    #[test]
    fn test_logical_equivalence_tracking() {
        // 3-qubit bit flip code
        let stab1 = pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 1)]);
        let stab2 = pauli_string(&[(Pauli::Z, 1), (Pauli::Z, 2)]);
        let logical_z = pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 1), (Pauli::Z, 2)]);
        let logical_x = pauli_string(&[(Pauli::X, 0), (Pauli::X, 1), (Pauli::X, 2)]);

        let code = StabilizerCodeSpec::new(3, vec![stab1, stab2], vec![logical_z], vec![logical_x])
            .unwrap();

        let config = DistanceSearchConfig::with_max_weight(2);
        let logicals = find_shortest_logicals(&code, &config, 0).unwrap();

        // Should find single-qubit Z errors (equivalent to Z0)
        // and single-qubit X errors (equivalent to X0)
        assert!(!logicals.is_empty());

        // All weight-1 operators should have exactly one equivalent logical
        for info in &logicals {
            assert_eq!(info.weight, 1);
            assert!(!info.equivalent_logicals.is_empty());

            // Check that the equivalence string is sensible
            let equiv_str = info.equivalence_string();
            assert!(equiv_str == "X0" || equiv_str == "Z0", "Got: {equiv_str}");
        }
    }

    #[test]
    fn test_logical_equivalence_string() {
        let info = LogicalOperatorInfo {
            operator: pauli_string(&[(Pauli::X, 0)]),
            weight: 1,
            equivalent_logicals: vec![('X', 0), ('Z', 1)],
        };
        assert_eq!(info.equivalence_string(), "X0*Z1");

        let info2 = LogicalOperatorInfo {
            operator: pauli_string(&[(Pauli::Z, 0)]),
            weight: 1,
            equivalent_logicals: vec![],
        };
        assert_eq!(info2.equivalence_string(), "I");
    }
}
