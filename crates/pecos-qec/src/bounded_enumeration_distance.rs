// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Exact binary code distance by bounded generator-row enumeration.
//!
//! This is the information-set method treated for quantum codes in
//! [arXiv:2408.10743](https://arxiv.org/abs/2408.10743). Its initial upper bound also tries a
//! fixed number of seeded random information sets, following the upper-bound idea in
//! [arXiv:2308.15140](https://arxiv.org/abs/2308.15140). Randomness can only improve the upper
//! bound; exactness follows entirely from the deterministic information-set lower bound.

use crate::code_distance::mechanisms_from_stabilizer_code;
use crate::{DistanceProblem, DistanceProblemError, ParityCheckMatrix, StabilizerCodeSpec};
use pecos_quantum::F2Matrix;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

const RANDOM_INFORMATION_SET_TRIALS: usize = 64;
const RANDOM_INFORMATION_SET_SEED: u64 = 0xB0A0_3E12_1A11_5EED;

/// Result of a bounded generator-row enumeration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BoundedEnumerationDistance {
    /// Matching native lower and upper bounds certify the exact distance.
    CertifiedByBounds {
        /// Exact minimum Hamming weight.
        distance: usize,
        /// Binary assignment attaining `distance`.
        witness: Vec<bool>,
        /// Native lower bound at termination.
        lower_bound: usize,
        /// Last fully enumerated generator-row level.
        level: usize,
        /// Always true: no external solver is trusted for the lower bound.
        lb_certified: bool,
    },
    /// The level budget ended before the bounds met.
    LevelLimitReached {
        /// Proven native lower bound on the distance.
        lower_bound: usize,
        /// Weight of the best natively verified witness found.
        upper_bound: usize,
        /// Binary assignment attaining `upper_bound`.
        witness: Vec<bool>,
        /// Requested maximum generator-row level.
        max_level: usize,
        /// Always true: the lower bound is established by native enumeration.
        lb_certified: bool,
    },
}

impl BoundedEnumerationDistance {
    /// Returns the proven lower bound.
    #[must_use]
    pub fn lower_bound(&self) -> usize {
        match self {
            Self::CertifiedByBounds { lower_bound, .. }
            | Self::LevelLimitReached { lower_bound, .. } => *lower_bound,
        }
    }

    /// Returns the best natively verified upper bound.
    #[must_use]
    pub fn upper_bound(&self) -> usize {
        match self {
            Self::CertifiedByBounds { distance, .. } => *distance,
            Self::LevelLimitReached { upper_bound, .. } => *upper_bound,
        }
    }

    /// Returns the witness attaining [`Self::upper_bound`].
    #[must_use]
    pub fn witness(&self) -> &[bool] {
        match self {
            Self::CertifiedByBounds { witness, .. } | Self::LevelLimitReached { witness, .. } => {
                witness
            }
        }
    }

    /// Returns whether the lower and upper bounds prove exact distance.
    #[must_use]
    pub fn is_certified(&self) -> bool {
        matches!(self, Self::CertifiedByBounds { .. })
    }
}

/// Packed rows from one information set supplied to a level-enumeration backend.
///
/// Each row occupies [`LevelEnumerationInput::row_stride_words`] consecutive `u32` words, with
/// column zero in the least-significant bit of the first word.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedSystematicGenerator {
    /// Concatenated packed generator rows.
    pub rows: Vec<u32>,
}

/// One bounded-enumeration level presented to a backend.
///
/// `active_systematic_indices` preserves the information-set order used by the exact CPU search.
/// A backend must examine every `level`-combination of the `dimension` rows for each listed
/// systematic generator and minimize full-codeword weight among combinations with a nonzero
/// logical effect.
#[derive(Clone, Copy, Debug)]
pub struct LevelEnumerationInput<'a> {
    /// Generator-row combination size for this level.
    pub level: usize,
    /// Number of rows in every systematic generator.
    pub dimension: usize,
    /// Number of binary columns in a codeword.
    pub codeword_bits: usize,
    /// Packed words occupied by one generator or logical row.
    pub row_stride_words: usize,
    /// All peeled systematic generators.
    pub systematic_generators: &'a [PackedSystematicGenerator],
    /// Indices of the systematic generators active at this level.
    pub active_systematic_indices: &'a [usize],
    /// Concatenated packed logical rows.
    pub logical_rows: &'a [u32],
    /// Number of packed logical rows.
    pub logical_count: usize,
}

/// Minimum found by a level-enumeration backend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LevelEnumerationMinimum {
    /// Minimum full-codeword weight with a nonzero logical effect, or `None` if none exists.
    pub weight: Option<usize>,
    /// Optional packed witness attaining `weight`.
    ///
    /// If supplied, this must be the first minimum in the systematic-generator and lexicographic
    /// combination order described by [`LevelEnumerationInput`]. Backends may omit it; the host
    /// then deterministically reconstructs it with a CPU re-scan when the minimum improves the
    /// current upper bound.
    pub witness: Option<Vec<u32>>,
}

/// Backend seam for the branchless combination loop in bounded enumeration.
///
/// The enumeration structure follows
/// [arXiv:2408.10743](https://arxiv.org/abs/2408.10743). Implementations replace only a level's
/// combination enumeration; bounds, termination, witness verification, and tie-breaking remain
/// under the native host algorithm's control.
pub trait LevelEnumerationBackend {
    /// Error returned when a level cannot be enumerated.
    type Error;

    /// Returns the minimum logical codeword weight over all combinations in `input`.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] if the backend cannot completely enumerate the requested level.
    fn enumerate_level(
        &mut self,
        input: LevelEnumerationInput<'_>,
    ) -> Result<LevelEnumerationMinimum, Self::Error>;
}

/// Error from bounded enumeration using an external level backend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BoundedEnumerationBackendError<E> {
    /// The requested distance problem is invalid.
    DistanceProblem(DistanceProblemError),
    /// The level backend failed.
    Backend(E),
}

impl<E: std::fmt::Display> std::fmt::Display for BoundedEnumerationBackendError<E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DistanceProblem(error) => error.fmt(formatter),
            Self::Backend(error) => write!(formatter, "level-enumeration backend failed: {error}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for BoundedEnumerationBackendError<E> {}

#[derive(Clone, Debug)]
struct SystematicGenerator {
    rows: Vec<Vec<u8>>,
    rank: usize,
}

fn restricted_systematic_generator(
    generator: &F2Matrix,
    columns: &[usize],
) -> (SystematicGenerator, Vec<usize>) {
    let restricted_width = columns.len();
    let full_width = generator.num_cols();
    let mut augmented = F2Matrix::zeros(generator.num_rows(), restricted_width + full_width);
    for row in 0..generator.num_rows() {
        for (restricted_column, &full_column) in columns.iter().enumerate() {
            augmented.set(row, restricted_column, generator.get(row, full_column));
        }
        for column in 0..full_width {
            augmented.set(row, restricted_width + column, generator.get(row, column));
        }
    }

    // The restricted block comes first, so its pivots are chosen before the full generator block.
    // Later pivots can change full rows but are zero on the restricted block, preserving its RREF.
    let (reduced, pivots) = augmented.row_reduce();
    let restricted_pivots: Vec<_> = pivots
        .into_iter()
        .take_while(|&pivot| pivot < restricted_width)
        .collect();
    let rows = (0..generator.num_rows())
        .map(|row| {
            (0..full_width)
                .map(|column| reduced.get(row, restricted_width + column))
                .collect()
        })
        .collect();
    (
        SystematicGenerator {
            rows,
            rank: restricted_pivots.len(),
        },
        restricted_pivots,
    )
}

fn peel_information_sets(generator: &F2Matrix) -> Vec<SystematicGenerator> {
    let mut active_columns: Vec<_> = (0..generator.num_cols()).collect();
    let mut systematic_generators = Vec::new();
    loop {
        let (systematic, restricted_pivots) =
            restricted_systematic_generator(generator, &active_columns);
        if systematic.rank == 0 {
            break;
        }
        let mut is_pivot = vec![false; active_columns.len()];
        for pivot in restricted_pivots {
            is_pivot[pivot] = true;
        }
        active_columns = active_columns
            .into_iter()
            .enumerate()
            .filter_map(|(index, column)| (!is_pivot[index]).then_some(column))
            .collect();
        systematic_generators.push(systematic);
    }
    systematic_generators
}

fn row_weight(row: &[u8]) -> usize {
    row.iter().map(|&bit| usize::from(bit)).sum()
}

fn has_logical_effect(row: &[u8], logicals: &F2Matrix) -> bool {
    (0..logicals.num_rows()).any(|logical| {
        row.iter()
            .enumerate()
            .filter(|&(column, &bit)| bit == 1 && logicals.get(logical, column) == 1)
            .count()
            % 2
            == 1
    })
}

fn update_upper_bound(candidate: &[u8], logicals: &F2Matrix, best: &mut Option<Vec<u8>>) {
    if !has_logical_effect(candidate, logicals) {
        return;
    }
    let candidate_weight = row_weight(candidate);
    if best
        .as_ref()
        .is_none_or(|current| candidate_weight < row_weight(current))
    {
        *best = Some(candidate.to_vec());
    }
}

fn seeded_upper_bound(
    generator: &F2Matrix,
    systematic_generators: &[SystematicGenerator],
    logicals: &F2Matrix,
) -> Option<Vec<u8>> {
    let mut best = None;
    for systematic in systematic_generators {
        for row in &systematic.rows {
            update_upper_bound(row, logicals, &mut best);
        }
    }

    let mut rng = SmallRng::seed_from_u64(RANDOM_INFORMATION_SET_SEED);
    let mut columns: Vec<_> = (0..generator.num_cols()).collect();
    for _ in 0..RANDOM_INFORMATION_SET_TRIALS {
        columns.shuffle(&mut rng);
        let (systematic, _) = restricted_systematic_generator(generator, &columns);
        for row in &systematic.rows {
            update_upper_bound(row, logicals, &mut best);
        }
    }
    best
}

fn for_each_combination(count: usize, choose: usize, mut visit: impl FnMut(&[usize])) {
    if choose > count {
        return;
    }
    if choose == 0 {
        visit(&[]);
        return;
    }
    let mut combination: Vec<_> = (0..choose).collect();
    loop {
        visit(&combination);
        let Some(position) = (0..choose)
            .rev()
            .find(|&position| combination[position] < count - choose + position)
        else {
            break;
        };
        combination[position] += 1;
        for index in position + 1..choose {
            combination[index] = combination[index - 1] + 1;
        }
    }
}

fn pack_rows(rows: &[Vec<u8>], row_stride_words: usize) -> Vec<u32> {
    let mut packed = vec![0; rows.len() * row_stride_words];
    for (row_index, row) in rows.iter().enumerate() {
        for (column, &bit) in row.iter().enumerate() {
            if bit != 0 {
                packed[row_index * row_stride_words + column / 32] |= 1 << (column % 32);
            }
        }
    }
    packed
}

fn pack_matrix(matrix: &F2Matrix, row_stride_words: usize) -> Vec<u32> {
    pack_rows(&matrix.rows(), row_stride_words)
}

fn packed_combination_codeword(
    rows: &[u32],
    row_stride_words: usize,
    combination: &[usize],
) -> Vec<u32> {
    let mut codeword = vec![0; row_stride_words];
    for &row in combination {
        for (word, &generator_word) in codeword
            .iter_mut()
            .zip(&rows[row * row_stride_words..][..row_stride_words])
        {
            *word ^= generator_word;
        }
    }
    codeword
}

fn packed_weight(row: &[u32]) -> usize {
    row.iter().map(|word| word.count_ones() as usize).sum()
}

fn packed_has_logical_effect(
    row: &[u32],
    logical_rows: &[u32],
    logical_count: usize,
    row_stride_words: usize,
) -> bool {
    (0..logical_count).any(|logical| {
        row.iter()
            .zip(&logical_rows[logical * row_stride_words..][..row_stride_words])
            .map(|(&word, &logical_word)| (word & logical_word).count_ones())
            .sum::<u32>()
            % 2
            == 1
    })
}

fn unpack_codeword(row: &[u32], codeword_bits: usize) -> Vec<u8> {
    (0..codeword_bits)
        .map(|column| ((row[column / 32] >> (column % 32)) & 1) as u8)
        .collect()
}

/// Native CPU implementation of [`LevelEnumerationBackend`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuLevelEnumerationBackend;

impl LevelEnumerationBackend for CpuLevelEnumerationBackend {
    type Error = std::convert::Infallible;

    fn enumerate_level(
        &mut self,
        input: LevelEnumerationInput<'_>,
    ) -> Result<LevelEnumerationMinimum, Self::Error> {
        let mut best_weight = None;
        let mut best_witness = None;
        for &systematic_index in input.active_systematic_indices {
            let rows = &input.systematic_generators[systematic_index].rows;
            for_each_combination(input.dimension, input.level, |combination| {
                let candidate =
                    packed_combination_codeword(rows, input.row_stride_words, combination);
                if packed_has_logical_effect(
                    &candidate,
                    input.logical_rows,
                    input.logical_count,
                    input.row_stride_words,
                ) {
                    let weight = packed_weight(&candidate);
                    if best_weight.is_none_or(|current| weight < current) {
                        best_weight = Some(weight);
                        best_witness = Some(candidate);
                    }
                }
            });
        }
        Ok(LevelEnumerationMinimum {
            weight: best_weight,
            witness: best_witness,
        })
    }
}

fn lower_bound_after_level(
    level: usize,
    dimension: usize,
    systematic_generators: &[SystematicGenerator],
    even_code: bool,
) -> usize {
    let mut lower_bound = systematic_generators
        .iter()
        .map(|systematic| (level + 1).saturating_sub(dimension - systematic.rank))
        .filter(|&contribution| contribution != 0)
        .sum();
    if even_code && lower_bound % 2 == 1 {
        lower_bound += 1;
    }
    lower_bound
}

fn verify_result_witness(
    h: &ParityCheckMatrix,
    l: &ParityCheckMatrix,
    candidate: &[u8],
) -> Vec<bool> {
    let witness: Vec<_> = candidate.iter().map(|&bit| bit == 1).collect();
    let problem = DistanceProblem::from_css_checks(h, l)
        .expect("bounded-enumeration matrices were width-checked before search");
    let verified_weight = problem
        .verify_witness(&witness)
        .expect("bounded enumeration produced an invalid witness");
    assert_eq!(verified_weight, row_weight(candidate));
    witness
}

fn parity_check_from_matrix(matrix: &F2Matrix) -> ParityCheckMatrix {
    if matrix.num_rows() == 0 {
        ParityCheckMatrix::zeros(0, matrix.num_cols())
    } else {
        ParityCheckMatrix::from_dense(matrix.rows()).expect("F2Matrix entries are binary")
    }
}

fn reconstruct_level_witness(input: LevelEnumerationInput<'_>, target_weight: usize) -> Vec<u8> {
    for &systematic_index in input.active_systematic_indices {
        let rows = &input.systematic_generators[systematic_index].rows;
        let mut witness = None;
        for_each_combination(input.dimension, input.level, |combination| {
            if witness.is_some() {
                return;
            }
            let candidate = packed_combination_codeword(rows, input.row_stride_words, combination);
            if packed_weight(&candidate) == target_weight
                && packed_has_logical_effect(
                    &candidate,
                    input.logical_rows,
                    input.logical_count,
                    input.row_stride_words,
                )
            {
                witness = Some(unpack_codeword(&candidate, input.codeword_bits));
            }
        });
        if let Some(witness) = witness {
            return witness;
        }
    }
    panic!("level-enumeration backend returned a minimum with no matching witness")
}

fn matrix_distance_with_backend<B: LevelEnumerationBackend>(
    h: &ParityCheckMatrix,
    l: &ParityCheckMatrix,
    max_level: usize,
    backend: &mut B,
) -> Result<Option<BoundedEnumerationDistance>, B::Error> {
    assert_eq!(
        h.num_qubits(),
        l.num_qubits(),
        "code-distance matrices must have matching widths"
    );
    let generator = F2Matrix::from_rows(h.matrix().kernel());
    let dimension = generator.num_rows();
    if dimension == 0 {
        return Ok(None);
    }
    let systematic_generators = peel_information_sets(&generator);
    let Some(mut best) = seeded_upper_bound(&generator, &systematic_generators, l.matrix()) else {
        return Ok(None);
    };
    let row_stride_words = generator.num_cols().div_ceil(32);
    let packed_systematic_generators: Vec<_> = systematic_generators
        .iter()
        .map(|systematic| PackedSystematicGenerator {
            rows: pack_rows(&systematic.rows, row_stride_words),
        })
        .collect();
    let logical_rows = pack_matrix(l.matrix(), row_stride_words);
    // The first systematic generator is a basis of the whole code. Weight parity is linear over
    // GF(2), so even basis rows prove that every generated codeword has even weight.
    let even_code = generator
        .rows()
        .iter()
        .all(|row| row_weight(row).is_multiple_of(2));
    let mut lower_bound = lower_bound_after_level(0, dimension, &systematic_generators, even_code);
    if lower_bound >= row_weight(&best) {
        let witness = verify_result_witness(h, l, &best);
        return Ok(Some(BoundedEnumerationDistance::CertifiedByBounds {
            distance: row_weight(&best),
            witness,
            lower_bound,
            level: 0,
            lb_certified: true,
        }));
    }

    for level in 1..=dimension {
        if level > max_level {
            let witness = verify_result_witness(h, l, &best);
            return Ok(Some(BoundedEnumerationDistance::LevelLimitReached {
                lower_bound,
                upper_bound: row_weight(&best),
                witness,
                max_level,
                lb_certified: true,
            }));
        }
        let active_systematic_indices: Vec<_> = systematic_generators
            .iter()
            .enumerate()
            .filter_map(|(index, systematic)| {
                let contribution = (level + 1).saturating_sub(dimension - systematic.rank);
                (contribution != 0).then_some(index)
            })
            .collect();
        // Information sets omitted here add zero to this level's lower bound. Their candidates
        // cannot weaken the certificate, while the full-rank first information set still makes
        // the upper-bound search complete.
        let input = LevelEnumerationInput {
            level,
            dimension,
            codeword_bits: generator.num_cols(),
            row_stride_words,
            systematic_generators: &packed_systematic_generators,
            active_systematic_indices: &active_systematic_indices,
            logical_rows: &logical_rows,
            logical_count: l.matrix().num_rows(),
        };
        let level_minimum = backend.enumerate_level(input)?;
        if let Some(weight) = level_minimum.weight
            && weight < row_weight(&best)
        {
            best = if let Some(witness) = level_minimum.witness {
                assert_eq!(witness.len(), row_stride_words);
                assert_eq!(packed_weight(&witness), weight);
                assert!(packed_has_logical_effect(
                    &witness,
                    &logical_rows,
                    l.matrix().num_rows(),
                    row_stride_words,
                ));
                unpack_codeword(&witness, generator.num_cols())
            } else {
                // A minimum-only backend (including the GPU backend) intentionally pays for this
                // bounded CPU replay only after improving the upper bound. It preserves the exact
                // witness and tie-breaking behavior of the native systematic/lexicographic loop.
                reconstruct_level_witness(input, weight)
            };
        }
        lower_bound = lower_bound_after_level(level, dimension, &systematic_generators, even_code);
        if lower_bound >= row_weight(&best) {
            let witness = verify_result_witness(h, l, &best);
            return Ok(Some(BoundedEnumerationDistance::CertifiedByBounds {
                distance: row_weight(&best),
                witness,
                lower_bound,
                level,
                lb_certified: true,
            }));
        }
    }
    unreachable!("enumerating every generator row must exhaust a finite binary code")
}

fn matrix_distance(
    h: &ParityCheckMatrix,
    l: &ParityCheckMatrix,
    max_level: usize,
) -> Option<BoundedEnumerationDistance> {
    let mut backend = CpuLevelEnumerationBackend;
    match matrix_distance_with_backend(h, l, max_level, &mut backend) {
        Ok(result) => result,
        Err(error) => match error {},
    }
}

/// Computes binary `(H, L)` distance by bounded generator-row enumeration.
///
/// `H e = 0` enforces undetectability and `L e != 0` enforces a nontrivial effect. `None` means
/// no nontrivial vector exists in `ker(H)`. Exceeding `max_level` returns a certified interval,
/// not an absent result.
///
/// # Panics
///
/// Panics if the matrices have different widths.
#[must_use]
pub fn bounded_enumeration_code_distance(
    h: &ParityCheckMatrix,
    l: &ParityCheckMatrix,
    max_level: usize,
) -> Option<BoundedEnumerationDistance> {
    matrix_distance(h, l, max_level)
}

/// Computes binary `(H, L)` distance using an external level-enumeration backend.
///
/// Bounds, deterministic witness selection, and termination remain identical to
/// [`bounded_enumeration_code_distance`].
///
/// # Errors
///
/// Returns an error if `backend` cannot enumerate a required level.
///
/// # Panics
///
/// Panics if the matrices have different widths.
pub fn bounded_enumeration_code_distance_with_backend<B: LevelEnumerationBackend>(
    h: &ParityCheckMatrix,
    l: &ParityCheckMatrix,
    max_level: usize,
    backend: &mut B,
) -> Result<Option<BoundedEnumerationDistance>, B::Error> {
    matrix_distance_with_backend(h, l, max_level, backend)
}

/// Computes pure-X bounded-enumeration distance for a CSS-form stabilizer code.
///
/// # Errors
///
/// Returns the same CSS-form and bounds errors as
/// [`crate::DistanceProblem::from_css_code_x_distance`].
pub fn bounded_enumeration_x_distance(
    code: &StabilizerCodeSpec,
    max_level: usize,
) -> Result<Option<BoundedEnumerationDistance>, DistanceProblemError> {
    let problem = DistanceProblem::from_css_code_x_distance(code)?;
    let (h, l) = problem.matrices();
    let h = parity_check_from_matrix(h);
    let l = parity_check_from_matrix(l);
    Ok(matrix_distance(&h, &l, max_level))
}

/// Computes pure-X bounded-enumeration distance using an external level backend.
///
/// # Errors
///
/// Returns CSS problem errors or errors reported by `backend`.
pub fn bounded_enumeration_x_distance_with_backend<B: LevelEnumerationBackend>(
    code: &StabilizerCodeSpec,
    max_level: usize,
    backend: &mut B,
) -> Result<Option<BoundedEnumerationDistance>, BoundedEnumerationBackendError<B::Error>> {
    let problem = DistanceProblem::from_css_code_x_distance(code)
        .map_err(BoundedEnumerationBackendError::DistanceProblem)?;
    let (h, l) = problem.matrices();
    let h = parity_check_from_matrix(h);
    let l = parity_check_from_matrix(l);
    matrix_distance_with_backend(&h, &l, max_level, backend)
        .map_err(BoundedEnumerationBackendError::Backend)
}

/// Computes pure-Z bounded-enumeration distance for a CSS-form stabilizer code.
///
/// # Errors
///
/// Returns the same CSS-form and bounds errors as
/// [`crate::DistanceProblem::from_css_code_z_distance`].
pub fn bounded_enumeration_z_distance(
    code: &StabilizerCodeSpec,
    max_level: usize,
) -> Result<Option<BoundedEnumerationDistance>, DistanceProblemError> {
    let problem = DistanceProblem::from_css_code_z_distance(code)?;
    let (h, l) = problem.matrices();
    let h = parity_check_from_matrix(h);
    let l = parity_check_from_matrix(l);
    Ok(matrix_distance(&h, &l, max_level))
}

/// Computes pure-Z bounded-enumeration distance using an external level backend.
///
/// # Errors
///
/// Returns CSS problem errors or errors reported by `backend`.
pub fn bounded_enumeration_z_distance_with_backend<B: LevelEnumerationBackend>(
    code: &StabilizerCodeSpec,
    max_level: usize,
    backend: &mut B,
) -> Result<Option<BoundedEnumerationDistance>, BoundedEnumerationBackendError<B::Error>> {
    let problem = DistanceProblem::from_css_code_z_distance(code)
        .map_err(BoundedEnumerationBackendError::DistanceProblem)?;
    let (h, l) = problem.matrices();
    let h = parity_check_from_matrix(h);
    let l = parity_check_from_matrix(l);
    matrix_distance_with_backend(&h, &l, max_level, backend)
        .map_err(BoundedEnumerationBackendError::Backend)
}

/// Computes bounded-enumeration distance for any stabilizer code specification.
///
/// Each qubit contributes X, Y, and Z binary columns, in that order. As in
/// [`crate::stabilizer_code_distance`], two selected columns on one qubit can always be replaced
/// by the third with the same `(H, L)` effect and lower weight, so the exact binary minimum equals
/// physical Pauli weight.
///
/// # Errors
///
/// Returns [`DistanceProblemError::QubitOutOfRange`] if an operator addresses a qubit outside the
/// declared code width.
pub fn bounded_enumeration_stabilizer_distance(
    code: &StabilizerCodeSpec,
    max_level: usize,
) -> Result<Option<BoundedEnumerationDistance>, DistanceProblemError> {
    code.verify_logical_completeness()?;
    let mechanisms = mechanisms_from_stabilizer_code(code)?;
    let mut h = F2Matrix::zeros(code.stabilizers().len(), mechanisms.len());
    let mut l = F2Matrix::zeros(
        code.logical_zs().len() + code.logical_xs().len(),
        mechanisms.len(),
    );
    for (column, mechanism) in mechanisms.iter().enumerate() {
        for &detector in &mechanism.detectors {
            h.set(detector as usize, column, 1);
        }
        for &output in &mechanism.dem_outputs {
            l.set(output as usize, column, 1);
        }
    }
    let h = parity_check_from_matrix(&h);
    let l = parity_check_from_matrix(&l);
    Ok(matrix_distance(&h, &l, max_level))
}

/// Computes general stabilizer-code distance using an external level backend.
///
/// # Errors
///
/// Returns stabilizer problem errors or errors reported by `backend`.
pub fn bounded_enumeration_stabilizer_distance_with_backend<B: LevelEnumerationBackend>(
    code: &StabilizerCodeSpec,
    max_level: usize,
    backend: &mut B,
) -> Result<Option<BoundedEnumerationDistance>, BoundedEnumerationBackendError<B::Error>> {
    code.verify_logical_completeness()
        .map_err(DistanceProblemError::from)
        .map_err(BoundedEnumerationBackendError::DistanceProblem)?;
    let mechanisms = mechanisms_from_stabilizer_code(code)
        .map_err(BoundedEnumerationBackendError::DistanceProblem)?;
    let mut h = F2Matrix::zeros(code.stabilizers().len(), mechanisms.len());
    let mut l = F2Matrix::zeros(
        code.logical_zs().len() + code.logical_xs().len(),
        mechanisms.len(),
    );
    for (column, mechanism) in mechanisms.iter().enumerate() {
        for &detector in &mechanism.detectors {
            h.set(detector as usize, column, 1);
        }
        for &output in &mechanism.dem_outputs {
            l.set(output as usize, column, 1);
        }
    }
    let h = parity_check_from_matrix(&h);
    let l = parity_check_from_matrix(&l);
    matrix_distance_with_backend(&h, &l, max_level, backend)
        .map_err(BoundedEnumerationBackendError::Backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BivariateBicycleCode, StabilizerCode, certified_distance, connected_cluster_code_distance,
        stabilizer_code_distance,
    };
    use rand::{RngExt, SeedableRng};
    use std::time::Instant;

    fn matrix_from_f2(matrix: &F2Matrix) -> ParityCheckMatrix {
        parity_check_from_matrix(matrix)
    }

    fn checks_for_generator(rows: Vec<Vec<u8>>) -> ParityCheckMatrix {
        let generator = F2Matrix::from_rows(rows);
        matrix_from_f2(&F2Matrix::from_rows(generator.kernel()))
    }

    fn assert_certified_distance(
        result: &BoundedEnumerationDistance,
        expected: usize,
    ) -> (&[bool], usize) {
        match result {
            BoundedEnumerationDistance::CertifiedByBounds {
                distance,
                witness,
                lower_bound,
                level,
                lb_certified,
            } => {
                assert_eq!(*distance, expected);
                assert!(*lower_bound >= *distance);
                assert!(*lb_certified);
                (witness, *level)
            }
            BoundedEnumerationDistance::LevelLimitReached { .. } => {
                panic!("expected exact distance, got a bounded interval")
            }
        }
    }

    fn steane_pair() -> (ParityCheckMatrix, ParityCheckMatrix) {
        (
            ParityCheckMatrix::from_dense(vec![
                vec![1, 0, 1, 0, 1, 0, 1],
                vec![0, 1, 1, 0, 0, 1, 1],
                vec![0, 0, 0, 1, 1, 1, 1],
            ])
            .unwrap(),
            ParityCheckMatrix::from_dense(vec![vec![1; 7]]).unwrap(),
        )
    }

    #[test]
    fn steane_agrees_with_connected_cluster_and_sat() {
        let (h, l) = steane_pair();
        let bounded = bounded_enumeration_code_distance(&h, &l, 4).unwrap();
        let connected = connected_cluster_code_distance(&h, &l, 3).unwrap();
        let problem = DistanceProblem::from_css_checks(&h, &l).unwrap();
        let sat = certified_distance(&problem, 3).unwrap().unwrap();
        let (witness, _) = assert_certified_distance(&bounded, 3);

        assert_eq!(connected.distance, 3);
        assert_eq!(sat.distance, 3);
        assert_eq!(problem.verify_witness(witness), Ok(3));
    }

    #[test]
    fn five_qubit_three_mechanism_reduction_agrees_with_other_searches() {
        let spec = StabilizerCodeSpec::from_stabilizer_code(&StabilizerCode::five_qubit()).unwrap();
        let bounded = bounded_enumeration_stabilizer_distance(&spec, 5)
            .unwrap()
            .unwrap();
        let connected = match stabilizer_code_distance(&spec, 3).unwrap() {
            crate::StabilizerDistanceSearchOutcome::Certified(result) => result,
            crate::StabilizerDistanceSearchOutcome::BudgetExhausted { max_weight } => {
                panic!("expected certified distance, exhausted weight {max_weight}")
            }
        };
        let symplectic = DistanceProblem::from_stabilizer_spec(&spec).unwrap();
        let sat = certified_distance(&symplectic, 3).unwrap().unwrap();

        let (witness, _) = assert_certified_distance(&bounded, 3);
        assert_eq!(connected.distance, 3);
        assert_eq!(sat.distance, 3);
        assert_eq!(witness.iter().filter(|&&selected| selected).count(), 3);
        assert_eq!(witness.len(), 3 * spec.num_qubits());
    }

    #[test]
    fn yy_code_uses_a_single_y_mechanism() {
        use pecos_core::{X, Y, Ys, Z};

        let spec = StabilizerCodeSpec::builder(2)
            .check(Ys([0, 1]))
            .logical_z(Y(0))
            .logical_x(X(0) & Z(1))
            .build_verified()
            .unwrap();
        let bounded = bounded_enumeration_stabilizer_distance(&spec, 2)
            .unwrap()
            .unwrap();
        let (witness, _) = assert_certified_distance(&bounded, 1);
        let selected = witness.iter().position(|&bit| bit).unwrap();

        assert_eq!(selected % 3, 1, "the minimum mechanism must be Y");
    }

    #[test]
    fn bb_72_agrees_with_connected_cluster_and_sat() {
        let code =
            BivariateBicycleCode::new(6, 6, &[(3, 0), (0, 1), (0, 2)], &[(0, 3), (1, 0), (2, 0)])
                .unwrap();
        let bounded = bounded_enumeration_code_distance(code.hx(), code.logical_x(), 6).unwrap();
        let connected = connected_cluster_code_distance(code.hx(), code.logical_x(), 6).unwrap();
        let problem = DistanceProblem::from_css_checks(code.hx(), code.logical_x()).unwrap();
        let sat = certified_distance(&problem, 6).unwrap().unwrap();
        let (witness, _) = assert_certified_distance(&bounded, 6);

        assert_eq!(connected.distance, 6);
        assert_eq!(sat.distance, 6);
        assert_eq!(problem.verify_witness(witness), Ok(6));
    }

    #[test]
    fn level_budget_returns_an_honest_interval() {
        let (h, l) = steane_pair();
        let result = bounded_enumeration_code_distance(&h, &l, 0).unwrap();
        let problem = DistanceProblem::from_css_checks(&h, &l).unwrap();

        match result {
            BoundedEnumerationDistance::LevelLimitReached {
                lower_bound,
                upper_bound,
                witness,
                max_level,
                lb_certified,
            } => {
                assert_eq!(lower_bound, 1);
                assert_eq!(upper_bound, 3);
                assert_eq!(max_level, 0);
                assert!(lb_certified);
                assert_eq!(problem.verify_witness(&witness), Ok(upper_bound));
            }
            BoundedEnumerationDistance::CertifiedByBounds { .. } => {
                panic!("Steane must not certify before row enumeration")
            }
        }
    }

    #[test]
    fn lower_bound_off_by_one_changes_the_hand_analyzed_termination_level() {
        // This [6,2,4] code has three disjoint full information sets. After enumerating level one,
        // the required (d+1) formula gives LB=6 and terminates at UB=4. Replacing d+1 by d gives
        // only LB=3 and therefore changes the asserted termination level.
        let h = checks_for_generator(vec![vec![1, 1, 1, 1, 0, 0], vec![0, 0, 1, 1, 1, 1]]);
        let l = ParityCheckMatrix::from_dense(vec![vec![1, 0, 0, 0, 0, 0]]).unwrap();
        let result = bounded_enumeration_code_distance(&h, &l, 2).unwrap();
        let (witness, level) = assert_certified_distance(&result, 4);

        assert_eq!(level, 1);
        assert_eq!(
            DistanceProblem::from_css_checks(&h, &l)
                .unwrap()
                .verify_witness(witness),
            Ok(4)
        );
    }

    #[test]
    fn even_weight_rounding_tightens_the_initial_certificate() {
        // The code generated by 1010 and 1001 is even. Its peeled ranks are [2,1], so the raw
        // level-zero bound is one; evenness raises the honest bound to the exact value two.
        let h = checks_for_generator(vec![vec![1, 0, 1, 0], vec![1, 0, 0, 1]]);
        let l = ParityCheckMatrix::from_dense(vec![vec![1, 0, 0, 0]]).unwrap();
        let result = bounded_enumeration_code_distance(&h, &l, 0).unwrap();
        let (_, level) = assert_certified_distance(&result, 2);

        assert_eq!(result.lower_bound(), 2);
        assert_eq!(level, 0);
    }

    fn exhaustive_distance(problem: &DistanceProblem) -> Option<usize> {
        (0..1usize << problem.num_vars())
            .filter_map(|mask| {
                let witness: Vec<_> = (0..problem.num_vars())
                    .map(|bit| mask & (1 << bit) != 0)
                    .collect();
                problem.verify_witness(&witness).ok()
            })
            .min()
    }

    #[test]
    fn seeded_random_pairs_agree_with_exhaustive_minimum() {
        let mut rng = SmallRng::seed_from_u64(0xB0A0_DED0_5EED_0026);
        for _case in 0..96 {
            let width = rng.random_range(1..=9);
            let h = ParityCheckMatrix::from_dense(
                (0..rng.random_range(1..=width))
                    .map(|_| (0..width).map(|_| u8::from(rng.random_bool(0.5))).collect())
                    .collect(),
            )
            .unwrap();
            let l = ParityCheckMatrix::from_dense(
                (0..rng.random_range(1..=3))
                    .map(|_| (0..width).map(|_| u8::from(rng.random_bool(0.5))).collect())
                    .collect(),
            )
            .unwrap();
            let problem = DistanceProblem::from_css_checks(&h, &l).unwrap();
            let exhaustive = exhaustive_distance(&problem);
            let bounded = bounded_enumeration_code_distance(&h, &l, width);

            assert_eq!(
                bounded
                    .as_ref()
                    .map(BoundedEnumerationDistance::upper_bound),
                exhaustive
            );
            if let Some(result) = bounded {
                let (witness, _) = assert_certified_distance(&result, exhaustive.unwrap());
                assert_eq!(problem.verify_witness(witness), Ok(exhaustive.unwrap()));
            }
        }
    }

    fn independent_dense_rows(rng: &mut SmallRng, row_count: usize, width: usize) -> Vec<Vec<u8>> {
        let mut rows = Vec::with_capacity(row_count);
        while rows.len() < row_count {
            let candidate: Vec<_> = (0..width).map(|_| u8::from(rng.random_bool(0.5))).collect();
            let mut extended = rows.clone();
            extended.push(candidate.clone());
            if F2Matrix::from_rows(extended).row_reduce().1.len() > rows.len() {
                rows.push(candidate);
            }
        }
        rows
    }

    fn dense_css_pair(seed: u64) -> (ParityCheckMatrix, ParityCheckMatrix) {
        const NUM_QUBITS: usize = 40;
        const NUM_LOGICALS: usize = 8;
        const X_CHECKS: usize = 8;
        const Z_CHECKS: usize = NUM_QUBITS - NUM_LOGICALS - X_CHECKS;

        let mut rng = SmallRng::seed_from_u64(seed);
        let hz_rows = independent_dense_rows(&mut rng, Z_CHECKS, NUM_QUBITS);
        let hz = F2Matrix::from_rows(hz_rows.clone());
        let hz_kernel = hz.kernel();
        assert_eq!(hz_kernel.len(), X_CHECKS + NUM_LOGICALS);

        // Choosing Hx from ker(Hz) makes Hx Hz^T=0 by construction. Taking only eight of the
        // sixteen kernel rows leaves eight quantum logical qubits in this seeded [[40,8]] CSS
        // code; the complement of row(Hz) inside ker(Hx) supplies logical-Z detector rows.
        let hx = F2Matrix::from_rows(hz_kernel[..X_CHECKS].to_vec());
        assert_eq!(hx.mul(&hz.transpose()), F2Matrix::zeros(X_CHECKS, Z_CHECKS));
        let mut span = hz_rows;
        let mut span_rank = Z_CHECKS;
        let mut logical_z = Vec::with_capacity(NUM_LOGICALS);
        for candidate in hx.kernel() {
            let mut extended = span.clone();
            extended.push(candidate.clone());
            let rank = F2Matrix::from_rows(extended).row_reduce().1.len();
            if rank > span_rank {
                span.push(candidate.clone());
                logical_z.push(candidate);
                span_rank = rank;
            }
        }
        assert_eq!(logical_z.len(), NUM_LOGICALS);
        (
            ParityCheckMatrix::from_dense(hz.rows()).unwrap(),
            ParityCheckMatrix::from_dense(logical_z).unwrap(),
        )
    }

    fn time_three_methods(label: &str, h: &ParityCheckMatrix, l: &ParityCheckMatrix) {
        let started = Instant::now();
        let bounded = bounded_enumeration_code_distance(h, l, h.num_qubits()).unwrap();
        let bounded_elapsed = started.elapsed();
        let distance = bounded.upper_bound();
        assert!(bounded.is_certified());
        println!("{label}: distance={distance}, BZ-style={bounded_elapsed:?}");

        let started = Instant::now();
        let connected = connected_cluster_code_distance(h, l, distance).unwrap();
        let connected_elapsed = started.elapsed();
        assert_eq!(connected.distance, distance);
        println!("{label}: CC={connected_elapsed:?}");

        let problem = DistanceProblem::from_css_checks(h, l).unwrap();
        let started = Instant::now();
        let sat = certified_distance(&problem, distance).unwrap().unwrap();
        let sat_elapsed = started.elapsed();
        assert_eq!(sat.distance, distance);
        println!("{label}: SAT={sat_elapsed:?}");
    }

    #[test]
    #[ignore = "CPU timing probe for bounded-enumeration code distance"]
    fn three_method_timing_probe() {
        let dense = dense_css_pair(0xDE05_EC55_0026_0004);
        time_three_methods("dense seeded CSS [[40,8]] X side", &dense.0, &dense.1);

        let steane = steane_pair();
        time_three_methods("Steane sparse X side", &steane.0, &steane.1);

        let bb =
            BivariateBicycleCode::new(6, 6, &[(3, 0), (0, 1), (0, 2)], &[(0, 3), (1, 0), (2, 0)])
                .unwrap();
        time_three_methods("BB [[72,12,6]] sparse X side", bb.hx(), bb.logical_x());
    }
}
