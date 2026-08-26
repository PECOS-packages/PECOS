// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Matsumoto--Amano exact synthesis for one-qubit Clifford+T operators.
//!
//! The normal form is `(T | epsilon) (HT | SHT)* C` in operator order, where
//! operators act from right to left. Public [`Gate`] words instead use execution
//! order, so an `HT` syllable is stored as `[T, H]`, and an `SHT` syllable as
//! `[T, H, S]`.

use std::collections::VecDeque;
use std::error::Error;
use std::fmt;
use std::sync::OnceLock;

use num_bigint::BigInt;
use num_traits::{One, Zero};

use crate::{DOmega, Gate, Matrix, ZOmega};

/// The result of exact Matsumoto--Amano synthesis.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactSynthesis {
    word: Vec<Gate>,
    scalar: u8,
    t_count: usize,
}

impl ExactSynthesis {
    /// Returns the normal-form gate word in execution order.
    #[must_use]
    pub fn word(&self) -> &[Gate] {
        &self.word
    }

    /// Returns `j` for the separate global scalar `omega^j`, with `0 <= j < 8`.
    #[must_use]
    pub const fn scalar(&self) -> u8 {
        self.scalar
    }

    /// Returns the number of `T` tokens in the normal-form word.
    #[must_use]
    pub const fn t_count(&self) -> usize {
        self.t_count
    }

    /// Reconstructs the exact matrix, including the separate global scalar.
    #[must_use]
    pub fn to_matrix(&self) -> Matrix {
        Matrix::from_word(&self.word).with_global_phase(self.scalar)
    }
}

/// A structured exact-synthesis failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SynthError {
    /// The supplied matrix does not satisfy `U^dagger U = I` exactly.
    NotUnitary,
    /// An entry of the Bloch representation violated its required real form.
    InvalidBlochEntry,
    /// A nonzero denominator exponent had no valid Matsumoto--Amano residue.
    InvalidResidue { denominator_exponent: u32 },
    /// Removing the selected syllable did not lower the denominator exponent.
    DenominatorDidNotDecrease { before: u32, after: u32 },
    /// Exact reduction ended at an operator absent from the Clifford table.
    CliffordSuffixNotFound,
    /// The generated projective Clifford table did not have 24 classes.
    CliffordTableIncomplete { classes: usize },
    /// Residue reduction produced a word outside the normal-form grammar.
    InvalidNormalForm,
}

impl fmt::Display for SynthError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotUnitary => write!(formatter, "matrix is not exactly unitary over D[omega]"),
            Self::InvalidBlochEntry => {
                write!(formatter, "Bloch representation contains a non-real entry")
            }
            Self::InvalidResidue {
                denominator_exponent,
            } => write!(
                formatter,
                "no Matsumoto--Amano residue at denominator exponent {denominator_exponent}"
            ),
            Self::DenominatorDidNotDecrease { before, after } => write!(
                formatter,
                "syllable reduction changed denominator exponent from {before} to {after}"
            ),
            Self::CliffordSuffixNotFound => {
                write!(formatter, "terminal operator is not in the Clifford table")
            }
            Self::CliffordTableIncomplete { classes } => write!(
                formatter,
                "projective Clifford table has {classes} classes instead of 24"
            ),
            Self::InvalidNormalForm => {
                write!(
                    formatter,
                    "residue reduction violated the normal-form grammar"
                )
            }
        }
    }
}

impl Error for SynthError {}

/// Synthesizes an exactly unitary `D[omega]` matrix into Matsumoto--Amano form.
///
/// The returned word contains `T` but never `Tdg`; its global `omega` scalar is
/// carried separately in [`ExactSynthesis`].
///
/// # Errors
///
/// Returns [`SynthError::NotUnitary`] if `u^dagger * u != I` in the exact ring.
/// The other structured variants report violated internal algebraic invariants.
pub fn exact_synthesize(u: &Matrix) -> Result<ExactSynthesis, SynthError> {
    if !u.is_unitary() {
        return Err(SynthError::NotUnitary);
    }

    let table = clifford_table();
    if table.len() != 24 {
        return Err(SynthError::CliffordTableIncomplete {
            classes: table.len(),
        });
    }

    let mut current = u.clone();
    let mut bloch = BlochMatrix::from_unitary(&current);
    let mut denominator_exponent = bloch.least_denominator_exponent();
    let mut leftmost_syllables = Vec::new();

    while denominator_exponent > 0 {
        let syllable = bloch.syllable(denominator_exponent)?;
        if syllable == Syllable::T && !leftmost_syllables.is_empty() {
            return Err(SynthError::InvalidNormalForm);
        }

        let syllable_matrix = Matrix::from_word(syllable.execution_word());
        current = &syllable_matrix.adjoint() * &current;
        bloch = BlochMatrix::from_unitary(&current);
        let next_exponent = bloch.least_denominator_exponent();
        if next_exponent.checked_add(1) != Some(denominator_exponent) {
            return Err(SynthError::DenominatorDidNotDecrease {
                before: denominator_exponent,
                after: next_exponent,
            });
        }
        leftmost_syllables.push(syllable);
        denominator_exponent = next_exponent;
    }

    let Some((suffix, scalar)) = table.iter().find_map(|entry| {
        phase_between(&current, &entry.matrix).map(|phase| (entry.word.clone(), phase))
    }) else {
        return Err(SynthError::CliffordSuffixNotFound);
    };

    let mut word = suffix;
    for syllable in leftmost_syllables.iter().rev() {
        word.extend_from_slice(syllable.execution_word());
    }

    let t_count = word.iter().filter(|gate| **gate == Gate::T).count();
    Ok(ExactSynthesis {
        word,
        scalar,
        t_count,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Syllable {
    T,
    HT,
    Sht,
}

impl Syllable {
    const fn execution_word(self) -> &'static [Gate] {
        match self {
            Self::T => &[Gate::T],
            Self::HT => &[Gate::T, Gate::H],
            Self::Sht => &[Gate::T, Gate::H, Gate::S],
        }
    }
}

struct BlochMatrix {
    entries: [[DOmega; 3]; 3],
}

impl BlochMatrix {
    fn from_unitary(unitary: &Matrix) -> Self {
        let paulis = [
            Matrix::from_gate(Gate::X),
            Matrix::from_gate(Gate::Y),
            Matrix::from_gate(Gate::Z),
        ];
        let adjoint = unitary.adjoint();
        let half = DOmega::new(ZOmega::one(), 2);
        let mut entries = std::array::from_fn(|_| std::array::from_fn(|_| DOmega::zero()));

        // U P_j U^dagger = sum_i B_ij P_i, so orthogonality of the Paulis gives
        // B_ij = Tr(P_i U P_j U^dagger) / 2.
        for column in 0..3 {
            let unitary_times_pauli = unitary * &paulis[column];
            let image = &unitary_times_pauli * &adjoint;
            for row in 0..3 {
                let product = &paulis[row] * &image;
                let trace = &product.entries()[0][0] + &product.entries()[1][1];
                entries[row][column] = &trace * &half;
            }
        }

        Self { entries }
    }

    fn least_denominator_exponent(&self) -> u32 {
        self.entries
            .iter()
            .flatten()
            .map(DOmega::least_denominator_exponent)
            .max()
            .unwrap_or(0)
    }

    fn syllable(&self, denominator_exponent: u32) -> Result<Syllable, SynthError> {
        // Giles--Selinger, Lemma 4.10: modulo the right Clifford action (a
        // permutation of columns), these three residues determine the leftmost
        // normal-form syllable. Their operator strings act right-to-left.
        const T_RESIDUE: [[bool; 3]; 3] = [
            [true, true, false],
            [true, true, false],
            [false, false, false],
        ];
        const H_RESIDUE: [[bool; 3]; 3] = [
            [false, false, false],
            [true, true, false],
            [true, true, false],
        ];
        const S_RESIDUE: [[bool; 3]; 3] = [
            [true, true, false],
            [false, false, false],
            [true, true, false],
        ];
        let residue = self.parity(denominator_exponent)?;

        if equivalent_by_column_permutation(&residue, &T_RESIDUE) {
            Ok(Syllable::T)
        } else if equivalent_by_column_permutation(&residue, &H_RESIDUE) {
            Ok(Syllable::HT)
        } else if equivalent_by_column_permutation(&residue, &S_RESIDUE) {
            Ok(Syllable::Sht)
        } else {
            Err(SynthError::InvalidResidue {
                denominator_exponent,
            })
        }
    }

    fn parity(&self, denominator_exponent: u32) -> Result<[[bool; 3]; 3], SynthError> {
        let mut residue = [[false; 3]; 3];
        for (row, residue_row) in residue.iter_mut().enumerate() {
            for (column, bit) in residue_row.iter_mut().enumerate() {
                *bit = real_numerator_parity(&self.entries[row][column], denominator_exponent)?;
            }
        }
        Ok(residue)
    }
}

fn real_numerator_parity(value: &DOmega, exponent: u32) -> Result<bool, SynthError> {
    let entry_exponent = value.least_denominator_exponent();
    let Some(extra_exponent) = exponent.checked_sub(entry_exponent) else {
        return Err(SynthError::InvalidBlochEntry);
    };

    let mut numerator = value.numerator().clone();
    for _ in 0..extra_exponent {
        numerator = &numerator * &ZOmega::sqrt2();
    }

    let coordinates = numerator.coordinates();
    // In the basis (1, omega, omega^2, omega^3),
    // a + b sqrt(2) = a + b omega - b omega^3 = [a, b, 0, -b].
    // Giles--Selinger's parity map keeps precisely the rational coefficient a
    // modulo 2.
    if !coordinates[2].is_zero() || coordinates[3] != -&coordinates[1] {
        return Err(SynthError::InvalidBlochEntry);
    }

    Ok(!(&coordinates[0] % BigInt::from(2_u8)).is_zero())
}

fn equivalent_by_column_permutation(left: &[[bool; 3]; 3], right: &[[bool; 3]; 3]) -> bool {
    const PERMUTATIONS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];

    PERMUTATIONS.iter().any(|permutation| {
        (0..3).all(|row| (0..3).all(|column| left[row][permutation[column]] == right[row][column]))
    })
}

#[derive(Clone)]
struct CliffordEntry {
    matrix: Matrix,
    word: Vec<Gate>,
}

fn clifford_table() -> &'static [CliffordEntry] {
    static TABLE: OnceLock<Vec<CliffordEntry>> = OnceLock::new();
    TABLE.get_or_init(generate_clifford_table)
}

fn generate_clifford_table() -> Vec<CliffordEntry> {
    // This order is the restriction of I < X < Y < Z < H < S < Sdg < T < Tdg
    // to the required suffix alphabet {H, S, X, Z}.
    const GENERATORS: [Gate; 4] = [Gate::X, Gate::Z, Gate::H, Gate::S];

    let mut entries = vec![CliffordEntry {
        matrix: Matrix::identity(),
        word: Vec::new(),
    }];
    let mut queue = VecDeque::from([0]);

    // Breadth-first traversal visits words by length and then lexicographically,
    // so the first representative of a projective class is its canonical word.
    while let Some(index) = queue.pop_front() {
        let entry = entries[index].clone();
        for generator in GENERATORS {
            let mut word = entry.word.clone();
            word.push(generator);
            let matrix = &Matrix::from_gate(generator) * &entry.matrix;
            if entries
                .iter()
                .all(|known| phase_between(&matrix, &known.matrix).is_none())
            {
                entries.push(CliffordEntry { matrix, word });
                queue.push_back(entries.len() - 1);
            }
        }
    }

    entries
}

fn phase_between(left: &Matrix, right: &Matrix) -> Option<u8> {
    (0..8).find(|exponent| left == &right.with_global_phase(*exponent))
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use num_traits::{One, Zero};

    use super::{SynthError, clifford_table, exact_synthesize, phase_between};
    use crate::{DOmega, Gate, Matrix};

    #[test]
    fn clifford_table_covers_24_projective_and_192_scalar_classes() {
        let table = clifford_table();
        assert_eq!(table.len(), 24);

        let mut scalar_complete = HashSet::new();
        for (index, entry) in table.iter().enumerate() {
            assert!(
                entry
                    .word
                    .iter()
                    .all(|gate| matches!(gate, Gate::H | Gate::S | Gate::X | Gate::Z))
            );
            assert_eq!(Matrix::from_word(&entry.word), entry.matrix);
            assert!(entry.matrix.is_unitary());

            for other in &table[..index] {
                assert!(phase_between(&entry.matrix, &other.matrix).is_none());
            }
            for scalar in 0..8 {
                assert!(scalar_complete.insert(entry.matrix.with_global_phase(scalar)));
            }
        }
        assert_eq!(scalar_complete.len(), 192);

        for entry in table {
            for generator in [Gate::X, Gate::Z, Gate::H, Gate::S] {
                let product = Matrix::from_gate(generator) * entry.matrix.clone();
                assert!(
                    table
                        .iter()
                        .any(|candidate| phase_between(&product, &candidate.matrix).is_some())
                );
            }
        }
    }

    #[test]
    fn table_words_are_shortest_then_lexicographically_first() {
        let table = clifford_table();
        let maximum_length = table
            .iter()
            .map(|entry| entry.word.len())
            .max()
            .expect("the Clifford table is nonempty");
        let generators = [Gate::X, Gate::Z, Gate::H, Gate::S];
        let mut first_words = vec![None; table.len()];
        let mut words = vec![Vec::new()];

        for length in 0..=maximum_length {
            for word in &words {
                let matrix = Matrix::from_word(word);
                let class = table
                    .iter()
                    .position(|entry| phase_between(&matrix, &entry.matrix).is_some())
                    .expect("every Clifford word belongs to the table");
                if first_words[class].is_none() {
                    first_words[class] = Some(word.clone());
                }
            }

            if length < maximum_length {
                words = words
                    .into_iter()
                    .flat_map(|word| {
                        generators.map(move |generator| {
                            let mut extended = word.clone();
                            extended.push(generator);
                            extended
                        })
                    })
                    .collect();
            }
        }

        for (entry, first_word) in table.iter().zip(first_words) {
            assert_eq!(Some(entry.word.clone()), first_word);
        }
    }

    #[test]
    fn exhaustive_normal_forms_match_count_and_round_trip_oracle() {
        const MAX_T_COUNT: usize = 4;
        let mut matrices = HashMap::new();
        let mut cumulative = 0_usize;

        for t_count in 0..=MAX_T_COUNT {
            let before = matrices.len();
            enumerate_exact_t_count(t_count, |word, scalar| {
                assert_eq!(
                    word.iter().filter(|gate| **gate == Gate::T).count(),
                    t_count
                );
                assert!(!word.contains(&Gate::Tdg));

                let matrix = Matrix::from_word(&word).with_global_phase(scalar);
                assert!(matrix.is_unitary());
                assert!(
                    matrices
                        .insert(matrix.clone(), (word.clone(), scalar))
                        .is_none()
                );

                let synthesis = exact_synthesize(&matrix).expect("normal form must synthesize");
                assert_eq!(synthesis.word(), word);
                assert_eq!(synthesis.scalar(), scalar);
                assert_eq!(synthesis.t_count(), t_count);
                assert_eq!(
                    synthesis.t_count(),
                    synthesis
                        .word()
                        .iter()
                        .filter(|gate| **gate == Gate::T)
                        .count()
                );
                assert_eq!(synthesis.to_matrix(), matrix);
            });

            let level_count = matrices.len() - before;
            let expected_level = if t_count == 0 {
                192
            } else {
                192 * 3 * (1_usize << (t_count - 1))
            };
            assert_eq!(level_count, expected_level);

            cumulative += level_count;
            let expected_cumulative = 192 * (3 * (1_usize << t_count) - 2);
            eprintln!(
                "T-count {t_count}: exact={level_count}, cumulative={cumulative}, expected={expected_cumulative}"
            );
            assert_eq!(cumulative, expected_cumulative);
            assert_eq!(matrices.len(), expected_cumulative);
        }
    }

    #[test]
    fn non_unitary_input_returns_structured_error() {
        let zero = DOmega::zero();
        let non_unitary = Matrix::new([[DOmega::one(), zero.clone()], [zero.clone(), zero]]);
        assert_eq!(exact_synthesize(&non_unitary), Err(SynthError::NotUnitary));
    }

    fn enumerate_exact_t_count(t_count: usize, mut visit: impl FnMut(Vec<Gate>, u8)) {
        if t_count == 0 {
            for suffix in clifford_table() {
                for scalar in 0..8 {
                    visit(suffix.word.clone(), scalar);
                }
            }
            return;
        }

        for leading_t in [false, true] {
            let star_length = t_count - usize::from(leading_t);
            for choices in 0..(1_usize << star_length) {
                for suffix in clifford_table() {
                    for scalar in 0..8 {
                        let mut word = suffix.word.clone();
                        for syllable in (0..star_length).rev() {
                            word.extend_from_slice(&[Gate::T, Gate::H]);
                            if choices & (1 << syllable) != 0 {
                                word.push(Gate::S);
                            }
                        }
                        if leading_t {
                            word.push(Gate::T);
                        }
                        visit(word, scalar);
                    }
                }
            }
        }
    }
}
