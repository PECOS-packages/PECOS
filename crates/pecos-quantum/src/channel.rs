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

//! Sparse Pauli-basis channel and operator representations.
//!
//! This module keeps three related representations separate:
//! - [`PauliSum`] stores arbitrary complex coefficients on Pauli operators.
//! - [`PauliChannel`] stores real Pauli error probabilities.
//! - [`DiagonalPtm`] stores real diagonal Pauli-transfer-matrix entries, also
//!   called Pauli fidelities for Pauli channels.
//! - [`Ptm`] stores a dense real Pauli-transfer matrix.
//! - [`KrausOps`] stores a concrete Kraus-operator channel representation.
//! - [`ChoiMatrix`] stores a concrete Choi representation.
//! - [`SuperOp`] stores a dense column-stacked superoperator.
//! - [`ChiMatrix`] stores a process matrix in the Pauli basis.
//! - [`Stinespring`] stores a Stinespring isometry.
//!
//! Pauli-channel probabilities and diagonal PTM entries are connected by an
//! explicit Walsh-Hadamard transform. They are not the same representation:
//! probabilities are non-negative and sum to one, while diagonal PTM entries
//! may be negative.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::f64::consts::TAU;
use std::fmt;
use std::ops::{Add, Mul};

use nalgebra::{DMatrix, SVD};
use num_complex::Complex64;
use pecos_core::{
    BitmaskStorage, ChannelExpr, Clifford, Pauli, PauliBitmaskSmall, PauliString, Phase as _,
    UnitaryRep,
};
use pecos_random::{Rng, RngExt as _};

use crate::unitary_matrix::to_matrix_with_size;

const DEFAULT_TOLERANCE: f64 = 1e-12;

/// Pauli basis ordering used by channel representations.
///
/// The current PECOS channel basis uses lexicographic Pauli digits with
/// `I=0`, `X=1`, `Y=2`, `Z=3`, and qubit 0 as the least-significant base-4
/// digit. For two qubits, labels displayed from high qubit to low qubit are:
/// `II, IX, IY, IZ, XI, XX, XY, XZ, ...`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PtmBasisOrder {
    /// Lexicographic Pauli basis with qubit 0 as the fastest-varying digit.
    #[default]
    LexicographicLittleEndian,
}

/// Error returned by channel representation constructors and conversions.
#[derive(Debug, Clone, PartialEq)]
pub enum ChannelError {
    /// The requested number of qubits would overflow a `usize` dimension.
    DimensionOverflow {
        /// Number of qubits supplied by the caller.
        num_qubits: usize,
    },
    /// A value was not a valid Pauli basis digit.
    InvalidBasisDigit {
        /// Invalid digit.
        digit: usize,
    },
    /// A Pauli-basis index is outside the basis for the requested qubit count.
    BasisIndexOutOfRange {
        /// Number of qubits supplied by the caller.
        num_qubits: usize,
        /// Basis length for that qubit count.
        basis_len: usize,
        /// Invalid index.
        index: usize,
    },
    /// A Pauli term acts outside the declared qubit range.
    QubitOutOfRange {
        /// Number of qubits supplied by the caller.
        num_qubits: usize,
        /// Highest qubit touched by the offending Pauli term.
        qubit: usize,
    },
    /// Two channel objects act on different numbers of qubits.
    QubitCountMismatch {
        /// Expected qubit count.
        expected: usize,
        /// Actual qubit count.
        actual: usize,
    },
    /// A coefficient or fidelity is not finite.
    InvalidCoefficient {
        /// Offending coefficient.
        value: Complex64,
    },
    /// A `PauliSum` coefficient was not real enough for a probability.
    NonRealCoefficient {
        /// Offending coefficient.
        value: Complex64,
        /// Allowed imaginary-part tolerance.
        tolerance: f64,
    },
    /// A probability value is invalid for a Pauli channel.
    InvalidProbability {
        /// Probability value.
        value: f64,
        /// Allowed negative tolerance.
        tolerance: f64,
    },
    /// A probability map does not sum to one within tolerance.
    ProbabilitySum {
        /// Observed probability sum.
        sum: f64,
        /// Allowed absolute tolerance.
        tolerance: f64,
    },
    /// A dense matrix does not match the expected channel or density-matrix
    /// shape.
    InvalidMatrixShape {
        /// Expected row count.
        expected_rows: usize,
        /// Expected column count.
        expected_cols: usize,
        /// Actual row count.
        rows: usize,
        /// Actual column count.
        cols: usize,
    },
    /// A Kraus channel was constructed without any Kraus operators.
    EmptyKrausSet,
    /// A numerical matrix decomposition failed.
    DecompositionFailed {
        /// Human-readable reason.
        reason: String,
    },
    /// A channel expression is outside the supported conversion subset.
    UnsupportedChannelExpr {
        /// Human-readable reason.
        reason: String,
    },
    /// A repeated subsystem was supplied.
    DuplicateSubsystem {
        /// Repeated qubit/subsystem index.
        qubit: usize,
    },
    /// A tomography reconstruction had the wrong number of basis samples.
    InvalidTomographySampleCount {
        /// Expected number of operator-basis outputs.
        expected: usize,
        /// Actual number of supplied outputs.
        actual: usize,
    },
    /// A tomography input index is outside the experiment design.
    TomographyInputOutOfRange {
        /// Number of tomography inputs in the design.
        num_inputs: usize,
        /// Invalid input index.
        index: usize,
    },
    /// A computational matrix-unit row/column is outside the Hilbert space.
    MatrixUnitOutOfRange {
        /// Hilbert-space dimension.
        dim: usize,
        /// Invalid row.
        row: usize,
        /// Invalid column.
        col: usize,
    },
}

impl fmt::Display for ChannelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionOverflow { num_qubits } => write!(
                f,
                "Pauli-basis dimension overflows usize for {num_qubits} qubits"
            ),
            Self::InvalidBasisDigit { digit } => write!(f, "invalid Pauli basis digit: {digit}"),
            Self::BasisIndexOutOfRange {
                num_qubits,
                basis_len,
                index,
            } => write!(
                f,
                "Pauli-basis index {index} is outside the {basis_len}-element basis for {num_qubits} qubits"
            ),
            Self::QubitOutOfRange { num_qubits, qubit } => write!(
                f,
                "Pauli term touches qubit {qubit}, outside declared {num_qubits}-qubit range"
            ),
            Self::QubitCountMismatch { expected, actual } => write!(
                f,
                "channel qubit count mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidCoefficient { value } => {
                write!(f, "invalid non-finite coefficient: {value}")
            }
            Self::NonRealCoefficient { value, tolerance } => write!(
                f,
                "coefficient {value} is not real within tolerance {tolerance}"
            ),
            Self::InvalidProbability { value, tolerance } => write!(
                f,
                "invalid Pauli-channel probability {value}; tolerance is {tolerance}"
            ),
            Self::ProbabilitySum { sum, tolerance } => write!(
                f,
                "Pauli-channel probabilities must sum to 1 within tolerance {tolerance}, got {sum}"
            ),
            Self::InvalidMatrixShape {
                expected_rows,
                expected_cols,
                rows,
                cols,
            } => write!(
                f,
                "invalid matrix shape {rows}x{cols}; expected {expected_rows}x{expected_cols}"
            ),
            Self::EmptyKrausSet => write!(f, "Kraus channel must contain at least one operator"),
            Self::DecompositionFailed { reason } => {
                write!(f, "matrix decomposition failed: {reason}")
            }
            Self::UnsupportedChannelExpr { reason } => {
                write!(f, "unsupported channel expression: {reason}")
            }
            Self::DuplicateSubsystem { qubit } => {
                write!(f, "duplicate subsystem/qubit index: {qubit}")
            }
            Self::InvalidTomographySampleCount { expected, actual } => write!(
                f,
                "invalid tomography sample count {actual}; expected {expected} operator-basis outputs"
            ),
            Self::TomographyInputOutOfRange { num_inputs, index } => write!(
                f,
                "tomography input index {index} is outside the {num_inputs}-input design"
            ),
            Self::MatrixUnitOutOfRange { dim, row, col } => write!(
                f,
                "matrix unit |{row}><{col}| is outside the {dim}-dimensional Hilbert space"
            ),
        }
    }
}

impl Error for ChannelError {}

/// Returns the number of Pauli basis elements for `num_qubits`.
///
/// This is `4^num_qubits`.
///
/// # Errors
///
/// Returns [`ChannelError::DimensionOverflow`] when `4^num_qubits` does not
/// fit in `usize`.
pub fn pauli_basis_len(num_qubits: usize) -> Result<usize, ChannelError> {
    4usize
        .checked_pow(
            num_qubits
                .try_into()
                .map_err(|_| ChannelError::DimensionOverflow { num_qubits })?,
        )
        .ok_or(ChannelError::DimensionOverflow { num_qubits })
}

/// Maps a Pauli value to the channel-basis digit `I=0, X=1, Y=2, Z=3`.
///
/// This intentionally differs from the internal [`Pauli`] discriminant order,
/// where `Z` and `Y` are stored in bitmask-friendly order.
#[must_use]
pub fn pauli_to_basis_digit(pauli: Pauli) -> usize {
    match pauli {
        Pauli::I => 0,
        Pauli::X => 1,
        Pauli::Y => 2,
        Pauli::Z => 3,
    }
}

/// Converts a channel-basis digit to a Pauli value.
///
/// # Errors
///
/// Returns [`ChannelError::InvalidBasisDigit`] when `digit` is not in `0..4`.
pub fn basis_digit_to_pauli(digit: usize) -> Result<Pauli, ChannelError> {
    match digit {
        0 => Ok(Pauli::I),
        1 => Ok(Pauli::X),
        2 => Ok(Pauli::Y),
        3 => Ok(Pauli::Z),
        _ => Err(ChannelError::InvalidBasisDigit { digit }),
    }
}

/// Returns the Pauli basis element at `index`.
///
/// The returned vector is indexed by qubit number: element 0 is the Pauli on
/// qubit 0. Qubit 0 is the fastest-varying base-4 digit.
///
/// # Errors
///
/// Returns an error when `4^num_qubits` overflows or `index` is outside the
/// basis.
pub fn basis_element(num_qubits: usize, index: usize) -> Result<Vec<Pauli>, ChannelError> {
    let basis_len = pauli_basis_len(num_qubits)?;
    if index >= basis_len {
        return Err(ChannelError::BasisIndexOutOfRange {
            num_qubits,
            basis_len,
            index,
        });
    }

    let mut remaining = index;
    let mut paulis = Vec::with_capacity(num_qubits);
    for _ in 0..num_qubits {
        paulis.push(basis_digit_to_pauli(remaining & 0b11)?);
        remaining >>= 2;
    }
    Ok(paulis)
}

/// Returns a display label for a Pauli basis element.
///
/// Labels are printed with the highest-numbered qubit first, matching common
/// ket-label display style. For example, basis index 1 on two qubits is `IX`
/// because it is identity on qubit 1 and X on qubit 0.
///
/// # Errors
///
/// Returns an error when `4^num_qubits` overflows or `index` is outside the
/// basis.
pub fn basis_label(num_qubits: usize, index: usize) -> Result<String, ChannelError> {
    let paulis = basis_element(num_qubits, index)?;
    Ok(paulis
        .iter()
        .rev()
        .map(|pauli| match pauli {
            Pauli::I => 'I',
            Pauli::X => 'X',
            Pauli::Y => 'Y',
            Pauli::Z => 'Z',
        })
        .collect())
}

/// Returns the Pauli bitmask basis element at `index`.
///
/// # Errors
///
/// Returns an error when `4^num_qubits` overflows or `index` is outside the
/// basis.
pub fn basis_bitmask(num_qubits: usize, index: usize) -> Result<PauliBitmaskSmall, ChannelError> {
    let paulis = basis_element(num_qubits, index)?;
    Ok(bitmask_from_paulis(&paulis))
}

/// Returns the Pauli-basis index for a bitmask in the canonical ordering.
///
/// # Errors
///
/// Returns [`ChannelError::QubitOutOfRange`] when the bitmask touches a qubit
/// outside `0..num_qubits`.
pub fn basis_index(num_qubits: usize, pauli: &PauliBitmaskSmall) -> Result<usize, ChannelError> {
    pauli_basis_len(num_qubits)?;
    validate_num_qubits(num_qubits, pauli)?;
    let mut index = 0usize;
    for qubit in 0..num_qubits {
        let digit = match (pauli.has_x(qubit), pauli.has_z(qubit)) {
            (false, false) => 0,
            (true, false) => 1,
            (true, true) => 2,
            (false, true) => 3,
        };
        index += digit << (2 * qubit);
    }
    Ok(index)
}

/// Returns the canonical display label for a bitmask.
///
/// # Errors
///
/// Returns [`ChannelError::QubitOutOfRange`] when the bitmask touches a qubit
/// outside `0..num_qubits`.
pub fn bitmask_label(num_qubits: usize, pauli: &PauliBitmaskSmall) -> Result<String, ChannelError> {
    basis_label(num_qubits, basis_index(num_qubits, pauli)?)
}

/// Converts a [`PauliString`] into the phase-free bitmask used by channel
/// representations.
///
/// # Errors
///
/// Returns [`ChannelError::QubitOutOfRange`] when the Pauli string touches a
/// qubit outside `0..num_qubits`.
pub fn pauli_string_to_bitmask(
    num_qubits: usize,
    pauli: &PauliString,
) -> Result<PauliBitmaskSmall, ChannelError> {
    let mut out = PauliBitmaskSmall::identity();
    for (p, q) in pauli.iter_pairs() {
        let q = q.index();
        if q >= num_qubits {
            return Err(ChannelError::QubitOutOfRange {
                num_qubits,
                qubit: q,
            });
        }
        match p {
            Pauli::I => {}
            Pauli::X => out.x_bits.set_bit(q),
            Pauli::Y => {
                out.x_bits.set_bit(q);
                out.z_bits.set_bit(q);
            }
            Pauli::Z => out.z_bits.set_bit(q),
        }
    }
    Ok(out)
}

/// Sparse complex sum of Pauli operators.
#[derive(Clone, Debug, PartialEq)]
pub struct PauliSum {
    num_qubits: usize,
    terms: BTreeMap<PauliBitmaskSmall, Complex64>,
}

impl PauliSum {
    /// Constructs an empty Pauli sum over `num_qubits`.
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            terms: BTreeMap::new(),
        }
    }

    /// Constructs a Pauli sum after validating term qubit ranges and
    /// simplifying near-zero coefficients.
    ///
    /// # Errors
    ///
    /// Returns an error when any term touches a qubit outside `0..num_qubits`
    /// or any coefficient is not finite.
    pub fn try_new(
        num_qubits: usize,
        terms: BTreeMap<PauliBitmaskSmall, Complex64>,
    ) -> Result<Self, ChannelError> {
        Self::try_new_with_tolerance(num_qubits, terms, DEFAULT_TOLERANCE)
    }

    /// Constructs a Pauli sum with an explicit zero-dropping tolerance.
    ///
    /// # Errors
    ///
    /// Returns an error when any term touches a qubit outside `0..num_qubits`
    /// or any coefficient is not finite.
    pub fn try_new_with_tolerance(
        num_qubits: usize,
        terms: BTreeMap<PauliBitmaskSmall, Complex64>,
        tolerance: f64,
    ) -> Result<Self, ChannelError> {
        let mut out = Self::new(num_qubits);
        for (pauli, coefficient) in terms {
            out.add_term_with_tolerance(pauli, coefficient, tolerance)?;
        }
        Ok(out)
    }

    /// Constructs a Pauli sum containing one [`PauliString`].
    ///
    /// The `PauliString` phase becomes the complex coefficient. The stored
    /// Pauli label itself is phase-free.
    ///
    /// # Errors
    ///
    /// Returns an error when `pauli` touches a qubit outside `0..num_qubits`.
    pub fn from_pauli_string(num_qubits: usize, pauli: &PauliString) -> Result<Self, ChannelError> {
        let label = pauli_string_to_bitmask(num_qubits, pauli)?;
        let coefficient = pauli.phase().to_complex();
        let mut terms = BTreeMap::new();
        terms.insert(label, coefficient);
        Self::try_new(num_qubits, terms)
    }

    /// Returns the number of qubits represented by this sum.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Returns the sparse Pauli terms and complex coefficients.
    #[must_use]
    pub fn terms(&self) -> &BTreeMap<PauliBitmaskSmall, Complex64> {
        &self.terms
    }

    /// Adds one term, merging with an existing coefficient if needed.
    ///
    /// # Errors
    ///
    /// Returns an error when `pauli` touches a qubit outside
    /// `0..self.num_qubits` or `coefficient` is not finite.
    pub fn add_term(
        &mut self,
        pauli: PauliBitmaskSmall,
        coefficient: Complex64,
    ) -> Result<(), ChannelError> {
        self.add_term_with_tolerance(pauli, coefficient, DEFAULT_TOLERANCE)
    }

    /// Adds one term with an explicit zero-dropping tolerance.
    ///
    /// # Errors
    ///
    /// Returns an error when `pauli` touches a qubit outside
    /// `0..self.num_qubits` or `coefficient` is not finite.
    pub fn add_term_with_tolerance(
        &mut self,
        pauli: PauliBitmaskSmall,
        coefficient: Complex64,
        tolerance: f64,
    ) -> Result<(), ChannelError> {
        validate_num_qubits(self.num_qubits, &pauli)?;
        validate_complex(coefficient)?;
        if coefficient.norm() <= tolerance {
            return Ok(());
        }

        match self.terms.entry(pauli) {
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                *entry.get_mut() += coefficient;
                if entry.get().norm() <= tolerance {
                    entry.remove();
                }
            }
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(coefficient);
            }
        }
        Ok(())
    }

    /// Drops near-zero coefficients in-place.
    pub fn simplify_with_tolerance(&mut self, tolerance: f64) {
        self.terms
            .retain(|_, coefficient| coefficient.norm() > tolerance);
    }

    /// Drops coefficients at the default tolerance and returns the simplified
    /// sum.
    #[must_use]
    pub fn simplify(mut self) -> Self {
        self.simplify_with_tolerance(DEFAULT_TOLERANCE);
        self
    }

    /// Greedily partitions terms into mutually commuting sums.
    ///
    /// Coefficients are preserved exactly. The grouping is a graph-coloring
    /// heuristic on the anticommutation graph, so it is not guaranteed to use
    /// the minimum possible number of groups.
    #[must_use]
    pub fn group_commuting(&self) -> Vec<Self> {
        let mut groups: Vec<BTreeMap<PauliBitmaskSmall, Complex64>> = Vec::new();

        'next_term: for (pauli, coefficient) in &self.terms {
            for group in &mut groups {
                if group.keys().all(|other| pauli.commutes_with(other)) {
                    group.insert(pauli.clone(), *coefficient);
                    continue 'next_term;
                }
            }
            groups.push(BTreeMap::from([(pauli.clone(), *coefficient)]));
        }

        groups
            .into_iter()
            .map(|terms| Self {
                num_qubits: self.num_qubits,
                terms,
            })
            .collect()
    }

    /// Returns the Pauli conjugation `P * self * P†`.
    ///
    /// Pauli conjugation preserves each Pauli label and flips the coefficient
    /// sign for terms that anticommute with `P`.
    ///
    /// # Errors
    ///
    /// Returns an error when `pauli` touches a qubit outside this sum's qubit
    /// range.
    pub fn conjugated_by_pauli_string(&self, pauli: &PauliString) -> Result<Self, ChannelError> {
        let label = pauli_string_to_bitmask(self.num_qubits, pauli)?;
        let mut terms = BTreeMap::new();
        for (term, coefficient) in &self.terms {
            let sign = if label.commutes_with(term) { 1.0 } else { -1.0 };
            terms.insert(term.clone(), *coefficient * sign);
        }
        Ok(Self {
            num_qubits: self.num_qubits,
            terms,
        })
    }

    /// Adds two Pauli sums after validating that they act on the same number
    /// of qubits.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::QubitCountMismatch`] when the two sums have
    /// different qubit counts.
    pub fn try_add(mut self, rhs: Self) -> Result<Self, ChannelError> {
        if self.num_qubits != rhs.num_qubits {
            return Err(ChannelError::QubitCountMismatch {
                expected: self.num_qubits,
                actual: rhs.num_qubits,
            });
        }
        for (pauli, coefficient) in rhs.terms {
            self.add_term(pauli, coefficient)?;
        }
        Ok(self)
    }

    /// Returns the trace of the represented operator.
    ///
    /// The trace is `identity_coefficient * 2^num_qubits`.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::DimensionOverflow`] when `2^num_qubits` cannot
    /// fit in `usize`.
    #[allow(clippy::cast_precision_loss)]
    pub fn trace(&self) -> Result<Complex64, ChannelError> {
        let dim = hilbert_dim(self.num_qubits)?;
        Ok(self
            .terms
            .get(&PauliBitmaskSmall::identity())
            .copied()
            .unwrap_or_else(|| Complex64::new(0.0, 0.0))
            * dim as f64)
    }
}

impl fmt::Display for PauliSum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }
        for (idx, (pauli, coefficient)) in self.terms.iter().enumerate() {
            if idx > 0 {
                write!(f, " + ")?;
            }
            let label = bitmask_label(self.num_qubits, pauli).map_err(|_| fmt::Error)?;
            write!(f, "({coefficient}){label}")?;
        }
        Ok(())
    }
}

impl Add for PauliSum {
    type Output = Self;

    /// Adds two Pauli sums.
    ///
    /// # Panics
    ///
    /// Panics when the sums have different qubit counts. Use
    /// [`PauliSum::try_add`] to handle this case without panicking.
    fn add(self, rhs: Self) -> Self::Output {
        self.try_add(rhs)
            .expect("cannot add PauliSum values with different qubit counts")
    }
}

impl Mul<Complex64> for PauliSum {
    type Output = Self;

    fn mul(mut self, rhs: Complex64) -> Self::Output {
        for coefficient in self.terms.values_mut() {
            *coefficient *= rhs;
        }
        self.simplify()
    }
}

impl Mul<PauliSum> for Complex64 {
    type Output = PauliSum;

    fn mul(self, rhs: PauliSum) -> Self::Output {
        rhs * self
    }
}

impl Mul<f64> for PauliSum {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        self * Complex64::new(rhs, 0.0)
    }
}

impl Mul<PauliSum> for f64 {
    type Output = PauliSum;

    fn mul(self, rhs: PauliSum) -> Self::Output {
        rhs * self
    }
}

/// Sparse Pauli error channel represented by probabilities.
#[derive(Clone, Debug, PartialEq)]
pub struct PauliChannel {
    num_qubits: usize,
    basis_order: PtmBasisOrder,
    probabilities: BTreeMap<PauliBitmaskSmall, f64>,
}

impl PauliChannel {
    /// Constructs a Pauli channel after validating probabilities.
    ///
    /// Missing Pauli terms are treated as zero probability. Stored
    /// probabilities must be finite, non-negative, and sum to one.
    ///
    /// # Errors
    ///
    /// Returns an error when a term is outside the declared qubit range, a
    /// probability is non-finite or negative, or probabilities do not sum to
    /// one.
    pub fn try_new(
        num_qubits: usize,
        probabilities: BTreeMap<PauliBitmaskSmall, f64>,
    ) -> Result<Self, ChannelError> {
        Self::try_new_with_tolerance(num_qubits, probabilities, DEFAULT_TOLERANCE)
    }

    /// Constructs a Pauli channel with an explicit tolerance.
    ///
    /// # Errors
    ///
    /// Returns an error when a term is outside the declared qubit range, a
    /// probability is non-finite or negative, or probabilities do not sum to
    /// one within `tolerance`.
    pub fn try_new_with_tolerance(
        num_qubits: usize,
        probabilities: BTreeMap<PauliBitmaskSmall, f64>,
        tolerance: f64,
    ) -> Result<Self, ChannelError> {
        let mut cleaned = BTreeMap::new();
        let mut sum = 0.0;
        for (pauli, probability) in probabilities {
            validate_num_qubits(num_qubits, &pauli)?;
            validate_probability(probability, tolerance)?;
            let probability = if probability.abs() <= tolerance {
                0.0
            } else {
                probability
            };
            if probability > 0.0 {
                cleaned.insert(pauli, probability);
            }
            sum += probability;
        }
        if (sum - 1.0).abs() > tolerance {
            return Err(ChannelError::ProbabilitySum { sum, tolerance });
        }
        Ok(Self {
            num_qubits,
            basis_order: PtmBasisOrder::default(),
            probabilities: cleaned,
        })
    }

    /// Constructs a one-qubit Pauli channel from non-identity probabilities.
    ///
    /// # Errors
    ///
    /// Returns an error when probabilities are invalid or do not leave a valid
    /// identity probability.
    pub fn one_qubit(px: f64, py: f64, pz: f64) -> Result<Self, ChannelError> {
        let mut probabilities = BTreeMap::new();
        probabilities.insert(PauliBitmaskSmall::identity(), 1.0 - px - py - pz);
        probabilities.insert(PauliBitmaskSmall::x(0), px);
        probabilities.insert(PauliBitmaskSmall::y(0), py);
        probabilities.insert(PauliBitmaskSmall::z(0), pz);
        Self::try_new(1, probabilities)
    }

    /// Converts a real, non-negative [`PauliSum`] into a Pauli channel.
    ///
    /// # Errors
    ///
    /// Returns an error when any coefficient has a non-negligible imaginary
    /// part, is negative, or the real coefficients do not sum to one.
    pub fn from_pauli_sum(sum: &PauliSum) -> Result<Self, ChannelError> {
        let mut probabilities = BTreeMap::new();
        for (pauli, coefficient) in sum.terms() {
            if coefficient.im.abs() > DEFAULT_TOLERANCE {
                return Err(ChannelError::NonRealCoefficient {
                    value: *coefficient,
                    tolerance: DEFAULT_TOLERANCE,
                });
            }
            probabilities.insert(pauli.clone(), coefficient.re);
        }
        Self::try_new(sum.num_qubits(), probabilities)
    }

    /// Constructs a Pauli channel from probabilities keyed by [`PauliString`].
    ///
    /// Pauli-string phases are ignored because Pauli channels apply
    /// `P rho P†`, where global phase cancels. Repeated Pauli keys are
    /// accumulated before validation.
    ///
    /// # Errors
    ///
    /// Returns an error when any Pauli string touches a qubit outside
    /// `0..num_qubits`, a probability is invalid, or probabilities do not sum
    /// to one.
    pub fn from_pauli_strings<I>(num_qubits: usize, probabilities: I) -> Result<Self, ChannelError>
    where
        I: IntoIterator<Item = (PauliString, f64)>,
    {
        let mut terms = BTreeMap::new();
        for (pauli, probability) in probabilities {
            let pauli = pauli_string_to_bitmask(num_qubits, &pauli)?;
            *terms.entry(pauli).or_insert(0.0) += probability;
        }
        Self::try_new(num_qubits, terms)
    }

    /// Converts a symbolic channel expression into a Pauli channel when it is
    /// a mixture of Pauli unitaries.
    ///
    /// This accepts the common constructors for bit-flip, dephasing,
    /// depolarizing, and general mixed-Pauli channels. Non-Pauli unitary
    /// mixtures are intentionally rejected instead of silently projecting them.
    ///
    /// # Errors
    ///
    /// Returns an error when `channel` is not a supported Pauli-unitary
    /// mixture or when its probabilities are invalid.
    pub fn from_channel_expr(channel: &ChannelExpr) -> Result<Self, ChannelError> {
        let num_qubits = channel_num_qubits(channel).max(1);
        match channel {
            ChannelExpr::Unitary(unitary) => {
                let pauli = unitary_rep_to_pauli_bitmask(num_qubits, unitary)?;
                let mut probabilities = BTreeMap::new();
                probabilities.insert(pauli, 1.0);
                Self::try_new(num_qubits, probabilities)
            }
            ChannelExpr::MixedUnitary(ops) => {
                let mut probabilities = BTreeMap::new();
                for (probability, unitary) in ops {
                    validate_probability(*probability, DEFAULT_TOLERANCE)?;
                    let pauli = unitary_rep_to_pauli_bitmask(num_qubits, unitary)?;
                    *probabilities.entry(pauli).or_insert(0.0) += *probability;
                }
                Self::try_new(num_qubits, probabilities)
            }
            _ => Err(ChannelError::UnsupportedChannelExpr {
                reason: "only Pauli-unitary channels can be converted to PauliChannel".to_string(),
            }),
        }
    }

    /// Converts Pauli probabilities to diagonal PTM entries.
    ///
    /// # Errors
    ///
    /// Returns an error if the Pauli-basis dimension overflows.
    pub fn to_diagonal_ptm(&self) -> Result<DiagonalPtm, ChannelError> {
        let basis_len = pauli_basis_len(self.num_qubits)?;
        let mut fidelities = BTreeMap::new();
        for basis_index in 0..basis_len {
            let basis = basis_bitmask(self.num_qubits, basis_index)?;
            let fidelity = self
                .probabilities
                .iter()
                .map(|(error, probability)| probability * commutation_character(error, &basis))
                .sum();
            fidelities.insert(basis, fidelity);
        }
        DiagonalPtm::try_new(self.num_qubits, fidelities)
    }

    /// Converts this Pauli channel to a dense PTM.
    ///
    /// # Errors
    ///
    /// Returns an error if the Pauli-basis dimension overflows.
    pub fn to_ptm(&self) -> Result<Ptm, ChannelError> {
        self.to_diagonal_ptm()?.to_ptm()
    }

    /// Returns the number of qubits represented by this channel.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Returns the PTM basis ordering.
    #[must_use]
    pub fn basis_order(&self) -> PtmBasisOrder {
        self.basis_order
    }

    /// Returns the sparse probability map.
    #[must_use]
    pub fn probabilities(&self) -> &BTreeMap<PauliBitmaskSmall, f64> {
        &self.probabilities
    }

    /// Returns the probability of a specific Pauli error.
    #[must_use]
    pub fn probability(&self, pauli: &PauliBitmaskSmall) -> f64 {
        self.probabilities.get(pauli).copied().unwrap_or(0.0)
    }

    /// Returns the total non-identity probability.
    #[must_use]
    pub fn total_error_rate(&self) -> f64 {
        self.probabilities
            .iter()
            .filter(|(pauli, _)| !pauli.is_identity())
            .map(|(_, probability)| probability)
            .sum()
    }
}

/// Sparse diagonal Pauli transfer matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct DiagonalPtm {
    num_qubits: usize,
    basis_order: PtmBasisOrder,
    fidelities: BTreeMap<PauliBitmaskSmall, f64>,
}

impl DiagonalPtm {
    /// Constructs a diagonal PTM after validating term qubit ranges.
    ///
    /// Missing Pauli terms are treated as zero fidelity.
    ///
    /// # Errors
    ///
    /// Returns an error when any term is outside the declared qubit range or
    /// any fidelity is non-finite.
    pub fn try_new(
        num_qubits: usize,
        fidelities: BTreeMap<PauliBitmaskSmall, f64>,
    ) -> Result<Self, ChannelError> {
        let mut cleaned = BTreeMap::new();
        for (pauli, fidelity) in fidelities {
            validate_num_qubits(num_qubits, &pauli)?;
            validate_real(fidelity)?;
            if fidelity.abs() > DEFAULT_TOLERANCE {
                cleaned.insert(pauli, fidelity);
            }
        }
        Ok(Self {
            num_qubits,
            basis_order: PtmBasisOrder::default(),
            fidelities: cleaned,
        })
    }

    /// Converts diagonal PTM entries to Pauli-channel probabilities.
    ///
    /// # Errors
    ///
    /// Returns an error if the Pauli-basis dimension overflows or the inverse
    /// Walsh-Hadamard transform does not produce valid probabilities.
    pub fn to_pauli_channel(&self) -> Result<PauliChannel, ChannelError> {
        let basis_len = pauli_basis_len(self.num_qubits)?;
        #[allow(clippy::cast_precision_loss)]
        let scale = basis_len as f64;
        let basis: Vec<PauliBitmaskSmall> = (0..basis_len)
            .map(|basis_index| basis_bitmask(self.num_qubits, basis_index))
            .collect::<Result<_, _>>()?;
        let mut probabilities = BTreeMap::new();
        for error in &basis {
            let probability: f64 = basis
                .iter()
                .map(|basis_element| {
                    self.fidelity(basis_element) * commutation_character(error, basis_element)
                })
                .sum::<f64>()
                / scale;
            probabilities.insert(error.clone(), probability);
        }
        PauliChannel::try_new(self.num_qubits, probabilities)
    }

    /// Converts a symbolic Pauli-unitary channel expression to diagonal PTM
    /// entries.
    ///
    /// # Errors
    ///
    /// Returns an error when the expression is not a supported Pauli-unitary
    /// mixture or when probabilities are invalid.
    pub fn from_channel_expr(channel: &ChannelExpr) -> Result<Self, ChannelError> {
        PauliChannel::from_channel_expr(channel)?.to_diagonal_ptm()
    }

    /// Expands this diagonal PTM into a dense PTM matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if the Pauli-basis dimension overflows.
    pub fn to_ptm(&self) -> Result<Ptm, ChannelError> {
        let basis_len = pauli_basis_len(self.num_qubits)?;
        let mut matrix = DMatrix::zeros(basis_len, basis_len);
        for basis_idx in 0..basis_len {
            let basis = basis_bitmask(self.num_qubits, basis_idx)?;
            matrix[(basis_idx, basis_idx)] = self.fidelity(&basis);
        }
        Ptm::try_new(self.num_qubits, matrix)
    }

    /// Returns the number of qubits represented by this diagonal PTM.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Returns the PTM basis ordering.
    #[must_use]
    pub fn basis_order(&self) -> PtmBasisOrder {
        self.basis_order
    }

    /// Returns the sparse fidelity map.
    #[must_use]
    pub fn fidelities(&self) -> &BTreeMap<PauliBitmaskSmall, f64> {
        &self.fidelities
    }

    /// Returns the fidelity for a specific Pauli basis element.
    #[must_use]
    pub fn fidelity(&self, pauli: &PauliBitmaskSmall) -> f64 {
        self.fidelities.get(pauli).copied().unwrap_or(0.0)
    }
}

/// Dense Pauli transfer matrix in the canonical PECOS Pauli basis.
///
/// PTM entries use the normalized convention
/// `R_ij = (1/d) Tr[P_i E(P_j)]`, where `d = 2^num_qubits`.
/// Rows index output Paulis, columns index input Paulis.
#[derive(Clone, Debug, PartialEq)]
pub struct Ptm {
    num_qubits: usize,
    basis_order: PtmBasisOrder,
    matrix: DMatrix<f64>,
}

impl Ptm {
    /// Constructs a dense PTM after validating the structural shape.
    ///
    /// # Errors
    ///
    /// Returns an error when `matrix` is not `4^num_qubits x 4^num_qubits`
    /// or contains non-finite entries.
    pub fn try_new(num_qubits: usize, matrix: DMatrix<f64>) -> Result<Self, ChannelError> {
        let basis_len = pauli_basis_len(num_qubits)?;
        if matrix.nrows() != basis_len || matrix.ncols() != basis_len {
            return Err(ChannelError::InvalidMatrixShape {
                expected_rows: basis_len,
                expected_cols: basis_len,
                rows: matrix.nrows(),
                cols: matrix.ncols(),
            });
        }
        for value in matrix.iter() {
            validate_real(*value)?;
        }
        Ok(Self {
            num_qubits,
            basis_order: PtmBasisOrder::default(),
            matrix,
        })
    }

    /// Constructs the identity channel PTM over `num_qubits`.
    ///
    /// # Errors
    ///
    /// Returns an error if the Pauli-basis dimension overflows.
    pub fn identity(num_qubits: usize) -> Result<Self, ChannelError> {
        let basis_len = pauli_basis_len(num_qubits)?;
        Self::try_new(num_qubits, DMatrix::identity(basis_len, basis_len))
    }

    /// Constructs a dense PTM from a diagonal PTM.
    ///
    /// # Errors
    ///
    /// Returns an error if the Pauli-basis dimension overflows.
    pub fn from_diagonal_ptm(diagonal: &DiagonalPtm) -> Result<Self, ChannelError> {
        diagonal.to_ptm()
    }

    /// Constructs a dense PTM from a Pauli channel.
    ///
    /// # Errors
    ///
    /// Returns an error if the Pauli-basis dimension overflows.
    pub fn from_pauli_channel(channel: &PauliChannel) -> Result<Self, ChannelError> {
        channel.to_ptm()
    }

    /// Constructs the PTM for a unitary conjugation channel.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions overflow or numerical entries have
    /// significant imaginary components.
    pub fn from_unitary(unitary: &UnitaryRep, num_qubits: usize) -> Result<Self, ChannelError> {
        let basis_len = pauli_basis_len(num_qubits)?;
        let dim = hilbert_dim(num_qubits)?;
        #[allow(clippy::cast_precision_loss)]
        let dim_f = dim as f64;
        let unitary_matrix = to_matrix_with_size(unitary, num_qubits).into_inner();
        if unitary_matrix.nrows() != dim || unitary_matrix.ncols() != dim {
            return Err(ChannelError::InvalidMatrixShape {
                expected_rows: dim,
                expected_cols: dim,
                rows: unitary_matrix.nrows(),
                cols: unitary_matrix.ncols(),
            });
        }
        let unitary_adjoint = unitary_matrix.adjoint();
        let basis_matrices = pauli_basis_matrices(num_qubits)?;
        let mut matrix = DMatrix::zeros(basis_len, basis_len);
        for input_idx in 0..basis_len {
            let evolved = &unitary_matrix * &basis_matrices[input_idx] * &unitary_adjoint;
            for output_idx in 0..basis_len {
                let entry = trace_complex(&(&basis_matrices[output_idx] * &evolved)) / dim_f;
                if entry.im.abs() > DEFAULT_TOLERANCE {
                    return Err(ChannelError::NonRealCoefficient {
                        value: entry,
                        tolerance: DEFAULT_TOLERANCE,
                    });
                }
                matrix[(output_idx, input_idx)] = entry.re;
            }
        }
        Self::try_new(num_qubits, matrix)
    }

    /// Converts a symbolic channel expression to a dense PTM when supported.
    ///
    /// This supports unitary, mixed-unitary, amplitude-damping, phase-damping,
    /// tensor, and composed channel expressions that can be represented by
    /// [`KrausOps`]. Erasure/leakage channels are intentionally rejected until
    /// PECOS has an explicit flag or extended-Hilbert-space representation.
    ///
    /// # Errors
    ///
    /// Returns an error when the expression is unsupported or structurally
    /// invalid.
    pub fn from_channel_expr(channel: &ChannelExpr) -> Result<Self, ChannelError> {
        let num_qubits = channel_num_qubits(channel).max(1);
        match channel {
            ChannelExpr::Unitary(unitary) => Self::from_unitary(unitary, num_qubits),
            ChannelExpr::MixedUnitary(ops) => {
                let basis_len = pauli_basis_len(num_qubits)?;
                let mut matrix = DMatrix::zeros(basis_len, basis_len);
                let mut total_probability = 0.0;
                for (probability, unitary) in ops {
                    validate_probability(*probability, DEFAULT_TOLERANCE)?;
                    let unitary_ptm = Self::from_unitary(unitary, num_qubits)?;
                    matrix += unitary_ptm.matrix * *probability;
                    total_probability += *probability;
                }
                if (total_probability - 1.0).abs() > DEFAULT_TOLERANCE {
                    return Err(ChannelError::ProbabilitySum {
                        sum: total_probability,
                        tolerance: DEFAULT_TOLERANCE,
                    });
                }
                Self::try_new(num_qubits, matrix)
            }
            ChannelExpr::AmplitudeDamping { .. }
            | ChannelExpr::PhaseDamping { .. }
            | ChannelExpr::Tensor(_)
            | ChannelExpr::Compose(_) => KrausOps::from_channel_expr(channel)?.to_ptm(),
            _ => Err(ChannelError::UnsupportedChannelExpr {
                reason: "dense PTM conversion supports unitary, mixed-unitary, amplitude-damping, phase-damping, tensor, and compose channels".to_string(),
            }),
        }
    }

    /// Returns the number of qubits represented by this PTM.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Returns the PTM basis ordering.
    #[must_use]
    pub fn basis_order(&self) -> PtmBasisOrder {
        self.basis_order
    }

    /// Returns the dense PTM matrix.
    #[must_use]
    pub fn matrix(&self) -> &DMatrix<f64> {
        &self.matrix
    }

    /// Consumes this PTM and returns its dense matrix.
    #[must_use]
    pub fn into_matrix(self) -> DMatrix<f64> {
        self.matrix
    }

    /// Returns one PTM entry by row/output and column/input basis indices.
    #[must_use]
    pub fn entry(&self, output: usize, input: usize) -> f64 {
        self.matrix[(output, input)]
    }

    /// Converts this PTM to a Choi matrix.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions overflow or the conversion encounters
    /// invalid matrix data.
    pub fn to_choi(&self) -> Result<ChoiMatrix, ChannelError> {
        ChoiMatrix::from_ptm(self)
    }

    /// Converts this PTM to Kraus operators through its Choi representation.
    ///
    /// # Errors
    ///
    /// Returns an error when the Choi conversion or numerical decomposition
    /// fails.
    pub fn to_kraus(&self) -> Result<KrausOps, ChannelError> {
        self.to_choi()?.to_kraus()
    }

    /// Converts this PTM to a column-stacked superoperator.
    ///
    /// # Errors
    ///
    /// Returns an error when the conversion through matrix units fails.
    pub fn to_superop(&self) -> Result<SuperOp, ChannelError> {
        SuperOp::from_ptm(self)
    }

    /// Converts this PTM to a Pauli-basis process matrix.
    ///
    /// # Errors
    ///
    /// Returns an error when the conversion through Choi/Kraus form fails.
    pub fn to_chi(&self) -> Result<ChiMatrix, ChannelError> {
        ChiMatrix::from_ptm(self)
    }
}

/// Concrete Kraus-operator representation of a quantum channel.
///
/// A Kraus channel applies
/// `E(rho) = sum_k K_k rho K_k†`.
/// Operators are full `2^n x 2^n` matrices using the same little-endian
/// computational-basis convention as [`UnitaryMatrix`](crate::UnitaryMatrix).
#[derive(Clone, Debug, PartialEq)]
pub struct KrausOps {
    num_qubits: usize,
    operators: Vec<DMatrix<Complex64>>,
}

impl KrausOps {
    /// Constructs a Kraus representation after structural validation.
    ///
    /// This validates only cheap structural properties: non-empty operator
    /// list, matrix shape, and finite entries. Trace-preservation and complete
    /// positivity are mathematical properties of the Kraus form; call
    /// [`Self::is_trace_preserving_with_tolerance`] when that check is needed.
    ///
    /// # Errors
    ///
    /// Returns an error when the operator list is empty, a matrix has the
    /// wrong shape, a dimension overflows, or an entry is not finite.
    pub fn try_new(
        num_qubits: usize,
        operators: Vec<DMatrix<Complex64>>,
    ) -> Result<Self, ChannelError> {
        if operators.is_empty() {
            return Err(ChannelError::EmptyKrausSet);
        }
        let dim = hilbert_dim(num_qubits)?;
        for operator in &operators {
            validate_complex_matrix(operator, dim, dim)?;
        }
        Ok(Self {
            num_qubits,
            operators,
        })
    }

    /// Constructs a one-operator Kraus representation for a unitary channel.
    ///
    /// # Errors
    ///
    /// Returns an error when the embedded unitary matrix has an invalid shape.
    pub fn from_unitary(unitary: &UnitaryRep, num_qubits: usize) -> Result<Self, ChannelError> {
        let operator = to_matrix_with_size(unitary, num_qubits).into_inner();
        Self::try_new(num_qubits, vec![operator])
    }

    /// Converts a symbolic channel expression to Kraus operators when
    /// supported.
    ///
    /// Supported variants are unitary, mixed-unitary, amplitude damping, phase
    /// damping, tensor, and compose. Measurement/preparation gate expressions,
    /// erasure, and leakage are intentionally rejected because they need
    /// instrument or flag semantics beyond a simple same-Hilbert-space Kraus
    /// channel.
    ///
    /// # Errors
    ///
    /// Returns an error when the expression is unsupported or invalid.
    pub fn from_channel_expr(channel: &ChannelExpr) -> Result<Self, ChannelError> {
        let num_qubits = channel_num_qubits(channel).max(1);
        kraus_from_channel_expr_with_size(channel, num_qubits)
    }

    /// Converts a symbolic channel expression to Kraus operators embedded in a
    /// specific system size.
    ///
    /// Use this when applying a local channel to a larger simulator state: a
    /// channel acting on qubit 3 of a 6-qubit system needs 6-qubit Kraus
    /// matrices, not the minimal 4-qubit representation implied by the highest
    /// touched qubit.
    ///
    /// # Errors
    ///
    /// Returns an error when the expression is unsupported, invalid, or touches
    /// a qubit outside `num_qubits`.
    pub fn from_channel_expr_with_num_qubits(
        channel: &ChannelExpr,
        num_qubits: usize,
    ) -> Result<Self, ChannelError> {
        for qubit in channel.qubits() {
            if qubit >= num_qubits {
                return Err(ChannelError::QubitOutOfRange { num_qubits, qubit });
            }
        }
        kraus_from_channel_expr_with_size(channel, num_qubits)
    }

    /// Converts this Kraus channel to a dense PTM.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions overflow or a PTM entry has a
    /// significant imaginary component.
    pub fn to_ptm(&self) -> Result<Ptm, ChannelError> {
        let basis_len = pauli_basis_len(self.num_qubits)?;
        let dim = hilbert_dim(self.num_qubits)?;
        #[allow(clippy::cast_precision_loss)]
        let dim_f = dim as f64;
        let basis_matrices = pauli_basis_matrices(self.num_qubits)?;
        let mut matrix = DMatrix::zeros(basis_len, basis_len);
        for input_idx in 0..basis_len {
            let mut evolved = DMatrix::zeros(dim, dim);
            for operator in &self.operators {
                evolved += operator * &basis_matrices[input_idx] * operator.adjoint();
            }
            for output_idx in 0..basis_len {
                let entry = trace_complex(&(&basis_matrices[output_idx] * &evolved)) / dim_f;
                if entry.im.abs() > DEFAULT_TOLERANCE {
                    return Err(ChannelError::NonRealCoefficient {
                        value: entry,
                        tolerance: DEFAULT_TOLERANCE,
                    });
                }
                matrix[(output_idx, input_idx)] = entry.re;
            }
        }
        Ptm::try_new(self.num_qubits, matrix)
    }

    /// Converts this Kraus channel to a Choi matrix.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions overflow.
    pub fn to_choi(&self) -> Result<ChoiMatrix, ChannelError> {
        ChoiMatrix::from_kraus(self)
    }

    /// Converts this Kraus channel to a column-stacked superoperator.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions overflow.
    pub fn to_superop(&self) -> Result<SuperOp, ChannelError> {
        SuperOp::from_kraus(self)
    }

    /// Converts this Kraus channel to a Pauli-basis process matrix.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions overflow.
    pub fn to_chi(&self) -> Result<ChiMatrix, ChannelError> {
        ChiMatrix::from_kraus(self)
    }

    /// Converts this Kraus channel to a Stinespring isometry.
    ///
    /// # Errors
    ///
    /// Returns an error when the stacked Kraus operators are not an isometry.
    pub fn to_stinespring(&self) -> Result<Stinespring, ChannelError> {
        Stinespring::from_kraus(self)
    }

    /// Returns whether `sum_k K_k† K_k = I` within the default tolerance.
    #[must_use]
    pub fn is_trace_preserving(&self) -> bool {
        self.is_trace_preserving_with_tolerance(1e-10)
    }

    /// Returns whether `sum_k K_k† K_k = I` within `tolerance`.
    #[must_use]
    pub fn is_trace_preserving_with_tolerance(&self, tolerance: f64) -> bool {
        let Ok(dim) = hilbert_dim(self.num_qubits) else {
            return false;
        };
        let mut accumulator = DMatrix::zeros(dim, dim);
        for operator in &self.operators {
            accumulator += operator.adjoint() * operator;
        }
        let identity = DMatrix::<Complex64>::identity(dim, dim);
        matrix_max_abs_diff(&accumulator, &identity) <= tolerance
    }

    /// Returns the number of qubits represented by this channel.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Returns the Kraus operators.
    #[must_use]
    pub fn operators(&self) -> &[DMatrix<Complex64>] {
        &self.operators
    }

    /// Consumes this value and returns the Kraus operators.
    #[must_use]
    pub fn into_operators(self) -> Vec<DMatrix<Complex64>> {
        self.operators
    }
}

/// Concrete Choi representation of a quantum channel.
///
/// PECOS stores the unnormalized Choi matrix
/// `J = sum_k vec(K_k) vec(K_k)†`, where `vec` is column-stacking. For a
/// trace-preserving channel on Hilbert dimension `d`, `Tr(J) = d` and
/// `Tr_output(J) = I_input`.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiMatrix {
    num_qubits: usize,
    matrix: DMatrix<Complex64>,
}

impl ChoiMatrix {
    /// Constructs a Choi matrix after structural validation.
    ///
    /// This validates only cheap structural properties: shape and finite
    /// entries. Complete positivity and trace preservation are explicit
    /// follow-up checks.
    ///
    /// # Errors
    ///
    /// Returns an error when the matrix shape is not `4^n x 4^n`, dimensions
    /// overflow, or an entry is not finite.
    pub fn try_new(num_qubits: usize, matrix: DMatrix<Complex64>) -> Result<Self, ChannelError> {
        let dim_squared = pauli_basis_len(num_qubits)?;
        validate_complex_matrix(&matrix, dim_squared, dim_squared)?;
        Ok(Self { num_qubits, matrix })
    }

    /// Converts Kraus operators to a Choi matrix.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions overflow.
    pub fn from_kraus(kraus: &KrausOps) -> Result<Self, ChannelError> {
        let dim = hilbert_dim(kraus.num_qubits)?;
        let dim_squared = pauli_basis_len(kraus.num_qubits)?;
        let mut matrix = DMatrix::zeros(dim_squared, dim_squared);
        for operator in kraus.operators() {
            for input_col in 0..dim {
                for output_row in 0..dim {
                    let row = choi_index(dim, output_row, input_col);
                    let row_value = operator[(output_row, input_col)];
                    for input_col_2 in 0..dim {
                        for output_row_2 in 0..dim {
                            let col = choi_index(dim, output_row_2, input_col_2);
                            matrix[(row, col)] +=
                                row_value * operator[(output_row_2, input_col_2)].conj();
                        }
                    }
                }
            }
        }
        Self::try_new(kraus.num_qubits, matrix)
    }

    /// Constructs a Choi matrix for a unitary channel.
    ///
    /// # Errors
    ///
    /// Returns an error when unitary embedding or Choi construction fails.
    pub fn from_unitary(unitary: &UnitaryRep, num_qubits: usize) -> Result<Self, ChannelError> {
        KrausOps::from_unitary(unitary, num_qubits)?.to_choi()
    }

    /// Converts a symbolic channel expression to a Choi matrix when supported.
    ///
    /// # Errors
    ///
    /// Returns an error when [`KrausOps::from_channel_expr`] rejects the
    /// expression or Choi construction fails.
    pub fn from_channel_expr(channel: &ChannelExpr) -> Result<Self, ChannelError> {
        KrausOps::from_channel_expr(channel)?.to_choi()
    }

    /// Converts a PTM to a Choi matrix.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions overflow.
    pub fn from_ptm(ptm: &Ptm) -> Result<Self, ChannelError> {
        let dim = hilbert_dim(ptm.num_qubits)?;
        let dim_squared = pauli_basis_len(ptm.num_qubits)?;
        let mut matrix = DMatrix::zeros(dim_squared, dim_squared);
        for input_row in 0..dim {
            for input_col in 0..dim {
                let mut matrix_unit = DMatrix::zeros(dim, dim);
                matrix_unit[(input_row, input_col)] = Complex64::new(1.0, 0.0);
                let evolved = apply_ptm_to_operator(ptm, &matrix_unit)?;
                for output_row in 0..dim {
                    for output_col in 0..dim {
                        matrix[(
                            choi_index(dim, output_row, input_row),
                            choi_index(dim, output_col, input_col),
                        )] = evolved[(output_row, output_col)];
                    }
                }
            }
        }
        Self::try_new(ptm.num_qubits, matrix)
    }

    /// Reconstructs a Choi matrix from complete operator-basis tomography data.
    ///
    /// `outputs` must contain `d^2` matrices, where `d = 2^num_qubits`.
    /// Entry `input_row + input_col * d` is the measured/reconstructed output
    /// operator `E(|input_row><input_col|)`. This is linear-inversion process
    /// tomography in the computational matrix-unit basis, using PECOS's
    /// column-stacked Choi convention.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions overflow, the sample count is not
    /// `d^2`, or any output matrix is not `d x d`.
    pub fn from_matrix_unit_outputs(
        num_qubits: usize,
        outputs: &[DMatrix<Complex64>],
    ) -> Result<Self, ChannelError> {
        let dim = hilbert_dim(num_qubits)?;
        let dim_squared = pauli_basis_len(num_qubits)?;
        if outputs.len() != dim_squared {
            return Err(ChannelError::InvalidTomographySampleCount {
                expected: dim_squared,
                actual: outputs.len(),
            });
        }

        let mut matrix = DMatrix::zeros(dim_squared, dim_squared);
        for input_col in 0..dim {
            for input_row in 0..dim {
                let output = &outputs[matrix_unit_index(dim, input_row, input_col)];
                validate_complex_matrix(output, dim, dim)?;
                for output_row in 0..dim {
                    for output_col in 0..dim {
                        matrix[(
                            choi_index(dim, output_row, input_row),
                            choi_index(dim, output_col, input_col),
                        )] = output[(output_row, output_col)];
                    }
                }
            }
        }
        Self::try_new(num_qubits, matrix)
    }

    /// Converts this Choi matrix to a dense PTM.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions overflow or a PTM entry has a
    /// significant imaginary component.
    pub fn to_ptm(&self) -> Result<Ptm, ChannelError> {
        let basis_len = pauli_basis_len(self.num_qubits)?;
        let dim = hilbert_dim(self.num_qubits)?;
        #[allow(clippy::cast_precision_loss)]
        let dim_f = dim as f64;
        let basis_matrices = pauli_basis_matrices(self.num_qubits)?;
        let mut matrix = DMatrix::zeros(basis_len, basis_len);
        for input_idx in 0..basis_len {
            let evolved = self.apply_to_operator(&basis_matrices[input_idx])?;
            for output_idx in 0..basis_len {
                let entry = trace_complex(&(&basis_matrices[output_idx] * &evolved)) / dim_f;
                if entry.im.abs() > DEFAULT_TOLERANCE {
                    return Err(ChannelError::NonRealCoefficient {
                        value: entry,
                        tolerance: DEFAULT_TOLERANCE,
                    });
                }
                matrix[(output_idx, input_idx)] = entry.re;
            }
        }
        Ptm::try_new(self.num_qubits, matrix)
    }

    /// Converts this Choi matrix to a Kraus representation using SVD.
    ///
    /// For a valid positive-semidefinite Choi matrix, the singular values are
    /// the Choi eigenvalues and the left singular vectors produce a Kraus
    /// decomposition. This method reconstructs the Choi matrix from the
    /// resulting Kraus operators and rejects inputs that are not positive
    /// semidefinite within the requested tolerance.
    ///
    /// # Errors
    ///
    /// Returns an error when numerical decomposition fails.
    pub fn to_kraus(&self) -> Result<KrausOps, ChannelError> {
        self.to_kraus_with_tolerance(DEFAULT_TOLERANCE)
    }

    /// Converts this Choi matrix to a column-stacked superoperator.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions overflow.
    pub fn to_superop(&self) -> Result<SuperOp, ChannelError> {
        SuperOp::from_choi(self)
    }

    /// Converts this Choi matrix to a Pauli-basis process matrix.
    ///
    /// # Errors
    ///
    /// Returns an error when conversion through Kraus form fails.
    pub fn to_chi(&self) -> Result<ChiMatrix, ChannelError> {
        ChiMatrix::from_choi(self)
    }

    /// Converts this Choi matrix to Kraus operators with an explicit
    /// singular-value cutoff.
    ///
    /// # Errors
    ///
    /// Returns an error when the tolerance is invalid, numerical decomposition
    /// fails, or the input is not positive semidefinite within tolerance.
    pub fn to_kraus_with_tolerance(&self, tolerance: f64) -> Result<KrausOps, ChannelError> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(ChannelError::DecompositionFailed {
                reason: format!("invalid Kraus decomposition tolerance: {tolerance}"),
            });
        }
        let dim = hilbert_dim(self.num_qubits)?;
        let svd = SVD::new(self.matrix.clone(), true, false);
        let u = svd.u.ok_or_else(|| ChannelError::DecompositionFailed {
            reason: "SVD did not return left singular vectors".to_string(),
        })?;
        let mut operators = Vec::new();
        for (idx, singular_value) in svd.singular_values.iter().copied().enumerate() {
            if singular_value <= tolerance {
                continue;
            }
            let scale = Complex64::new(singular_value.sqrt(), 0.0);
            let mut operator = DMatrix::zeros(dim, dim);
            for input_col in 0..dim {
                for output_row in 0..dim {
                    operator[(output_row, input_col)] =
                        u[(choi_index(dim, output_row, input_col), idx)] * scale;
                }
            }
            operators.push(operator);
        }
        if operators.is_empty() {
            operators.push(DMatrix::zeros(dim, dim));
        }
        let kraus = KrausOps::try_new(self.num_qubits, operators)?;
        let recovered = Self::from_kraus(&kraus)?;
        let reconstruction_tolerance = (10.0 * tolerance).max(1e-10);
        if matrix_max_abs_diff(recovered.matrix(), &self.matrix) > reconstruction_tolerance {
            return Err(ChannelError::DecompositionFailed {
                reason: "Choi matrix is not positive semidefinite within tolerance".to_string(),
            });
        }
        Ok(kraus)
    }

    /// Applies the represented channel to an operator matrix.
    ///
    /// # Errors
    ///
    /// Returns an error when the input operator shape is invalid.
    pub fn apply_to_operator(
        &self,
        operator: &DMatrix<Complex64>,
    ) -> Result<DMatrix<Complex64>, ChannelError> {
        let dim = hilbert_dim(self.num_qubits)?;
        validate_complex_matrix(operator, dim, dim)?;
        let mut out = DMatrix::zeros(dim, dim);
        for input_row in 0..dim {
            for input_col in 0..dim {
                let coefficient = operator[(input_row, input_col)];
                for output_row in 0..dim {
                    for output_col in 0..dim {
                        out[(output_row, output_col)] += coefficient
                            * self.matrix[(
                                choi_index(dim, output_row, input_row),
                                choi_index(dim, output_col, input_col),
                            )];
                    }
                }
            }
        }
        Ok(out)
    }

    /// Returns the output partial trace `Tr_output(J)`.
    ///
    /// With PECOS's column-stacked Choi convention, this equals the input-space
    /// identity for a trace-preserving channel.
    ///
    /// # Errors
    ///
    /// Returns an error when the Hilbert-space dimension overflows.
    pub fn partial_trace_output(&self) -> Result<DMatrix<Complex64>, ChannelError> {
        let dim = hilbert_dim(self.num_qubits)?;
        let mut reduced = DMatrix::zeros(dim, dim);
        for input_row in 0..dim {
            for input_col in 0..dim {
                let mut value = Complex64::new(0.0, 0.0);
                for output in 0..dim {
                    value += self.matrix[(
                        choi_index(dim, output, input_row),
                        choi_index(dim, output, input_col),
                    )];
                }
                reduced[(input_row, input_col)] = value;
            }
        }
        Ok(reduced)
    }

    /// Returns the input partial trace `Tr_input(J)`.
    ///
    /// With PECOS's column-stacked Choi convention, this equals `E(I)` and
    /// therefore equals the output-space identity for a unital channel.
    ///
    /// # Errors
    ///
    /// Returns an error when the Hilbert-space dimension overflows.
    pub fn partial_trace_input(&self) -> Result<DMatrix<Complex64>, ChannelError> {
        let dim = hilbert_dim(self.num_qubits)?;
        let mut reduced = DMatrix::zeros(dim, dim);
        for output_row in 0..dim {
            for output_col in 0..dim {
                let mut value = Complex64::new(0.0, 0.0);
                for input in 0..dim {
                    value += self.matrix[(
                        choi_index(dim, output_row, input),
                        choi_index(dim, output_col, input),
                    )];
                }
                reduced[(output_row, output_col)] = value;
            }
        }
        Ok(reduced)
    }

    /// Returns whether this Choi matrix is positive semidefinite within the
    /// default tolerance.
    #[must_use]
    pub fn is_completely_positive(&self) -> bool {
        self.is_completely_positive_with_tolerance(1e-10)
    }

    /// Returns whether this Choi matrix is positive semidefinite within
    /// `tolerance`.
    #[must_use]
    pub fn is_completely_positive_with_tolerance(&self, tolerance: f64) -> bool {
        self.to_kraus_with_tolerance(tolerance).is_ok()
    }

    /// Returns whether this Choi matrix is completely positive and
    /// trace-preserving within the default tolerance.
    #[must_use]
    pub fn is_cptp(&self) -> bool {
        self.is_cptp_with_tolerance(1e-10)
    }

    /// Returns whether this Choi matrix is completely positive and
    /// trace-preserving within `tolerance`.
    #[must_use]
    pub fn is_cptp_with_tolerance(&self, tolerance: f64) -> bool {
        self.is_completely_positive_with_tolerance(tolerance)
            && self.is_trace_preserving_with_tolerance(tolerance)
    }

    /// Returns whether `E(I) = I` within the default tolerance.
    #[must_use]
    pub fn is_unital(&self) -> bool {
        self.is_unital_with_tolerance(1e-10)
    }

    /// Returns whether `E(I) = I` within `tolerance`.
    #[must_use]
    pub fn is_unital_with_tolerance(&self, tolerance: f64) -> bool {
        let Ok(dim) = hilbert_dim(self.num_qubits) else {
            return false;
        };
        let Ok(reduced) = self.partial_trace_input() else {
            return false;
        };
        let identity = DMatrix::<Complex64>::identity(dim, dim);
        matrix_max_abs_diff(&reduced, &identity) <= tolerance
    }

    /// Returns whether `Tr_output(J) = I_input` within the default tolerance.
    #[must_use]
    pub fn is_trace_preserving(&self) -> bool {
        self.is_trace_preserving_with_tolerance(1e-10)
    }

    /// Returns whether `Tr_output(J) = I_input` within `tolerance`.
    #[must_use]
    pub fn is_trace_preserving_with_tolerance(&self, tolerance: f64) -> bool {
        let Ok(dim) = hilbert_dim(self.num_qubits) else {
            return false;
        };
        let Ok(reduced) = self.partial_trace_output() else {
            return false;
        };
        let identity = DMatrix::<Complex64>::identity(dim, dim);
        matrix_max_abs_diff(&reduced, &identity) <= tolerance
    }

    /// Returns the number of qubits represented by this Choi matrix.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Returns the dense Choi matrix.
    #[must_use]
    pub fn matrix(&self) -> &DMatrix<Complex64> {
        &self.matrix
    }

    /// Consumes this value and returns the dense Choi matrix.
    #[must_use]
    pub fn into_matrix(self) -> DMatrix<Complex64> {
        self.matrix
    }
}

/// Dense column-stacked superoperator representation.
///
/// `SuperOp` stores the matrix `S` satisfying `vec(E(A)) = S vec(A)`, where
/// `vec` uses column-stacking in PECOS's little-endian computational basis.
#[derive(Clone, Debug, PartialEq)]
pub struct SuperOp {
    num_qubits: usize,
    matrix: DMatrix<Complex64>,
}

impl SuperOp {
    /// Constructs a superoperator after structural validation.
    ///
    /// # Errors
    ///
    /// Returns an error when `matrix` is not `4^n x 4^n` or contains
    /// non-finite entries.
    pub fn try_new(num_qubits: usize, matrix: DMatrix<Complex64>) -> Result<Self, ChannelError> {
        let dim_squared = pauli_basis_len(num_qubits)?;
        validate_complex_matrix(&matrix, dim_squared, dim_squared)?;
        Ok(Self { num_qubits, matrix })
    }

    /// Constructs a superoperator from Kraus operators.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions overflow.
    pub fn from_kraus(kraus: &KrausOps) -> Result<Self, ChannelError> {
        let dim = hilbert_dim(kraus.num_qubits)?;
        let dim_squared = pauli_basis_len(kraus.num_qubits)?;
        let mut matrix = DMatrix::zeros(dim_squared, dim_squared);
        for input_col in 0..dim {
            for input_row in 0..dim {
                let input_idx = matrix_unit_index(dim, input_row, input_col);
                for output_col in 0..dim {
                    for output_row in 0..dim {
                        let output_idx = matrix_unit_index(dim, output_row, output_col);
                        let mut value = Complex64::new(0.0, 0.0);
                        for operator in kraus.operators() {
                            value += operator[(output_row, input_row)]
                                * operator[(output_col, input_col)].conj();
                        }
                        matrix[(output_idx, input_idx)] = value;
                    }
                }
            }
        }
        Self::try_new(kraus.num_qubits, matrix)
    }

    /// Constructs a superoperator from a Choi matrix.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions overflow.
    pub fn from_choi(choi: &ChoiMatrix) -> Result<Self, ChannelError> {
        let dim = hilbert_dim(choi.num_qubits)?;
        let dim_squared = pauli_basis_len(choi.num_qubits)?;
        let mut matrix = DMatrix::zeros(dim_squared, dim_squared);
        for input_col in 0..dim {
            for input_row in 0..dim {
                let input_idx = matrix_unit_index(dim, input_row, input_col);
                for output_col in 0..dim {
                    for output_row in 0..dim {
                        let output_idx = matrix_unit_index(dim, output_row, output_col);
                        matrix[(output_idx, input_idx)] = choi.matrix()[(
                            choi_index(dim, output_row, input_row),
                            choi_index(dim, output_col, input_col),
                        )];
                    }
                }
            }
        }
        Self::try_new(choi.num_qubits, matrix)
    }

    /// Constructs a superoperator from a PTM.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions overflow.
    pub fn from_ptm(ptm: &Ptm) -> Result<Self, ChannelError> {
        let dim = hilbert_dim(ptm.num_qubits)?;
        let dim_squared = pauli_basis_len(ptm.num_qubits)?;
        let mut matrix = DMatrix::zeros(dim_squared, dim_squared);
        for input_col in 0..dim {
            for input_row in 0..dim {
                let mut input = DMatrix::zeros(dim, dim);
                input[(input_row, input_col)] = Complex64::new(1.0, 0.0);
                let output = apply_ptm_to_operator(ptm, &input)?;
                let input_idx = matrix_unit_index(dim, input_row, input_col);
                for output_col in 0..dim {
                    for output_row in 0..dim {
                        let output_idx = matrix_unit_index(dim, output_row, output_col);
                        matrix[(output_idx, input_idx)] = output[(output_row, output_col)];
                    }
                }
            }
        }
        Self::try_new(ptm.num_qubits, matrix)
    }

    /// Constructs a superoperator from a supported channel expression.
    ///
    /// # Errors
    ///
    /// Returns an error when the expression cannot be represented as Kraus
    /// operators.
    pub fn from_channel_expr(channel: &ChannelExpr) -> Result<Self, ChannelError> {
        KrausOps::from_channel_expr(channel)?.to_superop()
    }

    /// Converts this superoperator to a Choi matrix.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions overflow.
    pub fn to_choi(&self) -> Result<ChoiMatrix, ChannelError> {
        let dim = hilbert_dim(self.num_qubits)?;
        let mut matrix = DMatrix::zeros(self.matrix.nrows(), self.matrix.ncols());
        for input_col in 0..dim {
            for input_row in 0..dim {
                let input_idx = matrix_unit_index(dim, input_row, input_col);
                for output_col in 0..dim {
                    for output_row in 0..dim {
                        let output_idx = matrix_unit_index(dim, output_row, output_col);
                        matrix[(
                            choi_index(dim, output_row, input_row),
                            choi_index(dim, output_col, input_col),
                        )] = self.matrix[(output_idx, input_idx)];
                    }
                }
            }
        }
        ChoiMatrix::try_new(self.num_qubits, matrix)
    }

    /// Converts this superoperator to a PTM.
    ///
    /// # Errors
    ///
    /// Returns an error when Choi/PTM conversion fails.
    pub fn to_ptm(&self) -> Result<Ptm, ChannelError> {
        self.to_choi()?.to_ptm()
    }

    /// Converts this superoperator to Kraus operators.
    ///
    /// # Errors
    ///
    /// Returns an error when Choi/Kraus conversion fails.
    pub fn to_kraus(&self) -> Result<KrausOps, ChannelError> {
        self.to_choi()?.to_kraus()
    }

    /// Returns the composition `self ∘ other`, applying `other` first.
    ///
    /// # Errors
    ///
    /// Returns an error when qubit counts differ.
    pub fn compose(&self, other: &Self) -> Result<Self, ChannelError> {
        if self.num_qubits != other.num_qubits {
            return Err(ChannelError::QubitCountMismatch {
                expected: self.num_qubits,
                actual: other.num_qubits,
            });
        }
        Self::try_new(self.num_qubits, &self.matrix * &other.matrix)
    }

    /// Returns the tensor product of two superoperators.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions overflow.
    pub fn tensor(&self, other: &Self) -> Result<Self, ChannelError> {
        Self::try_new(
            self.num_qubits + other.num_qubits,
            complex_kronecker(&self.matrix, &other.matrix),
        )
    }

    /// Returns the number of qubits represented by this superoperator.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Returns the dense superoperator matrix.
    #[must_use]
    pub fn matrix(&self) -> &DMatrix<Complex64> {
        &self.matrix
    }

    /// Consumes this value and returns its dense matrix.
    #[must_use]
    pub fn into_matrix(self) -> DMatrix<Complex64> {
        self.matrix
    }
}

/// Pauli-basis process matrix.
///
/// `ChiMatrix` stores coefficients `chi_ij` in
/// `E(rho) = sum_ij chi_ij P_i rho P_j†`, with the same Pauli basis ordering
/// as [`Ptm`].
#[derive(Clone, Debug, PartialEq)]
pub struct ChiMatrix {
    num_qubits: usize,
    basis_order: PtmBasisOrder,
    matrix: DMatrix<Complex64>,
}

impl ChiMatrix {
    /// Constructs a chi matrix after structural validation.
    ///
    /// # Errors
    ///
    /// Returns an error when `matrix` is not `4^n x 4^n` or contains
    /// non-finite entries.
    pub fn try_new(num_qubits: usize, matrix: DMatrix<Complex64>) -> Result<Self, ChannelError> {
        let basis_len = pauli_basis_len(num_qubits)?;
        validate_complex_matrix(&matrix, basis_len, basis_len)?;
        Ok(Self {
            num_qubits,
            basis_order: PtmBasisOrder::default(),
            matrix,
        })
    }

    /// Constructs a chi matrix from Kraus operators.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions overflow.
    pub fn from_kraus(kraus: &KrausOps) -> Result<Self, ChannelError> {
        let basis_len = pauli_basis_len(kraus.num_qubits)?;
        let dim = hilbert_dim(kraus.num_qubits)?;
        #[allow(clippy::cast_precision_loss)]
        let dim_f = dim as f64;
        let basis = pauli_basis_matrices(kraus.num_qubits)?;
        let mut matrix = DMatrix::zeros(basis_len, basis_len);
        for operator in kraus.operators() {
            let coefficients: Vec<Complex64> = basis
                .iter()
                .map(|pauli| trace_complex(&(pauli * operator)) / dim_f)
                .collect();
            for row in 0..basis_len {
                for col in 0..basis_len {
                    matrix[(row, col)] += coefficients[row] * coefficients[col].conj();
                }
            }
        }
        Self::try_new(kraus.num_qubits, matrix)
    }

    /// Constructs a chi matrix from a Choi matrix.
    ///
    /// # Errors
    ///
    /// Returns an error when Choi/Kraus conversion fails.
    pub fn from_choi(choi: &ChoiMatrix) -> Result<Self, ChannelError> {
        choi.to_kraus()?.to_chi()
    }

    /// Constructs a chi matrix from a PTM.
    ///
    /// # Errors
    ///
    /// Returns an error when PTM/Choi/Kraus conversion fails.
    pub fn from_ptm(ptm: &Ptm) -> Result<Self, ChannelError> {
        ptm.to_kraus()?.to_chi()
    }

    /// Constructs a chi matrix from a supported channel expression.
    ///
    /// # Errors
    ///
    /// Returns an error when the expression cannot be represented as Kraus
    /// operators.
    pub fn from_channel_expr(channel: &ChannelExpr) -> Result<Self, ChannelError> {
        KrausOps::from_channel_expr(channel)?.to_chi()
    }

    /// Converts this chi matrix to a Choi matrix.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions overflow.
    pub fn to_choi(&self) -> Result<ChoiMatrix, ChannelError> {
        let dim_squared = pauli_basis_len(self.num_qubits)?;
        let basis = pauli_basis_matrices(self.num_qubits)?;
        let basis_vectors: Vec<DMatrix<Complex64>> = basis.iter().map(vectorize_matrix).collect();
        let mut matrix = DMatrix::zeros(dim_squared, dim_squared);
        for row in 0..dim_squared {
            for col in 0..dim_squared {
                let coefficient = self.matrix[(row, col)];
                if coefficient.norm() <= DEFAULT_TOLERANCE {
                    continue;
                }
                matrix += &basis_vectors[row] * basis_vectors[col].adjoint() * coefficient;
            }
        }
        ChoiMatrix::try_new(self.num_qubits, matrix)
    }

    /// Converts this chi matrix to a PTM.
    ///
    /// # Errors
    ///
    /// Returns an error when Choi/PTM conversion fails.
    pub fn to_ptm(&self) -> Result<Ptm, ChannelError> {
        self.to_choi()?.to_ptm()
    }

    /// Returns the number of qubits represented by this chi matrix.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Returns the PTM basis ordering.
    #[must_use]
    pub fn basis_order(&self) -> PtmBasisOrder {
        self.basis_order
    }

    /// Returns the dense chi matrix.
    #[must_use]
    pub fn matrix(&self) -> &DMatrix<Complex64> {
        &self.matrix
    }

    /// Consumes this value and returns its dense matrix.
    #[must_use]
    pub fn into_matrix(self) -> DMatrix<Complex64> {
        self.matrix
    }
}

/// Stinespring isometry representation of a quantum channel.
///
/// The matrix has shape `(num_kraus * d) x d` and stacks Kraus operators
/// vertically, where `d = 2^num_qubits`.
#[derive(Clone, Debug, PartialEq)]
pub struct Stinespring {
    num_qubits: usize,
    environment_dim: usize,
    isometry: DMatrix<Complex64>,
}

impl Stinespring {
    /// Constructs a Stinespring isometry after structural validation.
    ///
    /// # Errors
    ///
    /// Returns an error when shape or isometry validation fails.
    pub fn try_new(num_qubits: usize, isometry: DMatrix<Complex64>) -> Result<Self, ChannelError> {
        let dim = hilbert_dim(num_qubits)?;
        if isometry.ncols() != dim || isometry.nrows() == 0 || !isometry.nrows().is_multiple_of(dim)
        {
            return Err(ChannelError::InvalidMatrixShape {
                expected_rows: dim,
                expected_cols: dim,
                rows: isometry.nrows(),
                cols: isometry.ncols(),
            });
        }
        validate_complex_matrix(&isometry, isometry.nrows(), dim)?;
        let identity = DMatrix::<Complex64>::identity(dim, dim);
        let gram = isometry.adjoint() * &isometry;
        if matrix_max_abs_diff(&gram, &identity) > 1e-10 {
            return Err(ChannelError::DecompositionFailed {
                reason: "Stinespring matrix is not an isometry".to_string(),
            });
        }
        Ok(Self {
            num_qubits,
            environment_dim: isometry.nrows() / dim,
            isometry,
        })
    }

    /// Constructs a Stinespring isometry by stacking Kraus operators.
    ///
    /// # Errors
    ///
    /// Returns an error when the Kraus operators are not trace preserving.
    pub fn from_kraus(kraus: &KrausOps) -> Result<Self, ChannelError> {
        let dim = hilbert_dim(kraus.num_qubits)?;
        let mut isometry = DMatrix::zeros(dim * kraus.operators().len(), dim);
        for (kraus_idx, operator) in kraus.operators().iter().enumerate() {
            for row in 0..dim {
                for col in 0..dim {
                    isometry[(kraus_idx * dim + row, col)] = operator[(row, col)];
                }
            }
        }
        Self::try_new(kraus.num_qubits, isometry)
    }

    /// Converts this Stinespring isometry to Kraus operators.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions overflow.
    pub fn to_kraus(&self) -> Result<KrausOps, ChannelError> {
        let dim = hilbert_dim(self.num_qubits)?;
        let operators = (0..self.environment_dim)
            .map(|kraus_idx| {
                DMatrix::from_fn(dim, dim, |row, col| {
                    self.isometry[(kraus_idx * dim + row, col)]
                })
            })
            .collect();
        KrausOps::try_new(self.num_qubits, operators)
    }

    /// Converts this Stinespring isometry to a Choi matrix.
    ///
    /// # Errors
    ///
    /// Returns an error when Kraus/Choi conversion fails.
    pub fn to_choi(&self) -> Result<ChoiMatrix, ChannelError> {
        self.to_kraus()?.to_choi()
    }

    /// Converts this Stinespring isometry to a superoperator.
    ///
    /// # Errors
    ///
    /// Returns an error when Kraus/superoperator conversion fails.
    pub fn to_superop(&self) -> Result<SuperOp, ChannelError> {
        self.to_kraus()?.to_superop()
    }

    /// Returns the number of qubits represented by this isometry.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Returns the environment dimension, equal to the number of Kraus blocks.
    #[must_use]
    pub fn environment_dim(&self) -> usize {
        self.environment_dim
    }

    /// Returns the dense Stinespring isometry.
    #[must_use]
    pub fn isometry(&self) -> &DMatrix<Complex64> {
        &self.isometry
    }

    /// Consumes this value and returns its dense isometry.
    #[must_use]
    pub fn into_isometry(self) -> DMatrix<Complex64> {
        self.isometry
    }
}

/// Returns the partial trace of a density matrix over selected qubits.
///
/// Qubit indexing is little-endian: qubit 0 is the least-significant bit of
/// the computational-basis index. The returned density matrix keeps the
/// untraced qubits in ascending qubit-index order.
///
/// # Errors
///
/// Returns an error when the matrix is not `2^num_qubits x 2^num_qubits`, a
/// traced qubit is outside range, or a traced qubit is repeated.
pub fn partial_trace(
    matrix: &DMatrix<Complex64>,
    num_qubits: usize,
    traced_qubits: &[usize],
) -> Result<DMatrix<Complex64>, ChannelError> {
    let dim = hilbert_dim(num_qubits)?;
    if matrix.nrows() != dim || matrix.ncols() != dim {
        return Err(ChannelError::InvalidMatrixShape {
            expected_rows: dim,
            expected_cols: dim,
            rows: matrix.nrows(),
            cols: matrix.ncols(),
        });
    }

    let mut traced = traced_qubits.to_vec();
    traced.sort_unstable();
    for window in traced.windows(2) {
        if window[0] == window[1] {
            return Err(ChannelError::DuplicateSubsystem { qubit: window[0] });
        }
    }
    for &qubit in &traced {
        if qubit >= num_qubits {
            return Err(ChannelError::QubitOutOfRange { num_qubits, qubit });
        }
    }

    let kept: Vec<usize> = (0..num_qubits)
        .filter(|qubit| traced.binary_search(qubit).is_err())
        .collect();
    let out_dim = 1usize << kept.len();
    let traced_dim = 1usize << traced.len();
    let mut out = DMatrix::zeros(out_dim, out_dim);
    for kept_row in 0..out_dim {
        for kept_col in 0..out_dim {
            let mut value = Complex64::new(0.0, 0.0);
            for traced_idx in 0..traced_dim {
                let row = embed_subsystem_index(&kept, kept_row, &traced, traced_idx);
                let col = embed_subsystem_index(&kept, kept_col, &traced, traced_idx);
                value += matrix[(row, col)];
            }
            out[(kept_row, kept_col)] = value;
        }
    }
    Ok(out)
}

/// Returns the computational matrix-unit operator basis.
///
/// The returned vector has length `d^2`, where `d = 2^num_qubits`. Entry
/// `row + col * d` is the matrix unit `|row><col|`. This order is the input
/// order expected by [`ChoiMatrix::from_matrix_unit_outputs`].
///
/// # Errors
///
/// Returns an error when the Hilbert-space dimension overflows.
pub fn matrix_unit_basis(num_qubits: usize) -> Result<Vec<DMatrix<Complex64>>, ChannelError> {
    let dim = hilbert_dim(num_qubits)?;
    let dim_squared = pauli_basis_len(num_qubits)?;
    let mut basis = Vec::with_capacity(dim_squared);
    for col in 0..dim {
        for row in 0..dim {
            let mut matrix = DMatrix::zeros(dim, dim);
            matrix[(row, col)] = Complex64::new(1.0, 0.0);
            basis.push(matrix);
        }
    }
    Ok(basis)
}

/// Metadata for one computational matrix-unit tomography input.
///
/// The input operator is `|row><col|`, and `index = row + col * dim`, where
/// `dim = 2^num_qubits`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatrixUnitTomographyInput {
    /// Input index in PECOS matrix-unit tomography order.
    pub index: usize,
    /// Ket row of the matrix unit.
    pub row: usize,
    /// Bra column of the matrix unit.
    pub col: usize,
}

/// Complete linear-inversion process-tomography design.
///
/// The current design uses the computational matrix-unit basis. It is useful
/// for exact channel characterization, simulator validation, and importing
/// reconstructed channel data. It is not a physical state-preparation recipe;
/// it records the linear operator basis and the PECOS ordering needed to
/// reconstruct a [`ChoiMatrix`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcessTomographyDesign {
    num_qubits: usize,
    dim: usize,
    num_inputs: usize,
}

impl ProcessTomographyDesign {
    /// Builds the complete computational matrix-unit design.
    ///
    /// # Errors
    ///
    /// Returns an error when the Hilbert-space dimension overflows.
    pub fn matrix_unit(num_qubits: usize) -> Result<Self, ChannelError> {
        let dim = hilbert_dim(num_qubits)?;
        let num_inputs = pauli_basis_len(num_qubits)?;
        Ok(Self {
            num_qubits,
            dim,
            num_inputs,
        })
    }

    /// Returns the number of qubits in the characterized channel.
    #[must_use]
    pub const fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Returns the Hilbert-space dimension `2^num_qubits`.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the number of matrix-unit input operators, `dim^2`.
    #[must_use]
    pub const fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    /// Returns the index for matrix unit `|row><col|`.
    ///
    /// # Errors
    ///
    /// Returns an error if `row` or `col` is outside the Hilbert space.
    pub fn input_index(&self, row: usize, col: usize) -> Result<usize, ChannelError> {
        if row >= self.dim || col >= self.dim {
            return Err(ChannelError::MatrixUnitOutOfRange {
                dim: self.dim,
                row,
                col,
            });
        }
        Ok(matrix_unit_index(self.dim, row, col))
    }

    /// Returns metadata for one matrix-unit input.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is outside the design.
    pub fn input_metadata(&self, index: usize) -> Result<MatrixUnitTomographyInput, ChannelError> {
        if index >= self.num_inputs {
            return Err(ChannelError::TomographyInputOutOfRange {
                num_inputs: self.num_inputs,
                index,
            });
        }
        Ok(MatrixUnitTomographyInput {
            index,
            row: index % self.dim,
            col: index / self.dim,
        })
    }

    /// Returns metadata for all matrix-unit inputs in reconstruction order.
    #[must_use]
    pub fn input_metadata_all(&self) -> Vec<MatrixUnitTomographyInput> {
        (0..self.num_inputs)
            .map(|index| MatrixUnitTomographyInput {
                index,
                row: index % self.dim,
                col: index / self.dim,
            })
            .collect()
    }

    /// Returns the matrix-unit input operator at `index`.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is outside the design.
    pub fn input_operator(&self, index: usize) -> Result<DMatrix<Complex64>, ChannelError> {
        let input = self.input_metadata(index)?;
        let mut matrix = DMatrix::zeros(self.dim, self.dim);
        matrix[(input.row, input.col)] = Complex64::new(1.0, 0.0);
        Ok(matrix)
    }

    /// Returns all matrix-unit input operators in reconstruction order.
    #[must_use]
    pub fn input_operators(&self) -> Vec<DMatrix<Complex64>> {
        (0..self.num_inputs)
            .map(|index| {
                let row = index % self.dim;
                let col = index / self.dim;
                let mut matrix = DMatrix::zeros(self.dim, self.dim);
                matrix[(row, col)] = Complex64::new(1.0, 0.0);
                matrix
            })
            .collect()
    }

    /// Applies `channel` to each design input in reconstruction order.
    ///
    /// # Errors
    ///
    /// Returns an error when `channel` acts on a different number of qubits or
    /// if channel application fails.
    pub fn simulate_outputs(
        &self,
        channel: &ChoiMatrix,
    ) -> Result<Vec<DMatrix<Complex64>>, ChannelError> {
        if channel.num_qubits() != self.num_qubits {
            return Err(ChannelError::QubitCountMismatch {
                expected: self.num_qubits,
                actual: channel.num_qubits(),
            });
        }
        self.input_operators()
            .iter()
            .map(|operator| channel.apply_to_operator(operator))
            .collect()
    }

    /// Reconstructs a Choi matrix from outputs ordered by this design.
    ///
    /// # Errors
    ///
    /// Returns an error when output count or shapes do not match the design.
    pub fn reconstruct_choi(
        &self,
        outputs: &[DMatrix<Complex64>],
    ) -> Result<ChoiMatrix, ChannelError> {
        ChoiMatrix::from_matrix_unit_outputs(self.num_qubits, outputs)
    }
}

/// Samples a Hilbert-Schmidt random density matrix on `num_qubits` qubits.
///
/// This samples a square complex Ginibre matrix `G` and returns
/// `G G† / Tr(G G†)`. The returned matrix uses the same little-endian
/// computational-basis order as PECOS's dense matrix helpers.
///
/// # Errors
///
/// Returns an error when the Hilbert-space dimension overflows.
pub fn random_density_matrix<R>(
    rng: &mut R,
    num_qubits: usize,
) -> Result<DMatrix<Complex64>, ChannelError>
where
    R: Rng + ?Sized,
{
    let dim = hilbert_dim(num_qubits)?;
    random_density_matrix_with_rank(rng, num_qubits, dim)
}

/// Samples a Hilbert-Schmidt random density matrix with explicit Ginibre rank.
///
/// A rank of `1` produces a random pure-state density matrix. Larger ranks
/// produce mixed states from a `dim x rank` complex Ginibre matrix.
///
/// # Errors
///
/// Returns an error when the Hilbert-space dimension overflows or `rank == 0`.
pub fn random_density_matrix_with_rank<R>(
    rng: &mut R,
    num_qubits: usize,
    rank: usize,
) -> Result<DMatrix<Complex64>, ChannelError>
where
    R: Rng + ?Sized,
{
    if rank == 0 {
        return Err(ChannelError::EmptyKrausSet);
    }
    let dim = hilbert_dim(num_qubits)?;
    let ginibre = DMatrix::from_fn(dim, rank, |_, _| standard_complex_normal(rng));
    let mut rho = &ginibre * ginibre.adjoint();
    let trace = trace_complex(&rho).re;
    if trace <= 0.0 || !trace.is_finite() {
        return Err(ChannelError::DecompositionFailed {
            reason: "random density matrix has invalid trace".to_string(),
        });
    }
    rho /= Complex64::new(trace, 0.0);
    Ok(rho)
}

/// Samples a random CPTP quantum channel in Kraus form.
///
/// The implementation samples a random Stinespring isometry by QR-decomposing a
/// complex Ginibre matrix of shape `(num_kraus * d) x d`, where
/// `d = 2^num_qubits`, then splits the isometry into `num_kraus` Kraus blocks.
/// The resulting operators satisfy `sum_k K_k† K_k = I` up to numerical
/// precision.
///
/// # Errors
///
/// Returns an error when dimensions overflow or `num_kraus == 0`.
pub fn random_quantum_channel<R>(
    rng: &mut R,
    num_qubits: usize,
    num_kraus: usize,
) -> Result<KrausOps, ChannelError>
where
    R: Rng + ?Sized,
{
    if num_kraus == 0 {
        return Err(ChannelError::EmptyKrausSet);
    }
    let dim = hilbert_dim(num_qubits)?;
    let rows = dim
        .checked_mul(num_kraus)
        .ok_or(ChannelError::DimensionOverflow { num_qubits })?;
    let ginibre = DMatrix::from_fn(rows, dim, |_, _| standard_complex_normal(rng));
    let (mut q, r) = ginibre.qr().unpack();

    for col in 0..dim {
        let diagonal = r[(col, col)];
        let norm = diagonal.norm();
        if norm > 0.0 {
            let phase = diagonal / norm;
            for row in 0..rows {
                q[(row, col)] *= phase;
            }
        }
    }

    let operators = (0..num_kraus)
        .map(|kraus_idx| {
            let start = kraus_idx * dim;
            DMatrix::from_fn(dim, dim, |row, col| q[(start + row, col)])
        })
        .collect();
    KrausOps::try_new(num_qubits, operators)
}

/// Samples a random `num_qubits`-qubit Pauli string.
///
/// Each qubit independently receives one of `I, X, Y, Z` with equal
/// probability. The all-identity Pauli is allowed.
pub fn random_pauli<R: Rng + ?Sized>(rng: &mut R, num_qubits: usize) -> PauliString {
    let paulis: Vec<Pauli> = (0..num_qubits)
        .map(|_| match rng.random_range(0..4) {
            0 => Pauli::I,
            1 => Pauli::X,
            2 => Pauli::Y,
            _ => Pauli::Z,
        })
        .collect();
    PauliString::from_paulis(&paulis)
}

/// Samples one of the 24 single-qubit Clifford gate primitives uniformly.
pub fn random_1q_clifford<R: Rng + ?Sized>(rng: &mut R) -> Clifford {
    let all = Clifford::all_1q();
    all[rng.random_range(0..all.len())]
}

/// Samples one of the standard two-qubit Clifford gate primitives uniformly.
pub fn random_2q_clifford<R: Rng + ?Sized>(rng: &mut R) -> Clifford {
    let all = Clifford::all_2q();
    all[rng.random_range(0..all.len())]
}

/// Samples one named Clifford gate primitive uniformly from the PECOS Clifford
/// enum.
pub fn random_clifford<R: Rng + ?Sized>(rng: &mut R) -> Clifford {
    let all = Clifford::all();
    all[rng.random_range(0..all.len())]
}

fn standard_complex_normal<R: Rng + ?Sized>(rng: &mut R) -> Complex64 {
    Complex64::new(standard_normal(rng), standard_normal(rng))
}

fn standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    loop {
        let u1 = rng.random::<f64>();
        if u1 > 0.0 {
            let u2 = rng.random::<f64>();
            return (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos();
        }
    }
}

fn bitmask_from_paulis(paulis: &[Pauli]) -> PauliBitmaskSmall {
    let mut out = PauliBitmaskSmall::identity();
    for (qubit, pauli) in paulis.iter().copied().enumerate() {
        match pauli {
            Pauli::I => {}
            Pauli::X => out.x_bits.set_bit(qubit),
            Pauli::Y => {
                out.x_bits.set_bit(qubit);
                out.z_bits.set_bit(qubit);
            }
            Pauli::Z => out.z_bits.set_bit(qubit),
        }
    }
    out
}

fn channel_num_qubits(channel: &ChannelExpr) -> usize {
    channel.qubits().into_iter().max().map_or(0, |q| q + 1)
}

fn bitmask_to_pauli_string(num_qubits: usize, pauli: &PauliBitmaskSmall) -> PauliString {
    let paulis: Vec<Pauli> = (0..num_qubits)
        .map(|qubit| match (pauli.has_x(qubit), pauli.has_z(qubit)) {
            (false, false) => Pauli::I,
            (true, false) => Pauli::X,
            (true, true) => Pauli::Y,
            (false, true) => Pauli::Z,
        })
        .collect();
    PauliString::from_paulis(&paulis)
}

fn pauli_basis_matrices(num_qubits: usize) -> Result<Vec<DMatrix<Complex64>>, ChannelError> {
    let basis_len = pauli_basis_len(num_qubits)?;
    let mut out = Vec::with_capacity(basis_len);
    for basis_idx in 0..basis_len {
        let bitmask = basis_bitmask(num_qubits, basis_idx)?;
        let pauli = bitmask_to_pauli_string(num_qubits, &bitmask);
        out.push(to_matrix_with_size(&UnitaryRep::Pauli(pauli), num_qubits).into_inner());
    }
    Ok(out)
}

fn apply_ptm_to_operator(
    ptm: &Ptm,
    operator: &DMatrix<Complex64>,
) -> Result<DMatrix<Complex64>, ChannelError> {
    let dim = hilbert_dim(ptm.num_qubits)?;
    validate_complex_matrix(operator, dim, dim)?;
    #[allow(clippy::cast_precision_loss)]
    let dim_f = dim as f64;
    let basis_matrices = pauli_basis_matrices(ptm.num_qubits)?;
    let mut out = DMatrix::zeros(dim, dim);
    for input_idx in 0..basis_matrices.len() {
        let coefficient = trace_complex(&(&basis_matrices[input_idx] * operator)) / dim_f;
        for (output_idx, basis_matrix) in basis_matrices.iter().enumerate() {
            let output_coefficient = coefficient * ptm.entry(output_idx, input_idx);
            out += basis_matrix * output_coefficient;
        }
    }
    Ok(out)
}

fn kraus_from_channel_expr_with_size(
    channel: &ChannelExpr,
    num_qubits: usize,
) -> Result<KrausOps, ChannelError> {
    match channel {
        ChannelExpr::Unitary(unitary) => KrausOps::from_unitary(unitary, num_qubits),
        ChannelExpr::MixedUnitary(ops) => {
            let mut total_probability = 0.0;
            let mut operators = Vec::with_capacity(ops.len());
            for (probability, unitary) in ops {
                validate_probability(*probability, DEFAULT_TOLERANCE)?;
                total_probability += *probability;
                if *probability > DEFAULT_TOLERANCE {
                    let scale = Complex64::new(probability.sqrt(), 0.0);
                    operators.push(to_matrix_with_size(unitary, num_qubits).into_inner() * scale);
                }
            }
            if (total_probability - 1.0).abs() > DEFAULT_TOLERANCE {
                return Err(ChannelError::ProbabilitySum {
                    sum: total_probability,
                    tolerance: DEFAULT_TOLERANCE,
                });
            }
            KrausOps::try_new(num_qubits, operators)
        }
        ChannelExpr::AmplitudeDamping { gamma, qubit } => {
            validate_unit_interval(*gamma)?;
            let sqrt_survival = (1.0 - gamma).sqrt();
            let sqrt_decay = gamma.sqrt();
            let k0 = DMatrix::from_row_slice(
                2,
                2,
                &[
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(sqrt_survival, 0.0),
                ],
            );
            let k1 = DMatrix::from_row_slice(
                2,
                2,
                &[
                    Complex64::new(0.0, 0.0),
                    Complex64::new(sqrt_decay, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            );
            KrausOps::try_new(
                num_qubits,
                vec![
                    embed_single_qubit_operator(num_qubits, *qubit, &k0)?,
                    embed_single_qubit_operator(num_qubits, *qubit, &k1)?,
                ],
            )
        }
        ChannelExpr::PhaseDamping { lambda, qubit } => {
            validate_unit_interval(*lambda)?;
            let sqrt_survival = (1.0 - lambda).sqrt();
            let sqrt_damp = lambda.sqrt();
            let k0 = DMatrix::from_row_slice(
                2,
                2,
                &[
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(sqrt_survival, 0.0),
                ],
            );
            let k1 = DMatrix::from_row_slice(
                2,
                2,
                &[
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(sqrt_damp, 0.0),
                ],
            );
            KrausOps::try_new(
                num_qubits,
                vec![
                    embed_single_qubit_operator(num_qubits, *qubit, &k0)?,
                    embed_single_qubit_operator(num_qubits, *qubit, &k1)?,
                ],
            )
        }
        ChannelExpr::Tensor(parts) => {
            if parts.is_empty() {
                return Err(ChannelError::UnsupportedChannelExpr {
                    reason: "empty channel tensor has no qubit context".to_string(),
                });
            }
            validate_disjoint_channel_parts(parts)?;
            let mut operators = vec![DMatrix::<Complex64>::identity(
                hilbert_dim(num_qubits)?,
                hilbert_dim(num_qubits)?,
            )];
            for part in parts {
                let part_ops = kraus_from_channel_expr_with_size(part, num_qubits)?;
                operators = compose_kraus_sets(&operators, part_ops.operators());
            }
            KrausOps::try_new(num_qubits, operators)
        }
        ChannelExpr::Compose(parts) => {
            if parts.is_empty() {
                return Err(ChannelError::UnsupportedChannelExpr {
                    reason: "empty channel composition has no qubit context".to_string(),
                });
            }
            let dim = hilbert_dim(num_qubits)?;
            let mut operators = vec![DMatrix::<Complex64>::identity(dim, dim)];
            for part in parts {
                let part_ops = kraus_from_channel_expr_with_size(part, num_qubits)?;
                operators = compose_kraus_sets(&operators, part_ops.operators());
            }
            KrausOps::try_new(num_qubits, operators)
        }
        ChannelExpr::Gate(_) | ChannelExpr::Erasure { .. } | ChannelExpr::Leakage { .. } => {
            Err(ChannelError::UnsupportedChannelExpr {
                reason:
                    "gate instruments, erasure, and leakage need explicit outcome/flag semantics"
                        .to_string(),
            })
        }
    }
}

fn validate_disjoint_channel_parts(parts: &[ChannelExpr]) -> Result<(), ChannelError> {
    let mut seen = BTreeSet::new();
    for part in parts {
        for qubit in part.qubits() {
            if !seen.insert(qubit) {
                return Err(ChannelError::DuplicateSubsystem { qubit });
            }
        }
    }
    Ok(())
}

fn compose_kraus_sets(
    current: &[DMatrix<Complex64>],
    next: &[DMatrix<Complex64>],
) -> Vec<DMatrix<Complex64>> {
    let mut out = Vec::with_capacity(current.len() * next.len());
    for next_op in next {
        for current_op in current {
            out.push(next_op * current_op);
        }
    }
    out
}

fn embed_single_qubit_operator(
    num_qubits: usize,
    qubit: usize,
    local: &DMatrix<Complex64>,
) -> Result<DMatrix<Complex64>, ChannelError> {
    if qubit >= num_qubits {
        return Err(ChannelError::QubitOutOfRange { num_qubits, qubit });
    }
    validate_complex_matrix(local, 2, 2)?;

    let dim = hilbert_dim(num_qubits)?;
    let qubit_mask = 1usize << qubit;
    let mut out = DMatrix::zeros(dim, dim);
    for row in 0..dim {
        for col in 0..dim {
            if row & !qubit_mask == col & !qubit_mask {
                let local_row = usize::from((row & qubit_mask) != 0);
                let local_col = usize::from((col & qubit_mask) != 0);
                out[(row, col)] = local[(local_row, local_col)];
            }
        }
    }
    Ok(out)
}

fn choi_index(dim: usize, output_index: usize, input_index: usize) -> usize {
    output_index + input_index * dim
}

fn matrix_unit_index(dim: usize, row: usize, col: usize) -> usize {
    row + col * dim
}

fn vectorize_matrix(matrix: &DMatrix<Complex64>) -> DMatrix<Complex64> {
    DMatrix::from_fn(matrix.nrows() * matrix.ncols(), 1, |idx, _| {
        let row = idx % matrix.nrows();
        let col = idx / matrix.nrows();
        matrix[(row, col)]
    })
}

fn complex_kronecker(left: &DMatrix<Complex64>, right: &DMatrix<Complex64>) -> DMatrix<Complex64> {
    let rows = left.nrows() * right.nrows();
    let cols = left.ncols() * right.ncols();
    let mut out = DMatrix::zeros(rows, cols);
    for left_row in 0..left.nrows() {
        for left_col in 0..left.ncols() {
            let scale = left[(left_row, left_col)];
            for right_row in 0..right.nrows() {
                for right_col in 0..right.ncols() {
                    out[(
                        left_row * right.nrows() + right_row,
                        left_col * right.ncols() + right_col,
                    )] = scale * right[(right_row, right_col)];
                }
            }
        }
    }
    out
}

fn trace_complex(matrix: &DMatrix<Complex64>) -> Complex64 {
    let n = matrix.nrows().min(matrix.ncols());
    (0..n).map(|idx| matrix[(idx, idx)]).sum()
}

fn hilbert_dim(num_qubits: usize) -> Result<usize, ChannelError> {
    2usize
        .checked_pow(
            num_qubits
                .try_into()
                .map_err(|_| ChannelError::DimensionOverflow { num_qubits })?,
        )
        .ok_or(ChannelError::DimensionOverflow { num_qubits })
}

fn embed_subsystem_index(
    kept_qubits: &[usize],
    kept_index: usize,
    traced_qubits: &[usize],
    traced_index: usize,
) -> usize {
    let mut out = 0usize;
    for (bit, qubit) in kept_qubits.iter().copied().enumerate() {
        if ((kept_index >> bit) & 1) != 0 {
            out |= 1usize << qubit;
        }
    }
    for (bit, qubit) in traced_qubits.iter().copied().enumerate() {
        if ((traced_index >> bit) & 1) != 0 {
            out |= 1usize << qubit;
        }
    }
    out
}

fn unitary_rep_to_pauli_bitmask(
    num_qubits: usize,
    unitary: &UnitaryRep,
) -> Result<PauliBitmaskSmall, ChannelError> {
    match unitary {
        unitary if unitary.is_identity() => Ok(PauliBitmaskSmall::identity()),
        UnitaryRep::Pauli(pauli) => {
            // Global Pauli phase cancels in the induced channel U rho U†.
            pauli_string_to_bitmask(num_qubits, pauli)
        }
        UnitaryRep::Tensor(parts) => {
            let mut out = PauliBitmaskSmall::identity();
            for part in parts {
                let part_mask = unitary_rep_to_pauli_bitmask(num_qubits, part)?;
                out = out.multiply(&part_mask);
            }
            Ok(out)
        }
        _ => Err(ChannelError::UnsupportedChannelExpr {
            reason: format!("unitary is not a Pauli operator: {unitary:?}"),
        }),
    }
}

fn validate_num_qubits(num_qubits: usize, pauli: &PauliBitmaskSmall) -> Result<(), ChannelError> {
    if let Some(qubit) = highest_qubit(pauli)
        && qubit >= num_qubits
    {
        return Err(ChannelError::QubitOutOfRange { num_qubits, qubit });
    }
    Ok(())
}

fn highest_qubit(pauli: &PauliBitmaskSmall) -> Option<usize> {
    [
        pauli.x_bits.highest_set_bit(),
        pauli.z_bits.highest_set_bit(),
    ]
    .into_iter()
    .flatten()
    .max()
}

fn validate_complex(value: Complex64) -> Result<(), ChannelError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(ChannelError::InvalidCoefficient { value })
    }
}

fn validate_complex_matrix(
    matrix: &DMatrix<Complex64>,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<(), ChannelError> {
    if matrix.nrows() != expected_rows || matrix.ncols() != expected_cols {
        return Err(ChannelError::InvalidMatrixShape {
            expected_rows,
            expected_cols,
            rows: matrix.nrows(),
            cols: matrix.ncols(),
        });
    }
    for value in matrix.iter() {
        validate_complex(*value)?;
    }
    Ok(())
}

fn validate_real(value: f64) -> Result<(), ChannelError> {
    validate_complex(Complex64::new(value, 0.0))
}

fn validate_probability(value: f64, tolerance: f64) -> Result<(), ChannelError> {
    validate_real(value)?;
    if value < -tolerance {
        return Err(ChannelError::InvalidProbability { value, tolerance });
    }
    Ok(())
}

fn validate_unit_interval(value: f64) -> Result<(), ChannelError> {
    validate_probability(value, DEFAULT_TOLERANCE)?;
    if value > 1.0 + DEFAULT_TOLERANCE {
        return Err(ChannelError::InvalidProbability {
            value,
            tolerance: DEFAULT_TOLERANCE,
        });
    }
    Ok(())
}

fn matrix_max_abs_diff(a: &DMatrix<Complex64>, b: &DMatrix<Complex64>) -> f64 {
    if a.shape() != b.shape() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(left, right)| (*left - *right).norm())
        .fold(0.0, f64::max)
}

fn commutation_character(a: &PauliBitmaskSmall, b: &PauliBitmaskSmall) -> f64 {
    if a.commutes_with(b) { 1.0 } else { -1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::op;
    use pecos_core::unitary;
    use pecos_core::{Op, QuarterPhase};
    use pecos_random::PecosRng;

    fn assert_close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-10, "{a} != {b}");
    }

    fn assert_complex_close(a: Complex64, b: Complex64) {
        assert_close(a.re, b.re);
        assert_close(a.im, b.im);
    }

    fn assert_matrix_close(a: &DMatrix<f64>, b: &DMatrix<f64>) {
        assert_eq!(a.shape(), b.shape());
        for row in 0..a.nrows() {
            for col in 0..a.ncols() {
                assert_close(a[(row, col)], b[(row, col)]);
            }
        }
    }

    fn assert_complex_matrix_close(a: &DMatrix<Complex64>, b: &DMatrix<Complex64>) {
        assert_eq!(a.shape(), b.shape());
        for row in 0..a.nrows() {
            for col in 0..a.ncols() {
                assert_complex_close(a[(row, col)], b[(row, col)]);
            }
        }
    }

    fn apply_kraus_direct(kraus: &KrausOps, operator: &DMatrix<Complex64>) -> DMatrix<Complex64> {
        let mut output = DMatrix::zeros(operator.nrows(), operator.ncols());
        for k in kraus.operators() {
            output += k * operator * k.adjoint();
        }
        output
    }

    fn direct_superop_from_kraus(kraus: &KrausOps) -> DMatrix<Complex64> {
        let dim = hilbert_dim(kraus.num_qubits()).unwrap();
        let dim_squared = dim * dim;
        let mut matrix = DMatrix::zeros(dim_squared, dim_squared);
        for input_col in 0..dim {
            for input_row in 0..dim {
                let mut input = DMatrix::zeros(dim, dim);
                input[(input_row, input_col)] = Complex64::new(1.0, 0.0);
                let output = apply_kraus_direct(kraus, &input);
                let input_idx = matrix_unit_index(dim, input_row, input_col);
                for output_col in 0..dim {
                    for output_row in 0..dim {
                        let output_idx = matrix_unit_index(dim, output_row, output_col);
                        matrix[(output_idx, input_idx)] = output[(output_row, output_col)];
                    }
                }
            }
        }
        matrix
    }

    fn direct_ptm_from_kraus(kraus: &KrausOps) -> DMatrix<f64> {
        let num_qubits = kraus.num_qubits();
        let basis = pauli_basis_matrices(num_qubits).unwrap();
        let dim = hilbert_dim(num_qubits).unwrap();
        #[allow(clippy::cast_precision_loss)]
        let dim_f = dim as f64;
        let mut matrix = DMatrix::zeros(basis.len(), basis.len());
        for input_idx in 0..basis.len() {
            let evolved = apply_kraus_direct(kraus, &basis[input_idx]);
            for output_idx in 0..basis.len() {
                let entry = trace_complex(&(&basis[output_idx] * &evolved)) / dim_f;
                assert!(
                    entry.im.abs() < 1e-10,
                    "PTM oracle produced complex entry {entry}"
                );
                matrix[(output_idx, input_idx)] = entry.re;
            }
        }
        matrix
    }

    fn direct_matrix_unit_outputs(kraus: &KrausOps) -> Vec<DMatrix<Complex64>> {
        let dim = hilbert_dim(kraus.num_qubits()).unwrap();
        let mut outputs = Vec::with_capacity(dim * dim);
        for input_col in 0..dim {
            for input_row in 0..dim {
                let mut input = DMatrix::zeros(dim, dim);
                input[(input_row, input_col)] = Complex64::new(1.0, 0.0);
                outputs.push(apply_kraus_direct(kraus, &input));
            }
        }
        outputs
    }

    fn assert_ptm_entry(ptm: &Ptm, output: &str, input: &str, expected: f64) {
        let output_idx = labels(ptm.num_qubits())
            .iter()
            .position(|label| label == output)
            .unwrap();
        let input_idx = labels(ptm.num_qubits())
            .iter()
            .position(|label| label == input)
            .unwrap();
        assert_close(ptm.entry(output_idx, input_idx), expected);
    }

    fn labels(num_qubits: usize) -> Vec<String> {
        (0..pauli_basis_len(num_qubits).unwrap())
            .map(|index| basis_label(num_qubits, index).unwrap())
            .collect()
    }

    #[test]
    fn one_qubit_basis_order_is_independent_of_pauli_discriminants() {
        assert_eq!(Pauli::Z as u8, 0b10);
        assert_eq!(Pauli::Y as u8, 0b11);

        assert_eq!(pauli_to_basis_digit(Pauli::I), 0);
        assert_eq!(pauli_to_basis_digit(Pauli::X), 1);
        assert_eq!(pauli_to_basis_digit(Pauli::Y), 2);
        assert_eq!(pauli_to_basis_digit(Pauli::Z), 3);

        assert_eq!(labels(1), ["I", "X", "Y", "Z"]);
    }

    #[test]
    fn two_qubit_basis_order_is_little_endian_lexicographic() {
        assert_eq!(
            labels(2),
            [
                "II", "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX",
                "ZY", "ZZ",
            ]
        );
    }

    #[test]
    fn basis_bitmask_and_index_round_trip() {
        for num_qubits in 1..=3 {
            for index in 0..pauli_basis_len(num_qubits).unwrap() {
                let pauli = basis_bitmask(num_qubits, index).unwrap();
                assert_eq!(basis_index(num_qubits, &pauli).unwrap(), index);
            }
        }
    }

    #[test]
    fn zero_qubit_channel_representations_are_scalar_identity() {
        let ptm = Ptm::identity(0).unwrap();
        assert_eq!(ptm.matrix().shape(), (1, 1));
        assert_close(ptm.entry(0, 0), 1.0);

        let mut probabilities = BTreeMap::new();
        probabilities.insert(PauliBitmaskSmall::identity(), 1.0);
        let pauli_channel = PauliChannel::try_new(0, probabilities).unwrap();
        assert_close(pauli_channel.total_error_rate(), 0.0);
        assert_matrix_close(pauli_channel.to_ptm().unwrap().matrix(), ptm.matrix());

        let kraus = KrausOps::try_new(
            0,
            vec![DMatrix::from_element(1, 1, Complex64::new(1.0, 0.0))],
        )
        .unwrap();
        assert!(kraus.is_trace_preserving());

        let choi = kraus.to_choi().unwrap();
        assert_eq!(choi.matrix().shape(), (1, 1));
        assert_complex_close(choi.matrix()[(0, 0)], Complex64::new(1.0, 0.0));
        assert!(choi.is_cptp());

        let superop = kraus.to_superop().unwrap();
        assert_eq!(superop.num_qubits(), 0);
        assert_complex_close(superop.matrix()[(0, 0)], Complex64::new(1.0, 0.0));

        let chi = kraus.to_chi().unwrap();
        assert_complex_close(chi.matrix()[(0, 0)], Complex64::new(1.0, 0.0));

        let stinespring = kraus.to_stinespring().unwrap();
        assert_eq!(stinespring.environment_dim(), 1);
        assert_complex_close(stinespring.isometry()[(0, 0)], Complex64::new(1.0, 0.0));
    }

    #[test]
    fn pauli_sum_add_scalar_simplify_and_trace() {
        let identity = PauliBitmaskSmall::identity();
        let x0 = PauliBitmaskSmall::x(0);

        let mut a = PauliSum::new(1);
        a.add_term(identity.clone(), Complex64::new(2.0, 0.0))
            .unwrap();
        a.add_term(x0.clone(), Complex64::new(1.0, 0.0)).unwrap();

        let mut b = PauliSum::new(1);
        b.add_term(x0.clone(), Complex64::new(-1.0, 0.0)).unwrap();
        b.add_term(PauliBitmaskSmall::z(0), Complex64::new(0.5, 0.0))
            .unwrap();

        let c = (a + b) * 2.0;
        assert_eq!(c.terms().len(), 2);
        assert_complex_close(*c.terms().get(&identity).unwrap(), Complex64::new(4.0, 0.0));
        assert_complex_close(c.trace().unwrap(), Complex64::new(8.0, 0.0));
    }

    #[test]
    fn pauli_sum_trace_reports_dimension_overflow() {
        let sum = PauliSum::new(usize::MAX);
        assert_eq!(
            sum.trace().unwrap_err(),
            ChannelError::DimensionOverflow {
                num_qubits: usize::MAX
            }
        );
    }

    #[test]
    fn pauli_sum_try_add_reports_qubit_mismatch() {
        let a = PauliSum::new(1);
        let b = PauliSum::new(2);

        assert_eq!(
            a.try_add(b).unwrap_err(),
            ChannelError::QubitCountMismatch {
                expected: 1,
                actual: 2
            }
        );
    }

    #[test]
    fn pauli_sum_try_add_merges_terms_and_drops_cancellations() {
        let identity = PauliBitmaskSmall::identity();
        let x0 = PauliBitmaskSmall::x(0);
        let z0 = PauliBitmaskSmall::z(0);

        let mut a = PauliSum::new(1);
        a.add_term(identity.clone(), Complex64::new(2.0, 0.0))
            .unwrap();
        a.add_term(x0.clone(), Complex64::new(1.0, 0.0)).unwrap();

        let mut b = PauliSum::new(1);
        b.add_term(x0.clone(), Complex64::new(-1.0, 0.0)).unwrap();
        b.add_term(z0.clone(), Complex64::new(0.5, 0.0)).unwrap();

        let sum = a.try_add(b).unwrap();
        assert_eq!(sum.terms().len(), 2);
        assert!(sum.terms().get(&x0).is_none());
        assert_complex_close(
            *sum.terms().get(&identity).unwrap(),
            Complex64::new(2.0, 0.0),
        );
        assert_complex_close(*sum.terms().get(&z0).unwrap(), Complex64::new(0.5, 0.0));
    }

    #[test]
    fn pauli_sum_from_pauli_string_preserves_phase_as_coefficient() {
        let pauli = PauliString::with_phase_and_paulis(
            QuarterPhase::PlusI,
            vec![(Pauli::X, 0.into()), (Pauli::Z, 1.into())],
        );
        let sum = PauliSum::from_pauli_string(2, &pauli).unwrap();
        let label = pauli_string_to_bitmask(2, &pauli).unwrap();
        assert_complex_close(*sum.terms().get(&label).unwrap(), Complex64::new(0.0, 1.0));
        assert_eq!(bitmask_label(2, &label).unwrap(), "ZX");
    }

    #[test]
    fn pauli_string_conjugates_pauli_sum_terms() {
        let mut sum = PauliSum::new(1);
        sum.add_term(PauliBitmaskSmall::x(0), Complex64::new(2.0, 0.0))
            .unwrap();
        sum.add_term(PauliBitmaskSmall::z(0), Complex64::new(3.0, 0.0))
            .unwrap();

        let conjugated = sum.conjugated_by_pauli_string(&PauliString::z(0)).unwrap();
        assert_complex_close(
            *conjugated.terms().get(&PauliBitmaskSmall::x(0)).unwrap(),
            Complex64::new(-2.0, 0.0),
        );
        assert_complex_close(
            *conjugated.terms().get(&PauliBitmaskSmall::z(0)).unwrap(),
            Complex64::new(3.0, 0.0),
        );
    }

    #[test]
    fn pauli_sum_group_commuting_preserves_coefficients() {
        let mut sum = PauliSum::new(2);
        sum.add_term(PauliBitmaskSmall::x(0), Complex64::new(2.0, 0.0))
            .unwrap();
        sum.add_term(PauliBitmaskSmall::z(0), Complex64::new(3.0, 0.0))
            .unwrap();
        sum.add_term(PauliBitmaskSmall::x(1), Complex64::new(5.0, 0.0))
            .unwrap();
        sum.add_term(PauliBitmaskSmall::z(1), Complex64::new(7.0, 0.0))
            .unwrap();

        let groups = sum.group_commuting();
        assert_eq!(groups.len(), 2);
        assert_eq!(
            groups
                .iter()
                .map(|group| group.terms().len())
                .sum::<usize>(),
            4
        );

        for group in &groups {
            for left in group.terms().keys() {
                for right in group.terms().keys() {
                    assert!(left.commutes_with(right));
                }
            }
        }

        let recovered: BTreeMap<_, _> = groups
            .iter()
            .flat_map(|group| {
                group
                    .terms()
                    .iter()
                    .map(|(pauli, coeff)| (pauli.clone(), *coeff))
            })
            .collect();
        assert_eq!(recovered, sum.terms().clone());
    }

    #[test]
    fn qubit_count_errors_fail_construction() {
        let mut terms = BTreeMap::new();
        terms.insert(PauliBitmaskSmall::x(2), Complex64::new(1.0, 0.0));
        let err = PauliSum::try_new(2, terms).unwrap_err();
        assert_eq!(
            err,
            ChannelError::QubitOutOfRange {
                num_qubits: 2,
                qubit: 2
            }
        );
    }

    #[test]
    fn one_qubit_pauli_channel_round_trips_through_diagonal_ptm() {
        let channel = PauliChannel::one_qubit(0.1, 0.2, 0.3).unwrap();
        let diagonal = channel.to_diagonal_ptm().unwrap();

        assert_close(diagonal.fidelity(&PauliBitmaskSmall::identity()), 1.0);
        assert_close(diagonal.fidelity(&PauliBitmaskSmall::x(0)), 0.0);
        assert_close(diagonal.fidelity(&PauliBitmaskSmall::y(0)), 0.2);
        assert_close(diagonal.fidelity(&PauliBitmaskSmall::z(0)), 0.4);

        let recovered = diagonal.to_pauli_channel().unwrap();
        assert_close(recovered.probability(&PauliBitmaskSmall::identity()), 0.4);
        assert_close(recovered.probability(&PauliBitmaskSmall::x(0)), 0.1);
        assert_close(recovered.probability(&PauliBitmaskSmall::y(0)), 0.2);
        assert_close(recovered.probability(&PauliBitmaskSmall::z(0)), 0.3);
        assert_close(recovered.total_error_rate(), 0.6);
    }

    #[test]
    fn two_qubit_pauli_channel_round_trips_through_diagonal_ptm() {
        let mut probabilities = BTreeMap::new();
        probabilities.insert(PauliBitmaskSmall::identity(), 0.7);
        probabilities.insert(PauliBitmaskSmall::x(0), 0.1);
        probabilities.insert(PauliBitmaskSmall::z(1), 0.05);
        probabilities.insert(
            PauliBitmaskSmall::y(0).multiply(&PauliBitmaskSmall::x(1)),
            0.15,
        );

        let channel = PauliChannel::try_new(2, probabilities).unwrap();
        let recovered = channel
            .to_diagonal_ptm()
            .unwrap()
            .to_pauli_channel()
            .unwrap();

        assert_close(recovered.probability(&PauliBitmaskSmall::identity()), 0.7);
        assert_close(recovered.probability(&PauliBitmaskSmall::x(0)), 0.1);
        assert_close(recovered.probability(&PauliBitmaskSmall::z(1)), 0.05);
        assert_close(
            recovered.probability(&PauliBitmaskSmall::y(0).multiply(&PauliBitmaskSmall::x(1))),
            0.15,
        );
    }

    #[test]
    fn pauli_channel_from_pauli_strings_accumulates_sparse_operator_keys() {
        use pecos_core::pauli::{I, X, Z};

        let channel = PauliChannel::from_pauli_strings(
            2,
            [(I(), 0.5), (X(0) & Z(1), 0.2), (X(0) & Z(1), 0.3)],
        )
        .unwrap();

        assert_close(channel.probability(&PauliBitmaskSmall::identity()), 0.5);
        assert_close(
            channel.probability(&PauliBitmaskSmall::x(0).multiply(&PauliBitmaskSmall::z(1))),
            0.5,
        );

        let err = PauliChannel::from_pauli_strings(1, [(Z(2), 1.0)]).unwrap_err();
        assert_eq!(
            err,
            ChannelError::QubitOutOfRange {
                num_qubits: 1,
                qubit: 2
            }
        );
    }

    #[test]
    fn diagonal_ptm_values_are_not_probabilities() {
        let mut fidelities = BTreeMap::new();
        fidelities.insert(PauliBitmaskSmall::identity(), 1.0);
        fidelities.insert(PauliBitmaskSmall::x(0), -0.5);
        fidelities.insert(PauliBitmaskSmall::y(0), 0.25);
        fidelities.insert(PauliBitmaskSmall::z(0), 0.75);

        let diagonal = DiagonalPtm::try_new(1, fidelities.clone()).unwrap();
        assert_close(diagonal.fidelity(&PauliBitmaskSmall::x(0)), -0.5);

        let err = PauliChannel::try_new(1, fidelities).unwrap_err();
        assert!(matches!(err, ChannelError::InvalidProbability { .. }));
    }

    #[test]
    fn pauli_channel_from_pauli_sum_rejects_complex_or_negative_coefficients() {
        let mut complex = PauliSum::new(1);
        complex
            .add_term(PauliBitmaskSmall::identity(), Complex64::new(1.0, 0.1))
            .unwrap();
        assert!(matches!(
            PauliChannel::from_pauli_sum(&complex).unwrap_err(),
            ChannelError::NonRealCoefficient { .. }
        ));

        let mut negative_terms = BTreeMap::new();
        negative_terms.insert(PauliBitmaskSmall::identity(), -0.1);
        negative_terms.insert(PauliBitmaskSmall::x(0), 1.1);
        assert!(matches!(
            PauliChannel::try_new(1, negative_terms).unwrap_err(),
            ChannelError::InvalidProbability { .. }
        ));
    }

    #[test]
    fn dense_ptm_identity_channel_is_identity_matrix() {
        let ptm = Ptm::identity(2).unwrap();
        assert_eq!(ptm.matrix().nrows(), 16);
        assert_eq!(ptm.matrix().ncols(), 16);
        for row in 0..16 {
            for col in 0..16 {
                assert_close(ptm.entry(row, col), if row == col { 1.0 } else { 0.0 });
            }
        }
    }

    #[test]
    fn bit_flip_channel_matches_hand_ptm_and_choi_references() {
        use pecos_core::pauli::{I, X};

        let p = 0.2;
        let channel = PauliChannel::from_pauli_strings(1, [(I(), 1.0 - p), (X(0), p)]).unwrap();
        let ptm = channel.to_ptm().unwrap();

        assert_ptm_entry(&ptm, "I", "I", 1.0);
        assert_ptm_entry(&ptm, "X", "X", 1.0);
        assert_ptm_entry(&ptm, "Y", "Y", 1.0 - 2.0 * p);
        assert_ptm_entry(&ptm, "Z", "Z", 1.0 - 2.0 * p);

        let choi = ptm.to_choi().unwrap();
        let matrix = choi.matrix();
        assert_complex_close(matrix[(0, 0)], Complex64::new(1.0 - p, 0.0));
        assert_complex_close(matrix[(0, 3)], Complex64::new(1.0 - p, 0.0));
        assert_complex_close(matrix[(3, 0)], Complex64::new(1.0 - p, 0.0));
        assert_complex_close(matrix[(3, 3)], Complex64::new(1.0 - p, 0.0));
        assert_complex_close(matrix[(1, 1)], Complex64::new(p, 0.0));
        assert_complex_close(matrix[(1, 2)], Complex64::new(p, 0.0));
        assert_complex_close(matrix[(2, 1)], Complex64::new(p, 0.0));
        assert_complex_close(matrix[(2, 2)], Complex64::new(p, 0.0));
    }

    #[test]
    fn depolarizing_channel_matches_hand_diagonal_ptm_reference() {
        use pecos_core::pauli::{I, X, Y, Z};

        let p = 0.1;
        let channel = PauliChannel::from_pauli_strings(
            1,
            [
                (I(), 1.0 - p),
                (X(0), p / 3.0),
                (Y(0), p / 3.0),
                (Z(0), p / 3.0),
            ],
        )
        .unwrap();
        let ptm = channel.to_ptm().unwrap();
        let non_identity_fidelity = 1.0 - 4.0 * p / 3.0;

        assert_ptm_entry(&ptm, "I", "I", 1.0);
        assert_ptm_entry(&ptm, "X", "X", non_identity_fidelity);
        assert_ptm_entry(&ptm, "Y", "Y", non_identity_fidelity);
        assert_ptm_entry(&ptm, "Z", "Z", non_identity_fidelity);
    }

    #[test]
    fn dense_ptm_unitary_conjugation_known_one_qubit_cliffords() {
        let h = Ptm::from_unitary(&unitary::H(0), 1).unwrap();
        assert_ptm_entry(&h, "I", "I", 1.0);
        assert_ptm_entry(&h, "Z", "X", 1.0);
        assert_ptm_entry(&h, "Y", "Y", -1.0);
        assert_ptm_entry(&h, "X", "Z", 1.0);

        let s = Ptm::from_unitary(&unitary::SZ(0), 1).unwrap();
        assert_ptm_entry(&s, "I", "I", 1.0);
        assert_ptm_entry(&s, "Y", "X", 1.0);
        assert_ptm_entry(&s, "X", "Y", -1.0);
        assert_ptm_entry(&s, "Z", "Z", 1.0);

        let x = Ptm::from_unitary(&unitary::X(0), 1).unwrap();
        assert_ptm_entry(&x, "I", "I", 1.0);
        assert_ptm_entry(&x, "X", "X", 1.0);
        assert_ptm_entry(&x, "Y", "Y", -1.0);
        assert_ptm_entry(&x, "Z", "Z", -1.0);
    }

    #[test]
    fn dense_ptm_qubit_order_matches_unitary_matrix_little_endian() {
        let x0 = Ptm::from_unitary(&(unitary::X(0) & unitary::I(1)), 2).unwrap();
        assert_ptm_entry(&x0, "IZ", "IZ", -1.0);
        assert_ptm_entry(&x0, "ZI", "ZI", 1.0);

        let x1 = Ptm::from_unitary(&(unitary::I(0) & unitary::X(1)), 2).unwrap();
        assert_ptm_entry(&x1, "IZ", "IZ", 1.0);
        assert_ptm_entry(&x1, "ZI", "ZI", -1.0);
    }

    #[test]
    fn invalid_dense_ptm_shape_fails_construction() {
        let err = Ptm::try_new(1, DMatrix::zeros(3, 3)).unwrap_err();
        assert!(matches!(err, ChannelError::InvalidMatrixShape { .. }));
    }

    #[test]
    fn channel_expr_pauli_channel_conversions_handle_common_constructors() {
        let Op::Channel(expr) = op::Depolarizing(0.3, 0) else {
            panic!("expected channel");
        };
        let channel = PauliChannel::from_channel_expr(&expr).unwrap();
        assert_close(channel.probability(&PauliBitmaskSmall::identity()), 0.7);
        assert_close(channel.probability(&PauliBitmaskSmall::x(0)), 0.1);
        assert_close(channel.probability(&PauliBitmaskSmall::y(0)), 0.1);
        assert_close(channel.probability(&PauliBitmaskSmall::z(0)), 0.1);

        let diagonal = DiagonalPtm::from_channel_expr(&expr).unwrap();
        let dense = Ptm::from_channel_expr(&expr).unwrap();
        assert_close(dense.entry(0, 0), 1.0);
        assert_close(
            dense.entry(1, 1),
            diagonal.fidelity(&PauliBitmaskSmall::x(0)),
        );
    }

    #[test]
    fn kraus_unitary_ptm_matches_direct_unitary_ptm() {
        let kraus = KrausOps::from_unitary(&unitary::H(0), 1).unwrap();
        assert_eq!(kraus.num_qubits(), 1);
        assert_eq!(kraus.operators().len(), 1);
        assert!(kraus.is_trace_preserving());

        let from_kraus = kraus.to_ptm().unwrap();
        let direct = Ptm::from_unitary(&unitary::H(0), 1).unwrap();
        assert_matrix_close(from_kraus.matrix(), direct.matrix());
    }

    #[test]
    fn kraus_mixed_unitary_ptm_matches_pauli_channel_ptm() {
        let Op::Channel(expr) = op::Depolarizing(0.3, 0) else {
            panic!("expected channel");
        };
        let kraus = KrausOps::from_channel_expr(&expr).unwrap();
        assert_eq!(kraus.operators().len(), 4);
        assert!(kraus.is_trace_preserving());

        let from_kraus = kraus.to_ptm().unwrap();
        let from_pauli = PauliChannel::from_channel_expr(&expr)
            .unwrap()
            .to_ptm()
            .unwrap();
        assert_matrix_close(from_kraus.matrix(), from_pauli.matrix());
    }

    #[test]
    fn amplitude_damping_kraus_and_ptm_have_known_values() {
        let gamma = 0.25;
        let Op::Channel(expr) = op::AmplitudeDamping(gamma, 0) else {
            panic!("expected channel");
        };
        let kraus = KrausOps::from_channel_expr(&expr).unwrap();
        assert_eq!(kraus.operators().len(), 2);
        assert!(kraus.is_trace_preserving());

        let ptm = Ptm::from_channel_expr(&expr).unwrap();
        assert_ptm_entry(&ptm, "I", "I", 1.0);
        assert_ptm_entry(&ptm, "Z", "I", gamma);
        assert_ptm_entry(&ptm, "X", "X", (1.0 - gamma).sqrt());
        assert_ptm_entry(&ptm, "Y", "Y", (1.0 - gamma).sqrt());
        assert_ptm_entry(&ptm, "Z", "Z", 1.0 - gamma);
    }

    #[test]
    fn phase_damping_kraus_and_ptm_have_known_values() {
        let lambda = 0.36;
        let Op::Channel(expr) = op::PhaseDamping(lambda, 0) else {
            panic!("expected channel");
        };
        let kraus = KrausOps::from_channel_expr(&expr).unwrap();
        assert_eq!(kraus.operators().len(), 2);
        assert!(kraus.is_trace_preserving());

        let ptm = Ptm::from_channel_expr(&expr).unwrap();
        assert_ptm_entry(&ptm, "I", "I", 1.0);
        assert_ptm_entry(&ptm, "X", "X", (1.0 - lambda).sqrt());
        assert_ptm_entry(&ptm, "Y", "Y", (1.0 - lambda).sqrt());
        assert_ptm_entry(&ptm, "Z", "Z", 1.0);
        assert_ptm_entry(&ptm, "Z", "I", 0.0);
    }

    #[test]
    fn kraus_tensor_and_compose_channels_are_trace_preserving() {
        let tensor = ChannelExpr::Tensor(vec![
            ChannelExpr::AmplitudeDamping {
                gamma: 0.2,
                qubit: 0,
            },
            ChannelExpr::PhaseDamping {
                lambda: 0.3,
                qubit: 1,
            },
        ]);
        let tensor_kraus = KrausOps::from_channel_expr(&tensor).unwrap();
        assert_eq!(tensor_kraus.num_qubits(), 2);
        assert_eq!(tensor_kraus.operators().len(), 4);
        assert!(tensor_kraus.is_trace_preserving());

        let compose = ChannelExpr::Compose(vec![
            ChannelExpr::AmplitudeDamping {
                gamma: 0.2,
                qubit: 0,
            },
            ChannelExpr::PhaseDamping {
                lambda: 0.3,
                qubit: 0,
            },
        ]);
        let compose_kraus = KrausOps::from_channel_expr(&compose).unwrap();
        assert_eq!(compose_kraus.num_qubits(), 1);
        assert_eq!(compose_kraus.operators().len(), 4);
        assert!(compose_kraus.is_trace_preserving());
    }

    #[test]
    fn kraus_tensor_rejects_manually_constructed_overlapping_subsystems() {
        let tensor = ChannelExpr::Tensor(vec![
            pecos_core::channel::BitFlip(0.1, 0),
            pecos_core::channel::Dephasing(0.2, 0),
        ]);

        assert!(matches!(
            KrausOps::from_channel_expr(&tensor),
            Err(ChannelError::DuplicateSubsystem { qubit: 0 })
        ));
    }

    #[test]
    fn kraus_tensor_and_compose_reject_empty_manual_exprs() {
        let tensor = ChannelExpr::Tensor(Vec::new());
        let compose = ChannelExpr::Compose(Vec::new());

        assert!(matches!(
            KrausOps::from_channel_expr(&tensor),
            Err(ChannelError::UnsupportedChannelExpr { .. })
        ));
        assert!(matches!(
            KrausOps::from_channel_expr(&compose),
            Err(ChannelError::UnsupportedChannelExpr { .. })
        ));
    }

    #[test]
    fn kraus_from_channel_expr_can_embed_in_larger_system() {
        let expr = pecos_core::channel::BitFlip(0.25, 2);
        let kraus = KrausOps::from_channel_expr_with_num_qubits(&expr, 3).unwrap();

        assert_eq!(kraus.num_qubits(), 3);
        assert!(kraus.is_trace_preserving());
        assert!(matches!(
            KrausOps::from_channel_expr_with_num_qubits(&expr, 2),
            Err(ChannelError::QubitOutOfRange {
                num_qubits: 2,
                qubit: 2
            })
        ));
    }

    #[test]
    fn pauli_channel_conversion_ignores_global_pauli_phase() {
        let pauli = PauliString::from_paulis_with_phase(QuarterPhase::PlusI, &[Pauli::X]);
        let expr = ChannelExpr::Unitary(UnitaryRep::Pauli(pauli));

        let channel = PauliChannel::from_channel_expr(&expr).unwrap();
        assert_close(channel.probability(&PauliBitmaskSmall::x(0)), 1.0);

        let dense = Ptm::from_channel_expr(&expr).unwrap();
        assert_ptm_entry(&dense, "X", "X", 1.0);
        assert_ptm_entry(&dense, "Y", "Y", -1.0);
        assert_ptm_entry(&dense, "Z", "Z", -1.0);
    }

    #[test]
    fn pauli_channel_rejects_non_pauli_channel_conversion() {
        let Op::Channel(expr) = op::AmplitudeDamping(0.1, 0) else {
            panic!("expected channel");
        };
        assert!(matches!(
            PauliChannel::from_channel_expr(&expr).unwrap_err(),
            ChannelError::UnsupportedChannelExpr { .. }
        ));
        assert!(Ptm::from_channel_expr(&expr).is_ok());
    }

    #[test]
    fn kraus_rejects_channels_with_non_kraus_semantics() {
        let Op::Gate(gate) = op::MZ(0) else {
            panic!("expected gate");
        };
        let gate_expr = ChannelExpr::Gate(gate);
        assert!(matches!(
            KrausOps::from_channel_expr(&gate_expr).unwrap_err(),
            ChannelError::UnsupportedChannelExpr { .. }
        ));

        let Op::Channel(erasure) = op::Erasure(0.1, 0) else {
            panic!("expected channel");
        };
        assert!(matches!(
            KrausOps::from_channel_expr(&erasure).unwrap_err(),
            ChannelError::UnsupportedChannelExpr { .. }
        ));

        let Op::Channel(leakage) = op::Leakage(0.1, 0) else {
            panic!("expected channel");
        };
        assert!(matches!(
            KrausOps::from_channel_expr(&leakage).unwrap_err(),
            ChannelError::UnsupportedChannelExpr { .. }
        ));
    }

    #[test]
    fn choi_identity_uses_column_stacked_kraus_convention() {
        let choi = ChoiMatrix::from_unitary(&unitary::I(0), 1).unwrap();
        assert_eq!(choi.num_qubits(), 1);
        assert_eq!(choi.matrix().shape(), (4, 4));
        assert!(choi.is_trace_preserving());
        assert!(choi.is_completely_positive());
        assert!(choi.is_cptp());
        assert!(choi.is_unital());
        assert_complex_close(trace_complex(choi.matrix()), Complex64::new(2.0, 0.0));

        let mut expected = DMatrix::zeros(4, 4);
        expected[(0, 0)] = Complex64::new(1.0, 0.0);
        expected[(0, 3)] = Complex64::new(1.0, 0.0);
        expected[(3, 0)] = Complex64::new(1.0, 0.0);
        expected[(3, 3)] = Complex64::new(1.0, 0.0);
        assert_complex_matrix_close(choi.matrix(), &expected);

        let identity = DMatrix::identity(2, 2);
        assert_complex_matrix_close(&choi.partial_trace_output().unwrap(), &identity);
        assert_complex_matrix_close(&choi.partial_trace_input().unwrap(), &identity);
    }

    #[test]
    fn choi_tomography_helpers_classify_basic_channels() {
        let zero = ChoiMatrix::try_new(1, DMatrix::zeros(4, 4)).unwrap();
        assert!(zero.is_completely_positive());
        assert!(!zero.is_trace_preserving());
        assert!(!zero.is_cptp());
        assert!(!zero.is_unital());

        let Op::Channel(expr) = op::AmplitudeDamping(0.25, 0) else {
            panic!("expected channel");
        };
        let damping = ChoiMatrix::from_channel_expr(&expr).unwrap();
        assert!(damping.is_completely_positive());
        assert!(damping.is_trace_preserving());
        assert!(damping.is_cptp());
        assert!(!damping.is_unital());

        let mut expected_trace_input = DMatrix::zeros(2, 2);
        expected_trace_input[(0, 0)] = Complex64::new(1.25, 0.0);
        expected_trace_input[(1, 1)] = Complex64::new(0.75, 0.0);
        assert_complex_matrix_close(
            &damping.partial_trace_input().unwrap(),
            &expected_trace_input,
        );
    }

    #[test]
    fn choi_transpose_map_is_trace_preserving_unital_but_not_cp() {
        let mut transpose_choi = DMatrix::zeros(4, 4);
        transpose_choi[(0, 0)] = Complex64::new(1.0, 0.0);
        transpose_choi[(1, 2)] = Complex64::new(1.0, 0.0);
        transpose_choi[(2, 1)] = Complex64::new(1.0, 0.0);
        transpose_choi[(3, 3)] = Complex64::new(1.0, 0.0);
        let choi = ChoiMatrix::try_new(1, transpose_choi).unwrap();

        assert!(choi.is_trace_preserving());
        assert!(choi.is_unital());
        assert!(!choi.is_completely_positive());
        assert!(!choi.is_cptp());
    }

    #[test]
    fn choi_ptm_round_trip_for_depolarizing_channel() {
        let Op::Channel(expr) = op::Depolarizing(0.3, 0) else {
            panic!("expected channel");
        };
        let ptm = Ptm::from_channel_expr(&expr).unwrap();
        let choi = ptm.to_choi().unwrap();
        assert!(choi.is_trace_preserving());

        let recovered = choi.to_ptm().unwrap();
        assert_matrix_close(recovered.matrix(), ptm.matrix());
    }

    #[test]
    fn choi_round_trips_amplitude_damping_through_kraus_and_ptm() {
        let Op::Channel(expr) = op::AmplitudeDamping(0.25, 0) else {
            panic!("expected channel");
        };
        let kraus = KrausOps::from_channel_expr(&expr).unwrap();
        let choi = kraus.to_choi().unwrap();
        assert!(choi.is_trace_preserving());

        let ptm_from_kraus = kraus.to_ptm().unwrap();
        let ptm_from_choi = choi.to_ptm().unwrap();
        assert_matrix_close(ptm_from_choi.matrix(), ptm_from_kraus.matrix());

        let recovered_kraus = choi.to_kraus().unwrap();
        assert!(recovered_kraus.is_trace_preserving());
        let recovered_ptm = recovered_kraus.to_ptm().unwrap();
        assert_matrix_close(recovered_ptm.matrix(), ptm_from_kraus.matrix());
    }

    #[test]
    fn ptm_to_kraus_round_trip_for_unitary_channel() {
        let ptm = Ptm::from_unitary(&unitary::H(0), 1).unwrap();
        let kraus = ptm.to_kraus().unwrap();
        assert!(kraus.is_trace_preserving());

        let recovered = kraus.to_ptm().unwrap();
        assert_matrix_close(recovered.matrix(), ptm.matrix());
    }

    #[test]
    fn superop_identity_round_trips_through_channel_representations() {
        let kraus = KrausOps::from_unitary(&unitary::I(0), 1).unwrap();
        let superop = SuperOp::from_kraus(&kraus).unwrap();
        let identity = DMatrix::<Complex64>::identity(4, 4);
        assert_complex_matrix_close(superop.matrix(), &identity);

        let choi = superop.to_choi().unwrap();
        let expected_choi = ChoiMatrix::from_kraus(&kraus).unwrap();
        assert_complex_matrix_close(choi.matrix(), expected_choi.matrix());

        let ptm = superop.to_ptm().unwrap();
        assert_matrix_close(ptm.matrix(), Ptm::identity(1).unwrap().matrix());
    }

    #[test]
    fn superop_choi_round_trip_for_amplitude_damping() {
        let Op::Channel(expr) = op::AmplitudeDamping(0.25, 0) else {
            panic!("expected channel");
        };
        let kraus = KrausOps::from_channel_expr(&expr).unwrap();
        let choi = kraus.to_choi().unwrap();

        let superop = SuperOp::from_choi(&choi).unwrap();
        let recovered = superop.to_choi().unwrap();

        assert_complex_matrix_close(recovered.matrix(), choi.matrix());
        assert_matrix_close(
            superop.to_ptm().unwrap().matrix(),
            kraus.to_ptm().unwrap().matrix(),
        );
    }

    #[test]
    fn superop_ptm_round_trip_for_depolarizing() {
        let Op::Channel(expr) = op::Depolarizing(0.3, 0) else {
            panic!("expected channel");
        };
        let ptm = Ptm::from_channel_expr(&expr).unwrap();

        let superop = SuperOp::from_ptm(&ptm).unwrap();
        let recovered = superop.to_ptm().unwrap();

        assert_matrix_close(recovered.matrix(), ptm.matrix());
    }

    #[test]
    fn random_channels_match_direct_kraus_oracles_for_small_systems() {
        for (num_qubits, num_kraus, seed) in [(1, 1, 11), (1, 3, 12), (2, 2, 21), (2, 4, 22)] {
            let mut rng = PecosRng::seed_from_u64(seed);
            let kraus = random_quantum_channel(&mut rng, num_qubits, num_kraus).unwrap();
            assert!(kraus.is_trace_preserving());

            let superop = kraus.to_superop().unwrap();
            assert_complex_matrix_close(superop.matrix(), &direct_superop_from_kraus(&kraus));

            let ptm = kraus.to_ptm().unwrap();
            assert_matrix_close(ptm.matrix(), &direct_ptm_from_kraus(&kraus));

            let choi = kraus.to_choi().unwrap();
            let outputs = direct_matrix_unit_outputs(&kraus);
            let reconstructed = ChoiMatrix::from_matrix_unit_outputs(num_qubits, &outputs).unwrap();
            assert_complex_matrix_close(choi.matrix(), reconstructed.matrix());

            for input in matrix_unit_basis(num_qubits).unwrap() {
                assert_complex_matrix_close(
                    &choi.apply_to_operator(&input).unwrap(),
                    &apply_kraus_direct(&kraus, &input),
                );
            }

            let stinespring_superop = kraus.to_stinespring().unwrap().to_superop().unwrap();
            assert_complex_matrix_close(stinespring_superop.matrix(), superop.matrix());
        }
    }

    #[test]
    fn three_qubit_random_channel_matches_direct_oracles() {
        let mut rng = PecosRng::seed_from_u64(1234);
        let kraus = random_quantum_channel(&mut rng, 3, 2).unwrap();
        assert!(kraus.is_trace_preserving());

        let superop = kraus.to_superop().unwrap();
        assert_eq!(superop.matrix().shape(), (64, 64));
        assert_complex_matrix_close(superop.matrix(), &direct_superop_from_kraus(&kraus));

        let ptm = kraus.to_ptm().unwrap();
        assert_eq!(ptm.matrix().shape(), (64, 64));
        assert_matrix_close(ptm.matrix(), &direct_ptm_from_kraus(&kraus));

        let choi = kraus.to_choi().unwrap();
        assert_eq!(choi.matrix().shape(), (64, 64));
        assert!(choi.is_cptp());
        assert_complex_matrix_close(
            &choi.partial_trace_output().unwrap(),
            &DMatrix::identity(8, 8),
        );
    }

    #[test]
    fn superop_compose_and_tensor_follow_matrix_semantics() {
        let x = KrausOps::from_unitary(&unitary::X(0), 1)
            .unwrap()
            .to_superop()
            .unwrap();
        let xx = x.compose(&x).unwrap();
        assert_complex_matrix_close(xx.matrix(), &DMatrix::<Complex64>::identity(4, 4));

        let identity = KrausOps::from_unitary(&unitary::I(0), 1)
            .unwrap()
            .to_superop()
            .unwrap();
        let tensor = identity.tensor(&identity).unwrap();
        assert_eq!(tensor.num_qubits(), 2);
        assert_complex_matrix_close(tensor.matrix(), &DMatrix::<Complex64>::identity(16, 16));

        assert_eq!(
            identity.compose(&tensor).unwrap_err(),
            ChannelError::QubitCountMismatch {
                expected: 1,
                actual: 2
            }
        );
    }

    #[test]
    fn stinespring_try_new_rejects_non_isometric_matrix() {
        let err = Stinespring::try_new(
            1,
            DMatrix::from_diagonal_element(2, 2, Complex64::new(2.0, 0.0)),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            ChannelError::DecompositionFailed { reason } if reason.contains("not an isometry")
        ));
    }

    #[test]
    fn chi_matrix_is_diagonal_for_pauli_mixture() {
        let expr = ChannelExpr::MixedUnitary(vec![(0.7, unitary::I(0)), (0.3, unitary::X(0))]);
        let chi = ChiMatrix::from_channel_expr(&expr).unwrap();
        let identity = basis_index(1, &PauliBitmaskSmall::identity()).unwrap();
        let x = basis_index(1, &PauliBitmaskSmall::x(0)).unwrap();

        assert_complex_close(chi.matrix()[(identity, identity)], Complex64::new(0.7, 0.0));
        assert_complex_close(chi.matrix()[(x, x)], Complex64::new(0.3, 0.0));
        for row in 0..chi.matrix().nrows() {
            for col in 0..chi.matrix().ncols() {
                if (row, col) != (identity, identity) && (row, col) != (x, x) {
                    assert!(chi.matrix()[(row, col)].norm() < 1e-10);
                }
            }
        }

        let recovered = chi.to_ptm().unwrap();
        let expected = Ptm::from_channel_expr(&expr).unwrap();
        assert_matrix_close(recovered.matrix(), expected.matrix());
    }

    #[test]
    fn chi_matrix_amplitude_damping_has_off_diagonal_terms_and_matches_ptm() {
        let Op::Channel(expr) = op::AmplitudeDamping(0.25, 0) else {
            panic!("expected channel");
        };
        let kraus = KrausOps::from_channel_expr(&expr).unwrap();
        let chi = ChiMatrix::from_kraus(&kraus).unwrap();

        let has_off_diagonal = (0..chi.matrix().nrows()).any(|row| {
            (0..chi.matrix().ncols())
                .any(|col| row != col && chi.matrix()[(row, col)].norm() > 1e-10)
        });
        assert!(
            has_off_diagonal,
            "amplitude damping should have off-diagonal chi entries"
        );

        let recovered = chi.to_ptm().unwrap();
        let expected = Ptm::from_channel_expr(&expr).unwrap();
        assert_matrix_close(recovered.matrix(), expected.matrix());
    }

    #[test]
    fn chi_choi_round_trip_for_amplitude_damping() {
        let Op::Channel(expr) = op::AmplitudeDamping(0.25, 0) else {
            panic!("expected channel");
        };
        let kraus = KrausOps::from_channel_expr(&expr).unwrap();
        let chi = ChiMatrix::from_kraus(&kraus).unwrap();

        let recovered = chi.to_choi().unwrap().to_chi().unwrap();

        assert_complex_matrix_close(recovered.matrix(), chi.matrix());
    }

    #[test]
    fn stinespring_round_trips_trace_preserving_kraus_channels() {
        let Op::Channel(expr) = op::AmplitudeDamping(0.25, 0) else {
            panic!("expected channel");
        };
        let kraus = KrausOps::from_channel_expr(&expr).unwrap();
        let stinespring = kraus.to_stinespring().unwrap();
        assert_eq!(stinespring.num_qubits(), 1);
        assert_eq!(stinespring.environment_dim(), kraus.operators().len());

        let recovered = stinespring.to_kraus().unwrap();
        let expected_choi = kraus.to_choi().unwrap();
        let recovered_choi = recovered.to_choi().unwrap();
        assert_complex_matrix_close(recovered_choi.matrix(), expected_choi.matrix());
    }

    #[test]
    fn invalid_choi_shape_fails_construction() {
        let err = ChoiMatrix::try_new(1, DMatrix::zeros(2, 2)).unwrap_err();
        assert!(matches!(err, ChannelError::InvalidMatrixShape { .. }));
    }

    #[test]
    fn choi_to_kraus_rejects_non_positive_choi_matrix() {
        let mut matrix = DMatrix::zeros(4, 4);
        matrix[(0, 0)] = Complex64::new(1.0, 0.0);
        matrix[(3, 3)] = Complex64::new(-1.0, 0.0);
        let choi = ChoiMatrix::try_new(1, matrix).unwrap();

        assert!(matches!(
            choi.to_kraus().unwrap_err(),
            ChannelError::DecompositionFailed { .. }
        ));
    }

    #[test]
    fn choi_to_kraus_rejects_invalid_tolerance() {
        let choi = ChoiMatrix::from_unitary(&unitary::I(0), 1).unwrap();

        assert!(matches!(
            choi.to_kraus_with_tolerance(f64::NAN).unwrap_err(),
            ChannelError::DecompositionFailed { .. }
        ));
        assert!(matches!(
            choi.to_kraus_with_tolerance(-1e-12).unwrap_err(),
            ChannelError::DecompositionFailed { .. }
        ));
    }

    #[test]
    fn matrix_unit_basis_uses_column_stacked_order() {
        let basis = matrix_unit_basis(1).unwrap();
        assert_eq!(basis.len(), 4);
        for (idx, (row, col)) in [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().enumerate() {
            let mut expected = DMatrix::zeros(2, 2);
            expected[(row, col)] = Complex64::new(1.0, 0.0);
            assert_complex_matrix_close(&basis[idx], &expected);
        }
    }

    #[test]
    fn process_tomography_design_exposes_matrix_unit_order() {
        let design = ProcessTomographyDesign::matrix_unit(1).unwrap();
        assert_eq!(design.num_qubits(), 1);
        assert_eq!(design.dim(), 2);
        assert_eq!(design.num_inputs(), 4);
        assert_eq!(design.input_index(0, 0).unwrap(), 0);
        assert_eq!(design.input_index(1, 0).unwrap(), 1);
        assert_eq!(design.input_index(0, 1).unwrap(), 2);
        assert_eq!(design.input_index(1, 1).unwrap(), 3);

        let metadata = design.input_metadata_all();
        assert_eq!(
            metadata,
            vec![
                MatrixUnitTomographyInput {
                    index: 0,
                    row: 0,
                    col: 0
                },
                MatrixUnitTomographyInput {
                    index: 1,
                    row: 1,
                    col: 0
                },
                MatrixUnitTomographyInput {
                    index: 2,
                    row: 0,
                    col: 1
                },
                MatrixUnitTomographyInput {
                    index: 3,
                    row: 1,
                    col: 1
                },
            ]
        );

        let from_design = design.input_operators();
        let from_free_function = matrix_unit_basis(1).unwrap();
        for (actual, expected) in from_design.iter().zip(from_free_function.iter()) {
            assert_complex_matrix_close(actual, expected);
        }
    }

    #[test]
    fn process_tomography_design_reconstructs_channel_outputs() {
        let Op::Channel(expr) = op::AmplitudeDamping(0.25, 0) else {
            panic!("expected channel");
        };
        let expected = ChoiMatrix::from_channel_expr(&expr).unwrap();
        let design = ProcessTomographyDesign::matrix_unit(1).unwrap();
        let outputs = design.simulate_outputs(&expected).unwrap();
        let reconstructed = design.reconstruct_choi(&outputs).unwrap();

        assert_complex_matrix_close(reconstructed.matrix(), expected.matrix());
        assert!(reconstructed.is_cptp());
        assert!(!reconstructed.is_unital());
    }

    #[test]
    fn process_tomography_design_reconstructs_two_qubit_tensor_channel() {
        let tensor = ChannelExpr::Tensor(vec![
            pecos_core::channel::AmplitudeDamping(0.25, 0),
            pecos_core::channel::PhaseDamping(0.4, 1),
        ]);
        let kraus = KrausOps::from_channel_expr(&tensor).unwrap();
        let expected = kraus.to_choi().unwrap();
        let design = ProcessTomographyDesign::matrix_unit(2).unwrap();

        let outputs = direct_matrix_unit_outputs(&kraus);
        let reconstructed = design.reconstruct_choi(&outputs).unwrap();
        assert_complex_matrix_close(reconstructed.matrix(), expected.matrix());

        let simulated_outputs = design.simulate_outputs(&expected).unwrap();
        for (direct, simulated) in outputs.iter().zip(simulated_outputs.iter()) {
            assert_complex_matrix_close(simulated, direct);
        }
    }

    #[test]
    fn process_tomography_design_rejects_invalid_inputs() {
        let design = ProcessTomographyDesign::matrix_unit(1).unwrap();
        assert!(matches!(
            design.input_metadata(4).unwrap_err(),
            ChannelError::TomographyInputOutOfRange {
                num_inputs: 4,
                index: 4
            }
        ));
        assert!(matches!(
            design.input_index(2, 0).unwrap_err(),
            ChannelError::MatrixUnitOutOfRange {
                dim: 2,
                row: 2,
                col: 0
            }
        ));

        let two_qubit_channel = ChoiMatrix::from_unitary(&unitary::I(0), 2).unwrap();
        assert!(matches!(
            design.simulate_outputs(&two_qubit_channel).unwrap_err(),
            ChannelError::QubitCountMismatch {
                expected: 1,
                actual: 2
            }
        ));
    }

    #[test]
    fn choi_reconstructs_identity_from_matrix_unit_outputs() {
        let inputs = matrix_unit_basis(1).unwrap();
        let reconstructed = ChoiMatrix::from_matrix_unit_outputs(1, &inputs).unwrap();
        let expected = ChoiMatrix::from_unitary(&unitary::I(0), 1).unwrap();

        assert_complex_matrix_close(reconstructed.matrix(), expected.matrix());
        assert!(reconstructed.is_cptp());
        assert!(reconstructed.is_unital());
    }

    #[test]
    fn choi_reconstructs_amplitude_damping_from_matrix_unit_outputs() {
        let Op::Channel(expr) = op::AmplitudeDamping(0.25, 0) else {
            panic!("expected channel");
        };
        let expected = ChoiMatrix::from_channel_expr(&expr).unwrap();
        let outputs: Vec<DMatrix<Complex64>> = matrix_unit_basis(1)
            .unwrap()
            .iter()
            .map(|operator| expected.apply_to_operator(operator).unwrap())
            .collect();

        let reconstructed = ChoiMatrix::from_matrix_unit_outputs(1, &outputs).unwrap();
        assert_complex_matrix_close(reconstructed.matrix(), expected.matrix());
        assert!(reconstructed.is_cptp());
        assert!(!reconstructed.is_unital());
    }

    #[test]
    fn choi_tomography_rejects_bad_sample_count_and_shape() {
        let err = ChoiMatrix::from_matrix_unit_outputs(1, &[]).unwrap_err();
        assert_eq!(
            err,
            ChannelError::InvalidTomographySampleCount {
                expected: 4,
                actual: 0
            }
        );

        let bad_outputs = vec![DMatrix::zeros(2, 2); 3]
            .into_iter()
            .chain(std::iter::once(DMatrix::zeros(3, 3)))
            .collect::<Vec<_>>();
        assert!(matches!(
            ChoiMatrix::from_matrix_unit_outputs(1, &bad_outputs).unwrap_err(),
            ChannelError::InvalidMatrixShape { .. }
        ));
    }

    #[test]
    fn partial_trace_of_bell_state_is_maximally_mixed() {
        let half = Complex64::new(0.5, 0.0);
        let mut rho = DMatrix::zeros(4, 4);
        rho[(0, 0)] = half;
        rho[(0, 3)] = half;
        rho[(3, 0)] = half;
        rho[(3, 3)] = half;

        let reduced = partial_trace(&rho, 2, &[1]).unwrap();
        assert_eq!(reduced.shape(), (2, 2));
        assert_complex_close(reduced[(0, 0)], half);
        assert_complex_close(reduced[(1, 1)], half);
        assert_complex_close(reduced[(0, 1)], Complex64::new(0.0, 0.0));
        assert_complex_close(reduced[(1, 0)], Complex64::new(0.0, 0.0));
    }

    #[test]
    fn partial_trace_respects_little_endian_qubit_ordering() {
        let mut rho = DMatrix::zeros(4, 4);
        rho[(1, 1)] = Complex64::new(1.0, 0.0); // |q1=0, q0=1><...|

        let keep_q0 = partial_trace(&rho, 2, &[1]).unwrap();
        assert_complex_close(keep_q0[(0, 0)], Complex64::new(0.0, 0.0));
        assert_complex_close(keep_q0[(1, 1)], Complex64::new(1.0, 0.0));

        let keep_q1 = partial_trace(&rho, 2, &[0]).unwrap();
        assert_complex_close(keep_q1[(0, 0)], Complex64::new(1.0, 0.0));
        assert_complex_close(keep_q1[(1, 1)], Complex64::new(0.0, 0.0));
    }

    #[test]
    fn random_density_matrix_is_normalized_hermitian_and_reproducible() {
        let mut rng = PecosRng::seed_from_u64(123);
        let rho = random_density_matrix(&mut rng, 2).unwrap();
        assert_eq!(rho.shape(), (4, 4));
        assert_complex_close(trace_complex(&rho), Complex64::new(1.0, 0.0));
        assert_complex_matrix_close(&rho, &rho.adjoint());

        let mut same_seed = PecosRng::seed_from_u64(123);
        let same = random_density_matrix(&mut same_seed, 2).unwrap();
        assert_complex_matrix_close(&rho, &same);

        let mut different_seed = PecosRng::seed_from_u64(456);
        let different = random_density_matrix(&mut different_seed, 2).unwrap();
        assert_ne!(rho, different);
    }

    #[test]
    fn random_density_matrix_rank_one_is_pure() {
        let mut rng = PecosRng::seed_from_u64(123);
        let rho = random_density_matrix_with_rank(&mut rng, 2, 1).unwrap();
        let purity = trace_complex(&(&rho * &rho)).re;
        assert_close(purity, 1.0);

        assert!(matches!(
            random_density_matrix_with_rank(&mut rng, 1, 0).unwrap_err(),
            ChannelError::EmptyKrausSet
        ));
    }

    #[test]
    fn random_density_matrix_three_qubit_rank_limited_is_stable() {
        let mut rng = PecosRng::seed_from_u64(789);
        let rho = random_density_matrix_with_rank(&mut rng, 3, 3).unwrap();
        assert_eq!(rho.shape(), (8, 8));
        assert_complex_close(trace_complex(&rho), Complex64::new(1.0, 0.0));
        assert_complex_matrix_close(&rho, &rho.adjoint());

        let purity = trace_complex(&(&rho * &rho));
        assert!(purity.im.abs() < 1e-10);
        assert!(
            purity.re > 1.0 / 8.0 - 1e-10 && purity.re <= 1.0 + 1e-10,
            "unexpected 3-qubit density-matrix purity: {purity}"
        );
    }

    #[test]
    fn random_quantum_channel_is_cptp_and_reproducible() {
        let mut rng = PecosRng::seed_from_u64(123);
        let channel = random_quantum_channel(&mut rng, 1, 3).unwrap();
        assert_eq!(channel.operators().len(), 3);
        assert!(channel.is_trace_preserving());
        assert!(channel.to_choi().unwrap().is_cptp());

        let mut same_seed = PecosRng::seed_from_u64(123);
        let same = random_quantum_channel(&mut same_seed, 1, 3).unwrap();
        for (left, right) in channel.operators().iter().zip(same.operators()) {
            assert_complex_matrix_close(left, right);
        }

        let mut different_seed = PecosRng::seed_from_u64(456);
        let different = random_quantum_channel(&mut different_seed, 1, 3).unwrap();
        assert_ne!(channel, different);
    }

    #[test]
    fn random_quantum_channel_rejects_zero_kraus_count() {
        let mut rng = PecosRng::seed_from_u64(123);
        assert!(matches!(
            random_quantum_channel(&mut rng, 1, 0).unwrap_err(),
            ChannelError::EmptyKrausSet
        ));
    }

    #[test]
    fn random_helpers_return_values_from_expected_sets() {
        let mut rng = PecosRng::seed_from_u64(123);
        let pauli = random_pauli(&mut rng, 5);
        assert!(pauli.qubits().into_iter().all(|q| q < 5));

        let c1 = random_1q_clifford(&mut rng);
        assert!(Clifford::all_1q().contains(&c1));

        let c2 = random_2q_clifford(&mut rng);
        assert!(Clifford::all_2q().contains(&c2));

        let c = random_clifford(&mut rng);
        assert!(Clifford::all().contains(&c));
    }
}
