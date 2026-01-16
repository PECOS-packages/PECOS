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

//! Gate expression algebra for quantum circuits.
//!
//! This module provides a lazy expression tree for building and manipulating
//! quantum gate sequences with algebraic simplification.
//!
//! # Representation Hierarchy
//!
//! 1. **Rotation gates**: `exp(-i θ/2 P)` where P is a Pauli - covers most gates
//! 2. **Control gates**: CX, CZ, SWAP, etc. - stay as named gates
//!
//! # Operators
//!
//! - `&` - Tensor product (operators on different qubits)
//! - `*` - Composition (matrix multiplication order: A * B means apply B then A)
//! - `.dg()` - Adjoint (Hermitian conjugate)
//!
//! # Examples
//!
//! ```
//! use pecos_core::operator::*;
//! use pecos_core::Angle64;
//!
//! // Build a circuit: H on q0, then CX(0,1), then T on q1
//! let circuit = T(1) * CX(0, 1) * H(0);
//!
//! // Check if it's Clifford
//! assert!(!circuit.is_clifford());  // T is not Clifford
//!
//! // Clifford circuit
//! let cliff = CX(0, 1) * H(0);
//! assert!(cliff.is_clifford());
//!
//! // Tensor product
//! let two_qubit = X(0) & Z(1);
//!
//! // Adjoint
//! let inv = circuit.dg();
//! ```

use crate::gate_type::GateType;
use crate::pauli::PauliOperator;
use crate::phase::Phase;
use crate::{Angle64, PauliString, QuarterPhase, QubitId};
use smallvec::SmallVec;
use std::ops::{BitAnd, Mul, Neg};

// ============================================================================
// Phase macros for exact arithmetic
// ============================================================================

/// Creates a `PhaseValue` from a pi-based expression for use with operators.
///
/// This is a convenience wrapper around `angle!` that returns a `PhaseValue`
/// which can be directly multiplied with gate expressions.
///
/// # Examples
///
/// ```
/// use pecos_core::{phase, Angle64};
/// use pecos_core::operator::X;
///
/// // e^{iπ/4} * X - exact, no floating point
/// let op = phase!(pi / 4) * X(0);
///
/// // e^{iπ/2} * X = i * X
/// let op = phase!(pi / 2) * X(0);
///
/// // e^{i * 2π/3} * X
/// let op = phase!(2 * pi / 3) * X(0);
/// ```
#[macro_export]
macro_rules! phase {
    ($($tokens:tt)*) => {
        $crate::operator::PhaseValue($crate::angle!($($tokens)*))
    };
}

/// Creates a `PhaseValue` from a turn-based fraction for use with operators.
///
/// This is a convenience wrapper around `turn!` that returns a `PhaseValue`
/// which can be directly multiplied with gate expressions.
///
/// # Examples
///
/// ```
/// use pecos_core::phase_turn;
/// use pecos_core::operator::X;
///
/// // T gate phase: e^{i * 2π/8} = e^{iπ/4}
/// let op = phase_turn!(1 / 8) * X(0);
///
/// // SZ gate phase: e^{i * 2π/4} = e^{iπ/2} = i
/// let op = phase_turn!(1 / 4) * X(0);
///
/// // Third of a turn: e^{i * 2π/3}
/// let op = phase_turn!(1 / 3) * X(0);
/// ```
#[macro_export]
macro_rules! phase_turn {
    ($($tokens:tt)*) => {
        $crate::operator::PhaseValue($crate::turn!($($tokens)*))
    };
}

/// Rotation gate types - gates parameterized by an angle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RotationType {
    /// Rotation around X axis: exp(-i θ/2 X)
    RX,
    /// Rotation around Y axis: exp(-i θ/2 Y)
    RY,
    /// Rotation around Z axis: exp(-i θ/2 Z)
    RZ,
    /// Two-qubit XX rotation: exp(-i θ/2 X⊗X)
    RXX,
    /// Two-qubit YY rotation: exp(-i θ/2 Y⊗Y)
    RYY,
    /// Two-qubit ZZ rotation: exp(-i θ/2 Z⊗Z)
    RZZ,
}

impl RotationType {
    /// Returns the number of qubits this rotation acts on.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        match self {
            Self::RX | Self::RY | Self::RZ => 1,
            Self::RXX | Self::RYY | Self::RZZ => 2,
        }
    }

    /// Returns the corresponding `GateType` for this rotation.
    #[must_use]
    pub fn to_gate_type(&self) -> GateType {
        match self {
            Self::RX => GateType::RX,
            Self::RY => GateType::RY,
            Self::RZ => GateType::RZ,
            Self::RXX => GateType::RXX,
            Self::RYY => GateType::RYY,
            Self::RZZ => GateType::RZZ,
        }
    }
}

// ============================================================================
// Commutativity
// ============================================================================

/// Result of checking whether two operators commute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Commutativity {
    /// Operators commute: AB = BA
    Commutes,
    /// Operators anti-commute: AB = -BA
    AntiCommutes,
    /// Commutativity cannot be determined (non-Pauli operators)
    Unknown,
}

// ============================================================================
// Qubit target types for polymorphic gate constructors
// ============================================================================

/// Wrapper for qubit targets that can be a single qubit or multiple qubits.
///
/// Enables pluralized gate functions to accept various qubit collections:
/// ```
/// use pecos_core::operator::*;
/// use pecos_core::QubitId;
///
/// // Multiple qubits via Xs - equivalent to X(0) & X(2) & X(5)
/// let x_multi = Xs([0, 2, 5]);
///
/// // Also works with QubitId arrays
/// let x_multi = Xs([QubitId(0), QubitId(2)]);
/// ```
#[derive(Debug, Clone)]
pub struct Qubits(SmallVec<[QubitId; 4]>);

impl From<usize> for Qubits {
    fn from(q: usize) -> Self {
        Qubits(smallvec::smallvec![QubitId(q)])
    }
}

impl From<QubitId> for Qubits {
    fn from(q: QubitId) -> Self {
        Qubits(smallvec::smallvec![q])
    }
}

impl<const N: usize> From<[usize; N]> for Qubits {
    fn from(qs: [usize; N]) -> Self {
        Qubits(qs.into_iter().map(QubitId).collect())
    }
}

impl<const N: usize> From<[QubitId; N]> for Qubits {
    fn from(qs: [QubitId; N]) -> Self {
        Qubits(qs.into_iter().collect())
    }
}

impl From<&[usize]> for Qubits {
    fn from(qs: &[usize]) -> Self {
        Qubits(qs.iter().copied().map(QubitId).collect())
    }
}

impl From<&[QubitId]> for Qubits {
    fn from(qs: &[QubitId]) -> Self {
        Qubits(qs.iter().copied().collect())
    }
}

impl From<Vec<usize>> for Qubits {
    fn from(qs: Vec<usize>) -> Self {
        Qubits(qs.into_iter().map(QubitId).collect())
    }
}

impl From<Vec<QubitId>> for Qubits {
    fn from(qs: Vec<QubitId>) -> Self {
        Qubits(qs.into_iter().collect())
    }
}

impl From<std::ops::Range<usize>> for Qubits {
    fn from(range: std::ops::Range<usize>) -> Self {
        assert!(!range.is_empty(), "empty range not allowed for Qubits");
        Qubits(range.map(QubitId).collect())
    }
}

impl From<std::ops::RangeInclusive<usize>> for Qubits {
    fn from(range: std::ops::RangeInclusive<usize>) -> Self {
        assert!(!range.is_empty(), "empty range not allowed for Qubits");
        Qubits(range.map(QubitId).collect())
    }
}

impl Qubits {
    /// Returns true if there are no qubits.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns the number of qubits.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns the qubits as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[QubitId] {
        &self.0
    }

    /// Applies a gate function to each qubit and returns the result.
    /// For a single qubit, returns the gate directly.
    /// For multiple qubits, returns a Tensor of the gates.
    #[must_use]
    pub fn apply<F>(self, gate_fn: F) -> Operator
    where
        F: Fn(usize) -> Operator,
    {
        match self.0.len() {
            0 => Operator::Pauli(PauliString::default()), // Identity
            1 => gate_fn(self.0[0].0),
            _ => Operator::Tensor(self.0.iter().map(|q| gate_fn(q.0)).collect()),
        }
    }
}

/// Wrapper for qubit pairs used by pluralized two-qubit gates.
///
/// ```
/// use pecos_core::operator::*;
/// use pecos_core::QubitId;
///
/// // Multiple CX gates via CXs
/// let cx_multi = CXs([(0, 1), (2, 3)]);
///
/// // Also works with QubitId pairs
/// let cx_multi = CXs([(QubitId(0), QubitId(1))]);
/// ```
#[derive(Debug, Clone)]
pub struct QubitPairs(SmallVec<[(QubitId, QubitId); 2]>);

impl From<(usize, usize)> for QubitPairs {
    fn from(pair: (usize, usize)) -> Self {
        QubitPairs(smallvec::smallvec![(QubitId(pair.0), QubitId(pair.1))])
    }
}

impl From<(QubitId, QubitId)> for QubitPairs {
    fn from(pair: (QubitId, QubitId)) -> Self {
        QubitPairs(smallvec::smallvec![pair])
    }
}

impl<const N: usize> From<[(usize, usize); N]> for QubitPairs {
    fn from(pairs: [(usize, usize); N]) -> Self {
        QubitPairs(
            pairs
                .into_iter()
                .map(|(a, b)| (QubitId(a), QubitId(b)))
                .collect(),
        )
    }
}

impl<const N: usize> From<[(QubitId, QubitId); N]> for QubitPairs {
    fn from(pairs: [(QubitId, QubitId); N]) -> Self {
        QubitPairs(pairs.into_iter().collect())
    }
}

impl From<&[(usize, usize)]> for QubitPairs {
    fn from(pairs: &[(usize, usize)]) -> Self {
        QubitPairs(
            pairs
                .iter()
                .map(|&(a, b)| (QubitId(a), QubitId(b)))
                .collect(),
        )
    }
}

impl From<&[(QubitId, QubitId)]> for QubitPairs {
    fn from(pairs: &[(QubitId, QubitId)]) -> Self {
        QubitPairs(pairs.iter().copied().collect())
    }
}

impl From<Vec<(usize, usize)>> for QubitPairs {
    fn from(pairs: Vec<(usize, usize)>) -> Self {
        QubitPairs(
            pairs
                .into_iter()
                .map(|(a, b)| (QubitId(a), QubitId(b)))
                .collect(),
        )
    }
}

impl From<Vec<(QubitId, QubitId)>> for QubitPairs {
    fn from(pairs: Vec<(QubitId, QubitId)>) -> Self {
        QubitPairs(pairs.into_iter().collect())
    }
}

impl QubitPairs {
    /// Returns true if there are no pairs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns the number of pairs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Applies a gate function to each pair and returns the result.
    /// For a single pair, returns the gate directly.
    /// For multiple pairs, returns a Tensor of the gates.
    #[must_use]
    pub fn apply<F>(self, gate_fn: F) -> Operator
    where
        F: Fn(usize, usize) -> Operator,
    {
        match self.0.len() {
            0 => Operator::Pauli(PauliString::default()), // Identity
            1 => gate_fn(self.0[0].0.0, self.0[0].1.0),
            _ => Operator::Tensor(self.0.iter().map(|(q0, q1)| gate_fn(q0.0, q1.0)).collect()),
        }
    }
}

/// A gate/operator expression - lazy representation of quantum operators.
///
/// This is the unified type for all quantum operators including Pauli operators,
/// Clifford gates, and general unitaries.
#[derive(Debug, Clone, PartialEq)]
pub enum Operator {
    /// Pauli operator (single or multi-qubit)
    /// Wraps `PauliString` for exact Pauli algebra
    Pauli(PauliString),

    /// Rotation gate with angle: exp(-i θ/2 P)
    Rotation {
        rotation_type: RotationType,
        angle: Angle64,
        qubits: SmallVec<[usize; 2]>,
    },

    /// Fixed gate (control gates, etc.) without angle parameter
    Gate {
        gate_type: GateType,
        qubits: SmallVec<[usize; 3]>,
    },

    /// Tensor product of expressions (operators on different qubits)
    Tensor(Vec<Operator>),

    /// Sequential composition (matrix multiplication order)
    /// Compose([A, B, C]) means apply A, then B, then C
    Compose(Vec<Operator>),

    /// Adjoint (Hermitian conjugate)
    Adjoint(Box<Operator>),

    /// Global phase: e^{i*phase} * inner
    /// Phase is represented as Angle64 for exact arithmetic
    Phase {
        phase: Angle64,
        inner: Box<Operator>,
    },
}

impl Operator {
    /// Creates a rotation gate expression.
    #[must_use]
    pub fn rotation(
        rotation_type: RotationType,
        angle: Angle64,
        qubits: impl Into<SmallVec<[usize; 2]>>,
    ) -> Self {
        Self::Rotation {
            rotation_type,
            angle,
            qubits: qubits.into(),
        }
    }

    /// Creates a fixed gate expression.
    #[must_use]
    pub fn gate(gate_type: GateType, qubits: impl Into<SmallVec<[usize; 3]>>) -> Self {
        Self::Gate {
            gate_type,
            qubits: qubits.into(),
        }
    }

    /// Returns the adjoint (Hermitian conjugate) of this expression.
    #[must_use]
    pub fn dg(&self) -> Self {
        match self {
            // Pauli adjoint: Paulis are Hermitian, but phase conjugates
            Self::Pauli(ps) => {
                let conj_phase = ps.phase().conjugate();
                Self::Pauli(PauliString::with_phase_and_paulis(
                    conj_phase,
                    ps.iter_pairs().collect(),
                ))
            }
            // Rotation adjoint: negate the angle
            Self::Rotation {
                rotation_type,
                angle,
                qubits,
            } => Self::Rotation {
                rotation_type: *rotation_type,
                angle: negate_angle(*angle),
                qubits: qubits.clone(),
            },
            // Gate adjoint: wrap or simplify for self-adjoint gates
            Self::Gate {
                gate_type,
                qubits: _,
            } => {
                if gate_type.is_self_adjoint() {
                    self.clone()
                } else {
                    Self::Adjoint(Box::new(self.clone()))
                }
            }
            // Tensor adjoint: adjoint of each part
            Self::Tensor(parts) => Self::Tensor(parts.iter().map(Operator::dg).collect()),
            // Compose adjoint: reverse order and adjoint each
            Self::Compose(parts) => Self::Compose(parts.iter().rev().map(Operator::dg).collect()),
            // Double adjoint: unwrap
            Self::Adjoint(inner) => (**inner).clone(),
            // Phase adjoint: conjugate phase (negate), adjoint inner
            Self::Phase { phase, inner } => Self::Phase {
                phase: negate_angle(*phase),
                inner: Box::new(inner.dg()),
            },
        }
    }

    /// Applies a global phase to this expression: e^{i*phase} * self
    #[must_use]
    pub fn with_phase(self, phase: Angle64) -> Self {
        if phase == Angle64::ZERO {
            return self;
        }

        // For Pauli variants, try to absorb the phase into the PauliString
        // if it's a multiple of π/2 (quarter turn)
        if let Self::Pauli(ps) = self {
            if let Some(quarter_phase) = angle_to_quarter_phase(phase) {
                let new_phase = ps.phase().multiply(&quarter_phase);
                return Self::Pauli(PauliString::with_phase_and_paulis(
                    new_phase,
                    ps.iter_pairs().collect(),
                ));
            }
            // Not a quarter turn multiple, wrap in Phase
            return Self::Phase {
                phase,
                inner: Box::new(Self::Pauli(ps)),
            };
        }

        Self::Phase {
            phase,
            inner: Box::new(self),
        }
    }

    /// Checks if this expression represents a Clifford operation.
    ///
    /// Clifford gates are those where all rotation angles are multiples of π/2.
    #[must_use]
    pub fn is_clifford(&self) -> bool {
        match self {
            // Paulis are always Clifford
            Self::Pauli(_) => true,
            Self::Rotation { angle, .. } => {
                // Clifford if angle is multiple of π/2 (quarter turn)
                is_multiple_of_quarter_turn(*angle)
            }
            Self::Gate { gate_type, .. } => gate_type.is_clifford(),
            Self::Tensor(parts) | Self::Compose(parts) => parts.iter().all(Operator::is_clifford),
            // Phase doesn't affect Clifford-ness (global phase)
            Self::Adjoint(inner) | Self::Phase { inner, .. } => inner.is_clifford(),
        }
    }

    /// Returns the qubits this expression acts on.
    #[must_use]
    pub fn qubits(&self) -> Vec<usize> {
        let mut result = Vec::new();
        self.collect_qubits(&mut result);
        result.sort_unstable();
        result.dedup();
        result
    }

    fn collect_qubits(&self, result: &mut Vec<usize>) {
        match self {
            Self::Pauli(ps) => {
                result.extend(ps.iter_pairs().map(|(_, q)| usize::from(q)));
            }
            Self::Rotation { qubits, .. } => result.extend(qubits.iter().copied()),
            Self::Gate { qubits, .. } => result.extend(qubits.iter().copied()),
            Self::Tensor(parts) | Self::Compose(parts) => {
                for part in parts {
                    part.collect_qubits(result);
                }
            }
            Self::Adjoint(inner) | Self::Phase { inner, .. } => inner.collect_qubits(result),
        }
    }
}

// ============================================================================
// Negation operator: -op (phase by π)
// ============================================================================

impl Neg for Operator {
    type Output = Operator;

    fn neg(self) -> Operator {
        self.with_phase(Angle64::HALF_TURN)
    }
}

impl Neg for &Operator {
    type Output = Operator;

    fn neg(self) -> Operator {
        self.clone().with_phase(Angle64::HALF_TURN)
    }
}

// ============================================================================
// Imaginary unit for phase multiplication
// ============================================================================

/// Imaginary unit for phase multiplication: i * op
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImaginaryUnit;

/// The imaginary unit `i`.
#[allow(non_upper_case_globals)]
pub const i: ImaginaryUnit = ImaginaryUnit;

impl Neg for ImaginaryUnit {
    type Output = NegImaginaryUnit;

    fn neg(self) -> NegImaginaryUnit {
        NegImaginaryUnit
    }
}

/// Negative imaginary unit (-i).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NegImaginaryUnit;

impl Mul<Operator> for ImaginaryUnit {
    type Output = Operator;

    fn mul(self, rhs: Operator) -> Operator {
        rhs.with_phase(Angle64::QUARTER_TURN) // i = e^{iπ/2}
    }
}

impl Mul<&Operator> for ImaginaryUnit {
    type Output = Operator;

    fn mul(self, rhs: &Operator) -> Operator {
        rhs.clone().with_phase(Angle64::QUARTER_TURN)
    }
}

impl Mul<Operator> for NegImaginaryUnit {
    type Output = Operator;

    #[allow(clippy::suspicious_arithmetic_impl)] // Adding angles for phase computation
    fn mul(self, rhs: Operator) -> Operator {
        rhs.with_phase(Angle64::QUARTER_TURN + Angle64::HALF_TURN) // -i = e^{i3π/2}
    }
}

impl Mul<&Operator> for NegImaginaryUnit {
    type Output = Operator;

    #[allow(clippy::suspicious_arithmetic_impl)] // Adding angles for phase computation
    fn mul(self, rhs: &Operator) -> Operator {
        rhs.clone()
            .with_phase(Angle64::QUARTER_TURN + Angle64::HALF_TURN)
    }
}

// ============================================================================
// General phase value for arbitrary phase multiplication
// ============================================================================

/// A phase value e^{i*angle} that can be multiplied with operators.
///
/// # Example
/// ```
/// use pecos_core::operator::{phase, X};
/// use pecos_core::Angle64;
///
/// // Create a phase of e^{iπ/4}
/// let eighth_turn = Angle64::HALF_TURN / 4;
/// let op = phase(eighth_turn) * X(0);  // e^{iπ/4} * X
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseValue(pub Angle64);

/// Creates a phase value e^{i*angle} that can be multiplied with operators.
///
/// The phase represents the complex number e^{i*angle} = cos(angle) + i*sin(angle).
///
/// # Example
/// ```
/// use pecos_core::operator::{phase, X, Z};
/// use pecos_core::Angle64;
///
/// // e^{iπ/4} * X
/// let op = phase(Angle64::HALF_TURN / 4) * X(0);
///
/// // e^{iπ/2} * Z = i * Z
/// let op = phase(Angle64::QUARTER_TURN) * Z(0);
/// ```
#[must_use]
pub fn phase(angle: Angle64) -> PhaseValue {
    PhaseValue(angle)
}

impl Neg for PhaseValue {
    type Output = PhaseValue;

    fn neg(self) -> PhaseValue {
        // -e^{iθ} = e^{i(θ + π)}
        PhaseValue(self.0 + Angle64::HALF_TURN)
    }
}

impl Mul<PhaseValue> for ImaginaryUnit {
    type Output = PhaseValue;

    #[allow(clippy::suspicious_arithmetic_impl)] // Adding angles for phase computation
    fn mul(self, rhs: PhaseValue) -> PhaseValue {
        // i * e^{iθ} = e^{i(θ + π/2)}
        PhaseValue(rhs.0 + Angle64::QUARTER_TURN)
    }
}

impl Mul<PhaseValue> for NegImaginaryUnit {
    type Output = PhaseValue;

    #[allow(clippy::suspicious_arithmetic_impl)] // Adding angles for phase computation
    fn mul(self, rhs: PhaseValue) -> PhaseValue {
        // -i * e^{iθ} = e^{i(θ + 3π/2)}
        PhaseValue(rhs.0 + Angle64::QUARTER_TURN + Angle64::HALF_TURN)
    }
}

impl Mul<Operator> for PhaseValue {
    type Output = Operator;

    fn mul(self, rhs: Operator) -> Operator {
        rhs.with_phase(self.0)
    }
}

impl Mul<&Operator> for PhaseValue {
    type Output = Operator;

    fn mul(self, rhs: &Operator) -> Operator {
        rhs.clone().with_phase(self.0)
    }
}

impl Operator {
    /// Attempts to convert a rotation to its named `GateType` equivalent.
    #[must_use]
    pub fn to_named_gate(&self) -> Option<GateType> {
        match self {
            Self::Pauli(ps) => {
                // Single-qubit Paulis map to named gates
                if ps.weight() == 1 && ps.phase() == QuarterPhase::PlusOne {
                    let (pauli, _qubit) = ps.iter_pairs().next()?;
                    match pauli {
                        crate::Pauli::I => Some(GateType::I),
                        crate::Pauli::X => Some(GateType::X),
                        crate::Pauli::Y => Some(GateType::Y),
                        crate::Pauli::Z => Some(GateType::Z),
                    }
                } else {
                    None
                }
            }
            Self::Rotation {
                rotation_type,
                angle,
                ..
            } => rotation_to_gate_type(*rotation_type, *angle),
            Self::Gate { gate_type, .. } => Some(*gate_type),
            _ => None,
        }
    }

    /// Returns a reference to the inner `PauliString` if this is a `Pauli` variant.
    #[must_use]
    pub fn as_pauli_string(&self) -> Option<&PauliString> {
        if let Self::Pauli(ps) = self {
            Some(ps)
        } else {
            None
        }
    }

    /// Consumes this `Operator` and returns the inner `PauliString` if this is a `Pauli` variant.
    #[must_use]
    pub fn into_pauli_string(self) -> Option<PauliString> {
        if let Self::Pauli(ps) = self {
            Some(ps)
        } else {
            None
        }
    }

    /// Attempts to convert this operator to a `PauliString`.
    ///
    /// This handles more cases than `into_pauli_string()`:
    /// - `Pauli(ps)` → returns `ps` directly
    /// - `Tensor([Pauli(a), Pauli(b), ...])` → merges into a single `PauliString`
    /// - `Phase { phase, inner: Pauli(ps) }` → applies phase to `ps`
    /// - Named Pauli gates (`X`, `Y`, `Z`) → corresponding single-qubit `PauliString`
    /// - Half-turn rotations (`RX(π)`, `RY(π)`, `RZ(π)`) → corresponding `PauliString`
    ///
    /// Returns `None` if the operator cannot be represented as a `PauliString`.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_core::{Xs, Zs, PauliOperator};
    ///
    /// // Tensor of Paulis on disjoint qubits
    /// let op = Xs(0..2) & Zs(2..4);
    /// let ps = op.try_to_pauli_string().unwrap();
    /// assert_eq!(ps.weight(), 4); // X on 0,1 and Z on 2,3
    /// ```
    #[must_use]
    pub fn try_to_pauli_string(self) -> Option<PauliString> {
        match self {
            Self::Pauli(ps) => Some(ps),

            Self::Tensor(parts) => {
                // Try to convert all parts to PauliStrings and merge
                let mut result = PauliString::new();
                for part in parts {
                    let ps = part.try_to_pauli_string()?;
                    // Merge: combine the Pauli operators
                    // For disjoint qubits, this is just concatenation
                    // For overlapping qubits, we multiply the Paulis
                    result = result * ps;
                }
                Some(result)
            }

            Self::Phase { phase, inner } => {
                let mut ps = inner.try_to_pauli_string()?;
                // Apply the global phase to the PauliString phase
                // phase is Angle64, we need to convert to QuarterPhase if possible
                // For now, only handle quarter-turn phases exactly
                let quarter_phase = if phase == Angle64::ZERO {
                    QuarterPhase::PlusOne
                } else if phase == Angle64::QUARTER_TURN {
                    QuarterPhase::PlusI
                } else if phase == Angle64::HALF_TURN {
                    QuarterPhase::MinusOne
                } else if phase == Angle64::THREE_QUARTERS_TURN {
                    QuarterPhase::MinusI
                } else {
                    // Non-quarter-turn phase, can't represent exactly
                    return None;
                };
                let new_phase = ps.phase().multiply(&quarter_phase);
                ps.set_phase(new_phase);
                Some(ps)
            }

            Self::Gate { gate_type, qubits } => {
                let qubit = qubits.first().copied()?;
                match gate_type {
                    GateType::X => Some(PauliString::x(qubit)),
                    GateType::Y => Some(PauliString::y(qubit)),
                    GateType::Z => Some(PauliString::z(qubit)),
                    GateType::I => Some(PauliString::identity()),
                    _ => None,
                }
            }

            Self::Rotation {
                rotation_type,
                angle,
                qubits,
            } => {
                // Only half-turn rotations are Pauli operators
                let half = Angle64::HALF_TURN;
                let neg_half = negate_angle(half);
                if angle != half && angle != neg_half {
                    return None;
                }
                let qubit = qubits.first().copied()?;
                match rotation_type {
                    RotationType::RX => Some(PauliString::x(qubit)),
                    RotationType::RY => Some(PauliString::y(qubit)),
                    RotationType::RZ => Some(PauliString::z(qubit)),
                    _ => None,
                }
            }

            Self::Adjoint(inner) => {
                // Paulis are Hermitian (self-adjoint), but phase conjugates
                let mut ps = inner.try_to_pauli_string()?;
                let conj_phase = ps.phase().conjugate();
                ps.set_phase(conj_phase);
                Some(ps)
            }

            Self::Compose(_) => {
                // Composition of Paulis requires multiplication
                // This is more complex; skip for now
                None
            }
        }
    }

    /// Checks if this operator is equivalent to a Pauli operator.
    ///
    /// Returns true for:
    /// - `Pauli` variants (any `PauliString`)
    /// - Half-turn rotations: `RX(π)`, `RY(π)`, `RZ(π)`
    /// - Named Pauli gates: `X`, `Y`, `Z`
    #[must_use]
    pub fn is_pauli_equivalent(&self) -> bool {
        match self {
            Self::Pauli(_) => true,
            Self::Rotation {
                rotation_type,
                angle,
                ..
            } => {
                let half = Angle64::HALF_TURN;
                let neg_half = negate_angle(half);
                (*angle == half || *angle == neg_half)
                    && matches!(
                        rotation_type,
                        RotationType::RX | RotationType::RY | RotationType::RZ
                    )
            }
            Self::Gate { gate_type, .. } => {
                matches!(gate_type, GateType::X | GateType::Y | GateType::Z)
            }
            _ => false,
        }
    }

    /// Attempts to convert this operator to a `Pauli` variant.
    ///
    /// Converts:
    /// - `Pauli` → returns as-is
    /// - `RX(π)` → `X`
    /// - `RY(π)` → `Y`
    /// - `RZ(π)` → `Z`
    /// - Named gates `X`, `Y`, `Z` → corresponding `Pauli` variant
    ///
    /// Returns `None` if the operator is not Pauli-equivalent.
    #[must_use]
    pub fn try_to_pauli(self) -> Option<Self> {
        match self {
            Self::Pauli(_) => Some(self),
            Self::Rotation {
                rotation_type,
                angle,
                qubits,
            } => {
                let half = Angle64::HALF_TURN;
                let neg_half = negate_angle(half);
                if angle != half && angle != neg_half {
                    return None;
                }
                let qubit = qubits[0];
                match rotation_type {
                    RotationType::RX => Some(X(qubit)),
                    RotationType::RY => Some(Y(qubit)),
                    RotationType::RZ => Some(Z(qubit)),
                    _ => None,
                }
            }
            Self::Gate { gate_type, qubits } => {
                let qubit = qubits[0];
                match gate_type {
                    GateType::X => Some(X(qubit)),
                    GateType::Y => Some(Y(qubit)),
                    GateType::Z => Some(Z(qubit)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Simplifies this gate expression by:
    /// - Merging adjacent rotations of the same type on the same qubits
    /// - Canceling inverse operations (rotation + its negation)
    /// - Removing identity operations (zero-angle rotations)
    /// - Flattening single-element containers
    #[must_use]
    #[allow(clippy::missing_panics_doc)] // Internal expects are guarded by length checks
    pub fn simplify(&self) -> Self {
        match self {
            // Pauli and Gate are already in simplified form
            Self::Pauli(_) | Self::Gate { .. } => self.clone(),

            Self::Rotation { angle, .. } => {
                // Remove identity rotations
                if *angle == Angle64::ZERO {
                    return self.clone(); // Keep as-is, will be filtered at Compose level
                }
                self.clone()
            }

            Self::Tensor(parts) => {
                // Simplify each part but preserve identities (they define the Hilbert space dimension)
                let simplified: Vec<_> = parts.iter().map(Operator::simplify).collect();

                match simplified.len() {
                    0 => Self::Pauli(PauliString::default()), // Empty tensor = identity
                    1 => simplified.into_iter().next().expect("length is 1"),
                    _ => Self::Tensor(simplified),
                }
            }

            Self::Compose(parts) => {
                // First simplify each part
                let simplified: Vec<_> = parts.iter().map(Operator::simplify).collect();

                // Flatten nested Compose nodes
                let flattened = flatten_compose(simplified);

                // Merge adjacent compatible rotations
                let merged = merge_adjacent_rotations(flattened);

                // Filter out identities
                let filtered: Vec<_> = merged.into_iter().filter(|p| !p.is_identity()).collect();

                match filtered.len() {
                    0 => I(0), // Empty composition = identity
                    1 => filtered.into_iter().next().expect("length is 1"),
                    _ => Self::Compose(filtered),
                }
            }

            Self::Adjoint(inner) => {
                // Simplify inner first, then take adjoint
                let simplified_inner = inner.simplify();
                simplified_inner.dg()
            }

            Self::Phase { phase, inner } => {
                // Simplify inner and preserve phase
                let simplified_inner = inner.simplify();
                if *phase == Angle64::ZERO || *phase == Angle64::FULL_TURN {
                    simplified_inner
                } else {
                    Self::Phase {
                        phase: *phase,
                        inner: Box::new(simplified_inner),
                    }
                }
            }
        }
    }

    /// Conjugates this operator by another: `gate * self * gate.dg()` (i.e., UAU†).
    ///
    /// This is the stabilizer update convention: when gate U is applied to a state,
    /// a stabilizer S transforms as S → U S U†.
    ///
    /// For Heisenberg picture evolution (U†AU), use [`conjdg`](Self::conjdg).
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_core::operator::{X, Z, H, T};
    ///
    /// // Stabilizer update: applying H to qubit 0
    /// let stabilizer = X(0) & Z(1);
    /// let updated = stabilizer.conj(&H(0));  // H * (X⊗Z) * H†
    ///
    /// // Works with any operators
    /// let a = T(0);
    /// let b = X(0);
    /// let conjugated = b.conj(&a);  // T X T†
    /// ```
    #[must_use]
    pub fn conj(&self, gate: &Operator) -> Self {
        gate.clone() * self.clone() * gate.dg()
    }

    /// Conjugates this operator by the adjoint of another: `gate.dg() * self * gate` (i.e., U†AU).
    ///
    /// This is the Heisenberg picture convention: operators evolve as A → U†AU.
    ///
    /// For stabilizer updates (UAU†), use [`conj`](Self::conj).
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_core::operator::{X, H};
    ///
    /// // Heisenberg evolution: how X evolves under H
    /// let evolved = X(0).conjdg(&H(0));  // H† X H
    /// ```
    #[must_use]
    pub fn conjdg(&self, gate: &Operator) -> Self {
        gate.dg() * self.clone() * gate.clone()
    }

    /// Returns the global phase of this operator.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_core::operator::{X, Y};
    /// use pecos_core::{GlobalPhase, QuarterPhase};
    ///
    /// let op = X(0);
    /// assert_eq!(op.phase(), GlobalPhase::one());
    ///
    /// let op = -Y(0);
    /// assert_eq!(op.phase(), GlobalPhase::minus_one());
    /// ```
    #[must_use]
    pub fn phase(&self) -> crate::GlobalPhase {
        use crate::GlobalPhase;

        match self {
            Self::Pauli(ps) => GlobalPhase::from(ps.phase()),
            Self::Phase { phase, .. } => GlobalPhase::from(*phase),
            _ => GlobalPhase::one(),
        }
    }

    /// Returns the weight (number of qubits) this operator acts on.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_core::operator::{X, Z, CX};
    ///
    /// assert_eq!(X(0).weight(), 1);
    /// assert_eq!((X(0) & Z(2)).weight(), 2);
    /// assert_eq!(CX(0, 1).weight(), 2);
    /// ```
    #[must_use]
    pub fn weight(&self) -> usize {
        self.qubits().len()
    }

    /// Checks if this is structurally the identity operator.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_core::operator::I;
    ///
    /// assert!(I(0).is_identity());
    /// ```
    #[must_use]
    pub fn is_identity(&self) -> bool {
        match self {
            Self::Pauli(ps) => ps.weight() == 0 && ps.phase() == crate::QuarterPhase::PlusOne,
            Self::Gate { gate_type, .. } => *gate_type == GateType::I,
            Self::Rotation { angle, .. } => *angle == Angle64::ZERO,
            Self::Tensor(parts) | Self::Compose(parts) => parts.iter().all(Operator::is_identity),
            Self::Adjoint(inner) => inner.is_identity(),
            Self::Phase { phase, inner } => *phase == Angle64::ZERO && inner.is_identity(),
        }
    }

    /// Checks if this operator is Hermitian (self-adjoint): A = A†.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_core::operator::{X, Y, Z, H, T};
    ///
    /// // Paulis are Hermitian
    /// assert!(X(0).is_hermitian());
    /// assert!(Y(0).is_hermitian());
    /// assert!(Z(0).is_hermitian());
    ///
    /// // H is Hermitian
    /// assert!(H(0).is_hermitian());
    ///
    /// // T is not Hermitian (T† ≠ T)
    /// assert!(!T(0).is_hermitian());
    /// ```
    #[must_use]
    pub fn is_hermitian(&self) -> bool {
        // A is Hermitian if A = A†
        // For structural comparison, we check known Hermitian operators
        match self {
            Self::Pauli(_) => true, // All Paulis are Hermitian
            Self::Gate { gate_type, .. } => matches!(
                gate_type,
                GateType::I
                    | GateType::X
                    | GateType::Y
                    | GateType::Z
                    | GateType::H
                    | GateType::CX
                    | GateType::CY
                    | GateType::CZ
                    | GateType::SWAP
            ),
            Self::Rotation { angle, .. } => {
                // Rotations are Hermitian only at angle 0 or π
                *angle == Angle64::ZERO || *angle == Angle64::HALF_TURN
            }
            Self::Tensor(parts) => parts.iter().all(Operator::is_hermitian),
            // Composition of Hermitians isn't generally Hermitian; phase factors break Hermiticity
            Self::Compose(_) | Self::Phase { .. } => false,
            Self::Adjoint(inner) => inner.is_hermitian(), // (A†)† = A, so same as inner
        }
    }

    /// Returns the operator raised to a power (repeated composition).
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_core::operator::{X, H};
    ///
    /// let x = X(0);
    /// let x2 = x.pow(2);  // X * X = I
    ///
    /// let h = H(0);
    /// let h3 = h.pow(3);  // H * H * H = H
    /// ```
    #[must_use]
    pub fn pow(&self, n: u32) -> Self {
        match n {
            0 => Self::Gate {
                gate_type: GateType::I,
                qubits: self
                    .qubits()
                    .into_iter()
                    .next()
                    .map_or(smallvec::smallvec![0], |q| smallvec::smallvec![q]),
            },
            1 => self.clone(),
            _ => {
                let mut result = self.clone();
                for _ in 1..n {
                    result = result * self.clone();
                }
                result
            }
        }
    }

    /// Checks whether this operator commutes with another.
    ///
    /// For Pauli operators, returns `Commutes` or `AntiCommutes`.
    /// For non-Pauli operators, returns `Unknown`.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_core::operator::{X, Z, Commutativity};
    ///
    /// let a = X(0);
    /// let b = Z(0);
    /// assert_eq!(a.commutes(&b), Commutativity::AntiCommutes);
    ///
    /// let c = X(0) & Z(1);
    /// let d = Z(0) & X(1);
    /// assert_eq!(c.commutes(&d), Commutativity::Commutes);
    /// ```
    #[must_use]
    pub fn commutes(&self, other: &Operator) -> Commutativity {
        use crate::PauliOperator;

        match (self, other) {
            (Self::Pauli(a), Self::Pauli(b)) => {
                if a.commutes_with(b) {
                    Commutativity::Commutes
                } else {
                    Commutativity::AntiCommutes
                }
            }
            _ => Commutativity::Unknown,
        }
    }

    /// Returns whether this operator is unitary.
    ///
    /// All operators in this enum are unitary by construction.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_core::operator::{X, H, RZ};
    /// use pecos_core::Angle64;
    ///
    /// assert!(X(0).is_unitary());
    /// assert!(H(0).is_unitary());
    /// assert!(RZ(Angle64::QUARTER_TURN, 0).is_unitary());
    /// ```
    #[must_use]
    pub fn is_unitary(&self) -> bool {
        true // All Operator variants are unitary by construction
    }

    /// Decomposes this operator into a sequence of primitive gates.
    ///
    /// Returns a flat vector of `Gate` structs representing the operator
    /// as a sequence of native gates.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_core::operator::{H, CX};
    ///
    /// let circuit = CX(0, 1) * H(0);  // H then CX
    /// let gates = circuit.decompose();
    /// assert_eq!(gates.len(), 2);
    /// ```
    #[must_use]
    pub fn decompose(&self) -> Vec<crate::Gate> {
        use crate::{Gate, Pauli};

        match self {
            Self::Pauli(ps) => {
                // Convert PauliString to individual gates
                let mut gates = Vec::new();
                for (pauli, qubit) in ps.iter_pairs() {
                    let gate = match pauli {
                        Pauli::I => continue, // Skip identity
                        Pauli::X => Gate::simple(GateType::X, smallvec::smallvec![qubit]),
                        Pauli::Y => Gate::simple(GateType::Y, smallvec::smallvec![qubit]),
                        Pauli::Z => Gate::simple(GateType::Z, smallvec::smallvec![qubit]),
                    };
                    gates.push(gate);
                }
                // Handle global phase if not +1
                // (Phase is tracked separately in PauliString but not representable in Gate)
                gates
            }

            Self::Rotation {
                rotation_type,
                angle,
                qubits,
            } => {
                let gate_type = match rotation_type {
                    RotationType::RX => GateType::RX,
                    RotationType::RY => GateType::RY,
                    RotationType::RZ => GateType::RZ,
                    RotationType::RXX => GateType::RXX,
                    RotationType::RYY => GateType::RYY,
                    RotationType::RZZ => GateType::RZZ,
                };
                let qubit_ids: crate::GateQubits =
                    qubits.iter().map(|&q| crate::QubitId(q)).collect();
                vec![Gate::with_angles(
                    gate_type,
                    smallvec::smallvec![*angle],
                    qubit_ids,
                )]
            }

            Self::Gate { gate_type, qubits } => {
                let qubit_ids: crate::GateQubits =
                    qubits.iter().map(|&q| crate::QubitId(q)).collect();
                vec![Gate::simple(*gate_type, qubit_ids)]
            }

            Self::Tensor(parts) => {
                // Decompose each part and concatenate
                parts.iter().flat_map(Operator::decompose).collect()
            }

            Self::Compose(parts) => {
                // Decompose each part in application order
                parts.iter().flat_map(Operator::decompose).collect()
            }

            Self::Adjoint(inner) => {
                // Decompose inner, reverse, and adjoint each gate
                let mut gates = inner.decompose();
                gates.reverse();
                for gate in &mut gates {
                    // Negate angles for rotation gates
                    for angle in &mut gate.angles {
                        *angle = Angle64::ZERO - *angle;
                    }
                    // Some gates need special handling
                    gate.gate_type = match gate.gate_type {
                        GateType::SX => GateType::SXdg,
                        GateType::SXdg => GateType::SX,
                        GateType::SY => GateType::SYdg,
                        GateType::SYdg => GateType::SY,
                        GateType::SZ => GateType::SZdg,
                        GateType::SZdg => GateType::SZ,
                        GateType::T => GateType::Tdg,
                        GateType::Tdg => GateType::T,
                        GateType::SZZ => GateType::SZZdg,
                        GateType::SZZdg => GateType::SZZ,
                        other => other, // Self-adjoint gates unchanged
                    };
                }
                gates
            }

            Self::Phase { inner, .. } => {
                // Global phase doesn't affect gate sequence
                // (Phase information is lost in decomposition)
                inner.decompose()
            }
        }
    }

    /// Converts this gate expression to a `CliffordRep` (generator propagation).
    ///
    /// Returns `None` if the expression contains non-Clifford operations.
    ///
    /// # Arguments
    /// * `num_qubits` - The total number of qubits in the system
    #[must_use]
    pub fn to_clifford_rep(&self, num_qubits: usize) -> Option<crate::clifford_rep::CliffordRep> {
        use crate::clifford_rep::CliffordRep;

        if !self.is_clifford() {
            return None;
        }

        match self {
            Self::Pauli(ps) => {
                // Convert PauliString to CliffordRep by composing single-qubit Paulis
                let mut result = CliffordRep::identity(num_qubits);
                for (pauli, qubit) in ps.iter_pairs() {
                    let q = usize::from(qubit);
                    let cliff = match pauli {
                        crate::Pauli::I => continue, // Skip identity
                        crate::Pauli::X => CliffordRep::x_on(q, num_qubits),
                        crate::Pauli::Y => CliffordRep::y_on(q, num_qubits),
                        crate::Pauli::Z => CliffordRep::z_on(q, num_qubits),
                    };
                    result = cliff.compose(&result);
                }
                Some(result)
            }

            Self::Rotation {
                rotation_type,
                angle,
                qubits,
            } => rotation_to_clifford_rep(*rotation_type, *angle, qubits, num_qubits),

            Self::Gate { gate_type, qubits } => {
                gate_type_to_clifford_rep(*gate_type, qubits, num_qubits)
            }

            Self::Tensor(parts) => {
                // For tensor products, compose all parts (they act on different qubits)
                let mut result = CliffordRep::identity(num_qubits);
                for part in parts {
                    if let Some(cliff) = part.to_clifford_rep(num_qubits) {
                        result = result.compose(&cliff);
                    } else {
                        return None;
                    }
                }
                Some(result)
            }

            Self::Compose(parts) => {
                // For composition, compose in order (parts are in application order)
                let mut result = CliffordRep::identity(num_qubits);
                for part in parts {
                    if let Some(cliff) = part.to_clifford_rep(num_qubits) {
                        result = cliff.compose(&result);
                    } else {
                        return None;
                    }
                }
                Some(result)
            }

            Self::Adjoint(inner) => {
                // Get the inner CliffordRep and take its inverse
                inner
                    .to_clifford_rep(num_qubits)
                    .map(|cliff| cliff.inverse())
            }

            Self::Phase { inner, .. } => {
                // Global phase is ignored in CliffordRep (Heisenberg picture)
                inner.to_clifford_rep(num_qubits)
            }
        }
    }
}

/// Convert a rotation to `CliffordRep` if it's Clifford.
fn rotation_to_clifford_rep(
    rotation_type: RotationType,
    angle: Angle64,
    qubits: &SmallVec<[usize; 2]>,
    num_qubits: usize,
) -> Option<crate::clifford_rep::CliffordRep> {
    use crate::clifford_rep::CliffordRep;

    // Check for Clifford angles (multiples of π/2)
    let quarter = Angle64::QUARTER_TURN;
    let neg_quarter = negate_angle(quarter);
    let half = Angle64::HALF_TURN;
    let neg_half = negate_angle(half);
    let three_quarter = quarter + half;
    let _neg_three_quarter = negate_angle(three_quarter);

    // Identity
    if angle == Angle64::ZERO {
        return Some(CliffordRep::identity(num_qubits));
    }

    match rotation_type {
        RotationType::RZ => {
            let qubit = qubits[0];
            let mut result = CliffordRep::identity(num_qubits);

            if angle == quarter {
                // S = RZ(π/2)
                result = apply_s(&result, qubit);
            } else if angle == neg_quarter || angle == three_quarter {
                // S† = RZ(-π/2) = RZ(3π/2)
                result = apply_sdg(&result, qubit);
            } else if angle == half || angle == neg_half {
                // Z = RZ(π)
                result = apply_z(&result, qubit);
            } else {
                return None; // Not a Clifford angle
            }
            Some(result)
        }

        RotationType::RX => {
            let qubit = qubits[0];
            let mut result = CliffordRep::identity(num_qubits);

            if angle == quarter {
                // SX = RX(π/2)
                result = apply_sx(&result, qubit);
            } else if angle == neg_quarter || angle == three_quarter {
                // SX† = RX(-π/2)
                result = apply_sxdg(&result, qubit);
            } else if angle == half || angle == neg_half {
                // X = RX(π)
                result = apply_x(&result, qubit);
            } else {
                return None;
            }
            Some(result)
        }

        RotationType::RY => {
            let qubit = qubits[0];
            let mut result = CliffordRep::identity(num_qubits);

            if angle == quarter {
                // SY = RY(π/2)
                result = apply_sy(&result, qubit);
            } else if angle == neg_quarter || angle == three_quarter {
                // SY† = RY(-π/2)
                result = apply_sydg(&result, qubit);
            } else if angle == half || angle == neg_half {
                // Y = RY(π)
                result = apply_y(&result, qubit);
            } else {
                return None;
            }
            Some(result)
        }

        RotationType::RZZ => {
            let q0 = qubits[0];
            let q1 = qubits[1];

            if angle == quarter {
                Some(CliffordRep::cz(q0, q1).compose(&CliffordRep::identity(num_qubits)))
            } else if angle == neg_quarter || angle == three_quarter {
                // SZZ† - CZ with phase adjustment
                Some(CliffordRep::cz(q0, q1).compose(&CliffordRep::identity(num_qubits)))
            } else {
                None
            }
        }

        _ => None, // RXX, RYY at non-zero angles are not standard Cliffords
    }
}

/// Convert a `GateType` to `CliffordRep`.
fn gate_type_to_clifford_rep(
    gate_type: GateType,
    qubits: &SmallVec<[usize; 3]>,
    num_qubits: usize,
) -> Option<crate::clifford_rep::CliffordRep> {
    use crate::clifford_rep::CliffordRep;

    match gate_type {
        GateType::I => Some(CliffordRep::identity(num_qubits)),
        GateType::X => {
            let mut result = CliffordRep::identity(num_qubits);
            result = apply_x(&result, qubits[0]);
            Some(result)
        }
        GateType::Y => {
            let mut result = CliffordRep::identity(num_qubits);
            result = apply_y(&result, qubits[0]);
            Some(result)
        }
        GateType::Z => {
            let mut result = CliffordRep::identity(num_qubits);
            result = apply_z(&result, qubits[0]);
            Some(result)
        }
        GateType::H => {
            let cliff = CliffordRep::h(qubits[0]);
            // Extend to num_qubits
            Some(extend_clifford(cliff, num_qubits))
        }
        GateType::SX => {
            let mut result = CliffordRep::identity(num_qubits);
            result = apply_sx(&result, qubits[0]);
            Some(result)
        }
        GateType::SY => {
            let mut result = CliffordRep::identity(num_qubits);
            result = apply_sy(&result, qubits[0]);
            Some(result)
        }
        GateType::SZ => {
            let mut result = CliffordRep::identity(num_qubits);
            result = apply_s(&result, qubits[0]);
            Some(result)
        }
        GateType::SXdg => {
            let mut result = CliffordRep::identity(num_qubits);
            result = apply_sxdg(&result, qubits[0]);
            Some(result)
        }
        GateType::SYdg => {
            let mut result = CliffordRep::identity(num_qubits);
            result = apply_sydg(&result, qubits[0]);
            Some(result)
        }
        GateType::SZdg => {
            let mut result = CliffordRep::identity(num_qubits);
            result = apply_sdg(&result, qubits[0]);
            Some(result)
        }
        GateType::CX => {
            let cliff = CliffordRep::cx(qubits[0], qubits[1]);
            Some(extend_clifford(cliff, num_qubits))
        }
        GateType::CY => {
            let cliff = CliffordRep::cy(qubits[0], qubits[1]);
            Some(extend_clifford(cliff, num_qubits))
        }
        GateType::CZ => {
            let cliff = CliffordRep::cz(qubits[0], qubits[1]);
            Some(extend_clifford(cliff, num_qubits))
        }
        GateType::SWAP => {
            let cliff = CliffordRep::swap(qubits[0], qubits[1]);
            Some(extend_clifford(cliff, num_qubits))
        }
        _ => None, // Non-Clifford or unsupported gate
    }
}

/// Extend a `CliffordRep` to act on more qubits (identity on new qubits).
fn extend_clifford(
    cliff: crate::clifford_rep::CliffordRep,
    target_qubits: usize,
) -> crate::clifford_rep::CliffordRep {
    use crate::clifford_rep::CliffordRep;

    if cliff.num_qubits() >= target_qubits {
        return cliff;
    }

    // Create a new CliffordRep with more qubits, copying the original images
    // and using identity for the additional qubits
    let mut result = CliffordRep::identity(target_qubits);

    // Copy the generator images from the original
    for q in 0..cliff.num_qubits() {
        result.set_x_image(q, cliff.x_image(q).clone());
        result.set_z_image(q, cliff.z_image(q).clone());
    }
    // Additional qubits remain as identity (already set by CliffordRep::identity)

    result
}

// Helper functions to apply single-qubit Cliffords to a CliffordRep
fn apply_x(
    cliff: &crate::clifford_rep::CliffordRep,
    qubit: usize,
) -> crate::clifford_rep::CliffordRep {
    let x_cliff = crate::clifford_rep::CliffordRep::x(qubit);
    extend_clifford(x_cliff, cliff.num_qubits()).compose(cliff)
}

fn apply_y(
    cliff: &crate::clifford_rep::CliffordRep,
    qubit: usize,
) -> crate::clifford_rep::CliffordRep {
    let y_cliff = crate::clifford_rep::CliffordRep::y(qubit);
    extend_clifford(y_cliff, cliff.num_qubits()).compose(cliff)
}

fn apply_z(
    cliff: &crate::clifford_rep::CliffordRep,
    qubit: usize,
) -> crate::clifford_rep::CliffordRep {
    let z_cliff = crate::clifford_rep::CliffordRep::z(qubit);
    extend_clifford(z_cliff, cliff.num_qubits()).compose(cliff)
}

fn apply_s(
    cliff: &crate::clifford_rep::CliffordRep,
    qubit: usize,
) -> crate::clifford_rep::CliffordRep {
    let s_cliff = crate::clifford_rep::CliffordRep::s(qubit);
    extend_clifford(s_cliff, cliff.num_qubits()).compose(cliff)
}

fn apply_sdg(
    cliff: &crate::clifford_rep::CliffordRep,
    qubit: usize,
) -> crate::clifford_rep::CliffordRep {
    let sdg_cliff = crate::clifford_rep::CliffordRep::sdg(qubit);
    extend_clifford(sdg_cliff, cliff.num_qubits()).compose(cliff)
}

fn apply_sx(
    cliff: &crate::clifford_rep::CliffordRep,
    qubit: usize,
) -> crate::clifford_rep::CliffordRep {
    let sx_cliff = crate::clifford_rep::CliffordRep::sx(qubit);
    extend_clifford(sx_cliff, cliff.num_qubits()).compose(cliff)
}

fn apply_sxdg(
    cliff: &crate::clifford_rep::CliffordRep,
    qubit: usize,
) -> crate::clifford_rep::CliffordRep {
    // SX† = SX^3 = SX * SX * SX
    let sx_cliff = crate::clifford_rep::CliffordRep::sx(qubit);
    let extended = extend_clifford(sx_cliff.clone(), cliff.num_qubits());
    extended
        .compose(&extended)
        .compose(&extended)
        .compose(cliff)
}

fn apply_sy(
    cliff: &crate::clifford_rep::CliffordRep,
    qubit: usize,
) -> crate::clifford_rep::CliffordRep {
    let sy_cliff = crate::clifford_rep::CliffordRep::sy(qubit);
    extend_clifford(sy_cliff, cliff.num_qubits()).compose(cliff)
}

fn apply_sydg(
    cliff: &crate::clifford_rep::CliffordRep,
    qubit: usize,
) -> crate::clifford_rep::CliffordRep {
    // SY† = SY^3
    let sy_cliff = crate::clifford_rep::CliffordRep::sy(qubit);
    let extended = extend_clifford(sy_cliff.clone(), cliff.num_qubits());
    extended
        .compose(&extended)
        .compose(&extended)
        .compose(cliff)
}

/// Flatten nested Compose nodes into a single level.
fn flatten_compose(parts: Vec<Operator>) -> Vec<Operator> {
    let mut result = Vec::new();
    for part in parts {
        match part {
            Operator::Compose(inner_parts) => {
                result.extend(flatten_compose(inner_parts));
            }
            other => result.push(other),
        }
    }
    result
}

/// Merge adjacent rotations of the same type on the same qubits.
fn merge_adjacent_rotations(parts: Vec<Operator>) -> Vec<Operator> {
    if parts.len() < 2 {
        return parts;
    }

    let mut result = Vec::with_capacity(parts.len());
    let mut idx = 0;

    while idx < parts.len() {
        let current = &parts[idx];

        // Check if next element can be merged with current
        if idx + 1 < parts.len()
            && let Some(merged) = try_merge_rotations(current, &parts[idx + 1])
        {
            // Skip the merged element
            if merged.is_identity() {
                // Both cancelled out, skip both
                idx += 2;
                continue;
            }
            result.push(merged);
            idx += 2;
            continue;
        }

        result.push(parts[idx].clone());
        idx += 1;
    }

    // Recurse if we made any merges (might enable more merges)
    if result.len() < parts.len() {
        merge_adjacent_rotations(result)
    } else {
        result
    }
}

/// Try to merge two rotations if they are compatible.
/// Returns None if they cannot be merged.
fn try_merge_rotations(a: &Operator, b: &Operator) -> Option<Operator> {
    match (a, b) {
        (
            Operator::Rotation {
                rotation_type: rt_a,
                angle: angle_a,
                qubits: qubits_a,
            },
            Operator::Rotation {
                rotation_type: rt_b,
                angle: angle_b,
                qubits: qubits_b,
            },
        ) => {
            // Can only merge if same rotation type and same qubits
            if rt_a == rt_b && qubits_a == qubits_b {
                let combined_angle = *angle_a + *angle_b;
                Some(Operator::Rotation {
                    rotation_type: *rt_a,
                    angle: combined_angle,
                    qubits: qubits_a.clone(),
                })
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Convert a rotation (type + angle) to a named `GateType` if one exists.
#[must_use]
pub fn rotation_to_gate_type(rotation_type: RotationType, angle: Angle64) -> Option<GateType> {
    // Check for standard angles
    let quarter = Angle64::QUARTER_TURN;
    let neg_quarter = negate_angle(quarter);
    let half = Angle64::HALF_TURN;
    let eighth = half / 4; // π/4
    let neg_eighth = negate_angle(eighth);

    match rotation_type {
        RotationType::RZ => {
            if angle == quarter {
                Some(GateType::SZ)
            } else if angle == neg_quarter {
                Some(GateType::SZdg)
            } else if angle == half {
                Some(GateType::Z)
            } else if angle == eighth {
                Some(GateType::T)
            } else if angle == neg_eighth {
                Some(GateType::Tdg)
            } else {
                None
            }
        }
        RotationType::RX => {
            if angle == quarter {
                Some(GateType::SX)
            } else if angle == neg_quarter {
                Some(GateType::SXdg)
            } else if angle == half {
                Some(GateType::X)
            } else {
                None
            }
        }
        RotationType::RY => {
            if angle == quarter {
                Some(GateType::SY)
            } else if angle == neg_quarter {
                Some(GateType::SYdg)
            } else if angle == half {
                Some(GateType::Y)
            } else {
                None
            }
        }
        RotationType::RZZ => {
            if angle == quarter {
                Some(GateType::SZZ)
            } else if angle == neg_quarter {
                Some(GateType::SZZdg)
            } else {
                None
            }
        }
        _ => None,
    }
}

// ============================================================================
// Gate type helpers
// ============================================================================

trait GateTypeExt {
    fn is_clifford(&self) -> bool;
    fn is_self_adjoint(&self) -> bool;
}

impl GateTypeExt for GateType {
    fn is_clifford(&self) -> bool {
        use GateType::{CX, CY, CZ, H, I, SWAP, SX, SXdg, SY, SYdg, SZ, SZZ, SZZdg, SZdg, X, Y, Z};
        matches!(
            self,
            I | X
                | Y
                | Z
                | H
                | SX
                | SXdg
                | SY
                | SYdg
                | SZ
                | SZdg
                | CX
                | CY
                | CZ
                | SWAP
                | SZZ
                | SZZdg
        )
    }

    fn is_self_adjoint(&self) -> bool {
        use GateType::{CX, CY, CZ, H, I, SWAP, X, Y, Z};
        matches!(self, I | X | Y | Z | H | CX | CY | CZ | SWAP)
    }
}

// ============================================================================
// Angle64 helpers
// ============================================================================

/// Check if an angle is a multiple of a quarter turn (π/2).
///
/// This is used to determine if a rotation is a Clifford gate.
fn is_multiple_of_quarter_turn(angle: Angle64) -> bool {
    let quarter_fraction = Angle64::QUARTER_TURN.fraction();
    if quarter_fraction == 0 {
        return true; // Edge case: if quarter is 0, everything is a multiple
    }
    angle.fraction().is_multiple_of(quarter_fraction)
}

/// Negate an angle (for adjoint operations).
fn negate_angle(angle: Angle64) -> Angle64 {
    Angle64::ZERO - angle
}

/// Convert an angle to a `QuarterPhase` if it's a multiple of π/2.
///
/// Returns None if the angle is not a multiple of π/2.
fn angle_to_quarter_phase(angle: Angle64) -> Option<QuarterPhase> {
    let quarter = Angle64::QUARTER_TURN;
    let half = Angle64::HALF_TURN;
    let three_quarters = quarter + half;

    if angle == Angle64::ZERO {
        Some(QuarterPhase::PlusOne)
    } else if angle == quarter {
        Some(QuarterPhase::PlusI)
    } else if angle == half {
        Some(QuarterPhase::MinusOne)
    } else if angle == three_quarters {
        Some(QuarterPhase::MinusI)
    } else {
        None
    }
}

// ============================================================================
// Gate constructors - Single qubit rotations
// ============================================================================

/// Rotation around X axis by the given angle.
///
/// For multiple qubits, use `RXs(angle, [0, 1, 2])`.
#[must_use]
#[allow(non_snake_case)]
pub fn RX(angle: Angle64, qubit: impl Into<QubitId>) -> Operator {
    Operator::rotation(RotationType::RX, angle, smallvec::smallvec![qubit.into().0])
}

/// RX rotations on multiple qubits.
///
/// `RXs(angle, [0, 1, 2])` is equivalent to `RX(angle, 0) & RX(angle, 1) & RX(angle, 2)`
#[must_use]
#[allow(non_snake_case)]
pub fn RXs(angle: Angle64, qubits: impl Into<Qubits>) -> Operator {
    qubits
        .into()
        .apply(|q| Operator::rotation(RotationType::RX, angle, smallvec::smallvec![q]))
}

/// Rotation around Y axis by the given angle.
///
/// For multiple qubits, use `RYs(angle, [0, 1, 2])`.
#[must_use]
#[allow(non_snake_case)]
pub fn RY(angle: Angle64, qubit: impl Into<QubitId>) -> Operator {
    Operator::rotation(RotationType::RY, angle, smallvec::smallvec![qubit.into().0])
}

/// RY rotations on multiple qubits.
///
/// `RYs(angle, [0, 1, 2])` is equivalent to `RY(angle, 0) & RY(angle, 1) & RY(angle, 2)`
#[must_use]
#[allow(non_snake_case)]
pub fn RYs(angle: Angle64, qubits: impl Into<Qubits>) -> Operator {
    qubits
        .into()
        .apply(|q| Operator::rotation(RotationType::RY, angle, smallvec::smallvec![q]))
}

/// Rotation around Z axis by the given angle.
///
/// For multiple qubits, use `RZs(angle, [0, 1, 2])`.
#[must_use]
#[allow(non_snake_case)]
pub fn RZ(angle: Angle64, qubit: impl Into<QubitId>) -> Operator {
    Operator::rotation(RotationType::RZ, angle, smallvec::smallvec![qubit.into().0])
}

/// RZ rotations on multiple qubits.
///
/// `RZs(angle, [0, 1, 2])` is equivalent to `RZ(angle, 0) & RZ(angle, 1) & RZ(angle, 2)`
#[must_use]
#[allow(non_snake_case)]
pub fn RZs(angle: Angle64, qubits: impl Into<Qubits>) -> Operator {
    qubits
        .into()
        .apply(|q| Operator::rotation(RotationType::RZ, angle, smallvec::smallvec![q]))
}

// ============================================================================
// Gate constructors - Two qubit rotations
// ============================================================================

/// Two-qubit XX rotation by the given angle.
///
/// For multiple pairs, use `RXXs(angle, [(0, 1), (2, 3)])` or tensor.
#[must_use]
#[allow(non_snake_case)]
pub fn RXX(angle: Angle64, q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> Operator {
    Operator::rotation(
        RotationType::RXX,
        angle,
        smallvec::smallvec![q0.into().0, q1.into().0],
    )
}

/// RXX rotations on multiple qubit pairs.
///
/// `RXXs(angle, [(0, 1), (2, 3)])` is equivalent to `RXX(angle, 0, 1) & RXX(angle, 2, 3)`
#[must_use]
#[allow(non_snake_case)]
pub fn RXXs(angle: Angle64, pairs: impl Into<QubitPairs>) -> Operator {
    pairs
        .into()
        .apply(|q0, q1| Operator::rotation(RotationType::RXX, angle, smallvec::smallvec![q0, q1]))
}

/// Two-qubit YY rotation by the given angle.
///
/// For multiple pairs, use `RYYs(angle, [(0, 1), (2, 3)])` or tensor.
#[must_use]
#[allow(non_snake_case)]
pub fn RYY(angle: Angle64, q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> Operator {
    Operator::rotation(
        RotationType::RYY,
        angle,
        smallvec::smallvec![q0.into().0, q1.into().0],
    )
}

/// RYY rotations on multiple qubit pairs.
///
/// `RYYs(angle, [(0, 1), (2, 3)])` is equivalent to `RYY(angle, 0, 1) & RYY(angle, 2, 3)`
#[must_use]
#[allow(non_snake_case)]
pub fn RYYs(angle: Angle64, pairs: impl Into<QubitPairs>) -> Operator {
    pairs
        .into()
        .apply(|q0, q1| Operator::rotation(RotationType::RYY, angle, smallvec::smallvec![q0, q1]))
}

/// Two-qubit ZZ rotation by the given angle.
///
/// For multiple pairs, use `RZZs(angle, [(0, 1), (2, 3)])` or tensor.
#[must_use]
#[allow(non_snake_case)]
pub fn RZZ(angle: Angle64, q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> Operator {
    Operator::rotation(
        RotationType::RZZ,
        angle,
        smallvec::smallvec![q0.into().0, q1.into().0],
    )
}

/// RZZ rotations on multiple qubit pairs.
///
/// `RZZs(angle, [(0, 1), (2, 3)])` is equivalent to `RZZ(angle, 0, 1) & RZZ(angle, 2, 3)`
#[must_use]
#[allow(non_snake_case)]
pub fn RZZs(angle: Angle64, pairs: impl Into<QubitPairs>) -> Operator {
    pairs
        .into()
        .apply(|q0, q1| Operator::rotation(RotationType::RZZ, angle, smallvec::smallvec![q0, q1]))
}

// ============================================================================
// Gate constructors - Named single-qubit Cliffords
// ============================================================================

/// Identity gate on a single qubit.
#[must_use]
#[allow(non_snake_case)]
pub fn I(qubit: impl Into<QubitId>) -> Operator {
    RZ(Angle64::ZERO, qubit.into().0)
}

/// Identity gates on multiple qubits.
#[must_use]
#[allow(non_snake_case)]
pub fn Is(qubits: impl Into<Qubits>) -> Operator {
    qubits.into().apply(|q| RZ(Angle64::ZERO, q))
}

/// Pauli X operator on a single qubit.
///
/// For multiple qubits, use `Xs([0, 2, 5])` or tensor: `X(0) & X(2) & X(5)`
#[must_use]
#[allow(non_snake_case)]
pub fn X(qubit: impl Into<QubitId>) -> Operator {
    Operator::Pauli(PauliString::x(qubit.into().0))
}

/// Pauli X operators on multiple qubits.
///
/// `Xs([0, 2, 5])` is equivalent to `X(0) & X(2) & X(5)`
#[must_use]
#[allow(non_snake_case)]
pub fn Xs(qubits: impl Into<Qubits>) -> Operator {
    let qs = qubits.into();
    if qs.0.is_empty() {
        Operator::Pauli(PauliString::default())
    } else {
        let mut ps = PauliString::x(qs.0[0].0);
        for q in &qs.0[1..] {
            ps = ps & PauliString::x(q.0);
        }
        Operator::Pauli(ps)
    }
}

/// Pauli Y operator on a single qubit.
///
/// For multiple qubits, use `Ys([0, 2, 5])` or tensor: `Y(0) & Y(2) & Y(5)`
#[must_use]
#[allow(non_snake_case)]
pub fn Y(qubit: impl Into<QubitId>) -> Operator {
    Operator::Pauli(PauliString::y(qubit.into().0))
}

/// Pauli Y operators on multiple qubits.
///
/// `Ys([0, 2, 5])` is equivalent to `Y(0) & Y(2) & Y(5)`
#[must_use]
#[allow(non_snake_case)]
pub fn Ys(qubits: impl Into<Qubits>) -> Operator {
    let qs = qubits.into();
    if qs.0.is_empty() {
        Operator::Pauli(PauliString::default())
    } else {
        let mut ps = PauliString::y(qs.0[0].0);
        for q in &qs.0[1..] {
            ps = ps & PauliString::y(q.0);
        }
        Operator::Pauli(ps)
    }
}

/// Pauli Z operator on a single qubit.
///
/// For multiple qubits, use `Zs([0, 2, 5])` or tensor: `Z(0) & Z(2) & Z(5)`
#[must_use]
#[allow(non_snake_case)]
pub fn Z(qubit: impl Into<QubitId>) -> Operator {
    Operator::Pauli(PauliString::z(qubit.into().0))
}

/// Pauli Z operators on multiple qubits.
///
/// `Zs([0, 2, 5])` is equivalent to `Z(0) & Z(2) & Z(5)`
#[must_use]
#[allow(non_snake_case)]
pub fn Zs(qubits: impl Into<Qubits>) -> Operator {
    let qs = qubits.into();
    if qs.0.is_empty() {
        Operator::Pauli(PauliString::default())
    } else {
        let mut ps = PauliString::z(qs.0[0].0);
        for q in &qs.0[1..] {
            ps = ps & PauliString::z(q.0);
        }
        Operator::Pauli(ps)
    }
}

/// SX gate (sqrt X): RX(π/2)
#[must_use]
#[allow(non_snake_case)]
pub fn SX(qubit: impl Into<QubitId>) -> Operator {
    RX(Angle64::QUARTER_TURN, qubit.into().0)
}

/// SX gates on multiple qubits.
#[must_use]
#[allow(non_snake_case)]
pub fn SXs(qubits: impl Into<Qubits>) -> Operator {
    qubits.into().apply(|q| RX(Angle64::QUARTER_TURN, q))
}

/// SY gate (sqrt Y): RY(π/2)
#[must_use]
#[allow(non_snake_case)]
pub fn SY(qubit: impl Into<QubitId>) -> Operator {
    RY(Angle64::QUARTER_TURN, qubit.into().0)
}

/// SY gates on multiple qubits.
#[must_use]
#[allow(non_snake_case)]
pub fn SYs(qubits: impl Into<Qubits>) -> Operator {
    qubits.into().apply(|q| RY(Angle64::QUARTER_TURN, q))
}

/// SZ gate (sqrt Z): RZ(π/2)
#[must_use]
#[allow(non_snake_case)]
pub fn SZ(qubit: impl Into<QubitId>) -> Operator {
    RZ(Angle64::QUARTER_TURN, qubit.into().0)
}

/// SZ gates on multiple qubits.
#[must_use]
#[allow(non_snake_case)]
pub fn SZs(qubits: impl Into<Qubits>) -> Operator {
    qubits.into().apply(|q| RZ(Angle64::QUARTER_TURN, q))
}

/// T gate: RZ(π/4)
#[must_use]
#[allow(non_snake_case)]
pub fn T(qubit: impl Into<QubitId>) -> Operator {
    RZ(Angle64::HALF_TURN / 4, qubit.into().0)
}

/// T gates on multiple qubits.
#[must_use]
#[allow(non_snake_case)]
pub fn Ts(qubits: impl Into<Qubits>) -> Operator {
    qubits.into().apply(|q| RZ(Angle64::HALF_TURN / 4, q))
}

/// Hadamard gate: RZ(π) * RY(π/2) (up to global phase)
#[must_use]
#[allow(non_snake_case)]
pub fn H(qubit: impl Into<QubitId>) -> Operator {
    let q = qubit.into().0;
    Operator::Gate {
        gate_type: GateType::H,
        qubits: smallvec::smallvec![q],
    }
}

/// Hadamard gates on multiple qubits.
#[must_use]
#[allow(non_snake_case)]
pub fn Hs(qubits: impl Into<Qubits>) -> Operator {
    qubits.into().apply(|q| {
        Operator::Compose(vec![
            RZ(Angle64::HALF_TURN, q),
            RY(Angle64::QUARTER_TURN, q),
        ])
    })
}

// ============================================================================
// Gate constructors - Named two-qubit gates
// ============================================================================

/// CNOT (CX) gate.
///
/// For multiple pairs, use `CXs([(0, 1), (2, 3)])` or tensor: `CX(0, 1) & CX(2, 3)`
#[must_use]
#[allow(non_snake_case)]
pub fn CX(control: impl Into<QubitId>, target: impl Into<QubitId>) -> Operator {
    Operator::gate(
        GateType::CX,
        smallvec::smallvec![control.into().0, target.into().0],
    )
}

/// CX gates on multiple qubit pairs.
///
/// `CXs([(0, 1), (2, 3)])` is equivalent to `CX(0, 1) & CX(2, 3)`
#[must_use]
#[allow(non_snake_case)]
pub fn CXs(pairs: impl Into<QubitPairs>) -> Operator {
    pairs
        .into()
        .apply(|ctrl, tgt| Operator::gate(GateType::CX, smallvec::smallvec![ctrl, tgt]))
}

/// Controlled-Y gate.
///
/// For multiple pairs, use `CYs([(0, 1), (2, 3)])` or tensor: `CY(0, 1) & CY(2, 3)`
#[must_use]
#[allow(non_snake_case)]
pub fn CY(control: impl Into<QubitId>, target: impl Into<QubitId>) -> Operator {
    Operator::gate(
        GateType::CY,
        smallvec::smallvec![control.into().0, target.into().0],
    )
}

/// CY gates on multiple qubit pairs.
///
/// `CYs([(0, 1), (2, 3)])` is equivalent to `CY(0, 1) & CY(2, 3)`
#[must_use]
#[allow(non_snake_case)]
pub fn CYs(pairs: impl Into<QubitPairs>) -> Operator {
    pairs
        .into()
        .apply(|ctrl, tgt| Operator::gate(GateType::CY, smallvec::smallvec![ctrl, tgt]))
}

/// Controlled-Z gate.
///
/// For multiple pairs, use `CZs([(0, 1), (2, 3)])` or tensor: `CZ(0, 1) & CZ(2, 3)`
#[must_use]
#[allow(non_snake_case)]
pub fn CZ(q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> Operator {
    Operator::gate(GateType::CZ, smallvec::smallvec![q0.into().0, q1.into().0])
}

/// CZ gates on multiple qubit pairs.
///
/// `CZs([(0, 1), (2, 3)])` is equivalent to `CZ(0, 1) & CZ(2, 3)`
#[must_use]
#[allow(non_snake_case)]
pub fn CZs(pairs: impl Into<QubitPairs>) -> Operator {
    pairs
        .into()
        .apply(|q0, q1| Operator::gate(GateType::CZ, smallvec::smallvec![q0, q1]))
}

/// SWAP gate.
///
/// For multiple pairs, use `SWAPs([(0, 1), (2, 3)])` or tensor: `SWAP(0, 1) & SWAP(2, 3)`
#[must_use]
#[allow(non_snake_case)]
pub fn SWAP(q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> Operator {
    Operator::gate(
        GateType::SWAP,
        smallvec::smallvec![q0.into().0, q1.into().0],
    )
}

/// SWAP gates on multiple qubit pairs.
///
/// `SWAPs([(0, 1), (2, 3)])` is equivalent to `SWAP(0, 1) & SWAP(2, 3)`
#[must_use]
#[allow(non_snake_case)]
pub fn SWAPs(pairs: impl Into<QubitPairs>) -> Operator {
    pairs
        .into()
        .apply(|q0, q1| Operator::gate(GateType::SWAP, smallvec::smallvec![q0, q1]))
}

/// SZZ gate: RZZ(π/2)
///
/// For multiple pairs, use `SZZs([(0, 1), (2, 3)])` or tensor: `SZZ(0, 1) & SZZ(2, 3)`
#[must_use]
#[allow(non_snake_case)]
pub fn SZZ(q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> Operator {
    Operator::rotation(
        RotationType::RZZ,
        Angle64::QUARTER_TURN,
        smallvec::smallvec![q0.into().0, q1.into().0],
    )
}

/// SZZ gates on multiple qubit pairs.
///
/// `SZZs([(0, 1), (2, 3)])` is equivalent to `SZZ(0, 1) & SZZ(2, 3)`
#[must_use]
#[allow(non_snake_case)]
pub fn SZZs(pairs: impl Into<QubitPairs>) -> Operator {
    pairs.into().apply(|q0, q1| {
        Operator::rotation(
            RotationType::RZZ,
            Angle64::QUARTER_TURN,
            smallvec::smallvec![q0, q1],
        )
    })
}

// ============================================================================
// Gate constructors - Three-qubit gates
// ============================================================================

/// Toffoli (CCX) gate.
#[must_use]
#[allow(non_snake_case)]
pub fn CCX(c0: impl Into<QubitId>, c1: impl Into<QubitId>, target: impl Into<QubitId>) -> Operator {
    Operator::gate(
        GateType::CCX,
        smallvec::smallvec![c0.into().0, c1.into().0, target.into().0],
    )
}

// ============================================================================
// Operator implementations
// ============================================================================

// Tensor product: &
impl BitAnd for Operator {
    type Output = Operator;

    fn bitand(self, rhs: Operator) -> Operator {
        match (self, rhs) {
            // Pauli & Pauli: use PauliString tensor product
            (Operator::Pauli(a), Operator::Pauli(b)) => Operator::Pauli(a & b),
            // Flatten nested tensors
            (Operator::Tensor(mut parts), Operator::Tensor(rhs_parts)) => {
                parts.extend(rhs_parts);
                Operator::Tensor(parts)
            }
            (Operator::Tensor(mut parts), rhs) => {
                parts.push(rhs);
                Operator::Tensor(parts)
            }
            (lhs, Operator::Tensor(mut parts)) => {
                parts.insert(0, lhs);
                Operator::Tensor(parts)
            }
            (lhs, rhs) => Operator::Tensor(vec![lhs, rhs]),
        }
    }
}

// Composition: *
impl Mul for Operator {
    type Output = Operator;

    fn mul(self, rhs: Operator) -> Operator {
        // A * B means apply B first, then A (matrix multiplication order)
        // So we store as [B, A] in the Compose vec (application order)
        match (self, rhs) {
            // Pauli * Pauli: use PauliString algebra
            (Operator::Pauli(a), Operator::Pauli(b)) => Operator::Pauli(a * b),
            // Flatten nested compositions
            (Operator::Compose(lhs_parts), Operator::Compose(rhs_parts)) => {
                // rhs applied first, then lhs
                let mut result = rhs_parts;
                result.extend(lhs_parts);
                Operator::Compose(result)
            }
            (Operator::Compose(lhs_parts), rhs) => {
                // rhs applied first
                let mut result = vec![rhs];
                result.extend(lhs_parts);
                Operator::Compose(result)
            }
            (lhs, Operator::Compose(mut rhs_parts)) => {
                // rhs_parts applied first, then lhs
                rhs_parts.push(lhs);
                Operator::Compose(rhs_parts)
            }
            (lhs, rhs) => {
                // rhs applied first, then lhs
                Operator::Compose(vec![rhs, lhs])
            }
        }
    }
}

// ============================================================================
// Circuit diagram generation
// ============================================================================

impl Operator {
    /// Generates an ASCII circuit diagram for this expression.
    #[must_use]
    pub fn to_diagram(&self, num_qubits: usize) -> String {
        let mut diagram = CircuitDiagram::new(num_qubits);
        self.add_to_diagram(&mut diagram);
        diagram.render()
    }

    fn add_to_diagram(&self, diagram: &mut CircuitDiagram) {
        match self {
            Self::Pauli(ps) => {
                // Draw each Pauli on its qubit
                for (pauli, qubit) in ps.iter_pairs() {
                    let q = usize::from(qubit);
                    let name = match pauli {
                        crate::Pauli::I => continue,
                        crate::Pauli::X => "X",
                        crate::Pauli::Y => "Y",
                        crate::Pauli::Z => "Z",
                    };
                    diagram.add_single_gate(q, name);
                }
            }
            Self::Rotation {
                rotation_type,
                angle,
                qubits,
            } => {
                let name = if let Some(gate_type) = rotation_to_gate_type(*rotation_type, *angle) {
                    format!("{gate_type:?}")
                } else {
                    format!("{rotation_type:?}")
                };

                if qubits.len() == 1 {
                    diagram.add_single_gate(qubits[0], &name);
                } else if qubits.len() == 2 {
                    diagram.add_two_qubit_gate(qubits[0], qubits[1], &name);
                }
            }
            Self::Gate { gate_type, qubits } => match gate_type {
                GateType::CX => {
                    diagram.add_controlled_gate(qubits[0], qubits[1], "X");
                }
                GateType::CY => {
                    diagram.add_controlled_gate(qubits[0], qubits[1], "Y");
                }
                GateType::CZ => {
                    diagram.add_controlled_gate(qubits[0], qubits[1], "Z");
                }
                GateType::SWAP => {
                    diagram.add_swap(qubits[0], qubits[1]);
                }
                GateType::CCX => {
                    diagram.add_toffoli(qubits[0], qubits[1], qubits[2]);
                }
                _ => {
                    if qubits.len() == 1 {
                        diagram.add_single_gate(qubits[0], &format!("{gate_type:?}"));
                    }
                }
            },
            Self::Tensor(parts) => {
                // Tensor products can be drawn simultaneously
                for part in parts {
                    part.add_to_diagram(diagram);
                }
            }
            Self::Compose(parts) => {
                // Sequential composition: draw in order
                for part in parts {
                    part.add_to_diagram(diagram);
                    diagram.advance();
                }
            }
            Self::Adjoint(inner) => {
                // Mark as adjoint somehow?
                inner.add_to_diagram(diagram);
            }
            Self::Phase { inner, .. } => {
                // Global phase doesn't appear in circuit diagrams
                inner.add_to_diagram(diagram);
            }
        }
    }
}

struct CircuitDiagram {
    num_qubits: usize,
    columns: Vec<Vec<String>>,
    current_col: usize,
}

impl CircuitDiagram {
    fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            columns: vec![vec![String::new(); num_qubits * 2 - 1]],
            current_col: 0,
        }
    }

    fn ensure_column(&mut self) {
        if self.current_col >= self.columns.len() {
            self.columns
                .push(vec![String::new(); self.num_qubits * 2 - 1]);
        }
    }

    fn advance(&mut self) {
        self.current_col += 1;
    }

    fn add_single_gate(&mut self, qubit: usize, name: &str) {
        self.ensure_column();
        let row = qubit * 2;
        if row < self.columns[self.current_col].len() {
            self.columns[self.current_col][row] = format!("[{name}]");
        }
    }

    fn add_controlled_gate(&mut self, control: usize, target: usize, target_name: &str) {
        self.ensure_column();
        let ctrl_row = control * 2;
        let targ_row = target * 2;

        if ctrl_row < self.columns[self.current_col].len() {
            self.columns[self.current_col][ctrl_row] = "●".to_string();
        }
        if targ_row < self.columns[self.current_col].len() {
            self.columns[self.current_col][targ_row] = format!("[{target_name}]");
        }

        // Draw vertical line
        let (min_row, max_row) = if ctrl_row < targ_row {
            (ctrl_row, targ_row)
        } else {
            (targ_row, ctrl_row)
        };
        for row in (min_row + 1)..max_row {
            if row % 2 == 1 && self.columns[self.current_col][row].is_empty() {
                self.columns[self.current_col][row] = "│".to_string();
            }
        }
    }

    fn add_swap(&mut self, q0: usize, q1: usize) {
        self.ensure_column();
        let row0 = q0 * 2;
        let row1 = q1 * 2;

        if row0 < self.columns[self.current_col].len() {
            self.columns[self.current_col][row0] = "×".to_string();
        }
        if row1 < self.columns[self.current_col].len() {
            self.columns[self.current_col][row1] = "×".to_string();
        }

        // Draw vertical line
        let (min_row, max_row) = (row0.min(row1), row0.max(row1));
        for row in (min_row + 1)..max_row {
            if row % 2 == 1 && self.columns[self.current_col][row].is_empty() {
                self.columns[self.current_col][row] = "│".to_string();
            }
        }
    }

    fn add_toffoli(&mut self, c0: usize, c1: usize, target: usize) {
        self.ensure_column();
        let c0_row = c0 * 2;
        let c1_row = c1 * 2;
        let targ_row = target * 2;

        if c0_row < self.columns[self.current_col].len() {
            self.columns[self.current_col][c0_row] = "●".to_string();
        }
        if c1_row < self.columns[self.current_col].len() {
            self.columns[self.current_col][c1_row] = "●".to_string();
        }
        if targ_row < self.columns[self.current_col].len() {
            self.columns[self.current_col][targ_row] = "[X]".to_string();
        }

        // Draw vertical lines
        let min_row = c0_row.min(c1_row).min(targ_row);
        let max_row = c0_row.max(c1_row).max(targ_row);
        for row in (min_row + 1)..max_row {
            if row % 2 == 1 && self.columns[self.current_col][row].is_empty() {
                self.columns[self.current_col][row] = "│".to_string();
            }
        }
    }

    fn add_two_qubit_gate(&mut self, q0: usize, q1: usize, name: &str) {
        self.ensure_column();
        let row0 = q0 * 2;
        let row1 = q1 * 2;

        if row0 < self.columns[self.current_col].len() {
            self.columns[self.current_col][row0] = format!("[{name}]");
        }
        if row1 < self.columns[self.current_col].len() {
            self.columns[self.current_col][row1] = format!("[{name}]");
        }

        // Draw vertical line
        let (min_row, max_row) = (row0.min(row1), row0.max(row1));
        for row in (min_row + 1)..max_row {
            if row % 2 == 1 && self.columns[self.current_col][row].is_empty() {
                self.columns[self.current_col][row] = "│".to_string();
            }
        }
    }

    fn render(&self) -> String {
        let lines: Vec<String> = (0..self.num_qubits).map(|q| format!("q{q}: ")).collect();

        // Add spacing lines between qubits
        let mut all_lines: Vec<String> = Vec::new();
        for (idx, line) in lines.iter().enumerate() {
            all_lines.push(line.clone());
            if idx < self.num_qubits - 1 {
                all_lines.push("    ".to_string()); // spacing line
            }
        }

        // Process each column
        for col in &self.columns {
            // Find max width in this column
            let max_width = col
                .iter()
                .map(|s| s.chars().count())
                .max()
                .unwrap_or(0)
                .max(3);

            for (row, cell) in col.iter().enumerate() {
                if row < all_lines.len() {
                    if cell.is_empty() {
                        // Wire or empty
                        if row % 2 == 0 {
                            all_lines[row].push_str(&"─".repeat(max_width));
                        } else {
                            all_lines[row].push_str(&" ".repeat(max_width));
                        }
                    } else {
                        // Center the cell content
                        let padding = max_width.saturating_sub(cell.chars().count());
                        let left_pad = padding / 2;
                        let right_pad = padding - left_pad;

                        if row % 2 == 0 {
                            // Qubit line
                            all_lines[row].push_str(&"─".repeat(left_pad));
                            all_lines[row].push_str(cell);
                            all_lines[row].push_str(&"─".repeat(right_pad));
                        } else {
                            // Spacing line
                            all_lines[row].push_str(&" ".repeat(left_pad));
                            all_lines[row].push_str(cell);
                            all_lines[row].push_str(&" ".repeat(right_pad));
                        }
                    }
                }
            }
        }

        // Add trailing wire
        for (idx, line) in all_lines.iter_mut().enumerate() {
            if idx % 2 == 0 {
                line.push('─');
            }
        }

        all_lines.join("\n")
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_qubit_gates() {
        let x = X(0);
        let z = Z(1);
        let h = H(0);

        assert!(x.is_clifford());
        assert!(z.is_clifford());
        assert!(h.is_clifford());
    }

    #[test]
    fn test_t_gate_not_clifford() {
        let t = T(0);
        assert!(!t.is_clifford());
    }

    #[test]
    fn test_tensor_product() {
        let op = X(0) & Z(1);
        assert!(op.is_clifford());

        // Pauli & Pauli now produces a single Pauli variant
        if let Operator::Pauli(ps) = &op {
            assert_eq!(ps.get(0), crate::Pauli::X);
            assert_eq!(ps.get(1), crate::Pauli::Z);
        } else {
            panic!("Expected Pauli, got {op:?}");
        }

        // Mixed tensor (Pauli with non-Pauli) produces Tensor
        let mixed = X(0) & H(1);
        assert!(matches!(mixed, Operator::Tensor(_)));
    }

    #[test]
    fn test_composition() {
        let circuit = T(0) * H(0);
        assert!(!circuit.is_clifford()); // T is not Clifford

        let cliff_circuit = SZ(0) * H(0);
        assert!(cliff_circuit.is_clifford());
    }

    #[test]
    fn test_control_gates() {
        let cx = CX(0, 1);
        let cz = CZ(0, 1);

        assert!(cx.is_clifford());
        assert!(cz.is_clifford());
    }

    #[test]
    fn test_adjoint() {
        let t = T(0);
        let t_dg = t.dg();

        // T† should have negated angle
        if let Operator::Rotation { angle, .. } = t_dg {
            let t_angle = Angle64::HALF_TURN / 4;
            let expected = negate_angle(t_angle);
            assert_eq!(angle, expected);
        } else {
            panic!("Expected Rotation");
        }
    }

    #[test]
    fn test_double_adjoint() {
        let h = H(0);
        let h_dg_dg = h.dg().dg();

        // H†† should equal H (structurally)
        assert_eq!(h, h_dg_dg);
    }

    #[test]
    fn test_qubits() {
        let circuit = CX(0, 1) * H(0) * T(2);
        let qubits = circuit.qubits();
        assert_eq!(qubits, vec![0, 1, 2]);
    }

    #[test]
    fn test_to_named_gate() {
        let sz = SZ(0);
        assert_eq!(sz.to_named_gate(), Some(GateType::SZ));

        let t = T(0);
        assert_eq!(t.to_named_gate(), Some(GateType::T));

        let x = X(0);
        assert_eq!(x.to_named_gate(), Some(GateType::X));
    }

    #[test]
    fn test_diagram_single_qubit() {
        let h = H(0);
        let diagram = h.to_diagram(1);
        assert!(diagram.contains("[H]"));
    }

    #[test]
    fn test_diagram_cx() {
        let cx = CX(0, 1);
        let diagram = cx.to_diagram(2);
        assert!(diagram.contains("●"));
        assert!(diagram.contains("[X]"));
    }

    #[test]
    fn test_diagram_complex() {
        // Build a circuit: H(0), CX(0,1), T(1)
        let circuit = T(1) * CX(0, 1) * H(0);
        let diagram = circuit.to_diagram(2);
        println!("Circuit diagram:\n{diagram}");

        // Also test a 3-qubit circuit
        let circuit3 = CCX(0, 1, 2) * H(0) * H(1);
        let diagram3 = circuit3.to_diagram(3);
        println!("\n3-qubit circuit:\n{diagram3}");
    }

    #[test]
    fn test_composition_order() {
        // A * B means apply B first, then A
        // Test composition with rotations
        let circuit = SZ(0) * SY(0) * SX(0); // Apply SX, then SY, then SZ

        if let Operator::Compose(parts) = circuit {
            assert_eq!(parts.len(), 3);
            // All should be rotations
            assert!(matches!(&parts[0], Operator::Rotation { .. }));
            assert!(matches!(&parts[1], Operator::Rotation { .. }));
            assert!(matches!(&parts[2], Operator::Rotation { .. }));
        } else {
            panic!("Expected Compose");
        }
    }

    // ========================================================================
    // Simplify tests
    // ========================================================================

    #[test]
    fn test_simplify_identity() {
        let id = I(0);
        assert!(id.is_identity());
    }

    #[test]
    fn test_simplify_cancellation() {
        // T * T† should simplify to identity
        let t = T(0);
        let t_dg = t.dg();
        let circuit = t.clone() * t_dg;
        let simplified = circuit.simplify();

        // Should be identity (zero-angle rotation)
        assert!(simplified.is_identity());
    }

    #[test]
    fn test_simplify_merge_adjacent() {
        // SZ * SZ = Z (RZ(π/2) + RZ(π/2) = RZ(π))
        let circuit = SZ(0) * SZ(0);
        let simplified = circuit.simplify();

        // Should merge into a single rotation
        if let Operator::Rotation {
            angle,
            rotation_type,
            ..
        } = simplified
        {
            assert_eq!(rotation_type, RotationType::RZ);
            assert_eq!(angle, Angle64::HALF_TURN); // π
        } else {
            panic!("Expected single Rotation, got {simplified:?}");
        }
    }

    #[test]
    fn test_simplify_preserves_different_qubits() {
        // RZ on different qubits shouldn't merge
        let circuit = RZ(Angle64::QUARTER_TURN, 0) * RZ(Angle64::QUARTER_TURN, 1);
        let simplified = circuit.simplify();

        // Should remain a Compose with 2 parts
        if let Operator::Compose(parts) = simplified {
            assert_eq!(parts.len(), 2);
        } else {
            panic!("Expected Compose");
        }
    }

    #[test]
    fn test_simplify_preserves_different_rotation_types() {
        // RX and RZ on same qubit shouldn't merge
        let circuit = RX(Angle64::QUARTER_TURN, 0) * RZ(Angle64::QUARTER_TURN, 0);
        let simplified = circuit.simplify();

        // Should remain a Compose with 2 parts
        if let Operator::Compose(parts) = simplified {
            assert_eq!(parts.len(), 2);
        } else {
            panic!("Expected Compose");
        }
    }

    #[test]
    fn test_simplify_multiple_merges() {
        // T * T * T * T = Z (4 * π/4 = π)
        let circuit = T(0) * T(0) * T(0) * T(0);
        let simplified = circuit.simplify();

        if let Operator::Rotation { angle, .. } = simplified {
            assert_eq!(angle, Angle64::HALF_TURN);
        } else {
            panic!("Expected single Rotation");
        }
    }

    #[test]
    fn test_simplify_tensor_preserves_identity() {
        // X(0) & I(1) should preserve the identity to maintain Hilbert space dimension
        let circuit = X(0) & I(1);
        let simplified = circuit.simplify();

        // Should still be a tensor with both parts (preserves 2-qubit space)
        let qubits = simplified.qubits();
        assert!(qubits.contains(&0));
        assert!(qubits.contains(&1));
    }

    #[test]
    fn test_simplify_compose_with_gate() {
        // CX doesn't merge with rotations
        let circuit = SZ(0) * CX(0, 1) * SZ(0);
        let simplified = circuit.simplify();

        // Should have 3 parts (S, CX, S)
        if let Operator::Compose(parts) = simplified {
            assert_eq!(parts.len(), 3);
        } else {
            panic!("Expected Compose");
        }
    }

    // ========================================================================
    // CliffordRep conversion tests
    // ========================================================================

    #[test]
    fn test_to_clifford_rep_non_clifford_returns_none() {
        let t = T(0);
        assert!(t.to_clifford_rep(1).is_none());
    }

    #[test]
    fn test_to_clifford_rep_identity() {
        let id = I(0);
        let cliff = id.to_clifford_rep(1).unwrap();

        // Identity should transform X -> X, Z -> Z
        let x0 = PauliString::x(0);
        let z0 = PauliString::z(0);

        let tx = cliff.apply(&x0);
        let tz = cliff.apply(&z0);

        assert_eq!(tx.get(0), crate::Pauli::X);
        assert_eq!(tz.get(0), crate::Pauli::Z);
    }

    #[test]
    fn test_to_clifford_rep_x_gate() {
        let x = X(0);
        let cliff = x.to_clifford_rep(1).unwrap();

        // X gate: X -> X, Z -> -Z
        let z0 = PauliString::z(0);

        let tz = cliff.apply(&z0);
        assert_eq!(tz.get(0), crate::Pauli::Z);
        assert_eq!(tz.phase(), crate::QuarterPhase::MinusOne);
    }

    #[test]
    fn test_to_clifford_rep_s_gate() {
        let s = SZ(0);
        let cliff = s.to_clifford_rep(1).unwrap();

        // SZ gate: X -> Y, Z -> Z
        let x0 = PauliString::x(0);

        let tx = cliff.apply(&x0);
        assert_eq!(tx.get(0), crate::Pauli::Y);
    }

    #[test]
    fn test_to_clifford_rep_cx_gate() {
        let cx = CX(0, 1);
        let cliff = cx.to_clifford_rep(2).unwrap();

        // CX: X_control -> X_control * X_target
        let x0 = PauliString::x(0);

        let tx = cliff.apply(&x0);
        // Should have X on both qubits
        assert_eq!(tx.get(0), crate::Pauli::X);
        assert_eq!(tx.get(1), crate::Pauli::X);
    }

    #[test]
    fn test_to_clifford_rep_composition() {
        // S * H should be convertible
        let circuit = SZ(0) * H(0);
        let cliff = circuit.to_clifford_rep(1);
        assert!(cliff.is_some());
    }

    // ========================================================================
    // Phase tests
    // ========================================================================

    #[test]
    fn test_phase_basic() {
        // phase(π/4) * X should create a phased operator
        // Since π/4 is not a quarter-turn multiple, it wraps in Phase
        let eighth_turn = Angle64::HALF_TURN / 4;
        let op = phase(eighth_turn) * X(0);

        if let Operator::Phase { phase: p, inner } = op {
            assert_eq!(p, eighth_turn);
            // Inner should be Pauli(X)
            assert!(matches!(*inner, Operator::Pauli(_)));
        } else {
            panic!("Expected Phase variant, got {op:?}");
        }
    }

    #[test]
    fn test_phase_negation() {
        // -phase(θ) = phase(θ + π)
        let quarter = Angle64::QUARTER_TURN;
        let p = phase(quarter);
        let neg_p = -p;

        assert_eq!(neg_p.0, quarter + Angle64::HALF_TURN);
    }

    #[test]
    fn test_phase_times_i() {
        // i * phase(θ) = phase(θ + π/2)
        let quarter = Angle64::QUARTER_TURN;
        let p = phase(quarter);
        let ip = i * p;

        assert_eq!(ip.0, quarter + Angle64::QUARTER_TURN);
    }

    #[test]
    fn test_phase_times_neg_i() {
        // -i * phase(θ) = phase(θ + 3π/2)
        let quarter = Angle64::QUARTER_TURN;
        let p = phase(quarter);
        let nip = -i * p;

        assert_eq!(nip.0, quarter + Angle64::QUARTER_TURN + Angle64::HALF_TURN);
    }

    #[test]
    fn test_phase_equivalence_with_i() {
        // phase(π/2) * X should be equivalent to i * X
        // Since π/2 is a quarter turn, the phase gets absorbed into the PauliString
        let op1 = phase(Angle64::QUARTER_TURN) * X(0);
        let op2 = i * X(0);

        // Both should be Pauli with phase +i
        if let (Operator::Pauli(ps1), Operator::Pauli(ps2)) = (&op1, &op2) {
            assert_eq!(ps1.phase(), QuarterPhase::PlusI);
            assert_eq!(ps2.phase(), QuarterPhase::PlusI);
        } else {
            panic!("Expected Pauli variants, got {op1:?} and {op2:?}");
        }
    }

    #[test]
    fn test_phase_zero_is_identity() {
        // phase(0) * X should simplify to just X
        let op = phase(Angle64::ZERO) * X(0);

        // with_phase returns self when phase is zero
        assert!(matches!(op, Operator::Pauli(_)));
        if let Operator::Pauli(ps) = op {
            assert_eq!(ps.phase(), QuarterPhase::PlusOne);
        }
    }

    // ========================================================================
    // Macro tests
    // ========================================================================

    #[test]
    fn test_angle_macro_pi() {
        assert_eq!(crate::angle!(pi), Angle64::HALF_TURN);
    }

    #[test]
    fn test_angle_macro_pi_over_2() {
        assert_eq!(crate::angle!(pi / 2), Angle64::QUARTER_TURN);
    }

    #[test]
    fn test_angle_macro_pi_over_4() {
        // pi/4 = 1/8 turn
        assert_eq!(crate::angle!(pi / 4), Angle64::from_turn_ratio(1, 8));
    }

    #[test]
    fn test_angle_macro_2_pi_over_3() {
        // 2*pi/3 = 2/6 = 1/3 turn
        assert_eq!(crate::angle!(2 * pi / 3), Angle64::from_turn_ratio(1, 3));
    }

    #[test]
    fn test_angle_macro_4_pi_over_3() {
        // 4*pi/3 = 4/6 = 2/3 turn
        assert_eq!(crate::angle!(4 * pi / 3), Angle64::from_turn_ratio(2, 3));
    }

    #[test]
    fn test_angle_macro_negative() {
        // -pi/2 should be the negative of pi/2
        let neg = crate::angle!(-pi / 2);
        let pos = crate::angle!(pi / 2);
        assert_eq!(neg, Angle64::ZERO - pos);
    }

    #[test]
    fn test_phase_macro_basic() {
        // phase!(pi/4) should create a PhaseValue
        let p = crate::phase!(pi / 4);
        assert_eq!(p.0, Angle64::from_turn_ratio(1, 8));
    }

    #[test]
    fn test_phase_macro_with_operator() {
        // phase!(pi/2) * X should be same as i * X
        // Since pi/2 is a quarter turn, the phase gets absorbed into PauliString
        let op1 = crate::phase!(pi / 2) * X(0);
        let op2 = i * X(0);

        // Both should be Pauli with phase +i
        if let (Operator::Pauli(ps1), Operator::Pauli(ps2)) = (&op1, &op2) {
            assert_eq!(ps1.phase(), QuarterPhase::PlusI);
            assert_eq!(ps2.phase(), QuarterPhase::PlusI);
        } else {
            panic!("Expected Pauli variants, got {op1:?} and {op2:?}");
        }
    }

    #[test]
    fn test_phase_macro_exact_cancellation() {
        // 8 * (pi/4) should exactly equal 2*pi = 0
        let eighth = crate::angle!(pi / 4);
        let full = eighth + eighth + eighth + eighth + eighth + eighth + eighth + eighth;
        assert_eq!(full, Angle64::ZERO);
    }

    // ========================================================================
    // Turn macro tests
    // ========================================================================

    #[test]
    fn test_turn_macro_quarter() {
        assert_eq!(crate::turn!(1 / 4), Angle64::QUARTER_TURN);
    }

    #[test]
    fn test_turn_macro_half() {
        assert_eq!(crate::turn!(1 / 2), Angle64::HALF_TURN);
    }

    #[test]
    fn test_turn_macro_eighth() {
        // 1/8 turn = T gate phase = pi/4 radians
        assert_eq!(crate::turn!(1 / 8), crate::angle!(pi / 4));
    }

    #[test]
    fn test_turn_macro_third() {
        // 1/3 turn
        assert_eq!(crate::turn!(1 / 3), Angle64::from_turn_ratio(1, 3));
    }

    #[test]
    fn test_turn_macro_two_thirds() {
        // 2/3 turn
        assert_eq!(crate::turn!(2 / 3), Angle64::from_turn_ratio(2, 3));
    }

    #[test]
    fn test_turn_vs_angle_equivalence() {
        // turn!(1/4) should equal angle!(pi/2)
        assert_eq!(crate::turn!(1 / 4), crate::angle!(pi / 2));

        // turn!(1/8) should equal angle!(pi/4)
        assert_eq!(crate::turn!(1 / 8), crate::angle!(pi / 4));

        // turn!(1/2) should equal angle!(pi)
        assert_eq!(crate::turn!(1 / 2), crate::angle!(pi));
    }

    #[test]
    fn test_phase_turn_macro_basic() {
        let p = crate::phase_turn!(1 / 8);
        assert_eq!(p.0, Angle64::from_turn_ratio(1, 8));
    }

    #[test]
    fn test_phase_turn_macro_with_operator() {
        // phase_turn!(1/4) * X should be same as i * X (quarter turn = i)
        // Since quarter turn is a quarter phase, it gets absorbed into the PauliString
        let op1 = crate::phase_turn!(1 / 4) * X(0);
        let op2 = i * X(0);

        // Both should be Pauli with phase +i
        if let (Operator::Pauli(ps1), Operator::Pauli(ps2)) = (&op1, &op2) {
            assert_eq!(ps1.phase(), QuarterPhase::PlusI);
            assert_eq!(ps2.phase(), QuarterPhase::PlusI);
        } else {
            panic!("Expected Pauli variants, got {op1:?} and {op2:?}");
        }
    }

    #[test]
    fn test_turn_exact_cancellation() {
        // 8 * (1/8 turn) should exactly equal 1 full turn = 0
        let eighth = crate::turn!(1 / 8);
        let full = eighth + eighth + eighth + eighth + eighth + eighth + eighth + eighth;
        assert_eq!(full, Angle64::ZERO);
    }

    // ========================================================================
    // Pauli equivalence tests
    // ========================================================================

    #[test]
    fn test_is_pauli_equivalent_pauli() {
        assert!(X(0).is_pauli_equivalent());
        assert!(Y(1).is_pauli_equivalent());
        assert!(Z(2).is_pauli_equivalent());
        assert!((X(0) & Y(1)).is_pauli_equivalent());
    }

    #[test]
    fn test_is_pauli_equivalent_rotation() {
        // Half-turn rotations are Pauli-equivalent
        assert!(RX(Angle64::HALF_TURN, 0).is_pauli_equivalent());
        assert!(RY(Angle64::HALF_TURN, 0).is_pauli_equivalent());
        assert!(RZ(Angle64::HALF_TURN, 0).is_pauli_equivalent());

        // Quarter-turn rotations are not
        assert!(!RX(Angle64::QUARTER_TURN, 0).is_pauli_equivalent());
        assert!(!RZ(Angle64::QUARTER_TURN, 0).is_pauli_equivalent());
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn test_try_to_pauli_rotation() {
        // RX(π) = X
        let rx_pi = RX(Angle64::HALF_TURN, 0);
        let converted = rx_pi.try_to_pauli().expect("Should convert");
        if let Operator::Pauli(ps) = converted {
            assert_eq!(ps.get(0), crate::Pauli::X);
        } else {
            panic!("Expected Pauli variant");
        }

        // RY(π) = Y
        let ry_pi = RY(Angle64::HALF_TURN, 1);
        let converted = ry_pi.try_to_pauli().expect("Should convert");
        if let Operator::Pauli(ps) = converted {
            assert_eq!(ps.get(1), crate::Pauli::Y);
        } else {
            panic!("Expected Pauli variant");
        }

        // RZ(π) = Z
        let rz_pi = RZ(Angle64::HALF_TURN, 2);
        let converted = rz_pi.try_to_pauli().expect("Should convert");
        if let Operator::Pauli(ps) = converted {
            assert_eq!(ps.get(2), crate::Pauli::Z);
        } else {
            panic!("Expected Pauli variant");
        }
    }

    #[test]
    fn test_try_to_pauli_non_pauli() {
        // Quarter-turn rotations should not convert
        assert!(RX(Angle64::QUARTER_TURN, 0).try_to_pauli().is_none());
        assert!(RZ(Angle64::QUARTER_TURN, 0).try_to_pauli().is_none());
    }

    // ========================================================================
    // Multi-qubit syntax tests
    // ========================================================================

    #[test]
    fn test_x_multi_qubit() {
        // Xs([0, 2, 5]) should be equivalent to X(0) & X(2) & X(5)
        let multi = Xs([0, 2, 5]);
        let tensor = X(0) & X(2) & X(5);

        // Both should be Pauli variants with the same content
        if let (Operator::Pauli(ps1), Operator::Pauli(ps2)) = (&multi, &tensor) {
            assert_eq!(ps1.get(0), crate::Pauli::X);
            assert_eq!(ps1.get(2), crate::Pauli::X);
            assert_eq!(ps1.get(5), crate::Pauli::X);
            assert_eq!(ps1, ps2);
        } else {
            panic!("Expected Pauli variants");
        }
    }

    #[test]
    fn test_t_multi_qubit() {
        // Ts([0, 1, 2]) should be a tensor of T gates
        let multi = Ts([0, 1, 2]);

        if let Operator::Tensor(parts) = multi {
            assert_eq!(parts.len(), 3);
            // Each should be a rotation
            assert!(matches!(&parts[0], Operator::Rotation { .. }));
            assert!(matches!(&parts[1], Operator::Rotation { .. }));
            assert!(matches!(&parts[2], Operator::Rotation { .. }));
        } else {
            panic!("Expected Tensor variant, got {multi:?}");
        }
    }

    #[test]
    fn test_h_multi_qubit() {
        // Hs([0, 1]) should be a tensor of H gates
        let multi = Hs([0, 1]);

        if let Operator::Tensor(parts) = multi {
            assert_eq!(parts.len(), 2);
            // Each should be a Compose (H = RZ * RY)
            assert!(matches!(&parts[0], Operator::Compose(_)));
            assert!(matches!(&parts[1], Operator::Compose(_)));
        } else {
            panic!("Expected Tensor variant, got {multi:?}");
        }
    }

    #[test]
    fn test_single_qubit_still_works() {
        // Single qubit syntax should still work
        let x = X(0);
        let t = T(1);
        let h = H(2);

        assert!(matches!(x, Operator::Pauli(_)));
        assert!(matches!(t, Operator::Rotation { .. }));
        assert!(matches!(
            h,
            Operator::Gate {
                gate_type: GateType::H,
                ..
            }
        ));
    }

    #[test]
    fn test_range_syntax() {
        // Range syntax: Xs(0..3) = X(0) & X(1) & X(2)
        let multi_range = Xs(0..3);
        let tensor = X(0) & X(1) & X(2);

        if let (Operator::Pauli(ps1), Operator::Pauli(ps2)) = (&multi_range, &tensor) {
            assert_eq!(ps1, ps2);
        } else {
            panic!("Expected Pauli variants");
        }
    }

    #[test]
    fn test_range_inclusive_syntax() {
        // RangeInclusive syntax: Zs(1..=3) = Z(1) & Z(2) & Z(3)
        let multi_range = Zs(1..=3);
        let tensor = Z(1) & Z(2) & Z(3);

        if let (Operator::Pauli(ps1), Operator::Pauli(ps2)) = (&multi_range, &tensor) {
            assert_eq!(ps1, ps2);
        } else {
            panic!("Expected Pauli variants");
        }
    }

    #[test]
    fn test_identity_range_syntax() {
        // Is(0..=2) should create identity operators on qubits 0, 1, 2
        let identities = Is(0..=2);

        if let Operator::Tensor(parts) = identities {
            assert_eq!(parts.len(), 3);
            for part in &parts {
                assert!(part.is_identity());
            }
        } else {
            panic!("Expected Tensor variant, got {identities:?}");
        }
    }

    #[test]
    #[should_panic(expected = "empty range not allowed")]
    fn test_empty_range_panics() {
        let _ = Xs(0..0);
    }

    #[test]
    #[should_panic(expected = "empty range not allowed")]
    #[allow(clippy::reversed_empty_ranges)] // Intentionally testing empty range
    fn test_empty_range_inclusive_panics() {
        // 1..=0 is empty
        let _ = Zs(1..=0);
    }

    #[test]
    fn test_single_element_range() {
        // Xs(0..1) should be equivalent to X(0)
        let from_range = Xs(0..1);
        let direct = X(0);

        if let (Operator::Pauli(ps1), Operator::Pauli(ps2)) = (&from_range, &direct) {
            assert_eq!(ps1, ps2);
        } else {
            panic!("Expected Pauli variants");
        }
    }

    #[test]
    fn test_single_element_range_inclusive() {
        // Ts(2..=2) should be equivalent to T(2)
        let from_range = Ts(2..=2);
        let direct = T(2);

        // Both should be Rotation variants
        if let (
            Operator::Rotation {
                angle: a1,
                rotation_type: r1,
                ..
            },
            Operator::Rotation {
                angle: a2,
                rotation_type: r2,
                ..
            },
        ) = (&from_range, &direct)
        {
            assert_eq!(a1, a2);
            assert_eq!(r1, r2);
        } else {
            panic!("Expected Rotation variants, got {from_range:?} and {direct:?}");
        }
    }

    // ========================================================================
    // Conjugation tests
    // ========================================================================
    // Two conjugation conventions:
    //   A.conj(U)   = U * A * U†  (stabilizer update: S →USU† when applying U)
    //   A.conjdg(U) = U† * A * U  (Heisenberg picture: A → U†AU)

    #[test]
    fn test_conj_pauli_by_pauli() {
        // X.conj(Z) = Z * X * Z† = Z * X * Z = -X (since Z is self-adjoint)
        // ZX = -XZ, so ZXZ = -XZZ = -X
        let x = X(0);
        let z = Z(0);
        let result = x.conj(&z);

        // conj returns a Compose, simplify to get the Pauli
        let simplified = result.simplify();
        if let Operator::Pauli(ps) = simplified {
            assert_eq!(ps.get(0), crate::Pauli::X);
            assert_eq!(ps.phase(), QuarterPhase::MinusOne);
        } else {
            panic!("Expected Pauli variant, got {simplified:?}");
        }
    }

    #[test]
    fn test_conjdg_pauli_by_pauli() {
        // X.conjdg(Z) = Z† * X * Z = Z * X * Z = -X (since Z is self-adjoint)
        // Same result as conj for self-adjoint gates
        let x = X(0);
        let z = Z(0);
        let result = x.conjdg(&z);

        let simplified = result.simplify();
        if let Operator::Pauli(ps) = simplified {
            assert_eq!(ps.get(0), crate::Pauli::X);
            assert_eq!(ps.phase(), QuarterPhase::MinusOne);
        } else {
            panic!("Expected Pauli variant, got {simplified:?}");
        }
    }

    #[test]
    fn test_conj_produces_compose() {
        // A.conj(B) = B * A * B† produces a Compose
        let x = X(0);
        let h = H(0);
        let result = x.conj(&h);

        // Result is Compose(H, X, H†)
        assert!(matches!(result, Operator::Compose(_)));
    }

    #[test]
    fn test_conj_z_by_z() {
        // Z.conj(Z) = Z * Z * Z† = Z * Z * Z = Z (since Z² = I, Z³ = Z)
        let z = Z(0);
        let result = z.conj(&z);

        let simplified = result.simplify();
        if let Operator::Pauli(ps) = simplified {
            assert_eq!(ps.get(0), crate::Pauli::Z);
            assert_eq!(ps.phase(), QuarterPhase::PlusOne);
        } else {
            panic!("Expected Pauli variant, got {simplified:?}");
        }
    }

    #[test]
    fn test_conj_structure_sz_gate() {
        // X.conj(SZ) = SZ * X * SZ† should produce Compose with SZ first, X middle, SZ† last
        let x = X(0);
        let sz = SZ(0);
        let result = x.conj(&sz);

        // Verify structure: Compose([SZ, X, SZ†])
        if let Operator::Compose(parts) = result {
            assert_eq!(parts.len(), 3);
            // First element is SZ (positive angle)
            assert!(matches!(
                &parts[0],
                Operator::Rotation {
                    rotation_type: RotationType::RZ,
                    ..
                }
            ));
            // Middle element is X
            assert!(matches!(&parts[1], Operator::Pauli(_)));
            // Last element is SZ† (negative angle)
            assert!(matches!(
                &parts[2],
                Operator::Rotation {
                    rotation_type: RotationType::RZ,
                    ..
                }
            ));
        } else {
            panic!("Expected Compose variant, got {result:?}");
        }
    }

    #[test]
    fn test_conjdg_structure_sz_gate() {
        // X.conjdg(SZ) = SZ† * X * SZ should produce Compose with SZ† first, X middle, SZ last
        let x = X(0);
        let sz = SZ(0);
        let result = x.conjdg(&sz);

        // Verify structure: Compose([SZ†, X, SZ])
        if let Operator::Compose(parts) = result {
            assert_eq!(parts.len(), 3);
            // First element is SZ† (negative angle)
            assert!(matches!(
                &parts[0],
                Operator::Rotation {
                    rotation_type: RotationType::RZ,
                    ..
                }
            ));
            // Middle element is X
            assert!(matches!(&parts[1], Operator::Pauli(_)));
            // Last element is SZ (positive angle)
            assert!(matches!(
                &parts[2],
                Operator::Rotation {
                    rotation_type: RotationType::RZ,
                    ..
                }
            ));
        } else {
            panic!("Expected Compose variant, got {result:?}");
        }
    }

    #[test]
    fn test_conj_conjdg_inverse_relationship() {
        // For any operator A and gate U:
        // A.conj(U).conjdg(U) should give back A (up to simplification)
        // Because (UAU†).conjdg(U) = U†(UAU†)U = A
        let x = X(0);
        let sz = SZ(0);

        let forward = x.clone().conj(&sz); // SZ X SZ†
        let back = forward.conjdg(&sz); // SZ† (SZ X SZ†) SZ = X

        let simplified = back.simplify();
        if let Operator::Pauli(ps) = simplified {
            assert_eq!(ps.get(0), crate::Pauli::X);
            assert_eq!(ps.phase(), QuarterPhase::PlusOne);
        } else {
            panic!("Expected Pauli variant, got {simplified:?}");
        }
    }

    // ========================================================================
    // Weight tests
    // ========================================================================

    #[test]
    fn test_weight_single_pauli() {
        assert_eq!(X(0).weight(), 1);
        assert_eq!(Y(1).weight(), 1);
        assert_eq!(Z(2).weight(), 1);
    }

    #[test]
    fn test_weight_identity() {
        // weight() returns number of qubits acted on
        // I(0) = RZ(0, 0) acts on qubit 0
        assert_eq!(I(0).weight(), 1);
    }

    #[test]
    fn test_weight_tensor_product() {
        let op = X(0) & Y(1) & Z(2);
        assert_eq!(op.weight(), 3);
    }

    #[test]
    fn test_weight_tensor_with_identity() {
        // Id tensored still counts as acting on that qubit
        let op = X(0) & I(1) & Z(2);
        assert_eq!(op.weight(), 3);
    }

    #[test]
    fn test_weight_rotation() {
        // Rotations have weight equal to the number of qubits they act on
        assert_eq!(RX(Angle64::QUARTER_TURN, 0).weight(), 1);
        assert_eq!(RZZ(Angle64::QUARTER_TURN, 0, 1).weight(), 2);
    }

    #[test]
    fn test_weight_gate() {
        assert_eq!(H(0).weight(), 1);
        assert_eq!(CX(0, 1).weight(), 2);
        assert_eq!(CCX(0, 1, 2).weight(), 3);
    }

    // ========================================================================
    // is_hermitian tests
    // ========================================================================

    #[test]
    fn test_is_hermitian_paulis() {
        // All Paulis are Hermitian
        assert!(X(0).is_hermitian());
        assert!(Y(0).is_hermitian());
        assert!(Z(0).is_hermitian());
        assert!(I(0).is_hermitian());
    }

    #[test]
    fn test_is_hermitian_pauli_tensor() {
        // Tensor products of Paulis are Hermitian
        let op = X(0) & Y(1) & Z(2);
        assert!(op.is_hermitian());
    }

    #[test]
    fn test_is_hermitian_hadamard() {
        // H is Hermitian (H = H†)
        assert!(H(0).is_hermitian());
    }

    #[test]
    fn test_is_hermitian_rotation_not() {
        // General rotations are not Hermitian (unless angle is 0 or π)
        assert!(!RZ(Angle64::QUARTER_TURN, 0).is_hermitian());
        assert!(!T(0).is_hermitian());
        assert!(!SZ(0).is_hermitian());
    }

    #[test]
    fn test_is_hermitian_rotation_half_turn() {
        // Half-turn rotations are Hermitian (up to global phase)
        // RX(π) = -iX, which is Hermitian
        assert!(RX(Angle64::HALF_TURN, 0).is_hermitian());
        assert!(RY(Angle64::HALF_TURN, 0).is_hermitian());
        assert!(RZ(Angle64::HALF_TURN, 0).is_hermitian());
    }

    // ========================================================================
    // pow tests
    // ========================================================================

    #[test]
    fn test_pow_zero() {
        // X^0 = I
        let x = X(0);
        let result = x.pow(0);
        assert!(result.is_identity());
    }

    #[test]
    fn test_pow_one() {
        // X^1 = X
        let x = X(0);
        let result = x.pow(1);
        if let Operator::Pauli(ps) = result {
            assert_eq!(ps.get(0), crate::Pauli::X);
        } else {
            panic!("Expected Pauli");
        }
    }

    #[test]
    fn test_pow_two_pauli() {
        // X^2 = I after simplification
        let x = X(0);
        let result = x.pow(2).simplify();
        assert!(result.is_identity());
    }

    #[test]
    fn test_pow_creates_compose() {
        // pow(n) creates a Compose of n copies without simplification
        let t = T(0);
        let result = t.pow(3);
        assert!(matches!(result, Operator::Compose(_)));
    }

    #[test]
    fn test_pow_rotation_simplify() {
        // T^2 = S (RZ(π/4)^2 = RZ(π/2)) after simplification
        let t = T(0);
        let result = t.pow(2).simplify();

        if let Operator::Rotation {
            angle,
            rotation_type,
            ..
        } = result
        {
            assert_eq!(rotation_type, RotationType::RZ);
            assert_eq!(angle, Angle64::QUARTER_TURN);
        } else {
            panic!("Expected Rotation, got {result:?}");
        }
    }

    #[test]
    fn test_pow_four_t_simplify() {
        // T^4 = Z (RZ(π/4)^4 = RZ(π)) after simplification
        let t = T(0);
        let result = t.pow(4).simplify();

        if let Operator::Rotation { angle, .. } = result {
            assert_eq!(angle, Angle64::HALF_TURN);
        } else {
            panic!("Expected Rotation, got {result:?}");
        }
    }

    #[test]
    fn test_pow_eight_t_simplify() {
        // T^8 = I (RZ(π/4)^8 = RZ(2π) = I) after simplification
        let t = T(0);
        let result = t.pow(8).simplify();
        assert!(result.is_identity());
    }

    // ========================================================================
    // commutes tests
    // ========================================================================

    #[test]
    fn test_commutes_same_pauli() {
        // X commutes with X
        let x1 = X(0);
        let x2 = X(0);
        assert_eq!(x1.commutes(&x2), Commutativity::Commutes);
    }

    #[test]
    fn test_commutes_different_paulis_same_qubit() {
        // X and Z anticommute on same qubit
        let x = X(0);
        let z = Z(0);
        assert_eq!(x.commutes(&z), Commutativity::AntiCommutes);

        // X and Y anticommute
        let y = Y(0);
        assert_eq!(x.commutes(&y), Commutativity::AntiCommutes);

        // Y and Z anticommute
        assert_eq!(y.commutes(&z), Commutativity::AntiCommutes);
    }

    #[test]
    fn test_commutes_different_qubits() {
        // Operators on different qubits always commute
        let x0 = X(0);
        let z1 = Z(1);
        assert_eq!(x0.commutes(&z1), Commutativity::Commutes);
    }

    #[test]
    fn test_commutes_non_pauli_unknown() {
        // Non-Pauli operators return Unknown
        // I(0) is RZ(0, 0), not a Pauli variant
        let id = I(0);
        let x = X(0);
        assert_eq!(id.commutes(&x), Commutativity::Unknown);

        let h = H(0);
        assert_eq!(h.commutes(&x), Commutativity::Unknown);
    }

    #[test]
    fn test_commutes_pauli_strings() {
        // XY and YX: overlap on both qubits, both anticommute -> commute
        let xy = X(0) & Y(1);
        let yx = Y(0) & X(1);
        assert_eq!(xy.commutes(&yx), Commutativity::Commutes);

        // XY and ZZ: both overlap, X-Z and Y-Z both anticommute -> commute
        let zz = Z(0) & Z(1);
        assert_eq!(xy.commutes(&zz), Commutativity::Commutes);

        // XZ and Y: only qubit 0 overlaps, X-Y anticommute
        let xz = X(0) & Z(1);
        let y = Y(0);
        assert_eq!(xz.commutes(&y), Commutativity::AntiCommutes);
    }

    // ========================================================================
    // decompose tests
    // ========================================================================

    #[test]
    fn test_decompose_single_pauli() {
        let x = X(0);
        let gates = x.decompose();
        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].gate_type, GateType::X);
    }

    #[test]
    fn test_decompose_pauli_tensor() {
        let op = X(0) & Y(1) & Z(2);
        let gates = op.decompose();
        assert_eq!(gates.len(), 3);

        let gate_types: Vec<_> = gates.iter().map(|g| g.gate_type).collect();
        assert!(gate_types.contains(&GateType::X));
        assert!(gate_types.contains(&GateType::Y));
        assert!(gate_types.contains(&GateType::Z));
    }

    #[test]
    fn test_decompose_identity() {
        // I(0) = RZ(0, 0), so it decomposes to an RZ gate with angle 0
        let id = I(0);
        let gates = id.decompose();
        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].gate_type, GateType::RZ);
        assert_eq!(gates[0].angles[0], Angle64::ZERO);
    }

    #[test]
    fn test_decompose_pauli_identity_empty() {
        // A true Pauli identity (from PauliString) decomposes to empty
        let ps = PauliString::identity();
        let op = Operator::Pauli(ps);
        let gates = op.decompose();
        assert!(gates.is_empty());
    }

    #[test]
    fn test_decompose_rotation() {
        let t = T(0);
        let gates = t.decompose();
        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].gate_type, GateType::RZ);
        assert_eq!(gates[0].angles.len(), 1);
    }

    #[test]
    fn test_decompose_gate() {
        let cx = CX(0, 1);
        let gates = cx.decompose();
        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].gate_type, GateType::CX);
    }

    #[test]
    fn test_decompose_composition() {
        let circuit = SZ(0) * H(0) * X(0); // X, then H, then S
        let gates = circuit.decompose();
        assert_eq!(gates.len(), 3);
    }

    #[test]
    fn test_decompose_adjoint() {
        let t = T(0);
        let t_dg = t.dg();
        let gates = t_dg.decompose();

        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].gate_type, GateType::RZ);
        // Angle should be negated
        let expected_angle = Angle64::ZERO - (Angle64::HALF_TURN / 4);
        assert_eq!(gates[0].angles[0], expected_angle);
    }

    #[test]
    fn test_decompose_adjoint_named_gate() {
        // S† should decompose to SZdg
        let s = SZ(0);
        let s_dg = s.dg();
        let gates = s_dg.decompose();

        // S is a rotation, S† negates the angle
        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].gate_type, GateType::RZ);
    }

    // ========================================================================
    // as_pauli_string / into_pauli_string tests
    // ========================================================================

    #[test]
    fn test_as_pauli_string_pauli() {
        let x = X(0);
        let ps = x.as_pauli_string();
        assert!(ps.is_some());
        let ps = ps.unwrap();
        assert_eq!(ps.get(0), crate::Pauli::X);
    }

    #[test]
    fn test_as_pauli_string_non_pauli() {
        let h = H(0);
        assert!(h.as_pauli_string().is_none());

        let t = T(0);
        assert!(t.as_pauli_string().is_none());
    }

    #[test]
    fn test_into_pauli_string_pauli() {
        let xy = X(0) & Y(1);
        let ps = xy.into_pauli_string();
        assert!(ps.is_some());
        let ps = ps.unwrap();
        assert_eq!(ps.get(0), crate::Pauli::X);
        assert_eq!(ps.get(1), crate::Pauli::Y);
    }

    #[test]
    fn test_into_pauli_string_with_phase() {
        let op = i * X(0);
        let ps = op.into_pauli_string();
        assert!(ps.is_some());
        let ps = ps.unwrap();
        assert_eq!(ps.get(0), crate::Pauli::X);
        assert_eq!(ps.phase(), QuarterPhase::PlusI);
    }
}
