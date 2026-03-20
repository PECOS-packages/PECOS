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
//! use pecos_core::unitary_rep::*;
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
use crate::{Angle64, Pauli, PauliString, QuarterPhase, QubitId};
use smallvec::SmallVec;
use std::ops::{BitAnd, Mul, Neg};
use std::str::FromStr;

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
/// use pecos_core::unitary_rep::X;
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
        $crate::unitary_rep::PhaseValue($crate::angle!($($tokens)*))
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
/// use pecos_core::unitary_rep::X;
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
        $crate::unitary_rep::PhaseValue($crate::turn!($($tokens)*))
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
// Unitary base type
// ============================================================================

/// Base unitary gate descriptor.
///
/// Parallels `Pauli` for `PauliString` and `Clifford` for `CliffordRep`.
/// Describes *what gate*, not *where it acts*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Unitary {
    /// Single-axis rotation: exp(-i theta/2 P)
    Rotation {
        rotation_type: RotationType,
        angle: Angle64,
    },
    /// XY-plane rotation: exp(-i theta/2 (cos(phi) X + sin(phi) Y))
    R1XY { theta: Angle64, phi: Angle64 },
    /// General single-qubit unitary U(theta, phi, lambda)
    /// Matrix: [[cos(t/2), -e^{il}sin(t/2)], [e^{ip}sin(t/2), e^{i(p+l)}cos(t/2)]]
    U3 {
        theta: Angle64,
        phi: Angle64,
        lambda: Angle64,
    },
    /// General 2-qubit Pauli rotation: exp(-i/2 * (alpha*XX + beta*YY + gamma*ZZ))
    RXXRYYRZZ {
        alpha: Angle64,
        beta: Angle64,
        gamma: Angle64,
    },
    /// General 2-qubit unitary via KAK decomposition:
    /// U = (U3(before[0]) x U3(before[1])) * RXXRYYRZZ(interaction) * (U3(after[0]) x U3(after[1]))
    /// Each [Angle64; 3] is [theta, phi, lambda] for a U3 gate.
    U2q {
        before: [[Angle64; 3]; 2],
        interaction: [Angle64; 3],
        after: [[Angle64; 3]; 2],
    },
    /// Named gate (H, CX, SWAP, etc.) without angle parameter
    Named(GateType),
}

impl Unitary {
    /// Returns the number of qubits this gate acts on.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        match self {
            Self::Rotation { rotation_type, .. } => rotation_type.num_qubits(),
            Self::R1XY { .. } | Self::U3 { .. } => 1,
            Self::RXXRYYRZZ { .. } | Self::U2q { .. } => 2,
            Self::Named(gate_type) => gate_type.quantum_arity(),
        }
    }

    /// Checks if this unitary is a Clifford operation.
    #[must_use]
    pub fn is_clifford(&self) -> bool {
        match self {
            Self::Rotation { angle, .. } => is_multiple_of_quarter_turn(*angle),
            Self::R1XY { theta, phi } => {
                is_multiple_of_quarter_turn(*theta) && is_multiple_of_quarter_turn(*phi)
            }
            Self::U3 { theta, phi, lambda } => {
                is_multiple_of_quarter_turn(*theta)
                    && is_multiple_of_quarter_turn(*phi)
                    && is_multiple_of_quarter_turn(*lambda)
            }
            Self::RXXRYYRZZ { alpha, beta, gamma } => {
                is_multiple_of_quarter_turn(*alpha)
                    && is_multiple_of_quarter_turn(*beta)
                    && is_multiple_of_quarter_turn(*gamma)
            }
            Self::U2q {
                before,
                interaction,
                after,
            } => {
                before
                    .iter()
                    .chain(after.iter())
                    .all(|u3| u3.iter().all(|a| is_multiple_of_quarter_turn(*a)))
                    && interaction.iter().all(|a| is_multiple_of_quarter_turn(*a))
            }
            Self::Named(gate_type) => gate_type.is_clifford(),
        }
    }

    /// Checks if this unitary is the identity operation.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        match self {
            Self::Rotation { angle, .. } => *angle == Angle64::ZERO,
            Self::R1XY { theta, .. } => *theta == Angle64::ZERO,
            Self::U3 { theta, phi, lambda } => {
                *theta == Angle64::ZERO && (*phi + *lambda) == Angle64::ZERO
            }
            Self::RXXRYYRZZ { alpha, beta, gamma } => {
                *alpha == Angle64::ZERO && *beta == Angle64::ZERO && *gamma == Angle64::ZERO
            }
            Self::U2q {
                before,
                interaction,
                after,
            } => {
                // Identity if all U3s are identity (theta=0, phi+lambda=0) and interaction is zero
                before
                    .iter()
                    .chain(after.iter())
                    .all(|u3| u3[0] == Angle64::ZERO && (u3[1] + u3[2]) == Angle64::ZERO)
                    && interaction.iter().all(|a| *a == Angle64::ZERO)
            }
            Self::Named(gate_type) => *gate_type == GateType::I,
        }
    }

    /// Checks if this unitary is a Pauli operation (I, X, Y, or Z).
    #[must_use]
    pub fn is_pauli(&self) -> bool {
        matches!(
            self,
            Self::Named(GateType::I | GateType::X | GateType::Y | GateType::Z)
        )
    }

    /// Returns the corresponding `Pauli` if this is a Pauli gate.
    #[must_use]
    pub fn try_to_pauli(&self) -> Option<Pauli> {
        match self {
            Self::Named(GateType::I) => Some(Pauli::I),
            Self::Named(GateType::X) => Some(Pauli::X),
            Self::Named(GateType::Y) => Some(Pauli::Y),
            Self::Named(GateType::Z) => Some(Pauli::Z),
            _ => None,
        }
    }

    /// Returns the corresponding `GateType` if one exists.
    #[must_use]
    pub fn to_gate_type(&self) -> Option<GateType> {
        match self {
            Self::Rotation {
                rotation_type,
                angle,
            } => rotation_to_gate_type(*rotation_type, *angle),
            Self::R1XY { .. } => Some(GateType::R1XY),
            Self::U3 { .. } => Some(GateType::U),
            Self::RXXRYYRZZ { .. } => Some(GateType::RXXRYYRZZ),
            Self::U2q { .. } => Some(GateType::U2q),
            Self::Named(gate_type) => Some(*gate_type),
        }
    }

    /// Embeds this gate on a specific qubit, returning a `UnitaryRep`.
    ///
    /// # Panics
    /// Panics if called on a two-qubit gate. Use [`on_qubits`](Unitary::on_qubits) instead.
    #[must_use]
    pub fn on_qubit(self, qubit: usize) -> UnitaryRep {
        assert!(
            self.num_qubits() == 1,
            "on_qubit called on {}-qubit gate; use on_qubits instead",
            self.num_qubits()
        );
        UnitaryRep::Gate(self, SmallVec::from_slice(&[qubit]))
    }

    /// Embeds this gate on specific qubits, returning a `UnitaryRep`.
    ///
    /// For single-qubit gates, the second qubit is ignored and
    /// [`on_qubit`](Unitary::on_qubit) is preferred.
    #[must_use]
    pub fn on_qubits(self, q0: usize, q1: usize) -> UnitaryRep {
        if self.num_qubits() == 1 {
            self.on_qubit(q0)
        } else {
            UnitaryRep::Gate(self, SmallVec::from_slice(&[q0, q1]))
        }
    }
}

// Composition: Unitary * Unitary -> UnitaryRep
// Both gates are placed on qubit 0 (and qubit 1 for 2-qubit gates).
impl Mul for Unitary {
    type Output = UnitaryRep;

    fn mul(self, rhs: Unitary) -> UnitaryRep {
        self.on_default_qubits() * rhs.on_default_qubits()
    }
}

// Tensor product: Unitary & Unitary -> UnitaryRep
// Gates are placed on consecutive qubits starting from 0.
impl BitAnd for Unitary {
    type Output = UnitaryRep;

    fn bitand(self, rhs: Unitary) -> UnitaryRep {
        let rhs_offset = self.num_qubits();
        self.on_default_qubits() & rhs.on_default_qubits_offset(rhs_offset)
    }
}

impl Unitary {
    /// Places this gate on default qubits starting from 0.
    pub(crate) fn on_default_qubits(self) -> UnitaryRep {
        let qubits: SmallVec<[usize; 3]> = (0..self.num_qubits()).collect();
        UnitaryRep::Gate(self, qubits)
    }

    /// Places this gate on default qubits starting from `offset`.
    pub(crate) fn on_default_qubits_offset(self, offset: usize) -> UnitaryRep {
        let qubits: SmallVec<[usize; 3]> = (offset..offset + self.num_qubits()).collect();
        UnitaryRep::Gate(self, qubits)
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
/// use pecos_core::unitary_rep::*;
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
    pub fn apply<F>(self, gate_fn: F) -> UnitaryRep
    where
        F: Fn(usize) -> UnitaryRep,
    {
        match self.0.len() {
            0 => UnitaryRep::Pauli(PauliString::default()), // Identity
            1 => gate_fn(self.0[0].0),
            _ => UnitaryRep::Tensor(self.0.iter().map(|q| gate_fn(q.0)).collect()),
        }
    }
}

/// Wrapper for qubit pairs used by pluralized two-qubit gates.
///
/// ```
/// use pecos_core::unitary_rep::*;
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
    pub fn apply<F>(self, gate_fn: F) -> UnitaryRep
    where
        F: Fn(usize, usize) -> UnitaryRep,
    {
        match self.0.len() {
            0 => UnitaryRep::Pauli(PauliString::default()), // Identity
            1 => gate_fn(self.0[0].0.0, self.0[0].1.0),
            _ => UnitaryRep::Tensor(self.0.iter().map(|(q0, q1)| gate_fn(q0.0, q1.0)).collect()),
        }
    }
}

/// A gate/operator expression - lazy representation of quantum operators.
///
/// This is the unified type for all quantum operators including Pauli operators,
/// Clifford gates, and general unitaries.
#[derive(Debug, Clone, PartialEq)]
pub enum UnitaryRep {
    /// Pauli operator (single or multi-qubit)
    /// Wraps `PauliString` for exact Pauli algebra
    Pauli(PauliString),

    /// A unitary gate acting on specific qubits.
    ///
    /// The `Unitary` descriptor says *what gate*, and the `SmallVec` says *where it acts*.
    Gate(Unitary, SmallVec<[usize; 3]>),

    /// Tensor product of expressions (operators on different qubits)
    Tensor(Vec<UnitaryRep>),

    /// Sequential composition (matrix multiplication order)
    /// Compose([A, B, C]) means apply A, then B, then C
    Compose(Vec<UnitaryRep>),

    /// Adjoint (Hermitian conjugate)
    Adjoint(Box<UnitaryRep>),

    /// Global phase: e^{i*phase} * inner
    /// Phase is represented as Angle64 for exact arithmetic
    Phase {
        phase: Angle64,
        inner: Box<UnitaryRep>,
    },
}

impl From<PauliString> for UnitaryRep {
    fn from(ps: PauliString) -> Self {
        UnitaryRep::Pauli(ps)
    }
}

/// Error type for parsing an [`UnitaryRep`] from a string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseUnitaryRepError {
    pub message: String,
}

impl std::fmt::Display for ParseUnitaryRepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ParseUnitaryRepError {}

/// Parses an angle expression like `pi`, `pi/4`, `2*pi/3`, `-pi/2`.
///
/// Returns the angle as an `Angle64`.
fn parse_angle_expr(s: &str) -> Result<Angle64, ParseUnitaryRepError> {
    let s = s.trim();
    if s.is_empty() {
        return Err(ParseUnitaryRepError {
            message: "Empty angle expression".to_string(),
        });
    }

    // Handle negative sign
    let (negative, s) = if let Some(rest) = s.strip_prefix('-') {
        (true, rest.trim())
    } else {
        (false, s)
    };

    // Try to parse as a pi expression: [N*]pi[/M]
    let angle = if let Some(rest) = s.strip_prefix("pi") {
        // "pi", "pi/N"
        let rest = rest.trim();
        if rest.is_empty() {
            Angle64::HALF_TURN
        } else if let Some(denom_str) = rest.strip_prefix('/') {
            let denom: u64 = denom_str.trim().parse().map_err(|_| ParseUnitaryRepError {
                message: format!("Invalid angle denominator: '{denom_str}'"),
            })?;
            if denom == 0 {
                return Err(ParseUnitaryRepError {
                    message: "Division by zero in angle".to_string(),
                });
            }
            Angle64::HALF_TURN / denom
        } else {
            return Err(ParseUnitaryRepError {
                message: format!("Invalid angle expression: '{s}'"),
            });
        }
    } else if s.contains("pi") {
        // "N*pi" or "N*pi/M"
        let parts: Vec<&str> = s.splitn(2, '*').collect();
        if parts.len() != 2 {
            return Err(ParseUnitaryRepError {
                message: format!("Invalid angle expression: '{s}'"),
            });
        }
        let numer: u64 = parts[0].trim().parse().map_err(|_| ParseUnitaryRepError {
            message: format!("Invalid angle numerator: '{}'", parts[0]),
        })?;
        let pi_part = parts[1].trim();
        let pi_rest = pi_part
            .strip_prefix("pi")
            .ok_or_else(|| ParseUnitaryRepError {
                message: format!("Expected 'pi' in angle expression: '{s}'"),
            })?;
        let pi_rest = pi_rest.trim();
        if pi_rest.is_empty() {
            Angle64::HALF_TURN * numer
        } else if let Some(denom_str) = pi_rest.strip_prefix('/') {
            let denom: u64 = denom_str.trim().parse().map_err(|_| ParseUnitaryRepError {
                message: format!("Invalid angle denominator: '{denom_str}'"),
            })?;
            if denom == 0 {
                return Err(ParseUnitaryRepError {
                    message: "Division by zero in angle".to_string(),
                });
            }
            Angle64::HALF_TURN * numer / denom
        } else {
            return Err(ParseUnitaryRepError {
                message: format!("Invalid angle expression: '{s}'"),
            });
        }
    } else {
        return Err(ParseUnitaryRepError {
            message: format!(
                "Unsupported angle format: '{s}' (expected pi expression like 'pi/4' or '2*pi/3')"
            ),
        });
    };

    if negative {
        Ok(Angle64::ZERO - angle)
    } else {
        Ok(angle)
    }
}

/// Parses qubit indices from whitespace-separated tokens.
fn parse_qubits(tokens: &[&str]) -> Result<Vec<usize>, ParseUnitaryRepError> {
    tokens
        .iter()
        .map(|t| {
            t.parse::<usize>().map_err(|_| ParseUnitaryRepError {
                message: format!("Invalid qubit index: '{t}'"),
            })
        })
        .collect()
}

impl FromStr for UnitaryRep {
    type Err = ParseUnitaryRepError;

    /// Parses an `UnitaryRep` from a string, supporting both gate and Pauli syntax.
    ///
    /// # Gate syntax
    ///
    /// Fixed gates: `"H 0"`, `"CX 0 1"`, `"SWAP 0 1"`, `"CCX 0 1 2"`
    ///
    /// Rotation gates: `"RX(pi/4) 0"`, `"RZ(pi/2) 0"`, `"RZZ(pi) 0 1"`
    ///
    /// Named rotations: `"T 0"`, `"Tdg 0"`, `"S 0"`, `"Sdg 0"`
    ///
    /// # Pauli syntax
    ///
    /// Sparse: `"X0 Z4 Y7"`, `"-i X2 Z4"`
    ///
    /// Dense: `"XYZZ"`, `"+iXXZI"`
    ///
    /// Single Pauli with space: `"X 0"`, `"Z 3"` (treated same as `"X0"`, `"Z3"`)
    ///
    /// **Note**: Gate names take priority over Pauli parsing. `"S 0"` parses as an
    /// S gate (RZ(pi/2)), not as Pauli S on qubit 0. Similarly for `"H 0"`, `"T 0"`, etc.
    /// Use sparse Pauli syntax without spaces (e.g., `"X0"`) to avoid ambiguity.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_core::UnitaryRep;
    /// use std::str::FromStr;
    ///
    /// // Gate syntax
    /// let h: UnitaryRep = "H 0".parse().unwrap();
    /// let cx: UnitaryRep = "CX 0 1".parse().unwrap();
    /// let t: UnitaryRep = "T 0".parse().unwrap();
    /// let rz: UnitaryRep = "RZ(pi/4) 0".parse().unwrap();
    ///
    /// // Pauli syntax (sparse)
    /// let p: UnitaryRep = "X0 Z1".parse().unwrap();
    ///
    /// // Pauli syntax (dense)
    /// let p: UnitaryRep = "XZI".parse().unwrap();
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.is_empty() {
            return Ok(UnitaryRep::Pauli(PauliString::new()));
        }

        // Extract the first word (gate name candidate), stopping at whitespace or '('
        let gate_end = s
            .find(|c: char| c.is_whitespace() || c == '(')
            .unwrap_or(s.len());
        let gate_name = &s[..gate_end];

        // Check for rotation gates with angle: GATE(angle) qubit...
        if let (Some(paren_start), Some(paren_end)) = (s.find('('), s.find(')')) {
            let rot_name = s[..paren_start].trim();
            let angle_str = &s[paren_start + 1..paren_end];
            let after_paren = s[paren_end + 1..].trim();
            let qubit_tokens: Vec<&str> = after_paren.split_whitespace().collect();

            let rot_type = match rot_name.to_uppercase().as_str() {
                "RX" => Some(RotationType::RX),
                "RY" => Some(RotationType::RY),
                "RZ" => Some(RotationType::RZ),
                "RXX" => Some(RotationType::RXX),
                "RYY" => Some(RotationType::RYY),
                "RZZ" => Some(RotationType::RZZ),
                _ => None,
            };

            if let Some(rot_type) = rot_type {
                let angle = parse_angle_expr(angle_str)?;
                let qubits = parse_qubits(&qubit_tokens)?;
                let expected = rot_type.num_qubits();
                if qubits.len() != expected {
                    return Err(ParseUnitaryRepError {
                        message: format!(
                            "{rot_name} requires {expected} qubit(s), got {}",
                            qubits.len()
                        ),
                    });
                }
                return Ok(UnitaryRep::rotation(
                    rot_type,
                    angle,
                    SmallVec::from_vec(qubits),
                ));
            }
        }

        // Try to match fixed/named gates (case-insensitive)
        let upper = gate_name.to_uppercase();
        let after_gate = s[gate_end..].trim();
        let qubit_tokens: Vec<&str> = after_gate.split_whitespace().collect();

        match upper.as_str() {
            // Single-qubit fixed gates
            "H" | "F" | "FDG" | "SX" | "SXDG" | "SY" | "SYDG" | "CH" => {
                let gate_type = match upper.as_str() {
                    "H" => GateType::H,
                    "F" => GateType::F,
                    "FDG" => GateType::Fdg,
                    "SX" => GateType::SX,
                    "SXDG" => GateType::SXdg,
                    "SY" => GateType::SY,
                    "SYDG" => GateType::SYdg,
                    "CH" => GateType::CH,
                    _ => unreachable!(),
                };
                let expected = if upper == "CH" { 2 } else { 1 };
                let qubits = parse_qubits(&qubit_tokens)?;
                if qubits.len() != expected {
                    return Err(ParseUnitaryRepError {
                        message: format!(
                            "{gate_name} requires {expected} qubit(s), got {}",
                            qubits.len()
                        ),
                    });
                }
                Ok(UnitaryRep::gate(gate_type, SmallVec::from_vec(qubits)))
            }

            // Named rotations (these produce Rotation variants)
            "T" => {
                let qubits = parse_qubits(&qubit_tokens)?;
                if qubits.len() != 1 {
                    return Err(ParseUnitaryRepError {
                        message: format!("T requires 1 qubit, got {}", qubits.len()),
                    });
                }
                Ok(UnitaryRep::rotation(
                    RotationType::RZ,
                    Angle64::HALF_TURN / 4,
                    smallvec::smallvec![qubits[0]],
                ))
            }
            "TDG" => {
                let qubits = parse_qubits(&qubit_tokens)?;
                if qubits.len() != 1 {
                    return Err(ParseUnitaryRepError {
                        message: format!("Tdg requires 1 qubit, got {}", qubits.len()),
                    });
                }
                Ok(UnitaryRep::rotation(
                    RotationType::RZ,
                    Angle64::ZERO - Angle64::HALF_TURN / 4,
                    smallvec::smallvec![qubits[0]],
                ))
            }
            "S" | "SZ" => {
                let qubits = parse_qubits(&qubit_tokens)?;
                if qubits.len() != 1 {
                    return Err(ParseUnitaryRepError {
                        message: format!("{gate_name} requires 1 qubit, got {}", qubits.len()),
                    });
                }
                Ok(UnitaryRep::rotation(
                    RotationType::RZ,
                    Angle64::QUARTER_TURN,
                    smallvec::smallvec![qubits[0]],
                ))
            }
            "SDG" | "SZDG" => {
                let qubits = parse_qubits(&qubit_tokens)?;
                if qubits.len() != 1 {
                    return Err(ParseUnitaryRepError {
                        message: format!("{gate_name} requires 1 qubit, got {}", qubits.len()),
                    });
                }
                Ok(UnitaryRep::rotation(
                    RotationType::RZ,
                    Angle64::ZERO - Angle64::QUARTER_TURN,
                    smallvec::smallvec![qubits[0]],
                ))
            }

            // Two-qubit fixed gates
            "CX" | "CNOT" => {
                let qubits = parse_qubits(&qubit_tokens)?;
                if qubits.len() != 2 {
                    return Err(ParseUnitaryRepError {
                        message: format!("{gate_name} requires 2 qubits, got {}", qubits.len()),
                    });
                }
                Ok(UnitaryRep::gate(GateType::CX, SmallVec::from_vec(qubits)))
            }
            "CY" => {
                let qubits = parse_qubits(&qubit_tokens)?;
                if qubits.len() != 2 {
                    return Err(ParseUnitaryRepError {
                        message: format!("CY requires 2 qubits, got {}", qubits.len()),
                    });
                }
                Ok(UnitaryRep::gate(GateType::CY, SmallVec::from_vec(qubits)))
            }
            "CZ" => {
                let qubits = parse_qubits(&qubit_tokens)?;
                if qubits.len() != 2 {
                    return Err(ParseUnitaryRepError {
                        message: format!("CZ requires 2 qubits, got {}", qubits.len()),
                    });
                }
                Ok(UnitaryRep::gate(GateType::CZ, SmallVec::from_vec(qubits)))
            }
            "SWAP" => {
                let qubits = parse_qubits(&qubit_tokens)?;
                if qubits.len() != 2 {
                    return Err(ParseUnitaryRepError {
                        message: format!("SWAP requires 2 qubits, got {}", qubits.len()),
                    });
                }
                Ok(UnitaryRep::gate(GateType::SWAP, SmallVec::from_vec(qubits)))
            }

            // Three-qubit gates
            "CCX" | "TOFFOLI" => {
                let qubits = parse_qubits(&qubit_tokens)?;
                if qubits.len() != 3 {
                    return Err(ParseUnitaryRepError {
                        message: format!("{gate_name} requires 3 qubits, got {}", qubits.len()),
                    });
                }
                Ok(UnitaryRep::gate(GateType::CCX, SmallVec::from_vec(qubits)))
            }

            // Not a recognized gate name -> try Pauli parsing.
            // This handles: "X0 Z1", "XYZZ", "-i X2 Z4", "X 0", "Z 3", etc.
            _ => PauliString::from_str(s)
                .map(UnitaryRep::Pauli)
                .map_err(|e| ParseUnitaryRepError { message: e.message }),
        }
    }
}

impl UnitaryRep {
    /// Creates a rotation gate expression.
    #[must_use]
    pub fn rotation(
        rotation_type: RotationType,
        angle: Angle64,
        qubits: impl Into<SmallVec<[usize; 3]>>,
    ) -> Self {
        Self::Gate(
            Unitary::Rotation {
                rotation_type,
                angle,
            },
            qubits.into(),
        )
    }

    /// Creates a fixed gate expression.
    #[must_use]
    pub fn gate(gate_type: GateType, qubits: impl Into<SmallVec<[usize; 3]>>) -> Self {
        Self::Gate(Unitary::Named(gate_type), qubits.into())
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
            // Gate adjoint: negate angle for rotations, wrap for non-self-adjoint named gates
            Self::Gate(
                Unitary::Rotation {
                    rotation_type,
                    angle,
                },
                qubits,
            ) => Self::Gate(
                Unitary::Rotation {
                    rotation_type: *rotation_type,
                    angle: negate_angle(*angle),
                },
                qubits.clone(),
            ),
            Self::Gate(Unitary::R1XY { theta, phi }, qubits) => Self::Gate(
                Unitary::R1XY {
                    theta: negate_angle(*theta),
                    phi: *phi,
                },
                qubits.clone(),
            ),
            // U(theta, phi, lambda)† = U(-theta, -lambda, -phi)
            Self::Gate(Unitary::U3 { theta, phi, lambda }, qubits) => Self::Gate(
                Unitary::U3 {
                    theta: negate_angle(*theta),
                    phi: negate_angle(*lambda),
                    lambda: negate_angle(*phi),
                },
                qubits.clone(),
            ),
            Self::Gate(Unitary::RXXRYYRZZ { alpha, beta, gamma }, qubits) => Self::Gate(
                Unitary::RXXRYYRZZ {
                    alpha: negate_angle(*alpha),
                    beta: negate_angle(*beta),
                    gamma: negate_angle(*gamma),
                },
                qubits.clone(),
            ),
            // U2q† = (B0†⊗B1†) * RXXRYYRZZ(-a,-b,-c) * (A0†⊗A1†)
            // where Ui† = U(-theta, -lambda, -phi), so: swap before/after and negate+swap phi/lambda
            Self::Gate(
                Unitary::U2q {
                    before,
                    interaction,
                    after,
                },
                qubits,
            ) => {
                let negate_u3 = |u3: &[Angle64; 3]| -> [Angle64; 3] {
                    [
                        negate_angle(u3[0]),
                        negate_angle(u3[2]),
                        negate_angle(u3[1]),
                    ]
                };
                Self::Gate(
                    Unitary::U2q {
                        before: [negate_u3(&after[0]), negate_u3(&after[1])],
                        interaction: [
                            negate_angle(interaction[0]),
                            negate_angle(interaction[1]),
                            negate_angle(interaction[2]),
                        ],
                        after: [negate_u3(&before[0]), negate_u3(&before[1])],
                    },
                    qubits.clone(),
                )
            }
            Self::Gate(Unitary::Named(gate_type), _) => {
                if gate_type.is_self_adjoint() {
                    self.clone()
                } else {
                    Self::Adjoint(Box::new(self.clone()))
                }
            }
            // Tensor adjoint: adjoint of each part
            Self::Tensor(parts) => Self::Tensor(parts.iter().map(UnitaryRep::dg).collect()),
            // Compose adjoint: reverse order and adjoint each
            Self::Compose(parts) => Self::Compose(parts.iter().rev().map(UnitaryRep::dg).collect()),
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
            Self::Gate(unitary, _) => unitary.is_clifford(),
            Self::Tensor(parts) | Self::Compose(parts) => parts.iter().all(UnitaryRep::is_clifford),
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
            Self::Gate(_, qubits) => result.extend(qubits.iter().copied()),
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

impl Neg for UnitaryRep {
    type Output = UnitaryRep;

    fn neg(self) -> UnitaryRep {
        self.with_phase(Angle64::HALF_TURN)
    }
}

impl Neg for &UnitaryRep {
    type Output = UnitaryRep;

    fn neg(self) -> UnitaryRep {
        self.clone().with_phase(Angle64::HALF_TURN)
    }
}

// ============================================================================
// Imaginary unit for phase multiplication
// ============================================================================

// Re-use the canonical types from pauli::algebra.
pub use crate::pauli::algebra::{ImaginaryUnit, NegImaginaryUnit, i};

impl Mul<UnitaryRep> for ImaginaryUnit {
    type Output = UnitaryRep;

    fn mul(self, rhs: UnitaryRep) -> UnitaryRep {
        rhs.with_phase(Angle64::QUARTER_TURN) // i = e^{iπ/2}
    }
}

impl Mul<&UnitaryRep> for ImaginaryUnit {
    type Output = UnitaryRep;

    fn mul(self, rhs: &UnitaryRep) -> UnitaryRep {
        rhs.clone().with_phase(Angle64::QUARTER_TURN)
    }
}

impl Mul<UnitaryRep> for NegImaginaryUnit {
    type Output = UnitaryRep;

    #[allow(clippy::suspicious_arithmetic_impl)] // Adding angles for phase computation
    fn mul(self, rhs: UnitaryRep) -> UnitaryRep {
        rhs.with_phase(Angle64::QUARTER_TURN + Angle64::HALF_TURN) // -i = e^{i3π/2}
    }
}

impl Mul<&UnitaryRep> for NegImaginaryUnit {
    type Output = UnitaryRep;

    #[allow(clippy::suspicious_arithmetic_impl)] // Adding angles for phase computation
    fn mul(self, rhs: &UnitaryRep) -> UnitaryRep {
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
/// use pecos_core::unitary_rep::{phase, X};
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
/// use pecos_core::unitary_rep::{phase, X, Z};
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

impl Mul<UnitaryRep> for PhaseValue {
    type Output = UnitaryRep;

    fn mul(self, rhs: UnitaryRep) -> UnitaryRep {
        rhs.with_phase(self.0)
    }
}

impl Mul<&UnitaryRep> for PhaseValue {
    type Output = UnitaryRep;

    fn mul(self, rhs: &UnitaryRep) -> UnitaryRep {
        rhs.clone().with_phase(self.0)
    }
}

impl UnitaryRep {
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
            Self::Gate(unitary, _) => unitary.to_gate_type(),
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

    /// Consumes this `UnitaryRep` and returns the inner `PauliString` if this is a `Pauli` variant.
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
    /// use pecos_core::unitary_rep::{Xs, Zs};
    /// use pecos_core::PauliOperator;
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

            Self::Gate(Unitary::Named(gate_type), qubits) => {
                let qubit = qubits.first().copied()?;
                match gate_type {
                    GateType::X => Some(PauliString::x(qubit)),
                    GateType::Y => Some(PauliString::y(qubit)),
                    GateType::Z => Some(PauliString::z(qubit)),
                    GateType::I => Some(PauliString::identity()),
                    _ => None,
                }
            }

            Self::Gate(
                Unitary::Rotation {
                    rotation_type,
                    angle,
                },
                qubits,
            ) => {
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

            // R1XY, U3, RXXRYYRZZ, and U2q are not Paulis
            Self::Gate(
                Unitary::R1XY { .. }
                | Unitary::U3 { .. }
                | Unitary::RXXRYYRZZ { .. }
                | Unitary::U2q { .. },
                _,
            ) => None,

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
            Self::Gate(
                Unitary::Rotation {
                    rotation_type,
                    angle,
                },
                _,
            ) => {
                let half = Angle64::HALF_TURN;
                let neg_half = negate_angle(half);
                (*angle == half || *angle == neg_half)
                    && matches!(
                        rotation_type,
                        RotationType::RX | RotationType::RY | RotationType::RZ
                    )
            }
            Self::Gate(Unitary::Named(gate_type), _) => {
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
            Self::Gate(
                Unitary::Rotation {
                    rotation_type,
                    angle,
                },
                qubits,
            ) => {
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
            Self::Gate(Unitary::Named(gate_type), qubits) => {
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
            Self::Pauli(_) | Self::Gate(_, _) => self.clone(),

            Self::Tensor(parts) => {
                // Simplify each part but preserve identities (they define the Hilbert space dimension)
                let simplified: Vec<_> = parts.iter().map(UnitaryRep::simplify).collect();

                match simplified.len() {
                    0 => Self::Pauli(PauliString::default()), // Empty tensor = identity
                    1 => simplified.into_iter().next().expect("length is 1"),
                    _ => Self::Tensor(simplified),
                }
            }

            Self::Compose(parts) => {
                // First simplify each part
                let simplified: Vec<_> = parts.iter().map(UnitaryRep::simplify).collect();

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
    /// use pecos_core::unitary_rep::{X, Z, H, T};
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
    pub fn conj(&self, gate: &UnitaryRep) -> Self {
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
    /// use pecos_core::unitary_rep::{X, H};
    ///
    /// // Heisenberg evolution: how X evolves under H
    /// let evolved = X(0).conjdg(&H(0));  // H† X H
    /// ```
    #[must_use]
    pub fn conjdg(&self, gate: &UnitaryRep) -> Self {
        gate.dg() * self.clone() * gate.clone()
    }

    /// Returns the global phase of this operator.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_core::unitary_rep::{X, Y};
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
    /// use pecos_core::unitary_rep::{X, Z, CX};
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
    /// use pecos_core::unitary_rep::I;
    ///
    /// assert!(I(0).is_identity());
    /// ```
    #[must_use]
    pub fn is_identity(&self) -> bool {
        match self {
            Self::Pauli(ps) => ps.weight() == 0 && ps.phase() == crate::QuarterPhase::PlusOne,
            Self::Gate(unitary, _) => unitary.is_identity(),
            Self::Tensor(parts) | Self::Compose(parts) => parts.iter().all(UnitaryRep::is_identity),
            Self::Adjoint(inner) => inner.is_identity(),
            Self::Phase { phase, inner } => *phase == Angle64::ZERO && inner.is_identity(),
        }
    }

    /// Checks if this operator is Hermitian (self-adjoint): A = A†.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_core::unitary_rep::{X, Y, Z, H, T};
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
            Self::Gate(Unitary::Named(gate_type), _) => matches!(
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
            Self::Gate(Unitary::Rotation { angle, .. }, _) => {
                // Rotations are Hermitian only at angle 0 or π
                *angle == Angle64::ZERO || *angle == Angle64::HALF_TURN
            }
            Self::Gate(Unitary::R1XY { theta, .. }, _) => {
                *theta == Angle64::ZERO || *theta == Angle64::HALF_TURN
            }
            // U(theta, phi, lambda) is Hermitian when U = U†, i.e. U(-theta, -lambda, -phi) = U(theta, phi, lambda)
            // This requires theta=0 (or pi) and phi=-lambda
            Self::Gate(Unitary::U3 { theta, phi, lambda }, _) => {
                (*theta == Angle64::ZERO || *theta == Angle64::HALF_TURN)
                    && *phi == negate_angle(*lambda)
            }
            Self::Gate(Unitary::RXXRYYRZZ { alpha, beta, gamma }, _) => {
                (*alpha == Angle64::ZERO || *alpha == Angle64::HALF_TURN)
                    && (*beta == Angle64::ZERO || *beta == Angle64::HALF_TURN)
                    && (*gamma == Angle64::ZERO || *gamma == Angle64::HALF_TURN)
            }
            Self::Tensor(parts) => parts.iter().all(UnitaryRep::is_hermitian),
            // U2q Hermiticity is structurally complex; conservatively return false.
            // Composition of Hermitians isn't generally Hermitian; phase factors break Hermiticity.
            Self::Gate(Unitary::U2q { .. }, _) | Self::Compose(_) | Self::Phase { .. } => false,
            Self::Adjoint(inner) => inner.is_hermitian(), // (A†)† = A, so same as inner
        }
    }

    /// Returns the operator raised to a power (repeated composition).
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_core::unitary_rep::{X, H};
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
            0 => Self::Gate(
                Unitary::Named(GateType::I),
                self.qubits()
                    .into_iter()
                    .next()
                    .map_or(smallvec::smallvec![0], |q| smallvec::smallvec![q]),
            ),
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
    /// use pecos_core::unitary_rep::{X, Z, Commutativity};
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
    pub fn commutes(&self, other: &UnitaryRep) -> Commutativity {
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
    /// use pecos_core::unitary_rep::{X, H, RZ};
    /// use pecos_core::Angle64;
    ///
    /// assert!(X(0).is_unitary());
    /// assert!(H(0).is_unitary());
    /// assert!(RZ(Angle64::QUARTER_TURN, 0).is_unitary());
    /// ```
    #[must_use]
    pub fn is_unitary(&self) -> bool {
        true // All UnitaryRep variants are unitary by construction
    }

    /// Decomposes this operator into a sequence of primitive gates.
    ///
    /// Returns a flat vector of `Gate` structs representing the operator
    /// as a sequence of native gates.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_core::unitary_rep::{H, CX};
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

            Self::Gate(
                Unitary::Rotation {
                    rotation_type,
                    angle,
                },
                qubits,
            ) => {
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

            Self::Gate(Unitary::R1XY { theta, phi }, qubits) => {
                let qubit_ids: crate::GateQubits =
                    qubits.iter().map(|&q| crate::QubitId(q)).collect();
                vec![Gate::with_angles(
                    GateType::R1XY,
                    smallvec::smallvec![*theta, *phi],
                    qubit_ids,
                )]
            }

            Self::Gate(Unitary::U3 { theta, phi, lambda }, qubits) => {
                let qubit_ids: crate::GateQubits =
                    qubits.iter().map(|&q| crate::QubitId(q)).collect();
                vec![Gate::with_angles(
                    GateType::U,
                    smallvec::smallvec![*theta, *phi, *lambda],
                    qubit_ids,
                )]
            }

            Self::Gate(Unitary::RXXRYYRZZ { alpha, beta, gamma }, qubits) => {
                let qubit_ids: crate::GateQubits =
                    qubits.iter().map(|&q| crate::QubitId(q)).collect();
                vec![Gate::with_angles(
                    GateType::RXXRYYRZZ,
                    smallvec::smallvec![*alpha, *beta, *gamma],
                    qubit_ids,
                )]
            }

            Self::Gate(
                Unitary::U2q {
                    before,
                    interaction,
                    after,
                },
                qubits,
            ) => {
                let qubit_ids: crate::GateQubits =
                    qubits.iter().map(|&q| crate::QubitId(q)).collect();
                vec![Gate::with_angles(
                    GateType::U2q,
                    smallvec::smallvec![
                        before[0][0],
                        before[0][1],
                        before[0][2],
                        before[1][0],
                        before[1][1],
                        before[1][2],
                        interaction[0],
                        interaction[1],
                        interaction[2],
                        after[0][0],
                        after[0][1],
                        after[0][2],
                        after[1][0],
                        after[1][1],
                        after[1][2]
                    ],
                    qubit_ids,
                )]
            }

            Self::Gate(Unitary::Named(gate_type), qubits) => {
                let qubit_ids: crate::GateQubits =
                    qubits.iter().map(|&q| crate::QubitId(q)).collect();
                vec![Gate::simple(*gate_type, qubit_ids)]
            }

            Self::Tensor(parts) => {
                // Decompose each part and concatenate
                parts.iter().flat_map(UnitaryRep::decompose).collect()
            }

            Self::Compose(parts) => {
                // Decompose each part in application order
                parts.iter().flat_map(UnitaryRep::decompose).collect()
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

            Self::Gate(
                Unitary::Rotation {
                    rotation_type,
                    angle,
                },
                qubits,
            ) => rotation_to_clifford_rep(*rotation_type, *angle, qubits, num_qubits),

            // R1XY Clifford case: decompose as RZ(-phi+pi/2) * RY(theta) * RZ(phi-pi/2)
            // and convert each to CliffordRep. Only reached when is_clifford() is true.
            Self::Gate(Unitary::R1XY { theta, phi }, qubits) => {
                let q = qubits[0];
                let rz1 = Unitary::Rotation {
                    rotation_type: RotationType::RZ,
                    angle: negate_angle(*phi) + Angle64::QUARTER_TURN,
                };
                let ry = Unitary::Rotation {
                    rotation_type: RotationType::RY,
                    angle: *theta,
                };
                let rz2 = Unitary::Rotation {
                    rotation_type: RotationType::RZ,
                    angle: *phi - Angle64::QUARTER_TURN,
                };
                let rep = UnitaryRep::Gate(rz1, SmallVec::from_slice(&[q]))
                    * UnitaryRep::Gate(ry, SmallVec::from_slice(&[q]))
                    * UnitaryRep::Gate(rz2, SmallVec::from_slice(&[q]));
                rep.to_clifford_rep(num_qubits)
            }

            // RXXRYYRZZ Clifford case: decompose as RXX(alpha) * RYY(beta) * RZZ(gamma)
            Self::Gate(Unitary::RXXRYYRZZ { alpha, beta, gamma }, qubits) => {
                let qs: SmallVec<[usize; 3]> = SmallVec::from_slice(&[qubits[0], qubits[1]]);
                let rxx = Unitary::Rotation {
                    rotation_type: RotationType::RXX,
                    angle: *alpha,
                };
                let ryy = Unitary::Rotation {
                    rotation_type: RotationType::RYY,
                    angle: *beta,
                };
                let rzz = Unitary::Rotation {
                    rotation_type: RotationType::RZZ,
                    angle: *gamma,
                };
                let rep = UnitaryRep::Gate(rxx, qs.clone())
                    * UnitaryRep::Gate(ryy, qs.clone())
                    * UnitaryRep::Gate(rzz, qs);
                rep.to_clifford_rep(num_qubits)
            }

            // U2q Clifford case: decompose as (U3⊗U3) * RXXRYYRZZ * (U3⊗U3)
            Self::Gate(
                Unitary::U2q {
                    before,
                    interaction,
                    after,
                },
                qubits,
            ) => {
                let qs: SmallVec<[usize; 3]> = SmallVec::from_slice(&[qubits[0], qubits[1]]);
                // Build the decomposition: after[0]⊗after[1], then RXXRYYRZZ, then before[0]⊗before[1]
                let u3_gate = |params: &[Angle64; 3], q: usize| -> UnitaryRep {
                    UnitaryRep::Gate(
                        Unitary::U3 {
                            theta: params[0],
                            phi: params[1],
                            lambda: params[2],
                        },
                        SmallVec::from_slice(&[q]),
                    )
                };
                let rxxryyrzz = UnitaryRep::Gate(
                    Unitary::RXXRYYRZZ {
                        alpha: interaction[0],
                        beta: interaction[1],
                        gamma: interaction[2],
                    },
                    qs,
                );
                let rep = u3_gate(&before[0], qubits[0])
                    * u3_gate(&before[1], qubits[1])
                    * rxxryyrzz
                    * u3_gate(&after[0], qubits[0])
                    * u3_gate(&after[1], qubits[1]);
                rep.to_clifford_rep(num_qubits)
            }

            // U3 Clifford case: decompose as RZ(phi) * RY(theta) * RZ(lambda)
            // (only reached when is_clifford() is true)
            Self::Gate(Unitary::U3 { theta, phi, lambda }, qubits) => {
                let q = qubits[0];
                let rz_phi = Unitary::Rotation {
                    rotation_type: RotationType::RZ,
                    angle: *phi,
                };
                let ry = Unitary::Rotation {
                    rotation_type: RotationType::RY,
                    angle: *theta,
                };
                let rz_lambda = Unitary::Rotation {
                    rotation_type: RotationType::RZ,
                    angle: *lambda,
                };
                let rep = UnitaryRep::Gate(rz_phi, SmallVec::from_slice(&[q]))
                    * UnitaryRep::Gate(ry, SmallVec::from_slice(&[q]))
                    * UnitaryRep::Gate(rz_lambda, SmallVec::from_slice(&[q]));
                rep.to_clifford_rep(num_qubits)
            }

            Self::Gate(Unitary::Named(gate_type), qubits) => {
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
    qubits: &SmallVec<[usize; 3]>,
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

        RotationType::RXX => {
            let q0 = qubits[0];
            let q1 = qubits[1];

            if angle == quarter {
                // SXX = RXX(π/2)
                let cliff = CliffordRep::sxx(q0, q1);
                Some(extend_clifford(cliff, num_qubits))
            } else if angle == neg_quarter || angle == three_quarter {
                // SXXdg = RXX(-π/2) = RXX(3π/2)
                let cliff = CliffordRep::sxxdg(q0, q1);
                Some(extend_clifford(cliff, num_qubits))
            } else if angle == half || angle == neg_half {
                // XX = RXX(π) = (X kron X)
                let mut result = CliffordRep::identity(num_qubits);
                result = apply_x(&result, q0);
                result = apply_x(&result, q1);
                Some(result)
            } else {
                None
            }
        }

        RotationType::RYY => {
            let q0 = qubits[0];
            let q1 = qubits[1];

            if angle == quarter {
                // SYY = RYY(π/2)
                let cliff = CliffordRep::syy(q0, q1);
                Some(extend_clifford(cliff, num_qubits))
            } else if angle == neg_quarter || angle == three_quarter {
                // SYYdg = RYY(-π/2) = RYY(3π/2)
                let cliff = CliffordRep::syydg(q0, q1);
                Some(extend_clifford(cliff, num_qubits))
            } else if angle == half || angle == neg_half {
                // YY = RYY(π) = (Y kron Y)
                let mut result = CliffordRep::identity(num_qubits);
                result = apply_y(&result, q0);
                result = apply_y(&result, q1);
                Some(result)
            } else {
                None
            }
        }

        RotationType::RZZ => {
            let q0 = qubits[0];
            let q1 = qubits[1];

            if angle == quarter {
                // SZZ = RZZ(π/2)
                let cliff = CliffordRep::szz(q0, q1);
                Some(extend_clifford(cliff, num_qubits))
            } else if angle == neg_quarter || angle == three_quarter {
                // SZZdg = RZZ(-π/2) = RZZ(3π/2)
                let cliff = CliffordRep::szzdg(q0, q1);
                Some(extend_clifford(cliff, num_qubits))
            } else if angle == half || angle == neg_half {
                // ZZ = RZZ(π) = (Z kron Z)
                let mut result = CliffordRep::identity(num_qubits);
                result = apply_z(&result, q0);
                result = apply_z(&result, q1);
                Some(result)
            } else {
                None
            }
        }
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
        GateType::F => {
            let cliff = CliffordRep::f(qubits[0]);
            Some(extend_clifford(cliff, num_qubits))
        }
        GateType::Fdg => {
            let cliff = CliffordRep::fdg(qubits[0]);
            Some(extend_clifford(cliff, num_qubits))
        }
        GateType::SXX => {
            let cliff = CliffordRep::sxx(qubits[0], qubits[1]);
            Some(extend_clifford(cliff, num_qubits))
        }
        GateType::SXXdg => {
            let cliff = CliffordRep::sxxdg(qubits[0], qubits[1]);
            Some(extend_clifford(cliff, num_qubits))
        }
        GateType::SYY => {
            let cliff = CliffordRep::syy(qubits[0], qubits[1]);
            Some(extend_clifford(cliff, num_qubits))
        }
        GateType::SYYdg => {
            let cliff = CliffordRep::syydg(qubits[0], qubits[1]);
            Some(extend_clifford(cliff, num_qubits))
        }
        GateType::SZZ => {
            let cliff = CliffordRep::szz(qubits[0], qubits[1]);
            Some(extend_clifford(cliff, num_qubits))
        }
        GateType::SZZdg => {
            let cliff = CliffordRep::szzdg(qubits[0], qubits[1]);
            Some(extend_clifford(cliff, num_qubits))
        }
        _ => None, // Non-Clifford or parameterized gate
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
    let s_cliff = crate::clifford_rep::CliffordRep::sz(qubit);
    extend_clifford(s_cliff, cliff.num_qubits()).compose(cliff)
}

fn apply_sdg(
    cliff: &crate::clifford_rep::CliffordRep,
    qubit: usize,
) -> crate::clifford_rep::CliffordRep {
    let sdg_cliff = crate::clifford_rep::CliffordRep::szdg(qubit);
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
fn flatten_compose(parts: Vec<UnitaryRep>) -> Vec<UnitaryRep> {
    let mut result = Vec::new();
    for part in parts {
        match part {
            UnitaryRep::Compose(inner_parts) => {
                result.extend(flatten_compose(inner_parts));
            }
            other => result.push(other),
        }
    }
    result
}

/// Merge adjacent rotations of the same type on the same qubits.
fn merge_adjacent_rotations(parts: Vec<UnitaryRep>) -> Vec<UnitaryRep> {
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
fn try_merge_rotations(a: &UnitaryRep, b: &UnitaryRep) -> Option<UnitaryRep> {
    match (a, b) {
        (
            UnitaryRep::Gate(
                Unitary::Rotation {
                    rotation_type: rt_a,
                    angle: angle_a,
                },
                qubits_a,
            ),
            UnitaryRep::Gate(
                Unitary::Rotation {
                    rotation_type: rt_b,
                    angle: angle_b,
                },
                qubits_b,
            ),
        ) => {
            // Can only merge if same rotation type and same qubits
            if rt_a == rt_b && qubits_a == qubits_b {
                let combined_angle = *angle_a + *angle_b;
                Some(UnitaryRep::Gate(
                    Unitary::Rotation {
                        rotation_type: *rt_a,
                        angle: combined_angle,
                    },
                    qubits_a.clone(),
                ))
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
        RotationType::RXX => {
            if angle == quarter {
                Some(GateType::SXX)
            } else if angle == neg_quarter {
                Some(GateType::SXXdg)
            } else {
                None
            }
        }
        RotationType::RYY => {
            if angle == quarter {
                Some(GateType::SYY)
            } else if angle == neg_quarter {
                Some(GateType::SYYdg)
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
        use GateType::{
            CX, CY, CZ, F, Fdg, H, I, SWAP, SX, SXX, SXXdg, SXdg, SY, SYY, SYYdg, SYdg, SZ, SZZ,
            SZZdg, SZdg, X, Y, Z,
        };
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
                | F
                | Fdg
                | CX
                | CY
                | CZ
                | SWAP
                | SXX
                | SXXdg
                | SYY
                | SYYdg
                | SZZ
                | SZZdg
        )
    }

    fn is_self_adjoint(&self) -> bool {
        use GateType::{CCX, CX, CY, CZ, H, I, SWAP, X, Y, Z};
        matches!(self, I | X | Y | Z | H | CX | CY | CZ | SWAP | CCX)
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
pub fn RX(angle: Angle64, qubit: impl Into<QubitId>) -> UnitaryRep {
    UnitaryRep::rotation(RotationType::RX, angle, smallvec::smallvec![qubit.into().0])
}

/// RX rotations on multiple qubits.
///
/// `RXs(angle, [0, 1, 2])` is equivalent to `RX(angle, 0) & RX(angle, 1) & RX(angle, 2)`
#[must_use]
#[allow(non_snake_case)]
pub fn RXs(angle: Angle64, qubits: impl Into<Qubits>) -> UnitaryRep {
    qubits
        .into()
        .apply(|q| UnitaryRep::rotation(RotationType::RX, angle, smallvec::smallvec![q]))
}

/// Rotation around Y axis by the given angle.
///
/// For multiple qubits, use `RYs(angle, [0, 1, 2])`.
#[must_use]
#[allow(non_snake_case)]
pub fn RY(angle: Angle64, qubit: impl Into<QubitId>) -> UnitaryRep {
    UnitaryRep::rotation(RotationType::RY, angle, smallvec::smallvec![qubit.into().0])
}

/// RY rotations on multiple qubits.
///
/// `RYs(angle, [0, 1, 2])` is equivalent to `RY(angle, 0) & RY(angle, 1) & RY(angle, 2)`
#[must_use]
#[allow(non_snake_case)]
pub fn RYs(angle: Angle64, qubits: impl Into<Qubits>) -> UnitaryRep {
    qubits
        .into()
        .apply(|q| UnitaryRep::rotation(RotationType::RY, angle, smallvec::smallvec![q]))
}

/// Rotation around Z axis by the given angle.
///
/// For multiple qubits, use `RZs(angle, [0, 1, 2])`.
#[must_use]
#[allow(non_snake_case)]
pub fn RZ(angle: Angle64, qubit: impl Into<QubitId>) -> UnitaryRep {
    UnitaryRep::rotation(RotationType::RZ, angle, smallvec::smallvec![qubit.into().0])
}

/// RZ rotations on multiple qubits.
///
/// `RZs(angle, [0, 1, 2])` is equivalent to `RZ(angle, 0) & RZ(angle, 1) & RZ(angle, 2)`
#[must_use]
#[allow(non_snake_case)]
pub fn RZs(angle: Angle64, qubits: impl Into<Qubits>) -> UnitaryRep {
    qubits
        .into()
        .apply(|q| UnitaryRep::rotation(RotationType::RZ, angle, smallvec::smallvec![q]))
}

// ============================================================================
// Gate constructors - Two qubit rotations
// ============================================================================

/// Two-qubit XX rotation by the given angle.
///
/// For multiple pairs, use `RXXs(angle, [(0, 1), (2, 3)])` or tensor.
#[must_use]
#[allow(non_snake_case)]
pub fn RXX(angle: Angle64, q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> UnitaryRep {
    UnitaryRep::rotation(
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
pub fn RXXs(angle: Angle64, pairs: impl Into<QubitPairs>) -> UnitaryRep {
    pairs
        .into()
        .apply(|q0, q1| UnitaryRep::rotation(RotationType::RXX, angle, smallvec::smallvec![q0, q1]))
}

/// Two-qubit YY rotation by the given angle.
///
/// For multiple pairs, use `RYYs(angle, [(0, 1), (2, 3)])` or tensor.
#[must_use]
#[allow(non_snake_case)]
pub fn RYY(angle: Angle64, q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> UnitaryRep {
    UnitaryRep::rotation(
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
pub fn RYYs(angle: Angle64, pairs: impl Into<QubitPairs>) -> UnitaryRep {
    pairs
        .into()
        .apply(|q0, q1| UnitaryRep::rotation(RotationType::RYY, angle, smallvec::smallvec![q0, q1]))
}

/// Two-qubit ZZ rotation by the given angle.
///
/// For multiple pairs, use `RZZs(angle, [(0, 1), (2, 3)])` or tensor.
#[must_use]
#[allow(non_snake_case)]
pub fn RZZ(angle: Angle64, q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> UnitaryRep {
    UnitaryRep::rotation(
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
pub fn RZZs(angle: Angle64, pairs: impl Into<QubitPairs>) -> UnitaryRep {
    pairs
        .into()
        .apply(|q0, q1| UnitaryRep::rotation(RotationType::RZZ, angle, smallvec::smallvec![q0, q1]))
}

// ============================================================================
// Gate constructors - Named single-qubit Cliffords
// ============================================================================

/// Identity gate on a single qubit.
#[must_use]
#[allow(non_snake_case)]
pub fn I(qubit: impl Into<QubitId>) -> UnitaryRep {
    RZ(Angle64::ZERO, qubit.into().0)
}

/// Identity gates on multiple qubits.
#[must_use]
#[allow(non_snake_case)]
pub fn Is(qubits: impl Into<Qubits>) -> UnitaryRep {
    qubits.into().apply(|q| RZ(Angle64::ZERO, q))
}

/// Pauli X operator on a single qubit.
///
/// For multiple qubits, use `Xs([0, 2, 5])` or tensor: `X(0) & X(2) & X(5)`
#[must_use]
#[allow(non_snake_case)]
pub fn X(qubit: impl Into<QubitId>) -> UnitaryRep {
    UnitaryRep::Pauli(PauliString::x(qubit.into().0))
}

/// Pauli X operators on multiple qubits.
///
/// `Xs([0, 2, 5])` is equivalent to `X(0) & X(2) & X(5)`
#[must_use]
#[allow(non_snake_case)]
pub fn Xs(qubits: impl Into<Qubits>) -> UnitaryRep {
    let qs = qubits.into();
    if qs.0.is_empty() {
        UnitaryRep::Pauli(PauliString::default())
    } else {
        let mut ps = PauliString::x(qs.0[0].0);
        for q in &qs.0[1..] {
            ps = ps & PauliString::x(q.0);
        }
        UnitaryRep::Pauli(ps)
    }
}

/// Pauli Y operator on a single qubit.
///
/// For multiple qubits, use `Ys([0, 2, 5])` or tensor: `Y(0) & Y(2) & Y(5)`
#[must_use]
#[allow(non_snake_case)]
pub fn Y(qubit: impl Into<QubitId>) -> UnitaryRep {
    UnitaryRep::Pauli(PauliString::y(qubit.into().0))
}

/// Pauli Y operators on multiple qubits.
///
/// `Ys([0, 2, 5])` is equivalent to `Y(0) & Y(2) & Y(5)`
#[must_use]
#[allow(non_snake_case)]
pub fn Ys(qubits: impl Into<Qubits>) -> UnitaryRep {
    let qs = qubits.into();
    if qs.0.is_empty() {
        UnitaryRep::Pauli(PauliString::default())
    } else {
        let mut ps = PauliString::y(qs.0[0].0);
        for q in &qs.0[1..] {
            ps = ps & PauliString::y(q.0);
        }
        UnitaryRep::Pauli(ps)
    }
}

/// Pauli Z operator on a single qubit.
///
/// For multiple qubits, use `Zs([0, 2, 5])` or tensor: `Z(0) & Z(2) & Z(5)`
#[must_use]
#[allow(non_snake_case)]
pub fn Z(qubit: impl Into<QubitId>) -> UnitaryRep {
    UnitaryRep::Pauli(PauliString::z(qubit.into().0))
}

/// Pauli Z operators on multiple qubits.
///
/// `Zs([0, 2, 5])` is equivalent to `Z(0) & Z(2) & Z(5)`
#[must_use]
#[allow(non_snake_case)]
pub fn Zs(qubits: impl Into<Qubits>) -> UnitaryRep {
    let qs = qubits.into();
    if qs.0.is_empty() {
        UnitaryRep::Pauli(PauliString::default())
    } else {
        let mut ps = PauliString::z(qs.0[0].0);
        for q in &qs.0[1..] {
            ps = ps & PauliString::z(q.0);
        }
        UnitaryRep::Pauli(ps)
    }
}

/// SX gate (sqrt X): RX(π/2)
#[must_use]
#[allow(non_snake_case)]
pub fn SX(qubit: impl Into<QubitId>) -> UnitaryRep {
    RX(Angle64::QUARTER_TURN, qubit.into().0)
}

/// SX gates on multiple qubits.
#[must_use]
#[allow(non_snake_case)]
pub fn SXs(qubits: impl Into<Qubits>) -> UnitaryRep {
    qubits.into().apply(|q| RX(Angle64::QUARTER_TURN, q))
}

/// SY gate (sqrt Y): RY(π/2)
#[must_use]
#[allow(non_snake_case)]
pub fn SY(qubit: impl Into<QubitId>) -> UnitaryRep {
    RY(Angle64::QUARTER_TURN, qubit.into().0)
}

/// SY gates on multiple qubits.
#[must_use]
#[allow(non_snake_case)]
pub fn SYs(qubits: impl Into<Qubits>) -> UnitaryRep {
    qubits.into().apply(|q| RY(Angle64::QUARTER_TURN, q))
}

/// SZ gate (sqrt Z): RZ(π/2)
#[must_use]
#[allow(non_snake_case)]
pub fn SZ(qubit: impl Into<QubitId>) -> UnitaryRep {
    RZ(Angle64::QUARTER_TURN, qubit.into().0)
}

/// SZ gates on multiple qubits.
#[must_use]
#[allow(non_snake_case)]
pub fn SZs(qubits: impl Into<Qubits>) -> UnitaryRep {
    qubits.into().apply(|q| RZ(Angle64::QUARTER_TURN, q))
}

/// T gate: RZ(π/4)
#[must_use]
#[allow(non_snake_case)]
pub fn T(qubit: impl Into<QubitId>) -> UnitaryRep {
    RZ(Angle64::HALF_TURN / 4, qubit.into().0)
}

/// T gates on multiple qubits.
#[must_use]
#[allow(non_snake_case)]
pub fn Ts(qubits: impl Into<Qubits>) -> UnitaryRep {
    qubits.into().apply(|q| RZ(Angle64::HALF_TURN / 4, q))
}

/// Hadamard gate: RZ(π) * RY(π/2) (up to global phase)
#[must_use]
#[allow(non_snake_case)]
pub fn H(qubit: impl Into<QubitId>) -> UnitaryRep {
    let q = qubit.into().0;
    UnitaryRep::Gate(Unitary::Named(GateType::H), smallvec::smallvec![q])
}

/// Hadamard gates on multiple qubits.
#[must_use]
#[allow(non_snake_case)]
pub fn Hs(qubits: impl Into<Qubits>) -> UnitaryRep {
    qubits.into().apply(|q| {
        UnitaryRep::Compose(vec![
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
pub fn CX(control: impl Into<QubitId>, target: impl Into<QubitId>) -> UnitaryRep {
    UnitaryRep::gate(
        GateType::CX,
        smallvec::smallvec![control.into().0, target.into().0],
    )
}

/// CX gates on multiple qubit pairs.
///
/// `CXs([(0, 1), (2, 3)])` is equivalent to `CX(0, 1) & CX(2, 3)`
#[must_use]
#[allow(non_snake_case)]
pub fn CXs(pairs: impl Into<QubitPairs>) -> UnitaryRep {
    pairs
        .into()
        .apply(|ctrl, tgt| UnitaryRep::gate(GateType::CX, smallvec::smallvec![ctrl, tgt]))
}

/// Controlled-Y gate.
///
/// For multiple pairs, use `CYs([(0, 1), (2, 3)])` or tensor: `CY(0, 1) & CY(2, 3)`
#[must_use]
#[allow(non_snake_case)]
pub fn CY(control: impl Into<QubitId>, target: impl Into<QubitId>) -> UnitaryRep {
    UnitaryRep::gate(
        GateType::CY,
        smallvec::smallvec![control.into().0, target.into().0],
    )
}

/// CY gates on multiple qubit pairs.
///
/// `CYs([(0, 1), (2, 3)])` is equivalent to `CY(0, 1) & CY(2, 3)`
#[must_use]
#[allow(non_snake_case)]
pub fn CYs(pairs: impl Into<QubitPairs>) -> UnitaryRep {
    pairs
        .into()
        .apply(|ctrl, tgt| UnitaryRep::gate(GateType::CY, smallvec::smallvec![ctrl, tgt]))
}

/// Controlled-Z gate.
///
/// For multiple pairs, use `CZs([(0, 1), (2, 3)])` or tensor: `CZ(0, 1) & CZ(2, 3)`
#[must_use]
#[allow(non_snake_case)]
pub fn CZ(q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> UnitaryRep {
    UnitaryRep::gate(GateType::CZ, smallvec::smallvec![q0.into().0, q1.into().0])
}

/// CZ gates on multiple qubit pairs.
///
/// `CZs([(0, 1), (2, 3)])` is equivalent to `CZ(0, 1) & CZ(2, 3)`
#[must_use]
#[allow(non_snake_case)]
pub fn CZs(pairs: impl Into<QubitPairs>) -> UnitaryRep {
    pairs
        .into()
        .apply(|q0, q1| UnitaryRep::gate(GateType::CZ, smallvec::smallvec![q0, q1]))
}

/// SWAP gate.
///
/// For multiple pairs, use `SWAPs([(0, 1), (2, 3)])` or tensor: `SWAP(0, 1) & SWAP(2, 3)`
#[must_use]
#[allow(non_snake_case)]
pub fn SWAP(q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> UnitaryRep {
    UnitaryRep::gate(
        GateType::SWAP,
        smallvec::smallvec![q0.into().0, q1.into().0],
    )
}

/// SWAP gates on multiple qubit pairs.
///
/// `SWAPs([(0, 1), (2, 3)])` is equivalent to `SWAP(0, 1) & SWAP(2, 3)`
#[must_use]
#[allow(non_snake_case)]
pub fn SWAPs(pairs: impl Into<QubitPairs>) -> UnitaryRep {
    pairs
        .into()
        .apply(|q0, q1| UnitaryRep::gate(GateType::SWAP, smallvec::smallvec![q0, q1]))
}

/// SZZ gate: RZZ(π/2)
///
/// For multiple pairs, use `SZZs([(0, 1), (2, 3)])` or tensor: `SZZ(0, 1) & SZZ(2, 3)`
#[must_use]
#[allow(non_snake_case)]
pub fn SZZ(q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> UnitaryRep {
    UnitaryRep::rotation(
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
pub fn SZZs(pairs: impl Into<QubitPairs>) -> UnitaryRep {
    pairs.into().apply(|q0, q1| {
        UnitaryRep::rotation(
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
pub fn CCX(
    c0: impl Into<QubitId>,
    c1: impl Into<QubitId>,
    target: impl Into<QubitId>,
) -> UnitaryRep {
    UnitaryRep::gate(
        GateType::CCX,
        smallvec::smallvec![c0.into().0, c1.into().0, target.into().0],
    )
}

// ============================================================================
// UnitaryRep implementations
// ============================================================================

// Tensor product: &
impl BitAnd for UnitaryRep {
    type Output = UnitaryRep;

    fn bitand(self, rhs: UnitaryRep) -> UnitaryRep {
        match (self, rhs) {
            // Pauli & Pauli: use PauliString tensor product
            (UnitaryRep::Pauli(a), UnitaryRep::Pauli(b)) => UnitaryRep::Pauli(a & b),
            // Flatten nested tensors
            (UnitaryRep::Tensor(mut parts), UnitaryRep::Tensor(rhs_parts)) => {
                parts.extend(rhs_parts);
                UnitaryRep::Tensor(parts)
            }
            (UnitaryRep::Tensor(mut parts), rhs) => {
                parts.push(rhs);
                UnitaryRep::Tensor(parts)
            }
            (lhs, UnitaryRep::Tensor(mut parts)) => {
                parts.insert(0, lhs);
                UnitaryRep::Tensor(parts)
            }
            (lhs, rhs) => UnitaryRep::Tensor(vec![lhs, rhs]),
        }
    }
}

// Composition: *
impl Mul for UnitaryRep {
    type Output = UnitaryRep;

    fn mul(self, rhs: UnitaryRep) -> UnitaryRep {
        // A * B means apply B first, then A (matrix multiplication order)
        // So we store as [B, A] in the Compose vec (application order)
        match (self, rhs) {
            // Pauli * Pauli: use PauliString algebra
            (UnitaryRep::Pauli(a), UnitaryRep::Pauli(b)) => UnitaryRep::Pauli(a * b),
            // Flatten nested compositions
            (UnitaryRep::Compose(lhs_parts), UnitaryRep::Compose(rhs_parts)) => {
                // rhs applied first, then lhs
                let mut result = rhs_parts;
                result.extend(lhs_parts);
                UnitaryRep::Compose(result)
            }
            (UnitaryRep::Compose(lhs_parts), rhs) => {
                // rhs applied first
                let mut result = vec![rhs];
                result.extend(lhs_parts);
                UnitaryRep::Compose(result)
            }
            (lhs, UnitaryRep::Compose(mut rhs_parts)) => {
                // rhs_parts applied first, then lhs
                rhs_parts.push(lhs);
                UnitaryRep::Compose(rhs_parts)
            }
            (lhs, rhs) => {
                // rhs applied first, then lhs
                UnitaryRep::Compose(vec![rhs, lhs])
            }
        }
    }
}

// ============================================================================
// Circuit diagram generation
// ============================================================================

use crate::circuit_diagram::{
    CellColor, CircuitDiagram, DiagramRenderer, DiagramStyle, GateFamily, SymbolSet,
};

/// Map a `GateType` to its axis color using PECOS color algebra.
fn gate_type_color(gt: GateType) -> CellColor {
    match gt {
        GateType::X | GateType::RX | GateType::RXX => CellColor::XAxis,
        GateType::Y | GateType::RY | GateType::RYY => CellColor::YAxis,
        GateType::Z
        | GateType::RZ
        | GateType::T
        | GateType::Tdg
        | GateType::RZZ
        | GateType::MZ
        | GateType::PZ
        | GateType::SZZ
        | GateType::SZZdg
        | GateType::CRZ => CellColor::ZAxis,
        GateType::SX | GateType::SXdg => CellColor::YZMix,
        GateType::SY | GateType::SYdg | GateType::H | GateType::CH => CellColor::XZMix,
        GateType::SZ | GateType::SZdg => CellColor::XYMix,
        _ => CellColor::None,
    }
}

/// Map a `GateType` to its `GateFamily` for diagram bracket/stroke styling.
///
/// Most gates use `Default` brackets (`[G]`). Only measurement and preparation
/// gates keep their asymmetric brackets (`|MZ)` and `(PZ|`).
fn gate_type_family(gt: GateType) -> GateFamily {
    match gt {
        GateType::MZ | GateType::MeasureLeaked | GateType::MeasureFree => GateFamily::Measurement,
        GateType::PZ | GateType::QAlloc | GateType::QFree => GateFamily::Preparation,
        _ => GateFamily::Default,
    }
}

impl UnitaryRep {
    /// Generates a Unicode circuit diagram for this expression.
    ///
    /// This is an alias for [`to_unicode`](Self::to_unicode).
    #[must_use]
    pub fn to_diagram(&self, num_qubits: usize) -> String {
        self.to_unicode(num_qubits)
    }

    /// Plain ASCII circuit diagram.
    #[must_use]
    pub fn to_ascii(&self, num_qubits: usize) -> String {
        self.render_with(num_qubits, &DiagramStyle::default())
            .ascii()
    }

    /// ASCII circuit diagram with ANSI colors.
    #[must_use]
    pub fn to_color_ascii(&self, num_qubits: usize) -> String {
        self.render_with(
            num_qubits,
            &DiagramStyle::builder().ansi_color(true).build(),
        )
        .ascii()
    }

    /// Unicode circuit diagram.
    #[must_use]
    pub fn to_unicode(&self, num_qubits: usize) -> String {
        self.render_with(
            num_qubits,
            &DiagramStyle::builder().symbols(SymbolSet::Unicode).build(),
        )
        .unicode()
    }

    /// Unicode circuit diagram with ANSI colors.
    #[must_use]
    pub fn to_color_unicode(&self, num_qubits: usize) -> String {
        self.render_with(
            num_qubits,
            &DiagramStyle::builder()
                .symbols(SymbolSet::Unicode)
                .ansi_color(true)
                .build(),
        )
        .unicode()
    }

    /// Export as an SVG circuit diagram.
    #[must_use]
    pub fn to_svg(&self, num_qubits: usize) -> String {
        self.render_with(num_qubits, &DiagramStyle::default()).svg()
    }

    /// Export as a `TikZ` `tikzpicture`.
    #[must_use]
    pub fn to_tikz(&self, num_qubits: usize) -> String {
        self.render_with(num_qubits, &DiagramStyle::default())
            .tikz()
    }

    /// Export as a Graphviz DOT digraph.
    #[must_use]
    pub fn to_dot(&self, num_qubits: usize) -> String {
        self.render_with(num_qubits, &DiagramStyle::default()).dot()
    }

    /// Create a [`DiagramRenderer`] bound to a custom [`DiagramStyle`].
    ///
    /// The renderer can produce text, SVG, `TikZ`, or DOT output using the
    /// given style configuration.
    #[must_use]
    pub fn render_with<'a>(
        &self,
        num_qubits: usize,
        style: &'a DiagramStyle,
    ) -> DiagramRenderer<'a> {
        let mut diagram = CircuitDiagram::new(num_qubits);
        self.add_to_diagram(&mut diagram);
        DiagramRenderer::new(diagram, String::new(), style)
    }

    fn add_to_diagram(&self, diagram: &mut CircuitDiagram) {
        match self {
            Self::Pauli(ps) => {
                for (pauli, qubit) in ps.iter_pairs() {
                    let q = usize::from(qubit);
                    let (name, color) = match pauli {
                        crate::Pauli::I => continue,
                        crate::Pauli::X => ("X", CellColor::XAxis),
                        crate::Pauli::Y => ("Y", CellColor::YAxis),
                        crate::Pauli::Z => ("Z", CellColor::ZAxis),
                    };
                    diagram.add_gate(q, name, color, GateFamily::Default);
                }
            }
            Self::Gate(
                Unitary::Rotation {
                    rotation_type,
                    angle,
                },
                qubits,
            ) => {
                let resolved_gt = rotation_to_gate_type(*rotation_type, *angle);
                let name = if let Some(gt) = resolved_gt {
                    format!("{gt:?}")
                } else {
                    format!("{rotation_type:?}")
                };
                let family = resolved_gt.map_or(GateFamily::Default, gate_type_family);
                let color = resolved_gt.map_or(CellColor::None, gate_type_color);

                if qubits.len() == 1 {
                    diagram.add_gate(qubits[0], &name, color, family);
                } else if qubits.len() == 2 {
                    diagram.add_gate(qubits[0], &name, color, family);
                    diagram.add_gate(qubits[1], &name, color, family);
                    diagram.connect_vertical(qubits[0], qubits[1], CellColor::None);
                }
            }
            Self::Gate(Unitary::R1XY { .. }, qubits) => {
                diagram.add_gate(qubits[0], "R1XY", CellColor::None, GateFamily::Default);
            }
            Self::Gate(Unitary::U3 { .. }, qubits) => {
                diagram.add_gate(qubits[0], "U", CellColor::None, GateFamily::Default);
            }
            Self::Gate(Unitary::RXXRYYRZZ { .. }, qubits) => {
                diagram.add_gate(qubits[0], "RXXRYYRZZ", CellColor::None, GateFamily::Default);
                diagram.add_gate(qubits[1], "RXXRYYRZZ", CellColor::None, GateFamily::Default);
                diagram.connect_vertical(qubits[0], qubits[1], CellColor::None);
            }
            Self::Gate(Unitary::U2q { .. }, qubits) => {
                diagram.add_gate(qubits[0], "U2q", CellColor::None, GateFamily::Default);
                diagram.add_gate(qubits[1], "U2q", CellColor::None, GateFamily::Default);
                diagram.connect_vertical(qubits[0], qubits[1], CellColor::None);
            }
            Self::Gate(Unitary::Named(gate_type), qubits) => match gate_type {
                GateType::CX => {
                    diagram.add_control(qubits[0]);
                    diagram.add_gate(qubits[1], "X", CellColor::XAxis, GateFamily::Default);
                    diagram.connect_vertical(qubits[0], qubits[1], CellColor::None);
                }
                GateType::CY => {
                    diagram.add_control(qubits[0]);
                    diagram.add_gate(qubits[1], "Y", CellColor::YAxis, GateFamily::Default);
                    diagram.connect_vertical(qubits[0], qubits[1], CellColor::None);
                }
                GateType::CZ => {
                    diagram.add_control(qubits[0]);
                    diagram.add_gate(qubits[1], "Z", CellColor::ZAxis, GateFamily::Default);
                    diagram.connect_vertical(qubits[0], qubits[1], CellColor::None);
                }
                GateType::SWAP => {
                    diagram.add_gate(qubits[0], "x", CellColor::None, GateFamily::Default);
                    diagram.add_gate(qubits[1], "x", CellColor::None, GateFamily::Default);
                    diagram.connect_vertical(qubits[0], qubits[1], CellColor::None);
                }
                GateType::CCX => {
                    diagram.add_control(qubits[0]);
                    diagram.add_control(qubits[1]);
                    diagram.add_gate(qubits[2], "X", CellColor::XAxis, GateFamily::Default);
                    let min_q = qubits[0].min(qubits[1]).min(qubits[2]);
                    let max_q = qubits[0].max(qubits[1]).max(qubits[2]);
                    diagram.connect_vertical(min_q, max_q, CellColor::None);
                }
                _ => {
                    if qubits.len() == 1 {
                        let family = gate_type_family(*gate_type);
                        let color = gate_type_color(*gate_type);
                        diagram.add_gate(qubits[0], &format!("{gate_type:?}"), color, family);
                    }
                }
            },
            Self::Tensor(parts) => {
                for part in parts {
                    part.add_to_diagram(diagram);
                }
            }
            Self::Compose(parts) => {
                for part in parts {
                    part.add_to_diagram(diagram);
                    diagram.advance();
                }
            }
            Self::Adjoint(inner) | Self::Phase { inner, .. } => {
                inner.add_to_diagram(diagram);
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pauli;

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
        if let UnitaryRep::Pauli(ps) = &op {
            assert_eq!(ps.get(0), crate::Pauli::X);
            assert_eq!(ps.get(1), crate::Pauli::Z);
        } else {
            panic!("Expected Pauli, got {op:?}");
        }

        // Mixed tensor (Pauli with non-Pauli) produces Tensor
        let mixed = X(0) & H(1);
        assert!(matches!(mixed, UnitaryRep::Tensor(_)));
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
        if let UnitaryRep::Gate(Unitary::Rotation { angle, .. }, _) = t_dg {
            let t_angle = Angle64::HALF_TURN / 4;
            let expected = negate_angle(t_angle);
            assert_eq!(angle, expected);
        } else {
            panic!("Expected Gate(Rotation)");
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
        assert!(diagram.contains("[H]")); // Default family
    }

    #[test]
    fn test_diagram_cx() {
        let cx = CX(0, 1);
        let diagram = cx.to_diagram(2);
        assert!(diagram.contains("\u{25CF}")); // control dot
        assert!(diagram.contains("[X]")); // Default family for controlled target
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

        if let UnitaryRep::Compose(parts) = circuit {
            assert_eq!(parts.len(), 3);
            // All should be rotations
            assert!(matches!(
                &parts[0],
                UnitaryRep::Gate(Unitary::Rotation { .. }, _)
            ));
            assert!(matches!(
                &parts[1],
                UnitaryRep::Gate(Unitary::Rotation { .. }, _)
            ));
            assert!(matches!(
                &parts[2],
                UnitaryRep::Gate(Unitary::Rotation { .. }, _)
            ));
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
        if let UnitaryRep::Gate(
            Unitary::Rotation {
                angle,
                rotation_type,
            },
            _,
        ) = simplified
        {
            assert_eq!(rotation_type, RotationType::RZ);
            assert_eq!(angle, Angle64::HALF_TURN); // π
        } else {
            panic!("Expected single Gate(Rotation), got {simplified:?}");
        }
    }

    #[test]
    fn test_simplify_preserves_different_qubits() {
        // RZ on different qubits shouldn't merge
        let circuit = RZ(Angle64::QUARTER_TURN, 0) * RZ(Angle64::QUARTER_TURN, 1);
        let simplified = circuit.simplify();

        // Should remain a Compose with 2 parts
        if let UnitaryRep::Compose(parts) = simplified {
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
        if let UnitaryRep::Compose(parts) = simplified {
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

        if let UnitaryRep::Gate(Unitary::Rotation { angle, .. }, _) = simplified {
            assert_eq!(angle, Angle64::HALF_TURN);
        } else {
            panic!("Expected single Gate(Rotation)");
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
        if let UnitaryRep::Compose(parts) = simplified {
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

        if let UnitaryRep::Phase { phase: p, inner } = op {
            assert_eq!(p, eighth_turn);
            // Inner should be Pauli(X)
            assert!(matches!(*inner, UnitaryRep::Pauli(_)));
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
        if let (UnitaryRep::Pauli(ps1), UnitaryRep::Pauli(ps2)) = (&op1, &op2) {
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
        assert!(matches!(op, UnitaryRep::Pauli(_)));
        if let UnitaryRep::Pauli(ps) = op {
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
        if let (UnitaryRep::Pauli(ps1), UnitaryRep::Pauli(ps2)) = (&op1, &op2) {
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
        if let (UnitaryRep::Pauli(ps1), UnitaryRep::Pauli(ps2)) = (&op1, &op2) {
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
        if let UnitaryRep::Pauli(ps) = converted {
            assert_eq!(ps.get(0), crate::Pauli::X);
        } else {
            panic!("Expected Pauli variant");
        }

        // RY(π) = Y
        let ry_pi = RY(Angle64::HALF_TURN, 1);
        let converted = ry_pi.try_to_pauli().expect("Should convert");
        if let UnitaryRep::Pauli(ps) = converted {
            assert_eq!(ps.get(1), crate::Pauli::Y);
        } else {
            panic!("Expected Pauli variant");
        }

        // RZ(π) = Z
        let rz_pi = RZ(Angle64::HALF_TURN, 2);
        let converted = rz_pi.try_to_pauli().expect("Should convert");
        if let UnitaryRep::Pauli(ps) = converted {
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
        if let (UnitaryRep::Pauli(ps1), UnitaryRep::Pauli(ps2)) = (&multi, &tensor) {
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

        if let UnitaryRep::Tensor(parts) = multi {
            assert_eq!(parts.len(), 3);
            // Each should be a rotation
            assert!(matches!(
                &parts[0],
                UnitaryRep::Gate(Unitary::Rotation { .. }, _)
            ));
            assert!(matches!(
                &parts[1],
                UnitaryRep::Gate(Unitary::Rotation { .. }, _)
            ));
            assert!(matches!(
                &parts[2],
                UnitaryRep::Gate(Unitary::Rotation { .. }, _)
            ));
        } else {
            panic!("Expected Tensor variant, got {multi:?}");
        }
    }

    #[test]
    fn test_h_multi_qubit() {
        // Hs([0, 1]) should be a tensor of H gates
        let multi = Hs([0, 1]);

        if let UnitaryRep::Tensor(parts) = multi {
            assert_eq!(parts.len(), 2);
            // Each should be a Compose (H = RZ * RY)
            assert!(matches!(&parts[0], UnitaryRep::Compose(_)));
            assert!(matches!(&parts[1], UnitaryRep::Compose(_)));
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

        assert!(matches!(x, UnitaryRep::Pauli(_)));
        assert!(matches!(t, UnitaryRep::Gate(Unitary::Rotation { .. }, _)));
        assert!(matches!(
            h,
            UnitaryRep::Gate(Unitary::Named(GateType::H), _)
        ));
    }

    #[test]
    fn test_range_syntax() {
        // Range syntax: Xs(0..3) = X(0) & X(1) & X(2)
        let multi_range = Xs(0..3);
        let tensor = X(0) & X(1) & X(2);

        if let (UnitaryRep::Pauli(ps1), UnitaryRep::Pauli(ps2)) = (&multi_range, &tensor) {
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

        if let (UnitaryRep::Pauli(ps1), UnitaryRep::Pauli(ps2)) = (&multi_range, &tensor) {
            assert_eq!(ps1, ps2);
        } else {
            panic!("Expected Pauli variants");
        }
    }

    #[test]
    fn test_identity_range_syntax() {
        // Is(0..=2) should create identity operators on qubits 0, 1, 2
        let identities = Is(0..=2);

        if let UnitaryRep::Tensor(parts) = identities {
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

        if let (UnitaryRep::Pauli(ps1), UnitaryRep::Pauli(ps2)) = (&from_range, &direct) {
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
            UnitaryRep::Gate(
                Unitary::Rotation {
                    angle: a1,
                    rotation_type: r1,
                },
                _,
            ),
            UnitaryRep::Gate(
                Unitary::Rotation {
                    angle: a2,
                    rotation_type: r2,
                },
                _,
            ),
        ) = (&from_range, &direct)
        {
            assert_eq!(a1, a2);
            assert_eq!(r1, r2);
        } else {
            panic!("Expected Gate(Rotation) variants, got {from_range:?} and {direct:?}");
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
        if let UnitaryRep::Pauli(ps) = simplified {
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
        if let UnitaryRep::Pauli(ps) = simplified {
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
        assert!(matches!(result, UnitaryRep::Compose(_)));
    }

    #[test]
    fn test_conj_z_by_z() {
        // Z.conj(Z) = Z * Z * Z† = Z * Z * Z = Z (since Z² = I, Z³ = Z)
        let z = Z(0);
        let result = z.conj(&z);

        let simplified = result.simplify();
        if let UnitaryRep::Pauli(ps) = simplified {
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
        if let UnitaryRep::Compose(parts) = result {
            assert_eq!(parts.len(), 3);
            // First element is SZ (positive angle)
            assert!(matches!(
                &parts[0],
                UnitaryRep::Gate(
                    Unitary::Rotation {
                        rotation_type: RotationType::RZ,
                        ..
                    },
                    _
                )
            ));
            // Middle element is X
            assert!(matches!(&parts[1], UnitaryRep::Pauli(_)));
            // Last element is SZ† (negative angle)
            assert!(matches!(
                &parts[2],
                UnitaryRep::Gate(
                    Unitary::Rotation {
                        rotation_type: RotationType::RZ,
                        ..
                    },
                    _
                )
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
        if let UnitaryRep::Compose(parts) = result {
            assert_eq!(parts.len(), 3);
            // First element is SZ† (negative angle)
            assert!(matches!(
                &parts[0],
                UnitaryRep::Gate(
                    Unitary::Rotation {
                        rotation_type: RotationType::RZ,
                        ..
                    },
                    _
                )
            ));
            // Middle element is X
            assert!(matches!(&parts[1], UnitaryRep::Pauli(_)));
            // Last element is SZ (positive angle)
            assert!(matches!(
                &parts[2],
                UnitaryRep::Gate(
                    Unitary::Rotation {
                        rotation_type: RotationType::RZ,
                        ..
                    },
                    _
                )
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
        if let UnitaryRep::Pauli(ps) = simplified {
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
        if let UnitaryRep::Pauli(ps) = result {
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
        assert!(matches!(result, UnitaryRep::Compose(_)));
    }

    #[test]
    fn test_pow_rotation_simplify() {
        // T^2 = S (RZ(π/4)^2 = RZ(π/2)) after simplification
        let t = T(0);
        let result = t.pow(2).simplify();

        if let UnitaryRep::Gate(
            Unitary::Rotation {
                angle,
                rotation_type,
            },
            _,
        ) = result
        {
            assert_eq!(rotation_type, RotationType::RZ);
            assert_eq!(angle, Angle64::QUARTER_TURN);
        } else {
            panic!("Expected Gate(Rotation), got {result:?}");
        }
    }

    #[test]
    fn test_pow_four_t_simplify() {
        // T^4 = Z (RZ(π/4)^4 = RZ(π)) after simplification
        let t = T(0);
        let result = t.pow(4).simplify();

        if let UnitaryRep::Gate(Unitary::Rotation { angle, .. }, _) = result {
            assert_eq!(angle, Angle64::HALF_TURN);
        } else {
            panic!("Expected Gate(Rotation), got {result:?}");
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
        let op = UnitaryRep::Pauli(ps);
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

    // ====================== SVG/TikZ/DOT export ======================

    #[test]
    fn operator_svg() {
        let op = H(0);
        let svg = op.to_svg(2);
        assert!(svg.contains("<svg"));
        assert!(svg.contains(">H</text>"));
        assert!(svg.contains("q0</text>"));
    }

    #[test]
    fn operator_tikz() {
        let op = H(0);
        let tikz = op.to_tikz(2);
        assert!(tikz.contains("\\begin{tikzpicture}"));
        assert!(tikz.contains("{H}"));
    }

    #[test]
    fn operator_dot() {
        let op = H(0);
        let dot = op.to_dot(2);
        assert!(dot.contains("digraph circuit"));
        assert!(dot.contains("label=\"H\""));
    }

    #[test]
    fn operator_cx_svg() {
        let op = CX(0, 1);
        let svg = op.to_svg(2);
        assert!(svg.contains("<circle")); // control dot
        assert!(svg.contains("<rect")); // gate box
    }

    // --- FromStr tests ---

    #[test]
    fn from_str_h_gate() {
        let op: UnitaryRep = "H 0".parse().unwrap();
        assert!(matches!(
            op,
            UnitaryRep::Gate(Unitary::Named(GateType::H), _)
        ));
    }

    #[test]
    fn from_str_cx_gate() {
        let op: UnitaryRep = "CX 0 1".parse().unwrap();
        assert!(matches!(
            op,
            UnitaryRep::Gate(Unitary::Named(GateType::CX), _)
        ));
    }

    #[test]
    fn from_str_cnot_alias() {
        let op: UnitaryRep = "CNOT 0 1".parse().unwrap();
        assert!(matches!(
            op,
            UnitaryRep::Gate(Unitary::Named(GateType::CX), _)
        ));
    }

    #[test]
    fn from_str_swap_gate() {
        let op: UnitaryRep = "SWAP 2 3".parse().unwrap();
        assert!(matches!(
            op,
            UnitaryRep::Gate(Unitary::Named(GateType::SWAP), _)
        ));
    }

    #[test]
    fn from_str_ccx_gate() {
        let op: UnitaryRep = "CCX 0 1 2".parse().unwrap();
        assert!(matches!(
            op,
            UnitaryRep::Gate(Unitary::Named(GateType::CCX), _)
        ));
    }

    #[test]
    fn from_str_t_gate() {
        let op: UnitaryRep = "T 0".parse().unwrap();
        assert!(matches!(
            op,
            UnitaryRep::Gate(
                Unitary::Rotation {
                    rotation_type: RotationType::RZ,
                    ..
                },
                _
            )
        ));
    }

    #[test]
    fn from_str_s_gate() {
        let op: UnitaryRep = "S 0".parse().unwrap();
        assert!(matches!(
            op,
            UnitaryRep::Gate(
                Unitary::Rotation {
                    rotation_type: RotationType::RZ,
                    ..
                },
                _
            )
        ));
    }

    #[test]
    fn from_str_rz_with_angle() {
        let op: UnitaryRep = "RZ(pi/4) 0".parse().unwrap();
        match op {
            UnitaryRep::Gate(
                Unitary::Rotation {
                    rotation_type,
                    angle,
                },
                qubits,
            ) => {
                assert_eq!(rotation_type, RotationType::RZ);
                assert_eq!(angle, Angle64::HALF_TURN / 4);
                assert_eq!(qubits[0], 0);
            }
            _ => panic!("Expected Gate(Rotation)"),
        }
    }

    #[test]
    fn from_str_rx_with_angle() {
        let op: UnitaryRep = "RX(pi/2) 3".parse().unwrap();
        match op {
            UnitaryRep::Gate(
                Unitary::Rotation {
                    rotation_type,
                    angle,
                },
                qubits,
            ) => {
                assert_eq!(rotation_type, RotationType::RX);
                assert_eq!(angle, Angle64::QUARTER_TURN);
                assert_eq!(qubits[0], 3);
            }
            _ => panic!("Expected Gate(Rotation)"),
        }
    }

    #[test]
    fn from_str_rzz_two_qubit() {
        let op: UnitaryRep = "RZZ(pi) 0 1".parse().unwrap();
        match op {
            UnitaryRep::Gate(
                Unitary::Rotation {
                    rotation_type,
                    angle,
                },
                qubits,
            ) => {
                assert_eq!(rotation_type, RotationType::RZZ);
                assert_eq!(angle, Angle64::HALF_TURN);
                assert_eq!(qubits.as_slice(), &[0, 1]);
            }
            _ => panic!("Expected Gate(Rotation)"),
        }
    }

    #[test]
    fn from_str_pauli_sparse() {
        let op: UnitaryRep = "X0 Z1".parse().unwrap();
        match op {
            UnitaryRep::Pauli(ps) => {
                assert_eq!(ps.get(0), Pauli::X);
                assert_eq!(ps.get(1), Pauli::Z);
            }
            _ => panic!("Expected Pauli"),
        }
    }

    #[test]
    fn from_str_pauli_dense() {
        let op: UnitaryRep = "XZI".parse().unwrap();
        match op {
            UnitaryRep::Pauli(ps) => {
                assert_eq!(ps.get(0), Pauli::X);
                assert_eq!(ps.get(1), Pauli::Z);
                assert_eq!(ps.get(2), Pauli::I);
            }
            _ => panic!("Expected Pauli"),
        }
    }

    #[test]
    fn from_str_x_space_qubit() {
        // "X 0" and "X0" should produce the same result
        let op1: UnitaryRep = "X 0".parse().unwrap();
        let op2: UnitaryRep = "X0".parse().unwrap();
        match (&op1, &op2) {
            (UnitaryRep::Pauli(ps1), UnitaryRep::Pauli(ps2)) => {
                assert_eq!(ps1.get(0), Pauli::X);
                assert_eq!(ps2.get(0), Pauli::X);
                assert_eq!(ps1.phase(), ps2.phase());
            }
            _ => panic!("Expected Pauli for both"),
        }
    }

    #[test]
    fn from_str_z_space_qubit() {
        let op: UnitaryRep = "Z 3".parse().unwrap();
        match op {
            UnitaryRep::Pauli(ps) => {
                assert_eq!(ps.get(3), Pauli::Z);
            }
            _ => panic!("Expected Pauli"),
        }
    }

    #[test]
    fn from_str_multi_pauli_spaced() {
        // "X 0 Z 1" should work the same as "X0 Z1"
        let op: UnitaryRep = "X 0 Z 1".parse().unwrap();
        match op {
            UnitaryRep::Pauli(ps) => {
                assert_eq!(ps.get(0), Pauli::X);
                assert_eq!(ps.get(1), Pauli::Z);
            }
            _ => panic!("Expected Pauli"),
        }
    }

    #[test]
    fn from_str_pauli_with_phase() {
        let op: UnitaryRep = "-i X2 Z4".parse().unwrap();
        match op {
            UnitaryRep::Pauli(ps) => {
                assert_eq!(ps.phase(), QuarterPhase::MinusI);
                assert_eq!(ps.get(2), Pauli::X);
                assert_eq!(ps.get(4), Pauli::Z);
            }
            _ => panic!("Expected Pauli"),
        }
    }

    #[test]
    fn from_str_wrong_qubit_count() {
        assert!("H 0 1".parse::<UnitaryRep>().is_err());
        assert!("CX 0".parse::<UnitaryRep>().is_err());
        assert!("CCX 0 1".parse::<UnitaryRep>().is_err());
    }

    #[test]
    fn from_str_case_insensitive() {
        assert!("h 0".parse::<UnitaryRep>().is_ok());
        assert!("cx 0 1".parse::<UnitaryRep>().is_ok());
        assert!("swap 0 1".parse::<UnitaryRep>().is_ok());
    }

    #[test]
    fn from_str_empty() {
        let op: UnitaryRep = "".parse().unwrap();
        assert!(matches!(op, UnitaryRep::Pauli(_)));
    }

    // --- parse_angle_expr tests ---

    #[test]
    fn parse_angle_pi() {
        let angle = parse_angle_expr("pi").unwrap();
        assert_eq!(angle, Angle64::HALF_TURN);
    }

    #[test]
    fn parse_angle_pi_over_4() {
        let angle = parse_angle_expr("pi/4").unwrap();
        assert_eq!(angle, Angle64::HALF_TURN / 4);
    }

    #[test]
    fn parse_angle_pi_over_2() {
        let angle = parse_angle_expr("pi/2").unwrap();
        assert_eq!(angle, Angle64::QUARTER_TURN);
    }

    #[test]
    fn parse_angle_2_pi_over_3() {
        let angle = parse_angle_expr("2*pi/3").unwrap();
        assert_eq!(angle, Angle64::HALF_TURN * 2 / 3);
    }

    #[test]
    fn parse_angle_negative_pi() {
        let angle = parse_angle_expr("-pi").unwrap();
        assert_eq!(angle, Angle64::ZERO - Angle64::HALF_TURN);
    }

    #[test]
    fn parse_angle_negative_pi_over_4() {
        let angle = parse_angle_expr("-pi/4").unwrap();
        assert_eq!(angle, Angle64::ZERO - Angle64::HALF_TURN / 4);
    }

    #[test]
    fn parse_angle_with_whitespace() {
        let angle = parse_angle_expr("  pi / 4  ").unwrap();
        assert_eq!(angle, Angle64::HALF_TURN / 4);
    }

    #[test]
    fn parse_angle_empty_is_error() {
        assert!(parse_angle_expr("").is_err());
    }

    #[test]
    fn parse_angle_division_by_zero() {
        assert!(parse_angle_expr("pi/0").is_err());
    }

    #[test]
    fn parse_angle_invalid_format() {
        assert!(parse_angle_expr("hello").is_err());
    }

    #[test]
    fn parse_angle_n_pi_division_by_zero() {
        assert!(parse_angle_expr("2*pi/0").is_err());
    }

    #[test]
    fn parse_angle_invalid_numerator() {
        assert!(parse_angle_expr("abc*pi/4").is_err());
    }

    #[test]
    fn parse_angle_invalid_denominator() {
        assert!(parse_angle_expr("pi/xyz").is_err());
    }

    #[test]
    fn parse_angle_3_pi() {
        let angle = parse_angle_expr("3*pi").unwrap();
        assert_eq!(angle, Angle64::HALF_TURN * 3);
    }

    // --- from_str additional edge cases ---

    #[test]
    fn from_str_tdg_gate() {
        let op: UnitaryRep = "Tdg 0".parse().unwrap();
        assert!(matches!(
            op,
            UnitaryRep::Gate(
                Unitary::Rotation {
                    rotation_type: RotationType::RZ,
                    ..
                },
                _
            )
        ));
    }

    #[test]
    fn from_str_sdg_gate() {
        let op: UnitaryRep = "Sdg 0".parse().unwrap();
        assert!(matches!(
            op,
            UnitaryRep::Gate(
                Unitary::Rotation {
                    rotation_type: RotationType::RZ,
                    ..
                },
                _
            )
        ));
    }

    #[test]
    fn from_str_ry_with_angle() {
        let op: UnitaryRep = "RY(pi/4) 0".parse().unwrap();
        assert!(matches!(
            op,
            UnitaryRep::Gate(
                Unitary::Rotation {
                    rotation_type: RotationType::RY,
                    ..
                },
                _
            )
        ));
    }

    #[test]
    fn from_str_rxx_two_qubit() {
        let op: UnitaryRep = "RXX(pi/2) 0 1".parse().unwrap();
        assert!(matches!(
            op,
            UnitaryRep::Gate(
                Unitary::Rotation {
                    rotation_type: RotationType::RXX,
                    ..
                },
                _
            )
        ));
    }

    #[test]
    fn from_str_ryy_two_qubit() {
        let op: UnitaryRep = "RYY(pi) 0 1".parse().unwrap();
        assert!(matches!(
            op,
            UnitaryRep::Gate(
                Unitary::Rotation {
                    rotation_type: RotationType::RYY,
                    ..
                },
                _
            )
        ));
    }

    #[test]
    fn from_str_rotation_wrong_qubit_count() {
        assert!("RZ(pi/4) 0 1".parse::<UnitaryRep>().is_err());
        assert!("RZZ(pi) 0".parse::<UnitaryRep>().is_err());
    }

    #[test]
    fn from_str_cz_gate() {
        let op: UnitaryRep = "CZ 0 1".parse().unwrap();
        assert!(matches!(
            op,
            UnitaryRep::Gate(Unitary::Named(GateType::CZ), _)
        ));
    }

    #[test]
    fn from_str_cy_gate() {
        let op: UnitaryRep = "CY 0 1".parse().unwrap();
        assert!(matches!(
            op,
            UnitaryRep::Gate(Unitary::Named(GateType::CY), _)
        ));
    }

    #[test]
    fn from_str_toffoli_alias() {
        let op: UnitaryRep = "TOFFOLI 0 1 2".parse().unwrap();
        assert!(matches!(
            op,
            UnitaryRep::Gate(Unitary::Named(GateType::CCX), _)
        ));
    }

    #[test]
    fn from_str_sx_gate() {
        let op: UnitaryRep = "SX 0".parse().unwrap();
        assert!(matches!(
            op,
            UnitaryRep::Gate(Unitary::Named(GateType::SX), _)
        ));
    }

    #[test]
    fn from_str_f_gate() {
        let op: UnitaryRep = "F 0".parse().unwrap();
        assert!(matches!(
            op,
            UnitaryRep::Gate(Unitary::Named(GateType::F), _)
        ));
    }

    // ========================================================================
    // Unitary base type tests
    // ========================================================================

    #[test]
    fn unitary_named_is_clifford() {
        assert!(Unitary::Named(GateType::H).is_clifford());
        assert!(Unitary::Named(GateType::X).is_clifford());
        assert!(Unitary::Named(GateType::CX).is_clifford());
        assert!(Unitary::Named(GateType::SWAP).is_clifford());
        assert!(!Unitary::Named(GateType::T).is_clifford());
    }

    #[test]
    fn unitary_rotation_is_clifford() {
        // Quarter-turn multiples are Clifford
        assert!(
            Unitary::Rotation {
                rotation_type: RotationType::RZ,
                angle: Angle64::QUARTER_TURN,
            }
            .is_clifford()
        );
        assert!(
            Unitary::Rotation {
                rotation_type: RotationType::RX,
                angle: Angle64::HALF_TURN,
            }
            .is_clifford()
        );
        assert!(
            Unitary::Rotation {
                rotation_type: RotationType::RY,
                angle: Angle64::ZERO,
            }
            .is_clifford()
        );

        // Non-quarter-turn is not Clifford
        assert!(
            !Unitary::Rotation {
                rotation_type: RotationType::RZ,
                angle: Angle64::from_turn_ratio(1, 8),
            }
            .is_clifford()
        );
    }

    #[test]
    fn unitary_is_identity() {
        assert!(Unitary::Named(GateType::I).is_identity());
        assert!(!Unitary::Named(GateType::H).is_identity());

        assert!(
            Unitary::Rotation {
                rotation_type: RotationType::RZ,
                angle: Angle64::ZERO,
            }
            .is_identity()
        );
        assert!(
            !Unitary::Rotation {
                rotation_type: RotationType::RZ,
                angle: Angle64::QUARTER_TURN,
            }
            .is_identity()
        );
    }

    #[test]
    fn unitary_to_gate_type() {
        assert_eq!(
            Unitary::Named(GateType::H).to_gate_type(),
            Some(GateType::H)
        );
        assert_eq!(
            Unitary::Named(GateType::CX).to_gate_type(),
            Some(GateType::CX)
        );

        // Quarter-turn RZ -> SZ
        assert_eq!(
            Unitary::Rotation {
                rotation_type: RotationType::RZ,
                angle: Angle64::QUARTER_TURN,
            }
            .to_gate_type(),
            Some(GateType::SZ)
        );

        // Half-turn RZ -> Z
        assert_eq!(
            Unitary::Rotation {
                rotation_type: RotationType::RZ,
                angle: Angle64::HALF_TURN,
            }
            .to_gate_type(),
            Some(GateType::Z)
        );

        // Non-Clifford rotation -> None
        assert_eq!(
            Unitary::Rotation {
                rotation_type: RotationType::RZ,
                angle: Angle64::from_turn_ratio(1, 5),
            }
            .to_gate_type(),
            None
        );
    }

    #[test]
    fn unitary_num_qubits() {
        assert_eq!(Unitary::Named(GateType::H).num_qubits(), 1);
        assert_eq!(Unitary::Named(GateType::X).num_qubits(), 1);
        assert_eq!(Unitary::Named(GateType::CX).num_qubits(), 2);
        assert_eq!(Unitary::Named(GateType::SWAP).num_qubits(), 2);
        assert_eq!(
            Unitary::Rotation {
                rotation_type: RotationType::RZ,
                angle: Angle64::QUARTER_TURN,
            }
            .num_qubits(),
            1
        );
        assert_eq!(
            Unitary::Rotation {
                rotation_type: RotationType::RZZ,
                angle: Angle64::QUARTER_TURN,
            }
            .num_qubits(),
            2
        );
    }

    #[test]
    fn unitary_on_qubit() {
        let h = Unitary::Named(GateType::H);
        let rep = h.on_qubit(3);
        assert!(
            matches!(rep, UnitaryRep::Gate(Unitary::Named(GateType::H), ref q) if q.as_slice() == [3])
        );
    }

    #[test]
    fn unitary_on_qubits() {
        let cx = Unitary::Named(GateType::CX);
        let rep = cx.on_qubits(2, 5);
        assert!(
            matches!(rep, UnitaryRep::Gate(Unitary::Named(GateType::CX), ref q) if q.as_slice() == [2, 5])
        );

        // 1q gate ignores second qubit
        let h = Unitary::Named(GateType::H);
        let rep = h.on_qubits(3, 99);
        assert!(
            matches!(rep, UnitaryRep::Gate(Unitary::Named(GateType::H), ref q) if q.as_slice() == [3])
        );
    }

    #[test]
    #[should_panic(expected = "on_qubit called on 2-qubit gate")]
    fn unitary_on_qubit_panics_for_2q() {
        let _ = Unitary::Named(GateType::CX).on_qubit(0);
    }

    #[test]
    fn unitary_mul_produces_compose() {
        let h = Unitary::Named(GateType::H);
        let sx = Unitary::Rotation {
            rotation_type: RotationType::RX,
            angle: Angle64::QUARTER_TURN,
        };
        let result = h * sx;
        // H * SX: apply SX first, then H, both on qubit 0
        if let UnitaryRep::Compose(parts) = &result {
            assert_eq!(parts.len(), 2);
            assert_eq!(result.qubits(), vec![0]);
        } else {
            panic!("Expected Compose, got {result:?}");
        }
    }

    #[test]
    fn unitary_tensor_produces_tensor() {
        let h = Unitary::Named(GateType::H);
        let x = Unitary::Named(GateType::X);
        let result = h & x;
        // H on qubit 0, X on qubit 1
        if let UnitaryRep::Tensor(parts) = &result {
            assert_eq!(parts.len(), 2);
        } else {
            panic!("Expected Tensor, got {result:?}");
        }
        let qubits = result.qubits();
        assert!(qubits.contains(&0));
        assert!(qubits.contains(&1));
    }

    #[test]
    fn unitary_tensor_2q_gates() {
        let cx = Unitary::Named(GateType::CX);
        let h = Unitary::Named(GateType::H);
        let result = cx & h;
        // CX on qubits 0,1 then H on qubit 2
        let qubits = result.qubits();
        assert_eq!(qubits, vec![0, 1, 2]);
    }

    #[test]
    fn unitary_eq_and_hash() {
        use std::collections::HashSet;

        let a = Unitary::Named(GateType::H);
        let b = Unitary::Named(GateType::H);
        let c = Unitary::Named(GateType::X);

        assert_eq!(a, b);
        assert_ne!(a, c);

        let mut set = HashSet::new();
        set.insert(a);
        assert!(set.contains(&b));
        assert!(!set.contains(&c));

        // Rotation variants
        let r1 = Unitary::Rotation {
            rotation_type: RotationType::RZ,
            angle: Angle64::QUARTER_TURN,
        };
        let r2 = Unitary::Rotation {
            rotation_type: RotationType::RZ,
            angle: Angle64::QUARTER_TURN,
        };
        let r3 = Unitary::Rotation {
            rotation_type: RotationType::RX,
            angle: Angle64::QUARTER_TURN,
        };

        assert_eq!(r1, r2);
        assert_ne!(r1, r3);
        set.insert(r1);
        assert!(set.contains(&r2));
        assert!(!set.contains(&r3));
    }
}
