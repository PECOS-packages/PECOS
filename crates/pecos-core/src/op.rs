// Copyright 2026 The PECOS Developers
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

//! Unified quantum operation algebra with automatic type promotion.
//!
//! [`Op`] wraps five algebraic levels — [`PauliString`], [`CliffordRep`],
//! [`UnitaryRep`], [`GateExpr`], and [`ChannelExpr`] — and automatically promotes to the
//! tightest level that can represent a combination.
//!
//! # Promotion Hierarchy
//!
//! ```text
//! Pauli  ⊂  Clifford  ⊂  Unitary  ⊂  Gate  ⊂  Channel
//! ```
//!
//! Combining two `Op` values via tensor (`&`) or composition (`*`) promotes
//! to the maximum level of the operands. The first three levels support full
//! algebraic operations including adjoint (`dg()`). The Gate and Channel levels
//! support tensor and composition but not adjoint.
//!
//! # Examples
//!
//! ```
//! use pecos_core::op::*;
//!
//! // Pauli & Pauli stays Pauli
//! let p = X(0) & Y(3);
//! assert!(p.is_pauli());
//!
//! // Pauli & Clifford promotes to Clifford
//! let c = X(0) & H(3);
//! assert!(c.is_clifford());
//!
//! // Adding a non-Clifford promotes to Unitary
//! let u = X(0) & H(3) & T(5);
//! assert!(u.is_unitary());
//!
//! // Adding a measurement promotes to Gate
//! let g = H(0) & MZ(1);
//! assert!(g.is_gate());
//!
//! // Adding a noise channel promotes to Channel
//! let ch = g & Depolarizing(0.01, 2);
//! assert!(ch.is_channel());
//! ```

use crate::clifford_rep::CliffordRep;
use crate::qubit_support::{assert_distinct_qubits, overlapping_qubits};
use crate::unitary_rep::{PhaseValue, UnitaryRep};
use crate::{Angle64, PauliString, QubitId};
use std::fmt;
use std::ops::{BitAnd, Mul, Neg};

// Re-export phase types so `use pecos_core::op::*` gives the full algebra vocabulary.
pub use crate::pauli::algebra::{ImaginaryUnit, NegImaginaryUnit};
pub use crate::unitary_rep::phase;

/// Unified quantum operation with automatic level promotion.
///
/// Wraps one of five algebraic levels and promotes to the tightest
/// level when combined via `&` (tensor) or `*` (composition).
///
/// The Clifford variant stores both a [`CliffordRep`] (for efficient Clifford
/// algebra) and a [`UnitaryRep`] (for promotion to the Unitary level).
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    /// Pauli level: tensor products of single-qubit Paulis with a phase.
    Pauli(PauliString),
    /// Clifford level: tableau representation paired with the equivalent expression tree.
    Clifford(CliffordRep, UnitaryRep),
    /// General unitary level: expression tree.
    Unitary(UnitaryRep),
    /// Gate level: ideal circuit operations such as unitary gates, preparation,
    /// measurement, and reset.
    Gate(GateExpr),
    /// Channel level: general CPTP maps and noise/decoherence operations.
    Channel(ChannelExpr),
}

/// The algebraic level of an [`Op`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Level {
    Pauli = 0,
    Clifford = 1,
    Unitary = 2,
    Gate = 3,
    Channel = 4,
}

/// Error returned by fallible tensor-product constructors when supports overlap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorProductError {
    overlapping_qubits: Vec<usize>,
}

impl TensorProductError {
    /// Qubits touched by both operands.
    #[must_use]
    pub fn overlapping_qubits(&self) -> &[usize] {
        &self.overlapping_qubits
    }
}

impl fmt::Display for TensorProductError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "tensor product requires disjoint operator support; overlapping qubits: {:?}",
            self.overlapping_qubits
        )
    }
}

impl std::error::Error for TensorProductError {}

/// An ideal circuit operation expression.
///
/// Gate expressions represent operations that can appear in an ideal circuit
/// block: unitaries, preparations, measurements, resets, and their tensor or
/// sequential combinations. Measurement record allocation is owned by the
/// surrounding circuit representation, not by this expression.
#[derive(Debug, Clone, PartialEq)]
pub enum GateExpr {
    /// A unitary operation lifted to the gate level.
    Unitary(UnitaryRep),
    /// Prepare qubit in a given basis eigenstate.
    Prep { basis: Basis, qubit: usize },
    /// Measure qubit in a given basis (produces a classical outcome).
    Measure { basis: Basis, qubit: usize },
    /// Reset qubit to the given basis eigenstate.
    Reset { basis: Basis, qubit: usize },
    /// Tensor product of gate expressions.
    Tensor(Vec<GateExpr>),
    /// Sequential composition: apply first element, then second, etc.
    Compose(Vec<GateExpr>),
}

/// A general quantum channel expression.
///
/// Channels include noise/decoherence maps, mixed unitaries, lifted ideal
/// gates, and their compositions. They compose and tensor like unitaries but
/// are not generally invertible.
#[derive(Debug, Clone, PartialEq)]
pub enum ChannelExpr {
    /// A unitary operation lifted to the channel level.
    Unitary(UnitaryRep),
    /// An ideal gate expression lifted to the channel level.
    Gate(GateExpr),
    /// Mixed-unitary channel: ρ → `Σ_k` `p_k` `U_k` ρ `U_k`†.
    ///
    /// Each entry is `(probability, unitary)` with probabilities summing to 1.
    /// Covers Pauli channels, depolarizing noise, dephasing, bit-flip, etc.
    MixedUnitary(Vec<(f64, UnitaryRep)>),
    /// Amplitude damping channel with parameter γ ∈ [0, 1].
    ///
    /// Kraus operators: K₀ = |0⟩⟨0| + √(1−γ)|1⟩⟨1|, K₁ = √γ |0⟩⟨1|.
    /// Models energy relaxation (T₁ decay).
    AmplitudeDamping { gamma: f64, qubit: usize },
    /// Phase damping channel with parameter λ ∈ [0, 1].
    ///
    /// Kraus operators: K₀ = diag(1, √(1−λ)), K₁ = diag(0, √λ).
    /// Models pure dephasing (T₂ process without T₁).
    PhaseDamping { lambda: f64, qubit: usize },
    /// Erasure channel with erasure probability p ∈ [0, 1].
    ///
    /// With probability (1−p) the qubit is untouched; with probability p it is
    /// replaced by the maximally mixed state and an erasure flag is raised.
    /// This is a heralded error — the location of the error is known.
    Erasure { prob: f64, qubit: usize },
    /// Leakage channel: qubit transitions to a non-computational state
    /// with probability `rate`.
    ///
    /// Models transitions |1⟩ → |2⟩ (or other leaked states) common in
    /// superconducting and trapped-ion qubits. The simulator must handle
    /// the extended Hilbert space.
    Leakage { rate: f64, qubit: usize },
    /// Tensor product of channel expressions on different qubits.
    Tensor(Vec<ChannelExpr>),
    /// Sequential composition: apply first element, then second, etc.
    Compose(Vec<ChannelExpr>),
}

/// Measurement/preparation basis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Basis {
    /// Computational basis (Z eigenstates |0>, |1>).
    Z,
    /// X basis (|+>, |->).
    X,
    /// Y basis.
    Y,
}

// --- Helper ---

fn cliff(cr: CliffordRep, ur: UnitaryRep) -> Op {
    Op::Clifford(cr, ur)
}

fn require_disjoint_support(lhs: &[usize], rhs: &[usize]) -> Result<(), TensorProductError> {
    let overlapping_qubits = overlapping_qubits(lhs.iter().copied(), rhs.iter().copied());
    if overlapping_qubits.is_empty() {
        Ok(())
    } else {
        Err(TensorProductError { overlapping_qubits })
    }
}

// --- Core methods ---

impl Op {
    /// Returns the algebraic level of this expression.
    #[must_use]
    pub fn level(&self) -> Level {
        match self {
            Op::Pauli(_) => Level::Pauli,
            Op::Clifford(..) => Level::Clifford,
            Op::Unitary(_) => Level::Unitary,
            Op::Gate(_) => Level::Gate,
            Op::Channel(_) => Level::Channel,
        }
    }

    #[must_use]
    pub fn is_pauli(&self) -> bool {
        matches!(self, Op::Pauli(_))
    }

    #[must_use]
    pub fn is_clifford(&self) -> bool {
        matches!(self, Op::Clifford(..))
    }

    #[must_use]
    pub fn is_unitary(&self) -> bool {
        matches!(self, Op::Unitary(_))
    }

    #[must_use]
    pub fn is_gate(&self) -> bool {
        matches!(self, Op::Gate(_))
    }

    #[must_use]
    pub fn is_channel(&self) -> bool {
        matches!(self, Op::Channel(_))
    }

    /// Extracts the inner `GateExpr`, if at the Gate level.
    #[must_use]
    pub fn as_gate(&self) -> Option<&GateExpr> {
        match self {
            Op::Gate(gate) => Some(gate),
            _ => None,
        }
    }

    /// Extracts the inner `ChannelExpr`, if at the Channel level.
    #[must_use]
    pub fn as_channel(&self) -> Option<&ChannelExpr> {
        match self {
            Op::Channel(ch) => Some(ch),
            _ => None,
        }
    }

    /// Extracts the inner `PauliString`, if at the Pauli level.
    #[must_use]
    pub fn as_pauli(&self) -> Option<&PauliString> {
        match self {
            Op::Pauli(ps) => Some(ps),
            _ => None,
        }
    }

    /// Extracts the inner `CliffordRep`, if at the Clifford level.
    #[must_use]
    pub fn as_clifford(&self) -> Option<&CliffordRep> {
        match self {
            Op::Clifford(cr, _) => Some(cr),
            _ => None,
        }
    }

    /// Extracts the inner `UnitaryRep`, if at the Unitary level.
    #[must_use]
    pub fn as_unitary(&self) -> Option<&UnitaryRep> {
        match self {
            Op::Unitary(ur) => Some(ur),
            _ => None,
        }
    }

    /// Consumes and returns the inner `PauliString`, if at the Pauli level.
    #[must_use]
    pub fn into_pauli(self) -> Option<PauliString> {
        match self {
            Op::Pauli(ps) => Some(ps),
            _ => None,
        }
    }

    /// Consumes and returns the inner `CliffordRep`.
    /// Pauli promotes to Clifford. Returns `None` for Unitary/Gate/Channel (cannot demote).
    #[must_use]
    pub fn into_clifford(self) -> Option<CliffordRep> {
        match self {
            Op::Pauli(ps) => Some(CliffordRep::from(ps)),
            Op::Clifford(cr, _) => Some(cr),
            Op::Unitary(_) | Op::Gate(_) | Op::Channel(_) => None,
        }
    }

    /// Consumes and returns a `UnitaryRep`.
    /// Returns `None` for Gate/Channel (cannot demote).
    #[must_use]
    pub fn into_unitary(self) -> Option<UnitaryRep> {
        match self {
            Op::Pauli(ps) => Some(UnitaryRep::from(ps)),
            Op::Clifford(_, ur) | Op::Unitary(ur) => Some(ur),
            Op::Gate(_) | Op::Channel(_) => None,
        }
    }

    /// Consumes and returns a `GateExpr`. Unitary and lower levels promote to
    /// `GateExpr::Unitary`; Channel cannot demote.
    #[must_use]
    pub fn into_gate(self) -> Option<GateExpr> {
        match self {
            Op::Pauli(ps) => Some(GateExpr::Unitary(UnitaryRep::from(ps))),
            Op::Clifford(_, ur) | Op::Unitary(ur) => Some(GateExpr::Unitary(ur)),
            Op::Gate(gate) => Some(gate),
            Op::Channel(_) => None,
        }
    }

    /// Consumes and returns a `ChannelExpr`. Always succeeds:
    /// unitary and lower levels promote to `ChannelExpr::Unitary`; Gate
    /// promotes to `ChannelExpr::Gate`.
    #[must_use]
    pub fn into_channel(self) -> ChannelExpr {
        match self {
            Op::Pauli(ps) => ChannelExpr::Unitary(UnitaryRep::from(ps)),
            Op::Clifford(_, ur) | Op::Unitary(ur) => ChannelExpr::Unitary(ur),
            Op::Gate(gate) => ChannelExpr::Gate(gate),
            Op::Channel(ch) => ch,
        }
    }

    /// Promotes this `Op` to at least the Clifford level.
    #[must_use]
    pub fn to_clifford_level(self) -> Op {
        match self {
            Op::Pauli(ps) => {
                let ur = UnitaryRep::from(ps.clone());
                cliff(CliffordRep::from(ps), ur)
            }
            other => other,
        }
    }

    /// Promotes this `Op` to at least the Unitary level.
    /// Returns `None` if at Gate or Channel level (cannot demote).
    #[must_use]
    pub fn to_unitary_level(self) -> Option<Op> {
        match self {
            Op::Pauli(ps) => Some(Op::Unitary(UnitaryRep::from(ps))),
            Op::Clifford(_, ur) | Op::Unitary(ur) => Some(Op::Unitary(ur)),
            Op::Gate(_) | Op::Channel(_) => None,
        }
    }

    /// Promotes this `Op` to the Gate level.
    #[must_use]
    pub fn to_gate_level(self) -> Option<Op> {
        self.into_gate().map(Op::Gate)
    }

    /// Promotes this `Op` to the Channel level.
    #[must_use]
    pub fn to_channel_level(self) -> Op {
        Op::Channel(self.into_channel())
    }

    /// Returns the tensor product of two operations.
    ///
    /// This is the fallible form of the `&` operator. Tensor products require
    /// disjoint qubit support; use `*` for sequential composition on the same
    /// qubits.
    ///
    /// # Errors
    ///
    /// Returns [`TensorProductError`] when the two operations touch any of the
    /// same qubits.
    pub fn try_tensor(self, rhs: Op) -> Result<Op, TensorProductError> {
        require_disjoint_support(&self.qubits(), &rhs.qubits())?;
        Ok(self.tensor_unchecked(rhs))
    }

    fn tensor_unchecked(self, rhs: Op) -> Op {
        let max_level = self.level().max(rhs.level());
        match max_level {
            Level::Pauli => {
                let a = self.into_pauli().expect("max_level is Pauli");
                let b = rhs.into_pauli().expect("max_level is Pauli");
                Op::Pauli(&a & &b)
            }
            Level::Clifford => {
                let (cr_a, ur_a) = match self {
                    Op::Pauli(ps) => pauli_to_cliff_pair(ps),
                    Op::Clifford(cr, ur) => (cr, ur),
                    _ => unreachable!(),
                };
                let (cr_b, ur_b) = match rhs {
                    Op::Pauli(ps) => pauli_to_cliff_pair(ps),
                    Op::Clifford(cr, ur) => (cr, ur),
                    _ => unreachable!(),
                };
                cliff(cr_a.compose(&cr_b), ur_a & ur_b)
            }
            Level::Unitary => {
                let a = self.into_unitary().expect("max_level is Unitary");
                let b = rhs.into_unitary().expect("max_level is Unitary");
                Op::Unitary(a & b)
            }
            Level::Gate => {
                let a = self.into_gate().expect("max_level is Gate");
                let b = rhs.into_gate().expect("max_level is Gate");
                Op::Gate(GateExpr::Tensor(vec![a, b]))
            }
            Level::Channel => {
                let a = self.into_channel();
                let b = rhs.into_channel();
                Op::Channel(ChannelExpr::Tensor(vec![a, b]))
            }
        }
    }

    /// Returns the adjoint (dagger) of this expression.
    ///
    /// # Panics
    /// Panics if called on a Gate-level or Channel-level `Op` (not generally invertible).
    #[must_use]
    pub fn dg(&self) -> Op {
        match self {
            Op::Pauli(ps) => Op::Pauli(ps.clone()),
            Op::Clifford(cr, ur) => cliff(cr.inverse(), ur.dg()),
            Op::Unitary(ur) => Op::Unitary(ur.dg()),
            Op::Gate(_) => panic!("dg() is not defined for Gate-level operations"),
            Op::Channel(_) => panic!("dg() is not defined for Channel-level operations"),
        }
    }

    /// Returns the adjoint if this is a unitary-level operation, `None` for gates/channels.
    #[must_use]
    pub fn try_dg(&self) -> Option<Op> {
        match self {
            Op::Pauli(ps) => Some(Op::Pauli(ps.clone())),
            Op::Clifford(cr, ur) => Some(cliff(cr.inverse(), ur.dg())),
            Op::Unitary(ur) => Some(Op::Unitary(ur.dg())),
            Op::Gate(_) | Op::Channel(_) => None,
        }
    }

    /// Returns the set of qubit indices this expression acts on.
    #[must_use]
    pub fn qubits(&self) -> Vec<usize> {
        match self {
            Op::Pauli(ps) => ps.qubits(),
            Op::Clifford(cr, ur) => {
                let mut qs = cr.support_qubits();
                qs.extend(ur.qubits());
                qs.sort_unstable();
                qs.dedup();
                qs
            }
            Op::Unitary(ur) => ur.qubits(),
            Op::Gate(gate) => gate.qubits(),
            Op::Channel(ch) => ch.qubits(),
        }
    }

    /// Returns the number of qubits this expression spans.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.qubits().into_iter().max().map_or(0, |q| q + 1)
    }
}

// --- GateExpr methods ---

impl GateExpr {
    /// Returns the set of qubit indices this gate expression acts on.
    #[must_use]
    pub fn qubits(&self) -> Vec<usize> {
        let mut qs = Vec::new();
        self.collect_qubits(&mut qs);
        qs.sort_unstable();
        qs.dedup();
        qs
    }

    fn collect_qubits(&self, out: &mut Vec<usize>) {
        match self {
            GateExpr::Prep { qubit, .. }
            | GateExpr::Measure { qubit, .. }
            | GateExpr::Reset { qubit, .. } => {
                out.push(*qubit);
            }
            GateExpr::Unitary(ur) => {
                out.extend(ur.qubits());
            }
            GateExpr::Tensor(parts) | GateExpr::Compose(parts) => {
                for part in parts {
                    part.collect_qubits(out);
                }
            }
        }
    }
}

impl fmt::Display for GateExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GateExpr::Unitary(ur) => write!(f, "{ur:?}"),
            GateExpr::Prep { basis, qubit } => write!(f, "P{basis:?}({qubit})"),
            GateExpr::Measure { basis, qubit } => write!(f, "M{basis:?}({qubit})"),
            GateExpr::Reset {
                basis: Basis::Z,
                qubit,
            } => write!(f, "Reset({qubit})"),
            GateExpr::Reset { basis, qubit } => write!(f, "Reset{basis:?}({qubit})"),
            GateExpr::Tensor(parts) => {
                for (i, part) in parts.iter().enumerate() {
                    if i > 0 {
                        write!(f, " & ")?;
                    }
                    write!(f, "{part}")?;
                }
                Ok(())
            }
            GateExpr::Compose(parts) => {
                for (i, part) in parts.iter().enumerate() {
                    if i > 0 {
                        write!(f, " * ")?;
                    }
                    write!(f, "{part}")?;
                }
                Ok(())
            }
        }
    }
}

// --- ChannelExpr methods ---

impl ChannelExpr {
    /// Returns the set of qubit indices this channel expression acts on.
    #[must_use]
    pub fn qubits(&self) -> Vec<usize> {
        let mut qs = Vec::new();
        self.collect_qubits(&mut qs);
        qs.sort_unstable();
        qs.dedup();
        qs
    }

    fn collect_qubits(&self, out: &mut Vec<usize>) {
        match self {
            ChannelExpr::AmplitudeDamping { qubit, .. }
            | ChannelExpr::PhaseDamping { qubit, .. }
            | ChannelExpr::Erasure { qubit, .. }
            | ChannelExpr::Leakage { qubit, .. } => {
                out.push(*qubit);
            }
            ChannelExpr::Unitary(ur) => {
                out.extend(ur.qubits());
            }
            ChannelExpr::Gate(gate) => {
                out.extend(gate.qubits());
            }
            ChannelExpr::MixedUnitary(ops) => {
                for (_, ur) in ops {
                    out.extend(ur.qubits());
                }
            }
            ChannelExpr::Tensor(parts) | ChannelExpr::Compose(parts) => {
                for part in parts {
                    part.collect_qubits(out);
                }
            }
        }
    }
}

impl fmt::Display for ChannelExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChannelExpr::Unitary(ur) => write!(f, "{ur:?}"),
            ChannelExpr::Gate(gate) => write!(f, "{gate}"),
            ChannelExpr::MixedUnitary(ops) => {
                write!(f, "MixedUnitary[")?;
                for (i, (p, ur)) in ops.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{p}*{ur:?}")?;
                }
                write!(f, "]")
            }
            ChannelExpr::AmplitudeDamping { gamma, qubit } => {
                write!(f, "AmplitudeDamping({gamma}, {qubit})")
            }
            ChannelExpr::PhaseDamping { lambda, qubit } => {
                write!(f, "PhaseDamping({lambda}, {qubit})")
            }
            ChannelExpr::Erasure { prob, qubit } => {
                write!(f, "Erasure({prob}, {qubit})")
            }
            ChannelExpr::Leakage { rate, qubit } => {
                write!(f, "Leakage({rate}, {qubit})")
            }
            ChannelExpr::Tensor(parts) => {
                for (i, part) in parts.iter().enumerate() {
                    if i > 0 {
                        write!(f, " & ")?;
                    }
                    write!(f, "{part}")?;
                }
                Ok(())
            }
            ChannelExpr::Compose(parts) => {
                for (i, part) in parts.iter().enumerate() {
                    if i > 0 {
                        write!(f, " * ")?;
                    }
                    write!(f, "{part}")?;
                }
                Ok(())
            }
        }
    }
}

// --- Promotion helpers ---

/// Promotes a `PauliString` to a `(CliffordRep, UnitaryRep)` pair.
fn pauli_to_cliff_pair(ps: PauliString) -> (CliffordRep, UnitaryRep) {
    let ur = UnitaryRep::from(ps.clone());
    (CliffordRep::from(ps), ur)
}

// --- Tensor product: & operator ---

impl BitAnd for Op {
    type Output = Op;

    fn bitand(self, rhs: Op) -> Op {
        self.try_tensor(rhs).unwrap_or_else(|err| panic!("{err}"))
    }
}

// --- Composition: * operator ---

impl Mul for Op {
    type Output = Op;

    fn mul(self, rhs: Op) -> Op {
        let max_level = self.level().max(rhs.level());
        match max_level {
            Level::Pauli => {
                let a = self.into_pauli().expect("max_level is Pauli");
                let b = rhs.into_pauli().expect("max_level is Pauli");
                Op::Pauli(a * &b)
            }
            Level::Clifford => {
                let (cr_a, ur_a) = match self {
                    Op::Pauli(ps) => pauli_to_cliff_pair(ps),
                    Op::Clifford(cr, ur) => (cr, ur),
                    _ => unreachable!(),
                };
                let (cr_b, ur_b) = match rhs {
                    Op::Pauli(ps) => pauli_to_cliff_pair(ps),
                    Op::Clifford(cr, ur) => (cr, ur),
                    _ => unreachable!(),
                };
                cliff(cr_a.compose(&cr_b), ur_a * ur_b)
            }
            Level::Unitary => {
                let a = self.into_unitary().expect("max_level is Unitary");
                let b = rhs.into_unitary().expect("max_level is Unitary");
                Op::Unitary(a * b)
            }
            Level::Gate => {
                let a = self.into_gate().expect("max_level is Gate");
                let b = rhs.into_gate().expect("max_level is Gate");
                Op::Gate(GateExpr::Compose(vec![a, b]))
            }
            Level::Channel => {
                let a = self.into_channel();
                let b = rhs.into_channel();
                Op::Channel(ChannelExpr::Compose(vec![a, b]))
            }
        }
    }
}

// --- Reference overloads for & (tensor) and * (compose) ---

impl BitAnd<&Op> for Op {
    type Output = Op;
    fn bitand(self, rhs: &Op) -> Op {
        self & rhs.clone()
    }
}

impl BitAnd<Op> for &Op {
    type Output = Op;
    fn bitand(self, rhs: Op) -> Op {
        self.clone() & rhs
    }
}

impl BitAnd<&Op> for &Op {
    type Output = Op;
    fn bitand(self, rhs: &Op) -> Op {
        self.clone() & rhs.clone()
    }
}

impl Mul<&Op> for Op {
    type Output = Op;
    fn mul(self, rhs: &Op) -> Op {
        self * rhs.clone()
    }
}

impl Mul<Op> for &Op {
    type Output = Op;
    fn mul(self, rhs: Op) -> Op {
        self.clone() * rhs
    }
}

impl Mul<&Op> for &Op {
    type Output = Op;
    fn mul(self, rhs: &Op) -> Op {
        self.clone() * rhs.clone()
    }
}

// --- Negation and phase multiplication ---

impl Neg for Op {
    type Output = Op;

    fn neg(self) -> Op {
        match self {
            Op::Pauli(ps) => Op::Pauli(-ps),
            Op::Clifford(cr, ur) => cliff(cr, -ur),
            Op::Unitary(ur) => Op::Unitary(-ur),
            Op::Gate(_) => panic!("negation is not defined for Gate-level operations"),
            Op::Channel(_) => panic!("negation is not defined for Channel-level operations"),
        }
    }
}

impl Neg for &Op {
    type Output = Op;

    fn neg(self) -> Op {
        -self.clone()
    }
}

impl Mul<Op> for ImaginaryUnit {
    type Output = Op;

    fn mul(self, rhs: Op) -> Op {
        match rhs {
            Op::Pauli(ps) => Op::Pauli(self * ps),
            Op::Clifford(cr, ur) => cliff(cr, self * ur),
            Op::Unitary(ur) => Op::Unitary(self * ur),
            Op::Gate(_) => {
                panic!("phase multiplication is not defined for Gate-level operations")
            }
            Op::Channel(_) => {
                panic!("phase multiplication is not defined for Channel-level operations")
            }
        }
    }
}

impl Mul<&Op> for ImaginaryUnit {
    type Output = Op;

    fn mul(self, rhs: &Op) -> Op {
        self * rhs.clone()
    }
}

impl Mul<Op> for NegImaginaryUnit {
    type Output = Op;

    fn mul(self, rhs: Op) -> Op {
        match rhs {
            Op::Pauli(ps) => Op::Pauli(self * ps),
            Op::Clifford(cr, ur) => cliff(cr, self * ur),
            Op::Unitary(ur) => Op::Unitary(self * ur),
            Op::Gate(_) => {
                panic!("phase multiplication is not defined for Gate-level operations")
            }
            Op::Channel(_) => {
                panic!("phase multiplication is not defined for Channel-level operations")
            }
        }
    }
}

impl Mul<&Op> for NegImaginaryUnit {
    type Output = Op;

    fn mul(self, rhs: &Op) -> Op {
        self * rhs.clone()
    }
}

/// Generic phase multiplication: `phase(angle) * op` promotes to Unitary.
///
/// Applies the global phase e^{i*angle} to the operation.
///
/// # Panics
/// Panics if applied to a Gate-level or Channel-level operation.
impl Mul<Op> for PhaseValue {
    type Output = Op;

    fn mul(self, rhs: Op) -> Op {
        match rhs {
            Op::Gate(_) => {
                panic!("phase multiplication is not defined for Gate-level operations")
            }
            Op::Channel(_) => {
                panic!("phase multiplication is not defined for Channel-level operations")
            }
            other => {
                let ur = other
                    .into_unitary()
                    .expect("non-Gate/non-Channel Op is convertible to Unitary");
                Op::Unitary(self * ur)
            }
        }
    }
}

impl Mul<&Op> for PhaseValue {
    type Output = Op;

    fn mul(self, rhs: &Op) -> Op {
        self * rhs.clone()
    }
}

/// Scalar multiplication: `1 * op` is identity, `-1 * op` is negation.
///
/// # Panics
/// Panics if the scalar is not `1` or `-1`.
impl Mul<Op> for i32 {
    type Output = Op;

    fn mul(self, rhs: Op) -> Op {
        match self {
            1 => rhs,
            -1 => -rhs,
            _ => panic!("only 1 and -1 are valid scalar multipliers for Op"),
        }
    }
}

impl Mul<&Op> for i32 {
    type Output = Op;

    fn mul(self, rhs: &Op) -> Op {
        self * rhs.clone()
    }
}

// --- From conversions ---

impl From<PauliString> for Op {
    fn from(ps: PauliString) -> Op {
        Op::Pauli(ps)
    }
}

impl From<UnitaryRep> for Op {
    fn from(ur: UnitaryRep) -> Op {
        Op::Unitary(ur)
    }
}

impl From<GateExpr> for Op {
    fn from(gate: GateExpr) -> Op {
        Op::Gate(gate)
    }
}

impl From<ChannelExpr> for Op {
    fn from(channel: ChannelExpr) -> Op {
        Op::Channel(channel)
    }
}

// --- Display ---

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Pauli(ps) => write!(f, "{ps}"),
            Op::Clifford(cr, _) => write!(f, "{cr}"),
            Op::Unitary(ur) => write!(f, "{ur:?}"),
            Op::Gate(gate) => write!(f, "{gate}"),
            Op::Channel(ch) => write!(f, "{ch}"),
        }
    }
}

// --- Gate constructors — Pauli level ---

/// Identity operator.
#[allow(non_snake_case)]
#[must_use]
pub fn I(qubit: impl Into<QubitId>) -> Op {
    let _q: QubitId = qubit.into();
    Op::Pauli(PauliString::identity())
}

/// Pauli X gate.
#[allow(non_snake_case)]
#[must_use]
pub fn X(qubit: impl Into<QubitId>) -> Op {
    Op::Pauli(PauliString::x(qubit.into().0))
}

/// Pauli Y gate.
#[allow(non_snake_case)]
#[must_use]
pub fn Y(qubit: impl Into<QubitId>) -> Op {
    Op::Pauli(PauliString::y(qubit.into().0))
}

/// Pauli Z gate.
#[allow(non_snake_case)]
#[must_use]
pub fn Z(qubit: impl Into<QubitId>) -> Op {
    Op::Pauli(PauliString::z(qubit.into().0))
}

// --- Gate constructors — Clifford level (1-qubit) ---

/// Hadamard gate.
#[allow(non_snake_case)]
#[must_use]
pub fn H(qubit: impl Into<QubitId>) -> Op {
    let q = qubit.into();
    cliff(CliffordRep::h(q.0), crate::unitary_rep::H(q))
}

/// sqrt(X) gate.
#[allow(non_snake_case)]
#[must_use]
pub fn SX(qubit: impl Into<QubitId>) -> Op {
    let q = qubit.into();
    cliff(CliffordRep::sx(q.0), crate::unitary_rep::SX(q))
}

/// sqrt(X)-dagger gate.
#[allow(non_snake_case)]
#[must_use]
pub fn SXdg(qubit: impl Into<QubitId>) -> Op {
    let q = qubit.into();
    cliff(CliffordRep::sxdg(q.0), crate::unitary_rep::SX(q).dg())
}

/// sqrt(Y) gate.
#[allow(non_snake_case)]
#[must_use]
pub fn SY(qubit: impl Into<QubitId>) -> Op {
    let q = qubit.into();
    cliff(CliffordRep::sy(q.0), crate::unitary_rep::SY(q))
}

/// sqrt(Y)-dagger gate.
#[allow(non_snake_case)]
#[must_use]
pub fn SYdg(qubit: impl Into<QubitId>) -> Op {
    let q = qubit.into();
    cliff(CliffordRep::sydg(q.0), crate::unitary_rep::SY(q).dg())
}

/// sqrt(Z) gate (S gate).
#[allow(non_snake_case)]
#[must_use]
pub fn SZ(qubit: impl Into<QubitId>) -> Op {
    let q = qubit.into();
    cliff(CliffordRep::sz(q.0), crate::unitary_rep::SZ(q))
}

/// sqrt(Z)-dagger gate (S-dagger gate).
#[allow(non_snake_case)]
#[must_use]
pub fn SZdg(qubit: impl Into<QubitId>) -> Op {
    let q = qubit.into();
    cliff(CliffordRep::szdg(q.0), crate::unitary_rep::SZ(q).dg())
}

/// H2 gate (SY * Z decomposition).
#[allow(non_snake_case)]
#[must_use]
pub fn H2(qubit: impl Into<QubitId>) -> Op {
    let q = qubit.into();
    cliff(
        CliffordRep::h2(q.0),
        crate::unitary_rep::Z(q) * crate::unitary_rep::SY(q),
    )
}

/// H3 gate (SZ * Y decomposition).
#[allow(non_snake_case)]
#[must_use]
pub fn H3(qubit: impl Into<QubitId>) -> Op {
    let q = qubit.into();
    cliff(
        CliffordRep::h3(q.0),
        crate::unitary_rep::Y(q) * crate::unitary_rep::SZ(q),
    )
}

/// H4 gate (SZ * X decomposition).
#[allow(non_snake_case)]
#[must_use]
pub fn H4(qubit: impl Into<QubitId>) -> Op {
    let q = qubit.into();
    cliff(
        CliffordRep::h4(q.0),
        crate::unitary_rep::X(q) * crate::unitary_rep::SZ(q),
    )
}

/// H5 gate (SX * Z decomposition).
#[allow(non_snake_case)]
#[must_use]
pub fn H5(qubit: impl Into<QubitId>) -> Op {
    let q = qubit.into();
    cliff(
        CliffordRep::h5(q.0),
        crate::unitary_rep::Z(q) * crate::unitary_rep::SX(q),
    )
}

/// H6 gate (SX * Y decomposition).
#[allow(non_snake_case)]
#[must_use]
pub fn H6(qubit: impl Into<QubitId>) -> Op {
    let q = qubit.into();
    cliff(
        CliffordRep::h6(q.0),
        crate::unitary_rep::Y(q) * crate::unitary_rep::SX(q),
    )
}

/// Face gate F (SX * SZ decomposition).
#[allow(non_snake_case)]
#[must_use]
pub fn F(qubit: impl Into<QubitId>) -> Op {
    let q = qubit.into();
    cliff(
        CliffordRep::f(q.0),
        crate::unitary_rep::SZ(q) * crate::unitary_rep::SX(q),
    )
}

/// Face gate F-dagger (`SZdg` * `SXdg` decomposition).
#[allow(non_snake_case)]
#[must_use]
pub fn Fdg(qubit: impl Into<QubitId>) -> Op {
    let q = qubit.into();
    cliff(
        CliffordRep::fdg(q.0),
        crate::unitary_rep::SX(q).dg() * crate::unitary_rep::SZ(q).dg(),
    )
}

/// F2 gate (`SXdg` * SY decomposition).
#[allow(non_snake_case)]
#[must_use]
pub fn F2(qubit: impl Into<QubitId>) -> Op {
    let q = qubit.into();
    cliff(
        CliffordRep::f2(q.0),
        crate::unitary_rep::SY(q) * crate::unitary_rep::SX(q).dg(),
    )
}

/// F2-dagger gate (`SYdg` * SX decomposition).
#[allow(non_snake_case)]
#[must_use]
pub fn F2dg(qubit: impl Into<QubitId>) -> Op {
    let q = qubit.into();
    cliff(
        CliffordRep::f2dg(q.0),
        crate::unitary_rep::SX(q) * crate::unitary_rep::SY(q).dg(),
    )
}

/// F3 gate (`SXdg` * SZ decomposition).
#[allow(non_snake_case)]
#[must_use]
pub fn F3(qubit: impl Into<QubitId>) -> Op {
    let q = qubit.into();
    cliff(
        CliffordRep::f3(q.0),
        crate::unitary_rep::SZ(q) * crate::unitary_rep::SX(q).dg(),
    )
}

/// F3-dagger gate (SX * `SZdg` decomposition).
#[allow(non_snake_case)]
#[must_use]
pub fn F3dg(qubit: impl Into<QubitId>) -> Op {
    let q = qubit.into();
    cliff(
        CliffordRep::f3dg(q.0),
        crate::unitary_rep::SX(q) * crate::unitary_rep::SZ(q).dg(),
    )
}

/// F4 gate (SX * SZ decomposition).
#[allow(non_snake_case)]
#[must_use]
pub fn F4(qubit: impl Into<QubitId>) -> Op {
    let q = qubit.into();
    cliff(
        CliffordRep::f4(q.0),
        crate::unitary_rep::SX(q) * crate::unitary_rep::SZ(q),
    )
}

/// F4-dagger gate (`SZdg` * `SXdg` decomposition).
#[allow(non_snake_case)]
#[must_use]
pub fn F4dg(qubit: impl Into<QubitId>) -> Op {
    let q = qubit.into();
    cliff(
        CliffordRep::f4dg(q.0),
        crate::unitary_rep::SZ(q).dg() * crate::unitary_rep::SX(q).dg(),
    )
}

// --- Gate constructors — Clifford level (2-qubit) ---

/// CNOT gate (controlled-X).
#[allow(non_snake_case)]
#[must_use]
pub fn CX(control: impl Into<QubitId>, target: impl Into<QubitId>) -> Op {
    let c = control.into();
    let t = target.into();
    cliff(CliffordRep::cx(c.0, t.0), crate::unitary_rep::CX(c, t))
}

/// Controlled-Y gate.
#[allow(non_snake_case)]
#[must_use]
pub fn CY(control: impl Into<QubitId>, target: impl Into<QubitId>) -> Op {
    let c = control.into();
    let t = target.into();
    cliff(CliffordRep::cy(c.0, t.0), crate::unitary_rep::CY(c, t))
}

/// Controlled-Z gate.
#[allow(non_snake_case)]
#[must_use]
pub fn CZ(q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> Op {
    let a = q0.into();
    let b = q1.into();
    cliff(CliffordRep::cz(a.0, b.0), crate::unitary_rep::CZ(a, b))
}

/// SWAP gate.
#[allow(non_snake_case)]
#[must_use]
pub fn SWAP(q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> Op {
    let a = q0.into();
    let b = q1.into();
    cliff(CliffordRep::swap(a.0, b.0), crate::unitary_rep::SWAP(a, b))
}

/// sqrt(XX) gate.
#[allow(non_snake_case)]
#[must_use]
pub fn SXX(q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> Op {
    let a = q0.into();
    let b = q1.into();
    cliff(
        CliffordRep::sxx(a.0, b.0),
        crate::unitary_rep::RXX(Angle64::QUARTER_TURN, a, b),
    )
}

/// sqrt(XX)-dagger gate.
#[allow(non_snake_case)]
#[must_use]
pub fn SXXdg(q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> Op {
    let a = q0.into();
    let b = q1.into();
    cliff(
        CliffordRep::sxxdg(a.0, b.0),
        crate::unitary_rep::RXX(Angle64::THREE_QUARTERS_TURN, a, b),
    )
}

/// sqrt(YY) gate.
#[allow(non_snake_case)]
#[must_use]
pub fn SYY(q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> Op {
    let a = q0.into();
    let b = q1.into();
    cliff(
        CliffordRep::syy(a.0, b.0),
        crate::unitary_rep::RYY(Angle64::QUARTER_TURN, a, b),
    )
}

/// sqrt(YY)-dagger gate.
#[allow(non_snake_case)]
#[must_use]
pub fn SYYdg(q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> Op {
    let a = q0.into();
    let b = q1.into();
    cliff(
        CliffordRep::syydg(a.0, b.0),
        crate::unitary_rep::RYY(Angle64::THREE_QUARTERS_TURN, a, b),
    )
}

/// sqrt(ZZ) gate.
#[allow(non_snake_case)]
#[must_use]
pub fn SZZ(q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> Op {
    let a = q0.into();
    let b = q1.into();
    cliff(CliffordRep::szz(a.0, b.0), crate::unitary_rep::SZZ(a, b))
}

/// sqrt(ZZ)-dagger gate.
#[allow(non_snake_case)]
#[must_use]
pub fn SZZdg(q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> Op {
    let a = q0.into();
    let b = q1.into();
    cliff(
        CliffordRep::szzdg(a.0, b.0),
        crate::unitary_rep::SZZ(a, b).dg(),
    )
}

/// iSWAP gate.
#[allow(non_snake_case)]
#[must_use]
pub fn ISWAP(q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> Op {
    let a = q0.into();
    let b = q1.into();
    // iSWAP = exp(+i*pi/4*(XX+YY)) = RXX(-pi/2) * RYY(-pi/2)
    cliff(
        CliffordRep::iswap(a.0, b.0),
        crate::unitary_rep::RXX(Angle64::THREE_QUARTERS_TURN, a, b)
            * crate::unitary_rep::RYY(Angle64::THREE_QUARTERS_TURN, a, b),
    )
}

/// iSWAP-dagger gate.
#[allow(non_snake_case)]
#[must_use]
pub fn ISWAPdg(q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> Op {
    let a = q0.into();
    let b = q1.into();
    cliff(
        CliffordRep::iswapdg(a.0, b.0),
        (crate::unitary_rep::RXX(Angle64::THREE_QUARTERS_TURN, a, b)
            * crate::unitary_rep::RYY(Angle64::THREE_QUARTERS_TURN, a, b))
        .dg(),
    )
}

/// G (Givens) gate.
#[allow(non_snake_case)]
#[must_use]
pub fn G(q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> Op {
    let a = q0.into();
    let b = q1.into();
    // G = CZ * H(q0) * H(q1) * CZ
    cliff(
        CliffordRep::g(a.0, b.0),
        crate::unitary_rep::CZ(a, b)
            * crate::unitary_rep::H(a)
            * crate::unitary_rep::H(b)
            * crate::unitary_rep::CZ(a, b),
    )
}

/// G (Givens)-dagger gate.
#[allow(non_snake_case)]
#[must_use]
pub fn Gdg(q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> Op {
    let a = q0.into();
    let b = q1.into();
    cliff(
        CliffordRep::gdg(a.0, b.0),
        (crate::unitary_rep::CZ(a, b)
            * crate::unitary_rep::H(a)
            * crate::unitary_rep::H(b)
            * crate::unitary_rep::CZ(a, b))
        .dg(),
    )
}

// --- Gate constructors — Unitary level ---

/// T gate (pi/8 gate).
#[allow(non_snake_case)]
#[must_use]
pub fn T(qubit: impl Into<QubitId>) -> Op {
    Op::Unitary(crate::unitary_rep::T(qubit))
}

/// T-dagger gate.
#[allow(non_snake_case)]
#[must_use]
pub fn Tdg(qubit: impl Into<QubitId>) -> Op {
    Op::Unitary(crate::unitary_rep::T(qubit).dg())
}

/// Rotation around X axis: exp(-i theta/2 X).
#[allow(non_snake_case)]
#[must_use]
pub fn RX(angle: Angle64, qubit: impl Into<QubitId>) -> Op {
    Op::Unitary(crate::unitary_rep::RX(angle, qubit))
}

/// Rotation around Y axis: exp(-i theta/2 Y).
#[allow(non_snake_case)]
#[must_use]
pub fn RY(angle: Angle64, qubit: impl Into<QubitId>) -> Op {
    Op::Unitary(crate::unitary_rep::RY(angle, qubit))
}

/// Rotation around Z axis: exp(-i theta/2 Z).
#[allow(non_snake_case)]
#[must_use]
pub fn RZ(angle: Angle64, qubit: impl Into<QubitId>) -> Op {
    Op::Unitary(crate::unitary_rep::RZ(angle, qubit))
}

/// Two-qubit XX rotation: exp(-i theta/2 XX).
#[allow(non_snake_case)]
#[must_use]
pub fn RXX(angle: Angle64, q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> Op {
    Op::Unitary(crate::unitary_rep::RXX(angle, q0, q1))
}

/// Two-qubit YY rotation: exp(-i theta/2 YY).
#[allow(non_snake_case)]
#[must_use]
pub fn RYY(angle: Angle64, q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> Op {
    Op::Unitary(crate::unitary_rep::RYY(angle, q0, q1))
}

/// Two-qubit ZZ rotation: exp(-i theta/2 ZZ).
#[allow(non_snake_case)]
#[must_use]
pub fn RZZ(angle: Angle64, q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> Op {
    Op::Unitary(crate::unitary_rep::RZZ(angle, q0, q1))
}

/// Toffoli gate (CCX, 3 qubits).
#[allow(non_snake_case)]
#[must_use]
pub fn CCX(c0: impl Into<QubitId>, c1: impl Into<QubitId>, target: impl Into<QubitId>) -> Op {
    Op::Unitary(crate::unitary_rep::CCX(c0, c1, target))
}

// --- Gate constructors — Gate level ---

/// Prepare qubit in the |0> state (Z-basis preparation).
#[allow(non_snake_case)]
#[must_use]
pub fn PZ(qubit: impl Into<QubitId>) -> Op {
    Op::Gate(GateExpr::Prep {
        basis: Basis::Z,
        qubit: qubit.into().0,
    })
}

/// Prepare qubit in the |+> state (X-basis preparation).
#[allow(non_snake_case)]
#[must_use]
pub fn PX(qubit: impl Into<QubitId>) -> Op {
    Op::Gate(GateExpr::Prep {
        basis: Basis::X,
        qubit: qubit.into().0,
    })
}

/// Prepare qubit in the Y-basis +1 eigenstate.
#[allow(non_snake_case)]
#[must_use]
pub fn PY(qubit: impl Into<QubitId>) -> Op {
    Op::Gate(GateExpr::Prep {
        basis: Basis::Y,
        qubit: qubit.into().0,
    })
}

/// Measure qubit in the Z basis (computational basis measurement).
#[allow(non_snake_case)]
#[must_use]
pub fn MZ(qubit: impl Into<QubitId>) -> Op {
    Op::Gate(GateExpr::Measure {
        basis: Basis::Z,
        qubit: qubit.into().0,
    })
}

/// Measure qubit in the X basis.
#[allow(non_snake_case)]
#[must_use]
pub fn MX(qubit: impl Into<QubitId>) -> Op {
    Op::Gate(GateExpr::Measure {
        basis: Basis::X,
        qubit: qubit.into().0,
    })
}

/// Measure qubit in the Y basis.
#[allow(non_snake_case)]
#[must_use]
pub fn MY(qubit: impl Into<QubitId>) -> Op {
    Op::Gate(GateExpr::Measure {
        basis: Basis::Y,
        qubit: qubit.into().0,
    })
}

// --- Noise channel constructors ---

/// Single-qubit depolarizing channel: ρ → (1−p)ρ + (p/3)(XρX + `YρY` + `ZρZ`).
///
/// # Panics
/// Panics if `p` is not in [0, 1].
#[allow(non_snake_case)]
#[must_use]
pub fn Depolarizing(p: f64, qubit: impl Into<QubitId>) -> Op {
    assert!((0.0..=1.0).contains(&p), "probability p must be in [0, 1]");
    let q = qubit.into();
    let p3 = p / 3.0;
    Op::Channel(ChannelExpr::MixedUnitary(vec![
        (1.0 - p, crate::unitary_rep::I(q)),
        (p3, crate::unitary_rep::X(q)),
        (p3, crate::unitary_rep::Y(q)),
        (p3, crate::unitary_rep::Z(q)),
    ]))
}

/// Dephasing (phase-flip) channel: ρ → (1−p)ρ + p `ZρZ`.
///
/// # Panics
/// Panics if `p` is not in [0, 1].
#[allow(non_snake_case)]
#[must_use]
pub fn Dephasing(p: f64, qubit: impl Into<QubitId>) -> Op {
    assert!((0.0..=1.0).contains(&p), "probability p must be in [0, 1]");
    let q = qubit.into();
    Op::Channel(ChannelExpr::MixedUnitary(vec![
        (1.0 - p, crate::unitary_rep::I(q)),
        (p, crate::unitary_rep::Z(q)),
    ]))
}

/// Bit-flip channel: ρ → (1−p)ρ + p `XρX`.
///
/// # Panics
/// Panics if `p` is not in [0, 1].
#[allow(non_snake_case)]
#[must_use]
pub fn BitFlip(p: f64, qubit: impl Into<QubitId>) -> Op {
    assert!((0.0..=1.0).contains(&p), "probability p must be in [0, 1]");
    let q = qubit.into();
    Op::Channel(ChannelExpr::MixedUnitary(vec![
        (1.0 - p, crate::unitary_rep::I(q)),
        (p, crate::unitary_rep::X(q)),
    ]))
}

/// General single-qubit Pauli channel: ρ → (1−px−py−pz)ρ + px `XρX` + py `YρY` + pz `ZρZ`.
///
/// # Panics
/// Panics if any probability is negative or if `px + py + pz > 1`.
#[allow(non_snake_case)]
#[must_use]
pub fn PauliChannel(px: f64, py: f64, pz: f64, qubit: impl Into<QubitId>) -> Op {
    assert!(
        px >= 0.0 && py >= 0.0 && pz >= 0.0,
        "probabilities must be non-negative"
    );
    let pi = 1.0 - px - py - pz;
    assert!(pi >= -1e-15, "probabilities must sum to at most 1");
    let pi = pi.max(0.0);
    let q = qubit.into();
    Op::Channel(ChannelExpr::MixedUnitary(vec![
        (pi, crate::unitary_rep::I(q)),
        (px, crate::unitary_rep::X(q)),
        (py, crate::unitary_rep::Y(q)),
        (pz, crate::unitary_rep::Z(q)),
    ]))
}

/// Amplitude damping channel with decay parameter γ ∈ [0, 1].
///
/// Models T₁ relaxation: qubit decays from |1⟩ to |0⟩ with probability γ.
/// Kraus operators: K₀ = |0⟩⟨0| + √(1−γ)|1⟩⟨1|, K₁ = √γ |0⟩⟨1|.
///
/// # Panics
/// Panics if `gamma` is not in [0, 1].
#[allow(non_snake_case)]
#[must_use]
pub fn AmplitudeDamping(gamma: f64, qubit: impl Into<QubitId>) -> Op {
    assert!((0.0..=1.0).contains(&gamma), "gamma must be in [0, 1]");
    Op::Channel(ChannelExpr::AmplitudeDamping {
        gamma,
        qubit: qubit.into().0,
    })
}

/// Bit-phase-flip channel: ρ → (1−p)ρ + p `YρY`.
///
/// # Panics
/// Panics if `p` is not in [0, 1].
#[allow(non_snake_case)]
#[must_use]
pub fn BitPhaseFlip(p: f64, qubit: impl Into<QubitId>) -> Op {
    assert!((0.0..=1.0).contains(&p), "probability p must be in [0, 1]");
    let q = qubit.into();
    Op::Channel(ChannelExpr::MixedUnitary(vec![
        (1.0 - p, crate::unitary_rep::I(q)),
        (p, crate::unitary_rep::Y(q)),
    ]))
}

/// Two-qubit depolarizing channel.
///
/// ρ → (1−p)ρ + (p/15) Σ_{P ≠ II} P ρ P†
///
/// where the sum runs over the 15 non-identity two-qubit Pauli operators.
///
/// # Panics
/// Panics if `p` is not in [0, 1].
#[allow(non_snake_case)]
#[must_use]
pub fn Depolarizing2(p: f64, q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> Op {
    use crate::unitary_rep;
    assert!((0.0..=1.0).contains(&p), "probability p must be in [0, 1]");
    let a = q0.into();
    let b = q1.into();
    assert_distinct_qubits("Depolarizing2", [a.0, b.0]);
    let p15 = p / 15.0;
    let paulis_1q = [
        unitary_rep::I,
        unitary_rep::X,
        unitary_rep::Y,
        unitary_rep::Z,
    ];
    let mut ops = Vec::with_capacity(16);
    for (idx_a, pi) in paulis_1q.iter().enumerate() {
        for (idx_b, pj) in paulis_1q.iter().enumerate() {
            let prob = if idx_a == 0 && idx_b == 0 {
                1.0 - p
            } else {
                p15
            };
            ops.push((prob, pi(a) & pj(b)));
        }
    }
    Op::Channel(ChannelExpr::MixedUnitary(ops))
}

/// Phase damping channel with parameter λ ∈ [0, 1].
///
/// Models pure dephasing (T₂ without T₁).
/// Kraus operators: K₀ = diag(1, √(1−λ)), K₁ = diag(0, √λ).
///
/// Note: for Pauli-noise approximations, use [`Dephasing`] instead.
///
/// # Panics
/// Panics if `lambda` is not in [0, 1].
#[allow(non_snake_case)]
#[must_use]
pub fn PhaseDamping(lambda: f64, qubit: impl Into<QubitId>) -> Op {
    assert!((0.0..=1.0).contains(&lambda), "lambda must be in [0, 1]");
    Op::Channel(ChannelExpr::PhaseDamping {
        lambda,
        qubit: qubit.into().0,
    })
}

/// Erasure channel with erasure probability p ∈ [0, 1].
///
/// With probability (1−p) the qubit passes through unchanged; with probability p
/// it is replaced by the maximally mixed state and an erasure flag is raised.
/// This is a heralded error — the error location is known to the decoder.
///
/// # Panics
/// Panics if `prob` is not in [0, 1].
#[allow(non_snake_case)]
#[must_use]
pub fn Erasure(prob: f64, qubit: impl Into<QubitId>) -> Op {
    assert!((0.0..=1.0).contains(&prob), "probability must be in [0, 1]");
    Op::Channel(ChannelExpr::Erasure {
        prob,
        qubit: qubit.into().0,
    })
}

/// Reset qubit to |0⟩ regardless of its current state.
///
/// Kraus operators: K₀ = |0⟩⟨0|, K₁ = |0⟩⟨1|.
#[allow(non_snake_case)]
#[must_use]
pub fn Reset(qubit: impl Into<QubitId>) -> Op {
    Op::Gate(GateExpr::Reset {
        basis: Basis::Z,
        qubit: qubit.into().0,
    })
}

/// Leakage channel: qubit transitions from the computational subspace to a
/// leaked state with the given rate.
///
/// Models |1⟩ → |2⟩ transitions common in superconducting and trapped-ion
/// qubits. The simulator is responsible for managing the extended Hilbert space.
///
/// # Panics
/// Panics if `rate` is not in [0, 1].
#[allow(non_snake_case)]
#[must_use]
pub fn Leakage(rate: f64, qubit: impl Into<QubitId>) -> Op {
    assert!((0.0..=1.0).contains(&rate), "rate must be in [0, 1]");
    Op::Channel(ChannelExpr::Leakage {
        rate,
        qubit: qubit.into().0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PauliOperator;
    use crate::pauli::algebra::i;

    // --- Level detection ---

    #[test]
    fn pauli_level() {
        assert!(X(0).is_pauli());
        assert!(Y(0).is_pauli());
        assert!(Z(0).is_pauli());
        assert!(I(0).is_pauli());
    }

    #[test]
    fn clifford_1q_level() {
        assert!(H(0).is_clifford());
        assert!(SX(0).is_clifford());
        assert!(SXdg(0).is_clifford());
        assert!(SY(0).is_clifford());
        assert!(SYdg(0).is_clifford());
        assert!(SZ(0).is_clifford());
        assert!(SZdg(0).is_clifford());
        assert!(H2(0).is_clifford());
        assert!(H3(0).is_clifford());
        assert!(H4(0).is_clifford());
        assert!(H5(0).is_clifford());
        assert!(H6(0).is_clifford());
        assert!(F(0).is_clifford());
        assert!(Fdg(0).is_clifford());
        assert!(F2(0).is_clifford());
        assert!(F2dg(0).is_clifford());
        assert!(F3(0).is_clifford());
        assert!(F3dg(0).is_clifford());
        assert!(F4(0).is_clifford());
        assert!(F4dg(0).is_clifford());
    }

    #[test]
    fn clifford_2q_level() {
        assert!(CX(0, 1).is_clifford());
        assert!(CY(0, 1).is_clifford());
        assert!(CZ(0, 1).is_clifford());
        assert!(SWAP(0, 1).is_clifford());
        assert!(SXX(0, 1).is_clifford());
        assert!(SXXdg(0, 1).is_clifford());
        assert!(SYY(0, 1).is_clifford());
        assert!(SYYdg(0, 1).is_clifford());
        assert!(SZZ(0, 1).is_clifford());
        assert!(SZZdg(0, 1).is_clifford());
        assert!(ISWAP(0, 1).is_clifford());
        assert!(ISWAPdg(0, 1).is_clifford());
        assert!(G(0, 1).is_clifford());
        assert!(Gdg(0, 1).is_clifford());
    }

    #[test]
    fn unitary_level() {
        assert!(T(0).is_unitary());
        assert!(Tdg(0).is_unitary());
        assert!(RX(Angle64::QUARTER_TURN, 0).is_unitary());
        assert!(RY(Angle64::QUARTER_TURN, 0).is_unitary());
        assert!(RZ(Angle64::QUARTER_TURN, 0).is_unitary());
        assert!(RXX(Angle64::QUARTER_TURN, 0, 1).is_unitary());
        assert!(RYY(Angle64::QUARTER_TURN, 0, 1).is_unitary());
        assert!(RZZ(Angle64::QUARTER_TURN, 0, 1).is_unitary());
        assert!(CCX(0, 1, 2).is_unitary());
    }

    // --- Tensor promotion ---

    #[test]
    fn pauli_tensor_stays_pauli() {
        let op = X(0) & Y(3);
        assert!(op.is_pauli());
        let ps = op.as_pauli().unwrap();
        assert_eq!(ps.weight(), 2);
    }

    #[test]
    fn pauli_clifford_tensor_promotes() {
        let op = X(0) & H(3);
        assert!(op.is_clifford());
    }

    #[test]
    fn clifford_clifford_tensor_stays_clifford() {
        let op = H(0) & SZ(3);
        assert!(op.is_clifford());
    }

    #[test]
    fn clifford_tensor_uses_actual_support_not_span() {
        let op = H(0) & SZ(3);
        let cr = op.as_clifford().unwrap();

        assert_eq!(op.qubits(), vec![0, 3]);
        assert_eq!(cr.apply(&PauliString::x(0)), PauliString::z(0));
        assert_eq!(cr.apply(&PauliString::z(0)), PauliString::x(0));
        assert_eq!(cr.apply(&PauliString::x(3)), PauliString::y(3));
        assert_eq!(cr.apply(&PauliString::z(3)), PauliString::z(3));
    }

    #[test]
    fn pauli_unitary_tensor_promotes() {
        let op = X(0) & T(3);
        assert!(op.is_unitary());
    }

    #[test]
    fn clifford_unitary_tensor_promotes() {
        let op = H(0) & T(3);
        assert!(op.is_unitary());
    }

    #[test]
    fn three_way_promotion() {
        let op = X(0) & H(3) & T(5);
        assert!(op.is_unitary());
    }

    #[test]
    fn try_tensor_reports_overlapping_qubits() {
        let err = X(0).try_tensor(Z(0)).unwrap_err();
        assert_eq!(err.overlapping_qubits(), &[0]);
    }

    #[test]
    fn try_tensor_rejects_mixed_level_overlaps() {
        let cases = [
            ("pauli-clifford", X(0), H(0), vec![0]),
            ("pauli-unitary", X(0), T(0), vec![0]),
            ("pauli-gate", X(0), MZ(0), vec![0]),
            ("pauli-channel", X(0), Depolarizing(0.01, 0), vec![0]),
            ("clifford-gate", H(0), MZ(0), vec![0]),
            ("gate-channel", MZ(0), Depolarizing(0.01, 0), vec![0]),
            ("partial-multi-qubit", CX(0, 2), H(2), vec![2]),
        ];

        for (name, lhs, rhs, expected_overlap) in cases {
            let err = lhs.try_tensor(rhs).unwrap_err();
            assert_eq!(
                err.overlapping_qubits(),
                expected_overlap.as_slice(),
                "{name}"
            );
        }
    }

    #[test]
    fn tensor_operator_panics_are_consistent_across_levels() {
        fn assert_overlap_panic(f: impl FnOnce() + std::panic::UnwindSafe, expected: &str) {
            let err = std::panic::catch_unwind(f).expect_err("expected tensor overlap panic");
            let message = if let Some(message) = err.downcast_ref::<String>() {
                message.as_str()
            } else if let Some(message) = err.downcast_ref::<&str>() {
                message
            } else {
                panic!("unexpected non-string panic payload");
            };
            assert!(
                message.contains("tensor product requires disjoint operator support"),
                "{message}"
            );
            assert!(message.contains(expected), "{message}");
        }

        assert_overlap_panic(
            || {
                let _ = X(0) & Z(0);
            },
            "[0]",
        );
        assert_overlap_panic(
            || {
                let _ = H(0) & T(0);
            },
            "[0]",
        );
        assert_overlap_panic(
            || {
                let _ = T(0) & MZ(0);
            },
            "[0]",
        );
        assert_overlap_panic(
            || {
                let _ = MZ(0) & Depolarizing(0.01, 0);
            },
            "[0]",
        );
        assert_overlap_panic(
            || {
                let _ = CX(0, 2) & Depolarizing(0.01, 2);
            },
            "[2]",
        );
    }

    #[test]
    fn try_tensor_accepts_mixed_level_disjoint_support() {
        assert!((X(0).try_tensor(H(1))).unwrap().is_clifford());
        assert!((H(0).try_tensor(T(1))).unwrap().is_unitary());
        assert!(
            (MZ(0).try_tensor(Depolarizing(0.01, 1)))
                .unwrap()
                .is_channel()
        );
    }

    #[test]
    #[should_panic(expected = "tensor product requires disjoint operator support")]
    fn pauli_tensor_rejects_overlapping_qubits() {
        let _ = X(0) & Z(0);
    }

    #[test]
    #[should_panic(expected = "tensor product requires disjoint operator support")]
    fn clifford_tensor_rejects_overlapping_qubits() {
        let _ = H(0) & SZ(0);
    }

    #[test]
    #[should_panic(expected = "tensor product requires disjoint operator support")]
    fn gate_tensor_rejects_overlapping_qubits() {
        let _ = H(0) & MZ(0);
    }

    #[test]
    #[should_panic(expected = "tensor product requires disjoint operator support")]
    fn channel_tensor_rejects_overlapping_qubits() {
        let _ = H(0) & Depolarizing(0.01, 0);
    }

    #[test]
    #[should_panic(expected = "SWAP requires distinct qubits")]
    fn op_two_qubit_gate_rejects_repeated_qubit() {
        let _ = SWAP(0, 0);
    }

    #[test]
    #[should_panic(expected = "Depolarizing2 requires distinct qubits")]
    fn op_two_qubit_channel_rejects_repeated_qubit() {
        let _ = Depolarizing2(0.01, 2, 2);
    }

    // --- Composition promotion ---

    #[test]
    fn pauli_compose_stays_pauli() {
        let op = X(0) * Z(0);
        assert!(op.is_pauli());
    }

    #[test]
    fn clifford_compose_stays_clifford() {
        let op = H(0) * SZ(0);
        assert!(op.is_clifford());
    }

    #[test]
    fn pauli_clifford_compose_promotes() {
        let op = X(0) * H(0);
        assert!(op.is_clifford());
    }

    // --- Extraction ---

    #[test]
    fn into_pauli_some_for_pauli() {
        assert!(X(0).into_pauli().is_some());
    }

    #[test]
    fn into_pauli_none_for_clifford() {
        assert!(H(0).into_pauli().is_none());
    }

    #[test]
    fn into_clifford_promotes_pauli() {
        assert!(X(0).into_clifford().is_some());
    }

    #[test]
    fn into_clifford_none_for_unitary() {
        assert!(T(0).into_clifford().is_none());
    }

    #[test]
    fn into_unitary_succeeds_for_unitary_levels() {
        assert!(X(0).into_unitary().is_some());
        assert!(H(0).into_unitary().is_some());
        assert!(T(0).into_unitary().is_some());
    }

    #[test]
    fn into_unitary_none_for_gate_and_channel() {
        assert!(MZ(0).into_unitary().is_none());
        assert!(Depolarizing(0.01, 0).into_unitary().is_none());
    }

    // --- Level promotion ---

    #[test]
    fn to_clifford_level_promotes_pauli() {
        assert!(X(0).to_clifford_level().is_clifford());
    }

    #[test]
    fn to_clifford_level_preserves_clifford() {
        assert!(H(0).to_clifford_level().is_clifford());
    }

    #[test]
    fn to_unitary_level_promotes() {
        assert!(X(0).to_unitary_level().unwrap().is_unitary());
        assert!(H(0).to_unitary_level().unwrap().is_unitary());
        assert!(T(0).to_unitary_level().unwrap().is_unitary());
    }

    #[test]
    fn to_unitary_level_none_for_gate_and_channel() {
        assert!(MZ(0).to_unitary_level().is_none());
        assert!(Depolarizing(0.01, 0).to_unitary_level().is_none());
    }

    // --- Adjoint ---

    #[test]
    fn dagger_preserves_level() {
        assert!(X(0).dg().is_pauli());
        assert!(H(0).dg().is_clifford());
        assert!(T(0).dg().is_unitary());
    }

    // --- Phase and negation ---

    #[test]
    fn neg_pauli_preserves_level() {
        let op = -X(0);
        assert!(op.is_pauli());
    }

    #[test]
    fn i_times_pauli() {
        let op = (i * X(2)) & Y(5) & Z(3);
        assert!(op.is_pauli());
    }

    #[test]
    fn neg_i_times_pauli() {
        let op = -i * (X(0) & Y(1));
        assert!(op.is_pauli());
    }

    #[test]
    fn neg_clifford() {
        let op = -H(0);
        assert!(op.is_clifford());
    }

    #[test]
    fn i_times_unitary() {
        let op = i * T(0);
        assert!(op.is_unitary());
    }

    #[test]
    fn phase_then_promote() {
        // -i * X(2) & Y(5) & Z(3) is Pauli, then promote to Clifford
        let op = -i * (X(2) & Y(5) & Z(3));
        assert!(op.is_pauli());
        let promoted = op.to_clifford_level();
        assert!(promoted.is_clifford());
    }

    #[test]
    fn ref_neg() {
        let a = X(0);
        let b = -&a;
        assert!(b.is_pauli());
        // original still usable
        assert!(a.is_pauli());
    }

    #[test]
    fn ref_i_mul() {
        let a = X(0);
        let b = i * &a;
        assert!(b.is_pauli());
        assert!(a.is_pauli());
    }

    #[test]
    fn minus_one_times_op() {
        let op = (-1 * X(9)) & Y(4);
        assert!(op.is_pauli());
    }

    #[test]
    fn plus_one_times_op() {
        let op = 1 * X(0);
        assert!(op.is_pauli());
    }

    #[test]
    #[should_panic(expected = "only 1 and -1")]
    fn invalid_scalar_panics() {
        let _ = 2 * X(0);
    }

    #[test]
    fn generic_phase_promotes_to_unitary() {
        // e^{iπ/8} * X(0) — not a quarter-turn phase, must promote
        let op = phase(Angle64::HALF_TURN / 4) * X(0);
        assert!(op.is_unitary());
    }

    #[test]
    fn generic_phase_on_clifford() {
        let op = phase(Angle64::HALF_TURN / 4) * H(0);
        assert!(op.is_unitary());
    }

    #[test]
    fn generic_phase_on_unitary() {
        let op = phase(Angle64::HALF_TURN / 3) * T(0);
        assert!(op.is_unitary());
    }

    #[test]
    fn phases_at_different_points() {
        // -Y(1) contributes phase -1, rest are +1
        let a = X(0) & -Y(1) & Z(2);
        assert!(a.is_pauli());
        let ps_a = a.as_pauli().unwrap();
        assert_eq!(
            ps_a.phase(),
            crate::phase::quarter_phase::QuarterPhase::MinusOne
        );

        // Two negations cancel: (-X) & (-Y) has phase (-1)*(-1) = +1
        let b = -X(0) & -Y(1);
        let ps_b = b.as_pauli().unwrap();
        assert_eq!(
            ps_b.phase(),
            crate::phase::quarter_phase::QuarterPhase::PlusOne
        );

        // i and -1 combine: i * (-1) = -i
        let c = (i * X(0)) & -Y(1);
        let ps_c = c.as_pauli().unwrap();
        assert_eq!(
            ps_c.phase(),
            crate::phase::quarter_phase::QuarterPhase::MinusI
        );

        // -i at one point, -1 at another: (-i)*(-1) = +i
        let d = (-i * X(0)) & -Z(1);
        let ps_d = d.as_pauli().unwrap();
        assert_eq!(
            ps_d.phase(),
            crate::phase::quarter_phase::QuarterPhase::PlusI
        );
    }

    // --- Level ordering ---

    #[test]
    fn level_ordering() {
        assert!(Level::Pauli < Level::Clifford);
        assert!(Level::Clifford < Level::Unitary);
        assert!(Level::Unitary < Level::Gate);
        assert!(Level::Gate < Level::Channel);
    }

    // --- From conversions ---

    #[test]
    fn from_pauli_string() {
        let op: Op = PauliString::x(0).into();
        assert!(op.is_pauli());
    }

    #[test]
    fn from_unitary_rep() {
        let op: Op = crate::unitary_rep::T(0).into();
        assert!(op.is_unitary());
    }

    // --- Clifford dual representation consistency ---

    #[test]
    fn clifford_compose_tableau_is_consistent() {
        // Composing two Cliffords via Op should give a tableau matching
        // direct CliffordRep composition
        let op = H(0) * SZ(0);
        let cr = op.as_clifford().unwrap();
        let expected = CliffordRep::h(0).compose(&CliffordRep::sz(0));
        assert_eq!(cr, &expected);
    }

    #[test]
    fn clifford_tensor_tableau_is_consistent() {
        let op = H(0) & SZ(1);
        let cr = op.as_clifford().unwrap();
        let expected = CliffordRep::h(0).compose(&CliffordRep::sz(1));
        assert_eq!(cr, &expected);
    }

    #[test]
    fn all_1q_clifford_constructors_have_valid_tableau() {
        let gates: Vec<Op> = vec![
            H(0),
            SX(0),
            SXdg(0),
            SY(0),
            SYdg(0),
            SZ(0),
            SZdg(0),
            H2(0),
            H3(0),
            H4(0),
            H5(0),
            H6(0),
            F(0),
            Fdg(0),
            F2(0),
            F2dg(0),
            F3(0),
            F3dg(0),
            F4(0),
            F4dg(0),
        ];
        for gate in &gates {
            let cr = gate.as_clifford().unwrap();
            assert!(cr.is_valid(), "Clifford tableau invalid for gate: {gate}");
        }
    }

    #[test]
    fn all_2q_clifford_constructors_have_valid_tableau() {
        let gates: Vec<Op> = vec![
            CX(0, 1),
            CY(0, 1),
            CZ(0, 1),
            SWAP(0, 1),
            SXX(0, 1),
            SXXdg(0, 1),
            SYY(0, 1),
            SYYdg(0, 1),
            SZZ(0, 1),
            SZZdg(0, 1),
            ISWAP(0, 1),
            ISWAPdg(0, 1),
            G(0, 1),
            Gdg(0, 1),
        ];
        for gate in &gates {
            let cr = gate.as_clifford().unwrap();
            assert!(cr.is_valid(), "Clifford tableau invalid for gate: {gate}");
        }
    }

    // --- Query methods ---

    #[test]
    fn qubits_pauli() {
        let op = X(0) & Z(3);
        let mut qs = op.qubits();
        qs.sort_unstable();
        assert_eq!(qs, vec![0, 3]);
    }

    #[test]
    fn qubits_clifford() {
        let op = CX(1, 3);
        assert_eq!(op.num_qubits(), 4); // spans qubits 0..4
    }

    #[test]
    fn qubits_unitary() {
        let op = T(5);
        assert_eq!(op.qubits(), vec![5]);
        assert_eq!(op.num_qubits(), 6);
    }

    // --- Reference overloads ---

    #[test]
    fn ref_bitand() {
        let a = X(0);
        let b = Y(1);
        let c = &a & &b;
        assert!(c.is_pauli());
        // originals still usable
        assert!(a.is_pauli());
        assert!(b.is_pauli());
    }

    #[test]
    fn ref_mul() {
        let a = H(0);
        let b = SZ(0);
        let c = &a * &b;
        assert!(c.is_clifford());
        // originals still usable
        assert!(a.is_clifford());
        assert!(b.is_clifford());
    }

    // --- Gate level ---

    #[test]
    fn gate_level() {
        assert!(MZ(0).is_gate());
        assert!(MX(0).is_gate());
        assert!(MY(0).is_gate());
        assert!(PZ(0).is_gate());
        assert!(PX(0).is_gate());
        assert!(PY(0).is_gate());
    }

    #[test]
    fn gate_tensor_stays_gate() {
        let op = MZ(0) & MZ(1);
        assert!(op.is_gate());
    }

    #[test]
    fn gate_compose_stays_gate() {
        let op = PZ(0) * MZ(0);
        assert!(op.is_gate());
    }

    #[test]
    fn unitary_gate_tensor_promotes() {
        let op = H(0) & MZ(1);
        assert!(op.is_gate());
    }

    #[test]
    fn non_clifford_unitary_gate_tensor_promotes_to_gate_tensor() {
        let op = T(0) & MZ(1);
        assert!(op.is_gate());
        assert!(matches!(op, Op::Gate(GateExpr::Tensor(parts)) if parts.len() == 2));
    }

    #[test]
    fn pauli_gate_tensor_promotes() {
        let op = X(0) & MZ(1);
        assert!(op.is_gate());
    }

    #[test]
    fn unitary_gate_compose_promotes() {
        let op = H(0) * MZ(0);
        assert!(op.is_gate());
    }

    #[test]
    fn non_clifford_unitary_gate_compose_promotes_to_gate_compose() {
        let op = T(0) * MZ(0);
        assert!(op.is_gate());
        assert!(matches!(op, Op::Gate(GateExpr::Compose(parts)) if parts.len() == 2));
    }

    #[test]
    fn into_channel_always_succeeds() {
        // All levels can promote to ChannelExpr
        let _ = X(0).into_channel();
        let _ = H(0).into_channel();
        let _ = T(0).into_channel();
        let _ = MZ(0).into_channel();
    }

    #[test]
    fn into_gate_promotes_unitaries_and_keeps_gates() {
        assert!(X(0).into_gate().is_some());
        assert!(H(0).into_gate().is_some());
        assert!(T(0).into_gate().is_some());
        assert!(MZ(0).into_gate().is_some());
        assert!(Depolarizing(0.01, 0).into_gate().is_none());
    }

    #[test]
    fn into_clifford_none_for_gate() {
        assert!(MZ(0).into_clifford().is_none());
    }

    #[test]
    fn try_dg_none_for_gate() {
        assert!(MZ(0).try_dg().is_none());
    }

    #[test]
    fn try_dg_some_for_unitary_levels() {
        assert!(X(0).try_dg().is_some());
        assert!(H(0).try_dg().is_some());
        assert!(T(0).try_dg().is_some());
    }

    #[test]
    #[should_panic(expected = "not defined for Gate")]
    fn dg_panics_for_gate() {
        let _ = MZ(0).dg();
    }

    #[test]
    fn gate_qubits() {
        let op = MZ(3);
        assert_eq!(op.qubits(), vec![3]);
        assert_eq!(op.num_qubits(), 4);
    }

    #[test]
    fn gate_tensor_qubits() {
        let op = PZ(0) & MZ(2);
        let mut qs = op.qubits();
        qs.sort_unstable();
        assert_eq!(qs, vec![0, 2]);
    }

    #[test]
    fn gate_channel_tensor_promotes_to_channel() {
        let op = MZ(0) & Depolarizing(0.01, 1);
        assert!(op.is_channel());
    }

    #[test]
    fn to_channel_level_promotes() {
        assert!(X(0).to_channel_level().is_channel());
        assert!(H(0).to_channel_level().is_channel());
        assert!(T(0).to_channel_level().is_channel());
        assert!(MZ(0).to_channel_level().is_channel());
    }

    // --- Noise channels ---

    #[test]
    fn depolarizing_is_channel() {
        let op = Depolarizing(0.1, 0);
        assert!(op.is_channel());
    }

    #[test]
    fn depolarizing_probabilities_sum_to_one() {
        let op = Depolarizing(0.3, 0);
        if let Op::Channel(ChannelExpr::MixedUnitary(ops)) = op {
            let total: f64 = ops.iter().map(|(p, _)| p).sum();
            assert!((total - 1.0).abs() < 1e-15);
        } else {
            panic!("expected MixedUnitary");
        }
    }

    #[test]
    fn depolarizing_zero_is_identity() {
        let op = Depolarizing(0.0, 0);
        if let Op::Channel(ChannelExpr::MixedUnitary(ops)) = op {
            assert!((ops[0].0 - 1.0).abs() < 1e-15);
            assert!(ops[1].0.abs() < 1e-15);
        } else {
            panic!("expected MixedUnitary");
        }
    }

    #[test]
    #[should_panic(expected = "probability p must be in [0, 1]")]
    fn depolarizing_rejects_negative() {
        let _ = Depolarizing(-0.1, 0);
    }

    #[test]
    #[should_panic(expected = "probability p must be in [0, 1]")]
    fn depolarizing_rejects_above_one() {
        let _ = Depolarizing(1.5, 0);
    }

    #[test]
    fn dephasing_is_channel() {
        let op = Dephasing(0.05, 0);
        assert!(op.is_channel());
    }

    #[test]
    fn bit_flip_is_channel() {
        let op = BitFlip(0.01, 0);
        assert!(op.is_channel());
    }

    #[test]
    fn pauli_channel_is_channel() {
        let op = PauliChannel(0.1, 0.05, 0.05, 0);
        assert!(op.is_channel());
    }

    #[test]
    #[should_panic(expected = "probabilities must sum to at most 1")]
    fn pauli_channel_rejects_overflow() {
        let _ = PauliChannel(0.5, 0.3, 0.3, 0);
    }

    #[test]
    fn amplitude_damping_is_channel() {
        let op = AmplitudeDamping(0.1, 0);
        assert!(op.is_channel());
    }

    #[test]
    fn amplitude_damping_qubits() {
        let op = AmplitudeDamping(0.5, 3);
        assert_eq!(op.qubits(), vec![3]);
    }

    #[test]
    #[should_panic(expected = "gamma must be in [0, 1]")]
    fn amplitude_damping_rejects_negative() {
        let _ = AmplitudeDamping(-0.1, 0);
    }

    #[test]
    fn noise_tensor_with_gate() {
        let op = H(0) & Depolarizing(0.1, 1);
        assert!(op.is_channel());
    }

    #[test]
    fn non_clifford_unitary_channel_tensor_promotes_to_channel_tensor() {
        let op = T(0) & Depolarizing(0.1, 1);
        assert!(op.is_channel());
        assert!(matches!(op, Op::Channel(ChannelExpr::Tensor(parts)) if parts.len() == 2));
    }

    #[test]
    fn noise_compose_with_gate() {
        let op = H(0) * Dephasing(0.05, 0);
        assert!(op.is_channel());
    }

    #[test]
    fn non_clifford_unitary_channel_compose_promotes_to_channel_compose() {
        let op = T(0) * Dephasing(0.05, 0);
        assert!(op.is_channel());
        assert!(matches!(op, Op::Channel(ChannelExpr::Compose(parts)) if parts.len() == 2));
    }

    #[test]
    fn mixed_unitary_qubits() {
        let op = Depolarizing(0.1, 5);
        assert_eq!(op.qubits(), vec![5]);
        assert_eq!(op.num_qubits(), 6);
    }

    #[test]
    fn bit_phase_flip_is_channel() {
        let op = BitPhaseFlip(0.05, 0);
        assert!(op.is_channel());
        if let Op::Channel(ChannelExpr::MixedUnitary(ops)) = op {
            assert_eq!(ops.len(), 2);
            let total: f64 = ops.iter().map(|(p, _)| p).sum();
            assert!((total - 1.0).abs() < 1e-15);
        } else {
            panic!("expected MixedUnitary");
        }
    }

    #[test]
    fn depolarizing2_is_channel() {
        let op = Depolarizing2(0.1, 0, 1);
        assert!(op.is_channel());
    }

    #[test]
    fn depolarizing2_has_16_terms() {
        let op = Depolarizing2(0.3, 0, 1);
        if let Op::Channel(ChannelExpr::MixedUnitary(ops)) = op {
            assert_eq!(ops.len(), 16);
            let total: f64 = ops.iter().map(|(p, _)| p).sum();
            assert!((total - 1.0).abs() < 1e-14);
        } else {
            panic!("expected MixedUnitary");
        }
    }

    #[test]
    fn depolarizing2_qubits() {
        let op = Depolarizing2(0.1, 2, 5);
        let mut qs = op.qubits();
        qs.sort_unstable();
        assert_eq!(qs, vec![2, 5]);
    }

    #[test]
    fn phase_damping_is_channel() {
        let op = PhaseDamping(0.1, 0);
        assert!(op.is_channel());
        assert_eq!(op.qubits(), vec![0]);
    }

    #[test]
    #[should_panic(expected = "lambda must be in [0, 1]")]
    fn phase_damping_rejects_negative() {
        let _ = PhaseDamping(-0.1, 0);
    }

    #[test]
    fn erasure_is_channel() {
        let op = Erasure(0.05, 0);
        assert!(op.is_channel());
        assert_eq!(op.qubits(), vec![0]);
    }

    #[test]
    #[should_panic(expected = "probability must be in [0, 1]")]
    fn erasure_rejects_negative() {
        let _ = Erasure(-0.1, 0);
    }

    #[test]
    fn reset_is_gate() {
        let op = Reset(0);
        assert!(op.is_gate());
        assert_eq!(op.qubits(), vec![0]);

        assert!(matches!(
            PX(1),
            Op::Gate(GateExpr::Prep {
                basis: Basis::X,
                qubit: 1
            })
        ));
        assert!(matches!(
            PY(2),
            Op::Gate(GateExpr::Prep {
                basis: Basis::Y,
                qubit: 2
            })
        ));
        assert!(matches!(
            PZ(3),
            Op::Gate(GateExpr::Prep {
                basis: Basis::Z,
                qubit: 3
            })
        ));
    }

    #[test]
    fn leakage_is_channel() {
        let op = Leakage(0.01, 0);
        assert!(op.is_channel());
        assert_eq!(op.qubits(), vec![0]);
    }

    #[test]
    #[should_panic(expected = "rate must be in [0, 1]")]
    fn leakage_rejects_negative() {
        let _ = Leakage(-0.1, 0);
    }

    #[test]
    fn symbolic_channels_compose_with_gates() {
        // All symbolic channels should compose/tensor with gates
        let ops = vec![
            PhaseDamping(0.1, 0),
            Erasure(0.05, 0),
            Leakage(0.01, 0),
            AmplitudeDamping(0.1, 0),
        ];
        for ch in ops {
            let tensored = H(1) & ch.clone();
            assert!(tensored.is_channel());
            let composed = H(0) * ch;
            assert!(composed.is_channel());
        }
    }

    // --- Reference overloads ---

    #[test]
    fn mixed_ref_ops() {
        let a = X(0);
        let b = H(1);
        // owned & ref
        let c = a.clone() & &b;
        assert!(c.is_clifford());
        // ref & owned
        let d = &a & b.clone();
        assert!(d.is_clifford());
    }

    // --- Pauli composition algebra ---

    #[test]
    fn x_times_y_is_iz() {
        let op = X(0) * Y(0);
        let ps = op.as_pauli().unwrap();
        assert_eq!(ps.phase(), crate::phase::quarter_phase::QuarterPhase::PlusI);
        assert_eq!(ps.weight(), 1);
        // The non-identity Pauli should be Z
        let (pauli, _) = ps.iter_pairs().next().unwrap();
        assert_eq!(pauli, crate::Pauli::Z);
    }

    #[test]
    fn y_times_z_is_ix() {
        let op = Y(0) * Z(0);
        let ps = op.as_pauli().unwrap();
        assert_eq!(ps.phase(), crate::phase::quarter_phase::QuarterPhase::PlusI);
        let (pauli, _) = ps.iter_pairs().next().unwrap();
        assert_eq!(pauli, crate::Pauli::X);
    }

    #[test]
    fn x_squared_is_identity() {
        let op = X(0) * X(0);
        let ps = op.as_pauli().unwrap();
        assert_eq!(
            ps.phase(),
            crate::phase::quarter_phase::QuarterPhase::PlusOne
        );
        assert_eq!(ps.weight(), 0);
    }

    // --- Identity algebra ---

    #[test]
    fn identity_compose_is_noop() {
        let op = I(0) * X(0);
        let ps = op.as_pauli().unwrap();
        assert_eq!(ps.weight(), 1);
        let (pauli, _) = ps.iter_pairs().next().unwrap();
        assert_eq!(pauli, crate::Pauli::X);
    }

    #[test]
    fn identity_tensor() {
        let op = I(0) & X(1);
        let ps = op.as_pauli().unwrap();
        assert_eq!(ps.weight(), 1);
    }

    #[test]
    fn identity_dagger_is_identity() {
        let op = I(0).dg();
        assert!(op.is_pauli());
        assert_eq!(op.as_pauli().unwrap().weight(), 0);
    }

    // --- Phase survival through promotion ---

    #[test]
    fn phased_pauli_promotes_to_clifford() {
        let phased = i * X(0);
        let promoted = phased.to_clifford_level();
        assert!(promoted.is_clifford());
        // Promote both to unitary and check they give the same matrix
        let ur = promoted.into_unitary().unwrap();
        let ur_direct = i * crate::unitary_rep::X(0);
        assert_eq!(ur, ur_direct);
    }

    #[test]
    fn phased_pauli_tensor_clifford_preserves_phase() {
        // (i*X(0)) & H(1) should promote to Clifford with phase retained
        let op = (i * X(0)) & H(1);
        assert!(op.is_clifford());
    }

    #[test]
    fn phased_pauli_promotes_to_unitary() {
        let op = (-i * X(0)) & T(1);
        assert!(op.is_unitary());
    }

    // --- Dagger of composed/tensored ops ---

    #[test]
    fn dagger_of_tensor() {
        let op = H(0) & SZ(1);
        let dg = op.dg();
        assert!(dg.is_clifford());
    }

    #[test]
    fn dagger_of_compose() {
        let op = H(0) * SZ(0);
        let dg = op.dg();
        assert!(dg.is_clifford());
    }

    #[test]
    fn dagger_of_unitary_compose() {
        let op = T(0) * H(0);
        let dg = op.dg();
        assert!(dg.is_unitary());
    }

    // --- Phase + Channel panics ---

    #[test]
    #[should_panic(expected = "not defined for Channel")]
    fn i_times_channel_panics() {
        let _ = i * Depolarizing(0.01, 0);
    }

    #[test]
    #[should_panic(expected = "not defined for Channel")]
    fn neg_channel_panics() {
        let _ = -Depolarizing(0.01, 0);
    }

    #[test]
    #[should_panic(expected = "not defined for Channel")]
    fn generic_phase_channel_panics() {
        let _ = phase(Angle64::QUARTER_TURN) * Depolarizing(0.01, 0);
    }

    #[test]
    #[should_panic(expected = "negation is not defined for Channel")]
    fn minus_one_channel_panics() {
        let _ = -1 * Depolarizing(0.01, 0);
    }

    // --- Noise boundary values ---

    #[test]
    fn noise_boundary_zero() {
        // p=0 should create valid channels (essentially identity)
        assert!(Depolarizing(0.0, 0).is_channel());
        assert!(Dephasing(0.0, 0).is_channel());
        assert!(BitFlip(0.0, 0).is_channel());
        assert!(BitPhaseFlip(0.0, 0).is_channel());
        assert!(Depolarizing2(0.0, 0, 1).is_channel());
        assert!(AmplitudeDamping(0.0, 0).is_channel());
        assert!(PhaseDamping(0.0, 0).is_channel());
        assert!(Erasure(0.0, 0).is_channel());
        assert!(Leakage(0.0, 0).is_channel());
    }

    #[test]
    fn noise_boundary_one() {
        // p=1 should create valid channels (maximal noise)
        assert!(Depolarizing(1.0, 0).is_channel());
        assert!(Dephasing(1.0, 0).is_channel());
        assert!(BitFlip(1.0, 0).is_channel());
        assert!(BitPhaseFlip(1.0, 0).is_channel());
        assert!(Depolarizing2(1.0, 0, 1).is_channel());
        assert!(AmplitudeDamping(1.0, 0).is_channel());
        assert!(PhaseDamping(1.0, 0).is_channel());
        assert!(Erasure(1.0, 0).is_channel());
        assert!(Leakage(1.0, 0).is_channel());
    }

    // --- MixedUnitary composition ---

    #[test]
    fn mixed_unitary_compose() {
        let op = Depolarizing(0.1, 0) * Dephasing(0.05, 0);
        assert!(op.is_channel());
        if let Op::Channel(ChannelExpr::Compose(parts)) = op {
            assert_eq!(parts.len(), 2);
        } else {
            panic!("expected Compose");
        }
    }

    #[test]
    fn mixed_unitary_tensor() {
        let op = Depolarizing(0.1, 0) & BitFlip(0.05, 1);
        assert!(op.is_channel());
        if let Op::Channel(ChannelExpr::Tensor(parts)) = op {
            assert_eq!(parts.len(), 2);
        } else {
            panic!("expected Tensor");
        }
    }

    // --- to_channel_level variant check ---

    #[test]
    fn to_channel_level_wraps_in_unitary_variant() {
        let op = H(0).to_channel_level();
        if let Op::Channel(ChannelExpr::Unitary(_)) = op {
            // correct
        } else {
            panic!("expected ChannelExpr::Unitary");
        }
    }
}
