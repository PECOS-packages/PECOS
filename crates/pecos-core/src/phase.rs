//! Phase factors for quantum operators.
//!
//! This module provides a hierarchy of phase types corresponding to the natural
//! stratification of phases in quantum computing:
//!
//! | Type | Values | Roots of unity | Used by |
//! |------|--------|----------------|---------|
//! | [`Sign`] | `{+1, -1}` | 2nd | [`PauliStabilizerGroup`] generators |
//! | [`QuarterPhase`] | `{+1, -1, +i, -i}` | 4th | [`PauliString`], [`PauliSequence`] |
//! | [`GlobalPhase`] | any `e^{i theta}` | continuous | [`UnitaryRep`] |
//!
//! The hierarchy forms a subtype chain: `Sign` < `QuarterPhase` < `GlobalPhase`,
//! with lossless widening conversions (`From`) going up and fallible narrowing
//! conversions (`TryFrom`) going down:
//!
//! ```
//! use pecos_core::{QuarterPhase, Sign, GlobalPhase};
//! use pecos_core::Phase;
//!
//! // Widening: Sign -> QuarterPhase (lossless)
//! let qp: QuarterPhase = Sign::PlusOne.into();
//! assert_eq!(qp, QuarterPhase::PlusOne);
//!
//! // Widening: QuarterPhase -> GlobalPhase (lossless)
//! let gp: GlobalPhase = QuarterPhase::PlusI.into();
//!
//! // Narrowing: QuarterPhase -> Sign (fallible)
//! assert!(Sign::try_from(QuarterPhase::PlusOne).is_ok());
//! assert!(Sign::try_from(QuarterPhase::PlusI).is_err());
//! ```
//!
//! ## Why three types?
//!
//! Multiplying Pauli operators naturally produces fourth roots of unity (`QuarterPhase`),
//! but stabilizer groups require that every element squares to +I, which restricts
//! generators to real phases (`Sign`). General quantum operators (rotations, etc.)
//! can carry arbitrary phases (`GlobalPhase`).
//!
//! All three types implement the [`Phase`] trait.
//!
//! [`Sign`]: sign::Sign
//! [`PauliString`]: crate::PauliString
//! [`UnitaryRep`]: crate::UnitaryRep
//! [`PauliSequence`]: https://docs.rs/pecos-quantum/latest/pecos_quantum/struct.PauliSequence.html
//! [`PauliStabilizerGroup`]: https://docs.rs/pecos-quantum/latest/pecos_quantum/struct.PauliStabilizerGroup.html

use crate::Angle64;
use num_complex::Complex64;

#[allow(clippy::module_name_repetitions)]
pub mod quarter_phase;
pub mod sign;

pub use quarter_phase::QuarterPhase;

/// A trait for phase factors that can be converted to complex numbers,
/// conjugated, and multiplied.
///
/// Implemented by [`Sign`](sign::Sign), [`QuarterPhase`], and [`GlobalPhase`].
pub trait Phase {
    #[must_use]
    fn phase(&self) -> &Self {
        self
    }
    /// Converts to a complex number representation.
    fn to_complex(&self) -> Complex64;
    /// Returns the complex conjugate of this phase.
    #[must_use]
    fn conjugate(&self) -> Self;
    /// Multiplies two phases.
    #[must_use]
    fn multiply(&self, other: &Self) -> Self;
}

/// A general phase factor `e^{i theta}`, used by [`UnitaryRep`](crate::UnitaryRep).
///
/// This is the most general phase type. When the phase happens to be a fourth
/// root of unity it is stored as a [`QuarterPhase`] for efficiency; otherwise
/// it is stored as an [`Angle64`] fixpoint angle.
///
/// Widens from: [`QuarterPhase`] (via `From`), [`Angle64`] (via `From`)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GlobalPhase {
    /// Discrete phase: +1, -1, +i, -i (efficient for Paulis)
    Quarter(QuarterPhase),
    /// Arbitrary phase: e^{i theta}
    Angle(Angle64),
}

impl GlobalPhase {
    /// Creates an identity phase (+1).
    #[must_use]
    pub fn one() -> Self {
        Self::Quarter(QuarterPhase::PlusOne)
    }

    /// Creates a phase of -1.
    #[must_use]
    pub fn minus_one() -> Self {
        Self::Quarter(QuarterPhase::MinusOne)
    }

    /// Creates a phase of +i.
    #[must_use]
    pub fn i() -> Self {
        Self::Quarter(QuarterPhase::PlusI)
    }

    /// Creates a phase of -i.
    #[must_use]
    pub fn minus_i() -> Self {
        Self::Quarter(QuarterPhase::MinusI)
    }

    /// Creates a phase from an angle.
    #[must_use]
    pub fn from_angle(angle: Angle64) -> Self {
        // Check if it's a quarter phase
        if angle == Angle64::ZERO {
            Self::Quarter(QuarterPhase::PlusOne)
        } else if angle == Angle64::QUARTER_TURN {
            Self::Quarter(QuarterPhase::PlusI)
        } else if angle == Angle64::HALF_TURN {
            Self::Quarter(QuarterPhase::MinusOne)
        } else if angle == Angle64::QUARTER_TURN + Angle64::HALF_TURN {
            Self::Quarter(QuarterPhase::MinusI)
        } else {
            Self::Angle(angle)
        }
    }

    /// Returns the quarter phase if this is one, None otherwise.
    #[must_use]
    pub fn as_quarter(&self) -> Option<QuarterPhase> {
        match self {
            Self::Quarter(qp) => Some(*qp),
            Self::Angle(_) => None,
        }
    }

    /// Returns the angle representation.
    #[must_use]
    pub fn to_angle(&self) -> Angle64 {
        match self {
            Self::Quarter(qp) => match qp {
                QuarterPhase::PlusOne => Angle64::ZERO,
                QuarterPhase::PlusI => Angle64::QUARTER_TURN,
                QuarterPhase::MinusOne => Angle64::HALF_TURN,
                QuarterPhase::MinusI => Angle64::QUARTER_TURN + Angle64::HALF_TURN,
            },
            Self::Angle(a) => *a,
        }
    }

    /// Converts to complex number.
    #[must_use]
    pub fn to_complex(&self) -> Complex64 {
        match self {
            Self::Quarter(qp) => qp.to_complex(),
            Self::Angle(a) => {
                let (s, c) = a.sin_cos();
                Complex64::new(c, s)
            }
        }
    }

    /// Returns the conjugate (negated angle).
    #[must_use]
    pub fn conjugate(&self) -> Self {
        match self {
            Self::Quarter(qp) => Self::Quarter(qp.conjugate()),
            Self::Angle(a) => Self::Angle(Angle64::ZERO - *a),
        }
    }

    /// Multiplies two phases.
    #[must_use]
    pub fn multiply(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Quarter(a), Self::Quarter(b)) => Self::Quarter(a.multiply(b)),
            _ => Self::from_angle(self.to_angle() + other.to_angle()),
        }
    }
}

impl From<sign::Sign> for GlobalPhase {
    fn from(s: sign::Sign) -> Self {
        Self::Quarter(QuarterPhase::from(s))
    }
}

impl From<QuarterPhase> for GlobalPhase {
    fn from(qp: QuarterPhase) -> Self {
        Self::Quarter(qp)
    }
}

impl From<Angle64> for GlobalPhase {
    fn from(angle: Angle64) -> Self {
        Self::from_angle(angle)
    }
}

impl Default for GlobalPhase {
    fn default() -> Self {
        Self::one()
    }
}
