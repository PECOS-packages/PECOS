use crate::Angle64;
use num_complex::Complex64;

#[allow(clippy::module_name_repetitions)]
pub mod quarter_phase;
pub mod sign;

pub use quarter_phase::QuarterPhase;

pub trait Phase {
    #[must_use]
    fn phase(&self) -> &Self {
        self
    }
    fn to_complex(&self) -> Complex64;
    #[must_use]
    fn conjugate(&self) -> Self;

    #[must_use]
    fn multiply(&self, other: &Self) -> Self;
}

/// A general phase factor, either a discrete quarter phase or arbitrary angle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GlobalPhase {
    /// Discrete phase: ±1, ±i (efficient for Paulis)
    Quarter(QuarterPhase),
    /// General phase: e^{iθ}
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
