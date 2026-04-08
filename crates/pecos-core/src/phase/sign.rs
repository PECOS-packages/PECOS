use crate::Phase;
use crate::QuarterPhase;
use num_complex::Complex64;
use num_complex::Complex64 as Complex;

/// Second roots of unity: `{+1, -1}`.
///
/// This is the phase constraint for stabilizer group generators. A stabilizer
/// must square to +I, which requires its phase to be real (+1 or -1). A generator
/// with phase +i would give `(iP)^2 = -I`, which stabilizes no quantum state.
///
/// Widens to: [`QuarterPhase`] (via `From`)
/// Narrows from: [`QuarterPhase`] (via `TryFrom`, fails on `+/-i`)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Sign {
    PlusOne = 0b00,
    MinusOne = 0b01,
}

impl Phase for Sign {
    fn to_complex(&self) -> Complex64 {
        use Sign::{MinusOne, PlusOne};
        match self {
            PlusOne => Complex::new(1.0, 0.0),
            MinusOne => Complex::new(-1.0, 0.0),
        }
    }

    fn conjugate(&self) -> Self {
        *self
    }

    /// Multiplies two `Sign` values using XOR.
    fn multiply(&self, other: &Self) -> Self {
        match (*self as u8) ^ (*other as u8) {
            0 => Sign::PlusOne,
            _ => Sign::MinusOne,
        }
    }
}

impl TryFrom<QuarterPhase> for Sign {
    type Error = &'static str;

    fn try_from(phase: QuarterPhase) -> Result<Self, Self::Error> {
        match phase {
            QuarterPhase::PlusOne => Ok(Sign::PlusOne),
            QuarterPhase::MinusOne => Ok(Sign::MinusOne),
            _ => Err("Invalid phase: Sign can only be PlusOne or MinusOne"),
        }
    }
}

impl From<Sign> for QuarterPhase {
    fn from(sign: Sign) -> QuarterPhase {
        match sign {
            Sign::PlusOne => QuarterPhase::PlusOne,
            Sign::MinusOne => QuarterPhase::MinusOne,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_multiply() {
        let sign = Sign::PlusOne;

        match sign {
            Sign::PlusOne => assert_eq!(sign.multiply(&Sign::PlusOne), Sign::PlusOne),
            Sign::MinusOne => assert_eq!(sign.multiply(&Sign::MinusOne), Sign::PlusOne),
        }
    }
}
