use crate::Phase;
use crate::QuarterPhase;
use num_complex::Complex64;
use num_complex::Complex64 as Complex;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)] // Ensures same binary representation as Phase
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

    /// Multiplies two `Sign` values using XOR (superfast).
    fn multiply(&self, other: &Self) -> Self {
        unsafe { std::mem::transmute((*self as u8) ^ (*other as u8)) }
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
        // Safe because `Sign` variants map directly to `Phase` variants
        unsafe { std::mem::transmute(sign as u8) }
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
