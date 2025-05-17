use crate::Phase;
use num_complex::Complex64 as Complex;
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum QuarterPhase {
    #[default]
    PlusOne = 0b00,
    MinusOne = 0b01,
    PlusI = 0b10,
    MinusI = 0b11,
}

impl Phase for QuarterPhase {
    fn to_complex(&self) -> Complex {
        use QuarterPhase::{MinusI, MinusOne, PlusI, PlusOne};
        match self {
            PlusOne => Complex::new(1.0, 0.0),
            MinusOne => Complex::new(-1.0, 0.0),
            PlusI => Complex::new(0.0, 1.0),
            MinusI => Complex::new(0.0, -1.0),
        }
    }

    fn conjugate(&self) -> Self {
        use QuarterPhase::{MinusI, MinusOne, PlusI, PlusOne};
        match self {
            PlusOne | MinusOne => *self, // Real phases are self-conjugate
            PlusI => MinusI,
            MinusI => PlusI,
        }
    }

    fn multiply(&self, other: &QuarterPhase) -> QuarterPhase {
        let lhs = *self as u8;
        let rhs = *other as u8;

        // XOR signs and adjust for imaginary overlap
        let real = (lhs ^ rhs) & 0b01 ^ (((lhs & rhs) >> 1) & 0b01);

        // XOR imaginary parts
        let imaginary = (lhs ^ rhs) & 0b10;

        let result = real | imaginary;

        // Cast back to Phase
        unsafe { std::mem::transmute(result) }
    }
}

// pub fn multiply(self, other: Phase) -> Phase {
//     // Precomputed multiplication table
//     const MULT_TABLE: [[Phase; 4]; 4] = [
//         [Phase::PlusOne, Phase::MinusOne, Phase::PlusI, Phase::MinusI], // PlusOne
//         [Phase::MinusOne, Phase::PlusOne, Phase::MinusI, Phase::PlusI], // MinusOne
//         [Phase::PlusI, Phase::MinusI, Phase::MinusOne, Phase::PlusOne], // PlusI
//         [Phase::MinusI, Phase::PlusI, Phase::PlusOne, Phase::MinusOne], // MinusI
//     ];
//
//     // Lookup result
//     unsafe { *MULT_TABLE.get_unchecked(self as usize).get_unchecked(other as usize) }
// }

// pub fn multiply(self, other: Self) -> Self {
//     use Phase::*;
//     match (self, other) {
//         (PlusOne, x) | (x, PlusOne) => x,
//         (MinusOne, MinusOne) => PlusOne,
//         (PlusI, MinusI) | (MinusI, PlusI) => PlusOne,
//         (PlusI, PlusI) | (MinusI, MinusI) => MinusOne,
//         (MinusOne, PlusI) | (PlusI, MinusOne) => MinusI,
//         (MinusOne, MinusI) | (MinusI, MinusOne) => PlusI,
//     }
// }

// pub fn multiply(self, other: Phase) -> Phase {
//     let lhs = self as u8;
//     let rhs = other as u8;
//
//     // XOR signs and adjust for imaginary overlap
//     let real = (lhs ^ rhs) & 0b01 ^ ((lhs & rhs) >> 1 & 0b01);
//
//     // XOR imaginary parts
//     let imaginary = (lhs ^ rhs) & 0b10;
//
//     let result = real | imaginary;
//
//     // Cast back to Phase
//     unsafe { std::mem::transmute(result) }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use QuarterPhase::*;

    #[test]
    fn test_default_phase() {
        assert_eq!(QuarterPhase::default(), PlusOne);
    }

    #[test]
    fn test_enum_indices() {
        assert_eq!(PlusOne as usize, 0);
        assert_eq!(MinusOne as usize, 1);
        assert_eq!(PlusI as usize, 2);
        assert_eq!(MinusI as usize, 3);
    }

    #[test]
    fn test_phase_multiplication() {
        let cases = [
            (PlusOne, PlusOne, PlusOne),
            (PlusOne, MinusOne, MinusOne),
            (PlusOne, PlusI, PlusI),
            (PlusOne, MinusI, MinusI),
            (MinusOne, PlusOne, MinusOne),
            (MinusOne, MinusOne, PlusOne),
            (MinusOne, PlusI, MinusI),
            (MinusOne, MinusI, PlusI),
            (PlusI, PlusI, MinusOne),
            (PlusI, MinusI, PlusOne),
            (PlusI, MinusOne, MinusI),
            (MinusI, PlusI, PlusOne),
            (MinusI, MinusI, MinusOne),
            (MinusI, MinusOne, PlusI),
        ];

        for &(lhs, rhs, expected) in &cases {
            let result = lhs.multiply(&rhs);
            assert_eq!(
                result, expected,
                "Failed for lhs={lhs:?}, rhs={rhs:?} (got {result:?}, expected {expected:?})"
            );
        }
    }

    #[test]
    fn test_phase_conjugation() {
        use QuarterPhase::*;

        assert_eq!(PlusOne.conjugate(), PlusOne);
        assert_eq!(MinusOne.conjugate(), MinusOne);
        assert_eq!(PlusI.conjugate(), MinusI);
        assert_eq!(MinusI.conjugate(), PlusI);
    }

    #[test]
    fn test_phase_to_complex() {
        use QuarterPhase::*;
        use num_complex::Complex;

        assert_eq!(PlusOne.to_complex(), Complex::new(1.0, 0.0));
        assert_eq!(MinusOne.to_complex(), Complex::new(-1.0, 0.0));
        assert_eq!(PlusI.to_complex(), Complex::new(0.0, 1.0));
        assert_eq!(MinusI.to_complex(), Complex::new(0.0, -1.0));
    }
}
