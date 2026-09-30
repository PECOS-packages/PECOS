//! Width-preserving conversions between register storage and expression values.

use crate::{BitUInt, ExprValue, errors::PecosError};
use ::bitvec::{field::BitField, prelude::*};

fn checked_width(width: usize) -> Result<u16, PecosError> {
    u16::try_from(width)
        .ok()
        .filter(|&width| width != 0)
        .ok_or_else(|| {
            PecosError::Input(format!(
                "Expression width must be in 1..=65535, got {width}"
            ))
        })
}

/// Read all bits, retaining the source width including high zeros.
///
/// # Errors
/// Rejects empty vectors and widths greater than 65535.
pub fn to_bituint(bits: &BitSlice<u8, Lsb0>) -> Result<BitUInt, PecosError> {
    let width = checked_width(bits.len())?;
    let words = bits.chunks(64).map(BitField::load_le::<u64>).collect();
    Ok(BitUInt::from_raw_words(width, words))
}

/// Store an expression at the destination width, truncating or extending by tag.
/// Booleans and unsigned values zero-extend; signed values sign-extend.
///
/// # Errors
/// Rejects zero destination width and widths greater than 65535.
pub fn from_expr_value(value: &ExprValue, width: usize) -> Result<BitVec<u8, Lsb0>, PecosError> {
    let width = checked_width(width)?;
    let bits = match value {
        ExprValue::Signed(bits) => bits.resize_sign_extend(width),
        ExprValue::Unsigned(bits) => {
            BitUInt::from_raw_words(width, bits.to_words().into_boxed_slice())
        }
        ExprValue::Boolean(value) => BitUInt::new(width, u64::from(*value)),
    };
    Ok((0..width).map(|bit| bits.get_bit(bit)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversion_width_bounds() {
        for width in [0, 65536, 65537] {
            assert!(to_bituint(&BitVec::repeat(false, width)).is_err());
            assert!(from_expr_value(&ExprValue::signed(1), width).is_err());
        }
        for width in [1, 65535] {
            let bits = BitVec::repeat(true, width);
            let value = to_bituint(&bits).unwrap();
            assert_eq!(usize::from(value.size()), width);
            assert_eq!(
                from_expr_value(&ExprValue::Unsigned(value), width).unwrap(),
                bits
            );
        }
    }

    #[test]
    fn exact_round_trip_across_words() {
        for width in [1, 63, 64, 65, 100, 128, 129, 200] {
            let mut bits = BitVec::repeat(false, width);
            for bit in [0, 62, 63, 64, 65, 99, 127, 128] {
                if bit < width {
                    bits.set(bit, true);
                }
            }
            let value = to_bituint(&bits).unwrap();
            assert_eq!(usize::from(value.size()), width);
            assert_eq!(
                from_expr_value(&ExprValue::Unsigned(value), width).unwrap(),
                bits
            );
            let zeros = BitVec::repeat(false, width);
            assert_eq!(usize::from(to_bituint(&zeros).unwrap().size()), width);
        }
    }

    #[test]
    fn stores_extend_by_tag_and_truncate() {
        let pattern = BitUInt::new(4, 15);
        assert_eq!(
            from_expr_value(&ExprValue::Signed(pattern.clone()), 100).unwrap(),
            BitVec::<u8, Lsb0>::repeat(true, 100)
        );
        let unsigned = from_expr_value(&ExprValue::Unsigned(pattern), 100).unwrap();
        assert!(unsigned[..4].all());
        assert!(unsigned[4..].not_any());
        for value in [ExprValue::signed(-1), ExprValue::unsigned(15)] {
            assert_eq!(from_expr_value(&value, 2).unwrap(), bitvec![u8, Lsb0; 1, 1]);
        }
        for value in [false, true] {
            let bits = from_expr_value(&ExprValue::Boolean(value), 100).unwrap();
            assert_eq!(bits[0], value);
            assert!(bits[1..].not_any());
        }
    }
}
