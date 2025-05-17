use crate::{Pauli, PauliOperator, Phase, QuarterPhase};

/// Represents a compact Pauli operator using bitmaps for up to 64 qubits.
///
/// The `BitSetPauli` struct uses `x_bits` and `z_bits` to track which qubits are affected
/// by X and Z components of the operator. Each bit corresponds to a qubit, where a set bit
/// indicates the qubit is affected by the respective component.
///
/// - `x_bits`: A 64-bit bitmap indicating qubits affected by X.
/// - `z_bits`: A 64-bit bitmap indicating qubits affected by Z.
/// - `phase`: Represents the overall phase of the operator (`+1`, `-1`, `+i`, or `-i`).
///
/// # Performance
/// This representation is optimized for fixed-size systems (up to 64 qubits), allowing
/// fast bitwise operations to compute multiplication, weight, and commutation properties.
#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Debug, PartialEq)]
pub struct PauliBitmap {
    phase: QuarterPhase,
    x_bits: u64,
    z_bits: u64,
}

impl PauliBitmap {
    /// Initializes a new empty Pauli operator, which is equivalent to the identity.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn get_x_bits(&self) -> u64 {
        self.x_bits
    }

    #[must_use]
    pub fn get_z_bits(&self) -> u64 {
        self.z_bits
    }

    /// Creates a `BitSetPauli` instance with the specified phase and qubit positions for X, Y, and Z operators.
    ///
    /// This method constructs a Pauli operator using the provided qubit positions:
    /// - `x`: Positions affected by the X operator.
    /// - `y`: Positions affected by both X and Z operators (added to both `x_positions` and `z_positions`).
    /// - `z`: Positions affected by the Z operator.
    ///
    /// The `phase` specifies the initial phase of the operator.
    ///
    /// # Parameters
    /// - `phase`: The phase of the Pauli operator (`+1`, `-1`, `+i`, or `-i`).
    /// - `x`: A slice of positions affected by the X operator.
    /// - `y`: A slice of positions affected by both X and Z operators.
    /// - `z`: A slice of positions affected by the Z operator.
    ///
    /// # Returns
    /// A `Result` containing a new `SetPauli` instance if the input is valid,
    /// or an error message as a `String` if the input is invalid.
    ///
    /// # Errors
    /// This method returns an `Err` if:
    /// - Any qubit positions in `x`, `y`, or `z` overlap. Such overlaps are not allowed
    ///   since it is assumed the user is inputting unique Pauli operators.
    ///
    /// # Examples
    /// ```
    /// use pecos_core::{PauliBitmap, QuarterPhase};
    ///
    /// let phase = QuarterPhase::PlusOne;
    /// let x = [1, 2];
    /// let y = [3];
    /// let z = [4];
    ///
    /// let pauli = PauliBitmap::with_operators(phase, &x, &y, &z).unwrap();
    /// ```
    ///
    /// # Panics
    /// This function does not panic under normal usage.
    pub fn with_operators(
        phase: QuarterPhase,
        x: &[u64],
        y: &[u64],
        z: &[u64],
    ) -> Result<Self, String> {
        for &pos in x.iter().chain(y).chain(z) {
            if pos >= 64 {
                return Err("position exceeds 64 qubits".to_string());
            }
        }

        let mut x_bits = x.iter().fold(0, |bits, &pos| bits | (1 << pos));
        let mut z_bits = z.iter().fold(0, |bits, &pos| bits | (1 << pos));

        if x_bits & z_bits != 0 {
            return Err("x and z share common elements".to_string());
        }

        let y_bits = y.iter().fold(0, |bits, &pos| bits | (1 << pos));
        x_bits |= y_bits;
        z_bits |= y_bits;

        Ok(Self {
            phase,
            x_bits,
            z_bits,
        })
    }
}

impl Default for PauliBitmap {
    fn default() -> Self {
        Self {
            phase: QuarterPhase::PlusOne,
            x_bits: 0,
            z_bits: 0,
        }
    }
}

impl PauliOperator for PauliBitmap {
    fn phase(&self) -> QuarterPhase {
        self.phase
    }

    /// Returns a vector of positions affected by the X operator.
    fn x_positions(&self) -> Vec<usize> {
        // Collect indices of set bits in x_bits
        (0..64).filter(|&i| (self.x_bits & (1 << i)) != 0).collect()
    }

    /// Returns a vector of positions affected by the Z operator.
    fn z_positions(&self) -> Vec<usize> {
        // Collect indices of set bits in z_bits
        (0..64).filter(|&i| (self.z_bits & (1 << i)) != 0).collect()
    }

    #[inline]
    fn multiply(&self, other: &Self) -> Self {
        let mut phase = self.phase.multiply(&other.phase);
        // Check anti-commutation from both X-Z and Z-X overlaps at single positions
        if (self.x_bits & other.z_bits).count_ones() % 2 == 1 {
            phase = phase.multiply(&QuarterPhase::MinusI);
        }
        if (self.z_bits & other.x_bits).count_ones() % 2 == 1 {
            phase = phase.multiply(&QuarterPhase::PlusI);
        }

        Self {
            phase,
            x_bits: self.x_bits ^ other.x_bits,
            z_bits: self.z_bits ^ other.z_bits,
        }
    }

    #[inline]
    fn weight(&self) -> usize {
        (self.x_bits | self.z_bits).count_ones() as usize
    }

    #[inline]
    fn commutes_with(&self, other: &Self) -> bool {
        let overlap_count =
            ((self.x_bits & other.z_bits) ^ (self.z_bits & other.x_bits)).count_ones();
        overlap_count % 2 == 0
    }

    /// Creates a `PauliBitmap` operator with a single qubit in the specified state.
    fn from_single(qubit: usize, pauli: Pauli) -> Self {
        assert!(qubit < 64, "Qubit index exceeds the limit of 64");

        let mut x_bits = 0u64;
        let mut z_bits = 0u64;

        match pauli {
            Pauli::X => x_bits |= 1 << qubit,
            Pauli::Z => z_bits |= 1 << qubit,
            Pauli::Y => {
                x_bits |= 1 << qubit;
                z_bits |= 1 << qubit;
            }
            Pauli::I => {} // Identity does not affect any qubit
        }

        Self {
            phase: QuarterPhase::PlusOne,
            x_bits,
            z_bits,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // BitSetPauli tests
    #[test]
    fn test_valid_pauli_bit_creation() {
        let pauli =
            PauliBitmap::with_operators(QuarterPhase::MinusOne, &[1, 2], &[3], &[4]).unwrap();
        assert_eq!(pauli.phase, QuarterPhase::MinusOne);
        assert_eq!(pauli.x_bits, 0b1110); // Bits 1,2,3 set
        assert_eq!(pauli.z_bits, 0b11000); // Bits 3,4 set
    }

    #[test]
    fn test_pauli_bit_commuting() {
        let p1 = PauliBitmap::with_operators(QuarterPhase::PlusOne, &[0, 1], &[], &[2]).unwrap();
        let p2 = PauliBitmap::with_operators(QuarterPhase::PlusOne, &[1], &[], &[3]).unwrap();
        assert!(p1.commutes_with(&p2));
    }

    #[test]
    fn test_pauli_bit_anticommuting() {
        let p1 = PauliBitmap::with_operators(QuarterPhase::PlusOne, &[0, 1], &[], &[2]).unwrap();
        let p2 = PauliBitmap::with_operators(QuarterPhase::PlusOne, &[1], &[], &[0]).unwrap();
        assert!(!p1.commutes_with(&p2));
    }

    #[test]
    fn test_palui_bit_overlap_detection() {
        let result = PauliBitmap::with_operators(QuarterPhase::PlusOne, &[1, 2], &[], &[2, 4]);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "x and z share common elements");
    }

    #[test]
    fn test_pauli_bit_range_check() {
        let result = PauliBitmap::with_operators(
            QuarterPhase::PlusOne,
            &[65], // Exceeds 64 qubits
            &[],
            &[2, 4],
        );
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "position exceeds 64 qubits");
    }

    #[test]
    fn test_pauli_bit_multiplication() {
        // +1XXZ
        let p1 = PauliBitmap::with_operators(QuarterPhase::PlusOne, &[0, 1], &[], &[2]).unwrap();
        // +1ZX
        let p2 = PauliBitmap::with_operators(QuarterPhase::PlusOne, &[1], &[], &[0]).unwrap();
        // (-i) +1YIZ
        let result = p1.multiply(&p2);
        assert_eq!(result.phase, QuarterPhase::MinusI);
        assert_eq!(result.x_bits, 0b1);
        assert_eq!(result.z_bits, 0b101); // Both bits 0 and 2
    }

    #[test]
    fn test_pauli_bit_weight() {
        let pauli =
            PauliBitmap::with_operators(QuarterPhase::PlusOne, &[1, 2], &[3], &[4]).unwrap();
        assert_eq!(pauli.weight(), 4); // Positions 1,2,3,4 (3 appears in both but counted once)
    }

    #[test]
    fn test_empty_pauli_bit() {
        let pauli = PauliBitmap::new();
        assert_eq!(pauli.phase, QuarterPhase::PlusOne);
        assert_eq!(pauli.x_bits, 0);
        assert_eq!(pauli.z_bits, 0);
        assert_eq!(pauli.weight(), 0);
    }

    #[test]
    fn test_pauli_sparse_commutes() {
        let p1 = PauliBitmap::with_operators(QuarterPhase::PlusOne, &[0, 1], &[], &[2]).unwrap();
        let p2 = PauliBitmap::with_operators(QuarterPhase::PlusOne, &[1], &[], &[3]).unwrap();
        assert!(p1.commutes_with(&p2));
    }

    #[test]
    fn test_pauli_bit_commutes() {
        let p1 = PauliBitmap::with_operators(QuarterPhase::PlusOne, &[0, 1], &[], &[2]).unwrap();
        let p2 = PauliBitmap::with_operators(QuarterPhase::PlusOne, &[1], &[], &[3]).unwrap();
        assert!(p1.commutes_with(&p2));
    }

    #[test]
    fn test_pauli_bit_anticommutes() {
        let p1 = PauliBitmap::with_operators(QuarterPhase::PlusOne, &[0, 1], &[], &[2]).unwrap();
        let p2 = PauliBitmap::with_operators(QuarterPhase::PlusOne, &[1], &[], &[0]).unwrap();
        assert!(!p1.commutes_with(&p2));
    }
}
