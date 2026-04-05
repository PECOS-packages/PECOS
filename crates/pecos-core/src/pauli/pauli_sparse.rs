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

use crate::{Pauli, PauliOperator, Phase, QuarterPhase, Set};
use std::ops::{BitAnd, BitOr, BitXor};

/// Represents a Pauli operator with positions for X and Z components.
///
/// The `PauliSparse` struct uses generic sets (`x_positions` and `z_positions`) to track qubit
/// positions affected by the X and Z components of the operator.
///
/// - Positions in `x_positions` are affected by the X operator.
/// - Positions in `z_positions` are affected by the Z operator.
/// - Positions in both are affected by the Y operator.
#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Debug, PartialEq)]
pub struct PauliSparse<T: for<'a> Set<'a, Element = usize>> {
    phase: QuarterPhase,
    x_positions: T,
    z_positions: T,
}

impl<T> Default for PauliSparse<T>
where
    T: for<'a> Set<'a, Element = usize> + Default,
{
    fn default() -> Self {
        Self {
            phase: QuarterPhase::PlusOne,
            x_positions: T::default(),
            z_positions: T::default(),
        }
    }
}

impl<T> PauliSparse<T>
where
    T: for<'a> Set<'a, Element = usize>,
{
    /// Returns a reference to the X positions set.
    ///
    /// Positions in `x_positions` are affected by the X operator.
    /// Positions in both `x_positions` and `z_positions` are affected by Y.
    #[inline]
    #[must_use]
    pub fn x_set(&self) -> &T {
        &self.x_positions
    }

    /// Returns a reference to the Z positions set.
    ///
    /// Positions in `z_positions` are affected by the Z operator.
    /// Positions in both `x_positions` and `z_positions` are affected by Y.
    #[inline]
    #[must_use]
    pub fn z_set(&self) -> &T {
        &self.z_positions
    }

    /// Creates a `PauliSparse` directly from X and Z position sets.
    ///
    /// This is the most efficient way to create a `PauliSparse` when you already
    /// have the X and Z sets (e.g., from a stabilizer tableau). Y operators are
    /// represented as positions present in both sets.
    ///
    /// # Parameters
    /// - `phase`: The phase of the Pauli operator (`+1`, `-1`, `+i`, or `-i`).
    /// - `x_positions`: Set of positions with X component.
    /// - `z_positions`: Set of positions with Z component.
    ///
    /// # Examples
    /// ```
    /// use pecos_core::{PauliSparse, QuarterPhase, VecSet};
    ///
    /// let x_set = VecSet::from_iter([0, 1]);
    /// let z_set = VecSet::from_iter([1, 2]);  // qubit 1 has Y (both X and Z)
    ///
    /// let pauli = PauliSparse::from_xz_sets(QuarterPhase::PlusOne, x_set, z_set);
    /// ```
    #[must_use]
    pub fn from_xz_sets(phase: QuarterPhase, x_positions: T, z_positions: T) -> Self {
        Self {
            phase,
            x_positions,
            z_positions,
        }
    }

    /// Returns `true` if this is the identity operator (no X, Y, or Z components).
    #[inline]
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.x_positions.is_empty() && self.z_positions.is_empty()
    }

    /// Sets the phase of this Pauli operator.
    #[inline]
    pub fn set_phase(&mut self, phase: QuarterPhase) {
        self.phase = phase;
    }
}

impl<T> PauliSparse<T>
where
    T: for<'a> Set<'a, Element = usize> + FromIterator<usize>,
    for<'a> &'a T: BitOr<Output = T>,
{
    /// Initializes a new empty Pauli operator, which is equivalent to the identity.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a `SetPauli` instance with the specified phase and qubit positions for X, Y, and Z operators.
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
    /// - Any qubit positions in `x` and `z` overlap. Such overlaps are not allowed
    ///   since a single qubit cannot simultaneously be affected by both X and Z components
    ///   in the same Pauli operator.
    ///
    /// # Examples
    /// ```
    /// use pecos_core::{PauliSparse, QuarterPhase, VecSet};
    ///
    /// let phase = QuarterPhase::PlusOne;
    /// let x = [1, 2];
    /// let y = [3];
    /// let z = [4];
    ///
    /// let pauli: PauliSparse<VecSet<usize>> = PauliSparse::with_operators(phase, &x, &y, &z).unwrap();
    /// ```
    ///
    /// # Panics
    /// This function does not panic under normal usage.
    pub fn with_operators(
        phase: QuarterPhase,
        x: &[usize],
        y: &[usize],
        z: &[usize],
    ) -> Result<Self, String> {
        let mut x_set: T = x.iter().copied().collect();
        let mut z_set: T = z.iter().copied().collect();

        if x_set.intersection(&z_set).next().is_some() {
            return Err("x and z share common elements".to_string());
        }

        for &elem in y {
            x_set = (&x_set | &T::from_iter([elem])).clone();
            z_set = (&z_set | &T::from_iter([elem])).clone();
        }

        Ok(Self {
            phase,
            x_positions: x_set,
            z_positions: z_set,
        })
    }
}

// TODO: Consider making a clear distinction between mutation in place and not

impl<T> PauliOperator for PauliSparse<T>
where
    T: for<'a> Set<'a, Element = usize> + FromIterator<usize>,
    for<'a> &'a T: BitAnd<Output = T> + BitXor<Output = T>,
{
    fn phase(&self) -> QuarterPhase {
        self.phase
    }

    /// Returns the X positions as a sorted `Vec<usize>`.
    fn x_positions(&self) -> Vec<usize> {
        self.x_positions.iter().copied().collect()
    }

    /// Returns the Z positions as a sorted `Vec<usize>`.
    fn z_positions(&self) -> Vec<usize> {
        self.z_positions.iter().copied().collect()
    }

    /// Multiplies two `SetPauli` operators and returns the result.
    ///
    /// # Parameters
    /// - `other`: The other `SetPauli` operator to multiply with.
    ///
    /// # Returns
    /// A new `SetPauli` operator representing the product.
    #[inline]
    fn multiply(&self, other: &Self) -> Self {
        let x_result = &self.x_positions ^ &other.x_positions;
        let z_result = &self.z_positions ^ &other.z_positions;

        // Phase formula derived from the Weyl convention: P(a,c) = i^{a·c} X^a Z^c
        // Product phase = i^{y_self + y_other - y_result} * (-1)^{|z_self ∩ x_other|}
        // where y = |x ∩ z| counts Y positions.
        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
        // qubit count fits in i32
        let y_self = (&self.x_positions & &self.z_positions).len() as i32;
        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
        // qubit count fits in i32
        let y_other = (&other.x_positions & &other.z_positions).len() as i32;
        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
        // qubit count fits in i32
        let y_result = (&x_result & &z_result).len() as i32;
        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
        // qubit count fits in i32
        let commute = (&self.z_positions & &other.x_positions).len() as i32;

        // Combined exponent of i (using (-1) = i^2)
        let exp = ((y_self + y_other - y_result + 2 * commute) % 4 + 4) % 4;
        let phase_correction = match exp {
            0 => QuarterPhase::PlusOne,
            1 => QuarterPhase::PlusI,
            2 => QuarterPhase::MinusOne,
            3 => QuarterPhase::MinusI,
            _ => unreachable!(),
        };

        Self {
            phase: self
                .phase
                .multiply(&other.phase)
                .multiply(&phase_correction),
            x_positions: x_result,
            z_positions: z_result,
        }
    }

    /// Calculates the weight of the `SetPauli` operator.
    ///
    /// The weight is the total number of unique positions affected by the X and Z components.
    ///
    /// # Returns
    /// The weight as a `usize`.
    #[inline]
    fn weight(&self) -> usize {
        self.x_positions.union(&self.z_positions).count()
    }

    /// Checks if this `SetPauli` operator commutes with another.
    ///
    /// # Parameters
    /// - `other`: The other `SetPauli` operator to check commutation with.
    ///
    /// # Returns
    /// `true` if the operators commute, `false` if they anti-commute.
    #[inline]
    fn commutes_with(&self, other: &Self) -> bool {
        // Check if the anti-commutation count is even (commutes) or odd (anti-commutes)
        let x_and_z = &self.x_positions & &other.z_positions;
        let z_and_x = &self.z_positions & &other.x_positions;

        (x_and_z.len() + z_and_x.len()) % 2 == 0
    }

    /// Creates a `PauliSparse` operator with a single qubit in the specified state.
    fn from_single(qubit: usize, pauli: Pauli) -> Self {
        let mut x_positions = T::default();
        let mut z_positions = T::default();

        match pauli {
            Pauli::X => x_positions.insert(qubit),
            Pauli::Z => z_positions.insert(qubit),
            Pauli::Y => {
                x_positions.insert(qubit);
                z_positions.insert(qubit);
            }
            Pauli::I => {} // Identity does not affect any positions
        }

        Self {
            phase: QuarterPhase::PlusOne,
            x_positions,
            z_positions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VecSet;
    use std::fmt::Debug;

    fn assert_sets_equal<E: Clone + Debug + PartialEq + Ord, T: for<'a> Set<'a, Element = E>>(
        left: &T,
        right: &T,
    ) {
        let mut left_elements: Vec<E> = left.iter().cloned().collect();
        let mut right_elements: Vec<E> = right.iter().cloned().collect();
        left_elements.sort();
        right_elements.sort();
        assert_eq!(left_elements, right_elements);
    }

    #[test]
    fn test_valid_pauli_creation() {
        let pauli =
            PauliSparse::with_operators(QuarterPhase::PlusOne, &[1usize, 2], &[3usize], &[4usize])
                .unwrap();

        assert_eq!(pauli.phase, QuarterPhase::PlusOne);
        assert_sets_equal(&pauli.x_positions, &VecSet::from_iter([1usize, 2, 3]));
        assert_sets_equal(&pauli.z_positions, &VecSet::from_iter([3usize, 4]));
    }

    #[test]
    fn test_overlap_in_x_and_z() {
        // Simply use Vec to avoid array size issues
        let result = PauliSparse::<VecSet<usize>>::with_operators(
            QuarterPhase::MinusOne,
            &[1usize, 2],
            &[3usize],
            &[2usize, 4], // Overlaps with x
        );
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "x and z share common elements");
    }

    #[test]
    fn test_y_addition_to_x_and_z() {
        let pauli =
            PauliSparse::with_operators(QuarterPhase::PlusOne, &[1usize], &[2usize], &[3usize])
                .unwrap();
        assert_sets_equal(&pauli.x_positions, &VecSet::from_iter([1usize, 2]));
        assert_sets_equal(&pauli.z_positions, &VecSet::from_iter([2usize, 3]));
    }

    #[test]
    fn test_empty_inputs() {
        // Test default/empty constructor
        let pauli = PauliSparse::<VecSet<usize>>::new();
        assert_eq!(pauli.phase, QuarterPhase::PlusOne);
        assert!(pauli.x_positions.is_empty());
        assert!(pauli.z_positions.is_empty());
    }

    #[test]
    fn test_partial_inputs() {
        let pauli = PauliSparse::<VecSet<usize>>::with_operators(
            QuarterPhase::MinusOne,
            &[1usize, 2],
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(pauli.phase, QuarterPhase::MinusOne);
        assert_eq!(pauli.x_positions, VecSet::from_iter([1usize, 2]));
        assert!(pauli.z_positions.is_empty());
    }

    #[test]
    fn test_pauli_sparse_anticommutes() {
        let p1 =
            PauliSparse::<VecSet<usize>>::with_operators(QuarterPhase::PlusOne, &[0, 1], &[], &[2])
                .unwrap();
        let p2 =
            PauliSparse::<VecSet<usize>>::with_operators(QuarterPhase::PlusOne, &[1], &[], &[0])
                .unwrap();
        assert!(!p1.commutes_with(&p2));
    }

    #[test]
    fn test_from_xz_sets() {
        let x_set = VecSet::from_iter([0usize, 1]);
        let z_set = VecSet::from_iter([1usize, 2]);

        let pauli = PauliSparse::from_xz_sets(QuarterPhase::MinusOne, x_set, z_set);

        assert_eq!(pauli.phase(), QuarterPhase::MinusOne);
        assert_sets_equal(pauli.x_set(), &VecSet::from_iter([0usize, 1]));
        assert_sets_equal(pauli.z_set(), &VecSet::from_iter([1usize, 2]));
        // Qubit 1 has Y (in both sets), weight should be 3
        assert_eq!(pauli.weight(), 3);
    }

    #[test]
    fn test_set_accessors() {
        let pauli =
            PauliSparse::with_operators(QuarterPhase::PlusOne, &[0usize, 1], &[2usize], &[3usize])
                .unwrap();

        // x_set should contain 0, 1, 2 (x positions + y position)
        assert_sets_equal(pauli.x_set(), &VecSet::from_iter([0usize, 1, 2]));
        // z_set should contain 2, 3 (z position + y position)
        assert_sets_equal(pauli.z_set(), &VecSet::from_iter([2usize, 3]));
    }

    #[test]
    fn test_is_identity() {
        let identity = PauliSparse::<VecSet<usize>>::new();
        assert!(identity.is_identity());

        let not_identity =
            PauliSparse::<VecSet<usize>>::with_operators(QuarterPhase::PlusOne, &[0], &[], &[])
                .unwrap();
        assert!(!not_identity.is_identity());
    }

    #[test]
    fn test_set_phase() {
        let mut pauli =
            PauliSparse::<VecSet<usize>>::with_operators(QuarterPhase::PlusOne, &[0], &[], &[])
                .unwrap();
        assert_eq!(pauli.phase(), QuarterPhase::PlusOne);

        pauli.set_phase(QuarterPhase::MinusI);
        assert_eq!(pauli.phase(), QuarterPhase::MinusI);
    }

    // ========================================================================
    // Bug-hunting: multiply with Y inputs
    // ========================================================================

    #[test]
    fn test_multiply_x_times_y() {
        // X * Y = iZ
        let x = PauliSparse::<VecSet<usize>>::with_operators(QuarterPhase::PlusOne, &[0], &[], &[])
            .unwrap();
        let y = PauliSparse::<VecSet<usize>>::with_operators(QuarterPhase::PlusOne, &[], &[0], &[])
            .unwrap();
        let result = x.multiply(&y);
        assert!(
            result.x_positions.is_empty(),
            "X*Y should give Z (no x-bit)"
        );
        assert_sets_equal(&result.z_positions, &VecSet::from_iter([0usize]));
        assert_eq!(
            result.phase,
            QuarterPhase::PlusI,
            "X*Y = iZ, phase should be +i"
        );
    }

    #[test]
    fn test_multiply_y_times_x() {
        // Y * X = -iZ
        let y = PauliSparse::<VecSet<usize>>::with_operators(QuarterPhase::PlusOne, &[], &[0], &[])
            .unwrap();
        let x = PauliSparse::<VecSet<usize>>::with_operators(QuarterPhase::PlusOne, &[0], &[], &[])
            .unwrap();
        let result = y.multiply(&x);
        assert!(result.x_positions.is_empty());
        assert_sets_equal(&result.z_positions, &VecSet::from_iter([0usize]));
        assert_eq!(
            result.phase,
            QuarterPhase::MinusI,
            "Y*X = -iZ, phase should be -i"
        );
    }

    #[test]
    fn test_multiply_y_times_z() {
        // Y * Z = iX
        let y = PauliSparse::<VecSet<usize>>::with_operators(QuarterPhase::PlusOne, &[], &[0], &[])
            .unwrap();
        let z = PauliSparse::<VecSet<usize>>::with_operators(QuarterPhase::PlusOne, &[], &[], &[0])
            .unwrap();
        let result = y.multiply(&z);
        assert_sets_equal(&result.x_positions, &VecSet::from_iter([0usize]));
        assert!(result.z_positions.is_empty());
        assert_eq!(
            result.phase,
            QuarterPhase::PlusI,
            "Y*Z = iX, phase should be +i"
        );
    }

    #[test]
    fn test_multiply_z_times_y() {
        // Z * Y = -iX
        let z = PauliSparse::<VecSet<usize>>::with_operators(QuarterPhase::PlusOne, &[], &[], &[0])
            .unwrap();
        let y = PauliSparse::<VecSet<usize>>::with_operators(QuarterPhase::PlusOne, &[], &[0], &[])
            .unwrap();
        let result = z.multiply(&y);
        assert_sets_equal(&result.x_positions, &VecSet::from_iter([0usize]));
        assert!(result.z_positions.is_empty());
        assert_eq!(
            result.phase,
            QuarterPhase::MinusI,
            "Z*Y = -iX, phase should be -i"
        );
    }

    #[test]
    fn test_multiply_y_times_y() {
        // Y * Y = I
        let y = PauliSparse::<VecSet<usize>>::with_operators(QuarterPhase::PlusOne, &[], &[0], &[])
            .unwrap();
        let result = y.multiply(&y);
        assert!(result.x_positions.is_empty());
        assert!(result.z_positions.is_empty());
        assert_eq!(
            result.phase,
            QuarterPhase::PlusOne,
            "Y*Y = I, phase should be +1"
        );
    }

    #[test]
    fn test_multiply_consistency_with_algebra() {
        // Cross-check: PauliSparse multiply vs the known-correct algebra
        // X * Z = -iY (no Y in inputs, should work)
        let x = PauliSparse::<VecSet<usize>>::with_operators(QuarterPhase::PlusOne, &[0], &[], &[])
            .unwrap();
        let z = PauliSparse::<VecSet<usize>>::with_operators(QuarterPhase::PlusOne, &[], &[], &[0])
            .unwrap();
        let result = x.multiply(&z);
        assert_sets_equal(&result.x_positions, &VecSet::from_iter([0usize]));
        assert_sets_equal(&result.z_positions, &VecSet::from_iter([0usize]));
        assert_eq!(result.phase, QuarterPhase::MinusI, "X*Z = -iY");
    }

    // ========================================================================
    // commutes_with with Y inputs
    // ========================================================================

    #[test]
    fn test_commutes_y_with_x() {
        let y = PauliSparse::<VecSet<usize>>::with_operators(QuarterPhase::PlusOne, &[], &[0], &[])
            .unwrap();
        let x = PauliSparse::<VecSet<usize>>::with_operators(QuarterPhase::PlusOne, &[0], &[], &[])
            .unwrap();
        assert!(!y.commutes_with(&x), "Y and X anticommute on same qubit");
    }

    #[test]
    fn test_commutes_y_with_z() {
        let y = PauliSparse::<VecSet<usize>>::with_operators(QuarterPhase::PlusOne, &[], &[0], &[])
            .unwrap();
        let z = PauliSparse::<VecSet<usize>>::with_operators(QuarterPhase::PlusOne, &[], &[], &[0])
            .unwrap();
        assert!(!y.commutes_with(&z), "Y and Z anticommute on same qubit");
    }

    #[test]
    fn test_commutes_y_with_y() {
        let y1 =
            PauliSparse::<VecSet<usize>>::with_operators(QuarterPhase::PlusOne, &[], &[0], &[])
                .unwrap();
        let y2 =
            PauliSparse::<VecSet<usize>>::with_operators(QuarterPhase::PlusOne, &[], &[0], &[])
                .unwrap();
        assert!(y1.commutes_with(&y2), "Y commutes with itself");
    }

    // ========================================================================
    // Multi-qubit multiply with multiple Y inputs (count > 1 paths)
    // ========================================================================

    #[test]
    fn test_multiply_double_y_inputs() {
        // (Y0, Y1) * (X0, X1): q0: Y*X=-iZ, q1: Y*X=-iZ -> phase = (-i)^2 = -1
        let p1 =
            PauliSparse::<VecSet<usize>>::with_operators(QuarterPhase::PlusOne, &[], &[0, 1], &[])
                .unwrap();
        let p2 =
            PauliSparse::<VecSet<usize>>::with_operators(QuarterPhase::PlusOne, &[0, 1], &[], &[])
                .unwrap();
        let result = p1.multiply(&p2);
        assert_eq!(
            result.phase,
            QuarterPhase::MinusOne,
            "(YY)*(XX) phase should be -1"
        );
        assert!(result.x_positions.is_empty());
        assert_sets_equal(&result.z_positions, &VecSet::from_iter([0usize, 1]));
    }
}
