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

use crate::{IndexableElement, Pauli, PauliOperator, Phase, QuarterPhase, Set};
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
pub struct PauliSparse<T: for<'a> Set<'a>> {
    phase: QuarterPhase,
    x_positions: T,
    z_positions: T,
}

impl<T> Default for PauliSparse<T>
where
    T: for<'a> Set<'a> + Default,
{
    fn default() -> Self {
        Self {
            phase: QuarterPhase::PlusOne,
            x_positions: T::default(),
            z_positions: T::default(),
        }
    }
}

impl<E, T> PauliSparse<T>
where
    T: for<'a> Set<'a, Element = E> + FromIterator<E>,
    for<'a> &'a T: BitOr<Output = T>,
    E: IndexableElement,
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
    pub fn with_operators(phase: QuarterPhase, x: &[E], y: &[E], z: &[E]) -> Result<Self, String> {
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

impl<E, T> PauliOperator for PauliSparse<T>
where
    T: for<'a> Set<'a, Element = E> + FromIterator<E>,
    for<'a> &'a T: BitAnd<Output = T> + BitXor<Output = T>,
    E: IndexableElement,
{
    fn phase(&self) -> QuarterPhase {
        self.phase
    }

    /// Returns the X positions as a sorted `Vec<usize>`.
    fn x_positions(&self) -> Vec<usize> {
        self.x_positions
            .iter()
            .map(super::super::element::IndexableElement::to_index)
            .collect()
    }

    /// Returns the Z positions as a sorted `Vec<usize>`.
    fn z_positions(&self) -> Vec<usize> {
        self.z_positions
            .iter()
            .map(super::super::element::IndexableElement::to_index)
            .collect()
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
        let mut phase = self.phase.multiply(&other.phase);

        // Calculate the overlap between X positions of `self` and Z positions of `other`
        let x_self_z_other = &self.x_positions & &other.z_positions; // => -i

        // Calculate the overlap between Z positions of `self` and X positions of `other`
        let z_self_x_other = &self.z_positions & &other.x_positions; // => +i

        // Anti-commutation occurs when the total overlap count is odd
        if x_self_z_other.len() % 2 == 1 {
            phase = phase.multiply(&QuarterPhase::MinusI);
        }
        if z_self_x_other.len() % 2 == 1 {
            phase = phase.multiply(&QuarterPhase::PlusI);
        }

        // Combine X and Z positions using XOR (symmetric difference)
        Self {
            phase,
            x_positions: &self.x_positions ^ &other.x_positions,
            z_positions: &self.z_positions ^ &other.z_positions,
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
            Pauli::X => x_positions.insert(E::from_index(qubit)),
            Pauli::Z => z_positions.insert(E::from_index(qubit)),
            Pauli::Y => {
                x_positions.insert(E::from_index(qubit));
                z_positions.insert(E::from_index(qubit));
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
}
