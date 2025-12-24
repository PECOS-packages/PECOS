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

#[allow(clippy::module_name_repetitions)]
pub mod pauli_bitmap;

#[allow(clippy::module_name_repetitions)]
pub mod pauli_sparse;

mod pauli_stabilizer_string;
#[allow(clippy::module_name_repetitions)]
pub mod pauli_string;

use crate::QuarterPhase;
use std::fmt::Debug;

/// Single-qubit Pauli operator
/// #[`allow(clippy::module_name_repetitions)`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
#[derive(Default)]
pub enum Pauli {
    #[default]
    I = 0b00,
    X = 0b01,
    Z = 0b10,
    Y = 0b11,
}

/// A trait for general Pauli operators.
///
/// This trait defines the behavior and properties of Pauli operators, including their
/// ability to be multiplied, determining their weight, and checking commutation relations.
#[allow(clippy::module_name_repetitions)]
pub trait PauliOperator: Clone + Debug {
    fn phase(&self) -> QuarterPhase;
    fn x_positions(&self) -> Vec<usize>;
    fn z_positions(&self) -> Vec<usize>;

    /// Multiplies two Pauli operators and returns the resulting operator.
    ///
    /// # Parameters
    /// - `other`: The other Pauli operator to multiply with.
    ///
    /// # Returns
    /// A new Pauli operator representing the product of the two.
    #[must_use]
    fn multiply(&self, other: &Self) -> Self;

    /// Calculates the weight of the Pauli operator.
    ///
    /// The weight is the number of positions where the operator acts non-trivially
    /// (i.e., acts as X, Y, or Z instead of the identity).
    ///
    /// # Returns
    /// The weight of the operator as a `usize`.
    fn weight(&self) -> usize;

    /// Determines whether this Pauli operator commutes with another.
    ///
    /// # Parameters
    /// - `other`: The other Pauli operator to check commutation with.
    ///
    /// # Returns
    /// `true` if the operators commute, `false` if they anti-commute.
    fn commutes_with(&self, other: &Self) -> bool;

    fn from_single(qubit: usize, pauli: Pauli) -> Self;
}
