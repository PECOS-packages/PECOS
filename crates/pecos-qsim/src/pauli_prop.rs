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

use super::clifford_gateable::{CliffordGateable, MeasurementResult};
use crate::quantum_simulator::QuantumSimulator;
use core::marker::PhantomData;
use pecos_core::{IndexableElement, Set, VecSet};
use std::collections::BTreeMap;
use std::fmt;

// TODO: Allow for the use of sets of elements of types other than usize

/// Type alias for the most common use case of `PauliProp` with standard vectors
#[expect(clippy::module_name_repetitions)]
pub type StdPauliProp = PauliProp<VecSet<usize>, usize>;

/// A simulator that tracks how Pauli operators transform under Clifford operations.
///
/// # Overview
/// The `PauliProp` simulator efficiently tracks the evolution of Pauli operators (X, Y, Z)
/// through Clifford quantum operations without maintaining the full quantum state. This makes
/// it particularly useful for:
/// - Simulating Pauli noise propagation in quantum circuits
/// - Tracking the evolution of Pauli observables
/// - Analyzing stabilizer states
/// - Verifying Clifford circuit implementations
///
/// # State Representation
/// The simulator maintains two sets to track Pauli operators:
/// - `xs`: Records qubits with X Pauli operators
/// - `zs`: Records qubits with Z Pauli operators
///
/// Y operators are implicitly represented by qubits present in both sets since Y = iXZ.
///
/// Optionally, the sign and phase can be tracked for full Pauli string representation.
///
/// # Type Parameters
/// - `T`: The set type used to store qubit indices (e.g., `VecSet`\<usize\>)
/// - `E`: The element type used for qubit indices (e.g., usize)
///
/// # Example
/// ```rust
/// use pecos_qsim::{StdPauliProp, CliffordGateable};
///
/// let mut sim = StdPauliProp::new();
/// sim.add_x(0);  // Track an X on qubit 0
/// sim.h(0);         // Apply Hadamard - transforms X to Z
/// assert!(sim.contains_z(0));  // Verify qubit 0 now has Z
/// ```
///
/// # Performance Characteristics
/// - Space complexity: O(n) where n is the number of qubits with non-identity operators
/// - Time complexity: O(1) for most gates
///
/// # References
/// - Gottesman, "The Heisenberg Representation of Quantum Computers"
///   <https://arxiv.org/abs/quant-ph/9807006>
#[derive(Clone, Debug)]
pub struct PauliProp<T, E>
where
    T: for<'a> Set<'a, Element = E>,
    E: IndexableElement,
{
    xs: T,
    zs: T,
    /// Optional tracking of the sign (false = +1, true = -1)
    sign: Option<bool>,
    /// Optional tracking of imaginary phase (0 = 1, 1 = i, 2 = -1, 3 = -i)
    img: Option<u8>,
    /// Maximum qubit index for string representation (optional)
    num_qubits: Option<usize>,
    _marker: PhantomData<E>,
}

impl<T, E> Default for PauliProp<T, E>
where
    E: IndexableElement,
    T: for<'a> Set<'a, Element = E>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, E> PauliProp<T, E>
where
    E: IndexableElement,
    T: for<'a> Set<'a, Element = E>,
{
    /// Creates a new `PauliProp` simulator.
    ///
    /// The simulator is initialized with no Pauli operators as the user needs to specify what
    /// observables to track.
    ///
    /// # Returns
    /// A new `PauliProp` instance
    #[must_use]
    pub fn new() -> Self {
        PauliProp {
            xs: T::new(),
            zs: T::new(),
            sign: None,
            img: None,
            num_qubits: None,
            _marker: PhantomData,
        }
    }

    /// Creates a new `PauliProp` simulator with sign tracking enabled.
    ///
    /// # Arguments
    /// * `num_qubits` - The total number of qubits (for string representation)
    ///
    /// # Returns
    /// A new `PauliProp` instance with sign tracking
    #[must_use]
    pub fn with_sign_tracking(num_qubits: usize) -> Self {
        PauliProp {
            xs: T::new(),
            zs: T::new(),
            sign: Some(false), // Start with +1
            img: Some(0),      // Start with no imaginary component
            num_qubits: Some(num_qubits),
            _marker: PhantomData,
        }
    }
}

impl<T, E> QuantumSimulator for PauliProp<T, E>
where
    E: IndexableElement,
    T: for<'a> Set<'a, Element = E>,
{
    /// Resets the state by clearing all Pauli all tracked X and Z operators.
    ///
    /// # Returns
    /// * `&mut Self` - Returns self for method chaining
    #[inline]
    fn reset(&mut self) -> &mut Self {
        self.xs.clear();
        self.zs.clear();
        if self.sign.is_some() {
            self.sign = Some(false);
        }
        if self.img.is_some() {
            self.img = Some(0);
        }
        self
    }
}

impl<T, E> PauliProp<T, E>
where
    T: for<'a> Set<'a, Element = E>,
    E: IndexableElement,
{
    /// Checks if the specified qubit has an X operator.
    ///
    /// # Arguments
    /// * `item` - The qubit index to check
    ///
    /// # Returns
    /// `true` if an X operator is present on the qubit
    #[inline]
    pub fn contains_x(&self, item: E) -> bool {
        self.xs.contains(&item)
    }

    /// Checks if the specified qubit has a Z operator.
    ///
    /// # Arguments
    /// * `item` - The qubit index to check
    ///
    /// # Returns
    /// `true` if a Z operator is present on the qubit
    #[inline]
    pub fn contains_z(&self, item: E) -> bool {
        self.zs.contains(&item)
    }

    /// Checks if the specified qubit has a Y operator.
    ///
    /// Since Y = iXZ, this checks for the presence of both X and Z operators.
    ///
    /// # Arguments
    /// * `item` - The qubit index to check
    ///
    /// # Returns
    /// `true` if both X and Z operators are present on the qubit
    #[inline]
    pub fn contains_y(&self, item: E) -> bool {
        self.contains_x(item) && self.contains_z(item)
    }

    /// Adds an X Pauli operator to be tracked to the specified qubit
    ///
    /// If the qubit already has:
    /// - No operator: Adds X
    /// - X operator: Removes X
    /// - Z operator: Creates Y (iXZ)
    /// - Y operator: Creates Z
    ///
    /// # Arguments
    /// * `item` - The qubit index to add the X operator to
    #[inline]
    pub fn add_x(&mut self, item: E) {
        self.xs.symmetric_difference_item_update(&item);
    }

    /// Adds a Z operator to the specified qubit.
    ///
    /// If the qubit already has:
    /// - No operator: Adds Z
    /// - Z operator: Removes Z
    /// - X operator: Creates Y (iXZ)
    /// - Y operator: Creates X
    ///
    /// # Arguments
    /// * `item` - The qubit index to add the Z operator to
    #[inline]
    pub fn add_z(&mut self, item: E) {
        self.zs.symmetric_difference_item_update(&item);
    }

    /// Adds a Y operator to the specified qubit.
    ///
    /// Since Y = iXZ, this adds both X and Z operators to the qubit.
    ///
    /// If the qubit already has:
    /// - No operator: Creates Y (Creates X and Z)
    /// - X operator: Removes X, Creates Z
    /// - Z operator: Removes Z, Creates X
    /// - Y operator: Removes X and Z
    ///
    /// # Arguments
    /// * `item` - The qubit index to add the Y operator to
    #[inline]
    pub fn add_y(&mut self, item: E) {
        self.add_x(item);
        self.add_z(item);
    }

    /// Flips the sign of the Pauli string (if sign tracking is enabled).
    #[inline]
    pub fn flip_sign(&mut self) {
        if let Some(ref mut sign) = self.sign {
            *sign = !*sign;
        }
    }

    /// Adds imaginary factors to the phase (if phase tracking is enabled).
    ///
    /// # Arguments
    /// * `num_is` - Number of i factors to add
    pub fn flip_img(&mut self, num_is: usize) {
        if let Some(img) = self.img.as_mut() {
            // Use modulo 4 on num_is first to ensure it fits in u8
            // Safe to cast since modulo 4 guarantees result is 0-3
            #[allow(clippy::cast_possible_truncation)]
            let num_is_mod = (num_is % 4) as u8;
            *img = (*img + num_is_mod) % 4;

            // If we've accumulated 2 or 3 i's, flip the sign
            let should_flip = *img == 2 || *img == 3;

            *img %= 2; // Keep only 0 or 1 for the imaginary part

            if should_flip {
                self.flip_sign();
            }
        }
    }

    /// Adds Pauli operators from a `BTreeMap` representation.
    ///
    /// The map should have keys "X", "Y", and "Z" with sets of qubit indices.
    /// This method properly handles operator composition with phase tracking if enabled.
    ///
    /// # Arguments
    /// * `paulis` - `BTreeMap` with "X", "Y", "Z" keys mapping to sets of qubit indices
    ///
    /// # Example
    /// ```rust
    /// use std::collections::BTreeMap;
    /// use pecos_qsim::StdPauliProp;
    /// use pecos_core::{VecSet, Set};
    ///
    /// let mut sim = StdPauliProp::with_sign_tracking(4);
    /// let mut paulis = BTreeMap::new();
    /// let mut x_set = VecSet::new();
    /// x_set.insert(0);
    /// x_set.insert(1);
    /// paulis.insert("X".to_string(), x_set);
    /// sim.add_paulis(&paulis);
    /// ```
    pub fn add_paulis(&mut self, paulis: &BTreeMap<String, T>)
    where
        T: Clone,
        E: Copy,
    {
        // Handle X operators
        if let Some(x_set) = paulis.get("X") {
            for &item in x_set.iter() {
                let was_y = self.contains_y(item);
                let was_z = self.contains_z(item) && !was_y;

                self.add_x(item);

                if self.sign.is_some() {
                    if was_y {
                        // Y·X = -iZ (applying X after Y)
                        self.flip_img(1);
                        self.flip_sign();
                    } else if was_z {
                        // Z·X = iY (applying X after Z)
                        self.flip_img(1);
                    }
                }
            }
        }

        // Handle Z operators
        if let Some(z_set) = paulis.get("Z") {
            for &item in z_set.iter() {
                let was_y = self.contains_y(item);
                let was_x = self.contains_x(item) && !was_y;

                self.add_z(item);

                if self.sign.is_some() {
                    if was_x {
                        // X·Z = -iY (applying Z after X)
                        self.flip_img(1);
                        self.flip_sign();
                    } else if was_y {
                        // Y·Z = iX (applying Z after Y)
                        self.flip_img(1);
                    }
                }
            }
        }

        // Handle Y operators
        if let Some(y_set) = paulis.get("Y") {
            for &item in y_set.iter() {
                let was_x = self.contains_x(item) && !self.contains_z(item);
                let was_z = self.contains_z(item) && !self.contains_x(item);

                self.add_y(item);

                if self.sign.is_some() {
                    if was_z {
                        // Z·Y = -iX (applying Y after Z)
                        self.flip_img(1);
                        self.flip_sign();
                    } else if was_x {
                        // X·Y = iZ (applying Y after X)
                        self.flip_img(1);
                    }
                }
            }
        }
    }

    /// Calculates the weight of the Pauli string (number of non-identity operators).
    ///
    /// # Returns
    /// The total number of qubits with non-identity Pauli operators
    pub fn weight(&self) -> usize {
        // Count X-only qubits
        let mut count = 0;
        for item in self.xs.iter() {
            if !self.zs.contains(item) {
                count += 1;
            }
        }

        // Count Z-only qubits
        for item in self.zs.iter() {
            if !self.xs.contains(item) {
                count += 1;
            }
        }

        // Count Y qubits (both X and Z)
        for item in self.xs.iter() {
            if self.zs.contains(item) {
                count += 1;
            }
        }

        count
    }

    /// Checks if this is the identity operator (no Pauli operators on any qubit).
    ///
    /// # Returns
    /// true if there are no X, Y, or Z operators on any qubit
    pub fn is_identity(&self) -> bool {
        self.xs.is_empty() && self.zs.is_empty()
    }

    /// Gets the sign as a boolean (false for +, true for -).
    ///
    /// # Returns
    /// false for positive sign, true for negative sign
    pub fn get_sign(&self) -> bool {
        self.sign.unwrap_or(false)
    }

    /// Gets the imaginary component (0 for real, 1 for imaginary).
    ///
    /// # Returns
    /// 0 for real, 1 for imaginary
    pub fn get_img(&self) -> u8 {
        self.img.unwrap_or(0)
    }

    /// Returns the sign string representation.
    ///
    /// # Returns
    /// A string like "+", "-", "+i", or "-i" depending on the phase
    pub fn sign_string(&self) -> String {
        match (self.sign, self.img) {
            (Some(false), Some(0) | None) => "+".to_string(),
            (Some(true), Some(0) | None) => "-".to_string(),
            (Some(false), Some(1)) => "+i".to_string(),
            (Some(true), Some(1)) => "-i".to_string(),
            _ => String::new(),
        }
    }

    /// Returns the operator string representation for sparse format.
    ///
    /// # Returns
    /// A string like "`X_0` `Z_2` `Y_3`" representing non-identity operators
    pub fn sparse_string(&self) -> String
    where
        E: Copy,
    {
        let mut entries = Vec::new();

        // Collect all qubit indices with operators
        for &item in self.xs.iter() {
            if self.contains_y(item) {
                entries.push((item, 'Y'));
            } else {
                entries.push((item, 'X'));
            }
        }

        for &item in self.zs.iter() {
            if !self.xs.contains(&item) {
                entries.push((item, 'Z'));
            }
        }

        if entries.is_empty() {
            "I".to_string()
        } else {
            // Format as sparse representation
            entries
                .iter()
                .map(|(idx, op)| format!("{op}{idx:?}"))
                .collect::<Vec<_>>()
                .join(" ")
        }
    }

    /// Returns the full Pauli string representation with sign and operators.
    ///
    /// # Returns
    /// A string like "+`X_0` `Z_2`" in sparse format
    pub fn to_pauli_string(&self) -> String
    where
        E: Copy,
    {
        format!("{}{}", self.sign_string(), self.sparse_string())
    }
}

// Specialized implementation for StdPauliProp (usize indices)
impl StdPauliProp {
    /// Get all qubits with X operators (including those with Y)
    #[must_use]
    pub fn get_x_qubits(&self) -> Vec<usize> {
        self.xs.iter().copied().collect()
    }

    /// Get all qubits with Z operators (including those with Y)
    #[must_use]
    pub fn get_z_qubits(&self) -> Vec<usize> {
        self.zs.iter().copied().collect()
    }

    /// Get all qubits with only X operators (not Y)
    #[must_use]
    pub fn get_x_only_qubits(&self) -> Vec<usize> {
        self.xs
            .iter()
            .filter(|&q| !self.contains_z(*q))
            .copied()
            .collect()
    }

    /// Get all qubits with only Z operators (not Y)
    #[must_use]
    pub fn get_z_only_qubits(&self) -> Vec<usize> {
        self.zs
            .iter()
            .filter(|&q| !self.contains_x(*q))
            .copied()
            .collect()
    }

    /// Get all qubits with Y operators (both X and Z)
    #[must_use]
    pub fn get_y_qubits(&self) -> Vec<usize> {
        self.xs
            .iter()
            .filter(|&q| self.contains_z(*q))
            .copied()
            .collect()
    }

    /// Returns the operator string as a dense representation.
    ///
    /// Requires `num_qubits` to be set.
    ///
    /// # Returns
    /// A string like "IXYZ" representing the Pauli operators on each qubit
    #[must_use]
    pub fn dense_string(&self) -> String {
        if let Some(n) = self.num_qubits {
            let mut result = String::with_capacity(n);
            for i in 0..n {
                if self.contains_y(i) {
                    result.push('Y');
                } else if self.contains_x(i) {
                    result.push('X');
                } else if self.contains_z(i) {
                    result.push('Z');
                } else {
                    result.push('I');
                }
            }
            result
        } else {
            self.sparse_string()
        }
    }

    /// Returns the full dense Pauli string with sign.
    ///
    /// # Returns
    /// A string like "+IXYZ" or "-iXYZ"
    #[must_use]
    pub fn to_dense_string(&self) -> String {
        format!("{}{}", self.sign_string(), self.dense_string())
    }
}

impl<T, E> fmt::Display for PauliProp<T, E>
where
    T: for<'a> Set<'a, Element = E>,
    E: IndexableElement + Copy,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_pauli_string())
    }
}

impl<T, E> CliffordGateable<E> for PauliProp<T, E>
where
    T: for<'a> Set<'a, Element = E>,
    E: IndexableElement,
{
    /// Applies the square root of Z gate (SZ or S gate) to the specified qubit.
    ///
    /// The SZ gate transforms Pauli operators as follows:
    /// ```text
    /// X -> Y
    /// Y -> -X
    /// Z -> Z
    /// ```
    ///
    /// Implementation: If the qubit has an X operator, toggle its Z operator
    ///
    /// # Arguments
    /// * `q` - The target qubit
    ///
    /// # Returns
    /// * `&mut Self` - Returns self for method chaining
    #[inline]
    fn sz(&mut self, q: E) -> &mut Self {
        if self.contains_x(q) {
            self.add_z(q);
        }
        self
    }

    /// Applies the Hadamard (H) gate to the specified qubit.
    ///
    /// The H gate transforms Pauli operators as follows:
    /// ```text
    /// X -> Z
    /// Z -> X
    /// Y -> -Y
    /// ```
    ///
    /// Implementation:
    /// - For X or Z: Swap between X and Z sets
    /// - For Y: Leave unchanged (Y transforms to -Y)
    ///
    /// # Arguments
    /// * `q` - The target qubit
    ///
    /// # Returns
    /// * `&mut Self` - Returns self for method chaining
    #[inline]
    #[expect(clippy::similar_names)]
    fn h(&mut self, q: E) -> &mut Self {
        let in_xs = self.contains_x(q);
        let in_zs = self.contains_z(q);

        if in_xs && in_zs {
        } else if in_xs {
            self.xs.remove(&q);
            self.zs.insert(q);
        } else if in_zs {
            self.zs.remove(&q);
            self.xs.insert(q);
        }
        self
    }

    /// Applies the controlled-X (CX) gate between two qubits
    ///
    /// The CX gate transforms Pauli operators as follows:
    /// ```text
    /// XI -> XX  (X on control propagates to target)
    /// IX -> IX  (X on target unchanged)
    /// ZI -> ZI  (Z on control unchanged)
    /// IZ -> ZZ  (Z on target propagates to control)
    /// ```
    ///
    /// Implementation:
    /// - If control has X: Toggle X on target
    /// - If target has Z: Toggle Z on control
    ///
    /// # Arguments
    /// * `q1` - The control qubit
    /// * `q2` - The target qubit
    ///
    /// # Returns
    /// * `&mut Self` - Returns self for method chaining
    #[inline]
    fn cx(&mut self, q1: E, q2: E) -> &mut Self {
        if self.contains_x(q1) {
            self.add_x(q2);
        }
        if self.contains_z(q2) {
            self.add_z(q1);
        }
        self
    }

    /// Performs a Z-basis measurement on the specified qubit.
    ///
    /// This simulates the effect of Pauli operators on measurement due to propagation.
    /// The outcome indicates whether an X operator has propagated to the measured
    /// qubit, which would flip the measurement result in the Z basis.
    ///
    /// Note: The outcomes are not actual measurements of the state but detect only if introduced
    /// operators might flip the value of measures and only correspond to valid measurements if they
    /// are originally deterministic.
    ///
    /// # Arguments
    /// * `q` - The qubit to measure
    ///
    /// # Returns
    /// * `MeasurementResult` containing:
    ///   - `outcome`: true if an X operator is present (measurement flipped)
    ///   - `is_deterministic`: always true for this simulator
    #[inline]
    fn mz(&mut self, q: E) -> MeasurementResult {
        let outcome = self.contains_x(q);
        MeasurementResult {
            outcome,
            is_deterministic: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn test_sign_tracking() {
        let mut sim = StdPauliProp::with_sign_tracking(4);

        // Initially should be +
        assert_eq!(sim.sign_string(), "+");

        // Flip sign
        sim.flip_sign();
        assert_eq!(sim.sign_string(), "-");

        // Add imaginary phase
        sim.flip_sign(); // Back to +
        sim.flip_img(1);
        assert_eq!(sim.sign_string(), "+i");

        // Two i's should give -1
        sim.flip_img(1);
        assert_eq!(sim.sign_string(), "-");
    }

    #[test]
    fn test_weight() {
        let mut sim = StdPauliProp::new();

        // Empty should have weight 0
        assert_eq!(sim.weight(), 0);

        // Add X on qubit 0
        sim.add_x(0);
        assert_eq!(sim.weight(), 1);

        // Add Z on qubit 1
        sim.add_z(1);
        assert_eq!(sim.weight(), 2);

        // Add Y on qubit 2 (both X and Z)
        sim.add_y(2);
        assert_eq!(sim.weight(), 3);

        // Adding X to qubit with Z makes Y
        sim.add_x(1);
        assert_eq!(sim.weight(), 3); // Still 3 operators
    }

    #[test]
    fn test_dense_string() {
        let mut sim = StdPauliProp::with_sign_tracking(4);

        sim.add_x(0);
        sim.add_z(2);
        sim.add_y(3);

        assert_eq!(sim.dense_string(), "XIZY");
        assert_eq!(sim.to_dense_string(), "+XIZY");

        sim.flip_sign();
        assert_eq!(sim.to_dense_string(), "-XIZY");
    }

    #[test]
    fn test_add_paulis() {
        let mut sim = StdPauliProp::with_sign_tracking(4);

        let mut paulis = BTreeMap::new();
        let mut x_set = VecSet::new();
        x_set.insert(0);
        x_set.insert(1);

        let mut z_set = VecSet::new();
        z_set.insert(2);

        paulis.insert("X".to_string(), x_set);
        paulis.insert("Z".to_string(), z_set);

        sim.add_paulis(&paulis);

        assert!(sim.contains_x(0));
        assert!(sim.contains_x(1));
        assert!(sim.contains_z(2));
        assert_eq!(sim.weight(), 3);
    }

    #[test]
    fn test_pauli_composition_with_phase() {
        let mut sim = StdPauliProp::with_sign_tracking(2);

        // Start with X on qubit 0
        sim.add_x(0);

        // Add Z to same qubit: X·Z = -iY (applying Z after X)
        let mut paulis = BTreeMap::new();
        let mut z_set = VecSet::new();
        z_set.insert(0);
        paulis.insert("Z".to_string(), z_set);

        sim.add_paulis(&paulis);

        // Should now have Y on qubit 0
        assert!(sim.contains_y(0));
        // Phase should be -i (X·Z = -iY)
        assert_eq!(sim.sign_string(), "-i");
    }
}
