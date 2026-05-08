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
use pecos_core::{QubitId, Set, VecSet};
use std::collections::BTreeMap;
use std::fmt;

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
/// # Example
/// ```rust
/// use pecos_core::qid;
/// use pecos_simulators::{PauliProp, CliffordGateable};
///
/// let mut sim = PauliProp::new();
/// sim.track_x(&[0]);  // Track an X on qubit 0
/// sim.h(&qid(0));    // Apply Hadamard - transforms X to Z
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
pub struct PauliProp {
    xs: VecSet<usize>,
    zs: VecSet<usize>,
    /// Optional tracking of the sign (false = +1, true = -1)
    sign: Option<bool>,
    /// Optional tracking of imaginary phase (0 = 1, 1 = i, 2 = -1, 3 = -i)
    img: Option<u8>,
    /// Maximum qubit index for string representation (optional)
    num_qubits: Option<usize>,
}

impl Default for PauliProp {
    fn default() -> Self {
        Self::new()
    }
}

impl PauliProp {
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
            xs: VecSet::new(),
            zs: VecSet::new(),
            sign: None,
            img: None,
            num_qubits: None,
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
            xs: VecSet::new(),
            zs: VecSet::new(),
            sign: Some(false), // Start with +1
            img: Some(0),      // Start with no imaginary component
            num_qubits: Some(num_qubits),
        }
    }
}

impl QuantumSimulator for PauliProp {
    fn num_qubits(&self) -> usize {
        self.num_qubits.unwrap_or(0)
    }

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

impl PauliProp {
    /// Checks if the specified qubit has an X operator.
    ///
    /// # Arguments
    /// * `item` - The qubit index to check
    ///
    /// # Returns
    /// `true` if an X operator is present on the qubit
    #[inline]
    #[must_use]
    pub fn contains_x(&self, item: usize) -> bool {
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
    #[must_use]
    pub fn contains_z(&self, item: usize) -> bool {
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
    #[must_use]
    pub fn contains_y(&self, item: usize) -> bool {
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
    /// * `qubits` - The qubit indices to track X operators on
    #[inline]
    pub fn track_x(&mut self, qubits: &[usize]) {
        for &q in qubits {
            self.xs.symmetric_difference_item_update(&q);
        }
    }

    /// Tracks Z operators on the specified qubits.
    ///
    /// For each qubit, if it already has:
    /// - No operator: Adds Z
    /// - Z operator: Removes Z
    /// - X operator: Creates Y (iXZ)
    /// - Y operator: Creates X
    ///
    /// # Arguments
    /// * `qubits` - The qubit indices to track Z operators on
    #[inline]
    pub fn track_z(&mut self, qubits: &[usize]) {
        for &q in qubits {
            self.zs.symmetric_difference_item_update(&q);
        }
    }

    /// Tracks Y operators on the specified qubits.
    ///
    /// Since Y = iXZ, this tracks both X and Z operators on each qubit.
    ///
    /// # Arguments
    /// * `qubits` - The qubit indices to track Y operators on
    #[inline]
    pub fn track_y(&mut self, qubits: &[usize]) {
        for &q in qubits {
            self.track_x(&[q]);
            self.track_z(&[q]);
        }
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
    /// use pecos_simulators::PauliProp;
    /// use pecos_core::{VecSet, Set};
    ///
    /// let mut sim = PauliProp::with_sign_tracking(4);
    /// let mut paulis = BTreeMap::new();
    /// let mut x_set = VecSet::new();
    /// x_set.insert(0);
    /// x_set.insert(1);
    /// paulis.insert("X".to_string(), x_set);
    /// sim.add_paulis(&paulis);
    /// ```
    pub fn add_paulis(&mut self, paulis: &BTreeMap<String, VecSet<usize>>) {
        // Handle X operators
        if let Some(x_set) = paulis.get("X") {
            for &item in x_set {
                let was_y = self.contains_y(item);
                let was_z = self.contains_z(item) && !was_y;

                self.track_x(&[item]);

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
            for &item in z_set {
                let was_y = self.contains_y(item);
                let was_x = self.contains_x(item) && !was_y;

                self.track_z(&[item]);

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
            for &item in y_set {
                let was_x = self.contains_x(item) && !self.contains_z(item);
                let was_z = self.contains_z(item) && !self.contains_x(item);

                self.track_y(&[item]);

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
    #[must_use]
    pub fn weight(&self) -> usize {
        // Count X-only qubits
        let mut count = 0;
        for item in &self.xs {
            if !self.zs.contains(item) {
                count += 1;
            }
        }

        // Count Z-only qubits
        for item in &self.zs {
            if !self.xs.contains(item) {
                count += 1;
            }
        }

        // Count Y qubits (both X and Z)
        for item in &self.xs {
            if self.zs.contains(item) {
                count += 1;
            }
        }

        count
    }

    /// Remove all Pauli operators from a specific qubit.
    ///
    /// Models reset (PZ) which absorbs any propagating error on that qubit.
    pub fn clear_qubit(&mut self, qubit: usize) {
        use pecos_core::sets::set::Set;
        self.xs.remove(&qubit);
        self.zs.remove(&qubit);
    }

    /// Checks if this is the identity operator (no Pauli operators on any qubit).
    ///
    /// # Returns
    /// true if there are no X, Y, or Z operators on any qubit
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.xs.is_empty() && self.zs.is_empty()
    }

    /// Gets the sign as a boolean (false for +, true for -).
    ///
    /// # Returns
    /// false for positive sign, true for negative sign
    #[must_use]
    pub fn get_sign(&self) -> bool {
        self.sign.unwrap_or(false)
    }

    /// Gets the imaginary component (0 for real, 1 for imaginary).
    ///
    /// # Returns
    /// 0 for real, 1 for imaginary
    #[must_use]
    pub fn get_img(&self) -> u8 {
        self.img.unwrap_or(0)
    }

    /// Returns the sign string representation.
    ///
    /// # Returns
    /// A string like "+", "-", "+i", or "-i" depending on the phase
    #[must_use]
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
    #[must_use]
    pub fn sparse_string(&self) -> String {
        let mut entries = Vec::new();

        // Collect all qubit indices with operators
        for &item in &self.xs {
            if self.contains_y(item) {
                entries.push((item, 'Y'));
            } else {
                entries.push((item, 'X'));
            }
        }

        for &item in &self.zs {
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
    #[must_use]
    pub fn to_pauli_string(&self) -> String {
        format!("{}{}", self.sign_string(), self.sparse_string())
    }
}

impl PauliProp {
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

    fn set_x_component(&mut self, q: usize, value: bool) {
        if self.contains_x(q) != value {
            self.track_x(&[q]);
        }
    }

    fn set_z_component(&mut self, q: usize, value: bool) {
        if self.contains_z(q) != value {
            self.track_z(&[q]);
        }
    }

    fn set_components(&mut self, q: usize, x: bool, z: bool) {
        self.set_x_component(q, x);
        self.set_z_component(q, z);
    }
}

impl fmt::Display for PauliProp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_pauli_string())
    }
}

impl CliffordGateable for PauliProp {
    /// Applies the square root of Z gate (SZ or S gate) to the specified qubits.
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
    /// * `qubits` - The target qubits
    ///
    /// # Returns
    /// * `&mut Self` - Returns self for method chaining
    #[inline]
    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            if self.contains_x(qu) {
                self.track_z(&[qu]);
            }
        }
        self
    }

    /// Applies the adjoint square root of Z gate.
    ///
    /// Ignoring global phase, `SZ` and `SZdg` have the same binary Pauli action:
    /// X <-> Y, Z -> Z.
    #[inline]
    fn szdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            if self.contains_x(qu) {
                self.track_z(&[qu]);
            }
        }
        self
    }

    /// Applies the Hadamard (H) gate to the specified qubits.
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
    /// * `qubits` - The target qubits
    ///
    /// # Returns
    /// * `&mut Self` - Returns self for method chaining
    #[inline]
    #[expect(clippy::similar_names)]
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            let in_xs = self.contains_x(qu);
            let in_zs = self.contains_z(qu);

            if in_xs && in_zs {
            } else if in_xs {
                self.xs.remove(&qu);
                self.zs.insert(qu);
            } else if in_zs {
                self.zs.remove(&qu);
                self.xs.insert(qu);
            }
        }
        self
    }

    /// Applies the square root of X gate.
    ///
    /// Binary Pauli action: X -> X, Z <-> Y.
    #[inline]
    fn sx(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            if self.contains_z(qu) {
                self.track_x(&[qu]);
            }
        }
        self
    }

    /// Applies the adjoint square root of X gate.
    ///
    /// Ignoring global phase, `SX` and `SXdg` have the same binary Pauli action.
    #[inline]
    fn sxdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            if self.contains_z(qu) {
                self.track_x(&[qu]);
            }
        }
        self
    }

    /// Applies the square root of Y gate.
    ///
    /// Binary Pauli action: X <-> Z, Y -> Y.
    #[inline]
    fn sy(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            let x = self.contains_x(qu);
            let z = self.contains_z(qu);
            self.set_components(qu, z, x);
        }
        self
    }

    /// Applies the adjoint square root of Y gate.
    ///
    /// Ignoring global phase, `SY` and `SYdg` have the same binary Pauli action.
    #[inline]
    fn sydg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            let x = self.contains_x(qu);
            let z = self.contains_z(qu);
            self.set_components(qu, z, x);
        }
        self
    }

    /// Applies the controlled-X (CX) gate between pairs of qubits
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
    /// * `qubits` - Pairs of (control, target) qubits
    ///
    /// # Returns
    /// * `&mut Self` - Returns self for method chaining
    #[inline]
    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q1, q2) in pairs {
            let q1 = q1.index();
            let q2 = q2.index();
            if self.contains_x(q1) {
                self.track_x(&[q2]);
            }
            if self.contains_z(q2) {
                self.track_z(&[q1]);
            }
        }
        self
    }

    /// Applies the controlled-Y gate.
    #[inline]
    fn cy(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q1, q2) in pairs {
            let q1 = q1.index();
            let q2 = q2.index();
            let x1 = self.contains_x(q1);
            let z1 = self.contains_z(q1);
            let x2 = self.contains_x(q2);
            let z2 = self.contains_z(q2);
            self.set_components(q1, x1, z1 ^ x2 ^ z2);
            self.set_components(q2, x2 ^ x1, z2 ^ x1);
        }
        self
    }

    /// Applies the controlled-Z gate.
    #[inline]
    fn cz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q1, q2) in pairs {
            let q1 = q1.index();
            let q2 = q2.index();
            let x1 = self.contains_x(q1);
            let z1 = self.contains_z(q1);
            let x2 = self.contains_x(q2);
            let z2 = self.contains_z(q2);
            self.set_components(q1, x1, z1 ^ x2);
            self.set_components(q2, x2, z2 ^ x1);
        }
        self
    }

    /// Applies the square root of XX gate.
    #[inline]
    fn sxx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q1, q2) in pairs {
            let q1 = q1.index();
            let q2 = q2.index();
            let x1 = self.contains_x(q1);
            let z1 = self.contains_z(q1);
            let x2 = self.contains_x(q2);
            let z2 = self.contains_z(q2);
            let affected = z1 ^ z2;
            self.set_components(q1, x1 ^ affected, z1);
            self.set_components(q2, x2 ^ affected, z2);
        }
        self
    }

    /// Applies the adjoint square root of XX gate.
    ///
    /// Ignoring global phase, `SXX` and `SXXdg` have the same binary Pauli action.
    #[inline]
    fn sxxdg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q1, q2) in pairs {
            let q1 = q1.index();
            let q2 = q2.index();
            let x1 = self.contains_x(q1);
            let z1 = self.contains_z(q1);
            let x2 = self.contains_x(q2);
            let z2 = self.contains_z(q2);
            let affected = z1 ^ z2;
            self.set_components(q1, x1 ^ affected, z1);
            self.set_components(q2, x2 ^ affected, z2);
        }
        self
    }

    /// Applies the square root of YY gate.
    #[inline]
    fn syy(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q1, q2) in pairs {
            let q1 = q1.index();
            let q2 = q2.index();
            let x1 = self.contains_x(q1);
            let z1 = self.contains_z(q1);
            let x2 = self.contains_x(q2);
            let z2 = self.contains_z(q2);
            self.set_components(q1, x2 ^ z1 ^ z2, x1 ^ x2 ^ z2);
            self.set_components(q2, x1 ^ z1 ^ z2, x1 ^ x2 ^ z1);
        }
        self
    }

    /// Applies the adjoint square root of YY gate.
    ///
    /// Ignoring global phase, `SYY` and `SYYdg` have the same binary Pauli action.
    #[inline]
    fn syydg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q1, q2) in pairs {
            let q1 = q1.index();
            let q2 = q2.index();
            let x1 = self.contains_x(q1);
            let z1 = self.contains_z(q1);
            let x2 = self.contains_x(q2);
            let z2 = self.contains_z(q2);
            self.set_components(q1, x2 ^ z1 ^ z2, x1 ^ x2 ^ z2);
            self.set_components(q2, x1 ^ z1 ^ z2, x1 ^ x2 ^ z1);
        }
        self
    }

    /// Applies the square root of ZZ gate.
    #[inline]
    fn szz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q1, q2) in pairs {
            let q1 = q1.index();
            let q2 = q2.index();
            let x1 = self.contains_x(q1);
            let z1 = self.contains_z(q1);
            let x2 = self.contains_x(q2);
            let z2 = self.contains_z(q2);
            let affected = x1 ^ x2;
            self.set_components(q1, x1, z1 ^ affected);
            self.set_components(q2, x2, z2 ^ affected);
        }
        self
    }

    /// Applies the adjoint square root of ZZ gate.
    ///
    /// Ignoring global phase, `SZZ` and `SZZdg` have the same binary Pauli action.
    #[inline]
    fn szzdg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q1, q2) in pairs {
            let q1 = q1.index();
            let q2 = q2.index();
            let x1 = self.contains_x(q1);
            let z1 = self.contains_z(q1);
            let x2 = self.contains_x(q2);
            let z2 = self.contains_z(q2);
            let affected = x1 ^ x2;
            self.set_components(q1, x1, z1 ^ affected);
            self.set_components(q2, x2, z2 ^ affected);
        }
        self
    }

    /// Applies the SWAP gate.
    #[inline]
    fn swap(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q1, q2) in pairs {
            let q1 = q1.index();
            let q2 = q2.index();
            let x1 = self.contains_x(q1);
            let z1 = self.contains_z(q1);
            let x2 = self.contains_x(q2);
            let z2 = self.contains_z(q2);
            self.set_components(q1, x2, z2);
            self.set_components(q2, x1, z1);
        }
        self
    }

    /// Performs a Z-basis measurement on the specified qubits.
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
    /// * `qubits` - The qubits to measure
    ///
    /// # Returns
    /// * `Vec<MeasurementResult>` containing:
    ///   - `outcome`: true if an X operator is present (measurement flipped)
    ///   - `is_deterministic`: always true for this simulator
    #[inline]
    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        qubits
            .iter()
            .map(|&q| {
                let outcome = self.contains_x(q.index());
                MeasurementResult {
                    outcome,
                    is_deterministic: true,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford_matrix_oracle::{CliffordMatrixGate, all_pauli_strings, conjugate_pauli};
    use std::collections::BTreeMap;

    fn prop_from_dense(input: &str) -> PauliProp {
        let mut prop = PauliProp::with_sign_tracking(input.len());
        for (q, p) in input.chars().enumerate() {
            match p {
                'I' => {}
                'X' => prop.track_x(&[q]),
                'Y' => prop.track_y(&[q]),
                'Z' => prop.track_z(&[q]),
                _ => panic!("invalid Pauli label {p}"),
            }
        }
        prop
    }

    fn assert_gate_table<F>(name: &str, table: &[(&str, &str)], mut apply: F)
    where
        F: FnMut(&mut PauliProp),
    {
        for &(input, expected) in table {
            let mut prop = prop_from_dense(input);
            apply(&mut prop);
            assert_eq!(prop.dense_string(), expected, "{name}: {input}");
        }
    }

    fn assert_gate_matches_matrix_oracle<F>(
        name: &str,
        gate: CliffordMatrixGate,
        num_qubits: usize,
        mut apply: F,
    ) where
        F: FnMut(&mut PauliProp),
    {
        for input in all_pauli_strings(num_qubits) {
            let expected = conjugate_pauli(gate, &input);
            let mut prop = prop_from_dense(&input);
            apply(&mut prop);
            assert_eq!(
                prop.dense_string(),
                expected.pauli,
                "{name}: {input}, oracle sign {}",
                expected.sign
            );
        }
    }

    fn reverse_two_qubit_pauli(pauli: &str) -> String {
        let labels: Vec<char> = pauli.chars().collect();
        assert_eq!(labels.len(), 2);
        [labels[1], labels[0]].into_iter().collect()
    }

    fn assert_reversed_pair_matches_matrix_oracle<F>(
        name: &str,
        gate: CliffordMatrixGate,
        mut apply: F,
    ) where
        F: FnMut(&mut PauliProp, &[(QubitId, QubitId)]),
    {
        let reversed_pair = [(QubitId(1), QubitId(0))];
        for input in all_pauli_strings(2) {
            let oracle_input = reverse_two_qubit_pauli(&input);
            let mut expected = conjugate_pauli(gate, &oracle_input);
            expected.pauli = reverse_two_qubit_pauli(&expected.pauli);

            let mut prop = prop_from_dense(&input);
            apply(&mut prop, &reversed_pair);
            assert_eq!(
                prop.dense_string(),
                expected.pauli,
                "{name} reversed pair: {input}, oracle sign {}",
                expected.sign
            );
        }
    }

    fn assert_two_pair_batch_matches_sequential<F>(name: &str, mut apply: F)
    where
        F: FnMut(&mut PauliProp, &[(QubitId, QubitId)]),
    {
        let pairs = [(QubitId(0), QubitId(1)), (QubitId(2), QubitId(3))];
        for input in all_pauli_strings(4) {
            let mut batched = prop_from_dense(&input);
            apply(&mut batched, &pairs);

            let mut sequential = prop_from_dense(&input);
            apply(&mut sequential, &pairs[0..1]);
            apply(&mut sequential, &pairs[1..2]);

            assert_eq!(
                batched.dense_string(),
                sequential.dense_string(),
                "{name} batched: {input}"
            );
        }
    }

    #[test]
    fn test_sign_tracking() {
        let mut sim = PauliProp::with_sign_tracking(4);

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
        let mut sim = PauliProp::new();

        // Empty should have weight 0
        assert_eq!(sim.weight(), 0);

        // Add X on qubit 0
        sim.track_x(&[0]);
        assert_eq!(sim.weight(), 1);

        // Add Z on qubit 1
        sim.track_z(&[1]);
        assert_eq!(sim.weight(), 2);

        // Add Y on qubit 2 (both X and Z)
        sim.track_y(&[2]);
        assert_eq!(sim.weight(), 3);

        // Adding X to qubit with Z makes Y
        sim.track_x(&[1]);
        assert_eq!(sim.weight(), 3); // Still 3 operators
    }

    #[test]
    fn test_dense_string() {
        let mut sim = PauliProp::with_sign_tracking(4);

        sim.track_x(&[0]);
        sim.track_z(&[2]);
        sim.track_y(&[3]);

        assert_eq!(sim.dense_string(), "XIZY");
        assert_eq!(sim.to_dense_string(), "+XIZY");

        sim.flip_sign();
        assert_eq!(sim.to_dense_string(), "-XIZY");
    }

    #[test]
    fn test_add_paulis() {
        let mut sim = PauliProp::with_sign_tracking(4);

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
        let mut sim = PauliProp::with_sign_tracking(2);

        // Start with X on qubit 0
        sim.track_x(&[0]);

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

    #[test]
    fn test_direct_clifford_gate_binary_truth_tables() {
        let q0 = QubitId(0);
        let q1 = QubitId(1);
        let pair = [(q0, q1)];

        assert_gate_table(
            "SZdg",
            &[("I", "I"), ("X", "Y"), ("Y", "X"), ("Z", "Z")],
            |prop| {
                prop.szdg(&[q0]);
            },
        );
        assert_gate_table(
            "SX",
            &[("I", "I"), ("X", "X"), ("Y", "Z"), ("Z", "Y")],
            |prop| {
                prop.sx(&[q0]);
            },
        );
        assert_gate_table(
            "SXdg",
            &[("I", "I"), ("X", "X"), ("Y", "Z"), ("Z", "Y")],
            |prop| {
                prop.sxdg(&[q0]);
            },
        );
        assert_gate_table(
            "SY",
            &[("I", "I"), ("X", "Z"), ("Y", "Y"), ("Z", "X")],
            |prop| {
                prop.sy(&[q0]);
            },
        );
        assert_gate_table(
            "SYdg",
            &[("I", "I"), ("X", "Z"), ("Y", "Y"), ("Z", "X")],
            |prop| {
                prop.sydg(&[q0]);
            },
        );
        assert_gate_table(
            "CY",
            &[("XI", "XY"), ("IX", "ZX"), ("ZI", "ZI"), ("IZ", "ZZ")],
            |prop| {
                prop.cy(&pair);
            },
        );
        assert_gate_table(
            "CZ",
            &[("XI", "XZ"), ("IX", "ZX"), ("ZI", "ZI"), ("IZ", "IZ")],
            |prop| {
                prop.cz(&pair);
            },
        );
        assert_gate_table(
            "SXX",
            &[("XI", "XI"), ("IX", "IX"), ("ZI", "YX"), ("IZ", "XY")],
            |prop| {
                prop.sxx(&pair);
            },
        );
        assert_gate_table(
            "SXXdg",
            &[("XI", "XI"), ("IX", "IX"), ("ZI", "YX"), ("IZ", "XY")],
            |prop| {
                prop.sxxdg(&pair);
            },
        );
        assert_gate_table(
            "SYY",
            &[("XI", "ZY"), ("IX", "YZ"), ("ZI", "XY"), ("IZ", "YX")],
            |prop| {
                prop.syy(&pair);
            },
        );
        assert_gate_table(
            "SYYdg",
            &[("XI", "ZY"), ("IX", "YZ"), ("ZI", "XY"), ("IZ", "YX")],
            |prop| {
                prop.syydg(&pair);
            },
        );
        assert_gate_table(
            "SZZ",
            &[("XI", "YZ"), ("IX", "ZY"), ("ZI", "ZI"), ("IZ", "IZ")],
            |prop| {
                prop.szz(&pair);
            },
        );
        assert_gate_table(
            "SZZdg",
            &[("XI", "YZ"), ("IX", "ZY"), ("ZI", "ZI"), ("IZ", "IZ")],
            |prop| {
                prop.szzdg(&pair);
            },
        );
        assert_gate_table(
            "SWAP",
            &[("XI", "IX"), ("IX", "XI"), ("ZI", "IZ"), ("IZ", "ZI")],
            |prop| {
                prop.swap(&pair);
            },
        );
    }

    #[test]
    fn test_direct_clifford_gates_match_matrix_oracle_for_all_paulis() {
        let q0 = QubitId(0);
        let q1 = QubitId(1);
        let pair = [(q0, q1)];

        assert_gate_matches_matrix_oracle("CX", CliffordMatrixGate::CX, 2, |prop| {
            prop.cx(&pair);
        });
        assert_gate_matches_matrix_oracle("SZdg", CliffordMatrixGate::SZdg, 1, |prop| {
            prop.szdg(&[q0]);
        });
        assert_gate_matches_matrix_oracle("F", CliffordMatrixGate::F, 1, |prop| {
            prop.f(&[q0]);
        });
        assert_gate_matches_matrix_oracle("Fdg", CliffordMatrixGate::Fdg, 1, |prop| {
            prop.fdg(&[q0]);
        });
        assert_gate_matches_matrix_oracle("SX", CliffordMatrixGate::SX, 1, |prop| {
            prop.sx(&[q0]);
        });
        assert_gate_matches_matrix_oracle("SXdg", CliffordMatrixGate::SXdg, 1, |prop| {
            prop.sxdg(&[q0]);
        });
        assert_gate_matches_matrix_oracle("SY", CliffordMatrixGate::SY, 1, |prop| {
            prop.sy(&[q0]);
        });
        assert_gate_matches_matrix_oracle("SYdg", CliffordMatrixGate::SYdg, 1, |prop| {
            prop.sydg(&[q0]);
        });
        assert_gate_matches_matrix_oracle("CY", CliffordMatrixGate::CY, 2, |prop| {
            prop.cy(&pair);
        });
        assert_gate_matches_matrix_oracle("CZ", CliffordMatrixGate::CZ, 2, |prop| {
            prop.cz(&pair);
        });
        assert_gate_matches_matrix_oracle("SXX", CliffordMatrixGate::SXX, 2, |prop| {
            prop.sxx(&pair);
        });
        assert_gate_matches_matrix_oracle("SXXdg", CliffordMatrixGate::SXXdg, 2, |prop| {
            prop.sxxdg(&pair);
        });
        assert_gate_matches_matrix_oracle("SYY", CliffordMatrixGate::SYY, 2, |prop| {
            prop.syy(&pair);
        });
        assert_gate_matches_matrix_oracle("SYYdg", CliffordMatrixGate::SYYdg, 2, |prop| {
            prop.syydg(&pair);
        });
        assert_gate_matches_matrix_oracle("SZZ", CliffordMatrixGate::SZZ, 2, |prop| {
            prop.szz(&pair);
        });
        assert_gate_matches_matrix_oracle("SZZdg", CliffordMatrixGate::SZZdg, 2, |prop| {
            prop.szzdg(&pair);
        });
        assert_gate_matches_matrix_oracle("SWAP", CliffordMatrixGate::SWAP, 2, |prop| {
            prop.swap(&pair);
        });
    }

    #[test]
    fn test_two_qubit_gates_reversed_pair_matches_matrix_oracle() {
        assert_reversed_pair_matches_matrix_oracle("CX", CliffordMatrixGate::CX, |prop, pairs| {
            prop.cx(pairs);
        });
        assert_reversed_pair_matches_matrix_oracle("CY", CliffordMatrixGate::CY, |prop, pairs| {
            prop.cy(pairs);
        });
        assert_reversed_pair_matches_matrix_oracle("CZ", CliffordMatrixGate::CZ, |prop, pairs| {
            prop.cz(pairs);
        });
        assert_reversed_pair_matches_matrix_oracle(
            "SXX",
            CliffordMatrixGate::SXX,
            |prop, pairs| {
                prop.sxx(pairs);
            },
        );
        assert_reversed_pair_matches_matrix_oracle(
            "SXXdg",
            CliffordMatrixGate::SXXdg,
            |prop, pairs| {
                prop.sxxdg(pairs);
            },
        );
        assert_reversed_pair_matches_matrix_oracle(
            "SYY",
            CliffordMatrixGate::SYY,
            |prop, pairs| {
                prop.syy(pairs);
            },
        );
        assert_reversed_pair_matches_matrix_oracle(
            "SYYdg",
            CliffordMatrixGate::SYYdg,
            |prop, pairs| {
                prop.syydg(pairs);
            },
        );
        assert_reversed_pair_matches_matrix_oracle(
            "SZZ",
            CliffordMatrixGate::SZZ,
            |prop, pairs| {
                prop.szz(pairs);
            },
        );
        assert_reversed_pair_matches_matrix_oracle(
            "SZZdg",
            CliffordMatrixGate::SZZdg,
            |prop, pairs| {
                prop.szzdg(pairs);
            },
        );
        assert_reversed_pair_matches_matrix_oracle(
            "SWAP",
            CliffordMatrixGate::SWAP,
            |prop, pairs| {
                prop.swap(pairs);
            },
        );
    }

    #[test]
    fn test_two_qubit_gate_batches_match_sequential_pairs() {
        assert_two_pair_batch_matches_sequential("CX", |prop, pairs| {
            prop.cx(pairs);
        });
        assert_two_pair_batch_matches_sequential("CY", |prop, pairs| {
            prop.cy(pairs);
        });
        assert_two_pair_batch_matches_sequential("CZ", |prop, pairs| {
            prop.cz(pairs);
        });
        assert_two_pair_batch_matches_sequential("SXX", |prop, pairs| {
            prop.sxx(pairs);
        });
        assert_two_pair_batch_matches_sequential("SXXdg", |prop, pairs| {
            prop.sxxdg(pairs);
        });
        assert_two_pair_batch_matches_sequential("SYY", |prop, pairs| {
            prop.syy(pairs);
        });
        assert_two_pair_batch_matches_sequential("SYYdg", |prop, pairs| {
            prop.syydg(pairs);
        });
        assert_two_pair_batch_matches_sequential("SZZ", |prop, pairs| {
            prop.szz(pairs);
        });
        assert_two_pair_batch_matches_sequential("SZZdg", |prop, pairs| {
            prop.szzdg(pairs);
        });
        assert_two_pair_batch_matches_sequential("SWAP", |prop, pairs| {
            prop.swap(pairs);
        });
    }
}
