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

use super::quantum_simulator::QuantumSimulator;
use pecos_core::QubitId;
use smallvec::SmallVec;

/// Stack-allocated qubit buffer for small batches (up to 8 qubits).
/// Falls back to heap allocation for larger batches.
/// Benchmarks show 5-10% improvement for 1-4 pairs vs Vec.
type QubitBuf = SmallVec<[QubitId; 8]>;

/// Stack-allocated pair buffer for small batches (up to 4 pairs).
type PairBuf = SmallVec<[(QubitId, QubitId); 4]>;

pub struct MeasurementResult {
    pub outcome: bool,
    pub is_deterministic: bool,
}

/// A simulator trait for quantum systems that implement Clifford operations.
///
/// # Overview
/// The Clifford group is a set of quantum operations that map Pauli operators to Pauli operators
/// under conjugation. A Clifford operation C transforms a Pauli operator P as:
/// ```text
/// C P C† = P'
/// ```
/// where P' is another Pauli operator (possibly with a phase ±1 or ±i).
///
/// # Gate Set
///
/// ## Single-qubit gates
/// - Pauli gates (X, Y, Z)
/// - Hadamard (H) and variants (H2-H6)
/// - Phase gates (SX, SY, SZ) and their adjoints
/// - Face (F) gates and variants (F, F2-F4) and their adjoints
///
/// ## Two-qubit gates
/// - CNOT (CX)
/// - Controlled-Y (CY)
/// - Controlled-Z (CZ)
/// - SWAP
/// - √XX, √YY, √ZZ and their adjoints
/// - G (a two-qubit Clifford)
///
/// ## Measurements and Preparations
/// - Measurements in X, Y, Z bases (including ± variants)
/// - State preparations in X, Y, Z bases (including ± variants)
///
/// # Slice-based API
/// All methods take `&[QubitId]` slices, allowing both single-qubit and batch operations:
///
/// - Single-qubit gates: `sim.h(&[QubitId(0)])` or `sim.h(&[QubitId(0), QubitId(1), QubitId(2)])`
/// - Two-qubit gates: `sim.cx(&[(control, target)])` or `sim.cx(&[(c0, t0), (c1, t1)])` for batches
///
/// # Gate Transformations
/// Gates transform Pauli operators according to their Heisenberg representation. For example:
///
/// Hadamard (H):
/// ```text
/// X → Z
/// Z → X
/// Y → -Y
/// ```
///
/// CNOT (with control c and target t):
/// ```text
/// Xc⊗It → Xc⊗Xt
/// Ic⊗Xt → Ic⊗Xt
/// Zc⊗It → Zc⊗It
/// Ic⊗Zt → Zc⊗Zt
/// ```
///
/// # Measurement Semantics
/// - Measurements return a `Vec<MeasurementResult>` containing:
///   - outcome: true for +1 eigenstate, false for -1 eigenstate
///   - deterministic: true if state was already in an eigenstate
///
/// # Examples
/// ```rust
/// use pecos_simulators::{CliffordGateable, SparseStab};
/// use pecos_core::QubitId;
///
/// let mut sim = SparseStab::new(2);
///
/// // Create Bell state
/// sim.h(&[QubitId(0)]).cx(&[(QubitId(0), QubitId(1))]);
///
/// // Measure in Z basis
/// let outcomes = sim.mz(&[QubitId(0)]);
/// ```
///
/// # Required Implementations
/// When implementing this trait, the following methods must be provided:
/// - `sz()`: Square root of Z gate (S or P gate)
/// - `h()`: Hadamard gate
/// - `cx()`: Controlled-NOT gate
/// - `mz()`: Z-basis measurement
///
/// All other operations have default implementations in terms of these basic gates.
/// Implementors may override any default implementation for efficiency.
///
/// # References
/// - Gottesman, "The Heisenberg Representation of Quantum Computers"
///   <https://arxiv.org/abs/quant-ph/9807006>
#[expect(clippy::min_ident_chars)]
pub trait CliffordGateable: QuantumSimulator {
    /// Applies the identity gate (I) to the specified qubits.
    ///
    /// The identity gate leaves the state unchanged.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn identity(&mut self, _qubits: &[QubitId]) -> &mut Self {
        self
    }

    /// Applies a Pauli X (NOT) gate to the specified qubits.
    ///
    /// The X gate is equivalent to a classical NOT operation in the computational basis.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Pauli Transformation
    /// ```text
    /// X → X
    /// Y → -Y
    /// Z → -Z
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// X = [[0, 1],
    ///      [1, 0]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.h(qubits).z(qubits).h(qubits)
    }

    /// Applies a Pauli Y gate to the specified qubits.
    ///
    /// The Y gate is a rotation by π radians around the Y axis of the Bloch sphere.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Pauli Transformation
    /// ```text
    /// X → -X
    /// Y → Y
    /// Z → -Z
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// Y = [[ 0, -i],
    ///      [+i,  0]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.z(qubits).x(qubits)
    }

    /// Applies a Pauli Z gate to the specified qubits.
    ///
    /// The Z gate applies a phase flip in the computational basis.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Pauli Transformation
    /// ```text
    /// X → -X
    /// Y → -Y
    /// Z → Z
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// Z = [[1,  0],
    ///      [0, -1]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.sz(qubits).sz(qubits)
    }

    /// Applies a square root of X (SX) gate to the specified qubits.
    ///
    /// The SX gate is equivalent to a rotation by π/2 radians around the X axis
    /// of the Bloch sphere.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Pauli Transformation
    /// ```text
    /// X → X
    /// Y → -Z
    /// Z → Y
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// SX = 1/2 [[1+i, 1-i],
    ///           [1-i, 1+i]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn sx(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.h(qubits).sz(qubits).h(qubits)
    }

    /// Applies the adjoint (inverse) of the square root of X gate.
    ///
    /// The SX† gate is equivalent to a rotation by -π/2 radians around the X axis
    /// of the Bloch sphere.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Pauli Transformation
    /// ```text
    /// X → X
    /// Y → Z
    /// Z → -Y
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// SX† = 1/2 [[1-i, 1+i],
    ///            [1+i, 1-i]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn sxdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.h(qubits).szdg(qubits).h(qubits)
    }

    /// Applies a square root of Y (SY) gate to the specified qubits.
    ///
    /// The SY gate is equivalent to a rotation by π/2 radians around the Y axis
    /// of the Bloch sphere.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Pauli Transformation
    /// ```text
    /// X → -Z
    /// Y → Y
    /// Z → X
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// SY = 1/√2 [[1,  -1],
    ///            [1,   1]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn sy(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.h(qubits).x(qubits)
    }

    /// Applies the adjoint (inverse) of the square root of Y gate.
    ///
    /// The SY† gate is equivalent to a rotation by -π/2 radians around the Y axis
    /// of the Bloch sphere.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Pauli Transformation
    /// ```text
    /// X → Z
    /// Y → Y
    /// Z → -X
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// SY† = 1/√2 [[ 1,  1],
    ///            [-1,  1]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn sydg(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.x(qubits).h(qubits)
    }

    /// Applies a square root of Z (SZ) gate to the specified qubits.
    ///
    /// The SZ gate (also known as the S gate) is equivalent to a rotation by π/2 radians
    /// around the Z axis of the Bloch sphere.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Pauli Transformation
    /// ```text
    /// X → Y
    /// Y → -X
    /// Z → Z
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// SZ = [[1, 0],
    ///       [0, i]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self;

    /// Applies the adjoint (inverse) of the square root of Z gate.
    ///
    /// The SZ† gate is equivalent to a rotation by -π/2 radians around the Z axis
    /// of the Bloch sphere.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Pauli Transformation
    /// ```text
    /// X → -Y
    /// Y → X
    /// Z → Z
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// SZ† = [[1,  0],
    ///        [0, -i]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn szdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.z(qubits).sz(qubits)
    }

    /// Applies the Hadamard gate (H or H1) to the specified qubits.
    ///
    /// The Hadamard gate creates an equal superposition of basis states and is fundamental
    /// to many quantum algorithms.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Pauli Transformation
    /// ```text
    /// X → Z
    /// Y → -Y
    /// Z → X
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// H = 1/√2 [[1,  1],
    ///           [1, -1]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self;

    /// Applies the H2 variant of the Hadamard gate to the specified qubits.
    ///
    /// H2 transforms between complementary measurement bases with an additional negative sign.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Pauli Transformation
    /// ```text
    /// X → -Z
    /// Y → -Y
    /// Z → -X
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// H2 = 1/√2 [[ 1, -1],
    ///            [-1,  1]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn h2(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.sy(qubits).z(qubits)
    }

    /// Applies the H3 variant of the Hadamard gate to the specified qubits.
    ///
    /// H3 performs a basis transformation in the XY plane of the Bloch sphere.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Pauli Transformation
    /// ```text
    /// X → Y
    /// Y → X
    /// Z → -Z
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// H3 = 1/√2 [[1,  i],
    ///            [i,  1]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn h3(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.sz(qubits).y(qubits)
    }

    /// Applies the H4 variant of the Hadamard gate to the specified qubits.
    ///
    /// H4 combines an XY-plane rotation with negative signs.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Pauli Transformation
    /// ```text
    /// X → -Y
    /// Y → -X
    /// Z → -Z
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// H4 = 1/√2 [[ 1, -i],
    ///            [-i,  1]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn h4(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.sz(qubits).x(qubits)
    }

    /// Applies the H5 variant of the Hadamard gate to the specified qubits.
    ///
    /// H5 performs a basis transformation in the YZ plane of the Bloch sphere.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Pauli Transformation
    /// ```text
    /// X → -X
    /// Y → Z
    /// Z → Y
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// H5 = 1/√2 [[-1,  1],
    ///            [ 1,  1]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn h5(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.sx(qubits).z(qubits)
    }

    /// Applies the H6 variant of the Hadamard gate to the specified qubits.
    ///
    /// H6 combines a YZ-plane rotation with negative signs.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Pauli Transformation
    /// ```text
    /// X → -X
    /// Y → -Z
    /// Z → -Y
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// H6 = 1/√2 [[-1, -1],
    ///            [-1,  1]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn h6(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.sx(qubits).y(qubits)
    }

    /// Applies the Face gate (F or F1) to the specified qubits.
    ///
    /// The Face gate performs a cyclic permutation of the Pauli operators.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Pauli Transformation
    /// ```text
    /// X → Y
    /// Y → Z
    /// Z → X
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// F = 1/√2 [[1,  -i],
    ///           [i,   1]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn f(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.sx(qubits).sz(qubits)
    }

    /// Applies the adjoint of the Face gate (F† or F1†) to the specified qubits.
    ///
    /// F† performs a counter-clockwise cyclic permutation of the Pauli operators.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Pauli Transformation
    /// ```text
    /// X → Z
    /// Y → X
    /// Z → Y
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// F† = 1/√2 [[1,   i],
    ///            [-i,  1]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn fdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.szdg(qubits).sxdg(qubits)
    }

    /// Applies the F2 variant of the Face gate to the specified qubits.
    ///
    /// F2 performs a cyclic permutation of the Pauli operators with one negative sign.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Pauli Transformation
    /// ```text
    /// X → -Z
    /// Y → -X
    /// Z → Y
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// F2 = 1/√2 [[-1,  -i],
    ///            [-i,   1]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn f2(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.sxdg(qubits).sy(qubits)
    }

    /// Applies the adjoint of the F2 gate (F2†) to the specified qubits.
    ///
    /// F2† performs a cyclic permutation with one negative sign in reverse.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Pauli Transformation
    /// ```text
    /// X → -Y
    /// Y → Z
    /// Z → -X
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// F2† = 1/√2 [[-1,   i],
    ///            [ i,   1]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn f2dg(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.sydg(qubits).sx(qubits)
    }

    /// Applies the F3 variant of the Face gate to the specified qubits.
    ///
    /// F3 performs a cyclic permutation with two negative signs.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Pauli Transformation
    /// ```text
    /// X → Y
    /// Y → -Z
    /// Z → -X
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// F3 = 1/√2 [[ 1,  -i],
    ///            [-i,  -1]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn f3(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.sxdg(qubits).sz(qubits)
    }

    /// Applies the adjoint of the F3 gate (F3†) to the specified qubits.
    ///
    /// F3† performs a cyclic permutation with two negative signs in reverse.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Pauli Transformation
    /// ```text
    /// X → -Z
    /// Y → X
    /// Z → -Y
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// F3† = 1/√2 [[ 1,   i],
    ///            [ i,  -1]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn f3dg(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.szdg(qubits).sx(qubits)
    }

    /// Applies the F4 variant of the Face gate to the specified qubits.
    ///
    /// F4 performs a cyclic permutation with three negative signs.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Pauli Transformation
    /// ```text
    /// X → Z
    /// Y → -X
    /// Z → -Y
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// F4 = 1/√2 [[-i,  -1],
    ///            [ 1,  -i]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn f4(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.sz(qubits).sx(qubits)
    }

    /// Applies the adjoint of the F4 gate (F4†) to the specified qubits.
    ///
    /// F4† performs a reverse cyclic permutation of the Pauli operators.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Pauli Transformation
    /// ```text
    /// X → -Y
    /// Y → -Z
    /// Z → X
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// F4† = 1/√2 [[ i,   1],
    ///            [-1,   i]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn f4dg(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.sxdg(qubits).szdg(qubits)
    }

    /// Applies a controlled-X (CNOT) operation between qubit pairs.
    ///
    /// The CX gate flips the target qubit if the control qubit is in state |1⟩.
    ///
    /// # Arguments
    /// * `pairs` - Pairs of (control, target) qubit indices: `[(c0, t0), (c1, t1), ...]`
    ///
    /// CX = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ X
    ///
    /// # Pauli Transformation
    /// ```text
    /// XI → XX
    /// IX → IX
    /// ZI → ZI
    /// IZ → ZZ
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// CX = [[1, 0, 0, 0],
    ///       [0, 1, 0, 0],
    ///       [0, 0, 0, 1],
    ///       [0, 0, 1, 0]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self;

    /// Applies a controlled-Y operation between qubit pairs.
    ///
    /// The CY gate applies a Y operation on the target qubit if the control qubit is in state |1⟩.
    ///
    /// # Arguments
    /// * `pairs` - Pairs of (control, target) qubit indices: `[(c0, t0), (c1, t1), ...]`
    ///
    /// CY = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ Y
    ///
    /// # Pauli Transformation
    /// ```text
    /// XI → XY
    /// IX → ZX
    /// ZI → ZI
    /// IZ → ZZ
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// CY = [[1,  0,  0,  0],
    ///       [0,  0,  0, -i],
    ///       [0,  0,  1,  0],
    ///       [0, +i,  0,  0]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn cy(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let targets: QubitBuf = pairs.iter().map(|&(_, q2)| q2).collect();
        // CY = (I ⊗ S†) CX (I ⊗ S) because S†XS = Y
        // Circuit order: S† on target, then CX, then S on target
        self.szdg(&targets).cx(pairs).sz(&targets)
    }

    /// Applies a controlled-Z operation between qubit pairs.
    ///
    /// The CZ gate applies a phase of -1 when both qubits are in state |1⟩.
    ///
    /// # Arguments
    /// * `pairs` - Pairs of qubit indices: `[(q0, q1), (q2, q3), ...]`
    ///
    /// CZ = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ Z
    ///
    /// # Pauli Transformation
    /// ```text
    /// XI → XZ
    /// IX → ZX
    /// ZI → ZI
    /// IZ → IZ
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// CZ = [[1,  0,  0,  0],
    ///       [0,  1,  0,  0],
    ///       [0,  0,  1,  0],
    ///       [0,  0,  0, -1]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn cz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let targets: QubitBuf = pairs.iter().map(|&(_, q2)| q2).collect();
        self.h(&targets).cx(pairs).h(&targets)
    }

    /// Applies a square root of XX (SXX) operation between qubit pairs.
    ///
    /// The SXX gate implements evolution under XX coupling for time π/4.
    ///
    /// # Arguments
    /// * `pairs` - Pairs of qubit indices: `[(q0, q1), (q2, q3), ...]`
    ///
    /// # Pauli Transformation
    /// ```text
    /// XI → XI
    /// IX → IX
    /// ZI → -YX
    /// IZ → -XY
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// SXX = 1/√2 [[1,  0,  0, -i],
    ///             [0,  1, -i,  0],
    ///             [0, -i,  1,  0],
    ///             [-i, 0,  0,  1]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn sxx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let q1s: QubitBuf = pairs.iter().map(|&(q1, _)| q1).collect();
        let q2s: QubitBuf = pairs.iter().map(|&(_, q2)| q2).collect();
        self.sx(&q1s).sx(&q2s).sydg(&q1s).cx(pairs).sy(&q1s)
    }

    /// Applies the adjoint of the square root of XX operation.
    ///
    /// The SXX† gate implements reverse evolution under XX coupling.
    ///
    /// # Arguments
    /// * `pairs` - Pairs of qubit indices: `[(q0, q1), (q2, q3), ...]`
    ///
    /// # Pauli Transformation
    /// ```text
    /// XI → XI
    /// IX → IX
    /// ZI → YX
    /// IZ → XY
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// SXX† = 1/√2 [[1,  0,  0,  i],
    ///              [0,  1,  i,  0],
    ///              [0,  i,  1,  0],
    ///              [i,  0,  0,  1]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn sxxdg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let q1s: QubitBuf = pairs.iter().map(|&(q1, _)| q1).collect();
        let q2s: QubitBuf = pairs.iter().map(|&(_, q2)| q2).collect();
        self.x(&q1s).x(&q2s).sxx(pairs)
    }

    /// Applies a square root of YY (SYY) operation between qubit pairs.
    ///
    /// The SYY gate implements evolution under YY coupling for time π/4.
    ///
    /// # Arguments
    /// * `pairs` - Pairs of qubit indices: `[(q0, q1), (q2, q3), ...]`
    ///
    /// # Pauli Transformation
    /// ```text
    /// XI → -ZY
    /// IX → -YZ
    /// ZI → XY
    /// IZ → YX
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// SYY = 1/√2 [[1,  0,   0, -i],
    ///             [0, -i,   1,  0],
    ///             [0,  1,  -i,  0],
    ///             [-i, 0,   0,  1]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn syy(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let q1s: QubitBuf = pairs.iter().map(|&(q1, _)| q1).collect();
        let q2s: QubitBuf = pairs.iter().map(|&(_, q2)| q2).collect();
        self.szdg(&q1s).szdg(&q2s).sxx(pairs).sz(&q1s).sz(&q2s)
    }

    /// Applies the adjoint of the square root of YY operation.
    ///
    /// The SYY† gate implements reverse evolution under YY coupling.
    ///
    /// # Arguments
    /// * `pairs` - Pairs of qubit indices: `[(q0, q1), (q2, q3), ...]`
    ///
    /// # Pauli Transformation
    /// ```text
    /// XI → ZY
    /// IX → YZ
    /// ZI → -XY
    /// IZ → -YX
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// SYY† = 1/√2 [[1,  0,  0,  i],
    ///              [0,  i,  1,  0],
    ///              [0,  1,  i,  0],
    ///              [i,  0,  0,  1]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn syydg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let q1s: QubitBuf = pairs.iter().map(|&(q1, _)| q1).collect();
        let q2s: QubitBuf = pairs.iter().map(|&(_, q2)| q2).collect();
        self.y(&q1s).y(&q2s).syy(pairs)
    }

    /// Applies a square root of ZZ (SZZ) operation between qubit pairs.
    ///
    /// The SZZ gate implements evolution under ZZ coupling for time π/4.
    ///
    /// # Arguments
    /// * `pairs` - Pairs of qubit indices: `[(q0, q1), (q2, q3), ...]`
    ///
    /// # Pauli Transformation
    /// ```text
    /// XI → YZ
    /// IX → ZY
    /// ZI → ZI
    /// IZ → IZ
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// SZZ = e^(-iπ/4) [[1,  0,  0,  0],
    ///                  [0, -i,  0,  0],
    ///                  [0,  0, -i,  0],
    ///                  [0,  0,  0,  1]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn szz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let q1s: QubitBuf = pairs.iter().map(|&(q1, _)| q1).collect();
        let q2s: QubitBuf = pairs.iter().map(|&(_, q2)| q2).collect();
        self.h(&q1s).h(&q2s).sxx(pairs).h(&q1s).h(&q2s)
    }

    /// Applies the adjoint of the square root of ZZ operation.
    ///
    /// The SZZ† gate implements reverse evolution under ZZ coupling.
    ///
    /// # Arguments
    /// * `pairs` - Pairs of qubit indices: `[(q0, q1), (q2, q3), ...]`
    ///
    /// # Pauli Transformation
    /// ```text
    /// XI → -YZ
    /// IX → -ZY
    /// ZI → ZI
    /// IZ → IZ
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// SZZ† = e^(iπ/4) [[1,  0,  0,  0],
    ///                  [0,  i,  0,  0],
    ///                  [0,  0,  i,  0],
    ///                  [0,  0,  0,  1]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn szzdg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let q1s: QubitBuf = pairs.iter().map(|&(q1, _)| q1).collect();
        let q2s: QubitBuf = pairs.iter().map(|&(_, q2)| q2).collect();
        self.z(&q1s).z(&q2s).szz(pairs)
    }

    /// Applies the SWAP operation between qubit pairs.
    ///
    /// The SWAP gate exchanges the quantum states of two qubits.
    ///
    /// # Arguments
    /// * `pairs` - Pairs of qubit indices: `[(q0, q1), (q2, q3), ...]`
    ///
    /// # Pauli Transformation
    /// ```text
    /// XI → IX
    /// IX → XI
    /// ZI → IZ
    /// IZ → ZI
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// SWAP = [[1, 0, 0, 0],
    ///         [0, 0, 1, 0],
    ///         [0, 1, 0, 0],
    ///         [0, 0, 0, 1]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn swap(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        // For SWAP, we need to apply cx in both directions
        // SWAP = CX(a,b) CX(b,a) CX(a,b)
        let reversed: PairBuf = pairs.iter().map(|&(a, b)| (b, a)).collect();
        self.cx(pairs).cx(&reversed).cx(pairs)
    }

    /// Applies the iSWAP two-qubit Clifford operation.
    ///
    /// The iSWAP gate swaps states with an additional i phase on the swapped states.
    ///
    /// # Arguments
    /// * `pairs` - Pairs of qubit indices: `[(q0, q1), (q2, q3), ...]`
    ///
    /// # Pauli Transformation
    /// ```text
    /// XI → ZY
    /// IX → YZ
    /// ZI → IZ
    /// IZ → ZI
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// iSWAP = [[1, 0,  0,  0],
    ///          [0, 0,  i,  0],
    ///          [0, i,  0,  0],
    ///          [0, 0,  0,  1]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    fn iswap(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let q1s: QubitBuf = pairs.iter().map(|&(q1, _)| q1).collect();
        let q2s: QubitBuf = pairs.iter().map(|&(_, q2)| q2).collect();
        let reversed: PairBuf = pairs.iter().map(|&(a, b)| (b, a)).collect();
        self.sz(&q1s)
            .sz(&q2s)
            .h(&q1s)
            .cx(pairs)
            .cx(&reversed)
            .h(&q2s)
    }

    /// Applies the G two-qubit Clifford operation.
    ///
    /// G is a symmetric two-qubit operation that implements a particular permutation
    /// of single-qubit Paulis.
    ///
    /// # Arguments
    /// * `pairs` - Pairs of qubit indices: `[(q0, q1), (q2, q3), ...]`
    ///
    /// # Pauli Transformation
    /// ```text
    /// XI → IX
    /// IX → XI
    /// ZI → XZ
    /// IZ → ZX
    /// ```
    ///
    /// # Matrix Representation
    /// ```text
    /// G = 1/2 [[1,  1,  1, -1],
    ///          [1, -1,  1,  1],
    ///          [1,  1, -1,  1],
    ///          [-1, 1,  1,  1]]
    /// ```
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn g(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let q1s: QubitBuf = pairs.iter().map(|&(q1, _)| q1).collect();
        let q2s: QubitBuf = pairs.iter().map(|&(_, q2)| q2).collect();
        self.cz(pairs).h(&q1s).h(&q2s).cz(pairs)
    }

    /// Applies the inverse (dagger) of the iSWAP gate.
    ///
    /// # Pauli Transformation
    /// ```text
    /// XI → -ZY    (vs iSWAP: XI → +ZY)
    /// IX → -YZ    (vs iSWAP: IX → +YZ)
    /// ZI → IZ     (same as iSWAP)
    /// IZ → ZI     (same as iSWAP)
    /// ```
    #[inline]
    fn iswapdg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let q1s: QubitBuf = pairs.iter().map(|&(q1, _)| q1).collect();
        let q2s: QubitBuf = pairs.iter().map(|&(_, q2)| q2).collect();
        let reversed: PairBuf = pairs.iter().map(|&(a, b)| (b, a)).collect();
        self.h(&q2s)
            .cx(&reversed)
            .cx(pairs)
            .h(&q1s)
            .szdg(&q2s)
            .szdg(&q1s)
    }

    /// Applies the dagger of the G gate. G is Hermitian (self-inverse), so Gdg = G.
    #[inline]
    fn gdg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self.g(pairs)
    }

    /// Measures the +X Pauli operator, projecting to the measured eigenstate.
    ///
    /// Projects the state into either the |+⟩ or |-⟩ eigenstate based on the
    /// measurement outcome.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Returns
    /// * `Vec<MeasurementResult>` - One result per qubit containing:
    ///   - `outcome`: true if projected to |-⟩, false if projected to |+⟩
    ///   - `is_deterministic`: true if state was already in an X eigenstate
    #[inline]
    fn mx(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        self.h(qubits);
        let results = self.mz(qubits);
        self.h(qubits);
        results
    }

    /// Measures the -X Pauli operator, projecting to the measured eigenstate.
    ///
    /// Projects the state into either the |+⟩ or |-⟩ eigenstate based on the
    /// measurement outcome.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Returns
    /// * `Vec<MeasurementResult>` - One result per qubit containing:
    ///   - `outcome`: true if projected to |+⟩, false if projected to |-⟩
    ///   - `is_deterministic`: true if state was already in an X eigenstate
    #[inline]
    fn mnx(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        self.h(qubits).x(qubits);
        let results = self.mz(qubits);
        self.x(qubits).h(qubits);
        results
    }

    /// Measures the +Y Pauli operator, projecting to the measured eigenstate.
    ///
    /// Projects the state into either the |+i⟩ or |-i⟩ eigenstate based on the
    /// measurement outcome.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Returns
    /// * `Vec<MeasurementResult>` - One result per qubit containing:
    ///   - `outcome`: true if projected to |-i⟩, false if projected to |+i⟩
    ///   - `is_deterministic`: true if state was already in a Y eigenstate
    #[inline]
    fn my(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        self.sx(qubits);
        let results = self.mz(qubits);
        self.sxdg(qubits);
        results
    }

    /// Measures the -Y Pauli operator, projecting to the measured eigenstate.
    ///
    /// Projects the state into either the |+i⟩ or |-i⟩ eigenstate based on the
    /// measurement outcome.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Returns
    /// * `Vec<MeasurementResult>` - One result per qubit containing:
    ///   - `outcome`: true if projected to |+i⟩, false if projected to |-i⟩
    ///   - `is_deterministic`: true if state was already in a Y eigenstate
    #[inline]
    fn mny(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        self.sxdg(qubits);
        let results = self.mz(qubits);
        self.sx(qubits);
        results
    }

    /// Measures the +Z Pauli operator, projecting to the measured eigenstate.
    ///
    /// Projects the state into either the |0⟩ or |1⟩ eigenstate based on the
    /// measurement outcome. This is the standard computational basis measurement.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Returns
    /// * `Vec<MeasurementResult>` - One result per qubit containing:
    ///   - `outcome`: true if projected to |1⟩, false if projected to |0⟩
    ///   - `is_deterministic`: true if state was already in a Z eigenstate
    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult>;

    /// Measures the -Z Pauli operator, projecting to the measured eigenstate.
    ///
    /// Projects the state into either the |0⟩ or |1⟩ eigenstate based on the
    /// measurement outcome.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Returns
    /// * `Vec<MeasurementResult>` - One result per qubit containing:
    ///   - `outcome`: true if projected to |0⟩, false if projected to |1⟩
    ///   - `is_deterministic`: true if state was already in a Z eigenstate
    #[inline]
    fn mnz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        self.x(qubits);
        let results = self.mz(qubits);
        self.x(qubits);
        results
    }

    /// Prepares qubits in the +1 eigenstate of the +X operator.
    ///
    /// Equivalent to preparing |+X⟩ = |+⟩ = (|0⟩ + |1⟩)/√2.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn px(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.mpx(qubits);
        self
    }

    /// Prepares qubits in the +1 eigenstate of -X.
    ///
    /// Equivalent to preparing |-X⟩ = |-⟩ = (|0⟩ - |1⟩)/√2.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn pnx(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.mpnx(qubits);
        self
    }

    /// Prepares qubits in the +1 eigenstate of +Y.
    ///
    /// Equivalent to preparing |+Y⟩ = |+i⟩ = (|0⟩ + i|1⟩)/√2.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn py(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.mpy(qubits);
        self
    }

    /// Prepares qubits in the +1 eigenstate of -Y.
    ///
    /// Equivalent to preparing |-Y⟩ = |-i⟩ = (|0⟩ - i|1⟩)/√2.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn pny(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.mpny(qubits);
        self
    }

    /// Prepares qubits in the +1 eigenstate of +Z.
    ///
    /// Equivalent to preparing |+Z⟩ = |0⟩.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn pz(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.mpz(qubits);
        self
    }

    /// Prepares qubits in the +1 eigenstate of -Z.
    ///
    /// Equivalent to preparing |-Z⟩ = |1⟩.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Returns
    /// * `&mut Self` - Returns the simulator for method chaining.
    #[inline]
    fn pnz(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.mpnz(qubits);
        self
    }

    /// Both measures +X and prepares the qubits in the |+⟩ state.
    ///
    /// After measurement, this operation always prepares the |+⟩ state regardless of
    /// measurement outcome.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Returns
    /// * `Vec<MeasurementResult>` - One result per qubit containing:
    ///   - `outcome`: true if Z correction was needed
    ///   - `is_deterministic`: true if state was already |+⟩
    #[inline]
    fn mpx(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        let results = self.mx(qubits);
        let corrections: QubitBuf = qubits
            .iter()
            .zip(results.iter())
            .filter(|(_, r)| r.outcome)
            .map(|(&q, _)| q)
            .collect();
        if !corrections.is_empty() {
            self.z(&corrections);
        }
        results
    }

    /// Both measures -X and prepares the qubits in the |-⟩ state.
    ///
    /// After measurement, this operation always prepares the |-⟩ state regardless of
    /// measurement outcome.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Returns
    /// * `Vec<MeasurementResult>` - One result per qubit containing:
    ///   - `outcome`: true if Z correction was needed
    ///   - `is_deterministic`: true if state was already |-⟩
    #[inline]
    fn mpnx(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        let results = self.mnx(qubits);
        let corrections: QubitBuf = qubits
            .iter()
            .zip(results.iter())
            .filter(|(_, r)| r.outcome)
            .map(|(&q, _)| q)
            .collect();
        if !corrections.is_empty() {
            self.z(&corrections);
        }
        results
    }

    /// Both measures +Y and prepares the qubits in the |+i⟩ state.
    ///
    /// After measurement, this operation always prepares the |+i⟩ state regardless of
    /// measurement outcome.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Returns
    /// * `Vec<MeasurementResult>` - One result per qubit containing:
    ///   - `outcome`: true if Z correction was needed
    ///   - `is_deterministic`: true if state was already |+i⟩
    #[inline]
    fn mpy(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        let results = self.my(qubits);
        let corrections: QubitBuf = qubits
            .iter()
            .zip(results.iter())
            .filter(|(_, r)| r.outcome)
            .map(|(&q, _)| q)
            .collect();
        if !corrections.is_empty() {
            self.z(&corrections);
        }
        results
    }

    /// Both measures -Y and prepares the qubits in the |-i⟩ state.
    ///
    /// After measurement, this operation always prepares the |-i⟩ state regardless of
    /// measurement outcome.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Returns
    /// * `Vec<MeasurementResult>` - One result per qubit containing:
    ///   - `outcome`: true if Z correction was needed
    ///   - `is_deterministic`: true if state was already |-i⟩
    #[inline]
    fn mpny(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        let results = self.mny(qubits);
        let corrections: QubitBuf = qubits
            .iter()
            .zip(results.iter())
            .filter(|(_, r)| r.outcome)
            .map(|(&q, _)| q)
            .collect();
        if !corrections.is_empty() {
            self.z(&corrections);
        }
        results
    }

    /// Both measures +Z and prepares the qubits in the |0⟩ state.
    ///
    /// After measurement, this operation always prepares the |0⟩ state regardless of
    /// measurement outcome.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Returns
    /// * `Vec<MeasurementResult>` - One result per qubit containing:
    ///   - `outcome`: true if X correction was needed
    ///   - `is_deterministic`: true if state was already |0⟩
    #[inline]
    fn mpz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        let results = self.mz(qubits);
        let corrections: QubitBuf = qubits
            .iter()
            .zip(results.iter())
            .filter(|(_, r)| r.outcome)
            .map(|(&q, _)| q)
            .collect();
        if !corrections.is_empty() {
            self.x(&corrections);
        }
        results
    }

    /// Both measures -Z and prepares the qubits in the |1⟩ state.
    ///
    /// After measurement, this operation always prepares the |1⟩ state regardless of
    /// measurement outcome.
    ///
    /// # Arguments
    /// * `qubits` - Target qubit indices.
    ///
    /// # Returns
    /// * `Vec<MeasurementResult>` - One result per qubit containing:
    ///   - `outcome`: true if X correction was needed
    ///   - `is_deterministic`: true if state was already |1⟩
    #[inline]
    fn mpnz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        let results = self.mnz(qubits);
        let corrections: QubitBuf = qubits
            .iter()
            .zip(results.iter())
            .filter(|(_, r)| r.outcome)
            .map(|(&q, _)| q)
            .collect();
        if !corrections.is_empty() {
            self.x(&corrections);
        }
        results
    }
}
