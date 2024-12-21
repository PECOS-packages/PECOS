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

use super::quantum_simulator_state::QuantumSimulatorState;
use pecos_core::IndexableElement;

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
/// This trait provides:
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
/// - G2 (a two-qubit Clifford)
///
/// ## Measurements and Preparations
/// - Measurements in X, Y, Z bases (including ± variants)
/// - State preparations in X, Y, Z bases (including ± variants)
///
/// # Type Parameters
/// - `T`: An indexable element type that can convert between qubit indices and usizes
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
/// - All measurements return (outcome, deterministic)
/// - outcome: true = +1 eigenstate, false = -1 eigenstate
/// - deterministic: true if state was already in an eigenstate
///
/// # Examples
/// ```rust
/// use pecos_qsim::{CliffordGateable, StdSparseStab};
/// let mut sim = StdSparseStab::new(2);
///
/// // Create Bell state
/// sim.h(0);
/// sim.cx(0, 1);
///
/// // Measure in Z basis
/// let outcome = sim.mz(0);
/// ```
///
/// # Required Implementations
/// When implementing this trait, the following methods must be provided at minimum:
/// - `mz()`: Z-basis measurement
/// - `x()`: Pauli X gate
/// - `y()`: Pauli Y gate
/// - `z()`: Pauli Z gate
/// - `h()`: Hadamard gate
/// - `sz()`: Square root of Z gate
/// - `cx()`: Controlled-NOT gate
///
/// All other operations have default implementations in terms of these basic gates.
/// Implementors may override any default implementation for efficiency.
///
/// # References
/// - Gottesman, "The Heisenberg Representation of Quantum Computers"
///   <https://arxiv.org/abs/quant-ph/9807006>
#[expect(clippy::min_ident_chars)]
pub trait CliffordGateable<T: IndexableElement>: QuantumSimulatorState {
    /// Identity on qubit q. X -> X, Z -> Z
    #[inline]
    fn identity(&mut self, _q: T) -> &mut Self {
        self
    }

    /// Applies a Pauli X (NOT) gate to the specified qubit.
    ///
    /// The X gate is equivalent to a classical NOT operation in the computational basis.
    /// It transforms the Pauli operators as follows:
    /// ```text
    /// X → X
    /// Y → -Y
    /// Z → -Z
    /// ```
    ///
    /// # Arguments
    /// * `q` - The target qubit
    ///
    /// # Examples
    /// ```rust
    /// use pecos_qsim::{CliffordGateable, StdSparseStab};
    /// let mut sim = StdSparseStab::new(1);
    /// sim.x(0); // Apply X gate to qubit 0
    /// ```
    #[inline]
    fn x(&mut self, q: T) -> &mut Self {
        self.h(q).z(q).h(q)
    }

    /// Applies a Pauli Y gate to the specified qubit.
    ///
    /// The Y gate is a rotation by π radians around the Y axis of the Bloch sphere.
    /// It transforms the Pauli operators as follows:
    /// ```text
    /// X → -X
    /// Y → Y
    /// Z → -Z
    /// ```
    ///
    /// # Arguments
    /// * `q` - The target qubit
    ///
    /// # Examples
    /// ```rust
    /// use pecos_qsim::{CliffordGateable, StdSparseStab};
    /// let mut sim = StdSparseStab::new(1);
    /// sim.y(0); // Apply Y gate to qubit 0
    /// ```
    #[inline]
    fn y(&mut self, q: T) -> &mut Self {
        self.z(q).x(q)
    }

    /// Applies a Pauli Z gate to the specified qubit.
    ///
    /// The Z gate applies a phase flip in the computational basis.
    /// It transforms the Pauli operators as follows:
    /// ```text
    /// X → -X
    /// Y → -Y
    /// Z → Z
    /// ```
    ///
    /// # Arguments
    /// * `q` - The target qubit
    ///
    /// # Examples
    /// ```rust
    /// use pecos_qsim::{CliffordGateable, StdSparseStab};
    /// let mut sim = StdSparseStab::new(1);
    /// sim.z(0); // Apply X gate to qubit 0
    /// ```
    #[inline]
    fn z(&mut self, q: T) -> &mut Self {
        self.sz(q).sz(q);
        self
    }

    /// Sqrt of X gate.
    ///     X -> X
    ///     Z -> -iW = -Y
    ///     W -> -iZ
    ///     Y -> Z
    #[inline]
    fn sx(&mut self, q: T) -> &mut Self {
        self.h(q);
        self.sz(q);
        self.h(q);
        self
    }

    /// Adjoint of Sqrt X gate.
    ///     X -> X
    ///     Z -> iW = Y
    ///     W -> iZ
    ///     Y -> -Z
    #[inline]
    fn sxdg(&mut self, q: T) -> &mut Self {
        self.h(q);
        self.szdg(q);
        self.h(q);
        self
    }

    /// Applies a square root of Y gate to the specified qubit.
    ///
    /// The SY gate is equivalent to a rotation by π/2 radians around the Y axis
    /// of the Bloch sphere. It transforms the Pauli operators as follows:
    /// ```text
    /// X → -Z
    /// Z → X
    /// Y → Y
    /// ```
    ///
    /// # Arguments
    /// * `q` - The target qubit
    ///
    /// # Mathematical Details
    /// The SY gate has the following matrix representation:
    /// ```text
    /// SY = 1/√2 [1  -1]
    ///          [1   1]
    /// ```
    ///
    /// # Examples
    /// ```rust
    /// use pecos_qsim::{CliffordGateable, StdSparseStab};
    /// let mut sim = StdSparseStab::new(1);
    /// sim.sy(0); // Apply square root of Y gate to qubit 0
    /// ```
    ///
    /// # Panics
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    fn sy(&mut self, q: T) -> &mut Self {
        self.h(q);
        self.x(q);
        self
    }

    /// Adjoint of sqrt of Y gate.
    ///     X -> Z
    ///     Z -> -X
    ///     W -> W
    ///     Y -> Y
    /// # Panics
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    fn sydg(&mut self, q: T) -> &mut Self {
        self.x(q);
        self.h(q);
        self
    }

    /// Applies a square root of Z gate (S or SZ gate) to the specified qubit.
    ///
    /// The SZ gate is equivalent to a rotation by π/2 radians around the Z axis
    /// of the Bloch sphere. It transforms the Pauli operators as follows:
    /// ```text
    /// X → Y
    /// Y → -X
    /// Z → Z
    /// ```
    ///
    /// # Arguments
    /// * `q` - The target qubit
    ///
    /// # Mathematical Details
    /// The S gate has the following matrix representation:
    /// ```text
    /// S = [1  0]
    ///     [0  i]
    /// ```
    fn sz(&mut self, q: T) -> &mut Self;

    /// Adjoint of Sqrt of Z gate. X -> ..., Z -> ...
    ///     X -> -iW = -Y
    ///     Z -> Z
    ///     W -> -iX
    ///     Y -> X
    #[inline]
    fn szdg(&mut self, q: T) -> &mut Self {
        self.z(q);
        self.sz(q);
        self
    }

    /// Applies a Hadamard gate to the specified qubit.
    ///
    /// The Hadamard gate creates an equal superposition of the computational basis
    /// states and is fundamental to many quantum algorithms. It transforms the
    /// Pauli operators as follows:
    /// ```text
    /// X → Z
    /// Y → -Y
    /// Z → X
    /// ```
    ///
    /// # Arguments
    /// * `q` - The target qubit
    ///
    /// # Mathematical Details
    /// The Hadamard gate has the following matrix representation:
    /// ```text
    /// H = 1/√2 [1   1]
    ///          [1  -1]
    /// ```
    fn h(&mut self, q: T) -> &mut Self;

    /// Applies a variant of the Hadamard gate (H2) to the specified qubit.
    ///
    /// H2 is part of the family of Hadamard-like gates that map X to Z and Z to X
    /// with different sign combinations. It transforms the Pauli operators as follows:
    /// ```text
    /// X → -Z
    /// Z → -X
    /// Y → -Y
    /// ```
    ///
    /// # Arguments
    /// * `q` - The target qubit
    ///
    /// # Notes
    /// - H2 is equivalent to the composition SY • Z  # TODO: Verify
    /// - H2 differs from H by introducing additional minus signs
    #[inline]
    fn h2(&mut self, q: T) -> &mut Self {
        self.sy(q);
        self.z(q);
        self
    }

    /// X -> Y, Z -> -Z, Y -> X
    #[inline]
    fn h3(&mut self, q: T) -> &mut Self {
        self.sz(q);
        self.y(q);
        self
    }

    /// X -> -Y, Z -> -Z, Y -> -X
    #[inline]
    fn h4(&mut self, q: T) -> &mut Self {
        self.sz(q);
        self.x(q);
        self
    }

    /// X -> -X, Z -> Y, Y -> Z
    #[inline]
    fn h5(&mut self, q: T) -> &mut Self {
        self.sx(q);
        self.z(q);
        self
    }

    /// X -> -X, Z -> -Y, Y -> -Z
    #[inline]
    fn h6(&mut self, q: T) -> &mut Self {
        self.sx(q);
        self.y(q);
        self
    }

    /// Applies a Face gate (F) to the specified qubit.
    ///
    /// The F gate is a member of the Clifford group that cyclically permutes
    /// the Pauli operators. It transforms them as follows:
    /// ```text
    /// X → Y
    /// Y → Z
    /// Z → X
    /// ```
    ///
    /// # Arguments
    /// * `q` - The target qubit
    ///
    /// # Mathematical Details
    /// The F gate can be implemented as F = SX • SZ  # TODO: verify
    ///
    /// # Examples
    /// ```rust
    /// use pecos_qsim::{CliffordGateable, StdSparseStab};
    /// let mut sim = StdSparseStab::new(1);
    /// sim.f(0); // Apply F gate to qubit 0
    /// ```
    #[inline]
    fn f(&mut self, q: T) -> &mut Self {
        self.sx(q).sz(q)
    }

    /// X -> Z, Z -> Y, Y -> X
    #[inline]
    fn fdg(&mut self, q: T) -> &mut Self {
        self.szdg(q).sxdg(q)
    }

    /// X -> -Z, Z -> Y, Y -> -X
    #[inline]
    fn f2(&mut self, q: T) -> &mut Self {
        self.sxdg(q).sy(q)
    }

    /// X -> -Y, Z -> -X, Y -> Z
    #[inline]
    fn f2dg(&mut self, q: T) -> &mut Self {
        self.sydg(q).sx(q)
    }

    /// X -> Y, Z -> -X, Y -> -Z
    #[inline]
    fn f3(&mut self, q: T) -> &mut Self {
        self.sxdg(q).sz(q)
    }

    /// X -> -Z, Z -> -Y, Y -> X
    #[inline]
    fn f3dg(&mut self, q: T) -> &mut Self {
        self.szdg(q).sx(q)
    }

    /// X -> Z, Z -> -Y, Y -> -X
    #[inline]
    fn f4(&mut self, q: T) -> &mut Self {
        self.sz(q).sx(q)
    }

    /// X -> -Y, Z -> X, Y -> -Z
    #[inline]
    fn f4dg(&mut self, q: T) -> &mut Self {
        self.sxdg(q).szdg(q)
    }

    /// Performs a controlled-NOT operation between two qubits.
    ///
    /// The operation performs:
    /// ```text
    /// |0⟩|b⟩ → |0⟩|b⟩
    /// |1⟩|b⟩ → |1⟩|b⊕1⟩
    /// ```
    ///
    /// In the Heisenberg picture, transforms Pauli operators as:
    /// ```text
    /// IX → IX  (X on target unchanged)
    /// XI → XX  (X on control propagates to target)
    /// IZ → ZZ  (Z on target propagates to control)
    /// ZI → ZI  (Z on control unchanged)
    /// ```
    ///
    /// # Arguments
    /// * `q1` - Control qubit
    /// * `q2` - Target qubit
    fn cx(&mut self, q1: T, q2: T) -> &mut Self;

    /// CY: +IX -> +ZX; +IZ -> +ZZ; +XI -> -XY; +ZI -> +ZI;
    #[inline]
    fn cy(&mut self, q1: T, q2: T) -> &mut Self {
        self.sz(q2).cx(q1, q2).szdg(q2)
    }

    /// CZ: +IX -> +ZX; +IZ -> +IZ; +XI -> +XZ; +ZI -> +ZI;
    #[inline]
    fn cz(&mut self, q1: T, q2: T) -> &mut Self {
        self.h(q2).cx(q1, q2).h(q2)
    }

    /// Performs a square root of XX operation (√XX).
    ///
    /// This is a symmetric two-qubit Clifford that implements:
    /// ```text
    /// SXX = exp(-iπ X⊗X/4)
    /// ```
    ///
    /// Transforms Pauli operators as:
    /// ```text
    /// XI → XI
    /// IX → IX
    /// ZI → -YX
    /// IZ → -XY
    /// ```
    ///
    /// # Arguments
    /// * q1 - First qubit
    /// * q2 - Second qubit
    #[inline]
    fn sxx(&mut self, q1: T, q2: T) -> &mut Self {
        self.sx(q1).sx(q2).sydg(q1).cx(q1, q2).sy(q1)
    }

    /// Performs an adjoint of the square root of XX operation (√XX†).
    ///
    /// This is a symmetric two-qubit Clifford that implements:
    /// ```text
    /// SXXdg = exp(+iπ X⊗X/4)
    /// ```
    ///
    /// Transforms Pauli operators as:
    /// ```text
    /// XI → XI
    /// IX → IX
    /// ZI → YX
    /// IZ → XY
    /// ```
    ///
    /// # Arguments
    /// * `q1` - First qubit
    /// * `q2` - Second qubit
    #[inline]
    fn sxxdg(&mut self, q1: T, q2: T) -> &mut Self {
        self.x(q1).x(q2).sxx(q1, q2)
    }

    /// Performs a square root of YY operation (√YY).
    ///
    /// This is a symmetric two-qubit Clifford that implements:
    /// ```text
    /// SYY = exp(-iπ Y⊗Y/4)
    /// ```
    ///
    /// Transforms Pauli operators as:
    /// ```text
    /// XI → -ZY
    /// IX → -YZ
    /// ZI → XY
    /// IZ → YX
    /// ```
    ///
    /// # Arguments
    /// * `q1` - First qubit
    /// * `q2` - Second qubit
    #[inline]
    fn syy(&mut self, q1: T, q2: T) -> &mut Self {
        self.szdg(q1).szdg(q2).sxx(q1, q2).sz(q1).sz(q2)
    }

    /// Performs an adjoint of the square root of YY operation (√YY†).
    ///
    /// This is a symmetric two-qubit Clifford that implements:
    /// ```text
    /// SYY = exp(+iπ Y⊗Y/4)
    /// ```
    ///
    /// Transforms Pauli operators as:
    /// ```text
    /// XI → ZY
    /// IX → YZ
    /// ZI → -XY
    /// IZ → -YX
    /// ```
    ///
    /// # Arguments
    /// * `q1` - First qubit
    /// * `q2` - Second qubit
    #[inline]
    fn syydg(&mut self, q1: T, q2: T) -> &mut Self {
        self.y(q1).y(q2).syy(q1, q2)
    }

    /// Performs a square root of ZZ operation (√ZZ).
    ///
    /// This is a symmetric two-qubit Clifford that implements:
    /// ```text
    /// SZZ = exp(-iπ Z⊗Z/4)
    /// ```
    ///
    /// Transforms Pauli operators as:
    /// ```text
    /// XI → YZ
    /// IX → ZY
    /// ZI → ZI
    /// IZ → IZ
    /// ```
    ///
    /// # Arguments
    /// * `q1` - First qubit
    /// * `q2` - Second qubit
    #[inline]
    fn szz(&mut self, q1: T, q2: T) -> &mut Self {
        self.sydg(q1).sydg(q2).sxx(q1, q2).sy(q1).sy(q2)
    }

    /// Performs an adjoint of the square root of ZZ operation (√ZZ).
    ///
    /// This is a symmetric two-qubit Clifford that implements:
    /// ```text
    /// SZZdg = exp(+iπ Z⊗Z/4)
    /// ```
    ///
    /// Transforms Pauli operators as:
    /// ```text
    /// XI → -ZY
    /// IX → -YZ TODO: verify
    /// ZI → ZI
    /// IZ → IZ
    /// ```
    ///
    /// # Arguments
    /// * `q1` - First qubit
    /// * `q2` - Second qubit
    #[inline]
    fn szzdg(&mut self, q1: T, q2: T) -> &mut Self {
        self.z(q1).z(q2).szz(q1, q2)
    }

    /// SWAP: +IX -> XI;
    ///       +IZ -> ZI;
    ///       +XI -> IX;
    ///       +ZI -> IZ;
    #[inline]
    fn swap(&mut self, q1: T, q2: T) -> &mut Self {
        self.cx(q1, q2).cx(q2, q1).cx(q1, q2)
    }

    /// Applies the G2 two-qubit Clifford operation.
    ///
    /// G2 is a symmetric two-qubit operation that implements a particular permutation
    /// of single-qubit Paulis. It transforms the Pauli operators as follows:
    /// ```text
    /// XI → IX
    /// IX → XI
    /// ZI → XZ
    /// IZ → ZX
    /// ```
    ///
    /// # Arguments
    /// * `q1` - First qubit
    /// * `q2` - Second qubit
    ///
    /// # Implementation Details
    /// G2 can be implemented as: CZ • (H⊗H) • CZ
    ///
    /// # Examples
    /// ```rust
    /// use pecos_qsim::{CliffordGateable, StdSparseStab};
    /// let mut sim = StdSparseStab::new(2);
    /// sim.g2(0, 1); // Apply G2 operation between qubits 0 and 1
    /// ```
    #[inline]
    fn g2(&mut self, q1: T, q2: T) -> &mut Self {
        self.cz(q1, q2).h(q1).h(q2).cz(q1, q2)
    }

    /// Measurement of the +`X_q` operator.
    #[inline]
    fn mx(&mut self, q: T) -> MeasurementResult {
        // +X -> +Z
        self.h(q);
        let meas = self.mz(q);
        // +Z -> +X
        self.h(q);

        meas
    }

    /// Measurement of the -`X_q` operator.
    #[inline]
    fn mnx(&mut self, q: T) -> MeasurementResult {
        // -X -> +Z
        self.h(q);
        self.x(q);
        let meas = self.mz(q);
        // +Z -> -X
        self.x(q);
        self.h(q);

        meas
    }

    /// Measurement of the +`Y_q` operator.
    /// # Panics
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    fn my(&mut self, q: T) -> MeasurementResult {
        // +Y -> +Z
        self.sx(q);
        let meas = self.mz(q);
        // +Z -> +Y
        self.sxdg(q);

        meas
    }

    /// Measurement of the -`Y_q` operator.
    /// # Panics
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    fn mny(&mut self, q: T) -> MeasurementResult {
        // -Y -> +Z
        self.sxdg(q);
        let meas = self.mz(q);
        // +Z -> -Y
        self.sx(q);

        meas
    }

    /// Performs a measurement in the Z basis on the specified qubit.
    ///
    /// This measurement projects the qubit state onto the +1 or -1 eigenstate
    /// of the Z operator (corresponding to |0⟩ or |1⟩ respectively).
    ///
    /// # Arguments
    /// * `q` - The qubit to measure
    ///
    /// For all measurement operations (mx, my, mz, mnx, mny, mnz):
    ///
    /// # Return Values
    /// Returns a tuple `(outcome, deterministic)` where:
    /// * `outcome`:
    ///   - `true` indicates projection onto the +1 eigenstate
    ///   - `false` indicates projection onto the -1 eigenstate
    /// * `deterministic`:
    ///   - `true` if the state was already in an eigenstate of the measured operator
    ///   - `false` if the measurement result was random (state was in superposition)
    ///
    /// # Examples
    /// ```rust
    /// use pecos_qsim::{CliffordGateable, StdSparseStab};
    /// let mut sim = StdSparseStab::new(1);
    /// let outcome = sim.mz(0);
    /// ```
    fn mz(&mut self, q: T) -> MeasurementResult;

    /// Measurement of the -`Z_q` operator.
    /// # Panics
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    fn mnz(&mut self, q: T) -> MeasurementResult {
        // -Z -> +Z
        self.x(q);
        let meas = self.mz(q);
        // +Z -> -Z
        self.x(q);

        meas
    }

    /// Prepares a qubit in the +1 eigenstate of the X operator.
    ///
    /// Equivalent to preparing |+⟩ = (|0⟩ + |1⟩)/√2
    ///
    /// # Arguments
    /// * `q` - Target qubit
    ///
    /// # Returns
    /// * `(bool, bool)` - (`measurement_outcome`, `was_deterministic`)
    ///
    /// # Panics
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    fn px(&mut self, q: T) -> MeasurementResult {
        let result = self.mx(q);
        if result.outcome {
            self.z(q);
        }
        result
    }

    /// Prepares a qubit in the -1 eigenstate of the X operator.
    ///
    /// Equivalent to preparing |-⟩ = (|0⟩ - |1⟩)/√2
    ///
    /// # Arguments
    /// * `q` - Target qubit
    ///
    /// # Returns
    /// * `(bool, bool)` - (`measurement_outcome`, `was_deterministic`)
    ///
    /// # Panics
    ///
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    fn pnx(&mut self, q: T) -> MeasurementResult {
        let result = self.mnx(q);
        if result.outcome {
            self.z(q);
        }
        result
    }

    /// Preparation of the +`Y_q` operator.
    /// # Panics
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    fn py(&mut self, q: T) -> MeasurementResult {
        let result = self.my(q);
        if result.outcome {
            self.z(q);
        }
        result
    }

    /// Preparation of the -`Y_q` operator.
    /// # Panics
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    fn pny(&mut self, q: T) -> MeasurementResult {
        let result = self.mny(q);
        if result.outcome {
            self.z(q);
        }
        result
    }

    /// Prepares a qubit in the +1 eigenstate of the Z operator.
    ///
    /// Equivalent to preparing |0⟩
    ///
    /// # Arguments
    /// * `q` - Target qubit
    ///
    /// # Returns
    /// * `(bool, bool)` - (`measurement_outcome`, `was_deterministic`)
    ///
    /// # Panics
    ///
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    fn pz(&mut self, q: T) -> MeasurementResult {
        let result = self.mz(q);
        if result.outcome {
            self.x(q);
        }
        result
    }

    /// Prepares a qubit in the -1 eigenstate of the Z operator.
    ///
    /// Equivalent to preparing |1⟩
    ///
    /// # Arguments
    /// * `q` - Target qubit
    ///
    /// # Returns
    /// * `(bool, bool)` - (`measurement_outcome`, `was_deterministic`)
    ///
    /// # Panics
    ///
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    fn pnz(&mut self, q: T) -> MeasurementResult {
        let result = self.mnz(q);
        if result.outcome {
            self.x(q);
        }
        result
    }
}
