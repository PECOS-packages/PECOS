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
use pecos_core::IndexableElement;

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
/// - T: An indexable element type that can convert between qubit indices and usizes
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
/// Xc → Xc⊗Xt
/// Xt → Xt
/// Zc → Zc
/// Zt → Zc⊗Zt
/// ```
///
/// # Measurement Semantics
/// - All measurements return (outcome, deterministic)
/// - outcome: true = +1 eigenstate, false = -1 eigenstate
/// - deterministic: true if state was already in an eigenstate
///
/// # Examples
/// ```rust
/// use pecos_core::VecSet;
/// use pecos_qsim::{CliffordSimulator, SparseStab};
///
/// let mut sim = SparseStab::<VecSet<u32>, u32>::new(2);
///
/// // Create Bell state
/// sim.h(0);
/// sim.cx(0, 1);
///
/// // Measure in Z basis
/// let (outcome, deterministic) = sim.mz(0);
/// ```
///
/// # Implementation Notes
/// - Most multi-qubit gates are implemented in terms of simpler gates
/// - The trait provides default implementations where possible
/// - Implementors must provide: `mz()`, `x()`, `y()`, `z()`, `h()`, `sz()`, `cx()`
/// - All other operations have default implementations
///
/// # References
/// - Gottesman, "The Heisenberg Representation of Quantum Computers"
///   <https://arxiv.org/abs/quant-ph/9807006>
#[expect(clippy::min_ident_chars)]
pub trait CliffordSimulator<T: IndexableElement>: QuantumSimulator {
    #[inline]
    #[must_use]
    fn new(num_qubits: usize) -> Self
    where
        Self: Sized,
    {
        <Self as QuantumSimulator>::new(num_qubits)
    }

    #[inline]
    fn num_qubits(&self) -> usize {
        <Self as QuantumSimulator>::num_qubits(self)
    }

    #[inline]
    fn reset(&mut self) -> &mut Self {
        <Self as QuantumSimulator>::reset(self)
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
    fn px(&mut self, q: T) -> (bool, bool) {
        let (meas, deter) = self.mx(q);
        if meas {
            self.z(q);
        }
        (meas, deter)
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
    fn pnx(&mut self, q: T) -> (bool, bool) {
        let (meas, deter) = self.mnx(q);
        if meas {
            self.z(q);
        }
        (meas, deter)
    }

    /// Preparation of the +`Y_q` operator.
    /// # Panics
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    fn py(&mut self, q: T) -> (bool, bool) {
        let (meas, deter) = self.my(q);
        if meas {
            self.z(q);
        }
        (meas, deter)
    }

    /// Preparation of the -`Y_q` operator.
    /// # Panics
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    fn pny(&mut self, q: T) -> (bool, bool) {
        let (meas, deter) = self.mny(q);
        if meas {
            self.z(q);
        }
        (meas, deter)
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
    fn pz(&mut self, q: T) -> (bool, bool) {
        let (meas, deter) = self.mz(q);
        if meas {
            self.x(q);
        }
        (meas, deter)
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
    fn pnz(&mut self, q: T) -> (bool, bool) {
        let (meas, deter) = self.mnz(q);
        if meas {
            self.x(q);
        }
        (meas, deter)
    }

    /// Measurement of the +`X_q` operator.
    #[inline]
    fn mx(&mut self, q: T) -> (bool, bool) {
        // +X -> +Z
        self.h(q);
        let (meas, deter) = self.mz(q);
        // +Z -> +X
        self.h(q);

        (meas, deter)
    }

    /// Measurement of the -`X_q` operator.
    #[inline]
    fn mnx(&mut self, q: T) -> (bool, bool) {
        // -X -> +Z
        self.h(q);
        self.x(q);
        let (meas, deter) = self.mz(q);
        // +Z -> -X
        self.x(q);
        self.h(q);

        (meas, deter)
    }

    /// Measurement of the +`Y_q` operator.
    /// # Panics
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    fn my(&mut self, q: T) -> (bool, bool) {
        // +Y -> +Z
        self.sx(q);
        let (meas, deter) = self.mz(q);
        // +Z -> +Y
        self.sxdg(q);

        (meas, deter)
    }

    /// Measurement of the -`Y_q` operator.
    /// # Panics
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    fn mny(&mut self, q: T) -> (bool, bool) {
        // -Y -> +Z
        self.sxdg(q);
        let (meas, deter) = self.mz(q);
        // +Z -> -Y
        self.sx(q);

        (meas, deter)
    }

    /// Measurement of the +`Z_q` operator.
    fn mz(&mut self, q: T) -> (bool, bool);

    /// Measurement of the -`Z_q` operator.
    /// # Panics
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    fn mnz(&mut self, q: T) -> (bool, bool) {
        // -Z -> +Z
        self.x(q);
        let (meas, deter) = self.mz(q);
        // +Z -> -Z
        self.x(q);

        (meas, deter)
    }

    /// Identity on qubit q. X -> X, Z -> Z
    #[inline]
    fn identity(&mut self, _q: T) {}

    /// Pauli X gate. X -> X, Z -> -Z
    fn x(&mut self, q: T);

    /// Pauli Y gate. X -> -X, Z -> -Z
    fn y(&mut self, q: T);

    /// Pauli Z gate. X -> -X, Z -> Z
    fn z(&mut self, q: T);

    /// Sqrt of X gate.
    ///     X -> X
    ///     Z -> -iW = -Y
    ///     W -> -iZ
    ///     Y -> Z
    #[inline]
    fn sx(&mut self, q: T) {
        self.h(q);
        self.sz(q);
        self.h(q);
    }

    /// Adjoint of Sqrt X gate.
    ///     X -> X
    ///     Z -> iW = Y
    ///     W -> iZ
    ///     Y -> -Z
    #[inline]
    fn sxdg(&mut self, q: T) {
        self.h(q);
        self.szdg(q);
        self.h(q);
    }

    /// Sqrt of Y gate.
    ///     X -> -Z
    ///     Z -> X
    ///     W -> W
    ///     Y -> Y
    /// # Panics
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    fn sy(&mut self, q: T) {
        self.h(q);
        self.x(q);
    }

    /// Adjoint of sqrt of Y gate.
    ///     X -> Z
    ///     Z -> -X
    ///     W -> W
    ///     Y -> Y
    /// # Panics
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    fn sydg(&mut self, q: T) {
        self.x(q);
        self.h(q);
    }

    /// Sqrt of Z gate. +X -> +Y; +Z -> +Z; +Y -> -X;
    fn sz(&mut self, q: T);

    /// Adjoint of Sqrt of Z gate. X -> ..., Z -> ...
    ///     X -> -iW = -Y
    ///     Z -> Z
    ///     W -> -iX
    ///     Y -> X
    #[inline]
    fn szdg(&mut self, q: T) {
        self.z(q);
        self.sz(q);
    }

    /// Hadamard gate. X -> Z, Z -> X
    fn h(&mut self, q: T);

    /// X -> -Z, Z -> -X, Y -> -Y
    #[inline]
    fn h2(&mut self, q: T) {
        self.sy(q);
        self.z(q);
    }

    /// X -> Y, Z -> -Z, Y -> X
    #[inline]
    fn h3(&mut self, q: T) {
        self.sz(q);
        self.y(q);
    }

    /// X -> -Y, Z -> -Z, Y -> -X
    #[inline]
    fn h4(&mut self, q: T) {
        self.sz(q);
        self.x(q);
    }

    /// X -> -X, Z -> Y, Y -> Z
    #[inline]
    fn h5(&mut self, q: T) {
        self.sx(q);
        self.z(q);
    }

    /// X -> -X, Z -> -Y, Y -> -Z
    #[inline]
    fn h6(&mut self, q: T) {
        self.sx(q);
        self.y(q);
    }

    /// X -> Y, Z -> X, Y -> Z
    #[inline]
    fn f(&mut self, q: T) {
        self.sx(q);
        self.sz(q);
    }

    /// X -> Z, Z -> Y, Y -> X
    #[inline]
    fn fdg(&mut self, q: T) {
        self.szdg(q);
        self.sxdg(q);
    }

    /// X -> -Z, Z -> Y, Y -> -X
    #[inline]
    fn f2(&mut self, q: T) {
        self.sxdg(q);
        self.sy(q);
    }

    /// X -> -Y, Z -> -X, Y -> Z
    #[inline]
    fn f2dg(&mut self, q: T) {
        self.sydg(q);
        self.sx(q);
    }

    /// X -> Y, Z -> -X, Y -> -Z
    #[inline]
    fn f3(&mut self, q: T) {
        self.sxdg(q);
        self.sz(q);
    }

    /// X -> -Z, Z -> -Y, Y -> X
    #[inline]
    fn f3dg(&mut self, q: T) {
        self.szdg(q);
        self.sx(q);
    }

    /// X -> Z, Z -> -Y, Y -> -X
    #[inline]
    fn f4(&mut self, q: T) {
        self.sz(q);
        self.sx(q);
    }

    /// X -> -Y, Z -> X, Y -> -Z
    #[inline]
    fn f4dg(&mut self, q: T) {
        self.sxdg(q);
        self.szdg(q);
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
    fn cx(&mut self, q1: T, q2: T);

    /// CY: +IX -> +ZX; +IZ -> +ZZ; +XI -> -XY; +ZI -> +ZI;
    #[inline]
    fn cy(&mut self, q1: T, q2: T) {
        self.sz(q2);
        self.cx(q1, q2);
        self.szdg(q2);
    }

    /// CZ: +IX -> +ZX; +IZ -> +IZ; +XI -> +XZ; +ZI -> +ZI;
    #[inline]
    fn cz(&mut self, q1: T, q2: T) {
        self.h(q2);
        self.cx(q1, q2);
        self.h(q2);
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
    fn sxx(&mut self, q1: T, q2: T) {
        self.sx(q1);
        self.sx(q2);
        self.sydg(q1);
        self.cx(q1, q2);
        self.sy(q1);
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
    fn sxxdg(&mut self, q1: T, q2: T) {
        self.x(q1);
        self.x(q2);
        self.sxx(q1, q2);
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
    fn syy(&mut self, q1: T, q2: T) {
        self.szdg(q1);
        self.szdg(q2);
        self.sxx(q1, q2);
        self.sz(q1);
        self.sz(q2);
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
    fn syydg(&mut self, q1: T, q2: T) {
        self.y(q1);
        self.y(q2);
        self.syy(q1, q2);
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
    fn szz(&mut self, q1: T, q2: T) {
        self.sydg(q1);
        self.sydg(q2);
        self.sxx(q1, q2);
        self.sy(q1);
        self.sy(q2);
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
    fn szzdg(&mut self, q1: T, q2: T) {
        self.z(q1);
        self.z(q2);
        self.szz(q1, q2);
    }

    /// SWAP: +IX -> XI;
    ///       +IZ -> ZI;
    ///       +XI -> IX;
    ///       +ZI -> IZ;
    #[inline]
    fn swap(&mut self, q1: T, q2: T) {
        self.cx(q1, q2);
        self.cx(q2, q1);
        self.cx(q1, q2);
    }

    /// G2: +XI -> +IX
    ///     +IX -> +XI
    ///     +ZI -> +XZ
    ///     +IZ -> +ZX
    #[inline]
    fn g2(&mut self, q1: T, q2: T) {
        self.cz(q1, q2);
        self.h(q1);
        self.h(q2);
        self.cz(q1, q2);
    }
}
