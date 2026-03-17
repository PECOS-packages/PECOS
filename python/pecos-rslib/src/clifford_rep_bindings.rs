// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Python bindings for `CliffordRep` — Heisenberg-picture Clifford gate representation.

use pecos::core::clifford_rep::CliffordRep as RustCliffordRep;
use pyo3::prelude::*;

use crate::pauli_bindings::PauliString;
use crate::stabilizer_group_bindings::PyPauliStabilizerGroup;

/// A Clifford gate in the Heisenberg picture.
///
/// Represents a Clifford gate by its action on Pauli generators:
/// for each qubit i, stores how `X_i` and `Z_i` transform under
/// conjugation `C P C†`.
///
/// Examples:
///     >>> from pecos_rslib import CliffordRep
///     >>> h = CliffordRep.h(0)  # Hadamard on qubit 0
///     >>> s = CliffordRep.s(0)  # S gate on qubit 0
///     >>> hs = h.compose(s)     # HS composition
///     >>> inv = h.inverse()     # H† = H
#[allow(clippy::doc_markdown)]
#[pyclass(name = "CliffordRep", module = "pecos_rslib", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyCliffordRep {
    inner: RustCliffordRep,
}

unsafe impl Send for PyCliffordRep {}
unsafe impl Sync for PyCliffordRep {}

#[pymethods]
impl PyCliffordRep {
    // ========================================================================
    // Constructors
    // ========================================================================

    /// Identity Clifford on n qubits.
    #[staticmethod]
    fn identity(num_qubits: usize) -> Self {
        Self {
            inner: RustCliffordRep::identity(num_qubits),
        }
    }

    /// Hadamard gate on qubit q.
    #[staticmethod]
    fn h(q: usize) -> Self {
        Self {
            inner: RustCliffordRep::h(q),
        }
    }

    /// Pauli X gate on qubit q.
    #[staticmethod]
    fn x(q: usize) -> Self {
        Self {
            inner: RustCliffordRep::x(q),
        }
    }

    /// Pauli Y gate on qubit q.
    #[staticmethod]
    fn y(q: usize) -> Self {
        Self {
            inner: RustCliffordRep::y(q),
        }
    }

    /// Pauli Z gate on qubit q.
    #[staticmethod]
    fn z(q: usize) -> Self {
        Self {
            inner: RustCliffordRep::z(q),
        }
    }

    /// sqrt(X) gate on qubit q.
    #[staticmethod]
    fn sx(q: usize) -> Self {
        Self {
            inner: RustCliffordRep::sx(q),
        }
    }

    /// sqrt(Y) gate on qubit q.
    #[staticmethod]
    fn sy(q: usize) -> Self {
        Self {
            inner: RustCliffordRep::sy(q),
        }
    }

    /// sqrt(Z) gate on qubit q.
    #[staticmethod]
    fn sz(q: usize) -> Self {
        Self {
            inner: RustCliffordRep::sz(q),
        }
    }

    /// SX† gate on qubit q.
    #[staticmethod]
    fn sxdg(q: usize) -> Self {
        Self {
            inner: RustCliffordRep::sxdg(q),
        }
    }

    /// SY† gate on qubit q.
    #[staticmethod]
    fn sydg(q: usize) -> Self {
        Self {
            inner: RustCliffordRep::sydg(q),
        }
    }

    /// SZ† gate on qubit q.
    #[staticmethod]
    fn szdg(q: usize) -> Self {
        Self {
            inner: RustCliffordRep::szdg(q),
        }
    }

    /// H2 gate on qubit q.
    #[staticmethod]
    fn h2(q: usize) -> Self {
        Self {
            inner: RustCliffordRep::h2(q),
        }
    }

    /// H3 gate on qubit q.
    #[staticmethod]
    fn h3(q: usize) -> Self {
        Self {
            inner: RustCliffordRep::h3(q),
        }
    }

    /// H4 gate on qubit q.
    #[staticmethod]
    fn h4(q: usize) -> Self {
        Self {
            inner: RustCliffordRep::h4(q),
        }
    }

    /// H5 gate on qubit q.
    #[staticmethod]
    fn h5(q: usize) -> Self {
        Self {
            inner: RustCliffordRep::h5(q),
        }
    }

    /// H6 gate on qubit q.
    #[staticmethod]
    fn h6(q: usize) -> Self {
        Self {
            inner: RustCliffordRep::h6(q),
        }
    }

    /// F (Face) gate on qubit q.
    #[staticmethod]
    fn f(q: usize) -> Self {
        Self {
            inner: RustCliffordRep::f(q),
        }
    }

    /// F† gate on qubit q.
    #[staticmethod]
    fn fdg(q: usize) -> Self {
        Self {
            inner: RustCliffordRep::fdg(q),
        }
    }

    /// F2 gate on qubit q.
    #[staticmethod]
    fn f2(q: usize) -> Self {
        Self {
            inner: RustCliffordRep::f2(q),
        }
    }

    /// F2† gate on qubit q.
    #[staticmethod]
    fn f2dg(q: usize) -> Self {
        Self {
            inner: RustCliffordRep::f2dg(q),
        }
    }

    /// F3 gate on qubit q.
    #[staticmethod]
    fn f3(q: usize) -> Self {
        Self {
            inner: RustCliffordRep::f3(q),
        }
    }

    /// F3† gate on qubit q.
    #[staticmethod]
    fn f3dg(q: usize) -> Self {
        Self {
            inner: RustCliffordRep::f3dg(q),
        }
    }

    /// F4 gate on qubit q.
    #[staticmethod]
    fn f4(q: usize) -> Self {
        Self {
            inner: RustCliffordRep::f4(q),
        }
    }

    /// F4† gate on qubit q.
    #[staticmethod]
    fn f4dg(q: usize) -> Self {
        Self {
            inner: RustCliffordRep::f4dg(q),
        }
    }

    /// CNOT (controlled-X) gate with control c and target t.
    #[staticmethod]
    fn cx(c: usize, t: usize) -> Self {
        Self {
            inner: RustCliffordRep::cx(c, t),
        }
    }

    /// Controlled-Z gate on qubits a and b.
    #[staticmethod]
    fn cz(a: usize, b: usize) -> Self {
        Self {
            inner: RustCliffordRep::cz(a, b),
        }
    }

    /// Controlled-Y gate with control c and target t.
    #[staticmethod]
    fn cy(c: usize, t: usize) -> Self {
        Self {
            inner: RustCliffordRep::cy(c, t),
        }
    }

    /// SWAP gate on qubits a and b.
    #[staticmethod]
    fn swap(a: usize, b: usize) -> Self {
        Self {
            inner: RustCliffordRep::swap(a, b),
        }
    }

    /// SXX gate on qubits a and b.
    #[staticmethod]
    fn sxx(a: usize, b: usize) -> Self {
        Self {
            inner: RustCliffordRep::sxx(a, b),
        }
    }

    /// SXX† gate on qubits a and b.
    #[staticmethod]
    fn sxxdg(a: usize, b: usize) -> Self {
        Self {
            inner: RustCliffordRep::sxxdg(a, b),
        }
    }

    /// SYY gate on qubits a and b.
    #[staticmethod]
    fn syy(a: usize, b: usize) -> Self {
        Self {
            inner: RustCliffordRep::syy(a, b),
        }
    }

    /// SYY† gate on qubits a and b.
    #[staticmethod]
    fn syydg(a: usize, b: usize) -> Self {
        Self {
            inner: RustCliffordRep::syydg(a, b),
        }
    }

    /// SZZ gate on qubits a and b.
    #[staticmethod]
    fn szz(a: usize, b: usize) -> Self {
        Self {
            inner: RustCliffordRep::szz(a, b),
        }
    }

    /// SZZ† gate on qubits a and b.
    #[staticmethod]
    fn szzdg(a: usize, b: usize) -> Self {
        Self {
            inner: RustCliffordRep::szzdg(a, b),
        }
    }

    /// iSWAP gate on qubits a and b.
    #[staticmethod]
    fn iswap(a: usize, b: usize) -> Self {
        Self {
            inner: RustCliffordRep::iswap(a, b),
        }
    }

    /// G gate on qubits a and b.
    #[staticmethod]
    fn g(a: usize, b: usize) -> Self {
        Self {
            inner: RustCliffordRep::g(a, b),
        }
    }

    // ========================================================================
    // Enumeration and random sampling
    // ========================================================================

    /// All 24 single-qubit Clifford gates on the given qubit.
    ///
    /// The 24 elements form the single-qubit Clifford group (modulo global phase).
    /// Index 0 is the identity.
    ///
    /// Args:
    ///     qubit: Which qubit the gates act on
    #[staticmethod]
    fn single_qubit_cliffords(qubit: usize) -> Vec<PyCliffordRep> {
        RustCliffordRep::single_qubit_cliffords(qubit)
            .into_iter()
            .map(|c| PyCliffordRep { inner: c })
            .collect()
    }

    /// A random single-qubit Clifford on the given qubit.
    ///
    /// Uniformly samples from the 24 single-qubit Cliffords.
    ///
    /// Args:
    ///     qubit: Which qubit the gate acts on
    #[staticmethod]
    fn random_single_qubit(qubit: usize) -> PyCliffordRep {
        let mut rng = rand::rng();
        PyCliffordRep {
            inner: RustCliffordRep::random_single_qubit(qubit, &mut rng),
        }
    }

    /// A random n-qubit Clifford by composing random gate layers.
    ///
    /// Uses `depth` layers of random single-qubit Cliffords and CZ gates.
    /// With sufficient depth (typically depth >= 2*n), this generates a
    /// distribution covering the full Clifford group.
    ///
    /// Args:
    ///     num_qubits: Number of qubits
    ///     depth: Number of random layers (default: 2 * num_qubits)
    #[allow(clippy::doc_markdown)]
    #[staticmethod]
    #[pyo3(signature = (num_qubits, depth=None))]
    fn random(num_qubits: usize, depth: Option<usize>) -> PyCliffordRep {
        let d = depth.unwrap_or(2 * num_qubits);
        let mut rng = rand::rng();
        PyCliffordRep {
            inner: RustCliffordRep::random(num_qubits, d, &mut rng),
        }
    }

    // ========================================================================
    // Operations
    // ========================================================================

    /// Compose this Clifford with another: self followed by other.
    ///
    /// Automatically extends to the larger qubit count if sizes differ.
    fn compose(&self, other: &PyCliffordRep) -> Self {
        Self {
            inner: self.inner.compose(&other.inner),
        }
    }

    /// Compute the inverse Clifford (C such that self * C = identity).
    fn inverse(&self) -> Self {
        Self {
            inner: self.inner.inverse(),
        }
    }

    /// Extend to n qubits by padding with identity.
    fn extended_to(&self, n: usize) -> Self {
        Self {
            inner: self.inner.extended_to(n),
        }
    }

    /// Apply this Clifford to a `PauliString`: `C P C†`.
    fn apply(&self, pauli: &PauliString) -> PauliString {
        PauliString::from_rust(self.inner.apply(&pauli.to_rust()))
    }

    /// Apply this Clifford to a `PauliStabilizerGroup`.
    ///
    /// Transforms all generators: `g_i` -> `C g_i C†`.
    fn apply_to_group(&self, group: &PyPauliStabilizerGroup) -> PyPauliStabilizerGroup {
        PyPauliStabilizerGroup {
            inner: group.inner.apply_clifford(&self.inner),
        }
    }

    /// Check if this is a valid Clifford (preserves commutation relations).
    fn is_valid(&self) -> bool {
        self.inner.is_valid()
    }

    /// Number of qubits this Clifford acts on.
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Create a Clifford from a `PauliString` (Paulis are Cliffords).
    ///
    /// The resulting Clifford acts as conjugation by the Pauli: `P * Q * P†`.
    #[staticmethod]
    fn from_pauli_string(pauli: &PauliString) -> Self {
        Self {
            inner: RustCliffordRep::from(pauli.to_rust()),
        }
    }

    /// Get the X generator image for the given qubit.
    fn x_image(&self, qubit: usize) -> PyResult<PauliString> {
        if qubit >= self.inner.num_qubits() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "qubit {qubit} out of range for {}-qubit Clifford",
                self.inner.num_qubits()
            )));
        }
        Ok(PauliString::from_rust(self.inner.x_image(qubit).clone()))
    }

    /// Get the Z generator image for the given qubit.
    fn z_image(&self, qubit: usize) -> PyResult<PauliString> {
        if qubit >= self.inner.num_qubits() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "qubit {qubit} out of range for {}-qubit Clifford",
                self.inner.num_qubits()
            )));
        }
        Ok(PauliString::from_rust(self.inner.z_image(qubit).clone()))
    }

    /// Compose two Cliffords: `self * other`.
    fn __mul__(&self, other: &PyCliffordRep) -> Self {
        Self {
            inner: &self.inner * &other.inner,
        }
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!("CliffordRep(num_qubits={})", self.inner.num_qubits())
    }
}

// ============================================================================
// Free constructor functions: H(0), SZ(0), CX(0,1), etc.
// ============================================================================

macro_rules! clifford_1q {
    ($py_name:ident, $export_name:literal, $method:ident) => {
        #[pyfunction]
        #[pyo3(name = $export_name)]
        fn $py_name(q: usize) -> PyCliffordRep {
            PyCliffordRep {
                inner: RustCliffordRep::$method(q),
            }
        }
    };
}

macro_rules! clifford_2q {
    ($py_name:ident, $export_name:literal, $method:ident) => {
        #[pyfunction]
        #[pyo3(name = $export_name)]
        fn $py_name(a: usize, b: usize) -> PyCliffordRep {
            PyCliffordRep {
                inner: RustCliffordRep::$method(a, b),
            }
        }
    };
}

// Single-qubit gates
clifford_1q!(py_h, "H", h);
clifford_1q!(py_h2, "H2", h2);
clifford_1q!(py_h3, "H3", h3);
clifford_1q!(py_h4, "H4", h4);
clifford_1q!(py_h5, "H5", h5);
clifford_1q!(py_h6, "H6", h6);
clifford_1q!(py_sx, "SX", sx);
clifford_1q!(py_sxdg, "SXdg", sxdg);
clifford_1q!(py_sy, "SY", sy);
clifford_1q!(py_sydg, "SYdg", sydg);
clifford_1q!(py_sz, "SZ", sz);
clifford_1q!(py_szdg, "SZdg", szdg);
clifford_1q!(py_f, "F", f);
clifford_1q!(py_fdg, "Fdg", fdg);
clifford_1q!(py_f2, "F2", f2);
clifford_1q!(py_f2dg, "F2dg", f2dg);
clifford_1q!(py_f3, "F3", f3);
clifford_1q!(py_f3dg, "F3dg", f3dg);
clifford_1q!(py_f4, "F4", f4);
clifford_1q!(py_f4dg, "F4dg", f4dg);

// Two-qubit gates
clifford_2q!(py_cx, "CX", cx);
clifford_2q!(py_cy, "CY", cy);
clifford_2q!(py_cz, "CZ", cz);
clifford_2q!(py_swap, "SWAP", swap);
clifford_2q!(py_sxx, "SXX", sxx);
clifford_2q!(py_sxxdg, "SXXdg", sxxdg);
clifford_2q!(py_syy, "SYY", syy);
clifford_2q!(py_syydg, "SYYdg", syydg);
clifford_2q!(py_szz, "SZZ", szz);
clifford_2q!(py_szzdg, "SZZdg", szzdg);
clifford_2q!(py_iswap, "ISWAP", iswap);
clifford_2q!(py_g, "G", g);

/// Register Clifford types with Python module.
pub fn register_clifford_types(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCliffordRep>()?;
    // Single-qubit constructors
    m.add_function(wrap_pyfunction!(py_h, m)?)?;
    m.add_function(wrap_pyfunction!(py_h2, m)?)?;
    m.add_function(wrap_pyfunction!(py_h3, m)?)?;
    m.add_function(wrap_pyfunction!(py_h4, m)?)?;
    m.add_function(wrap_pyfunction!(py_h5, m)?)?;
    m.add_function(wrap_pyfunction!(py_h6, m)?)?;
    m.add_function(wrap_pyfunction!(py_sx, m)?)?;
    m.add_function(wrap_pyfunction!(py_sxdg, m)?)?;
    m.add_function(wrap_pyfunction!(py_sy, m)?)?;
    m.add_function(wrap_pyfunction!(py_sydg, m)?)?;
    m.add_function(wrap_pyfunction!(py_sz, m)?)?;
    m.add_function(wrap_pyfunction!(py_szdg, m)?)?;
    m.add_function(wrap_pyfunction!(py_f, m)?)?;
    m.add_function(wrap_pyfunction!(py_fdg, m)?)?;
    m.add_function(wrap_pyfunction!(py_f2, m)?)?;
    m.add_function(wrap_pyfunction!(py_f2dg, m)?)?;
    m.add_function(wrap_pyfunction!(py_f3, m)?)?;
    m.add_function(wrap_pyfunction!(py_f3dg, m)?)?;
    m.add_function(wrap_pyfunction!(py_f4, m)?)?;
    m.add_function(wrap_pyfunction!(py_f4dg, m)?)?;
    // Two-qubit constructors
    m.add_function(wrap_pyfunction!(py_cx, m)?)?;
    m.add_function(wrap_pyfunction!(py_cy, m)?)?;
    m.add_function(wrap_pyfunction!(py_cz, m)?)?;
    m.add_function(wrap_pyfunction!(py_swap, m)?)?;
    m.add_function(wrap_pyfunction!(py_sxx, m)?)?;
    m.add_function(wrap_pyfunction!(py_sxxdg, m)?)?;
    m.add_function(wrap_pyfunction!(py_syy, m)?)?;
    m.add_function(wrap_pyfunction!(py_syydg, m)?)?;
    m.add_function(wrap_pyfunction!(py_szz, m)?)?;
    m.add_function(wrap_pyfunction!(py_szzdg, m)?)?;
    m.add_function(wrap_pyfunction!(py_iswap, m)?)?;
    m.add_function(wrap_pyfunction!(py_g, m)?)?;
    Ok(())
}
