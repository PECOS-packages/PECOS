//! Python-implemented simulator plugin.
//!
//! Wraps a Python object that implements the Clifford gate methods
//! into a Rust type that implements [`CliffordGateable`] (and optionally
//! [`ArbitraryRotationGateable`]).
//!
//! The Python author implements 4-5 methods and gets all 56 Clifford gates
//! for free via the trait default decompositions.

use pecos_core::{Angle64, QubitId};
use pecos_simulators::clifford_gateable::MeasurementResult;
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, QuantumSimulator};
use pyo3::prelude::*;

/// A quantum simulator implemented in Python, usable by PECOS's Rust engine.
///
/// The Python object must implement these Clifford methods:
/// - `sz(qubits: list[int])` -- S gate
/// - `h(qubits: list[int])` -- Hadamard
/// - `cx(pairs: list[tuple[int, int]])` -- CNOT
/// - `mz(qubits: list[int]) -> list[tuple[bool, bool]]` -- Z-measurement `(outcome, is_deterministic)`
/// - `reset()` -- reset to initial state
///
/// Optionally, for rotation support:
/// - `rx(theta: float, qubits: list[int])`
/// - `rz(theta: float, qubits: list[int])`
/// - `rzz(theta: float, pairs: list[tuple[int, int]])`
///
/// All other Clifford gates (X, Y, Z, SX, CZ, SWAP, etc.) are decomposed
/// automatically into the 4 primitives.
///
/// # Example (Python side)
///
/// ```python
/// class MySim:
///     def __init__(self, n):
///         self.bits = [False] * n
///     def sz(self, qubits): pass
///     def h(self, qubits):
///         for q in qubits: self.bits[q] = not self.bits[q]
///     def cx(self, pairs):
///         for c, t in pairs:
///             if self.bits[c]: self.bits[t] = not self.bits[t]
///     def mz(self, qubits):
///         return [(self.bits[q], True) for q in qubits]
///     def reset(self):
///         self.bits = [False] * len(self.bits)
///
/// sim = pecos_rslib.PyForeignSimulator(MySim(10))
/// ```
#[pyclass(name = "PyForeignSimulator", module = "pecos_rslib")]
pub struct PyForeignSimulator {
    inner: Py<PyAny>,
    supports_rotations: bool,
}

// SAFETY: Py<PyAny> is Send. GIL acquired via Python::attach() before access.
unsafe impl Send for PyForeignSimulator {}

#[pymethods]
impl PyForeignSimulator {
    /// Wrap a Python simulator object for use in PECOS.
    ///
    /// Validates required methods exist and probes for optional rotation support.
    #[new]
    fn new(py_obj: Py<PyAny>) -> PyResult<Self> {
        Python::attach(|py| {
            let obj = py_obj.bind(py);

            for method in &["sz", "h", "cx", "mz", "reset"] {
                if !obj.hasattr(*method)? {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                        "Python simulator must have a '{method}' method"
                    )));
                }
            }

            let supports_rotations =
                obj.hasattr("rx")? && obj.hasattr("rz")? && obj.hasattr("rzz")?;

            Ok(Self {
                inner: py_obj,
                supports_rotations,
            })
        })
    }

    #[getter]
    fn supports_rotations(&self) -> bool {
        self.supports_rotations
    }

    fn __repr__(&self) -> String {
        format!("PyForeignSimulator(rotations={})", self.supports_rotations)
    }
}

// -- Helpers --

fn qubit_indices(qubits: &[QubitId]) -> Vec<usize> {
    qubits.iter().map(QubitId::index).collect()
}

fn pair_tuples(pairs: &[(QubitId, QubitId)]) -> Vec<(usize, usize)> {
    pairs.iter().map(|(c, t)| (c.index(), t.index())).collect()
}

// -- Trait impls --

impl QuantumSimulator for PyForeignSimulator {
    fn reset(&mut self) -> &mut Self {
        Python::attach(|py| {
            self.inner
                .call_method0(py, "reset")
                .expect("Python simulator reset() failed");
        });
        self
    }
}

impl CliffordGateable for PyForeignSimulator {
    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        let indices = qubit_indices(qubits);
        Python::attach(|py| {
            self.inner
                .call_method1(py, "sz", (indices,))
                .expect("Python simulator sz() failed");
        });
        self
    }

    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        let indices = qubit_indices(qubits);
        Python::attach(|py| {
            self.inner
                .call_method1(py, "h", (indices,))
                .expect("Python simulator h() failed");
        });
        self
    }

    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let tuples = pair_tuples(pairs);
        Python::attach(|py| {
            self.inner
                .call_method1(py, "cx", (tuples,))
                .expect("Python simulator cx() failed");
        });
        self
    }

    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        let indices = qubit_indices(qubits);
        Python::attach(|py| {
            let result = self
                .inner
                .call_method1(py, "mz", (indices,))
                .expect("Python simulator mz() failed");

            let tuples: Vec<(bool, bool)> = result
                .extract(py)
                .expect("mz() must return list[tuple[bool, bool]]");

            tuples
                .into_iter()
                .map(|(outcome, is_deterministic)| MeasurementResult {
                    outcome,
                    is_deterministic,
                })
                .collect()
        })
    }
}

impl ArbitraryRotationGateable for PyForeignSimulator {
    fn rx(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        assert!(
            self.supports_rotations,
            "Python simulator does not support rotation gates (no rx method)"
        );
        let indices = qubit_indices(qubits);
        let radians = theta.to_radians();
        Python::attach(|py| {
            self.inner
                .call_method1(py, "rx", (radians, indices))
                .expect("Python simulator rx() failed");
        });
        self
    }

    fn rz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        assert!(
            self.supports_rotations,
            "Python simulator does not support rotation gates (no rz method)"
        );
        let indices = qubit_indices(qubits);
        let radians = theta.to_radians();
        Python::attach(|py| {
            self.inner
                .call_method1(py, "rz", (radians, indices))
                .expect("Python simulator rz() failed");
        });
        self
    }

    fn rzz(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        assert!(
            self.supports_rotations,
            "Python simulator does not support rotation gates (no rzz method)"
        );
        let tuples = pair_tuples(pairs);
        let radians = theta.to_radians();
        Python::attach(|py| {
            self.inner
                .call_method1(py, "rzz", (radians, tuples))
                .expect("Python simulator rzz() failed");
        });
        self
    }
}
