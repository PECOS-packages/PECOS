// Copyright 2025 The PECOS Developers
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

//! Python bindings for PECOS Pauli operators.
//!
//! This module exposes the fundamental Pauli types (I, X, Y, Z) and `PauliString`
//! to Python, allowing quantum error models to use native Pauli representations
//! instead of string-based arrays.

use pecos::prelude::{
    IndexableElement, Pauli as RustPauli, PauliString as RustPauliString, QuarterPhase, QubitId,
};
use pyo3::prelude::*;

/// Single-qubit Pauli operator (I, X, Y, Z)
///
/// This represents the four single-qubit Pauli operators:
/// - I: Identity (no error)
/// - X: Bit flip
/// - Z: Phase flip
/// - Y: Both bit and phase flip (Y = iXZ)
///
/// Internally represented as 2 bits:
/// - I = 0b00
/// - X = 0b01
/// - Z = 0b10
/// - Y = 0b11
///
/// Examples:
///     >>> from `pecos_rslib` import Pauli
///     >>> x = Pauli.X
///     >>> z = Pauli.Z
///     >>> print(x)  # "X"
#[pyclass(name = "Pauli", module = "pecos_rslib", frozen, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Pauli(RustPauli);

// SAFETY: Pauli is a simple Copy type wrapping a 2-bit enum.
// It contains no Python objects or mutable state, so it's safe to send across threads.
unsafe impl Send for Pauli {}
unsafe impl Sync for Pauli {}

#[pymethods]
#[allow(clippy::trivially_copy_pass_by_ref)] // PyO3 requires &self for special methods
impl Pauli {
    /// Identity operator (no error)
    #[classattr]
    const I: Pauli = Pauli(RustPauli::I);

    /// Pauli X (bit flip)
    #[classattr]
    const X: Pauli = Pauli(RustPauli::X);

    /// Pauli Z (phase flip)
    #[classattr]
    const Z: Pauli = Pauli(RustPauli::Z);

    /// Pauli Y (both bit and phase flip)
    #[classattr]
    const Y: Pauli = Pauli(RustPauli::Y);

    /// Create a Pauli from a string
    ///
    /// Args:
    ///     s: String "I", "X", "Y", or "Z"
    ///
    /// Returns:
    ///     Pauli operator
    ///
    /// Raises:
    ///     `ValueError`: If string is not a valid Pauli
    #[staticmethod]
    pub fn from_str(s: &str) -> PyResult<Self> {
        match s {
            "I" => Ok(Pauli(RustPauli::I)),
            "X" => Ok(Pauli(RustPauli::X)),
            "Y" => Ok(Pauli(RustPauli::Y)),
            "Z" => Ok(Pauli(RustPauli::Z)),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid Pauli string: '{s}'. Must be 'I', 'X', 'Y', or 'Z'"
            ))),
        }
    }

    /// String representation
    fn __str__(&self) -> &'static str {
        match self.0 {
            RustPauli::I => "I",
            RustPauli::X => "X",
            RustPauli::Y => "Y",
            RustPauli::Z => "Z",
        }
    }

    /// Repr for debugging
    fn __repr__(&self) -> String {
        format!("Pauli.{}", self.__str__())
    }

    /// Hash for use in dicts/sets
    fn __hash__(&self) -> u8 {
        self.0 as u8
    }

    /// Equality comparison
    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    /// Convert to integer (0=I, 1=X, 2=Z, 3=Y)
    #[allow(clippy::wrong_self_convention)] // PyO3 requires &self for all methods
    fn to_int(&self) -> u8 {
        self.0 as u8
    }

    /// Create from integer (0=I, 1=X, 2=Z, 3=Y)
    #[staticmethod]
    fn from_int(val: u8) -> PyResult<Self> {
        match val {
            0 => Ok(Pauli(RustPauli::I)),
            1 => Ok(Pauli(RustPauli::X)),
            2 => Ok(Pauli(RustPauli::Z)),
            3 => Ok(Pauli(RustPauli::Y)),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid Pauli integer: {val}. Must be 0 (I), 1 (X), 2 (Z), or 3 (Y)"
            ))),
        }
    }
}

/// Multi-qubit Pauli string
///
/// Represents a tensor product of Pauli operators acting on multiple qubits.
/// For example, "IXZ" means I on qubit 0, X on qubit 1, Z on qubit 2.
///
/// Can also represent sparse Pauli strings where only non-identity operators
/// are stored. For example, X on qubit 1 and Z on qubit 5 in a 10-qubit system.
///
/// Examples:
///
/// ```ignore
/// >>> from pecos_rslib import Pauli, PauliString
/// >>> # Create X on qubit 0, Z on qubit 1
/// >>> ps = PauliString([(Pauli.X, 0), (Pauli.Z, 1)])
/// >>> print(ps)  # "XZ"
///
/// >>> # Create from string (assumes sequential qubits starting at 0)
/// >>> ps2 = PauliString.from_str("XYZ")
/// >>> print(ps2)  # "XYZ"
/// ```
#[pyclass(name = "PauliString", module = "pecos_rslib", from_py_object)]
#[derive(Debug, Clone)]
pub struct PauliString {
    inner: RustPauliString,
}

// SAFETY: PauliString wraps RustPauliString which is thread-safe
unsafe impl Send for PauliString {}
unsafe impl Sync for PauliString {}

#[pymethods]
impl PauliString {
    /// Create a new `PauliString`
    ///
    /// Args:
    ///     paulis: Either:
    ///         - List of (Pauli, `qubit_index`) tuples for explicit qubit numbering
    ///         - List of Pauli operators for implicit sequential numbering (0, 1, 2, ...)
    ///         - None for identity
    ///     phase: Optional phase factor (0, 1, 2, 3 for +1, +i, -1, -i)
    ///
    /// Examples:
    ///     >>> # Explicit qubit indices (sparse representation)
    ///     >>> ps1 = `PauliString`([(Pauli.X, 0), (Pauli.Z, 2)])
    ///     >>> # Implicit sequential indices (dense representation)
    ///     >>> ps2 = `PauliString`([Pauli.X, Pauli.Y, Pauli.Z])  # qubits 0, 1, 2
    ///     >>> # With phase
    ///     >>> ps3 = `PauliString`([Pauli.Y], phase=2)  # -Y on qubit 0
    #[new]
    #[pyo3(signature = (paulis=None, phase=0))]
    fn new(paulis: Option<&Bound<'_, PyAny>>, phase: u8) -> PyResult<Self> {
        let rust_phase = match phase {
            0 => QuarterPhase::PlusOne,
            1 => QuarterPhase::PlusI,
            2 => QuarterPhase::MinusOne,
            3 => QuarterPhase::MinusI,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid phase: {phase}. Must be 0 (+1), 1 (+i), 2 (-1), or 3 (-i)"
                )));
            }
        };

        // Build PauliString from input
        let rust_paulis = if let Some(pauli_input) = paulis {
            use pyo3::types::PyList;

            // Try to extract as a list - using cast() per PyO3 0.27 API
            let Ok(list) = pauli_input.cast::<PyList>() else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "paulis must be a list",
                ));
            };

            if list.is_empty() {
                Vec::new()
            } else {
                // Check first element to determine format
                let first = list.get_item(0)?;

                // Try to extract as tuple (explicit qubit indices)
                if first.extract::<(Pauli, usize)>().is_ok() {
                    // Format: [(Pauli, qubit_id), ...]
                    list.iter()
                        .map(|item| {
                            let (pauli, qubit): (Pauli, usize) = item.extract()?;
                            Ok((pauli.0, QubitId::from_index(qubit)))
                        })
                        .collect::<PyResult<Vec<_>>>()?
                }
                // Try to extract as Pauli (implicit sequential indices)
                else if first.extract::<Pauli>().is_ok() {
                    // Format: [Pauli, ...] with implicit 0, 1, 2, ...
                    list.iter()
                        .enumerate()
                        .map(|(idx, item)| {
                            let pauli: Pauli = item.extract()?;
                            Ok((pauli.0, QubitId::from_index(idx)))
                        })
                        .collect::<PyResult<Vec<_>>>()?
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "paulis must be a list of Pauli objects or (Pauli, qubit_id) tuples",
                    ));
                }
            }
        } else {
            Vec::new()
        };

        // Construct RustPauliString using the new constructor
        let inner = RustPauliString::with_phase_and_paulis(rust_phase, rust_paulis);

        Ok(PauliString { inner })
    }

    /// Create `PauliString` from a string like "XYZ" or "IXZI"
    ///
    /// Args:
    ///     s: String of Pauli operators (I, X, Y, Z)
    ///
    /// Returns:
    ///     `PauliString` with operators on sequential qubits starting at 0
    ///
    /// Examples:
    ///     >>> ps = `PauliString.from_str("XYZ`")
    ///     >>> # X on qubit 0, Y on qubit 1, Z on qubit 2
    #[staticmethod]
    fn from_str(s: &str) -> PyResult<Self> {
        // Parse string character by character
        let mut paulis = Vec::new();

        for (i, c) in s.chars().enumerate() {
            let pauli = match c {
                'I' | 'i' => RustPauli::I,
                'X' | 'x' => RustPauli::X,
                'Y' | 'y' => RustPauli::Y,
                'Z' | 'z' => RustPauli::Z,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid Pauli character '{c}' at position {i}. Must be 'I', 'X', 'Y', or 'Z'"
                    )));
                }
            };

            // Only store non-identity operators (sparse representation)
            if pauli != RustPauli::I {
                paulis.push((pauli, QubitId::from_index(i)));
            }
        }

        let inner = RustPauliString::with_phase_and_paulis(QuarterPhase::PlusOne, paulis);

        Ok(PauliString { inner })
    }

    /// String representation
    fn __str__(&self) -> String {
        // Build string representation
        let phase_str = match self.inner.get_phase() {
            QuarterPhase::PlusOne => "",
            QuarterPhase::PlusI => "+i*",
            QuarterPhase::MinusOne => "-",
            QuarterPhase::MinusI => "-i*",
        };

        let paulis = self.inner.get_paulis();
        if paulis.is_empty() {
            return format!("{phase_str}I");
        }

        // Build sparse representation showing only non-identity operators
        let pauli_str: String = paulis
            .iter()
            .map(|(p, q)| {
                let p_char = match p {
                    RustPauli::I => 'I',
                    RustPauli::X => 'X',
                    RustPauli::Y => 'Y',
                    RustPauli::Z => 'Z',
                };
                format!("{}_{}", p_char, q.to_index())
            })
            .collect::<Vec<_>>()
            .join(" ");

        format!("{phase_str}{pauli_str}")
    }

    /// Repr for debugging
    fn __repr__(&self) -> String {
        let phase = self.get_phase();
        let paulis = self.get_paulis();

        if paulis.is_empty() {
            if phase == 0 {
                return "PauliString()".to_string();
            }
            return format!("PauliString(phase={phase})");
        }

        let paulis_repr: String = paulis
            .iter()
            .map(|(p, q)| {
                let p_str = match p.0 {
                    RustPauli::I => "Pauli.I",
                    RustPauli::X => "Pauli.X",
                    RustPauli::Y => "Pauli.Y",
                    RustPauli::Z => "Pauli.Z",
                };
                format!("({p_str}, {q})")
            })
            .collect::<Vec<_>>()
            .join(", ");

        if phase == 0 {
            format!("PauliString([{paulis_repr}])")
        } else {
            format!("PauliString([{paulis_repr}], phase={phase})")
        }
    }

    /// Get the phase as an integer (0, 1, 2, 3)
    fn get_phase(&self) -> u8 {
        match self.inner.get_phase() {
            QuarterPhase::PlusOne => 0,
            QuarterPhase::PlusI => 1,
            QuarterPhase::MinusOne => 2,
            QuarterPhase::MinusI => 3,
        }
    }

    /// Get the list of (Pauli, qubit) tuples
    fn get_paulis(&self) -> Vec<(Pauli, usize)> {
        self.inner
            .get_paulis()
            .iter()
            .map(|(p, q)| (Pauli(*p), q.to_index()))
            .collect()
    }
}

/// Register Pauli types with Python module
pub fn register_pauli_types(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Pauli>()?;
    m.add_class::<PauliString>()?;
    Ok(())
}
