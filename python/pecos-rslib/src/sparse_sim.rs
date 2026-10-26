// Copyright 2026 The PECOS Developers
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

use crate::prelude::*;
use crate::simulator_utils::{
    SymbolEntry, extract_forced_outcome, supports_exact, validate_gate_locations,
};
use pecos_core::BitSet;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PySet, PyTuple};

#[pyclass(module = "pecos_rslib")]
pub struct SparseSim {
    inner: SparseStab,
}

#[cfg(test)]
crate::simulator_utils::direct_surface_test!(
    direct_surface_matches_predicate,
    { SparseSim::new(2) },
    supports_forcing = true
);

#[cfg(test)]
crate::simulator_utils::forced_z_surface_test!(
    forced_mz_matches_aliases,
    |seed| SparseSim {
        inner: SparseStab::with_seed(2, seed)
    },
    "MZ",
    &["MZForced"],
    true
);

#[cfg(test)]
crate::simulator_utils::forced_z_surface_test!(
    forced_pz_matches_aliases,
    |seed| SparseSim {
        inner: SparseStab::with_seed(2, seed)
    },
    "PZ",
    &["PZForced"],
    false
);

#[cfg(test)]
crate::simulator_utils::random_z_surface_test!(
    random_z_measurement_sentinel,
    |seed| SparseSim {
        inner: SparseStab::with_seed(2, seed)
    },
    true
);

#[cfg(test)]
crate::simulator_utils::random_z_surface_test!(
    random_z_preparation_sentinel,
    |seed| SparseSim {
        inner: SparseStab::with_seed(2, seed)
    },
    false
);

#[cfg(test)]
crate::simulator_utils::invalid_forced_z_surface_test!(SparseSim {
    inner: SparseStab::with_seed(2, 0)
});

// This legacy class is not exported by the Python module, so exercise its
// Python boundary directly instead of adding a public API solely for testing.
#[test]
fn repeated_pair_is_python_value_error() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        for entry in crate::simulator_utils::SYMBOL_ENTRIES {
            if entry.qubit_count() != 2 || !supports(entry) {
                continue;
            }
            let mut sim = SparseSim::new(2);
            let location = PyTuple::new(py, [1_usize, 1_usize]).expect("qubit pair");
            let error = sim
                .run_2q_gate(entry.spelling, &location, None)
                .expect_err("repeated pair must raise ValueError");
            assert!(error.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            let message = error.to_string();
            assert!(message.contains(entry.spelling), "{message}");
            assert!(message.contains("repeated qubit 1"), "{message}");
        }
    });
}

#[test]
fn invalid_pair_batches_are_python_errors_without_mutation() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        for (invalid, message) in [
            (
                (2_usize, 2_usize),
                "CX requires distinct qubits; repeated qubit 2",
            ),
            ((2, 3), "CX: qubit 3 is outside the 3-qubit register"),
            ((3, 2), "CX: qubit 3 is outside the 3-qubit register"),
        ] {
            let mut sim = SparseSim::new(3);
            sim.inner.h(&[QubitId(0)]);
            let before = sim.inner.full_tableau();
            let locations = PySet::new(py, [(0_usize, 1_usize), invalid]).expect("gate locations");
            let error = sim
                .run_gate("CX", locations.as_any(), None, py)
                .expect_err("invalid batch must raise ValueError before mutation");
            assert!(error.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(error.to_string().contains(message), "{error}");
            assert_eq!(sim.inner.full_tableau(), before);

            let pair = PyTuple::new(py, [invalid.0, invalid.1]).expect("qubit pair");
            let error = sim
                .run_2q_gate("CX", &pair, None)
                .expect_err("invalid direct pair");
            assert!(error.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(error.to_string().contains(message), "{error}");
            assert_eq!(sim.inner.full_tableau(), before);
        }
    });
}

#[test]
fn pair_operands_are_extracted_once() {
    Python::initialize();
    Python::attach(|py| {
        let namespace = PyDict::new(py);
        py.run(pyo3::ffi::c_str!("class Index:\n    def __init__(self, value):\n        self.value = value\n        self.calls = 0\n    def __index__(self):\n        self.calls += 1\n        return self.value if self.calls == 1 else 2\na, b = Index(0), Index(1)\nlocations = {(a, b)}"), Some(&namespace), None).expect("index objects");
        let locations = namespace
            .get_item("locations")
            .expect("namespace")
            .expect("locations");
        let mut sim = SparseSim::new(3);
        sim.inner.h(&[QubitId(0)]);
        let mut expected = sim.inner.clone();
        expected.cx(&[(QubitId(0), QubitId(1))]);
        sim.run_gate("CX", &locations, None, py)
            .expect("validated CX");
        let calls: (usize, usize) = py
            .eval(
                pyo3::ffi::c_str!("(a.calls, b.calls)"),
                Some(&namespace),
                None,
            )
            .expect("call counts")
            .extract()
            .expect("integer counts");
        assert_eq!(calls, (1, 1));
        assert_eq!(sim.inner.full_tableau(), expected.full_tableau());
    });
}

#[test]
fn malformed_pair_batches_are_atomic() {
    Python::initialize();
    Python::attach(|py| {
        let malformed = py
            .eval(
                pyo3::ffi::c_str!("[(0,), (0, 1, 2), (-1, 0), (0, 1.5), 'bad', 0]"),
                None,
                None,
            )
            .expect("malformed locations");
        for location in malformed.try_iter().expect("locations") {
            let location = location.expect("location");
            let mut sim = SparseSim::new(3);
            sim.inner.h(&[QubitId(0)]);
            let before = sim.inner.full_tableau();
            let valid = PyTuple::new(py, [0, 1]).expect("valid pair");
            let locations = PySet::new(py, [valid.as_any(), &location]).expect("batch");
            let error = sim
                .run_gate("CX", locations.as_any(), None, py)
                .expect_err("malformed batch must fail");
            assert!(error.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            let message = error.to_string();
            assert!(message.contains("CX: expected a 2-tuple"), "{message}");
            assert!(
                message.contains(&location.repr().expect("repr").to_string()),
                "{message}"
            );
            assert_eq!(sim.inner.full_tableau(), before);
            assert!(
                sim.run_2q_gate("CX", &location, None)
                    .expect_err("malformed direct pair")
                    .is_instance_of::<pyo3::exceptions::PyValueError>(py)
            );
        }
    });
}

fn supports(entry: &SymbolEntry) -> bool {
    supports_exact! { entry;
        X => ["X"];
        Y => ["Y"];
        Z => ["Z"];
        H => ["H"];
        H2 => ["H2"];
        H3 => ["H3"];
        H4 => ["H4"];
        H5 => ["H5"];
        H6 => ["H6"];
        F => ["F"];
        Fdg => ["Fdg"];
        F2 => ["F2"];
        F2dg => ["F2dg"];
        F3 => ["F3"];
        F3dg => ["F3dg"];
        F4 => ["F4"];
        F4dg => ["F4dg"];
        Sx => ["SX"];
        Sxdg => ["SXdg"];
        Sy => ["SY"];
        Sydg => ["SYdg"];
        Sz => ["SZ"];
        Szdg => ["SZdg"];
        Pz => ["PZ"];
        PzForced => ["PZForced"];
        Pnz => ["PNZ"];
        Px => ["PX"];
        Pnx => ["PNX"];
        Py => ["PY"];
        Pny => ["PNY"];
        Mz => ["MZ"];
        MzForced => ["MZForced"];
        Mx => ["MX"];
        My => ["MY"];
        Cx => ["CX"];
        Cy => ["CY"];
        Cz => ["CZ"];
        Sxx => ["SXX"];
        Sxxdg => ["SXXdg"];
        Syy => ["SYY"];
        Syydg => ["SYYdg"];
        Szz => ["SZZ"];
        Szzdg => ["SZZdg"];
        Swap => ["SWAP"];
        G => ["G2"];
    }
}

#[pymethods]
impl SparseSim {
    #[new]
    fn new(num_qubits: usize) -> Self {
        SparseSim {
            inner: SparseStab::new(num_qubits),
        }
    }

    fn reset(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner.reset();
        slf
    }

    fn __repr__(&self) -> String {
        format!("SparseSim(num_qubits={})", self.inner.num_qubits())
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    #[allow(clippy::too_many_lines)]
    #[pyo3(signature = (symbol, location, params=None))]
    fn run_1q_gate(
        &mut self,
        symbol: &str,
        location: usize,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Option<u8>> {
        crate::simulator_utils::validate_qubit(symbol, location, self.inner.qubit_capacity())?;
        let q = &[QubitId(location)];
        match symbol {
            "X" => {
                self.inner.x(q);
                Ok(None)
            }
            "Y" => {
                self.inner.y(q);
                Ok(None)
            }
            "Z" => {
                self.inner.z(q);
                Ok(None)
            }
            "H" => {
                self.inner.h(q);
                Ok(None)
            }
            "H2" => {
                self.inner.h2(q);
                Ok(None)
            }
            "H3" => {
                self.inner.h3(q);
                Ok(None)
            }
            "H4" => {
                self.inner.h4(q);
                Ok(None)
            }
            "H5" => {
                self.inner.h5(q);
                Ok(None)
            }
            "H6" => {
                self.inner.h6(q);
                Ok(None)
            }
            "F" => {
                self.inner.f(q);
                Ok(None)
            }
            "Fdg" => {
                self.inner.fdg(q);
                Ok(None)
            }
            "F2" => {
                self.inner.f2(q);
                Ok(None)
            }
            "F2dg" => {
                self.inner.f2dg(q);
                Ok(None)
            }
            "F3" => {
                self.inner.f3(q);
                Ok(None)
            }
            "F3dg" => {
                self.inner.f3dg(q);
                Ok(None)
            }
            "F4" => {
                self.inner.f4(q);
                Ok(None)
            }
            "F4dg" => {
                self.inner.f4dg(q);
                Ok(None)
            }
            "SX" => {
                self.inner.sx(q);
                Ok(None)
            }
            "SXdg" => {
                self.inner.sxdg(q);
                Ok(None)
            }
            "SY" => {
                self.inner.sy(q);
                Ok(None)
            }
            "SYdg" => {
                self.inner.sydg(q);
                Ok(None)
            }
            "SZ" => {
                self.inner.sz(q);
                Ok(None)
            }
            "SZdg" => {
                self.inner.szdg(q);
                Ok(None)
            }
            "PZ" | "PZForced" => {
                if let Some(forced_value) =
                    extract_forced_outcome(params, symbol, symbol == "PZForced")?
                {
                    self.inner.pz_forced(location, forced_value);
                    return Ok(None);
                }
                self.inner.pz(q);
                Ok(None)
            }
            "PX" => {
                self.inner.px(q);
                Ok(None)
            }
            "PY" => {
                self.inner.py(q);
                Ok(None)
            }
            "PNZ" => {
                self.inner.pnz(q);
                Ok(None)
            }
            "PNX" => {
                self.inner.pnx(q);
                Ok(None)
            }
            "PNY" => {
                self.inner.pny(q);
                Ok(None)
            }
            "MZ" | "MX" | "MY" | "MZForced" => {
                let result = match symbol {
                    "MZ" | "MZForced" => {
                        if let Some(forced_value) =
                            extract_forced_outcome(params, symbol, symbol == "MZForced")?
                        {
                            self.inner.mz_forced(location, forced_value)
                        } else {
                            self.inner
                                .mz(q)
                                .into_iter()
                                .next()
                                .expect("measurement returned no results")
                        }
                    }
                    "MX" => self
                        .inner
                        .mx(q)
                        .into_iter()
                        .next()
                        .expect("measurement returned no results"),
                    "MY" => self
                        .inner
                        .my(q)
                        .into_iter()
                        .next()
                        .expect("measurement returned no results"),
                    _ => unreachable!(),
                };
                Ok(Some(u8::from(result.outcome)))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unsupported single-qubit gate",
            )),
        }
    }

    #[pyo3(signature = (symbol, location, _params))]
    fn run_2q_gate(
        &mut self,
        symbol: &str,
        location: &Bound<'_, PyAny>,
        _params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Option<u8>> {
        let pair = crate::simulator_utils::extract_pair(symbol, location)?;
        crate::simulator_utils::validate_pair(
            symbol,
            pair.0.0,
            pair.1.0,
            self.inner.qubit_capacity(),
        )?;
        self.dispatch_2q_gate(symbol, pair, None)
    }

    /// Internal gate dispatcher (tuple-based) - for internal use
    #[pyo3(signature = (symbol, location, params=None))]
    fn run_gate_internal(
        &mut self,
        symbol: &str,
        location: &Bound<'_, PyTuple>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Option<u8>> {
        match location.len() {
            1 => {
                let qubit: usize = location.get_item(0)?.extract()?;
                self.run_1q_gate(symbol, qubit, params)
            }
            2 => self.run_2q_gate(symbol, location, params),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Gate location must be specified for either 1 or 2 qubits",
            )),
        }
    }

    /// High-level `run_gate` that accepts a set of locations (Python wrapper compatible).
    /// Unsupported symbols raise for empty locations; supported symbols return an empty result.
    #[pyo3(signature = (symbol, locations, **params))]
    fn run_gate(
        &mut self,
        symbol: &str,
        locations: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyDict>>,
        py: Python<'_>,
    ) -> PyResult<Py<PyDict>> {
        self.run_gate_highlevel(symbol, locations, params, py)
    }

    fn stab_tableau(&self) -> String {
        self.inner.stab_tableau()
    }

    fn destab_tableau(&self) -> String {
        self.inner.destab_tableau()
    }

    #[pyo3(signature = (verbose=None, _print_y=None, print_destabs=None))]
    fn print_stabs(
        &self,
        verbose: Option<bool>,
        _print_y: Option<bool>,
        print_destabs: Option<bool>,
    ) -> Vec<String> {
        let verbose = verbose.unwrap_or(true);
        // let print_y = print_y.unwrap_or(true);
        let print_destabs = print_destabs.unwrap_or(false);

        let stabs = self.inner.stab_tableau();
        let stab_lines: Vec<String> = stabs.lines().map(String::from).collect();

        if print_destabs {
            let destabs = self.inner.destab_tableau();
            let destab_lines: Vec<String> = destabs.lines().map(String::from).collect();

            if verbose {
                log::debug!("Stabilizers:");
                for line in &stab_lines {
                    log::debug!("{line}");
                }
                log::debug!("Destabilizers:");
                for line in &destab_lines {
                    log::debug!("{line}");
                }
            }

            [stab_lines, destab_lines].concat()
        } else {
            if verbose {
                log::debug!("Stabilizers:");
                for line in &stab_lines {
                    log::debug!("{line}");
                }
            }

            stab_lines
        }
    }

    /// High-level `run_gate` method that accepts a set of locations
    #[pyo3(signature = (symbol, locations, **params))]
    fn run_gate_highlevel(
        &mut self,
        symbol: &str,
        locations: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyDict>>,
        py: Python<'_>,
    ) -> PyResult<Py<PyDict>> {
        let output = PyDict::new(py);
        let locations_set: Bound<PySet> = locations.clone().cast_into()?;
        let crate::simulator_utils::ValidatedLocations { qubits, pairs } = validate_gate_locations(
            symbol,
            &locations_set,
            supports,
            self.inner.qubit_capacity(),
        )?;

        // Check if simulate_gate is False
        if let Some(p) = params
            && let Ok(Some(sg)) = p.get_item("simulate_gate")
            && let Ok(false) = sg.extract::<bool>()
        {
            return Ok(output.into());
        }

        if let Some(pairs) = pairs {
            for pair in pairs {
                if let Some(value) = self.dispatch_2q_gate(symbol, pair, params)? {
                    output.set_item((pair.0.0, pair.1.0), value)?;
                }
            }
            return Ok(output.into());
        }

        for (&qubit, location) in qubits.iter().zip(locations_set.iter()) {
            let result = self.run_1q_gate(symbol, qubit, params)?;

            if let Some(value) = result {
                output.set_item(location, value)?;
            }
        }

        Ok(output.into())
    }

    /// Execute a quantum circuit
    #[pyo3(signature = (circuit, removed_locations=None))]
    fn run_circuit(
        &mut self,
        circuit: &Bound<'_, PyAny>,
        removed_locations: Option<&Bound<'_, PySet>>,
        py: Python<'_>,
    ) -> PyResult<Py<PyDict>> {
        let results = PyDict::new(py);

        // Iterate over circuit items
        for item in circuit.call_method0("items")?.try_iter()? {
            let item = item?;
            let tuple: Bound<PyTuple> = item.clone().cast_into()?;

            let symbol: String = tuple.get_item(0)?.extract()?;
            let locations_item = tuple.get_item(1)?;
            let locations: Bound<PySet> = locations_item.clone().cast_into()?;
            let params_item = tuple.get_item(2)?;
            let params: Bound<PyDict> = params_item.clone().cast_into()?;

            // Subtract removed_locations if provided
            let final_locations = if let Some(removed) = removed_locations {
                locations.call_method1("__sub__", (removed,))?
            } else {
                locations.clone().into_any()
            };

            // Run the gate
            let gate_results =
                self.run_gate_highlevel(&symbol, &final_locations, Some(&params), py)?;

            // Update results
            results.call_method1("update", (gate_results,))?;
        }

        Ok(results.into())
    }

    /// Add faults by running a circuit
    #[pyo3(signature = (circuit, removed_locations=None))]
    fn add_faults(
        &mut self,
        circuit: &Bound<'_, PyAny>,
        removed_locations: Option<&Bound<'_, PySet>>,
        py: Python<'_>,
    ) -> PyResult<()> {
        self.run_circuit(circuit, removed_locations, py)?;
        Ok(())
    }

    /// Returns the raw gens data (`col_x`, `col_z`, `row_x`, `row_z`) for stabs or destabs.
    fn _gens_data(&self, is_stab: bool) -> crate::simulator_utils::GensData {
        let gens = if is_stab {
            self.inner.stabs()
        } else {
            self.inner.destabs()
        };
        let to_vecs = |sets: &[BitSet]| -> Vec<Vec<usize>> {
            sets.iter().map(|s| s.iter().collect()).collect()
        };
        (
            to_vecs(&gens.col_x),
            to_vecs(&gens.col_z),
            to_vecs(&gens.row_x),
            to_vecs(&gens.row_z),
        )
    }

    #[getter]
    fn bindings(slf: PyRef<'_, Self>) -> PyResult<crate::simulator_utils::GateBindingsDict> {
        // Create a Rust GateBindingsDict directly
        let py = slf.py();
        let sim_obj: Py<PyAny> = slf.into_bound_py_any(py)?.unbind();
        Ok(crate::simulator_utils::GateBindingsDict::new(
            sim_obj, supports,
        ))
    }

    #[getter]
    fn stabs(slf: PyRef<'_, Self>) -> PyResult<crate::simulator_utils::TableauWrapper> {
        // Create a Rust TableauWrapper directly with is_stab=true
        let py = slf.py();
        let sim_obj: Py<PyAny> = slf.into_bound_py_any(py)?.unbind();
        Ok(crate::simulator_utils::TableauWrapper::new(sim_obj, true))
    }

    #[getter]
    fn destabs(slf: PyRef<'_, Self>) -> PyResult<crate::simulator_utils::TableauWrapper> {
        // Create a Rust TableauWrapper directly with is_stab=false
        let py = slf.py();
        let sim_obj: Py<PyAny> = slf.into_bound_py_any(py)?.unbind();
        Ok(crate::simulator_utils::TableauWrapper::new(sim_obj, false))
    }
}

impl SparseSim {
    // Operands have already been extracted and validated for the entire batch.
    fn dispatch_2q_gate(
        &mut self,
        symbol: &str,
        operands: (QubitId, QubitId),
        _params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Option<u8>> {
        let pair = &[operands];
        match symbol {
            "CX" => {
                self.inner.cx(pair);
                Ok(None)
            }
            "CY" => {
                self.inner.cy(pair);
                Ok(None)
            }
            "CZ" => {
                self.inner.cz(pair);
                Ok(None)
            }
            "SXX" => {
                self.inner.sxx(pair);
                Ok(None)
            }
            "SXXdg" => {
                self.inner.sxxdg(pair);
                Ok(None)
            }
            "SYY" => {
                self.inner.syy(pair);
                Ok(None)
            }
            "SYYdg" => {
                self.inner.syydg(pair);
                Ok(None)
            }
            "SZZ" => {
                self.inner.szz(pair);
                Ok(None)
            }
            "SZZdg" => {
                self.inner.szzdg(pair);
                Ok(None)
            }
            "SWAP" => {
                self.inner.swap(pair);
                Ok(None)
            }
            "G2" => {
                self.inner.g(pair);
                Ok(None)
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unsupported two-qubit gate",
            )),
        }
    }
}
