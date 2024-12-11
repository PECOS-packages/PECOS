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

use pecos::prelude::*;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

#[pyclass]
pub struct SparseSim {
    inner: SparseStab<VecSet<usize>, usize>,
}

#[pymethods]
impl SparseSim {
    #[new]
    fn new(num_qubits: usize) -> Self {
        SparseSim {
            inner: SparseStab::<VecSet<usize>, usize>::new(num_qubits),
        }
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[allow(clippy::too_many_lines)]
    #[pyo3(signature = (symbol, location, params=None))]
    fn run_1q_gate(
        &mut self,
        symbol: &str,
        location: usize,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Option<u8>> {
        match symbol {
            "X" => {
                self.inner.x(location);
                Ok(None)
            }
            "Y" => {
                self.inner.y(location);
                Ok(None)
            }
            "Z" => {
                self.inner.z(location);
                Ok(None)
            }
            "H" => {
                self.inner.h(location);
                Ok(None)
            }
            "H2" => {
                self.inner.h2(location);
                Ok(None)
            }
            "H3" => {
                self.inner.h3(location);
                Ok(None)
            }
            "H4" => {
                self.inner.h4(location);
                Ok(None)
            }
            "H5" => {
                self.inner.h5(location);
                Ok(None)
            }
            "H6" => {
                self.inner.h6(location);
                Ok(None)
            }
            "F" => {
                self.inner.f(location);
                Ok(None)
            }
            "Fdg" => {
                self.inner.fdg(location);
                Ok(None)
            }
            "F2" => {
                self.inner.f2(location);
                Ok(None)
            }
            "F2dg" => {
                self.inner.f2dg(location);
                Ok(None)
            }
            "F3" => {
                self.inner.f3(location);
                Ok(None)
            }
            "F3dg" => {
                self.inner.f3dg(location);
                Ok(None)
            }
            "F4" => {
                self.inner.f4(location);
                Ok(None)
            }
            "F4dg" => {
                self.inner.f4dg(location);
                Ok(None)
            }
            "SX" => {
                self.inner.sx(location);
                Ok(None)
            }
            "SXdg" => {
                self.inner.sxdg(location);
                Ok(None)
            }
            "SY" => {
                self.inner.sy(location);
                Ok(None)
            }
            "SYdg" => {
                self.inner.sydg(location);
                Ok(None)
            }
            "SZ" => {
                self.inner.sz(location);
                Ok(None)
            }
            "SZdg" => {
                self.inner.szdg(location);
                Ok(None)
            }
            "MZ" | "MX" | "MY" | "MZForced" | "PZ" | "PX" | "PY" | "PZForced" | "PnZ" | "PnX"
            | "PnY" => {
                let (result, _) = match symbol {
                    "MZ" => self.inner.mz(location),
                    "MX" => self.inner.mx(location),
                    "MY" => self.inner.my(location),
                    "MZForced" => {
                        let forced_value = params
                            .ok_or_else(|| {
                                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "MZForced requires params",
                                )
                            })?
                            .get_item("forced_outcome")?
                            .ok_or_else(|| {
                                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "MZForced requires a 'forced_outcome' parameter",
                                )
                            })?
                            .call_method0("__bool__")?
                            .extract::<bool>()?;
                        self.inner.mz_forced(location, forced_value)
                    }
                    "PZ" => self.inner.pz(location),
                    "PX" => self.inner.px(location),
                    "PY" => self.inner.py(location),
                    "PZForced" => {
                        let forced_value = params
                            .ok_or_else(|| {
                                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "PZForced requires params",
                                )
                            })?
                            .get_item("forced_outcome")?
                            .ok_or_else(|| {
                                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "PZForced requires a 'forced_outcome' parameter",
                                )
                            })?
                            .call_method0("__bool__")?
                            .extract::<bool>()?;
                        self.inner.pz_forced(location, forced_value)
                    }
                    "PnZ" => self.inner.pnz(location),
                    "PnX" => self.inner.pnx(location),
                    "PnY" => self.inner.pny(location),
                    _ => unreachable!(),
                };
                Ok(Some(u8::from(result)))
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
        location: &Bound<'_, PyTuple>,
        _params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Option<u8>> {
        if location.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Two-qubit gate requires exactly 2 qubit locations",
            ));
        }

        let q1: usize = location.get_item(0)?.extract()?;
        let q2: usize = location.get_item(1)?.extract()?;

        match symbol {
            "CX" => {
                self.inner.cx(q1, q2);
                Ok(None)
            }
            "CY" => {
                self.inner.cy(q1, q2);
                Ok(None)
            }
            "CZ" => {
                self.inner.cz(q1, q2);
                Ok(None)
            }
            "SXX" => {
                self.inner.sxx(q1, q2);
                Ok(None)
            }
            "SXXdg" => {
                self.inner.sxxdg(q1, q2);
                Ok(None)
            }
            "SYY" => {
                self.inner.syy(q1, q2);
                Ok(None)
            }
            "SYYdg" => {
                self.inner.syydg(q1, q2);
                Ok(None)
            }
            "SZZ" => {
                self.inner.szz(q1, q2);
                Ok(None)
            }
            "SZZdg" => {
                self.inner.szzdg(q1, q2);
                Ok(None)
            }
            "SWAP" => {
                self.inner.swap(q1, q2);
                Ok(None)
            }
            "G2" => {
                self.inner.g2(q1, q2);
                Ok(None)
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unsupported two-qubit gate",
            )),
        }
    }

    #[pyo3(signature = (symbol, location, params=None))]
    fn run_gate(
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
                println!("Stabilizers:");
                for line in &stab_lines {
                    println!("{line}");
                }
                println!("Destabilizers:");
                for line in &destab_lines {
                    println!("{line}");
                }
            }

            [stab_lines, destab_lines].concat()
        } else {
            if verbose {
                println!("Stabilizers:");
                for line in &stab_lines {
                    println!("{line}");
                }
            }

            stab_lines
        }
    }
}
