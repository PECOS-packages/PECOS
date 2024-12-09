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

// use pecos_core::VecSet;
// use pecos_qsims::CliffordSimulator;
// use pecos_qsims::SparseStab;
use pecos::prelude::*;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::collections::HashMap;

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

    #[allow(clippy::too_many_lines)]
    #[pyo3(signature = (symbol, location, params=None))]
    fn run_gate(
        &mut self,
        symbol: &str,
        location: &Bound<'_, PyTuple>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Option<HashMap<usize, u8>>> {
        match (symbol, location.len()) {
            ("X", 1) => {
                self.inner.x(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("Y", 1) => {
                self.inner.y(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("Z", 1) => {
                self.inner.z(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("H", 1) => {
                self.inner.h(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("H2", 1) => {
                self.inner.h2(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("H3", 1) => {
                self.inner.h3(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("H4", 1) => {
                self.inner.h4(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("H5", 1) => {
                self.inner.h5(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("H6", 1) => {
                self.inner.h6(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("F", 1) => {
                self.inner.f(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("Fdg", 1) => {
                self.inner.fdg(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("F2", 1) => {
                self.inner.f2(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("F2dg", 1) => {
                self.inner.f2dg(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("F3", 1) => {
                self.inner.f3(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("F3dg", 1) => {
                self.inner.f3dg(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("F4", 1) => {
                self.inner.f4(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("F4dg", 1) => {
                self.inner.f4dg(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("SX", 1) => {
                self.inner.sx(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("SXdg", 1) => {
                self.inner.sxdg(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("SY", 1) => {
                self.inner.sy(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("SYdg", 1) => {
                self.inner.sydg(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("SZ", 1) => {
                self.inner.sz(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("SZdg", 1) => {
                self.inner.szdg(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("CX", 2) => {
                self.inner.cx(
                    location.get_item(0)?.extract()?,
                    location.get_item(1)?.extract()?,
                );
                Ok(None)
            }
            ("CY", 2) => {
                self.inner.cy(
                    location.get_item(0)?.extract()?,
                    location.get_item(1)?.extract()?,
                );
                Ok(None)
            }
            ("CZ", 2) => {
                self.inner.cz(
                    location.get_item(0)?.extract()?,
                    location.get_item(1)?.extract()?,
                );
                Ok(None)
            }
            ("SXX", 2) => {
                self.inner.sxx(
                    location.get_item(0)?.extract()?,
                    location.get_item(1)?.extract()?,
                );
                Ok(None)
            }
            ("SXXdg", 2) => {
                self.inner.sxxdg(
                    location.get_item(0)?.extract()?,
                    location.get_item(1)?.extract()?,
                );
                Ok(None)
            }
            ("SYY", 2) => {
                self.inner.syy(
                    location.get_item(0)?.extract()?,
                    location.get_item(1)?.extract()?,
                );
                Ok(None)
            }
            ("SYYdg", 2) => {
                self.inner.syydg(
                    location.get_item(0)?.extract()?,
                    location.get_item(1)?.extract()?,
                );
                Ok(None)
            }
            ("SZZ", 2) => {
                self.inner.szz(
                    location.get_item(0)?.extract()?,
                    location.get_item(1)?.extract()?,
                );
                Ok(None)
            }
            ("SZZdg", 2) => {
                self.inner.szzdg(
                    location.get_item(0)?.extract()?,
                    location.get_item(1)?.extract()?,
                );
                Ok(None)
            }
            ("SWAP", 2) => {
                self.inner.swap(
                    location.get_item(0)?.extract()?,
                    location.get_item(1)?.extract()?,
                );
                Ok(None)
            }

            ("G2", 2) => {
                self.inner.g2(
                    location.get_item(0)?.extract()?,
                    location.get_item(1)?.extract()?,
                );
                Ok(None)
            }
            (
                "MZ" | "MX" | "MY" | "MZForced" | "PZ" | "PX" | "PY" | "PZForced" | "PnZ" | "PnX"
                | "PnY",
                1,
            ) => {
                let qubit: usize = location.get_item(0)?.extract()?;
                let (result, _) = match symbol {
                    "MZ" => self.inner.mz(qubit),
                    "MX" => self.inner.mx(qubit),
                    "MY" => self.inner.my(qubit),
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
                        self.inner.mz_forced(qubit, forced_value)
                    }
                    "PZ" => self.inner.pz(qubit),
                    "PX" => self.inner.px(qubit),
                    "PY" => self.inner.py(qubit),
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
                        self.inner.pz_forced(qubit, forced_value)
                    }
                    "PnZ" => self.inner.pnz(qubit),
                    "PnX" => self.inner.pnx(qubit),
                    "PnY" => self.inner.pny(qubit),
                    _ => unreachable!(),
                };
                let mut map = HashMap::new();
                if result {
                    map.insert(qubit, 1);
                }
                Ok(Some(map))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unsupported gate or incorrect number of qubits",
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
