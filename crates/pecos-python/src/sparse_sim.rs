use pecos_core::VecSet;
use pecos_qsims::CliffordSimulator;
use pecos_qsims::SparseStab;
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

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn x(&mut self, qubit: usize) {
        self.inner.x(qubit);
    }

    fn y(&mut self, qubit: usize) {
        self.inner.y(qubit);
    }

    fn z(&mut self, qubit: usize) {
        self.inner.z(qubit);
    }

    fn h(&mut self, qubit: usize) {
        self.inner.h(qubit);
    }

    fn h2(&mut self, qubit: usize) {
        self.inner.h2(qubit);
    }

    fn h3(&mut self, qubit: usize) {
        self.inner.h3(qubit);
    }

    fn h4(&mut self, qubit: usize) {
        self.inner.h4(qubit);
    }

    fn h5(&mut self, qubit: usize) {
        self.inner.h5(qubit);
    }

    fn h6(&mut self, qubit: usize) {
        self.inner.h6(qubit);
    }

    fn f(&mut self, qubit: usize) {
        self.inner.f(qubit);
    }

    fn fdg(&mut self, qubit: usize) {
        self.inner.fdg(qubit);
    }

    fn f2(&mut self, qubit: usize) {
        self.inner.f2(qubit);
    }

    fn f2dg(&mut self, qubit: usize) {
        self.inner.f2dg(qubit);
    }

    fn f3(&mut self, qubit: usize) {
        self.inner.f3(qubit);
    }

    fn f3dg(&mut self, qubit: usize) {
        self.inner.f3dg(qubit);
    }

    fn f4(&mut self, qubit: usize) {
        self.inner.f4(qubit);
    }

    fn f4dg(&mut self, qubit: usize) {
        self.inner.f4dg(qubit);
    }

    fn sx(&mut self, qubit: usize) {
        self.inner.sx(qubit);
    }

    fn sxdg(&mut self, qubit: usize) {
        self.inner.sxdg(qubit);
    }

    fn sy(&mut self, qubit: usize) {
        self.inner.sy(qubit);
    }

    fn sydg(&mut self, qubit: usize) {
        self.inner.sydg(qubit);
    }

    fn sz(&mut self, qubit: usize) {
        self.inner.sz(qubit);
    }

    fn szdg(&mut self, qubit: usize) {
        self.inner.szdg(qubit);
    }

    fn cx(&mut self, control: usize, target: usize) {
        self.inner.cx(control, target);
    }

    fn cy(&mut self, control: usize, target: usize) {
        self.inner.cy(control, target);
    }

    fn cz(&mut self, control: usize, target: usize) {
        self.inner.cz(control, target);
    }

    fn sxx(&mut self, qubit1: usize, qubit2: usize) {
        self.inner.sxx(qubit1, qubit2);
    }

    fn sxxdg(&mut self, qubit1: usize, qubit2: usize) {
        self.inner.sxxdg(qubit1, qubit2);
    }

    fn syy(&mut self, qubit1: usize, qubit2: usize) {
        self.inner.syy(qubit1, qubit2);
    }

    fn syydg(&mut self, qubit1: usize, qubit2: usize) {
        self.inner.syydg(qubit1, qubit2);
    }

    fn szz(&mut self, qubit1: usize, qubit2: usize) {
        self.inner.szz(qubit1, qubit2);
    }

    fn szzdg(&mut self, qubit1: usize, qubit2: usize) {
        self.inner.szzdg(qubit1, qubit2);
    }

    fn swap(&mut self, qubit1: usize, qubit2: usize) {
        self.inner.swap(qubit1, qubit2);
    }

    fn g2(&mut self, qubit1: usize, qubit2: usize) {
        self.inner.g2(qubit1, qubit2);
    }

    fn mz(&mut self, qubit: usize) -> (bool, bool) {
        self.inner.mz(qubit)
    }

    fn mx(&mut self, qubit: usize) -> (bool, bool) {
        self.inner.mx(qubit)
    }

    fn my(&mut self, qubit: usize) -> (bool, bool) {
        self.inner.my(qubit)
    }

    fn pz(&mut self, qubit: usize) -> (bool, bool) {
        self.inner.pz(qubit)
    }

    fn px(&mut self, qubit: usize) -> (bool, bool) {
        self.inner.px(qubit)
    }

    fn py(&mut self, qubit: usize) -> (bool, bool) {
        self.inner.py(qubit)
    }

    fn pnz(&mut self, qubit: usize) -> (bool, bool) {
        self.inner.pnz(qubit)
    }

    fn pnx(&mut self, qubit: usize) -> (bool, bool) {
        self.inner.pnx(qubit)
    }

    fn pny(&mut self, qubit: usize) -> (bool, bool) {
        self.inner.pny(qubit)
    }

    fn mz_forced(&mut self, qubit: usize, forced_outcome: bool) -> (bool, bool) {
        self.inner.mz_forced(qubit, forced_outcome)
    }

    fn pz_forced(&mut self, qubit: usize, forced_outcome: bool) -> (bool, bool) {
        self.inner.pz_forced(qubit, forced_outcome)
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
                self.x(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("Y", 1) => {
                self.y(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("Z", 1) => {
                self.z(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("H", 1) => {
                self.h(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("H2", 1) => {
                self.h2(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("H3", 1) => {
                self.h3(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("H4", 1) => {
                self.h4(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("H5", 1) => {
                self.h5(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("H6", 1) => {
                self.h6(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("F", 1) => {
                self.f(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("Fdg", 1) => {
                self.fdg(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("F2", 1) => {
                self.f2(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("F2dg", 1) => {
                self.f2dg(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("F3", 1) => {
                self.f3(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("F3dg", 1) => {
                self.f3dg(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("F4", 1) => {
                self.f4(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("F4dg", 1) => {
                self.f4dg(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("SX", 1) => {
                self.sx(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("SXdg", 1) => {
                self.sxdg(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("SY", 1) => {
                self.sy(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("SYdg", 1) => {
                self.sydg(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("SZ", 1) => {
                self.sz(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("SZdg", 1) => {
                self.szdg(location.get_item(0)?.extract()?);
                Ok(None)
            }
            ("CX", 2) => {
                self.cx(
                    location.get_item(0)?.extract()?,
                    location.get_item(1)?.extract()?,
                );
                Ok(None)
            }
            ("CY", 2) => {
                self.cy(
                    location.get_item(0)?.extract()?,
                    location.get_item(1)?.extract()?,
                );
                Ok(None)
            }
            ("CZ", 2) => {
                self.cz(
                    location.get_item(0)?.extract()?,
                    location.get_item(1)?.extract()?,
                );
                Ok(None)
            }
            ("SXX", 2) => {
                self.sxx(
                    location.get_item(0)?.extract()?,
                    location.get_item(1)?.extract()?,
                );
                Ok(None)
            }
            ("SXXdg", 2) => {
                self.sxxdg(
                    location.get_item(0)?.extract()?,
                    location.get_item(1)?.extract()?,
                );
                Ok(None)
            }
            ("SYY", 2) => {
                self.syy(
                    location.get_item(0)?.extract()?,
                    location.get_item(1)?.extract()?,
                );
                Ok(None)
            }
            ("SYYdg", 2) => {
                self.syydg(
                    location.get_item(0)?.extract()?,
                    location.get_item(1)?.extract()?,
                );
                Ok(None)
            }
            ("SZZ", 2) => {
                self.szz(
                    location.get_item(0)?.extract()?,
                    location.get_item(1)?.extract()?,
                );
                Ok(None)
            }
            ("SZZdg", 2) => {
                self.szzdg(
                    location.get_item(0)?.extract()?,
                    location.get_item(1)?.extract()?,
                );
                Ok(None)
            }
            ("SWAP", 2) => {
                self.swap(
                    location.get_item(0)?.extract()?,
                    location.get_item(1)?.extract()?,
                );
                Ok(None)
            }

            ("G2", 2) => {
                self.g2(
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
                    "MZ" => self.mz(qubit),
                    "MX" => self.mx(qubit),
                    "MY" => self.my(qubit),
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
                        self.mz_forced(qubit, forced_value)
                    }
                    "PZ" => self.pz(qubit),
                    "PX" => self.px(qubit),
                    "PY" => self.py(qubit),
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
                        self.pz_forced(qubit, forced_value)
                    }
                    "PnZ" => self.pnz(qubit),
                    "PnX" => self.pnx(qubit),
                    "PnY" => self.pny(qubit),
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
