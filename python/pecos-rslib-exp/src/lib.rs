// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Python bindings for experimental PECOS simulators.
//!
//! Exposes `StabMps` (stabilizer + MPS hybrid) and `Mast` (magic state
//! injection) from `pecos-stab-tn` via `PyO3`.

mod mast_bindings;
mod stab_mps_bindings;

use pecos_core::Angle64;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(crate) fn extract_angle(
    params: Option<&Bound<'_, PyDict>>,
    gate_name: &str,
) -> PyResult<Angle64> {
    let params = params.ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{gate_name} requires params with 'angle'"
        ))
    })?;
    let py_any = params.get_item("angle")?.ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{gate_name} requires an 'angle' parameter"
        ))
    })?;
    let radians: f64 = py_any.extract().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Expected a float 'angle' parameter for {gate_name}"
        ))
    })?;
    Ok(Angle64::from_radians(radians))
}

#[pymodule]
fn pecos_rslib_exp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<stab_mps_bindings::PyStabMps>()?;
    m.add_class::<mast_bindings::PyMast>()?;
    Ok(())
}
