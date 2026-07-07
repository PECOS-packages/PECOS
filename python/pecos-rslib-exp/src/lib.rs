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

// Experimental PyO3 binding signatures are constrained by Python-callable APIs
// and generated method wrappers. Python docstrings also contain Python snippets
// that Clippy's Rust-doc Markdown lint misclassifies. Keep this list limited to
// binding/docs shape lints.
#![allow(
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps,
    clippy::unused_self
)]

//! Python bindings for experimental PECOS simulators.
//!
//! Exposes `StabMps` (stabilizer + MPS hybrid) and `Mast` (magic state
//! injection) from `pecos-stab-tn` via `PyO3`.

mod coherent_idle_channel;
mod eeg_bindings;
mod mast_bindings;
mod sim_neo_bindings;
mod stab_mps_bindings;
pub mod stabmps_builder;

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
    m.add_class::<sim_neo_bindings::PySimNeoBuilder>()?;
    m.add_class::<sim_neo_bindings::PyStabMpsBuilder>()?;
    m.add_class::<sim_neo_bindings::PyNoiseModelBuilder>()?;
    m.add_function(wrap_pyfunction!(sim_neo_bindings::py_sim_neo, m)?)?;
    m.add_class::<sim_neo_bindings::PyMonteCarloBuilder>()?;
    m.add_function(wrap_pyfunction!(sim_neo_bindings::monte_carlo, m)?)?;
    m.add_class::<sim_neo_bindings::PyPathEnumerationBuilder>()?;
    m.add_function(wrap_pyfunction!(sim_neo_bindings::path_enumeration, m)?)?;
    m.add_class::<sim_neo_bindings::PySubsetSimulationBuilder>()?;
    m.add_class::<sim_neo_bindings::PySubsetResult>()?;
    m.add_function(wrap_pyfunction!(sim_neo_bindings::subset_simulation, m)?)?;
    m.add_function(wrap_pyfunction!(sim_neo_bindings::stab_mps, m)?)?;
    m.add_function(wrap_pyfunction!(sim_neo_bindings::depolarizing, m)?)?;
    m.add_class::<sim_neo_bindings::PyStateVecBuilder>()?;
    m.add_class::<sim_neo_bindings::PyStabilizerBuilder>()?;
    m.add_function(wrap_pyfunction!(sim_neo_bindings::statevec, m)?)?;
    m.add_function(wrap_pyfunction!(sim_neo_bindings::stabilizer, m)?)?;
    m.add_class::<sim_neo_bindings::PyMeasSamplingBuilder>()?;
    m.add_class::<sim_neo_bindings::PyRawMeasurementResult>()?;
    m.add_function(wrap_pyfunction!(sim_neo_bindings::meas_sampling, m)?)?;
    m.add_class::<sim_neo_bindings::PyFaultCatalog>()?;
    m.add_class::<sim_neo_bindings::PyFaultLocation>()?;
    m.add_class::<sim_neo_bindings::PyFaultAlternative>()?;
    m.add_class::<sim_neo_bindings::PyFaultConfiguration>()?;
    m.add_class::<sim_neo_bindings::PyFaultConfigurationIter>()?;
    m.add_function(wrap_pyfunction!(sim_neo_bindings::fault_catalog, m)?)?;
    // DEM generation functions
    m.add_function(wrap_pyfunction!(eeg_bindings::exact_detection_rates, m)?)?;
    m.add_function(wrap_pyfunction!(eeg_bindings::exact_pairwise_rates, m)?)?;
    m.add_function(wrap_pyfunction!(eeg_bindings::coherent_dem_exact, m)?)?;
    m.add_function(wrap_pyfunction!(eeg_bindings::coherent_dem_decomposed, m)?)?;
    m.add_function(wrap_pyfunction!(eeg_bindings::exact_correlation_table, m)?)?;
    m.add_function(wrap_pyfunction!(eeg_bindings::correlation_matching_dem, m)?)?;
    m.add_function(wrap_pyfunction!(eeg_bindings::noise_characterization, m)?)?;
    m.add_function(wrap_pyfunction!(eeg_bindings::compress_noise, m)?)?;
    m.add_function(wrap_pyfunction!(eeg_bindings::perturbative_dem, m)?)?;
    m.add_function(wrap_pyfunction!(eeg_bindings::perturbative_dem_events, m)?)?;
    m.add_function(wrap_pyfunction!(eeg_bindings::eeg_summary, m)?)?;
    m.add_function(wrap_pyfunction!(eeg_bindings::eeg_event_diagnostics, m)?)?;
    m.add_function(wrap_pyfunction!(eeg_bindings::eeg_per_detector, m)?)?;
    Ok(())
}
