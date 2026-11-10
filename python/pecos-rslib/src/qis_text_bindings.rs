//! LLVM-independent QIS text analysis in the base bindings wheel.

use pecos_qis::text::{self, ResultTagMeasurements};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::BTreeMap;

pyo3::create_exception!(pecos_rslib.qis, ResultTagCertificationError, PyValueError);

/// Return sorted result tags, scalar measurement ordinals and the total count.
/// Unsupported programs raise `ResultTagCertificationError`.
#[pyfunction]
fn extract_result_tag_measurements(qis: &str) -> PyResult<ResultTagMeasurements> {
    text::extract_result_tag_measurements(qis)
        .map_err(|error| ResultTagCertificationError::new_err(error.to_string()))
}

/// Normalize private PECOS helper symbols and validate their runtime ABIs.
/// Incompatible declarations or unsupported placements raise `ValueError`.
#[pyfunction]
fn normalize_pecos_helper_symbols(ir: &str) -> PyResult<String> {
    text::normalize_pecos_helper_symbols(ir)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

pub(crate) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let module = PyModule::new(py, "qis")?;
    module.add(
        "ResultTagCertificationError",
        py.get_type::<ResultTagCertificationError>(),
    )?;
    module.add(
        "PECOS_HELPER_ABIS",
        text::PECOS_HELPER_ABIS
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
    )?;
    module.add_function(wrap_pyfunction!(extract_result_tag_measurements, &module)?)?;
    module.add_function(wrap_pyfunction!(normalize_pecos_helper_symbols, &module)?)?;
    parent.add_submodule(&module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("pecos_rslib.qis", module)?;
    Ok(())
}
