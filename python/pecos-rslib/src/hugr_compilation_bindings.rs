// Python bindings for HUGR to LLVM compilation
use pecos::prelude::*;
use std::fs;

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Compile HUGR to QIS (LLVM IR with quantum instructions)
///
/// This function takes HUGR bytes (envelope format) and compiles them to QIS,
/// which is LLVM IR with quantum instruction set extensions.
///
/// Args:
///     `hugr_bytes`: HUGR program as envelope bytes
///     `output_path`: Optional path to write the QIS output
///
/// Returns:
///     QIS (LLVM IR) as a string
#[pyfunction(name = "compile_hugr_to_qis", signature = (hugr_bytes, output_path=None))]
pub fn py_compile_hugr_to_qis(hugr_bytes: &[u8], output_path: Option<&str>) -> PyResult<String> {
    let llvm_ir = compile_hugr_bytes_to_string(hugr_bytes)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    if let Some(path) = output_path {
        fs::write(path, &llvm_ir)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    }

    Ok(llvm_ir)
}

/// Get information about available compilation backends
#[pyfunction]
pub fn get_compilation_backends(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let result = PyDict::new(py);
    result.set_item("default_backend", "phir")?;

    let backends = PyDict::new(py);

    let phir_backend = PyDict::new(py);
    phir_backend.set_item("available", true)?;
    phir_backend.set_item("description", "PHIR pipeline: HUGR → PHIR → LLVM IR")?;
    backends.set_item("phir", phir_backend)?;

    let hugr_llvm_backend = PyDict::new(py);
    hugr_llvm_backend.set_item("available", true)?;
    hugr_llvm_backend.set_item("description", "HUGR-LLVM pipeline: HUGR → LLVM IR")?;
    backends.set_item("hugr-llvm", hugr_llvm_backend)?;

    result.set_item("backends", backends)?;

    Ok(result.into())
}

/// Register HUGR compilation functions with the Python module
pub fn register_hugr_compilation_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // HUGR to QIS compilation
    m.add_function(wrap_pyfunction!(py_compile_hugr_to_qis, m)?)?;
    m.add_function(wrap_pyfunction!(get_compilation_backends, m)?)?;

    Ok(())
}
