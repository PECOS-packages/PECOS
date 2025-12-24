// Python bindings for HUGR to LLVM compilation
use pecos::prelude::*;
use std::fs;

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Compile HUGR to LLVM IR
///
/// This function takes HUGR bytes (envelope format) and compiles them to LLVM IR
/// using the PECOS HUGR compiler that generates QIS-compatible output.
///
/// Args:
///     `hugr_bytes`: HUGR program as envelope bytes
///
/// Returns:
///     LLVM IR as a string
#[pyfunction(name = "compile_hugr_to_llvm", signature = (hugr_bytes, output_path=None))]
pub fn py_compile_hugr_to_llvm(hugr_bytes: &[u8], output_path: Option<&str>) -> PyResult<String> {
    let llvm_ir = compile_hugr_bytes_to_string(hugr_bytes)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    if let Some(path) = output_path {
        fs::write(path, &llvm_ir)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    }

    Ok(llvm_ir)
}

/// Check if Rust HUGR backend is available
#[pyfunction]
pub fn check_rust_hugr_availability() -> (bool, String) {
    (true, "HUGR support available via sim() API".to_string())
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
    let compile_fn = wrap_pyfunction!(py_compile_hugr_to_llvm, m)?;
    m.add_function(compile_fn.clone())?;
    // Add backwards-compatible alias
    m.add("compile_hugr_to_llvm_rust", compile_fn)?;

    m.add_function(wrap_pyfunction!(check_rust_hugr_availability, m)?)?;
    m.add_function(wrap_pyfunction!(get_compilation_backends, m)?)?;

    // Add availability constants
    m.add("RUST_HUGR_AVAILABLE", true)?;
    m.add("HUGR_LLVM_PIPELINE_AVAILABLE", true)?;

    Ok(())
}
