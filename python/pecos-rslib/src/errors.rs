//! Shared Python exception mapping for simulation construction and execution.

use pecos_core::errors::PecosError;
use pyo3::PyErr;
use pyo3::exceptions::{PyRuntimeError, PyValueError};

pub(crate) fn simulation_error(error: PecosError, context: &str) -> PyErr {
    let message = format!("{context}: {error}");
    match error {
        PecosError::Input(_) => PyValueError::new_err(message),
        _ => PyRuntimeError::new_err(message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulation_errors_distinguish_input_from_runtime() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            for context in ["Simulation failed", "Failed to build simulation"] {
                let input =
                    simulation_error(PecosError::Input("bad count 3; use 2".into()), context);
                assert!(input.is_instance_of::<PyValueError>(py));
                assert!(input.to_string().contains("bad count 3; use 2"));
                let runtime =
                    simulation_error(PecosError::Processing("execution failed".into()), context);
                assert!(runtime.is_instance_of::<PyRuntimeError>(py));
            }
        });
    }
}
