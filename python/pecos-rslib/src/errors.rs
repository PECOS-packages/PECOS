//! Shared Python exception mapping for simulation construction and execution.

use pecos_core::errors::PecosError;
use pyo3::PyErr;
use pyo3::exceptions::{PyRuntimeError, PyValueError};

pub(crate) fn simulation_error(error: PecosError, context: &str) -> PyErr {
    match error {
        PecosError::Input(message) => PyValueError::new_err(format!("{context}: {message}")),
        error => PyRuntimeError::new_err(format!("{context}: {error}")),
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
                assert_eq!(
                    input.to_string(),
                    format!("ValueError: {context}: bad count 3; use 2")
                );
                let runtime =
                    simulation_error(PecosError::Processing("execution failed".into()), context);
                assert!(runtime.is_instance_of::<PyRuntimeError>(py));
                assert_eq!(
                    runtime.to_string(),
                    format!("RuntimeError: {context}: Processing error: execution failed")
                );
            }
        });
    }
}
