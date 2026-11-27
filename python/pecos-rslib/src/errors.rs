//! Shared Python exception mapping for simulation and decoder operations.

use pecos_core::errors::PecosError;
use pecos_decoders::batch::ShotDecodeError;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::{PyErr, Python};

/// The Python exception a Python-implemented decoder raised, carried unchanged in
/// [`DecoderError::Other`](pecos_decoder_core::DecoderError::Other).
fn python_exception(error: &pecos_decoder_core::DecoderError) -> Option<PyErr> {
    let pecos_decoder_core::DecoderError::Other(source) = error else {
        return None;
    };
    let exception = source.downcast_ref::<PyErr>()?;
    Some(Python::attach(|py| exception.clone_ref(py)))
}

/// Convert a failed shot. A Python exception the decoder raised reaches the caller
/// as the same exception object, with the shot index added as a note.
pub(crate) fn shot_decode_error(error: &ShotDecodeError) -> PyErr {
    let Some(exception) = python_exception(&error.source) else {
        return decoder_error(&error.source, error.to_string());
    };
    Python::attach(|py| {
        match exception.add_note(py, format!("decoder failed on shot {}", error.shot_index)) {
            Ok(()) => exception,
            // As in Python, a failing `add_note` replaces the exception it was
            // called on, which becomes the new exception's context.
            Err(note_error) => {
                note_error.set_context(py, Some(exception));
                note_error
            }
        }
    })
}

/// Classify the original decoder error while retaining boundary-specific context.
///
/// A Python exception a Python-implemented decoder raised is returned unchanged.
pub(crate) fn decoder_error(error: &pecos_decoder_core::DecoderError, message: String) -> PyErr {
    use pecos_decoder_core::DecoderError;
    if let Some(exception) = python_exception(error) {
        return exception;
    }
    match error {
        DecoderError::BackendUnavailable { family: "mwpf", .. } => PyValueError::new_err(
            "MWPF decoder is not available in this build. \
             Install cmake (run `pecos setup`) and rebuild. \
             See: https://github.com/PECOS-packages/PECOS/blob/dev/docs/user-guide/cmake-setup.md",
        ),
        // A backend missing from this build is reported as a configuration error, as before.
        DecoderError::BackendUnavailable { .. }
        | DecoderError::InvalidDimensions { .. }
        | DecoderError::InvalidDemSyntax(_)
        | DecoderError::InvalidConfiguration(_)
        | DecoderError::InvalidGraph(_)
        | DecoderError::MatrixError(_)
        | DecoderError::BatchSizeMismatch { .. }
        | DecoderError::InvalidSyndrome(_)
        | DecoderError::InvalidNodeIndex { .. }
        | DecoderError::InvalidEdge(_) => PyValueError::new_err(message),
        DecoderError::InternalError(_)
        | DecoderError::DecodingFailed(_)
        | DecoderError::FfiError(_)
        | DecoderError::ConvergenceFailure { .. }
        | DecoderError::NotImplemented(_)
        | DecoderError::IoError(_)
        | DecoderError::Other(_) => PyRuntimeError::new_err(message),
    }
}

pub(crate) fn simulation_error(error: PecosError, context: &str) -> PyErr {
    if let Some(reference) = result_reference_error(&error) {
        let message = format!("PHIR interpreter failed: {reference}");
        return if reference.wrong_type {
            PyTypeError::new_err(message)
        } else {
            PyValueError::new_err(message)
        };
    }
    match error {
        PecosError::Input(message) => PyValueError::new_err(format!("{context}: {message}")),
        error => PyRuntimeError::new_err(format!("{context}: {error}")),
    }
}

pub(crate) fn result_reference_error(
    error: &PecosError,
) -> Option<pecos_phir_json::v0_1::classical_validation::ResultReferenceError> {
    match error {
        PecosError::Input(message) => {
            pecos_phir_json::v0_1::classical_validation::ResultReferenceError::from_input_message(
                message,
            )
        }
        PecosError::WithContext { source, .. } => source
            .downcast_ref::<PecosError>()
            .and_then(result_reference_error),
        _ => None,
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
