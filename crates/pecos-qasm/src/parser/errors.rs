use pecos_core::errors::PecosError;

/// Common error helpers for the QASM parser
const QASM_OPERATION: &str = "QASM operation";

/// Create an invalid operation error
pub fn invalid_operation(reason: impl Into<String>) -> PecosError {
    PecosError::CompileInvalidOperation {
        operation: QASM_OPERATION.to_string(),
        reason: reason.into(),
    }
}

/// Create an error for unknown register
#[must_use]
pub fn unknown_register(reg_type: &str, name: &str) -> PecosError {
    invalid_operation(format!("Unknown {reg_type} register: {name}"))
}

/// Create an error for an undeclared classical register.
#[must_use]
pub fn unknown_classical_register(name: &str) -> PecosError {
    invalid_operation(format!(
        "Unknown classical register: {name}. Declare '{name}' with creg before using it"
    ))
}

/// Create an error for register index out of bounds
#[must_use]
pub fn index_out_of_bounds(reg_name: &str, idx: usize, size: usize) -> PecosError {
    invalid_operation(format!(
        "Register index out of bounds: {reg_name}[{idx}] (register size: {size})"
    ))
}

/// Create an error for a bit outside a classical register's declaration.
#[must_use]
pub fn classical_bit_out_of_bounds(name: &str, index: usize, size: usize) -> PecosError {
    invalid_operation(format!(
        "Classical bit '{name}[{index}]' is out of bounds for register size {size}. \
         Declare '{name}' with creg large enough to contain bit {index}"
    ))
}

/// Create an error for undefined gate
#[must_use]
pub fn undefined_gate(name: &str) -> PecosError {
    invalid_operation(format!(
        "Undefined gate '{name}' - gate is neither native nor user-defined. Did you forget to include qelib1.inc?"
    ))
}

/// Create an error for wrong number of gate parameters
#[must_use]
pub fn wrong_param_count(gate: &str, expected: usize, actual: usize) -> PecosError {
    invalid_operation(format!(
        "Gate '{gate}' expects {expected} parameters but got {actual}"
    ))
}

/// Create an error for wrong number of qubits
#[must_use]
pub fn wrong_qubit_count(gate: &str, expected: usize, actual: usize) -> PecosError {
    invalid_operation(format!(
        "Gate '{gate}' expects {expected} qubits but got {actual}"
    ))
}

/// Create an error for register size mismatch
#[must_use]
pub fn register_size_mismatch(operation: &str, details: &str) -> PecosError {
    invalid_operation(format!("Register size mismatch in {operation}: {details}"))
}
