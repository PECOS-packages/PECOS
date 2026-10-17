//! Register-bound validation for simulator operands.

/// Check an operand before narrowing it or accessing simulator state.
///
/// # Panics
/// Panics if `index >= num_qubits`, naming the simulator, index, capacity and remedy.
#[inline]
pub fn check_qubit(index: usize, num_qubits: usize, simulator: &str) {
    assert!(
        index < num_qubits,
        "{simulator}: qubit {index} is out of range for a {num_qubits}-qubit register; use a qubit index below {num_qubits}, or construct the simulator with more qubits"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_last_operand() {
        check_qubit(16383, 16384, "GpuStab");
    }

    #[test]
    #[should_panic(
        expected = "GpuStab: qubit 16384 is out of range for a 16384-qubit register; use a qubit index below 16384, or construct the simulator with more qubits"
    )]
    fn first_invalid_operand() {
        check_qubit(16384, 16384, "GpuStab");
    }
}
