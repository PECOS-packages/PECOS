pub(crate) use pecos_simulators::qubit_validation::check_qubit;

pub(crate) fn checked_index(index: usize, num_qubits: usize, simulator: &str) -> u32 {
    check_qubit(index, num_qubits, simulator);
    u32::try_from(index).expect("GPU register sizes are representable as u32")
}

pub(crate) const PACKED_QUBIT_CAPACITY: usize = 1 << 14;

/// Register owners establish the 14-bit capacity at construction. Operand bounds
/// are checked here even when a public batch has already been validated.
pub(crate) fn pack_gate(
    gate_type: u32,
    target: usize,
    control: Option<usize>,
    num_qubits: usize,
    simulator: &str,
) -> u32 {
    let target = checked_index(target, num_qubits, simulator);
    let control = control.map(|q| checked_index(q, num_qubits, simulator));
    (gate_type & 0xF) | (target << 4) | (control.unwrap_or(0) << 18)
}

#[cfg(test)]
pub(crate) mod gpu_free_tests {
    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    pub(crate) fn assert_bounds_panic(
        simulator: &str,
        index: usize,
        capacity: usize,
        action: impl FnOnce(),
    ) {
        let panic =
            catch_unwind(AssertUnwindSafe(action)).expect_err("out-of-range operands must panic");
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .expect("panic must have a message");
        assert_eq!(
            message,
            format!(
                "{simulator}: qubit {index} is out of range for a {capacity}-qubit register; use a qubit index below {capacity}, or construct the simulator with more qubits"
            )
        );
    }

    #[test]
    fn bounds_helper_rejects_before_narrowing() {
        for simulator in [
            "GpuStab",
            "GpuStabMulti",
            "GpuStateVec32",
            "GpuStateVec64",
            "GpuPauliProp",
            "CuStateVec",
        ] {
            for (index, capacity) in [
                (0, 0),
                (100, 100),
                (120, 100),
                (16383, 16383),
                (16384, 16384),
                (usize::MAX, 100),
            ] {
                assert_bounds_panic(simulator, index, capacity, || {
                    checked_index(index, capacity, simulator);
                });
            }
            if let Ok(large) = usize::try_from(u64::from(u32::MAX) + 1) {
                assert_bounds_panic(simulator, large, 100, || {
                    checked_index(large, 100, simulator);
                });
            }
        }
        assert_eq!(checked_index(16383, 16384, "GpuStab"), 16383);
    }

    #[test]
    fn packed_operands_keep_all_fourteen_bits() {
        assert_eq!(
            pack_gate(6, 16383, Some(16382), 16384, "GpuStab"),
            6 | (16383 << 4) | (16382 << 18)
        );
        assert_eq!(pack_gate(0, 16383, None, 16384, "GpuStab"), 16383 << 4);
    }

    #[test]
    fn packed_operands_cannot_alias() {
        for simulator in ["GpuStab", "GpuStabMulti"] {
            for bad in [100, 16383, 16384, usize::MAX] {
                assert_bounds_panic(simulator, bad, 100, || {
                    pack_gate(0, bad, None, 100, simulator);
                });
                assert_bounds_panic(simulator, bad, 100, || {
                    pack_gate(6, 0, Some(bad), 100, simulator);
                });
                assert_bounds_panic(simulator, bad, 100, || {
                    pack_gate(6, bad, Some(0), 100, simulator);
                });
            }
            if let Ok(bad) = usize::try_from(u64::from(u32::MAX) + 1) {
                assert_bounds_panic(simulator, bad, 100, || {
                    pack_gate(0, bad, None, 100, simulator);
                });
            }
        }
    }
}
