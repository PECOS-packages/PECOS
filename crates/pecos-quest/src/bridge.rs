//! CXX bridge definitions for `QuEST` simulator

#[cxx::bridge]
pub mod ffi {
    // QuEST environment struct
    #[derive(Debug, Clone)]
    struct QuESTEnvInfo {
        pub is_multithreaded: bool,
        pub is_gpu_accelerated: bool,
        pub is_distributed: bool,
        pub rank: i32,
        pub num_nodes: i32,
    }

    // Qureg info struct for reporting parameters
    #[derive(Debug, Clone)]
    struct QuregInfo {
        pub num_qubits: i32,
        pub num_amps: i64,
        pub is_density_matrix: bool,
    }

    // Complex number representation
    #[derive(Debug, Clone, Copy)]
    struct Complex {
        pub real: f64,
        pub imag: f64,
    }

    #[allow(clippy::missing_safety_doc)]
    unsafe extern "C++" {
        include!("quest_ffi.h");

        // Environment management
        #[must_use]
        fn quest_create_env() -> *mut u8;
        unsafe fn quest_destroy_env(env: *mut u8);
        unsafe fn quest_get_env_info(env: *mut u8) -> QuESTEnvInfo;
        unsafe fn quest_sync_env(env: *mut u8);

        // Qureg creation and destruction
        unsafe fn quest_create_qureg(env: *mut u8, num_qubits: i32) -> *mut u8;
        unsafe fn quest_create_density_qureg(env: *mut u8, num_qubits: i32) -> *mut u8;
        unsafe fn quest_destroy_qureg(qureg: *mut u8);
        unsafe fn quest_clone_qureg(qureg: *mut u8) -> *mut u8;
        unsafe fn quest_get_qureg_info(qureg: *mut u8) -> QuregInfo;

        // State initialization
        unsafe fn quest_init_zero_state(qureg: *mut u8);
        unsafe fn quest_init_plus_state(qureg: *mut u8);
        unsafe fn quest_init_classical_state(qureg: *mut u8, state_ind: i64);
        unsafe fn quest_init_pure_state(qureg: *mut u8, pure_qureg: *mut u8);
        unsafe fn quest_init_random_state(qureg: *mut u8, seed: &[u64]);

        // Single-qubit gates
        unsafe fn quest_apply_pauli_x(qureg: *mut u8, qubit: i32);
        unsafe fn quest_apply_pauli_y(qureg: *mut u8, qubit: i32);
        unsafe fn quest_apply_pauli_z(qureg: *mut u8, qubit: i32);
        unsafe fn quest_apply_hadamard(qureg: *mut u8, qubit: i32);
        unsafe fn quest_apply_s_gate(qureg: *mut u8, qubit: i32);
        unsafe fn quest_apply_t_gate(qureg: *mut u8, qubit: i32);
        unsafe fn quest_apply_phase_shift(qureg: *mut u8, qubit: i32, angle: f64);
        unsafe fn quest_apply_rotation_x(qureg: *mut u8, qubit: i32, angle: f64);
        unsafe fn quest_apply_rotation_y(qureg: *mut u8, qubit: i32, angle: f64);
        unsafe fn quest_apply_rotation_z(qureg: *mut u8, qubit: i32, angle: f64);

        // Two-qubit gates
        unsafe fn quest_apply_cnot(qureg: *mut u8, control: i32, target: i32);
        unsafe fn quest_apply_cz(qureg: *mut u8, control: i32, target: i32);
        unsafe fn quest_apply_swap(qureg: *mut u8, qubit1: i32, qubit2: i32);
        unsafe fn quest_apply_controlled_phase_shift(
            qureg: *mut u8,
            control: i32,
            target: i32,
            angle: f64,
        );

        // Multi-controlled gates
        unsafe fn quest_apply_multi_controlled_pauli_z(
            qureg: *mut u8,
            controls: &[i32],
            target: i32,
        );

        // Measurements
        unsafe fn quest_measure(qureg: *mut u8, qubit: i32) -> i32;
        unsafe fn quest_measure_with_stats(
            qureg: *mut u8,
            qubit: i32,
            outcome_prob: &mut f64,
        ) -> i32;
        unsafe fn quest_calc_prob_of_outcome(qureg: *mut u8, qubit: i32, outcome: i32) -> f64;
        unsafe fn quest_apply_forced_measurement(qureg: *mut u8, qubit: i32, outcome: i32) -> f64;
        unsafe fn quest_calc_total_prob(qureg: *mut u8) -> f64;

        // Amplitude access
        unsafe fn quest_get_real_amp(qureg: *mut u8, index: i64) -> f64;
        unsafe fn quest_get_imag_amp(qureg: *mut u8, index: i64) -> f64;
        unsafe fn quest_get_complex_amp(qureg: *mut u8, index: i64) -> Complex;
        unsafe fn quest_get_prob_amp(qureg: *mut u8, index: i64) -> f64;

        // State vector properties
        unsafe fn quest_get_num_amps(qureg: *mut u8) -> i64;
        unsafe fn quest_get_num_qubits(qureg: *mut u8) -> i32;
        unsafe fn quest_is_density_matrix(qureg: *mut u8) -> bool;

        // Utility functions
        unsafe fn quest_calc_inner_product(qureg1: *mut u8, qureg2: *mut u8) -> Complex;
        unsafe fn quest_calc_fidelity(qureg1: *mut u8, qureg2: *mut u8) -> f64;
        unsafe fn quest_calc_purity(qureg: *mut u8) -> f64;
    }
}
