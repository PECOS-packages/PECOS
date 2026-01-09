//! CXX bridge for Qulacs C++ library bindings.

#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("qulacs_wrapper.h");

        type QulacsState;

        // Constructor and destructor
        fn create_quantum_state(num_qubits: usize) -> UniquePtr<QulacsState>;
        fn clone_quantum_state(state: &QulacsState) -> UniquePtr<QulacsState>;

        // RNG management
        fn set_seed(state: Pin<&mut QulacsState>, seed: u32);

        // State operations
        fn reset(state: Pin<&mut QulacsState>);
        #[allow(dead_code)]
        fn set_zero_state(state: Pin<&mut QulacsState>);
        fn set_computational_basis(state: Pin<&mut QulacsState>, basis: u64);

        // Get state information
        #[allow(dead_code)]
        fn get_num_qubits(state: &QulacsState) -> usize;
        #[allow(dead_code)]
        fn get_squared_norm(state: &QulacsState) -> f64;
        fn get_vector_size(state: &QulacsState) -> usize;
        fn get_amplitude(state: &QulacsState, index: u64) -> [f64; 2];
        fn get_marginal_probability(state: &QulacsState, qubit: usize) -> f64;

        // Single-qubit gates
        fn apply_x(state: Pin<&mut QulacsState>, qubit: usize);
        fn apply_y(state: Pin<&mut QulacsState>, qubit: usize);
        fn apply_z(state: Pin<&mut QulacsState>, qubit: usize);
        fn apply_h(state: Pin<&mut QulacsState>, qubit: usize);
        fn apply_s(state: Pin<&mut QulacsState>, qubit: usize);
        fn apply_sdag(state: Pin<&mut QulacsState>, qubit: usize);
        fn apply_t(state: Pin<&mut QulacsState>, qubit: usize);
        fn apply_tdag(state: Pin<&mut QulacsState>, qubit: usize);

        // NOTE: sqrt_x, sqrt_xdag, sqrt_y, sqrt_ydag removed - we use trait
        // decompositions instead for consistency with StateVec.

        // Rotation gates
        fn apply_rx(state: Pin<&mut QulacsState>, qubit: usize, angle: f64);
        fn apply_ry(state: Pin<&mut QulacsState>, qubit: usize, angle: f64);
        fn apply_rz(state: Pin<&mut QulacsState>, qubit: usize, angle: f64);

        // Global phase
        #[allow(dead_code)]
        fn apply_global_phase(state: Pin<&mut QulacsState>, angle: f64);

        // Two-qubit gates
        fn apply_cnot(state: Pin<&mut QulacsState>, control: usize, target: usize);
        fn apply_cz(state: Pin<&mut QulacsState>, control: usize, target: usize);
        fn apply_swap(state: Pin<&mut QulacsState>, qubit1: usize, qubit2: usize);

        // Measurement
        fn measure_z(state: Pin<&mut QulacsState>, qubit: usize) -> u8;
    }
}
