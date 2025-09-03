#pragma once
#include <memory>
#include <cstdint>
#include <array>

// Forward declaration of Qulacs QuantumStateCpu
class QuantumStateCpu;

// Wrapper class for C++/Rust interop
class QulacsState {
private:
    std::unique_ptr<QuantumStateCpu> state;
    uint32_t rng_seed;  // Store seed for measurements

public:
    QulacsState(size_t n_qubits);
    ~QulacsState();

    QuantumStateCpu* get_state() { return state.get(); }
    const QuantumStateCpu* get_state() const { return state.get(); }

    void set_rng_seed(uint32_t seed) { rng_seed = seed; }
    uint32_t get_rng_seed() const { return rng_seed; }
};

// Factory functions
std::unique_ptr<QulacsState> create_quantum_state(size_t n_qubits);
std::unique_ptr<QulacsState> clone_quantum_state(const QulacsState& state);

// RNG management
void set_seed(QulacsState& state, uint32_t seed);

// State operations
void reset(QulacsState& state);
void set_zero_state(QulacsState& state);
void set_computational_basis(QulacsState& state, uint64_t basis);

// Get state information
size_t get_num_qubits(const QulacsState& state);
double get_squared_norm(const QulacsState& state);
size_t get_vector_size(const QulacsState& state);
std::array<double, 2> get_amplitude(const QulacsState& state, uint64_t index);
double get_marginal_probability(const QulacsState& state, size_t qubit);

// Single-qubit gates
void apply_x(QulacsState& state, size_t qubit);
void apply_y(QulacsState& state, size_t qubit);
void apply_z(QulacsState& state, size_t qubit);
void apply_h(QulacsState& state, size_t qubit);
void apply_s(QulacsState& state, size_t qubit);
void apply_sdag(QulacsState& state, size_t qubit);
void apply_t(QulacsState& state, size_t qubit);
void apply_tdag(QulacsState& state, size_t qubit);
void apply_sqrt_x(QulacsState& state, size_t qubit);
void apply_sqrt_xdag(QulacsState& state, size_t qubit);
void apply_sqrt_y(QulacsState& state, size_t qubit);
void apply_sqrt_ydag(QulacsState& state, size_t qubit);

// Rotation gates
void apply_rx(QulacsState& state, size_t qubit, double angle);
void apply_ry(QulacsState& state, size_t qubit, double angle);
void apply_rz(QulacsState& state, size_t qubit, double angle);

// Global phase
void apply_global_phase(QulacsState& state, double angle);

// Two-qubit gates
void apply_cnot(QulacsState& state, size_t control, size_t target);
void apply_cz(QulacsState& state, size_t control, size_t target);
void apply_swap(QulacsState& state, size_t qubit1, size_t qubit2);

// Measurement
uint8_t measure_z(QulacsState& state, size_t qubit);
