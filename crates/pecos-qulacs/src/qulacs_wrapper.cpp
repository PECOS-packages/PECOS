#include "qulacs_wrapper.h"
#include "cppsim/state.hpp"
#include "cppsim/gate_factory.hpp"
#include <complex>
#include <array>

// Constructor and destructor
QulacsState::QulacsState(size_t n_qubits)
    : state(std::make_unique<QuantumStateCpu>(n_qubits)), rng_seed(0) {
}

QulacsState::~QulacsState() = default;

// Factory functions
std::unique_ptr<QulacsState> create_quantum_state(size_t n_qubits) {
    return std::make_unique<QulacsState>(n_qubits);
}

std::unique_ptr<QulacsState> clone_quantum_state(const QulacsState& state) {
    size_t n_qubits = state.get_state()->qubit_count;
    auto new_state = std::make_unique<QulacsState>(n_qubits);

    // Copy the quantum state using Qulacs' copy functionality
    new_state->get_state()->load(state.get_state());

    // Copy the RNG seed as well
    new_state->set_rng_seed(state.get_rng_seed());

    return new_state;
}

// State operations
void reset(QulacsState& state) {
    state.get_state()->set_zero_state();
}

void set_zero_state(QulacsState& state) {
    state.get_state()->set_zero_state();
}

void set_computational_basis(QulacsState& state, uint64_t basis) {
    state.get_state()->set_computational_basis(basis);
}

// Get state information
size_t get_num_qubits(const QulacsState& state) {
    return state.get_state()->qubit_count;
}

double get_squared_norm(const QulacsState& state) {
    return state.get_state()->get_squared_norm();
}

size_t get_vector_size(const QulacsState& state) {
    return state.get_state()->dim;
}

std::array<double, 2> get_amplitude(const QulacsState& state, uint64_t index) {
    // Access the raw data and get the amplitude directly
    auto* data = state.get_state()->data_cpp();
    auto amp = data[index];
    return {amp.real(), amp.imag()};
}

double get_marginal_probability(const QulacsState& state, size_t qubit) {
    return state.get_state()->get_zero_probability((UINT)qubit);
}

// Single-qubit gates - using Qulacs gate functions
void apply_x(QulacsState& state, size_t qubit) {
    auto gate = gate::X(qubit);
    gate->update_quantum_state(state.get_state());
    delete gate;
}

void apply_y(QulacsState& state, size_t qubit) {
    auto gate = gate::Y(qubit);
    gate->update_quantum_state(state.get_state());
    delete gate;
}

void apply_z(QulacsState& state, size_t qubit) {
    auto gate = gate::Z(qubit);
    gate->update_quantum_state(state.get_state());
    delete gate;
}

void apply_h(QulacsState& state, size_t qubit) {
    auto gate = gate::H(qubit);
    gate->update_quantum_state(state.get_state());
    delete gate;
}

void apply_s(QulacsState& state, size_t qubit) {
    auto gate = gate::S(qubit);
    gate->update_quantum_state(state.get_state());
    delete gate;
}

void apply_sdag(QulacsState& state, size_t qubit) {
    auto gate = gate::Sdag(qubit);
    gate->update_quantum_state(state.get_state());
    delete gate;
}

void apply_t(QulacsState& state, size_t qubit) {
    auto gate = gate::T(qubit);
    gate->update_quantum_state(state.get_state());
    delete gate;
}

void apply_tdag(QulacsState& state, size_t qubit) {
    auto gate = gate::Tdag(qubit);
    gate->update_quantum_state(state.get_state());
    delete gate;
}

void apply_sqrt_x(QulacsState& state, size_t qubit) {
    auto gate = gate::sqrtX(qubit);
    gate->update_quantum_state(state.get_state());
    delete gate;
}

void apply_sqrt_xdag(QulacsState& state, size_t qubit) {
    auto gate = gate::sqrtXdag(qubit);
    gate->update_quantum_state(state.get_state());
    delete gate;
}

void apply_sqrt_y(QulacsState& state, size_t qubit) {
    auto gate = gate::sqrtY(qubit);
    gate->update_quantum_state(state.get_state());
    delete gate;
}

void apply_sqrt_ydag(QulacsState& state, size_t qubit) {
    auto gate = gate::sqrtYdag(qubit);
    gate->update_quantum_state(state.get_state());
    delete gate;
}

// Rotation gates
// Note: Qulacs uses opposite sign convention, so we negate the angle
void apply_rx(QulacsState& state, size_t qubit, double angle) {
    auto gate = gate::RX(qubit, -angle);
    gate->update_quantum_state(state.get_state());
    delete gate;
}

void apply_ry(QulacsState& state, size_t qubit, double angle) {
    auto gate = gate::RY(qubit, -angle);
    gate->update_quantum_state(state.get_state());
    delete gate;
}

void apply_rz(QulacsState& state, size_t qubit, double angle) {
    auto gate = gate::RZ(qubit, -angle);
    gate->update_quantum_state(state.get_state());
    delete gate;
}

void apply_global_phase(QulacsState& state, double angle) {
    // Apply a global phase e^(i*angle) to all amplitudes
    auto* data = state.get_state()->data_cpp();
    size_t dim = state.get_state()->dim;
    std::complex<double> phase = std::exp(std::complex<double>(0, angle));

    for (size_t i = 0; i < dim; ++i) {
        data[i] *= phase;
    }
}

// Two-qubit gates
void apply_cnot(QulacsState& state, size_t control, size_t target) {
    auto gate = gate::CNOT(control, target);
    gate->update_quantum_state(state.get_state());
    delete gate;
}

void apply_cz(QulacsState& state, size_t control, size_t target) {
    auto gate = gate::CZ(control, target);
    gate->update_quantum_state(state.get_state());
    delete gate;
}

void apply_swap(QulacsState& state, size_t qubit1, size_t qubit2) {
    auto gate = gate::SWAP(qubit1, qubit2);
    gate->update_quantum_state(state.get_state());
    delete gate;
}

// RNG management
void set_seed(QulacsState& state, uint32_t seed) {
    // Store the seed to use when sampling
    state.set_rng_seed(seed);
}

// Measurement
uint8_t measure_z(QulacsState& state, size_t qubit) {

    // Use Qulacs' built-in sampling to get a measurement outcome
    auto* cpu_state = dynamic_cast<QuantumStateCpu*>(state.get_state());
    if (cpu_state) {
        // Sample one outcome using Qulacs' sampling with our stored seed
        // Note: We increment the seed after each use to get different results
        uint32_t current_seed = state.get_rng_seed();
        state.set_rng_seed(current_seed + 1);  // Increment for next measurement

        auto samples = cpu_state->sampling(1, current_seed);
        bool outcome = (samples[0] >> qubit) & 1;

        // Manually collapse the state by zeroing out incompatible amplitudes
        auto* data = cpu_state->data_cpp();
        double norm_factor = 0.0;

        // First pass: zero out incompatible amplitudes and calculate normalization
        for (ITYPE i = 0; i < cpu_state->dim; ++i) {
            bool state_bit = (i >> qubit) & 1;
            if (state_bit != outcome) {
                data[i] = CPPCTYPE(0.0, 0.0);
            } else {
                norm_factor += std::norm(data[i]);
            }
        }

        // Second pass: normalize remaining amplitudes
        if (norm_factor > 1e-15) {
            double inv_norm = 1.0 / std::sqrt(norm_factor);
            for (ITYPE i = 0; i < cpu_state->dim; ++i) {
                bool state_bit = (i >> qubit) & 1;
                if (state_bit == outcome) {
                    data[i] *= inv_norm;
                }
            }
        }

        return outcome ? 1 : 0;
    }

    // Fallback: just return 0
    return 0;
}
