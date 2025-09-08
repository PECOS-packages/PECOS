#ifndef QUEST_FFI_H
#define QUEST_FFI_H

#include <memory>
#include <cstdint>
#include "rust/cxx.h"

// Include CXX-generated structs
#include "pecos-quest/src/bridge.rs.h"

// Simple functions that work with pointers to opaque handles
// The handles will be managed by the C++ implementation

// Environment management
uint8_t* quest_create_env();
void quest_destroy_env(uint8_t* env);
QuESTEnvInfo quest_get_env_info(uint8_t* env);
void quest_sync_env(uint8_t* env);

// Qureg creation and destruction
uint8_t* quest_create_qureg(uint8_t* env, int32_t num_qubits);
uint8_t* quest_create_density_qureg(uint8_t* env, int32_t num_qubits);
void quest_destroy_qureg(uint8_t* qureg);
uint8_t* quest_clone_qureg(uint8_t* qureg);
QuregInfo quest_get_qureg_info(uint8_t* qureg);

// State initialization
void quest_init_zero_state(uint8_t* qureg);
void quest_init_plus_state(uint8_t* qureg);
void quest_init_classical_state(uint8_t* qureg, int64_t state_ind);
void quest_init_pure_state(uint8_t* qureg, uint8_t* pure_qureg);
void quest_init_random_state(uint8_t* qureg, rust::Slice<const uint64_t> seed);

// Single-qubit gates
void quest_apply_pauli_x(uint8_t* qureg, int32_t qubit);
void quest_apply_pauli_y(uint8_t* qureg, int32_t qubit);
void quest_apply_pauli_z(uint8_t* qureg, int32_t qubit);
void quest_apply_hadamard(uint8_t* qureg, int32_t qubit);
void quest_apply_s_gate(uint8_t* qureg, int32_t qubit);
void quest_apply_t_gate(uint8_t* qureg, int32_t qubit);
void quest_apply_phase_shift(uint8_t* qureg, int32_t qubit, double angle);
void quest_apply_rotation_x(uint8_t* qureg, int32_t qubit, double angle);
void quest_apply_rotation_y(uint8_t* qureg, int32_t qubit, double angle);
void quest_apply_rotation_z(uint8_t* qureg, int32_t qubit, double angle);

// Two-qubit gates
void quest_apply_cnot(uint8_t* qureg, int32_t control, int32_t target);
void quest_apply_cz(uint8_t* qureg, int32_t control, int32_t target);
void quest_apply_swap(uint8_t* qureg, int32_t qubit1, int32_t qubit2);
void quest_apply_controlled_phase_shift(uint8_t* qureg, int32_t control, int32_t target, double angle);

// Multi-controlled gates
void quest_apply_multi_controlled_pauli_z(uint8_t* qureg, rust::Slice<const int32_t> controls, int32_t target);

// Measurements
int32_t quest_measure(uint8_t* qureg, int32_t qubit);
int32_t quest_measure_with_stats(uint8_t* qureg, int32_t qubit, double& outcome_prob);
double quest_calc_prob_of_outcome(uint8_t* qureg, int32_t qubit, int32_t outcome);
double quest_calc_total_prob(uint8_t* qureg);

// Amplitude access
double quest_get_real_amp(uint8_t* qureg, int64_t index);
double quest_get_imag_amp(uint8_t* qureg, int64_t index);
Complex quest_get_complex_amp(uint8_t* qureg, int64_t index);
double quest_get_prob_amp(uint8_t* qureg, int64_t index);

// State vector properties
int64_t quest_get_num_amps(uint8_t* qureg);
int32_t quest_get_num_qubits(uint8_t* qureg);
bool quest_is_density_matrix(uint8_t* qureg);

// Utility functions
Complex quest_calc_inner_product(uint8_t* qureg1, uint8_t* qureg2);
double quest_calc_fidelity(uint8_t* qureg1, uint8_t* qureg2);
double quest_calc_purity(uint8_t* qureg);

#endif // QUEST_FFI_H
