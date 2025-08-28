// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

#include "cxx_shim.h"

StateWrapper::StateWrapper(std::uint64_t num_qubits, std::int32_t reserve_buckets)
    : state(static_cast<int_num>(num_qubits), static_cast<int>(reserve_buckets)) {}

void StateWrapper::set_seed(std::uint32_t seed) {
    // Set the instance RNG seed
    state.set_seed(static_cast<unsigned int>(seed));
}

void StateWrapper::clear() {
    state.clear();
}

void StateWrapper::hadamard(std::uint64_t qubit) {
    state.hadamard(static_cast<int_num>(qubit));
}

void StateWrapper::bitflip(std::uint64_t qubit) {
    state.bitflip(static_cast<int_num>(qubit));
}

void StateWrapper::phaseflip(std::uint64_t qubit) {
    state.phaseflip(static_cast<int_num>(qubit));
}

void StateWrapper::Y(std::uint64_t qubit) {
    state.Y(static_cast<int_num>(qubit));
}

void StateWrapper::phaserot(std::uint64_t qubit) {
    state.phaserot(static_cast<int_num>(qubit));
}

void StateWrapper::SZdg(std::uint64_t qubit) {
    state.SZdg(static_cast<int_num>(qubit));
}

void StateWrapper::SY(std::uint64_t qubit) {
    state.SY(static_cast<int_num>(qubit));
}

void StateWrapper::SYdg(std::uint64_t qubit) {
    state.SYdg(static_cast<int_num>(qubit));
}

void StateWrapper::SX(std::uint64_t qubit) {
    state.SX(static_cast<int_num>(qubit));
}

void StateWrapper::SXdg(std::uint64_t qubit) {
    state.SXdg(static_cast<int_num>(qubit));
}

void StateWrapper::H2(std::uint64_t qubit) {
    state.H2(static_cast<int_num>(qubit));
}

void StateWrapper::H3(std::uint64_t qubit) {
    state.H3(static_cast<int_num>(qubit));
}

void StateWrapper::H4(std::uint64_t qubit) {
    state.H4(static_cast<int_num>(qubit));
}

void StateWrapper::H5(std::uint64_t qubit) {
    state.H5(static_cast<int_num>(qubit));
}

void StateWrapper::H6(std::uint64_t qubit) {
    state.H6(static_cast<int_num>(qubit));
}

void StateWrapper::F(std::uint64_t qubit) {
    state.F(static_cast<int_num>(qubit));
}

void StateWrapper::F2(std::uint64_t qubit) {
    state.F2(static_cast<int_num>(qubit));
}

void StateWrapper::F3(std::uint64_t qubit) {
    state.F3(static_cast<int_num>(qubit));
}

void StateWrapper::F4(std::uint64_t qubit) {
    state.F4(static_cast<int_num>(qubit));
}

void StateWrapper::Fdg(std::uint64_t qubit) {
    state.Fdg(static_cast<int_num>(qubit));
}

void StateWrapper::F2dg(std::uint64_t qubit) {
    state.F2dg(static_cast<int_num>(qubit));
}

void StateWrapper::F3dg(std::uint64_t qubit) {
    state.F3dg(static_cast<int_num>(qubit));
}

void StateWrapper::F4dg(std::uint64_t qubit) {
    state.F4dg(static_cast<int_num>(qubit));
}

void StateWrapper::cx(std::uint64_t control, std::uint64_t target) {
    // The C++ cx function uses confusing parameter names but actually expects (control, target)
    state.cx(static_cast<int_num>(control), static_cast<int_num>(target));
}

void StateWrapper::cy(std::uint64_t control, std::uint64_t target) {
    // CY = (I ⊗ SYdg) CX (I ⊗ SY)
    state.SYdg(static_cast<int_num>(target));
    state.cx(static_cast<int_num>(control), static_cast<int_num>(target));
    state.SY(static_cast<int_num>(target));
}

void StateWrapper::cz(std::uint64_t qubit1, std::uint64_t qubit2) {
    // CZ = H(qubit2) CX(qubit1, qubit2) H(qubit2)
    state.hadamard(static_cast<int_num>(qubit2));
    state.cx(static_cast<int_num>(qubit1), static_cast<int_num>(qubit2));
    state.hadamard(static_cast<int_num>(qubit2));
}

void StateWrapper::swap(std::uint64_t qubit1, std::uint64_t qubit2) {
    state.swap(static_cast<int_num>(qubit1), static_cast<int_num>(qubit2));
}

void StateWrapper::g2(std::uint64_t qubit1, std::uint64_t qubit2) {
    // G2 gate decomposition: H(q1), CX(q2, q1), CX(q1, q2), H(q2)
    state.hadamard(static_cast<int_num>(qubit1));
    state.cx(static_cast<int_num>(qubit2), static_cast<int_num>(qubit1));
    state.cx(static_cast<int_num>(qubit1), static_cast<int_num>(qubit2));
    state.hadamard(static_cast<int_num>(qubit2));
}

void StateWrapper::sxx(std::uint64_t qubit1, std::uint64_t qubit2) {
    // SXX = SX(q1).SX(q2).SYdg(q1).CX(q1, q2).SY(q1)
    state.SX(static_cast<int_num>(qubit1));
    state.SX(static_cast<int_num>(qubit2));
    state.SYdg(static_cast<int_num>(qubit1));
    state.cx(static_cast<int_num>(qubit1), static_cast<int_num>(qubit2));
    state.SY(static_cast<int_num>(qubit1));
}

void StateWrapper::sxxdg(std::uint64_t qubit1, std::uint64_t qubit2) {
    // SXXdg = X(q1).X(q2).SXX(q1, q2)
    state.bitflip(static_cast<int_num>(qubit1));
    state.bitflip(static_cast<int_num>(qubit2));
    sxx(qubit1, qubit2);  // Call the wrapper's sxx implementation
}

std::uint32_t StateWrapper::measure(std::uint64_t qubit, std::int32_t forced_outcome, bool collapse) {
    // Simple wrapper - just return the measurement outcome
    unsigned int outcome = state.measure(static_cast<int_num>(qubit), static_cast<int>(forced_outcome), collapse);
    return static_cast<std::uint32_t>(outcome);
}

std::uint64_t StateWrapper::get_num_qubits() const {
    return static_cast<std::uint64_t>(state.num_qubits);
}

bool StateWrapper::has_stab_x(std::uint64_t gen_id, std::uint64_t qubit) const {
    const auto& row_set = state.stabs.row_x[static_cast<int_num>(gen_id)];
    return row_set.count(static_cast<int_num>(qubit)) > 0;
}

bool StateWrapper::has_stab_z(std::uint64_t gen_id, std::uint64_t qubit) const {
    const auto& row_set = state.stabs.row_z[static_cast<int_num>(gen_id)];
    return row_set.count(static_cast<int_num>(qubit)) > 0;
}

bool StateWrapper::has_destab_x(std::uint64_t gen_id, std::uint64_t qubit) const {
    const auto& row_set = state.destabs.row_x[static_cast<int_num>(gen_id)];
    return row_set.count(static_cast<int_num>(qubit)) > 0;
}

bool StateWrapper::has_destab_z(std::uint64_t gen_id, std::uint64_t qubit) const {
    const auto& row_set = state.destabs.row_z[static_cast<int_num>(gen_id)];
    return row_set.count(static_cast<int_num>(qubit)) > 0;
}

bool StateWrapper::get_sign_minus(std::uint64_t gen_id) const {
    return state.signs_minus.count(static_cast<int_num>(gen_id)) > 0;
}

bool StateWrapper::get_sign_i(std::uint64_t gen_id) const {
    return state.signs_i.count(static_cast<int_num>(gen_id)) > 0;
}

// Factory function
std::unique_ptr<StateWrapper> create_state_wrapper(std::uint64_t num_qubits, std::int32_t reserve_buckets) {
    return std::make_unique<StateWrapper>(num_qubits, reserve_buckets);
}
