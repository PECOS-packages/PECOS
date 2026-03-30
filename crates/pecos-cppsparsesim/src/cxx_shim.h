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

#pragma once

#include "sparsesim.h"
#include <memory>
#include <cstdint>

// Wrapper class that cxx can understand
class StateWrapper {
private:
    State state;

public:
    StateWrapper(std::uint64_t num_qubits, std::int32_t reserve_buckets);
    void set_seed(std::uint32_t seed);

    void clear();
    void hadamard(std::uint64_t qubit);
    void bitflip(std::uint64_t qubit);
    void phaseflip(std::uint64_t qubit);
    void Y(std::uint64_t qubit);
    void phaserot(std::uint64_t qubit);
    void SZdg(std::uint64_t qubit);
    void SY(std::uint64_t qubit);
    void SYdg(std::uint64_t qubit);
    void SX(std::uint64_t qubit);
    void SXdg(std::uint64_t qubit);
    void H2(std::uint64_t qubit);
    void H3(std::uint64_t qubit);
    void H4(std::uint64_t qubit);
    void H5(std::uint64_t qubit);
    void H6(std::uint64_t qubit);
    void F(std::uint64_t qubit);
    void F2(std::uint64_t qubit);
    void F3(std::uint64_t qubit);
    void F4(std::uint64_t qubit);
    void Fdg(std::uint64_t qubit);
    void F2dg(std::uint64_t qubit);
    void F3dg(std::uint64_t qubit);
    void F4dg(std::uint64_t qubit);
    void cx(std::uint64_t control, std::uint64_t target);
    void cy(std::uint64_t control, std::uint64_t target);
    void cz(std::uint64_t qubit1, std::uint64_t qubit2);
    void swap(std::uint64_t qubit1, std::uint64_t qubit2);
    void g2(std::uint64_t qubit1, std::uint64_t qubit2);
    void sxx(std::uint64_t qubit1, std::uint64_t qubit2);
    void sxxdg(std::uint64_t qubit1, std::uint64_t qubit2);
    std::uint32_t mz(std::uint64_t qubit, std::int32_t forced_outcome, bool collapse);

    // Tableau access methods
    std::uint64_t get_num_qubits() const;
    bool has_stab_x(std::uint64_t gen_id, std::uint64_t qubit) const;
    bool has_stab_z(std::uint64_t gen_id, std::uint64_t qubit) const;
    bool has_destab_x(std::uint64_t gen_id, std::uint64_t qubit) const;
    bool has_destab_z(std::uint64_t gen_id, std::uint64_t qubit) const;
    bool get_sign_minus(std::uint64_t gen_id) const;
    bool get_sign_i(std::uint64_t gen_id) const;

    // Get access to internal state for checking deterministic measurement
    const State& get_state() const { return state; }
};

// Factory function
std::unique_ptr<StateWrapper> create_state_wrapper(std::uint64_t num_qubits, std::int32_t reserve_buckets);
