// Copyright 2022 The PECOS developers
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
// the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

// Initial author: Tyson Lawrence

#include "custatevec_wrapper.h"

#include "quantum_volume.hpp"
#include "utils.hpp"

int run_quantum_volume(int num_qubits, int num_shots, double results[], int len_results)
{
    // Initialize the workspace
    CuStatevecWorkspace ws;

    // Initialize the state vector to |000...0>
    StateVector sv(num_qubits);
    sv.init_on_device();

    // Create, apply, and free the QV circuit
    QuantumVolume qv(num_qubits);
    qv.copy_to_device();
    qv.apply(sv,ws);
    qv.free_on_device();

    std::vector<double> probs = sv.get_probabilities(ws);
    for (int n = 0; n < len_results; n++)
        results[n] = probs[n];

    // // Make measurements for the shots
    // constexpr bool collapse = false;
    // double randnum = rand_in_range(0.0, 1.0);
    // // std::vector<int32_t> bit_string(num_qubits, -1);
    // BasisBits bit_string(num_qubits, -1);
    // for (size_t shot = 0; shot < num_shots; shot++) {
    //     randnum = rand_in_range(0.0, 1.0);
    //     sv.batch_measure_all(ws, bit_string, randnum, collapse);
    //     results[shot] = compress_bit_string(bit_string);
    // }

    sv.read_from_device();
    sv.free_on_device();

    return 0;
}

int run_quantum_volume_2(int num_qubits, double angles[][8][3], int targets[], double results[], int len_results)
{
    // Initialize the workspace
    CuStatevecWorkspace ws;

    // Initialize the state vector to |000...0>
    StateVector sv = create_zero_state_vector(num_qubits);
    sv.copy_to_device();

    // Convert the C-array to a Targets vector.  
    Targets _targets(targets, targets+num_qubits);

    // Create, apply, and free the QV circuit
    QuantumVolume qv(num_qubits, angles, _targets);
    qv.copy_to_device();
    qv.apply(sv,ws);
    qv.free_on_device();

    std::vector<double> probs = sv.get_probabilities(ws);
    for (int n = 0; n < len_results; n++)
        results[n] = probs[n];

    sv.read_from_device();
    sv.free_on_device();

    return 0;
}