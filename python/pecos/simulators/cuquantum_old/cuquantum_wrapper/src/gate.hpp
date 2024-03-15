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

#ifndef GATE_HPP
#define GATE_HPP

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecApplyMatrix

#include "state_vector.hpp"
#include "custatevec_workspace.hpp"
#include "utils.hpp"


/*
 *
 * Typedefs
 *
 */
using Targets = std::vector<int32_t>;
using Controls = Targets;


/*
 *
 * Gate wrapper class
 *
 */
class Gate
{
    // Gate matrix on host
    Eigen::MatrixXcd mat;

    // Gate matrix (pointer) on device
    void *d_mat = nullptr;

public:
    // Control and target qubits for gate
    Controls controls;
    Targets targets;

    /*
     *
     * Constructors
     *
     */
    Gate();
    Gate(size_t rows, size_t cols);
    Gate(const Eigen::MatrixXcd m);

    /*
     *
     * Member functions
     *
     */
    // Set the on host gate
    void set(const Eigen::MatrixXcd mat_new);

    // Get the size and size in memory
    size_t size();
    size_t mem_size();

    // Copy and free the state vector on the device
    void copy_to_device();
    void free_on_device();

    // Apply the gate to the state vector on the host
    void apply(StateVector &sv);

    // Apply the gate to the state vector on the device
    void apply(StateVector &sv, CuStatevecWorkspace &workspace,
               bool adjoint=false);

    void apply(StateVector &sv, CuStatevecWorkspace &workspace,
               const Controls &cs, const Targets &ts, bool adjoint=false);

    // Print the gate to stdout
    void print();

};


/*
 *
 * Factory functions for specific gates
 *
 */
constexpr double S2 = 0.707106781186547524401; // 1/sqrt(2)

// Single qubit gates
Gate Rx(double theta);
Gate Ry(double theta);
Gate Rz(double theta);
Gate U1q(double theta, double phi);
Gate U(double theta, double phi, double lambda);

Gate create_random_U();

Gate Hadamard();

Gate PauliX();
Gate PauliY();
Gate PauliZ();

// Two qubit gates
Gate SqrtZZ();
Gate RZZ(double theta);
Gate CNOT();
Gate Toffoli();

#endif
