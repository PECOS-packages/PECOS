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

#ifndef STATE_VECTOR_HPP
#define STATE_VECTOR_HPP

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecApplyMatrix

#include "Eigen/Dense"
#include "custatevec_workspace.hpp"
#include "utils.hpp"

/*
 *
 * Typedefs
 *
 */
using BasisBits = std::vector<int32_t>;


/*
 *
 * StateVector wrapper class
 *
 */
class StateVector 
{
    friend class Gate;

    // State vector on host
    Eigen::VectorXcd sv;

    // State vector (pointer) on device
    cuDoubleComplex *d_sv = nullptr;

    // Helper to get the cuda op codes for collapse operations
    custatevecCollapseOp_t get_collapse_op(bool collapse);

public:
    size_t num_bits;

    /*
     *
     * Constructors and destructor
     *
     */
    StateVector();
    StateVector(size_t num_bits);
    StateVector(const Eigen::VectorXcd v);
    ~StateVector();

    // Overload comparison to compare on host sv
    bool operator==(const StateVector& other) const
    {
        return (this->sv == other.sv);
    }

    /*
     *
     * Member functions
     *
     */
    // Set the on host state vector
    void set(const Eigen::VectorXcd v_new);
    // Get the on host state vector
    Eigen::VectorXcd get();

    // Get the size and size in memory
    size_t size();
    size_t mem_size();

    // Initiailze the state vector on the device to the zero state
    void init_on_device();
    
    // Copy, read, and free the state vector on the device
    void copy_to_device();
    void read_from_device();
    void free_on_device();

    // Measure on the Z-basis
    int32_t measure(CuStatevecWorkspace &workspace, const BasisBits &basis_bits, 
                         double randnum=0, bool collapse=true);

    void measure(CuStatevecWorkspace &workspace, int32_t &parity, 
                 const BasisBits &basis_bits, double randnum=0, 
                 bool collapse=true);

    // Batch measure multiple qubits on the Z-basis
    std::vector<int32_t> batch_measure(CuStatevecWorkspace &workspace, const BasisBits &bit_ordering, 
                            double randnum=0, bool collapse=true);

    void batch_measure(CuStatevecWorkspace &workspace, BasisBits &bit_string, 
                       const BasisBits &bit_ordering, double randnum=0, 
                       bool collapse=true);

    // Batch measure all qubits on the Z-basis
    void batch_measure_all(CuStatevecWorkspace &workspace, BasisBits &bit_string, 
                           double randnum=0, bool collapse=true);

    // Get the state vector probabilities
    std::vector<double> get_probabilities(CuStatevecWorkspace &workspace);

    // Reset to the zero state
    void reset(CuStatevecWorkspace &workspace);

    // Print the state vector in a nicely formatted manner to stdout
    void print();

};


/*
 *
 * StateVector helper functions
 *
 */
StateVector create_random_state_vector(size_t num_bits);
StateVector create_zero_state_vector(size_t num_bits);


#endif
