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

#include <iostream>
#include <iomanip>
#include <complex>
#include <numeric>

#include "state_vector.hpp"
#include "custom_kernels.hpp"

/*
 *
 * Constructors
 *
 */
StateVector::StateVector() :
    sv(Eigen::VectorXcd())
    {}

StateVector::StateVector(size_t num_bits) :
    sv(Eigen::VectorXcd(static_cast<size_t>(std::pow(2,num_bits)))),
    num_bits(num_bits)
    {}

StateVector::StateVector(const Eigen::VectorXcd sv) :
    sv(sv),
    num_bits(get_power_of_two_exponent(sv.size()))
    {}

StateVector::~StateVector()
{
    free_on_device();
}

/*
 *
 * Member functions
 *
 */
void StateVector::set(const Eigen::VectorXcd sv_new)
{
    sv = sv_new;
}

Eigen::VectorXcd StateVector::get()
{
    return sv;
}

size_t StateVector::size()
{
    return sv.size();
}

size_t StateVector::mem_size()
{
    return size() * sizeof(cuDoubleComplex);
}

void StateVector::init_on_device()
{
    // Allocate space on the device
    cudaError_t status = cudaMalloc((void **)&d_sv, sizeof(cuDoubleComplex)*size());
    if(status != cudaSuccess)
        throw std::runtime_error("cudaMalloc error");

    // TODO does the following need to be const?
    constexpr int32_t threads_per_block = 256;   // number of threads per CUDA thread block, typically 128/256/512/1024.
    uint32_t n_blocks = (size() + threads_per_block - 1) / threads_per_block;   // ceiling divide
    initialize_sv(d_sv, size());   // launch the CUDA kernel on the GPU
}

void StateVector::copy_to_device()
{
    // Allocate space on the device
    cudaError_t status = cudaMalloc((void **)&d_sv, mem_size());
    if(status != cudaSuccess)
        throw std::runtime_error("cudaMalloc error");

    // Copy state vector to device
    cudaMemcpy(d_sv, sv.data(), mem_size(), cudaMemcpyHostToDevice);
}

void StateVector::read_from_device()
{
    // Copy state vector back to host
    cudaMemcpy(sv.data(), d_sv, mem_size(), cudaMemcpyDeviceToHost);
}

void StateVector::free_on_device()
{
    // Free memory on device
    cudaFree(d_sv);
    d_sv = nullptr;
}

custatevecCollapseOp_t StateVector::get_collapse_op(bool collapse)
{
    custatevecCollapseOp_t op = CUSTATEVEC_COLLAPSE_NONE;
    if (collapse)
        op = CUSTATEVEC_COLLAPSE_NORMALIZE_AND_ZERO;
    return op;
}

/*
 *
 * Measure on the Z-basis
 *
 */
int32_t StateVector::measure(CuStatevecWorkspace &workspace, const BasisBits &basis_bits,
                             double randnum, bool collapse)
{
    int32_t parity;
    custatevecMeasureOnZBasis(
                workspace.handle, d_sv, CUDA_C_64F, num_bits, &parity,
                basis_bits.data(), basis_bits.size(),
                randnum, get_collapse_op(collapse));
    return parity;
}

void StateVector::measure(CuStatevecWorkspace &workspace, int32_t &parity,
                          const BasisBits &basis_bits, double randnum,
                          bool collapse)
{
    custatevecMeasureOnZBasis(
                workspace.handle, d_sv, CUDA_C_64F, num_bits, &parity,
                basis_bits.data(), basis_bits.size(),
                randnum, get_collapse_op(collapse));

}

/*
 *
 * Batch measure on the Z-basis
 *
 */
std::vector<int32_t> StateVector::batch_measure(CuStatevecWorkspace &workspace, const BasisBits &bit_ordering,
                                     double randnum, bool collapse)
{
    BasisBits bit_string(bit_ordering.size(), -1);

    custatevecBatchMeasure(
        workspace.handle, d_sv, CUDA_C_64F, num_bits,
        bit_string.data(), bit_ordering.data(), bit_string.size(),
        randnum, get_collapse_op(collapse));

    return bit_string;

}

void StateVector::batch_measure(CuStatevecWorkspace &workspace, BasisBits &bit_string,
                                const BasisBits &bit_ordering, double randnum,
                                bool collapse)
{
    if (bit_string.size() < bit_ordering.size())
        bit_string.resize(bit_ordering.size());

    custatevecBatchMeasure(
        workspace.handle, d_sv, CUDA_C_64F, num_bits,
        bit_string.data(), bit_ordering.data(), bit_string.size(),
        randnum, get_collapse_op(collapse));

}

/*
 *
 * Batch measure all qubits on the Z-basis
 *
 */
void StateVector::batch_measure_all(CuStatevecWorkspace &workspace, BasisBits &bit_string,
                                     double randnum, bool collapse)
{
    BasisBits bit_ordering = arange<int32_t>(num_bits-1, -1, -1);
    batch_measure(workspace, bit_string, bit_ordering, randnum, collapse);
}

/*
 *
 * Get probabilities for states
 *
 */
std::vector<double> StateVector::get_probabilities(CuStatevecWorkspace &workspace)
{
    // Calculate the probabilities for each state
    BasisBits bit_string_order = arange<int32_t>(num_bits-1, -1, -1);
    std::vector<double> probs(size(), 0);
    custatevecAbs2SumArray(
        workspace.handle, d_sv, CUDA_C_64F, num_bits,
        probs.data(), bit_string_order.data(), num_bits,
        NULL, NULL, 0);

    return probs;
}

/*
 *
 * Reset the state vector to the zero state on the device
 *
 */
void StateVector::reset(CuStatevecWorkspace &workspace)
{
    // Calculate the probabilities for each state
    BasisBits bit_string_order = arange<int32_t>(num_bits-1, -1, -1);
    std::vector<double> abs2sum(size(), 0);
    custatevecAbs2SumArray(
        workspace.handle, d_sv, CUDA_C_64F, num_bits,
        abs2sum.data(), bit_string_order.data(), num_bits,
        NULL, NULL, 0);

    // Normalization factor here is the probability of the |00...0>
    // state, which is 0th index
    double norm = abs2sum[0];

    // Collapse to |00...0>
    BasisBits bit_string(num_bits, 0);
    custatevecCollapseByBitString(
        workspace.handle, d_sv, CUDA_C_64F, num_bits,
        bit_string.data(), bit_string_order.data(), num_bits, norm);
}

/*
 *
 * Print the state vector to stdout in a formatted manner
 *
 */
void StateVector::print()
{
    for (size_t i = 0; i < size(); i++) {
        std::cout << "|" << num_to_binary(i, num_bits) << ">  ";
        std::cout << sv.coeff(i) << std::endl;;
    }

}

/*
 *
 * StateVector helper functions
 *
 */
StateVector create_random_state_vector(size_t num_bits)
{
    srand((unsigned int) time(0));
    size_t num_states = std::pow(2, num_bits);
    Eigen::VectorXcd v = Eigen::VectorXcd::Random(num_states).normalized();
    StateVector sv(v);
    return sv;
}

StateVector create_zero_state_vector(size_t num_bits)
{
    size_t num_states = std::pow(2, num_bits);
    Eigen::VectorXcd v = Eigen::VectorXcd::Zero(num_states);
    v[0] = 1;
    StateVector sv(v);
    return sv;
}
