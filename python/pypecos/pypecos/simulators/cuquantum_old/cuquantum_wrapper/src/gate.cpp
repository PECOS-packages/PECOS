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

#include "gate.hpp"
#include "gate_matrices.hpp"

#include "Eigen/Dense"

using Eigen::Matrix2cd;
using Eigen::Matrix4cd;

// #include "gate_matrices.hpp"

/*
 *
 * Constructors
 *
 */
Gate::Gate() :
    mat(Eigen::MatrixXcd())
    {}

Gate::Gate(size_t rows, size_t cols) :
    mat(Eigen::MatrixXcd(rows,cols))
    {}

Gate::Gate(const Eigen::MatrixXcd mat) :
    mat(mat)
    {}

/*
 *
 * Device functions
 *
 */
void Gate::set(const Eigen::MatrixXcd mat_new)
{
    mat = mat_new;
}

size_t Gate::size()
{
    return mat.size();
}

size_t Gate::mem_size()
{
    return size() * sizeof(cuDoubleComplex);
}

void Gate::copy_to_device()
{
    // Allocate space on the device
    cudaError_t status = cudaMalloc((void **)&d_mat, mem_size());
    if(status != cudaSuccess)
        throw std::runtime_error("cudaMalloc error");

    // Copy gate matrix to the device
    cudaMemcpy(d_mat, mat.data(), mem_size(), cudaMemcpyHostToDevice);
}

void Gate::free_on_device()
{
    // Free memory on device
    cudaFree(d_mat);
    d_mat = nullptr;
}

/*
 *
 * Apply the gate to a state vector on the host
 *
 */
void Gate::apply(StateVector &sv)
{
    sv.sv = mat * sv.sv;
}

/*
 *
 * Apply the gate to a state vector in the provided workspace using pre-configured
 * control and target bits. Optionally, apply the adjoint of the gate
 *
 */
void Gate::apply(StateVector &sv, CuStatevecWorkspace &workspace, bool adjoint)
{
    apply(sv, workspace, controls, targets, adjoint);
}


/*
 *
 * Apply the gate to a state vector in the provided workspace using the provided
 * control and target bits. Optionally, apply the adjoint of the gate
 *
 */
void Gate::apply(StateVector &sv, CuStatevecWorkspace &workspace,
                 const Controls &controls, const Targets &targets,
                 bool adjoint)
{
    // If the gate has been copied to the device, use the on device copy
    void *ptr = mat.data();
    if (d_mat != nullptr)
        ptr = d_mat;

    // First, check to see if workspace can handle the gate
    custatevecApplyMatrixGetWorkspaceSize(
                    workspace.handle, CUDA_C_64F, sv.num_bits, ptr,
                    CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_COL,
                    adjoint, targets.size(), controls.size(), CUSTATEVEC_COMPUTE_64F,
                    &workspace.extra_sz);

    // Allocate external workspace if necessary
    if (workspace.extra_sz > 0)
        cudaMalloc(&workspace.extra, workspace.extra_sz);

    int32_t csize = controls.size();

    // Apply the gate
    custatevecApplyMatrix(workspace.handle, sv.d_sv, CUDA_C_64F,
			  sv.num_bits, ptr, CUDA_C_64F,
			  CUSTATEVEC_MATRIX_LAYOUT_COL, adjoint,
			  targets.data(), targets.size(),
			  controls.data(), nullptr, controls.size(),
			  CUSTATEVEC_COMPUTE_64F,
			  &workspace.extra, workspace.extra_sz);
}


void Gate::print()
{
    std::cout << mat << std::endl;
}

/*
 *
 * Factory functions for specific gates
 *
 */

static constexpr std::complex<double> i = {0,1};

Gate Rx(double theta)
{
    return Gate(gate_matrices::Rx(theta));
}


Gate Ry(double theta)
{
    return Gate(gate_matrices::Ry(theta));
}


Gate Rz(double theta)
{
    return Gate(gate_matrices::Rz(theta));
}


Gate U(double theta, double phi, double lambda)
{
    Matrix2cd m(2,2);
    m = gate_matrices::Rx(theta) * gate_matrices::Ry(phi) * gate_matrices::Rz(lambda);
    return Gate(m);
}

Gate create_random_U()
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<> dist(0, 2*M_PI);

    double theta = dist(rng);
    double phi = dist(rng);
    double lambda = dist(rng);

    return U(theta, phi, lambda);
}


Gate U1q(double theta, double phi)
{
    Matrix2cd m(2,2);
    m = gate_matrices::Rz(phi-M_PI/2) * gate_matrices::Ry(theta) * gate_matrices::Rz(-phi+M_PI/2);
    return Gate(m);
}


Gate Hadamard()
{
    return Gate(gate_matrices::Hadamard());
}


Gate CNOT()
{
    return Gate(gate_matrices::CNOT());
}


Gate Toffoli()
{
    return CNOT();
}


Gate PauliX()
{
    return Gate(gate_matrices::PauliX());
}


Gate PauliY()
{
    return Gate(gate_matrices::PauliY());
}


Gate PauliZ()
{
    return Gate(gate_matrices::PauliZ());
}


/*
 *
 * Two qubit gates
 *
 */
Gate SqrtZZ()
{
    return Gate(gate_matrices::SqrtZZ());
}

Gate RZZ(double theta)
{
    return Gate(gate_matrices::RZZ(theta));
}
