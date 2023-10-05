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

#include "custatevec_workspace.hpp"
#include "gate.hpp"
#include "state_vector.hpp"
#include "utils.hpp"
#include "quantum_volume.hpp"
#include "version.h"

#include "Eigen/KroneckerProduct"

using Eigen::KroneckerProduct;

// macro to print simple things to stdout
#define PRINT(x) std::cout << x << std::endl;
#define ELAPSED(timer) std::cout << "Elapsed: " << timer.elapsed() << " s" << std::endl;

void bell_state(double randnum=0)
{
    // Initialize the workspace
    CuStatevecWorkspace workspace;

    StateVector sv = create_zero_state_vector(2);

    PRINT("Initial State");
    sv.print();

    sv.copy_to_device();

    // Create and apply the gates
    Gate h = Hadamard();
    Gate cnot = CNOT();

    h.copy_to_device();
    cnot.copy_to_device();

    h.apply(sv, workspace, {}, {0});
    cnot.apply(sv, workspace, {0}, {1});

    sv.read_from_device();

    PRINT("Bell state");
    sv.print();

    BasisBits basis_bits = {0,1};
    int parity0 = -1;
    int parity1 = -1;
    sv.measure(workspace, parity0, {0}, randnum);
    sv.measure(workspace, parity1, {1}, randnum);

    sv.read_from_device();

    PRINT("Measurement");
    std::cout << "Random Number: " << randnum << std::endl;
    std::cout << "        Bit 0: " << parity0 << std::endl;
    std::cout << "        Bit 1: " << parity1 << std::endl;
    PRINT("Final State Vector");
    sv.print();
}


int main(int argc, char *argv[])
{
    // bell_state();

    double randnum = 0;
    if (argc > 1)
        randnum = std::atof(argv[1]);

    if (randnum < 0)
        randnum = 0;

    if (randnum >= 1)
        randnum = 0.9999;

    bell_state(randnum);

    return 0;
}
