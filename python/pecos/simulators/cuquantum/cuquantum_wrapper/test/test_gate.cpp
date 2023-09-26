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

#include "catch.hpp"

#include "custatevec_workspace.hpp"
#include "gate.hpp"         
#include "state_vector.hpp"         

#include "cuda_helper.hpp"         // HANDLE_ERROR, HANDLE_CUDA_ERROR

using namespace std;

using CD = std::complex<double>;

/*
 *
 * Single gate tests
 *
 */

// Helper to run a single gate test
void run_gate_test(Gate &gate, StateVector &sv)
{
    // Copy state vector to device
    sv.copy_to_device();

    // Initialize the workspace
    CuStatevecWorkspace workspace;

    // Apply gate
    gate.copy_to_device();
    gate.apply(sv, workspace);
    gate.free_on_device();

    // Copy state vector back to host
    sv.read_from_device();
}


TEST_CASE("Hadamard") 
{
    // Create the gate
    Gate gate = Hadamard();
    gate.targets = {0};
    gate.controls = {};

    // Create the state vector 
    StateVector sv = create_zero_state_vector(1);
    StateVector sv_expected = create_zero_state_vector(1);

    // Directly calculate the expected result before running test
    gate.apply(sv_expected);

    // Modifies the state vector
    run_gate_test(gate, sv);

    REQUIRE(sv == sv_expected);
}


TEST_CASE("Toffoli From Docs") 
{
    // Create the gate
    Gate gate = Toffoli();
    gate.controls = {0,1};
    gate.targets = {2};

    // Create the state vector
    Eigen::VectorXcd v(8);
    v << (0.0, 0.0), (0.0, 0.1), (0.1, 0.1), (0.1, 0.2), 
         (0.2, 0.2), (0.3, 0.3), (0.3, 0.4), (0.4, 0.5);
    StateVector sv(v);
    sv.copy_to_device();

    run_gate_test(gate, sv);

    // Check results
    StateVector ex(3);
    Eigen::VectorXcd vex(8);
    vex << (0.0, 0.0), (0.0, 0.1), (0.1, 0.1), (0.4, 0.5), 
           (0.2, 0.2), (0.3, 0.3), (0.3, 0.4), (0.1, 0.2);
    StateVector svex(vex);

    REQUIRE(sv == svex);
}


// TODO FIXME Broken after change to composition
// TEST_CASE("Toffoli") 
// {
//     // Create the gate
//     Gate gate = Toffoli();
//     gate.controls = {0,1};
//     gate.targets = {2};

//     // Create the input and expected state vectors
//     StateVector sv(3);
//     StateVector ex(3);
    
//     // 000 -> 000
//     sv << 1, 0, 0, 0, 0, 0, 0, 0;
//     ex = sv;
//     run_gate_test(gate, sv);
//     REQUIRE(sv == ex);

//     // 010 -> 010
//     sv << 0, 1, 0, 0, 0, 0, 0, 0;
//     ex = sv;
//     run_gate_test(gate, sv);
//     REQUIRE(sv == ex);
    
//     // skip to the ones that flip bits

//     // 111 -> 110
//     sv << 0, 0, 0, 0, 0, 0, 0, 1;
//     ex << 0, 0, 0, 1, 0, 0, 0, 0;
//     run_gate_test(gate, sv);
//     REQUIRE(sv == ex);

//     // 110 -> 111
//     sv << 0, 0, 0, 1, 0, 0, 0, 0;
//     ex << 0, 0, 0, 0, 0, 0, 0, 1;
//     run_gate_test(gate, sv);
//     REQUIRE(sv == ex);
// }


// TODO FIXME Broken after change to composition
// TEST_CASE("CNOT") 
// {
//     Gate gate = CNOT();
//     gate.targets = {1};
//     gate.controls = {0};

//     // Create the input and expected state vectors
//     StateVector sv(2);
//     StateVector ex(2);

//     // 00 -> 00
//     sv << 1, 0, 0, 0;
//     ex = sv;
//     run_gate_test(gate, sv);
//     REQUIRE(sv == ex);

//     // 01 -> 01
//     sv << 0, 0, 1, 0;
//     ex = sv;
//     run_gate_test(gate, sv);
//     REQUIRE(sv == ex);

//     // 10 -> 11
//     sv << 0, 1, 0, 0;
//     ex << 0, 0, 0, 1;
//     run_gate_test(gate, sv);
//     REQUIRE(sv == ex);

//     // 11 -> 10
//     sv << 0, 0, 0, 1;
//     ex << 0, 1, 0, 0;
//     run_gate_test(gate, sv);
//     REQUIRE(sv == ex);

// }


/*
 *
 * Multi-gate (circuit) tests
 *
 */

TEST_CASE("Bell State")
{
    Gate h = Hadamard();
    h.targets = {0};
    h.controls = {};

    Gate cnot = CNOT(); 
    cnot.targets = {1};
    cnot.controls = {0};

    Eigen::VectorXcd v(4); 
    v << 1, 0, 0, 0;
    StateVector sv(v); 

    sv.copy_to_device();

    // Initialize the workspace
    CuStatevecWorkspace workspace;

    // Copy gates to device
    h.copy_to_device();
    cnot.copy_to_device();

    // Apply gates
    h.apply(sv, workspace);
    cnot.apply(sv, workspace);

    sv.read_from_device();

    // Bell state
    v << S2, 0, 0, S2;
    StateVector sv_bell(v); 

    REQUIRE(sv == sv_bell);
}
