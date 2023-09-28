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

#include <limits>
#include <numeric>
#include <algorithm>

#include "custatevec_workspace.hpp"
#include "state_vector.hpp"         
#include "utils.hpp"         

#include "cuda_helper.hpp"         // HANDLE_ERROR, HANDLE_CUDA_ERROR

using namespace std;

using CD = std::complex<double>;

/*
 *
 * StateVector tests
 *
 */

TEST_CASE("StateVector") 
{
    // Create the state vectors
    Eigen::VectorXcd v(8);
    v << CD(1.3, 3.7), CD(7.1, 2.1), CD(4.3, 0.5), CD(6.1, 9.2), 
         CD(9.0, 8.8), CD(3.3, 3.3), CD(6.6, 6.6), CD(2.3, 8.5);
    StateVector sv(v);
    StateVector sv_expected = sv;
    
    REQUIRE(sv.num_bits == 3);
    REQUIRE(sv.size() == 8);

    sv.copy_to_device();
    sv.read_from_device();
    sv.free_on_device();

    REQUIRE(sv == sv_expected);
}


// **************************************
// THIS TEST IS BROKEN DUE TO COMPOSITION CHANGE
// **************************************
// TEST_CASE("StateVector memory") 
// {
//     // C-array
//     const int nIndexBits   = 3;
//     const int nSvSize      = (1 << nIndexBits);
//     cuDoubleComplex sv_orig[] = {{ 0.0, 0.0}, { 0.0, 0.1}, { 0.3, 0.4}, { 0.1, 0.2}, 
//                                    { 0.2, 0.2}, { 0.3, 0.3}, { 0.1, 0.1}, { 0.4, 0.5}};

//     // Create the state vector
//     Eigen::VectorXcd v(8);
//     v << CD(0.0, 0.0), CD(0.0, 0.1), CD(0.3, 0.4), CD(0.1, 0.2), 
//          CD(0.2, 0.2), CD(0.3, 0.3), CD(0.1, 0.1), CD(0.4, 0.5);
//     StateVector sv(v);

//     // Sanity check that the C-style and StateVector representations are the same
//     bool same = true;
//     for (int i = 0; i < nSvSize; i++) {
//         cuDoubleComplex *v = reinterpret_cast<cuDoubleComplex*>(&sv(i));
//         if (!almost_equal(*v, sv_orig[i])) 
//             same = false;
//     }

//     // Print to stdout if not the same
//     if ( !same ) {
//         for (int i = 0; i < nSvSize; i++) {
//             cuDoubleComplex *v = reinterpret_cast<cuDoubleComplex*>(&sv(i));
//             std::cout << sv(i) << "   " ;
//             std::cout << std::setw(3) << sv_orig[i].x << "," << sv_orig[i].y <<  "\n";
//         }
//     }

//     REQUIRE(same);
// }


TEST_CASE("Measurement") 
{
    const BasisBits basis_bits = {0, 1, 2};

    int parity = -1;

    // In real appliction, random number in range [0, 1) will be used.
    const double randnum = 0.2;

    // Create the state vector
    Eigen::VectorXcd v(8);
    v << CD(0.0, 0.0), CD(0.0, 0.1), CD(0.3, 0.4), CD(0.1, 0.2), 
         CD(0.2, 0.2), CD(0.3, 0.3), CD(0.1, 0.1), CD(0.4, 0.5);
    StateVector sv(v);

    sv.copy_to_device();

    // Initialize the workspace
    CuStatevecWorkspace workspace;

    // Measure
    sv.measure(workspace, parity, basis_bits, randnum);

    sv.read_from_device();

    // Check results
    Eigen::VectorXcd v_expected(8);
    v_expected << CD(0.0, 0.0), CD(0.0, 0.0), CD(0.0, 0.0), CD(0.2, 0.4), 
                  CD(0.0, 0.0), CD(0.6, 0.6), CD(0.2, 0.2), CD(0.0, 0.0);
    StateVector sv_expected(v_expected);
    int parity_expected = 0;

    REQUIRE(sv ==  sv_expected);
    REQUIRE(parity ==  parity_expected);
}


void run_measure_test(double randnum)
{
    // Create the state vector
    // 60% chance |00> and 40% change |01>
    Eigen::VectorXcd v(4);
    v << std::sqrt(0.6), 0, std::sqrt(0.4), 0;
    StateVector sv(v);

    // Expected results
    Eigen::VectorXcd v_00(4);
    v_00 << 1, 0, 0, 0;
    StateVector sv_00(v_00);

    Eigen::VectorXcd v_01(4);
    v_01 << 0, 0, 1, 0;
    StateVector sv_01(v_01);

    const BasisBits basis_bits = {0, 1};
    int parity = -1;

    sv.copy_to_device();

    // Initialize the workspace
    CuStatevecWorkspace workspace;

    // Perform measurement
    sv.measure(workspace, parity, basis_bits, randnum);

    sv.read_from_device();

    if (randnum <= 0.6)
        REQUIRE(sv == sv_00);
    else
        REQUIRE(sv == sv_01);

}

TEST_CASE("Measurement on Z - Randnums") 
{
    double step = 0.05;
    for (double randnum = 0; randnum <= 1; randnum = randnum + step) 
        run_measure_test(randnum);
}

TEST_CASE("Batch Measurement") 
{
    Eigen::VectorXcd v(8);
    v << CD(0.0, 0.0), CD(0.0, 0.1), CD(0.1, 0.1), CD(0.1, 0.2),
         CD(0.2, 0.2), CD(0.3, 0.3), CD(0.3, 0.4), CD(0.4, 0.5);
    StateVector sv(v);

    Eigen::VectorXcd v_expected(8);
    v_expected << CD(0.0, 0.0), CD(0.0, 0.0), CD(0.0, 0.0), CD(0.0, 0.0),
                  CD(0.0, 0.0), CD(0.0, 0.0), CD(0.6, 0.8), CD(0.0, 0.0);
    StateVector sv_expected(v_expected);

    const double randnum = 0.5;
    const bool collapse = true;

    const BasisBits bit_ordering{2, 1, 0};
    BasisBits bit_string;
    const BasisBits bit_string_expected{1, 1, 0};

    CuStatevecWorkspace workspace;
    sv.copy_to_device();

    sv.batch_measure(workspace, bit_string, bit_ordering, randnum, collapse);

    sv.read_from_device();

    REQUIRE(sv == sv_expected);
    REQUIRE(bit_string == bit_string_expected);

}

TEST_CASE("Reset state")
{
    constexpr size_t NUM_BITS = 5;
    constexpr size_t TEST_CASES = 5;

    const size_t num_states = 1 << NUM_BITS;

    CuStatevecWorkspace ws;
    StateVector sv(NUM_BITS);
    BasisBits bs(NUM_BITS, -1);

    bool check, all_zeros;

    // No reset
    for (int i  = 0; i < TEST_CASES; i++) {
        sv = create_random_state_vector(NUM_BITS);
        sv.copy_to_device();

        check = true;

        // Make multiple measurements to ensure not in the zero state
        for (float r = 0; r < 0.99; r+=0.05) {
            // Reset bit sting
            std::fill(bs.begin(), bs.end(), -1);
            // Measure without collapsing
            sv.batch_measure_all(ws, bs, r, false);
            // Is |00...0> state?
            all_zeros = std::all_of(bs.begin(), bs.end(), [](int32_t b) { return b==0; });
            check &= all_zeros;
        }

        // Should be false, because the state is not in the zerio state
        // NOTE: There is a small probability that this will fail b/c 
        // create random vector might create the zero state
        REQUIRE_FALSE(check);

        sv.read_from_device();
        sv.free_on_device();
    }

    // Reset
    for (int i  = 0; i < TEST_CASES; i++) {
        sv = create_random_state_vector(NUM_BITS);
        sv.copy_to_device();

        sv.reset(ws);

        check = true;

        // Measure to ensure in |00...0> state
        for (float r = 0; r < 0.99; r+=0.05) {
            // Reset bit sting
            std::fill(bs.begin(), bs.end(), -1);
            // Measure ... no collapse
            sv.batch_measure_all(ws, bs, r, false);
            // Is |00...0> state?
            all_zeros = std::all_of(bs.begin(), bs.end(), [](int32_t b) { return b==0; });
            check &= all_zeros;
        }

        // Should be true, because the state was reset to the zero state 
        // before measurements. Thus all measurements should yield the 
        // zero state
        REQUIRE(check);

        sv.read_from_device();
        sv.free_on_device();
    }
}