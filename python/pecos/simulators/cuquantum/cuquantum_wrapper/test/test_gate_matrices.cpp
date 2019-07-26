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

#include <complex>
#include <iostream>

#include "gate_matrices.hpp"

using namespace gate_matrices;

const Eigen::Matrix2cd I = Eigen::Matrix2d::Identity();
const std::complex<double> i = {0,1};
const auto X = PauliX();
const auto Y = PauliY();
const auto Z = PauliZ();

using namespace std;

TEST_CASE("Rotation matrices") 
{
    // No rotation should be identity
    REQUIRE(Rx(0) == I);
    REQUIRE(Ry(0) == I);
    REQUIRE(Rz(0) == I);

    REQUIRE(Rz(M_PI).isApprox(-i * Z));
}

TEST_CASE("Pauli matrices") 
{
    REQUIRE(X*X == I);
    REQUIRE(Y*Y == I);
    REQUIRE(Z*Z == I);

    REQUIRE(X*Y == -Y*X);
    REQUIRE(X*Z == -Z*X);
    REQUIRE(Y*Z == -Z*Y);
    REQUIRE(Y == i*X*Z);
    REQUIRE(X == i*Z*Y);
    REQUIRE(Z == i*Y*X);
}

TEST_CASE("U1q") 
{
    Matrix2cd m;
    Matrix2cd m_expected;

    m = U1q(0,0);
    REQUIRE(m == I);

    m = U1q(M_PI,0);
    REQUIRE(m.isApprox(-i * X));

    m = U1q(M_PI,M_PI/2);
    REQUIRE(m.isApprox(-i * Y));
}

TEST_CASE("SqrtZZ") {
    auto m = SqrtZZ();
    /* cout << m << endl; */
    REQUIRE(true);
}
