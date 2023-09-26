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

#ifndef QUANTUM_VOLUME_HPP
#define QUANTUM_VOLUME_HPP

#include "gate.hpp"

/*
 *
 * SU(4) gate
 * KAK-like decomposition ???
 *
 */
class SU4Gate 
{
    Gate szz;
    std::array<Gate,8> us;

public:
    SU4Gate();
    SU4Gate(double angles[8][3]);
    // Randomize the U gates within the SU(4)
    void randomize_us();
    // Create the 8 U gates within the given angles (theta, phi, lambda)
    void create_us(double angles[8][3]);
    void copy_to_device();
    void free_on_device();
    void print();
    void apply(StateVector &sv, CuStatevecWorkspace &ws, const Targets &ts);
};

/*
 *
 * Quantum Volume circuit
 *
 */
class QuantumVolume
{
    size_t num_qubits;
    std::vector<std::vector<SU4Gate>> circuit;
    Targets targets; 

    void create_circuit();
    void create_circuit(double angles[][8][3]);

public:
    QuantumVolume(size_t num_qubits);
    QuantumVolume(size_t num_qubits, double angles[][8][3], Targets ts);
    void copy_to_device();
    void free_on_device();
    void print();
    void apply(StateVector &sv, CuStatevecWorkspace &ws);
};

#endif // QUANTUM_VOLUME_HPP
