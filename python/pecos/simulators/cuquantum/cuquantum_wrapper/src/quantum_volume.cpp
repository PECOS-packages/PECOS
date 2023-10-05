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

#include "quantum_volume.hpp"

#include "utils.hpp"

/*
 *
 * Helper functions
 *
 */
/*
 *
 * SU(4) gate
 * KAK-like decomposition ???
 *
 */
SU4Gate::SU4Gate() : szz(SqrtZZ())
{
    randomize_us();
}

SU4Gate::SU4Gate(double angles[8][3]) : szz(SqrtZZ())
{
    create_us(angles);
}

/*
 *
 * Create randomized U gates for the SU(4)
 *
 */
void SU4Gate::randomize_us()
{
    for (auto &u : us) {
        u.free_on_device();
        u = create_random_U();
    }
}

void SU4Gate::create_us(double angles[8][3])
{
    for (int n = 0; n < 8; n++) {
        us[n] = U(angles[n][0], angles[n][1], angles[n][2]);
    }
}

/*
 *
 * Copy and free the gate on the device
 *
 */
void SU4Gate::copy_to_device()
{
    free_on_device();

    szz.copy_to_device();
    for (auto &u : us)
        u.copy_to_device();
}

void SU4Gate::free_on_device()
{
    szz.free_on_device();
    for (auto &u : us)
        u.free_on_device();
}

/*
 *
 * Print the gate matrices to stdout
 *
 */
void SU4Gate::print()
{
    szz.print();
    for (auto &u : us)
        u.print();
}

/*
 *
 * Apply the gate to the target bits (ts) in the given state vector within
 * the given workspace
 *
 */
void SU4Gate::apply(StateVector &sv, CuStatevecWorkspace &ws, const Targets &ts)
{
    us[0].apply(sv, ws, {}, {ts[0]});
    us[1].apply(sv, ws, {}, {ts[1]});
    szz.apply(sv, ws, {}, ts);
    us[2].apply(sv, ws, {}, {ts[0]});
    us[3].apply(sv, ws, {}, {ts[1]});
    szz.apply(sv, ws, {}, ts);
    us[4].apply(sv, ws, {}, {ts[0]});
    us[5].apply(sv, ws, {}, {ts[1]});
    szz.apply(sv, ws, {}, ts);
    us[6].apply(sv, ws, {}, {ts[0]});
    us[7].apply(sv, ws, {}, {ts[1]});
}


/*
 *
 * Quantum Volume Circuit
 *
 */
QuantumVolume::QuantumVolume(size_t num_qubits) : num_qubits(num_qubits)
{
    targets = arange<int32_t>(0, num_qubits);
    create_circuit();
}

QuantumVolume::QuantumVolume(size_t num_qubits, double angles[][8][3], Targets ts) : num_qubits(num_qubits), targets(ts)
{
    create_circuit(angles);
}

/*
 *
 * Private method to create the actual circuit. Circuit is comprised of layers
 * of SU(4) gates. The number of layers equals the number of qubits.
 *
 */
void QuantumVolume::create_circuit()
{
    for (size_t i = 0; i < num_qubits; i++) {
        std::vector<SU4Gate> layer;
        for (size_t j = 0; j < num_qubits/2; j++)
            layer.push_back(SU4Gate());
        circuit.push_back(layer);
    }
}

void QuantumVolume::create_circuit(double angles[][8][3])
{
    for (size_t i = 0; i < num_qubits; i++) {
        std::vector<SU4Gate> layer;
        for (size_t j = 0; j < num_qubits/2; j++) {
            layer.push_back(SU4Gate(angles[i]));
        }
        circuit.push_back(layer);
    }
}

/*
 *
 * Copy and free the circuit on the device
 *
 */
void QuantumVolume::copy_to_device()
{
    for (auto &layer : circuit)
        for (auto &gate : layer)
            gate.copy_to_device();
}

void QuantumVolume::free_on_device()
{
    for (auto &layer : circuit)
        for (auto &gate : layer)
            gate.free_on_device();
}

/*
 *
 * Print the gate matrices to stdout
 *
 */
void QuantumVolume::print()
{
    for (auto &layer : circuit)
        for (auto &gate : layer)
            gate.print();
}

/*
 *
 * Apply the circuit to the target state vector within the target workspace
 *
 */
void QuantumVolume::apply(StateVector &sv, CuStatevecWorkspace &ws)
{
    for (auto &layer : circuit) {
        std::random_shuffle(targets.begin(), targets.end());
        for (int32_t i = 0; i < layer.size(); i++)
            layer[i].apply(sv, ws, {2*i, 2*i+1});
    }
}
