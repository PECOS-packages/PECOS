// Copyright 2018 The PECOS Developers
// Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract
// DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
// the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <iostream>
#include <cstdlib>
#include <vector>
#include <unordered_set>

using namespace std;

typedef unsigned long long int_num;


// The trivial hash (identity) for int_set.
// This is a perfect hash function for integers.
struct TrivialHash{
    public: size_t operator()(int_num & val) const noexcept { 
        return static_cast<size_t>(val); 
    }
};

//typedef unordered_set<int_num, TrivialHash, std::equal_to<unsigned long>> int_set;
typedef unordered_set<int_num> int_set;

// typedef unordered_set<int_num> int_set;
typedef vector<int_set> int_set_vec;


int_set_vec build_empty(int_num size, int reserve_buckets); // Builds empty int_set_vecs.
int_set_vec build_ones(int_num size, int reserve_buckets); // Builds an int_set_vecs with the diagonal filled.

struct Generators {
        int_set_vec col_x;  // qubit id -> set of gen ids
        int_set_vec col_z;  // qubit id -> set of gen ids
        int_set_vec row_x;  // gen id -> set of qubit ids
        int_set_vec row_z;  // gen id -> set of qubit ids
};


class State {
    // ~State () {};
    public:
        State(const int_num& num_qubits, const int& reserve_buckets=0);
        
        // Figure out constructors...
        // How to intialize the stabs and destabs...
        
        const int_num num_qubits;  // Total number of qubits.
        const int reserve_buckets; // Wether to reserve buckets.
        Generators stabs, destabs;  // Stabilizers and destabilizer generator matrices.
        int_set signs_minus, signs_i;  // A column that stores minuses and is.
        // Methods
        void clear();
        void hadamard(const int_num& qubit); // H
        void bitflip(const int_num& qubit); // X
        void phaseflip(const int_num& qubit); // Z
        void Y(const int_num& qubit);  // Y
        void phaserot(const int_num& qubit);  // S
        void SZd(const int_num& qubit);  // Sd
        void SY(const int_num& qubit);  // R
        void SYdg(const int_num& qubit);  // Rd
        void SX(const int_num& qubit);  // Q
        void SXdg(const int_num& qubit);  // Qd
        void H2(const int_num& qubit);  // H2
        void H3(const int_num& qubit);  // H3
        void H4(const int_num& qubit);  // H4
        void H5(const int_num& qubit);  // H5
        void H6(const int_num& qubit);  // H6
        void F(const int_num& qubit);  // F1
        void F2(const int_num& qubit);  // F2
        void F3(const int_num& qubit);  // F3
        void F4(const int_num& qubit);  // F4
        void Fdg(const int_num& qubit);  // F1d
        void F2dg(const int_num& qubit);  // F2d
        void F3dg(const int_num& qubit);  // F3d
        void F4dg(const int_num& qubit);  // F4d
        void cx(const int_num& tqubit, const int_num& cqubit);
        void swap(const int_num& qubit1, const int_num& qubit2);
        unsigned int measure(const int_num& qubit, int forced_outcome, bool collapse);
        
    private:
        unsigned int deterministic_measure(const int_num& qubit);
        unsigned int nondeterministic_measure(const int_num& qubit, int forced_outcome);
};

void hadamard_gen_mod(Generators& gen, const int_num& qubit);
void phaserot_gen_mod(Generators& gen, const int_num& qubit);
void Q_gen_mod(Generators& gen, const int_num& qubit);
void cnot_gen_mod(Generators& gen, const int_num& tqubit, const int_num& cqubit);
void swap_gen_mod(Generators& gen, const int_num& qubit1, const int_num& qubit2);
void F1_gen_mod(Generators& gen, const int_num& qubit);
void F2_gen_mod(Generators& gen, const int_num& qubit);
unsigned int random_outcome(void);