# distutils: language = c++
#cython: language_level=3, boundscheck=False, nonecheck=False, wraparound=False

#  =========================================================================  #
#   Copyright 2018 National Technology & Engineering Solutions of Sandia,
#   LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
#   the U.S. Government retains certain rights in this software.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#  =========================================================================  #

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set

cdef extern from "sparsesim.h":

    ctypedef unsigned long long int_num
    ctypedef unordered_set[int_num] int_set
    ctypedef vector[int_set] int_set_vec

    unsigned int random_outcome()

    struct Generators:
        int_set_vec col_x
        int_set_vec col_z
        int_set_vec row_x
        int_set_vec row_z

    cdef cppclass State:
        # State() except +
        State(int_num, bool) except +
        State(int_num) except +
        int_num num_qubits
        int reserve_buckets
        Generators stabs, destabs
        int_set signs_minus, signs_i

        # Methods
        void hadamard(const int_num &qubit)
        void bitflip(const int_num &qubit)  # X
        void phaseflip(const int_num& qubit)  # Z
        void Y(const int_num& qubit)  # Y
        void phaserot(const int_num& qubit)  # S
        void SZdg(const int_num& qubit)  # S
        void SY(const int_num& qubit)  # R
        void SYdg(const int_num& qubit)  # Rd
        void SX(const int_num& qubit)  # Q
        void SXdg(const int_num& qubit)  # Qd
        void H2(const int_num &qubit) # H2
        void H3(const int_num &qubit) # H3
        void H4(const int_num &qubit) # H4
        void H5(const int_num &qubit) # H5
        void H6(const int_num &qubit) # H6
        void F(const int_num &qubit) # F1
        void F2(const int_num &qubit) # F2
        void F3(const int_num &qubit) # F3
        void F4(const int_num &qubit) # F4
        void Fdg(const int_num &qubit) # F1d
        void F2dg(const int_num &qubit) # F2d
        void F3dg(const int_num &qubit) # F3d
        void F4dg(const int_num &qubit) # F4d
        void cx(const int_num& tqubit, const int_num& cqubit)
        void swap(const int_num& qubit1, const int_num& qubit2)
        unsigned int measure(const int_num& qubit, int forced_outcome, bool collapse)
