# distutils: language = c++
#cython: language_level=3, boundscheck=False, nonecheck=False, wraparound=False

# Copyright 2018 The PECOS Developers
# Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract
# DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

cimport cysparsesim_header as s
from cysparsesim_header cimport int_num

from .src.logical_sign import find_logical_signs

cdef dict gate_dict = {
                'I': State.I,
                'X': State.X,
                'Y': State.Y,
                'Z': State.Z,

                'II': State.II,
                'CNOT': State.cnot,
                'CZ': State.cz,
                'SWAP': State.swap,
                'G': State.g2,

                'H': State.hadamard,
                'H1': State.hadamard,
                'H2': State.H2,
                'H3': State.H3,
                'H4': State.H4,
                'H5': State.H5,
                'H6': State.H6,

                'H+z+x': State.hadamard,
                'H-z-x': State.H2,
                'H+y-z': State.H3,
                'H-y-z': State.H4,
                'H-x+y': State.H5,
                'H-x-y': State.H6,

                'F1': State.F1,
                'F2': State.F2,
                'F3': State.F3,
                'F4': State.F4,

                'F1d': State.F1d,
                'F2d': State.F2d,
                'F3d': State.F3d,
                'F4d': State.F4d,

                'R': State.R,
                'Rd': State.Rd,
                'Q': State.Q,
                'Qd': State.Qd,
                'S': State.S,
                'Sd': State.Sd,

                'measure X': State.measureX,
                'measure Y': State.measureY,
                'measure Z': State.measure,
                'force output': State.force_output,

                'init |0>': State.initzero,
                'init |1>': State.initone,
                'init |+>': State.initplus,
                'init |->': State.initminus,
                'init |+i>': State.initplusi,
                'init |-i>': State.initminusi,

            }

cdef class State:

    cdef s.State* _c_state

    cdef public:
        int_num num_qubits
        dict gate_dict

    def __cinit__(self, int_num num_qubits):
        self._c_state = new s.State(num_qubits)

        self.num_qubits = self._c_state.num_qubits
        self.gate_dict = gate_dict

    cdef void hadamard(self, int_num qubit):
        self._c_state.hadamard(qubit)

    cdef void H2(self, int_num qubit):
        self._c_state.H2(qubit)

    cdef void H3(self, int_num qubit):
        self._c_state.H3(qubit)

    cdef void H4(self, int_num qubit):
        self._c_state.H4(qubit)

    cdef void H5(self, int_num qubit):
        self._c_state.H5(qubit)

    cdef void H6(self, int_num qubit):
        self._c_state.H6(qubit)

    cdef void F1(self, int_num qubit):
        self._c_state.F1(qubit)

    cdef void F2(self, int_num qubit):
        self._c_state.F2(qubit)

    cdef void F3(self, int_num qubit):
        self._c_state.F3(qubit)

    cdef void F4(self, int_num qubit):
        self._c_state.F4(qubit)

    cdef void F1d(self, int_num qubit):
        self._c_state.F1d(qubit)

    cdef void F2d(self, int_num qubit):
        self._c_state.F2d(qubit)

    cdef void F3d(self, int_num qubit):
        self._c_state.F3d(qubit)

    cdef void F4d(self, int_num qubit):
        self._c_state.F4d(qubit)

    cdef void I(self, int_num qubit):
        pass

    cdef void X(self, int_num qubit):
        self._c_state.bitflip(qubit)

    cdef void Y(self, int_num qubit):
        self._c_state.Y(qubit)

    cdef void Z(self, int_num qubit):
        self._c_state.phaseflip(qubit)

    cdef void S(self, int_num qubit):
        self._c_state.phaserot(qubit)

    cdef void Sd(self, int_num qubit):
        self._c_state.Sd(qubit)

    cdef void R(self, int_num qubit):
        self._c_state.R(qubit)

    cdef void Rd(self, int_num qubit):
        self._c_state.Rd(qubit)

    cdef void Q(self, int_num qubit):
        self._c_state.Q(qubit)

    cdef void Qd(self, int_num qubit):
        self._c_state.Qd(qubit)

    cdef void cnot(self, tuple qubits):
        cdef int_num cqubit = qubits[0]
        cdef int_num tqubit = qubits[1]

        self._c_state.cnot(cqubit, tqubit)

    cdef void cz(self, tuple qubits):
        cdef int_num cqubit = qubits[0]
        cdef int_num tqubit = qubits[1]

        self._c_state.hadamard(tqubit)
        self._c_state.cnot(cqubit, tqubit)
        self._c_state.hadamard(tqubit)

    cdef void g2(self, tuple qubits):
        cdef int_num cqubit = qubits[0]
        cdef int_num tqubit = qubits[1]

        self._c_state.hadamard(cqubit)
        self._c_state.cnot(tqubit, cqubit)
        self._c_state.cnot(cqubit, tqubit)
        self._c_state.hadamard(tqubit)

    cdef void swap(self, tuple qubits):
        cdef int_num qubit1 = qubits[0]
        cdef int_num qubit2 = qubits[1]

        self._c_state.swap(qubit1, qubit2)

    cdef void II(self, tuple qubits):
        pass

    # cpdef unsigned int measure(self, const s.int_num qubit,
    def measure(self, const int_num qubit, int random_outcome=0):
        return self._c_state.measure(qubit, random_outcome)

    def measureX(self, const int_num qubit, int random_outcome=0):

        cdef unsigned int result

        self._c_state.hadamard(qubit)
        result = self._c_state.measure(qubit, random_outcome)
        self._c_state.hadamard(qubit)
        return result

    def measureY(self, const int_num qubit, int random_outcome=0):

        cdef unsigned int result

        self._c_state.H5(qubit)
        result = self._c_state.measure(qubit, random_outcome)
        self._c_state.H5(qubit)
        return result

    cdef void initzero(self, const int_num qubit):
        cdef unsigned int result
        result = self._c_state.measure(qubit, force=0)

        if result:
            self._c_state.bitflip(qubit)

    cdef void initone(self, const int_num qubit):
        cdef unsigned int result
        result = self._c_state.measure(qubit, force=0)

        if not result:
            self._c_state.bitflip(qubit)

    cdef void initplus(self, const int_num qubit):
        self.initzero(qubit)
        self._c_state.hadamard(qubit)

    cdef void initminus(self, const int_num qubit):
        self.initone(qubit)
        self._c_state.hadamard(qubit)

    cdef void initplusi(self, const int_num qubit):
        self.initzero(qubit)
        self._c_state.H5(qubit)

    cdef void initminusi(self, const int_num qubit):
        self.initone(qubit)
        self._c_state.H5(qubit)

    def logical_sign(self, logical_op, delogical_op):
        """

        Args:
            logical_op:
            delogical_op:

        Returns:

        """
        return find_logical_signs(self, logical_op, delogical_op)

    def run_gate(self, symbol, locations, output=True, **gate_kwargs):
        """

        Args:
            symbol:
            locations:
            output:
            **gate_kwargs:

        Returns:

        """

        if output:
            output = {}
            for location in locations:
                results = self.gate_dict[symbol](self, location, **gate_kwargs)
                if results:
                    output[location] = results

            return output

        else:
            for location in locations:
                self.gate_dict[symbol](self, location, **gate_kwargs)

            return {}

    @property
    def signs_minus(self):
        return self._c_state.signs_minus

    @property
    def signs_i(self):
        return self._c_state.signs_i

    @property
    def stabs(self):
        return self._c_state.stabs

    @property
    def destabs(self):
        return self._c_state.destabs

    def _pauli_sign(self, gen, i_gen):

        if i_gen in self.signs_minus:
            if i_gen in self.signs_i:
                sign = '-i'
            else:
                sign = ' -'
        else:

            if i_gen in self.signs_i:
                sign = ' i'
            else:
                sign = '  '

        return sign

    def force_output(self, int_num qubit, output=-1):
        """
        Outputs value.

        Used for error generators to generate outputs when replacing measurements.

        Args:
            state:
            qubit:
            output:

        Returns:

        """
        return output

    def col_string(self, gen):
        """
        Prints out the stabilizers for the column-wise sparse representation.

        :param num_qubits:
        :return:
        """

        col_x = gen['col_x']
        col_z = gen['col_z']

        result = []

        num_qubits = self.num_qubits

        for i_gen in range(num_qubits):

            stab_letters = []

            # ---- Signs ---- #
            sign = self._pauli_sign(gen, i_gen)
            stab_letters.append(sign)

            # ---- Paulis ---- #
            for qubit in range(num_qubits):

                letter = 'U'

                if i_gen in col_x[qubit] and i_gen not in col_z[qubit]:
                    letter = 'X'

                elif i_gen not in col_x[qubit] and i_gen in col_z[qubit]:
                    letter = 'Z'

                elif i_gen in col_x[qubit] and i_gen in col_z[qubit]:
                    letter = 'W'

                elif i_gen not in col_x[qubit] and i_gen not in col_z[qubit]:
                    letter = 'I'

                stab_letters.append(letter)

            # print(''.join(stab_letters))
            result.append(''.join(stab_letters))

        return result

    def print_tableau(self, gen, verbose=True):
        """
        Prints out the stabilizers.
        :return:
        """

        col_str = self.col_string(gen)
        row_str = self.row_string(gen)

        if col_str != row_str:
            print('col')
            for line in col_str:
                print(line)

            print('\nrow')
            for line in row_str:
                print(line)

            raise Exception('Something bad happened! String representation of the row-wise vs column-wile '
                            'spare stabilizers do not match!')

        if verbose:
            for line in col_str:
                print(line)

        return col_str

    def row_string(self, gen):
        """
        Prints out the stabilizers for the row-wise sparse representation.

        :param num_qubits:
        :return:
        """

        row_x = gen['row_x']
        row_z = gen['row_z']

        result = []

        num_qubits = self.num_qubits

        for i_gen in range(num_qubits):

            stab_letters = []

            # ---- Signs ---- #
            sign = self._pauli_sign(gen, i_gen)
            stab_letters.append(sign)

            # ---- Paulis ---- #
            for qubit in range(num_qubits):

                letter = 'U'

                if qubit in row_x[i_gen] and qubit not in row_z[i_gen]:
                    letter = 'X'

                elif qubit not in row_x[i_gen] and qubit in row_z[i_gen]:
                    letter = 'Z'

                elif qubit in row_x[i_gen] and qubit in row_z[i_gen]:
                    letter = 'W'

                elif qubit not in row_x[i_gen] and qubit not in row_z[i_gen]:
                    letter = 'I'

                stab_letters.append(letter)

            # print(''.join(stab_letters))
            result.append(''.join(stab_letters))

        return result

    def __dealloc__(self):
        del self._c_state
