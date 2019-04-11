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

"""
:file: state_data_structure.py

A data structure to track stabilizer states for codes made up of qubits.

This is based on CHP; however, sparsity is taken advantage of. LDPC codes tend to be are sparse both row-wise and
column-wise.

For the paper on CHP read: http://arxiv.org/abs/quant-ph/0406196

Author(s): Ciaran Ryan-Anderson <ciaranra@unm.edu; cryanan@sandia.gov>
Date Created:  02/24/2015
Last modified: 05/24/2017

Revision history

Date        Author  Comment
----------  ------  ---------------------------------------------------------------------------------------------------
02/24/2015  CRA     File created.

03/30/2016  CRA     Separated data structure stuff from main file.

05/24/2017  CRA     Simplified string outputs. Added the method ``gate`` to ``State`` to give an 

"""
from typing import Any, Union, Set, Tuple, List, Optional
from ...circuits import QuantumCircuit
from .. import BaseSim
from . import bindings
from .logical_sign import find_logical_signs
from .refactor import refactor as refactor_generators
from .refactor import find_stab as find_stabilizer


class SparseSim(BaseSim):
    """
    Represents the stabilizer state.

    Attributes:
        num_qubits (int):
        gate_dict (dict):
        stabs (Gens):
        destabs (Gens):
        gens (Tuple[Gens, Gens]):
    """

    def __init__(self, num_qubits: int) -> None:
        """
        Initializes the stabilizer state.

        Args:
            num_qubits (int): Number of qubits being represented.

        Returns:

        """

        super().__init__()

        if not isinstance(num_qubits, int):
            raise Exception('``num_qubits`` should be of type ``int.``')

        self.num_qubits = num_qubits

        self.bindings = bindings.gate_dict  # TODO: check to see if it makes a difference in performance if this is
        # moved to a class variable.

        # Represent a stabilizer state with ``num_qubits`` qubits.
        self.stabs = Gens(num_qubits)
        self.destabs = Gens(num_qubits)
        self.gens = (self.stabs, self.destabs)

        # Initialize all qubits in the zero state
        self.stabs.init_all_z()
        self.destabs.init_all_x()

    def logical_sign(self,
                     logical_op: QuantumCircuit,
                     # delogical_op: Optional[QuantumCircuit] = None
                     ) -> int:
        """

        Args:
            logical_op:

        Returns:

        """
        return find_logical_signs(self, logical_op,
                                  # delogical_op
                                  )

    def refactor(self,
                 xs: Set[int],
                 zs: Set[int],
                 choose=None,
                 prefer=None,
                 protected=None):

        return refactor_generators(self, xs, zs, choose, prefer, protected)

    def find_stab(self,
                  xs: Set[int],
                  zs: Set[int]):

        return find_stabilizer(self, xs, zs)

    def run_direct(self,
                   symbol: str,
                   location: Set[Union[int, Tuple[int, ...]]],
                   **gate_kwargs: Any):
        self.bindings[symbol](self, location, **gate_kwargs)

    def copy(self):

        new = SparseSim(self.num_qubits)

        old_stabs = self.stabs
        old_destabs = self.destabs

        new_gens = (new.stabs, new.destabs)

        for i, gen in enumerate([old_stabs, old_destabs]):

            new_gen = new_gens[i]

            new_gen.signs_minus = set()
            new_gen.signs_i = set()

            new_gen.signs_minus.update(gen.signs_minus)
            new_gen.signs_i.update(gen.signs_i)

            for j in range(self.num_qubits):

                new_gen.row_x[j] = set()
                new_gen.row_z[j] = set()
                new_gen.col_x[j] = set()
                new_gen.col_z[j] = set()

                new_gen.row_x[j].update(gen.row_x[j])
                new_gen.row_z[j].update(gen.row_z[j])
                new_gen.col_x[j].update(gen.col_x[j])
                new_gen.col_z[j].update(gen.col_z[j])

    @staticmethod
    def _pauli_sign(gen,
                    i_gen: int) -> str:

        if i_gen in gen.signs_minus:
            if i_gen in gen.signs_i:
                sign = '-i'
            else:
                sign = ' -'
        else:

            if i_gen in gen.signs_i:
                sign = ' i'
            else:
                sign = '  '

        return sign

    def col_string(self,
                   gen,
                   num_qubits: Optional[int] = None,
                   print_signs: bool = True,
                   print_y: bool = False):
        """
        Prints out the stabilizers for the column-wise sparse representation.

        Args:
            gen (Gens): A generator instance.
            num_qubits (Optional[int]): number of qubits.
            print_signs (bool): Whether to print the signs of the generators.
            print_y (bool):

        Returns:

        """

        col_x = gen.col_x
        col_z = gen.col_z

        result = []

        if num_qubits is None:
            num_qubits = self.num_qubits

        for i_gen in range(num_qubits):

            stab_letters = []

            # ---- Signs ---- #
            if print_signs:
                sign = self._pauli_sign(gen, i_gen)
            else:
                sign = '  '

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

            if print_y:
                num_minus_is = 0
                for i, letter in enumerate(stab_letters):
                    if letter == 'W':
                        stab_letters[i] = 'Y'
                        num_minus_is += 1

                if print_signs:
                    # 1, i, -1, -i  ... i
                    # 1, -i, -1, i
                    has_i = num_minus_is % 2
                    has_minus = int(num_minus_is % 4 == 1 or num_minus_is % 4 == 2)

                    if sign == ' -':
                        has_minus += 1
                    elif sign == ' i':
                        has_i += 1
                    elif sign == '-i':
                        has_minus += 1
                        has_i += 1

                    has_i %= 2
                    has_minus %= 2

                    if has_i:
                        if has_minus:
                            stab_letters[0] = '-i'
                        else:
                            stab_letters[0] = ' i'
                    else:
                        if has_minus:
                            stab_letters[0] = ' -'
                        else:
                            stab_letters[0] = '  '

            # print(''.join(stab_letters))
            result.append(''.join(stab_letters))

        return result

    def print_stabs(self,
                    verbose: bool = True,
                    print_y: bool = True,
                    print_destabs: bool = False):

        str_s = self.print_tableau(self.stabs, verbose=verbose, print_y=print_y)

        if print_destabs:
            if verbose:
                print('-------------------------------')
            str_d = self.print_tableau(self.destabs, verbose=verbose, print_signs=False, print_y=print_y)

            return str_s, str_d

        return str_s

    def print_tableau(self,
                      gen,
                      verbose: bool = True,
                      print_signs: bool = True,
                      print_y: bool = True):
        """
        Prints out the stabilizers.
        :return:
        """

        col_str = self.col_string(gen, print_signs=print_signs, print_y=print_y)
        row_str = self.row_string(gen, print_signs=print_signs, print_y=print_y)

        if col_str != row_str:
            print('col')
            for line in col_str:
                print(line)

            print('\nrow')
            for line in row_str:
                print(line)

            raise Exception('Something bad happened! String representation of the row-wise vs column-wise '
                            'stabilizers do not match!')

        if verbose:
            for line in col_str:
                print(line)

        return col_str

    def row_string(self,
                   gen,
                   num_qubits: Optional[int] = None,
                   print_signs: bool = True,
                   print_y: bool = False) -> List[str]:
        """
        Prints out the stabilizers for the row-wise sparse representation.

        Args:
            gen (Gens): A generator instance.
            num_qubits (int): number of qubits.
            print_signs (bool): Whether to print the signs of the generators.
            print_y (bool):

        Returns:

        """

        row_x = gen.row_x
        row_z = gen.row_z

        result = []

        if num_qubits is None:
            num_qubits = gen.num_qubits

        for i_gen in range(num_qubits):

            stab_letters = []

            # ---- Signs ---- #
            if print_signs:
                sign = self._pauli_sign(gen, i_gen)
            else:
                sign = '  '
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

            if print_y:
                num_minus_is = 0
                for i, letter in enumerate(stab_letters):
                    if letter == 'W':
                        stab_letters[i] = 'Y'
                        num_minus_is += 1

                if print_signs:
                    # 1, i, -1, -i  ... i
                    # 1, -i, -1, i
                    has_i = num_minus_is % 2
                    has_minus = int(num_minus_is % 4 == 1 or num_minus_is % 4 == 2)

                    if sign == ' -':
                        has_minus += 1
                    elif sign == ' i':
                        has_i += 1
                    elif sign == '-i':
                        has_minus += 1
                        has_i += 1

                    has_i %= 2
                    has_minus %= 2

                    if has_i:
                        if has_minus:
                            stab_letters[0] = '-i'
                        else:
                            stab_letters[0] = ' i'
                    else:
                        if has_minus:
                            stab_letters[0] = ' -'
                        else:
                            stab_letters[0] = '  '

            # print(''.join(stab_letters))
            result.append(''.join(stab_letters))

        return result


class Gens(object):
    """
    This class is the data structure used for tracking stabilizer/destabilizer generators.
    """

    def __init__(self, num_qubits: int) -> None:
        """
        :param num_qubits: Number of qubits to simulate.
        """

        self.num_qubits = num_qubits

        self.col_x = [set() for _ in range(num_qubits)]
        self.col_z = [set() for _ in range(num_qubits)]

        self.row_x = [set() for _ in range(num_qubits)]
        self.row_z = [set() for _ in range(num_qubits)]

        self.signs_minus = set()
        self.signs_i = set()

    def init_all_z(self) -> None:
        """
        Used to initiate stabilizers to all Zs.

        :return:
        :rtype:
        """

        self.signs_minus = set()
        self.signs_i = set()

        self.col_x = [set() for _ in range(self.num_qubits)]
        self.col_z = [{i} for i in range(self.num_qubits)]

        self.row_x = [set() for _ in range(self.num_qubits)]
        self.row_z = [{i} for i in range(self.num_qubits)]

    def init_all_x(self) -> None:
        """
        Used to initiate destabilizers to all Xs.

        :return:
        :rtype:
        """

        self.signs_minus = set()
        self.signs_i = set()

        self.col_x = [{i} for i in range(self.num_qubits)]
        self.col_z = [set() for _ in range(self.num_qubits)]

        self.row_x = [{i} for i in range(self.num_qubits)]
        self.row_z = [set() for _ in range(self.num_qubits)]

    def _pauli_sign(self, i_gen: int) -> str:

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

    def col_string(self, num_qubits: Optional[int] = None) -> List[str]:
        """
        Prints out the stabilizers for the column-wise sparse representation.

        :param num_qubits:
        :return:
        """

        result = []

        if num_qubits is None:
            num_qubits = self.num_qubits

        for i_gen in range(num_qubits):

            stab_letters = []

            # ---- Signs ---- #
            sign = self._pauli_sign(i_gen)
            stab_letters.append(sign)

            # ---- Paulis ---- #
            for qubit in range(num_qubits):

                letter = 'U'

                if i_gen in self.col_x[qubit] and i_gen not in self.col_z[qubit]:
                    letter = 'X'

                elif i_gen not in self.col_x[qubit] and i_gen in self.col_z[qubit]:
                    letter = 'Z'

                elif i_gen in self.col_x[qubit] and i_gen in self.col_z[qubit]:
                    letter = 'W'

                elif i_gen not in self.col_x[qubit] and i_gen not in self.col_z[qubit]:
                    letter = 'I'

                stab_letters.append(letter)

            # print(''.join(stab_letters))
            result.append(''.join(stab_letters))

        return result

    def print_tableau(self, verbose: bool = True) -> List[str]:
        """
        Prints out the stabilizers.
        :return:
        """

        col_str = self.col_string()
        row_str = self.row_string()

        if col_str != row_str:
            raise Exception('Something bad happened! String representation of the row-wise vs column-wile '
                            'spare stabilizers do not match!')

        if verbose:
            for line in col_str:
                print(line)

        return col_str

    def row_string(self, num_qubits: Optional[int] = None) -> List[str]:
        """
        Prints out the stabilizers for the row-wise sparse representation.

        :param num_qubits:
        :return:
        """

        result = []

        if num_qubits is None:
            num_qubits = self.num_qubits

        for i_gen in range(num_qubits):

            stab_letters = []

            # ---- Signs ---- #
            sign = self._pauli_sign(i_gen)
            stab_letters.append(sign)

            # ---- Paulis ---- #
            for qubit in range(num_qubits):

                letter = 'U'

                if qubit in self.row_x[i_gen] and qubit not in self.row_z[i_gen]:
                    letter = 'X'

                elif qubit not in self.row_x[i_gen] and qubit in self.row_z[i_gen]:
                    letter = 'Z'

                elif qubit in self.row_x[i_gen] and qubit in self.row_z[i_gen]:
                    letter = 'W'

                elif qubit not in self.row_x[i_gen] and qubit not in self.row_z[i_gen]:
                    letter = 'I'

                stab_letters.append(letter)

            result.append(''.join(stab_letters))

        return result
