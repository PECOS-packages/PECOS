#  =========================================================================  #
#   Copyright 2018 Ciarán Ryan-Anderson
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

from collections import OrderedDict


class Checks(object):
    """
    Data structure used to record checks for Pauli qubit stabilizer-codes.

    Maybe expanded in the future to non-Pauli, qudit checks.
    """

    def __init__(self):

        self._checks = []

    def add(self, paulis, qubits, **params):

        check_dict = {
            'Paulis': {},
            'qubits': OrderedDict(),
            'Pauli type': None,
            'params': params
        }

        lenq = len(qubits)
        if lenq != len(set(qubits)):
            raise Exception('Qubit ids are not unique!')

        if isinstance(paulis, str):

            paulis = paulis.upper()

            if paulis not in ['X', 'Y', 'Z']:
                raise Exception('Pauli type should be "X", "Y" or "Z"!')

            check_dict['type'] = paulis

            for q in qubits:
                if not isinstance(q, int):
                    raise Exception('Qubit ids must be integers!')

                check_dict['qubits'][q] = paulis

            check_dict['Paulis'][paulis] = set(qubits)

        else:

            if len(paulis) != len(qubits):
                raise Exception('Number of Paulis must match number of qubits!')

            for p, q in zip(paulis, qubits):

                p = p.upper()

                if p not in ['X', 'Y', 'Z']:
                    raise Exception('Pauli should be "X", "Y" or "Z"!')

                if not isinstance(q, int):
                    raise Exception('Qubit ids must be integers!')

                if p not in check_dict['Paulis']:
                    check_dict['Paulis'][p] = {q, }
                else:
                    check_dict['Paulis'][p].add(q)

            if len(check_dict['Paulis']) == 1:
                check_dict['type'] = check_dict['Paulis'][0]

    def __repr__(self):
        return self._checks

    def __str__(self):
        return str(self._checks)
