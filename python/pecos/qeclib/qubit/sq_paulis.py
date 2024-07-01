# Copyright 2024 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from numpy import array

from pecos.qeclib.qubit.metaclasses import SQPauliGate


class XGate(SQPauliGate):
    """The Pauli X unitary."""

    def __init__(self):
        super().__init__("X", qasm_sym="x")

        self.matrix = array(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=complex,
        )

        self.pauli_rules = {
            "X": "+X",
            "Z": "-Z",
        }


X = XGate()


class YGate(SQPauliGate):
    """The Pauli Y unitary."""

    def __init__(self):
        super().__init__("Y", qasm_sym="y")

        self.matrix = array(
            [
                [0, -1j],
                [1j, 0],
            ],
            dtype=complex,
        )

        self.pauli_rules = {
            "X": "-X",
            "Z": "-Z",
        }


Y = YGate()


class ZGate(SQPauliGate):
    """The Pauli Z unitary."""

    def __init__(self):
        super().__init__("Z", qasm_sym="z")

        self.matrix = array(
            [
                [1, 0],
                [0, -1],
            ],
            dtype=complex,
        )

        self.pauli_rules = {
            "X": "-X",
            "Z": "+Z",
        }


Z = ZGate()
