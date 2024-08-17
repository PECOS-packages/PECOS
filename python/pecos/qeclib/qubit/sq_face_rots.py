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

from pecos.qeclib.qubit.metaclasses import SQCliffordGate


class FGate(SQCliffordGate):
    """Face  rotation.

    Sdg; H; = SX; SZ; = SY; SX; = SZ; SY;

    X -> Y -> Z -> X

    X -> Y
    Z -> X
    Y -> Z
    """

    def qasm(self):
        str_list = []
        for q in self.qargs:
            str_list.append(f"rx(pi/2) {str(q)};\nrz(pi/2) {str(q)};")

        return " ".join(str_list)


F = FGate("F")


class FdgGate(SQCliffordGate):
    """Adjoint of the face  rotations.

    H; S; = SXdg; SYdg = SYdg; SZdg; = SZdg; SXdg;

    X -> Z -> Y -> X

    X -> Z
    Z -> Y
    Y -> X
    """

    def qasm(self):
        str_list = []
        for q in self.qargs:
            str_list.append(f"ry(-pi/2) {str(q)};\nrz(-pi/2) {str(q)};")

        return " ".join(str_list)


Fdg = FdgGate("Fdg")


class F4Gate(SQCliffordGate):
    """Face  4 rotation.

    H; Sdg = SX; SYdg = SYdg; SZ; = SZ; SX;

    X -> Z -> -Y -> X

    X -> Z
    Z -> -Y
    Y -> -X
    """

    def qasm(self):
        str_list = []
        for q in self.qargs:
            str_list.append(f"ry(-pi/2) {str(q)};\nrz(pi/2) {str(q)};")

        return " ".join(str_list)


F4 = F4Gate("F4")


class F4dgGate(SQCliffordGate):
    """Adjoint of the face 4  rotation.

    S; H; = SY; SXdg = SZdg; SY; = SXdg; SZdg;

    X -> -Y -> Z -> X

    X -> -Y
    Z -> X

    Y -> -Z
    """

    def qasm(self):
        str_list = []
        for q in self.qargs:
            str_list.append(f"rx(-pi/2) {str(q)};\nrz(-pi/2) {str(q)};")

        return " ".join(str_list)


F4dg = F4dgGate("F4dg")
