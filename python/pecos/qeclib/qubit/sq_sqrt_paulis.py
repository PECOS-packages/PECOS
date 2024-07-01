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


class SXGate(SQCliffordGate):
    """
    X -> X
    Z -> -Y
    Y -> Z
    """


SX = SXGate(qasm_sym="rx(pi/2)")


class SYGate(SQCliffordGate): ...


SY = SYGate(qasm_sym="ry(pi/2)")


class SZGate(SQCliffordGate): ...


S = SZ = SZGate(qasm_sym="rz(pi/2)")


class SXdgGate(SQCliffordGate): ...


SXdg = SXdgGate(qasm_sym="rx(-pi/2)")


class SYdgGate(SQCliffordGate): ...


SYdg = SYdgGate(qasm_sym="ry(-pi/2)")


class SZdgGate(SQCliffordGate): ...


Sdg = SZdg = SZdgGate(qasm_sym="rz(-pi/2)")
