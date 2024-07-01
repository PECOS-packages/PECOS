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

from pecos.qeclib.qubit.metaclasses import SingleQubitUnitary, TwoQubitUnitary


class RXGate(SingleQubitUnitary): ...


RX = RXGate(qasm_sym="rx")


class RYGate(SingleQubitUnitary): ...


RY = RYGate(qasm_sym="ry")


class RZGate(SingleQubitUnitary): ...


RZ = RZGate(qasm_sym="rz")


class RZZGate(TwoQubitUnitary): ...


RZZ = RZZGate(qasm_sym="rzz")
