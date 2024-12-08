# Copyright 2021 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# ruff: noqa: F401

from pecos.circuits.qasm.barrier import Barrier
from pecos.circuits.qasm.conditionals import CIf, CIfExpect
from pecos.circuits.qasm.expr import Assign
from pecos.circuits.qasm.gates import ArgGate, Gate, MeasGate, ResetGate
from pecos.circuits.qasm.misc import Comment

Measure = MeasGate()
Reset = ResetGate()

X = Gate("x", 1)
Y = Gate("y", 1)
Z = Gate("z", 1)

H = Gate("h", 1)
S = Gate("s", 1)
Sdg = Gate("sdg", 1)
T = Gate("t", 1)
Tdg = Gate("tdg", 1)

CNOT = Gate("cx", 2)
CX = Gate("cx", 2)
CY = Gate("cy", 2)
CZ = Gate("cz", 2)

RX = ArgGate("rx", 1, 1)
RY = ArgGate("ry", 1, 1)
RZ = ArgGate("rz", 1, 1)
