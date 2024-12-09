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

from pecos.qeclib.qubit.measures import Measure
from pecos.qeclib.qubit.preps import Prep
from pecos.qeclib.qubit.rots import RX, RY, RZ, RZZ
from pecos.qeclib.qubit.sq_face_rots import F4, F, F4dg, Fdg
from pecos.qeclib.qubit.sq_hadamards import H
from pecos.qeclib.qubit.sq_noncliffords import T, Tdg
from pecos.qeclib.qubit.sq_paulis import X, Y, Z
from pecos.qeclib.qubit.sq_sqrt_paulis import SX, SY, SZ, SXdg, SYdg, SZdg
from pecos.qeclib.qubit.tq_cliffords import (
    CX,
    CY,
    CZ,
    SXX,
    SYY,
    SZZ,
    SXXdg,
    SYYdg,
    SZZdg,
)
from pecos.qeclib.qubit.tq_noncliffords import CH

__all__ = [
    "CH",
    "CX",
    "CY",
    "CZ",
    "F4",
    "RX",
    "RY",
    "RZ",
    "RZZ",
    "SX",
    "SXX",
    "SY",
    "SYY",
    "SZ",
    "SZZ",
    "F",
    "F4dg",
    "Fdg",
    "H",
    "Measure",
    "Prep",
    "SXXdg",
    "SXdg",
    "SYYdg",
    "SYdg",
    "SZZdg",
    "SZdg",
    "T",
    "Tdg",
    "X",
    "Y",
    "Z",
]
