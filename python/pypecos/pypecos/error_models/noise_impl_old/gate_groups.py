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

from __future__ import annotations

import numpy as np

two_qubits = {
    "CNOT",
    "CZ",
    "SWAP",
    "G",
    "MS",
    "SqrtXX",
    "SqrtZZ",
    "RXX",
    "RYY",
    "RZZ",
    "CX",
    "SXX",
    "SXXdg",
    "SYY",
    "SYYdg",
    "SZZ",
    "SZZdg",
}

one_qubits = {
    "I",
    "X",
    "Y",
    "Z",
    "Q",
    "Qd",
    "R",
    "Rd",
    "S",
    "Sd",
    "H",
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
    "H6",
    "H+z+x",
    "H-z-x",
    "H+y-z",
    "H-y-z",
    "H-x+y",
    "H-x-y",
    "F1",
    "F1d",
    "F2",
    "F2d",
    "F3",
    "F3d",
    "F4",
    "F4d",
    "SqrtX",
    "SqrtXd",
    "SqrtY",
    "SqrtYd",
    "SqrtZ",
    "SqrtZd",
    "RZ",
    "U1q",
    "RXY1Q",
    "SX",
    "SXdg",
    "SY",
    "SYdg",
    "SZ",
    "SZdg",
    "F",
    "Fdg",
    "R1XY",
}

zbasis_rotations = {"Z", "S", "Sd", "SqrtZ", "SqrtZd", "RZ", "SZ", "SZdg"}

measurementsx = {"measure Z", "Measure", "Measure +Z"}
measurementsz = {"measure X", "measure Y", "Measure +X", "Measure +Y"}
initsx = {"init |0>", "init |1>"}
initsz = {"init |+>", "init |->", "init |+i>", "init |-i>"}

error_two_paulis_collection = np.array(
    [
        (None, "X"),
        (None, "Y"),
        (None, "Z"),
        ("X", None),
        ("X", "X"),
        ("X", "Y"),
        ("X", "Z"),
        ("Y", None),
        ("Y", "X"),
        ("Y", "Y"),
        ("Y", "Z"),
        ("Z", None),
        ("Z", "X"),
        ("Z", "Y"),
        ("Z", "Z"),
    ],
)

error_one_paulis_collection = np.array(["X", "Y", "Z"])

# Residual one-qubit gates that are applied due to leakage cause the MS gate not to be applied but the non-leaked
# qubits still gets the wrapper one-gates applied.
residual_gates_dict = {
    "MS": None,
    "SqrtZZ": None,
    "CNOT": ("Sd", "Qd"),
    "CZ": ("Sd", "Sd"),
    "CY": ("Sd", "Rd"),
}
