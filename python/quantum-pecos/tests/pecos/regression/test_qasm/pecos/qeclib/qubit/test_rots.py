# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""QASM regression tests for qubit rotation gates."""

from collections.abc import Callable

import pecos as pc
from pecos.qeclib import qubit
from pecos.slr import QReg


def test_RX(compare_qasm: Callable[..., None]) -> None:
    """Test RX rotation gate QASM regression."""
    q = QReg("q_test", 1)
    prog = qubit.RX[pc.f64.pi / 3](q[0])
    compare_qasm(prog)


def test_RY(compare_qasm: Callable[..., None]) -> None:
    """Test RY rotation gate QASM regression."""
    q = QReg("q_test", 1)
    prog = qubit.RY[pc.f64.pi / 3](q[0])
    compare_qasm(prog)


def test_RZ(compare_qasm: Callable[..., None]) -> None:
    """Test RZ rotation gate QASM regression."""
    q = QReg("q_test", 1)
    prog = qubit.RZ[pc.f64.pi / 3](q[0])
    compare_qasm(prog)


def test_RZZ(compare_qasm: Callable[..., None]) -> None:
    """Test RZZ two-qubit rotation gate QASM regression."""
    q = QReg("q_test", 4)
    prog = qubit.RZZ[pc.f64.pi / 3](q[1], q[3])
    compare_qasm(prog)
