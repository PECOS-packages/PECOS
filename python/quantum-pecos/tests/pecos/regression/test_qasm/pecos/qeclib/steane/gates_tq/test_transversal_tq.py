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

"""QASM regression tests for Steane transversal two-qubit gates."""

from collections.abc import Callable

from pecos.qeclib.steane.gates_tq.transversal_tq import CX, CY, CZ, SZZ
from pecos.slr import QReg


def test_CX(compare_qasm: Callable[..., None]) -> None:
    """Test Steane transversal CX gate QASM regression."""
    q1 = QReg("q1_test", 7)
    q2 = QReg("q2_test", 7)

    for barrier in [True, False]:
        block = CX(q1, q2, barrier=barrier)
        compare_qasm(block, barrier)


def test_CY(compare_qasm: Callable[..., None]) -> None:
    """Test Steane transversal CY gate QASM regression."""
    q1 = QReg("q1_test", 7)
    q2 = QReg("q2_test", 7)

    block = CY(q1, q2)
    compare_qasm(block)


def test_CZ(compare_qasm: Callable[..., None]) -> None:
    """Test Steane transversal CZ gate QASM regression."""
    q1 = QReg("q1_test", 7)
    q2 = QReg("q2_test", 7)

    block = CZ(q1, q2)
    compare_qasm(block)


def test_SZZ(compare_qasm: Callable[..., None]) -> None:
    """Test Steane transversal SZZ gate QASM regression."""
    q1 = QReg("q1_test", 7)
    q2 = QReg("q2_test", 7)

    block = SZZ(q1, q2)
    compare_qasm(block)
