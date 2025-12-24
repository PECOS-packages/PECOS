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

"""QASM regression tests for Steane single-qubit Pauli gates."""

from collections.abc import Callable

from pecos.qeclib.steane.gates_sq.paulis import X, Y, Z
from pecos.slr import QReg


def test_X(compare_qasm: Callable[..., None]) -> None:
    """Test Steane X Pauli gate QASM regression."""
    q = QReg("q_test", 7)

    block = X(q)
    compare_qasm(block)


def test_Y(compare_qasm: Callable[..., None]) -> None:
    """Test Steane Y Pauli gate QASM regression."""
    q = QReg("q_test", 7)

    block = Y(q)
    compare_qasm(block)


def test_Z(compare_qasm: Callable[..., None]) -> None:
    """Test Steane Z Pauli gate QASM regression."""
    q = QReg("q_test", 7)

    block = Z(q)
    compare_qasm(block)
