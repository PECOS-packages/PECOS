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

"""QASM regression tests for Steane single-qubit square root Pauli gates."""

from collections.abc import Callable

from pecos.qeclib.steane.gates_sq.sqrt_paulis import SX, SY, SZ, SXdg, SYdg, SZdg
from pecos.slr import QReg


def test_SX(compare_qasm: Callable[..., None]) -> None:
    """Test Steane SX square root Pauli gate QASM regression."""
    q = QReg("q_test", 7)

    block = SX(q)
    compare_qasm(block)


def test_SXdg(compare_qasm: Callable[..., None]) -> None:
    """Test Steane SXdg square root Pauli gate QASM regression."""
    q = QReg("q_test", 7)

    block = SXdg(q)
    compare_qasm(block)


def test_SY(compare_qasm: Callable[..., None]) -> None:
    """Test Steane SY square root Pauli gate QASM regression."""
    q = QReg("q_test", 7)

    block = SY(q)
    compare_qasm(block)


def test_SYdg(compare_qasm: Callable[..., None]) -> None:
    """Test Steane SYdg square root Pauli gate QASM regression."""
    q = QReg("q_test", 7)

    block = SYdg(q)
    compare_qasm(block)


def test_SZ(compare_qasm: Callable[..., None]) -> None:
    """Test Steane SZ square root Pauli gate QASM regression."""
    q = QReg("q_test", 7)

    block = SZ(q)
    compare_qasm(block)


def test_SZdg(compare_qasm: Callable[..., None]) -> None:
    """Test Steane SZdg square root Pauli gate QASM regression."""
    q = QReg("q_test", 7)

    block = SZdg(q)
    compare_qasm(block)
