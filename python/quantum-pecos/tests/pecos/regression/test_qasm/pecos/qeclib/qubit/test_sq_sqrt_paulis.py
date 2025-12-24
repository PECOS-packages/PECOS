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

"""QASM regression tests for single-qubit square root Pauli gates."""

from collections.abc import Callable

from pecos.qeclib import qubit
from pecos.slr import QReg


def test_SX(compare_qasm: Callable[..., None]) -> None:
    """Test SX square root Pauli gate QASM regression."""
    q = QReg("q_test", 2)
    prog = qubit.SX(q[1])
    compare_qasm(prog)


def test_SXdg(compare_qasm: Callable[..., None]) -> None:
    """Test SXdg square root Pauli gate QASM regression."""
    q = QReg("q_test", 2)
    prog = qubit.SXdg(q[1])
    compare_qasm(prog)


def test_SY(compare_qasm: Callable[..., None]) -> None:
    """Test SY square root Pauli gate QASM regression."""
    q = QReg("q_test", 2)
    prog = qubit.SY(q[1])
    compare_qasm(prog)


def test_SYdg(compare_qasm: Callable[..., None]) -> None:
    """Test SYdg square root Pauli gate QASM regression."""
    q = QReg("q_test", 2)
    prog = qubit.SYdg(q[1])
    compare_qasm(prog)


def test_SZ(compare_qasm: Callable[..., None]) -> None:
    """Test SZ square root Pauli gate QASM regression."""
    q = QReg("q_test", 2)
    prog = qubit.SZ(q[1])
    compare_qasm(prog)


def test_SZdg(compare_qasm: Callable[..., None]) -> None:
    """Test SZdg square root Pauli gate QASM regression."""
    q = QReg("q_test", 2)
    prog = qubit.SZdg(q[1])
    compare_qasm(prog)
