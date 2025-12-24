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

"""QASM regression tests for single-qubit face rotation gates."""

from collections.abc import Callable

from pecos.qeclib import qubit
from pecos.slr import QReg


def test_F(compare_qasm: Callable[..., None]) -> None:
    """Test F face rotation gate QASM regression."""
    q = QReg("q_test", 2)

    prog = qubit.F(q[1])
    compare_qasm(prog)


def test_Fdg(compare_qasm: Callable[..., None]) -> None:
    """Test Fdg face rotation gate QASM regression."""
    q = QReg("q_test", 2)

    prog = qubit.Fdg(q[1])
    compare_qasm(prog)


def test_F4(compare_qasm: Callable[..., None]) -> None:
    """Test F4 face rotation gate QASM regression."""
    q = QReg("q_test", 2)

    prog = qubit.F4(q[1])
    compare_qasm(prog)


def test_F4dg(compare_qasm: Callable[..., None]) -> None:
    """Test F4dg face rotation gate QASM regression."""
    q = QReg("q_test", 2)

    prog = qubit.F4dg(q[1])
    compare_qasm(prog)
