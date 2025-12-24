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

"""QASM regression tests for qubit measurement operations."""

from collections.abc import Callable

from pecos.qeclib import qubit
from pecos.slr import CReg, QReg


def test_Measure(compare_qasm: Callable[..., None]) -> None:
    """Test Measure gate QASM regression."""
    q = QReg("q_test", 1)
    m = CReg("m_test", 1)

    prog = qubit.Measure(q[0]) > m[0]
    compare_qasm(prog)
