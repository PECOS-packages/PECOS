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

from pecos.slr import CReg, QReg
from pecos.slr.qeclib import qubit


def test_Measure(compare_qasm: Callable[..., None]) -> None:
    """Test MeasureZ against the original measurement QASM golden."""
    q = QReg("q_test", 1)
    m = CReg("m_test", 1)

    prog = qubit.MeasureZ(q[0]) > m[0]
    compare_qasm(prog, filename="pecos.slr.qeclib.qubit.measures.Measure")
