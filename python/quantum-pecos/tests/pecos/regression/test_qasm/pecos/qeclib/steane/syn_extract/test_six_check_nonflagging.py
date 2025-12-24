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

"""QASM regression tests for Steane six-check non-flagged syndrome extraction."""

from collections.abc import Callable

from pecos.qeclib.steane.syn_extract.six_check_nonflagging import SixUnflaggedSyn
from pecos.slr import CReg, QReg


def test_SixUnflaggedSyn(compare_qasm: Callable[..., None]) -> None:
    """Test Steane six-check non-flagged syndrome extraction QASM regression."""
    q = QReg("q_test", 7)
    a = QReg("a_test", 3)
    syn_x = CReg("syn_x_test", 3)
    syn_z = CReg("syn_z_test", 3)

    block = SixUnflaggedSyn(q, a, syn_x, syn_z)
    compare_qasm(block)
