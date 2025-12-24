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

"""QASM regression tests for Steane three-parallel flagged syndrome extraction."""

from collections.abc import Callable

from pecos.qeclib.steane.syn_extract.three_parallel_flagging import (
    ThreeParallelFlaggingXZZ,
    ThreeParallelFlaggingZXX,
)
from pecos.slr import CReg, QReg


def test_ThreeParallelFlaggingXZZ(compare_qasm: Callable[..., None]) -> None:
    """Test Steane three-parallel XZZ flagged syndrome extraction QASM regression."""
    q = QReg("q_test", 7)
    a = QReg("a_test", 3)
    flag_x = CReg("flag_x_test", 3)
    flag_z = CReg("flag_z_test", 3)
    flags = CReg("flags_test", 3)
    last_raw_syn_x = CReg("last_raw_syn_x_test", 3)
    last_raw_syn_z = CReg("last_raw_syn_z_test", 3)

    block = ThreeParallelFlaggingXZZ(
        q,
        a,
        flag_x,
        flag_z,
        flags,
        last_raw_syn_x,
        last_raw_syn_z,
    )
    compare_qasm(block)


def test_ThreeParallelFlaggingZXX(compare_qasm: Callable[..., None]) -> None:
    """Test Steane three-parallel ZXX flagged syndrome extraction QASM regression."""
    q = QReg("q_test", 7)
    a = QReg("a_test", 3)
    flag_x = CReg("flag_x_test", 3)
    flag_z = CReg("flag_z_test", 3)
    flags = CReg("flags_test", 3)
    last_raw_syn_x = CReg("last_raw_syn_x_test", 3)
    last_raw_syn_z = CReg("last_raw_syn_z_test", 3)

    block = ThreeParallelFlaggingZXX(
        q,
        a,
        flag_x,
        flag_z,
        flags,
        last_raw_syn_x,
        last_raw_syn_z,
    )
    compare_qasm(block)
