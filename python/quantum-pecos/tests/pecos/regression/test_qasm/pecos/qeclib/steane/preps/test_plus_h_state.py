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

"""QASM regression tests for Steane H+ state preparation."""

from collections.abc import Callable

from pecos.qeclib.steane.preps.plus_h_state import PrepHStateFT
from pecos.slr import CReg, QReg


def test_PrepHStateFT(compare_qasm: Callable[..., None]) -> None:
    """Test Steane fault-tolerant H state preparation QASM regression."""
    q = QReg("q_test", 7)
    a = QReg("a_test", 3)
    out = CReg("out_test", 2)
    reject = CReg("reject_test", 1)
    flag_x = CReg("flag_x_test", 3)
    flag_z = CReg("flag_z_test", 3)
    flags = CReg("flags_test", 3)
    last_raw_syn_x = CReg("last_raw_syn_x_test", 3)
    last_raw_syn_z = CReg("last_raw_syn_z_test", 3)
    block = PrepHStateFT(
        q,
        a,
        out,
        reject[0],
        flag_x,
        flag_z,
        flags,
        last_raw_syn_x,
        last_raw_syn_z,
    )
    compare_qasm(block)
