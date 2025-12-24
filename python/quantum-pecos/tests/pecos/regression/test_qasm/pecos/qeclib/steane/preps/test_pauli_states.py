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

"""QASM regression tests for Steane Pauli state preparations."""

from collections.abc import Callable

from pecos.qeclib.steane.preps.pauli_states import (
    LogZeroRot,
    PrepEncodingFTZero,
    PrepEncodingNonFTZero,
    PrepRUS,
    PrepZeroVerify,
)
from pecos.slr import CReg, QReg


def test_PrepEncodingNonFTZero(compare_qasm: Callable[..., None]) -> None:
    """Test Steane non-fault-tolerant zero state preparation QASM regression."""
    q = QReg("q_test", 7)
    block = PrepEncodingNonFTZero(q)
    compare_qasm(block)


def test_PrepZeroVerify(compare_qasm: Callable[..., None]) -> None:
    """Test Steane zero state verification QASM regression."""
    q = QReg("q_test", 7)
    a = QReg("a_test", 1)
    init_bit = CReg("init_bit", 1)
    for reset_ancilla in [True, False]:
        block = PrepZeroVerify(q, a[0], init_bit[0], reset_ancilla=reset_ancilla)
        compare_qasm(block, reset_ancilla)


def test_PrepEncodingFTZero(compare_qasm: Callable[..., None]) -> None:
    """Test Steane fault-tolerant zero state preparation QASM regression."""
    q = QReg("q_test", 7)
    a = QReg("a_test", 1)
    init_bit = CReg("init_bit_test", 1)

    for reset in [True, False]:
        block = PrepEncodingFTZero(q, a[0], init_bit[0], reset=reset)
        compare_qasm(block, reset)


def test_PrepRUS(compare_qasm: Callable[..., None]) -> None:
    """Test Steane repeat-until-success state preparation QASM regression."""
    q = QReg("q_test", 7)
    a = QReg("a_test", 1)
    init = CReg("init_test", 1)

    for limit in [1, 2, 3]:
        for state in ["-Z", "+Z", "+X", "-X", "+Y", "-Y"]:
            for first_round_reset in [True, False]:
                block = PrepRUS(
                    q,
                    a[0],
                    init[0],
                    limit,
                    state,
                    first_round_reset=first_round_reset,
                )
                compare_qasm(block, limit, state, first_round_reset)


def test_LogZeroRot(compare_qasm: Callable[..., None]) -> None:
    """Test Steane logical zero rotation QASM regression."""
    q = QReg("q_test", 7)

    for state in ["-Z", "+Z", "+X", "-X", "+Y", "-Y"]:
        block = LogZeroRot(q, state)
        compare_qasm(block, state)
