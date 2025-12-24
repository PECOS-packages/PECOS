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

"""QASM regression tests for Steane code lookup decoders."""

from collections.abc import Callable

from pecos.qeclib.steane.decoders.lookup import (
    FlagLookupQASM,
    FlagLookupQASMActiveCorrectionX,
    FlagLookupQASMActiveCorrectionZ,
)
from pecos.slr import CReg, QReg


def test_FlagLookupQASM(compare_qasm: Callable[..., None]) -> None:
    """Test Steane flag lookup decoder QASM regression."""
    syn = CReg("syn_test", 3)
    syndromes = CReg("syndromes_test", 3)
    raw_syn = CReg("raw_syn_test", 3)
    pf = CReg("pf_test", 2)
    flag = CReg("flag_test", 1)
    flags = CReg("flags_test", 3)
    scratch = CReg("scratch_test", 32)

    for basis in ["X", "Y", "Z"]:
        block = FlagLookupQASM(
            basis,
            syn,
            syndromes,
            raw_syn,
            pf[0],
            flag,
            flags,
            scratch,
        )
        compare_qasm(block, basis)


def test_FlagLookupQASMActiveCorrectionX(compare_qasm: Callable[..., None]) -> None:
    """Test Steane flag lookup decoder with X correction QASM regression."""
    q = QReg("q_test", 7)
    syn = CReg("syn_test", 3)
    syndromes = CReg("syndromes_test", 3)
    raw_syn = CReg("raw_syn_test", 3)
    pf = CReg("pf_test", 2)
    flag = CReg("flag_test", 1)
    flags = CReg("flags_test", 3)
    scratch = CReg("scratch_test", 32)
    pf_copy = CReg("pf_copy_test", 1)

    for pf_bit_copy in [None, pf_copy]:
        block = FlagLookupQASMActiveCorrectionX(
            q,
            syn,
            syndromes,
            raw_syn,
            pf[0],
            flag,
            flags,
            scratch,
            pf_bit_copy,
        )
        compare_qasm(block, pf_bit_copy)


def test_FlagLookupQASMActiveCorrectionZ(compare_qasm: Callable[..., None]) -> None:
    """Test Steane flag lookup decoder with Z correction QASM regression."""
    q = QReg("q_test", 7)
    syn = CReg("syn_test", 3)
    syndromes = CReg("syndromes_test", 3)
    raw_syn = CReg("raw_syn_test", 3)
    pf = CReg("pf_test", 2)
    flag = CReg("flag_test", 1)
    flags = CReg("flags_test", 3)
    scratch = CReg("scratch_test", 32)
    pf_copy = CReg("pf_copy_test", 1)

    for pf_bit_copy in [None, pf_copy]:
        block = FlagLookupQASMActiveCorrectionZ(
            q,
            syn,
            syndromes,
            raw_syn,
            pf[0],
            flag,
            flags,
            scratch,
            pf_bit_copy,
        )
        compare_qasm(block, pf_bit_copy)
