# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Register-fidelity regression tests for the Rust ``run_phir_sim`` engine path.

This path (``PhirJsonEngine``) is separate from the ``HybridEngine`` classical
interpreter. These tests lock in that classical registers export losslessly at
full 64-bit width and that ``Result`` operations use the destination register.
"""

from __future__ import annotations

import json

import pytest
from pecos_rslib import run_phir_sim


def _run(ops: list[dict]) -> dict:
    return run_phir_sim(json.dumps({"format": "PHIR/JSON", "version": "0.1.0", "ops": ops}), shots=1)


def _as_int(result: dict, name: str) -> int:
    return int(result[name][0], 2)


def test_whole_register_result_preserves_high_bits() -> None:
    """A whole-register Result of a >32-bit source is not truncated to u32."""
    value = (1 << 40) + 5
    res = _run(
        [
            {"data": "cvar_define", "data_type": "u64", "variable": "m", "size": 64},
            {"cop": "=", "returns": ["m"], "args": [value]},
            {"cop": "Result", "args": ["m"], "returns": ["out"]},
        ],
    )
    assert _as_int(res, "out") == value


def test_indexed_result_exports_destination() -> None:
    """An indexed Result exports the accumulated destination, not the source value."""
    res = _run(
        [
            {"data": "cvar_define", "data_type": "u64", "variable": "m", "size": 2},
            {"cop": "=", "returns": [["m", 0]], "args": [1]},
            {"cop": "Result", "args": [["m", 0]], "returns": [["dst", 40]]},
        ],
    )
    # dst bit 40 set from m[0]; exporting the source m would wrongly give 1.
    assert _as_int(res, "dst") == (1 << 40)


def test_multiple_indexed_results_accumulate() -> None:
    """Multiple indexed Results into one register accumulate correctly."""
    res = _run(
        [
            {"data": "cvar_define", "data_type": "u64", "variable": "m", "size": 2},
            {"cop": "=", "returns": [["m", 0]], "args": [1]},
            {"cop": "=", "returns": [["m", 1]], "args": [1]},
            {"cop": "Result", "args": [["m", 0]], "returns": [["c", 0]]},
            {"cop": "Result", "args": [["m", 1]], "returns": [["c", 1]]},
        ],
    )
    assert _as_int(res, "c") == 0b11


def test_out_of_range_result_bit_fails_fast() -> None:
    """A Result bit index that exceeds the 64-bit backing type errors, not panics/drops."""
    with pytest.raises(RuntimeError, match=r"does not fit|backing type|out of range"):
        _run(
            [
                {"data": "cvar_define", "data_type": "u64", "variable": "m", "size": 2},
                {"cop": "=", "returns": [["m", 0]], "args": [1]},
                {"cop": "Result", "args": [["m", 0]], "returns": [["dst", 70]]},
            ],
        )
