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


# ── Measurement ordering: classical ops must read post-measurement values ────


def _qc(name: str, size: int) -> dict:
    return {"data": "qvar_define", "data_type": "qubits", "variable": name, "size": size}


def _cd(name: str, size: int) -> dict:
    return {"data": "cvar_define", "data_type": "u64", "variable": name, "size": size}


def test_measured_whole_register_result() -> None:
    """A Result reading a register measured in the same batch sees the measurement."""
    res = _run(
        [
            _qc("q", 1),
            _cd("m", 1),
            _cd("c", 1),
            {"qop": "X", "args": [["q", 0]]},  # deterministic measurement -> 1
            {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
            {"cop": "Result", "args": ["m"], "returns": ["c"]},
        ],
    )
    assert _as_int(res, "c") == 1


def test_measured_indexed_result() -> None:
    """An indexed Result of a just-measured bit places it correctly (ordering fix)."""
    res = _run(
        [
            _qc("q", 1),
            _cd("m", 1),
            _cd("c", 41),
            {"qop": "X", "args": [["q", 0]]},
            {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
            {"cop": "Result", "args": [["m", 0]], "returns": [["c", 40]]},
        ],
    )
    assert _as_int(res, "c") == (1 << 40)


def test_measured_assignment_reads_measurement() -> None:
    """An assignment reading a just-measured register sees the measurement (ordering)."""
    res = _run(
        [
            _qc("q", 1),
            _cd("m", 1),
            _cd("rr", 8),
            {"qop": "X", "args": [["q", 0]]},
            {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
            {"cop": "=", "args": [{"cop": "+", "args": ["m", 1]}], "returns": ["rr"]},
        ],
    )
    assert _as_int(res, "rr") == 2


def test_conditional_reads_measurement() -> None:
    """An if-block condition reads the just-measured register (ordering fix)."""
    res = _run(
        [
            _qc("q", 2),
            _cd("m", 1),
            _cd("m2", 1),
            {"qop": "X", "args": [["q", 0]]},
            {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
            {
                "block": "if",
                "condition": {"cop": "==", "args": [["m", 0], 1]},
                "true_branch": [{"qop": "X", "args": [["q", 1]]}],
            },
            {"qop": "Measure", "args": [["q", 1]], "returns": [["m2", 0]]},
            {"cop": "Result", "args": ["m2"], "returns": ["out"]},
        ],
    )
    assert _as_int(res, "out") == 1


def test_mixed_whole_and_indexed_result_same_destination() -> None:
    """Whole-register then indexed Result into the same destination both land."""
    res = _run(
        [
            _qc("q", 1),
            _cd("a", 8),
            _cd("m", 1),
            _cd("c", 8),
            {"cop": "=", "args": [5], "returns": ["a"]},
            {"cop": "Result", "args": ["a"], "returns": ["c"]},  # c = 5
            {"qop": "X", "args": [["q", 0]]},
            {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
            {"cop": "Result", "args": [["m", 0]], "returns": [["c", 6]]},  # c |= 1<<6
        ],
    )
    assert _as_int(res, "c") == (5 | (1 << 6))


# ── Nested-block measurement ordering (execution-cursor rework) ──────


def test_measurement_inside_if_branch_is_mapped() -> None:
    """A Measure emitted inside an if-branch maps to its return register."""
    res = _run(
        [
            _qc("q", 2),
            _cd("m", 1),
            _cd("m2", 1),
            {"qop": "X", "args": [["q", 0]]},
            {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
            {
                "block": "if",
                "condition": {"cop": "==", "args": [["m", 0], 1]},
                "true_branch": [
                    {"qop": "X", "args": [["q", 1]]},
                    {"qop": "Measure", "args": [["q", 1]], "returns": [["m2", 0]]},
                ],
            },
            {"cop": "Result", "args": ["m2"], "returns": ["out"]},
        ],
    )
    assert _as_int(res, "out") == 1


def test_result_inside_sequence_reads_measurement() -> None:
    """A Result inside a sequence block runs after the measurement lands."""
    res = _run(
        [
            _qc("q", 1),
            _cd("m", 1),
            _cd("c", 1),
            {"qop": "X", "args": [["q", 0]]},
            {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
            {"block": "sequence", "ops": [{"cop": "Result", "args": ["m"], "returns": ["c"]}]},
        ],
    )
    assert _as_int(res, "c") == 1


def test_branch_local_classical_reads_branch_measurement() -> None:
    """A classical op inside a branch reads a measurement queued in the same branch."""
    res = _run(
        [
            _qc("q", 2),
            _cd("m", 1),
            _cd("m2", 1),
            _cd("cc", 8),
            {"qop": "X", "args": [["q", 0]]},
            {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
            {
                "block": "if",
                "condition": {"cop": "==", "args": [["m", 0], 1]},
                "true_branch": [
                    {"qop": "X", "args": [["q", 1]]},
                    {"qop": "Measure", "args": [["q", 1]], "returns": [["m2", 0]]},
                    {"cop": "=", "args": [{"cop": "+", "args": ["m2", 5]}], "returns": ["cc"]},
                ],
            },
            {"cop": "Result", "args": ["cc"], "returns": ["out"]},
        ],
    )
    assert _as_int(res, "out") == 6


def test_nested_sequence_if_measure() -> None:
    """A block nested inside a block (depth > 1) executes correctly."""
    res = _run(
        [
            _qc("q", 2),
            _cd("m", 1),
            _cd("m2", 1),
            {"qop": "X", "args": [["q", 0]]},
            {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
            {
                "block": "sequence",
                "ops": [
                    {
                        "block": "if",
                        "condition": {"cop": "==", "args": [["m", 0], 1]},
                        "true_branch": [
                            {"qop": "X", "args": [["q", 1]]},
                            {"qop": "Measure", "args": [["q", 1]], "returns": [["m2", 0]]},
                        ],
                    },
                    {"cop": "Result", "args": ["m2"], "returns": ["out"]},
                ],
            },
        ],
    )
    assert _as_int(res, "out") == 1


def test_multi_qubit_measure() -> None:
    """A multi-qubit Measure measures all qubits (not just the first)."""
    res = _run(
        [
            _qc("q", 2),
            _cd("mm", 2),
            {"qop": "X", "args": [["q", 0]]},  # q0 -> 1, q1 -> 0
            {"qop": "Measure", "args": [["q", 0], ["q", 1]], "returns": [["mm", 0], ["mm", 1]]},
            {"cop": "Result", "args": ["mm"], "returns": ["out"]},
        ],
    )
    assert _as_int(res, "out") == 0b01


def test_no_return_measure_keeps_later_outcomes_aligned() -> None:
    """A no-return Measure reserves an outcome slot so later measures stay aligned.

    q0=1 (via X) is measured without a return; q1=0 is measured into m2. The
    outcomes must map by measured-qubit order, so m2 must see q1's 0, not q0's 1.
    """
    res = _run(
        [
            _qc("q", 2),
            _cd("m2", 1),
            {"qop": "X", "args": [["q", 0]]},  # q0 -> 1, q1 -> 0
            {"qop": "Measure", "args": [["q", 0]]},  # no returns
            {"qop": "Measure", "args": [["q", 1]], "returns": [["m2", 0]]},
            {"cop": "Result", "args": ["m2"], "returns": ["out"]},
        ],
    )
    assert _as_int(res, "out") == 0


def test_unknown_block_type_fails_fast() -> None:
    """An unrecognized block type is a hard error, not a silent skip."""
    with pytest.raises(Exception, match="Unknown block"):
        _run(
            [
                _qc("q", 1),
                {"block": "not_a_real_block", "ops": [{"qop": "X", "args": [["q", 0]]}]},
            ],
        )
