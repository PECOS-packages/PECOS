# Copyright 2024 The PECOS developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Tests to ensure the sequence of operations are as expected."""

from typing import Any

import pytest
from pecos.classical_interpreters.phir_classical_interpreter import (
    PhirClassicalInterpreter,
)
from pecos.reps.pyphir.op_types import COp, EMOp, FFCall, MOp, Op, QOp, SOp


def get_seq(program: dict[str, Any]) -> list[list[tuple[Any, ...]]]:
    """Get the sequences of operations produced by using the PHIR interpreter."""
    interp = PhirClassicalInterpreter()
    interp.init(program)

    ops_seq = []
    for buffered_ops in interp.execute(interp.program.ops):
        subseq = []

        # Create a simple identifier for operations
        for op in buffered_ops:
            op_ident = [op.name]
            # print("angles", op.angles)
            if op.angles is not None:
                op_ident.append(op.angles)
            if hasattr(op, "args") and op.args is not None:
                op_ident.append(op.args)
            if hasattr(op, "returns") and op.returns is not None:
                op_ident.append(op.returns)
            subseq.append(tuple(op_ident))
        ops_seq.append(subseq)
    return ops_seq


def test_seq() -> None:
    """Test that the expected sequence of operations is returned with a program using Sequence Blocks."""
    program = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "metadata": {"source": "pytket-phir v0.2.0", "strict_parallelism": "true"},
        "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
            {"data": "cvar_define", "data_type": "u32", "variable": "m", "size": 2},
            {"qop": "RZ", "angles": [[1.0], "pi"], "args": [["q", 0], ["q", 1]]},
            {
                "block": "sequence",
                "ops": [
                    {"qop": "R1XY", "angles": [[0.5, 0.5], "pi"], "args": [["q", 0]]},
                    {
                        "qop": "Measure",
                        "args": [["q", 0], ["q", 1]],
                        "returns": [["m", 0], ["m", 1]],
                    },
                    {"qop": "R1XY", "angles": [[0.5, 0.5], "pi"], "args": [["q", 0]]},
                ],
            },
            {
                "qop": "Measure",
                "args": [["q", 0], ["q", 1]],
                "returns": [["m", 0], ["m", 1]],
            },
            {"qop": "X", "args": [["q", 0]]},
        ],
    }

    assert get_seq(program) == [
        [
            ("RZ", (3.141592653589793,), [0, 1]),
            ("R1XY", (1.5707963267948966, 1.5707963267948966), [0]),
            ("Measure", [0, 1], [["m", 0], ["m", 1]]),
        ],
        [
            ("R1XY", (1.5707963267948966, 1.5707963267948966), [0]),
            ("Measure", [0, 1], [["m", 0], ["m", 1]]),
        ],
        [
            ("X", [0]),
        ],
    ]


def test_qparallel() -> None:
    """Test that the expected sequence of operations is returned with a program using Sequence Blocks."""
    program = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "metadata": {"source": "pytket-phir v0.2.0", "strict_parallelism": "true"},
        "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
            {"data": "cvar_define", "data_type": "u32", "variable": "m", "size": 2},
            {"qop": "RZ", "angles": [[1.0], "pi"], "args": [["q", 0], ["q", 1]]},
            {
                "block": "qparallel",
                "ops": [
                    {"qop": "R1XY", "angles": [[0.5, 0.5], "pi"], "args": [["q", 0]]},
                    {
                        "qop": "Measure",
                        "args": [["q", 0], ["q", 1]],
                        "returns": [["m", 0], ["m", 1]],
                    },
                    {"qop": "R1XY", "angles": [[0.5, 0.5], "pi"], "args": [["q", 0]]},
                ],
            },
            {
                "qop": "Measure",
                "args": [["q", 0], ["q", 1]],
                "returns": [["m", 0], ["m", 1]],
            },
            {"qop": "X", "args": [["q", 0]]},
        ],
    }

    assert get_seq(program) == [
        [
            ("RZ", (3.141592653589793,), [0, 1]),
            ("R1XY", (1.5707963267948966, 1.5707963267948966), [0]),
            ("Measure", [0, 1], [["m", 0], ["m", 1]]),
        ],
        [
            ("R1XY", (1.5707963267948966, 1.5707963267948966), [0]),
            ("Measure", [0, 1], [["m", 0], ["m", 1]]),
        ],
        [
            ("X", [0]),
        ],
    ]


def test_if_true() -> None:
    """Test PHIR conditional block execution when condition is true."""
    program = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "metadata": {"source": "pytket-phir v0.2.0", "strict_parallelism": "true"},
        "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
            {"data": "cvar_define", "data_type": "u32", "variable": "v", "size": 1},
            {"data": "cvar_define", "data_type": "u32", "variable": "m", "size": 2},
            {"cop": "=", "returns": [["v", 0]], "args": [1]},
            {"qop": "H", "angles": None, "args": [["q", 0]]},
            {"qop": "CX", "angles": None, "args": [[["q", 0], ["q", 1]]]},
            {"qop": "Measure", "returns": [["m", 1]], "args": [["q", 1]]},
            {
                "block": "if",
                "condition": {"cop": "==", "args": [["v", 0], 1]},
                "true_branch": [
                    {"qop": "X", "angles": None, "args": [["q", 0]]},
                ],
                "false_branch": [
                    {"qop": "Y", "angles": None, "args": [["q", 0]]},
                ],
            },
            {"qop": "Measure", "returns": [["m", 0]], "args": [["q", 0]]},
        ],
    }

    assert get_seq(program) == [
        [
            ("H", [0]),
            ("CX", [[0, 1]]),
            ("Measure", [1], [["m", 1]]),
        ],
        [
            ("X", [0]),
            (
                "Measure",
                [0],
                [["m", 0]],
            ),
        ],
    ]


def test_if_false() -> None:
    """Test PHIR conditional block execution when condition is false."""
    program = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "metadata": {"source": "pytket-phir v0.2.0", "strict_parallelism": "true"},
        "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
            {"data": "cvar_define", "data_type": "u32", "variable": "v", "size": 1},
            {"data": "cvar_define", "data_type": "u32", "variable": "m", "size": 2},
            {"cop": "=", "returns": [["v", 0]], "args": [1]},
            {"qop": "H", "angles": None, "args": [["q", 0]]},
            {"qop": "CX", "angles": None, "args": [[["q", 0], ["q", 1]]]},
            {"qop": "Measure", "returns": [["m", 1]], "args": [["q", 1]]},
            {
                "block": "if",
                "condition": {"cop": "==", "args": [["v", 0], 0]},
                "true_branch": [
                    {"qop": "X", "angles": None, "args": [["q", 0]]},
                ],
                "false_branch": [
                    {"qop": "Y", "angles": None, "args": [["q", 0]]},
                ],
            },
            {"qop": "Measure", "returns": [["m", 0]], "args": [["q", 0]]},
        ],
    }

    assert get_seq(program) == [
        [
            ("H", [0]),
            ("CX", [[0, 1]]),
            ("Measure", [1], [["m", 1]]),
        ],
        [
            ("Y", [0]),
            (
                "Measure",
                [0],
                [["m", 0]],
            ),
        ],
    ]


def test_if_no_false() -> None:
    """Test PHIR conditional block with no false branch."""
    program = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "metadata": {"source": "pytket-phir v0.2.0", "strict_parallelism": "true"},
        "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
            {"data": "cvar_define", "data_type": "u32", "variable": "v", "size": 1},
            {"data": "cvar_define", "data_type": "u32", "variable": "m", "size": 2},
            {"cop": "=", "returns": [["v", 0]], "args": [1]},
            {"qop": "H", "angles": None, "args": [["q", 0]]},
            {"qop": "CX", "angles": None, "args": [[["q", 0], ["q", 1]]]},
            {"qop": "Measure", "returns": [["m", 1]], "args": [["q", 1]]},
            {
                "block": "if",
                "condition": {"cop": "==", "args": [["v", 0], 0]},
                "true_branch": [
                    {"qop": "X", "angles": None, "args": [["q", 0]]},
                ],
            },
            {"qop": "Measure", "returns": [["m", 0]], "args": [["q", 0]]},
        ],
    }

    assert get_seq(program) == [
        [
            ("H", [0]),
            ("CX", [[0, 1]]),
            ("Measure", [1], [["m", 1]]),
        ],
        [
            (
                "Measure",
                [0],
                [["m", 0]],
            ),
        ],
    ]


class FalseyDict(dict):
    """Keep metadata entries even when a valid dict reports false."""

    def __bool__(self) -> bool:
        """Report false regardless of contents."""
        return False


@pytest.mark.parametrize("op_type", [Op, QOp, MOp, COp, FFCall, EMOp, SOp])
@pytest.mark.parametrize("kwargs", [{}, {"metadata": None}])
def test_operation_metadata_defaults(op_type: type, kwargs: dict) -> None:
    """Omitted and explicit None metadata produce independent empty dictionaries."""
    first = op_type("H", [0], **kwargs)
    second = op_type("H", [0], **kwargs)
    assert isinstance(first.metadata, dict)
    assert first.metadata == {}
    assert second.metadata == {}
    first.metadata["angle"] = 0.5
    assert second.metadata == {}


@pytest.mark.parametrize("op_type", [Op, QOp, MOp, COp, FFCall, EMOp, SOp])
@pytest.mark.parametrize("dict_type", [dict, FalseyDict])
def test_operation_metadata_identity(op_type: type, dict_type: type) -> None:
    """Explicit metadata retains identity and entries, including falsey subclasses."""
    metadata = dict_type(angle=0.5, bitflips=[1])
    op = op_type("H", [0], metadata=metadata)
    assert op.metadata == {"angle": 0.5, "bitflips": [1]}
    assert op.metadata is metadata


@pytest.mark.parametrize("op_type", [Op, QOp, MOp, COp, FFCall, EMOp, SOp])
@pytest.mark.parametrize("metadata", [False, 0, [], "", "invalid", [1]])
def test_operation_metadata_rejects_non_dict(op_type: type, metadata: object) -> None:
    """Reject non-dicts at construction and name the received type."""
    with pytest.raises(TypeError, match=type(metadata).__name__):
        op_type("H", [0], metadata=metadata)
