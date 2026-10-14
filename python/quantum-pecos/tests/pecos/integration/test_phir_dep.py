# Copyright 2023 The PECOS developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Test the use of the external phir package for validating and using PHIR."""

import json
from pathlib import Path

import pytest
from pecos.typing import PhirModel

this_dir = Path(__file__).parent


def test_spec_example() -> None:
    """Test PHIR specification example for dependency validation."""
    # From https://github.com/Quantinuum/phir/blob/main/phir_spec_qasm.md#overall-phir-example-with-quantinuums-extended-openqasm-20
    data = json.load(Path.open(this_dir / "phir/spec_example.phir.json"))

    PhirModel.model_validate(data)


def test_pecos_result_cop_top_level() -> None:
    """PECOS Result cop validates at the top level of a PHIR program.

    The upstream ``phir.model.PHIRModel`` rejects Result because it is a
    PECOS extension, not part of the spec. ``pecos.typing.PhirModel``
    subclasses the upstream model to add Result support.
    """
    data = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "cvar_define", "data_type": "u32", "variable": "m", "size": 1},
            {"data": "cvar_define", "data_type": "u32", "variable": "c", "size": 1},
            {"cop": "Result", "args": ["m"], "returns": ["c"]},
        ],
    }

    PhirModel.model_validate(data)


def test_pecos_result_cop_inside_seqblock() -> None:
    """Result cop validates when nested inside a SeqBlock."""
    data = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "cvar_define", "data_type": "u32", "variable": "m", "size": 1},
            {"data": "cvar_define", "data_type": "u32", "variable": "c", "size": 1},
            {
                "block": "sequence",
                "ops": [{"cop": "Result", "args": ["m"], "returns": ["c"]}],
            },
        ],
    }

    PhirModel.model_validate(data)


@pytest.mark.parametrize(
    ("dtype", "size"),
    [
        ("i8", 8),
        ("u8", 8),
        ("i16", 16),
        ("u16", 16),
        ("i32", 32),
        ("u32", 32),
        ("i64", 64),
        ("u64", 64),
    ],
)
def test_pecos_cvar_define_small_dtypes(dtype: str, size: int) -> None:
    """PECOS extends CVarDefine to support 8-bit and 16-bit integer dtypes.

    The upstream ``phir.model.CVarDefine`` only permits ``i32``, ``i64``,
    ``u32``, ``u64``. PECOS programs use 8/16-bit dtypes too.
    """
    data = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [{"data": "cvar_define", "data_type": dtype, "variable": "v", "size": size}],
    }

    PhirModel.model_validate(data)


def test_cvar_define_size_exceeding_dtype_rejected() -> None:
    """Extension remains strict: size must fit in the declared dtype."""
    from pydantic import ValidationError

    for dtype, bad_size in [("i8", 16), ("u8", 9), ("i16", 32), ("u16", 17)]:
        data = {
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "ops": [{"data": "cvar_define", "data_type": dtype, "variable": "v", "size": bad_size}],
        }
        with pytest.raises(ValidationError):
            PhirModel.model_validate(data)


def test_malformed_result_cop_still_rejected() -> None:
    """Extension remains strict: Result cop missing required fields is rejected."""
    from pydantic import ValidationError

    data = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [{"cop": "Result"}],  # missing args and returns
    }

    with pytest.raises(ValidationError):
        PhirModel.model_validate(data)


@pytest.mark.parametrize("unit", ["rad", "pi"])
@pytest.mark.parametrize("block", [None, "sequence", "qparallel", "if"])
def test_pecos_rxyxy2q_schema(unit: str, block: str | None) -> None:
    """The PECOS extension validates at top level and inside every block kind."""
    gate = {"qop": "RXYXY2Q", "angles": [[-0.73, 0.41], unit], "args": [[["q", 2], ["q", 0]]]}
    operation = gate
    if block == "if":
        operation = {"block": block, "condition": {"cop": "==", "args": [0, 0]}, "true_branch": [gate]}
    elif block is not None:
        operation = {"block": block, "ops": [gate]}
    data = {"format": "PHIR/JSON", "version": "0.1.0", "ops": [operation]}
    model = PhirModel.model_validate(data)
    validated = model.model_dump(mode="json", exclude_none=True)["ops"][0]
    if block is not None:
        validated = validated["true_branch" if block == "if" else "ops"][0]
    assert validated["qop"] == gate["qop"]
    assert validated["angles"] == gate["angles"]
    assert validated["args"] == gate["args"]


@pytest.mark.parametrize(
    "changes",
    [
        {"angles": None},
        {"angles": [[], "rad"]},
        {"angles": [[-0.73], "rad"]},
        {"angles": [[-0.73, 0.41, 0.2], "rad"]},
        {"angles": [[-0.73, 0.41], "invalid"]},
        {"args": []},
        {"args": [[["q", 2]]]},
        {"args": [[["q", 2], ["q", 0], ["q", 1]]]},
        {"qop": "unknown"},
        {"qop": "RZZ"},
        {"qop": "CX"},
    ],
)
def test_pecos_rxyxy2q_schema_stays_strict(changes: dict) -> None:
    """The extension rejects malformed payloads without relaxing upstream gates."""
    from pydantic import ValidationError

    gate = {"qop": "RXYXY2Q", "angles": [[-0.73, 0.41], "rad"], "args": [[["q", 2], ["q", 0]]]}
    gate.update(changes)
    with pytest.raises(ValidationError):
        PhirModel.model_validate({"format": "PHIR/JSON", "version": "0.1.0", "ops": [gate]})


@pytest.mark.parametrize("test_mode", [False, True])
def test_rxyxy2q_public_python_bridge(test_mode: bool) -> None:
    """The public bridge accepts the schema extension and preserves its payload."""
    import math

    import pecos_rslib

    data = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 3},
            {"qop": "RXYXY2Q", "angles": [[-0.73, 0.41], "rad"], "args": [[["q", 2], ["q", 0]]]},
        ],
    }
    if test_mode:
        data["ops"].insert(1, {"data": "cvar_define", "data_type": "u32", "variable": "m", "size": 3})
        data["ops"].append({"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]})
    commands = pecos_rslib.PhirJsonEngine(json.dumps(data)).process_program()
    assert len(commands) == (2 if test_mode else 1)
    assert commands[0]["gate_type"] == "RXYXY2Q"
    assert commands[0]["qubits"] == [2, 0]
    theta, phi = commands[0]["params"]["angles"]
    assert math.remainder(theta, math.tau) == pytest.approx(-0.73)
    assert phi == pytest.approx(0.41)
