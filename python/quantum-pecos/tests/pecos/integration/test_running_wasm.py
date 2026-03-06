"""Integration tests for WASM value passing through the quantum-classical pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pecos import HybridEngine, QuantumCircuit
from pecos.simulators import SparseSim

THIS_DIR = Path(__file__).parent


def _make_wat_circuit(cvar_spec: dict[str, int], gates: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a QuantumCircuit dict with the given cvars and gates."""
    return {
        "prog_type": "PECOS.QuantumCircuit",
        "prog_metadata": {
            "cvar_spec": cvar_spec,
            "cvar_spec_type": {},
            "num_qubits": 1,
        },
        "gates": [
            {"sym": "init |0>", "qubits": [0], "metadata": {"cond": None, "start_init": True}},
            *gates,
        ],
    }


def _assign(var: str, value: int) -> dict[str, Any]:
    """Create a classical assignment gate: var = value."""
    return {"sym": "cop", "qubits": [], "metadata": {"expr": {"t": var, "a": value, "op": "="}, "cond": None}}


def _wasm_call(func: str, assign_vars: list[str], args: list[str]) -> dict[str, Any]:
    """Create a WASM function call gate."""
    return {
        "sym": "cop",
        "qubits": [],
        "metadata": {
            "cop_type": "CFunc",
            "wrapper": "WASM",
            "func": func,
            "assign_vars": assign_vars,
            "args": args,
            "cond": None,
        },
    }


def _export(var: str) -> dict[str, Any]:
    """Create an export gate."""
    return {"sym": "cop", "qubits": [], "metadata": {"cop_type": "ExportCVar", "export": var, "cond": None}}


def _run_wat_circuit(
    cvar_spec: dict[str, int],
    gates: list[dict[str, Any]],
    wat_file: str = "test_values.wat",
) -> dict[str, Any]:
    """Build and run a circuit backed by a WAT file, return shot output."""
    wat_path = THIS_DIR / "wat" / wat_file
    wat_bytes = Path.read_bytes(wat_path)

    qc_dict = _make_wat_circuit(cvar_spec, gates)
    qc = QuantumCircuit.from_json_str(json.dumps(qc_dict))
    qc.metadata["ccop"] = wat_bytes
    qc.metadata["ccop_type"] = "wasmtime"

    state = SparseSim(num_qubits=1)
    runner = HybridEngine()
    shot_output, _ = runner.run(state, qc, shot_id=0)
    return shot_output


def test_wat_1bit_identity() -> None:
    """Verify a 1-bit register with value 1 round-trips through WASM as 1, not -1."""
    output = _run_wat_circuit(
        cvar_spec={"x": 1, "y": 1},
        gates=[
            _assign("x", 1),
            _wasm_call("identity", ["y"], ["x"]),
            _export("x"),
            _export("y"),
        ],
    )
    assert int(output["x"]) == 1
    assert int(output["y"]) == 1


def test_wat_add() -> None:
    """Verify two 8-bit values pass through a WASM add function correctly."""
    output = _run_wat_circuit(
        cvar_spec={"a": 8, "b": 8, "result": 8},
        gates=[
            _assign("a", 100),
            _assign("b", 50),
            _wasm_call("add", ["result"], ["a", "b"]),
            _export("a"),
            _export("b"),
            _export("result"),
        ],
    )
    assert int(output["a"]) == 100
    assert int(output["b"]) == 50
    assert int(output["result"]) == 150


def test_wat_identity_preserves_value() -> None:
    """Verify an 8-bit value round-trips through WASM identity unchanged."""
    output = _run_wat_circuit(
        cvar_spec={"v": 8, "out": 8},
        gates=[
            _assign("v", 200),
            _wasm_call("identity", ["out"], ["v"]),
            _export("v"),
            _export("out"),
        ],
    )
    assert int(output["v"]) == 200
    assert int(output["out"]) == 200


def test_wat_measurement_through_wasm() -> None:
    """Verify a measurement result passes through WASM identity correctly."""
    output = _run_wat_circuit(
        cvar_spec={"m": 1, "out": 1},
        gates=[
            {"sym": "X", "qubits": [0], "metadata": {"cond": None}},
            {
                "sym": "measure Z",
                "qubits": [0],
                "metadata": {"cond": None, "var_output": {"0": ["m", 0]}, "mid_circuit": True},
            },
            {"sym": "cop", "qubits": [0], "metadata": {"cop_type": "Idle", "cond": None, "active_sym": "MeasureZ"}},
            _wasm_call("identity", ["out"], ["m"]),
            _export("m"),
            _export("out"),
        ],
    )
    assert int(output["m"]) == 1
    assert int(output["out"]) == 1


def test_wat_add_with_measurement() -> None:
    """Verify a measurement result can be added with another register via WASM."""
    output = _run_wat_circuit(
        cvar_spec={"m": 1, "offset": 8, "result": 8},
        gates=[
            _assign("offset", 41),
            {"sym": "X", "qubits": [0], "metadata": {"cond": None}},
            {
                "sym": "measure Z",
                "qubits": [0],
                "metadata": {"cond": None, "var_output": {"0": ["m", 0]}, "mid_circuit": True},
            },
            {"sym": "cop", "qubits": [0], "metadata": {"cop_type": "Idle", "cond": None, "active_sym": "MeasureZ"}},
            _wasm_call("add", ["result"], ["m", "offset"]),
            _export("m"),
            _export("result"),
        ],
    )
    assert int(output["m"]) == 1
    assert int(output["result"]) == 42  # 1 + 41
