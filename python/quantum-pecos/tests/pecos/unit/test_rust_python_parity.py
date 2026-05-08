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

"""Tests that the Rust and Python PhirClassicalInterpreter implementations produce identical results.

These tests run the same PHIR programs through both interpreters and compare outputs.
Any difference is a bug in one or the other.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pecos.classical_interpreters.phir_classical_interpreter import PhirClassicalInterpreter
from pecos.engines.hybrid_engine import HybridEngine
from pecos_rslib import RustPhirClassicalInterpreter

if TYPE_CHECKING:
    from pecos.protocols import ForeignObjectProtocol


def run_both(
    phir: dict | str,
    *,
    shots: int = 1,
    seed: int = 42,
    qsim: str | None = None,
    foreign_object: ForeignObjectProtocol | None = None,
    return_int: bool = False,
) -> tuple[dict, dict]:
    """Run the same program through both interpreters and return both results."""
    kw = {}
    if qsim:
        kw["qsim"] = qsim

    py_i = PhirClassicalInterpreter()
    py_r = HybridEngine(cinterp=py_i, **kw).run(
        phir,
        foreign_object=foreign_object,
        shots=shots,
        seed=seed,
        return_int=return_int,
    )

    rs_i = RustPhirClassicalInterpreter()
    rs_r = HybridEngine(cinterp=rs_i, **kw).run(
        phir,
        foreign_object=foreign_object,
        shots=shots,
        seed=seed,
        return_int=return_int,
    )

    return py_r, rs_r


# ── Integration PHIR files ───────────────────────────────────────────

PHIR_DIR = Path(__file__).parent.parent / "integration" / "phir"


@pytest.mark.parametrize(
    "filename",
    [
        "bell_qparallel.phir.json",
        "bell_qparallel_cliff.phir.json",
        "bell_qparallel_cliff_barrier.phir.json",
        "bell_qparallel_cliff_ifbarrier.phir.json",
        "classical_00_11.phir.json",
        "qparallel.phir.json",
        "recording_random_meas.phir.json",
        "example1_no_wasm.phir.json",
    ],
)
def test_phir_integration_files(filename: str) -> None:
    """Test that integration PHIR files produce identical results."""
    phir = json.loads((PHIR_DIR / filename).read_text())
    py_r, rs_r = run_both(phir, shots=20, seed=42)
    assert py_r == rs_r, f"Mismatch on {filename}"


# ── WASM foreign function calls ─────────────────────────────────────

WAT_DIR = Path(__file__).parent.parent / "integration" / "wat"


def test_wasm_spec_example() -> None:
    """Test spec_example.phir.json with WASM foreign object."""
    from pecos import WasmForeignObject

    phir = json.loads((PHIR_DIR / "spec_example.phir.json").read_text())
    math_wat = WAT_DIR / "math.wat"

    py_i = PhirClassicalInterpreter()
    py_r = HybridEngine(cinterp=py_i).run(
        phir,
        foreign_object=WasmForeignObject(math_wat),
        shots=20,
        seed=42,
    )

    rs_i = RustPhirClassicalInterpreter()
    rs_r = HybridEngine(cinterp=rs_i).run(
        phir,
        foreign_object=WasmForeignObject(math_wat),
        shots=20,
        seed=42,
    )

    assert py_r == rs_r


# ── Classical operations ─────────────────────────────────────────────


@pytest.mark.parametrize(
    ("cop", "a", "b", "expected"),
    [
        ("+", 100, 23, 123),
        ("-", 100, 23, 77),
        ("*", 7, 6, 42),
        ("/", 42, 7, 6),
        ("%", 17, 5, 2),
        ("&", 0xFF, 0x0F, 0x0F),
        ("|", 0xF0, 0x0F, 0xFF),
        ("^", 0xFF, 0xAA, 0x55),
        (">>", 256, 3, 32),
        ("<<", 1, 10, 1024),
        ("==", 42, 42, 1),
        ("==", 42, 43, 0),
        ("!=", 42, 43, 1),
        ("<", 5, 10, 1),
        (">", 10, 5, 1),
        ("<=", 5, 5, 1),
        (">=", 6, 5, 1),
    ],
)
def test_classical_binary_ops(cop: str, a: int, b: int, expected: int) -> None:
    """Test that all binary classical operations produce identical results."""
    phir = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "cvar_define", "data_type": "i32", "variable": "x", "size": 32},
            {"data": "cvar_define", "data_type": "i32", "variable": "r", "size": 32},
            {"cop": "=", "returns": ["x"], "args": [a]},
            {"cop": "=", "returns": ["r"], "args": [{"cop": cop, "args": ["x", b]}]},
        ],
    }
    py_r, rs_r = run_both(phir, qsim="stabilizer", return_int=True)
    assert py_r == rs_r
    assert int(py_r["r"][0]) == expected, f"{a} {cop} {b}: expected {expected}, got {int(py_r['r'][0])}"


# ── Data types and extreme values ────────────────────────────────────


@pytest.mark.parametrize(
    ("dtype", "size", "val"),
    [
        ("i8", 8, 127),
        ("i8", 8, -128),
        ("u8", 8, 255),
        ("u8", 8, 0),
        ("i16", 16, 32767),
        ("i16", 16, -32768),
        ("u16", 16, 65535),
        ("i32", 32, 2**31 - 1),
        ("i32", 32, -(2**31)),
        ("u32", 32, 2**32 - 1),
        ("i64", 64, 2**63 - 1),
        ("i64", 64, -(2**63)),
        ("u64", 64, 2**64 - 1),
    ],
)
def test_data_types_extreme_values(dtype: str, size: int, val: int) -> None:
    """Test that extreme values for all data types produce identical results."""
    phir = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "cvar_define", "data_type": dtype, "variable": "v", "size": size},
            {"cop": "=", "returns": ["v"], "args": [val]},
        ],
    }
    py_r, rs_r = run_both(phir, qsim="stabilizer")
    assert py_r == rs_r


# ── Size masking ─────────────────────────────────────────────────────


@pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 7, 8, 10, 16])
def test_unsigned_size_masking(size: int) -> None:
    """Test that unsigned values are masked to the declared register size."""
    phir = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "cvar_define", "data_type": "u32", "variable": "c", "size": size},
            {"cop": "=", "returns": ["c"], "args": [2**32 - 1]},
        ],
    }
    py_r, rs_r = run_both(phir, qsim="stabilizer")
    assert py_r == rs_r


# ── Multi-assignment eval order ──────────────────────────────────────


def test_multi_assign_eval_order() -> None:
    """Test that multi-assignment evaluates all args before assigning any."""
    phir = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "cvar_define", "data_type": "i32", "variable": "a", "size": 32},
            {"data": "cvar_define", "data_type": "i32", "variable": "b", "size": 32},
            {"data": "cvar_define", "data_type": "i32", "variable": "c", "size": 32},
            {
                "cop": "=",
                "returns": ["a", "b", "c"],
                "args": [
                    10,
                    {"cop": "+", "args": ["a", 1]},
                    {"cop": "*", "args": ["b", 2]},
                ],
            },
        ],
    }
    py_r, rs_r = run_both(phir, qsim="stabilizer")
    assert py_r == rs_r


# ── Conditional branching ────────────────────────────────────────────


def test_conditional_measurement_feedback() -> None:
    """Test conditional gate based on measurement result."""
    phir = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
            {"data": "cvar_define", "data_type": "u32", "variable": "m", "size": 2},
            {"qop": "H", "args": [["q", 0]]},
            {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
            {
                "block": "if",
                "condition": {"cop": "==", "args": [["m", 0], 1]},
                "true_branch": [{"qop": "X", "args": [["q", 1]]}],
            },
            {"qop": "Measure", "args": [["q", 1]], "returns": [["m", 1]]},
        ],
    }
    py_r, rs_r = run_both(phir, shots=50, seed=42)
    assert py_r == rs_r


def test_nested_conditional() -> None:
    """Test nested if blocks with measurement feedback."""
    phir = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 3},
            {"data": "cvar_define", "data_type": "u32", "variable": "m", "size": 3},
            {"qop": "H", "args": [["q", 0]]},
            {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
            {
                "block": "if",
                "condition": {"cop": "==", "args": [["m", 0], 1]},
                "true_branch": [
                    {"qop": "H", "args": [["q", 1]]},
                    {"qop": "Measure", "args": [["q", 1]], "returns": [["m", 1]]},
                    {
                        "block": "if",
                        "condition": {"cop": "==", "args": [["m", 1], 1]},
                        "true_branch": [{"qop": "X", "args": [["q", 2]]}],
                        "false_branch": [{"qop": "H", "args": [["q", 2]]}],
                    },
                ],
                "false_branch": [{"qop": "X", "args": [["q", 1]]}],
            },
            {"qop": "Measure", "args": [["q", 2]], "returns": [["m", 2]]},
        ],
    }
    py_r, rs_r = run_both(phir, shots=200, seed=42)
    assert py_r == rs_r


# ── Seed determinism ─────────────────────────────────────────────────


@pytest.mark.parametrize("seed", [0, 1, 42, 999, 12345, 2**31 - 1])
def test_seed_determinism(seed: int) -> None:
    """Test that both interpreters produce identical results for various seeds."""
    phir = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
            {"data": "cvar_define", "data_type": "u32", "variable": "m", "size": 2},
            {"qop": "H", "args": [["q", 0]]},
            {"qop": "Measure", "args": [["q", 0], ["q", 1]], "returns": [["m", 0], ["m", 1]]},
        ],
    }
    py_r, rs_r = run_both(phir, shots=100, seed=seed)
    assert py_r == rs_r


# ── return_int mode ──────────────────────────────────────────────────


def test_return_int_types() -> None:
    """Test that return_int=True produces matching typed results."""
    phir = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "cvar_define", "data_type": "i32", "variable": "a", "size": 32},
            {"data": "cvar_define", "data_type": "u32", "variable": "b", "size": 32},
            {"cop": "=", "returns": ["a"], "args": [42]},
            {"cop": "=", "returns": ["b"], "args": [7]},
        ],
    }
    py_r, rs_r = run_both(phir, qsim="stabilizer", return_int=True)
    assert py_r == rs_r


# ── Fuzz testing ─────────────────────────────────────────────────────

FUZZ_DTYPES = ["i32", "u32", "i64", "u64"]
FUZZ_COPS = ["+", "-", "*", "&", "|", "^", ">>", "<<", "==", "!=", "<", ">", "<=", ">="]


def _make_random_classical_program(rng: random.Random) -> dict:
    """Generate a random classical-only PHIR program."""
    nvars = rng.randint(2, 5)
    ops = []
    vars_info = []
    for i in range(nvars):
        dtype = rng.choice(FUZZ_DTYPES)
        tw = int(dtype[1:])
        size = rng.randint(1, tw)
        name = f"v{i}"
        ops.append({"data": "cvar_define", "data_type": dtype, "variable": name, "size": size})
        vars_info.append((name, dtype, size))

    for _ in range(rng.randint(3, 10)):
        target = rng.choice(vars_info)
        tname, tdtype, tsize = target
        tw = int(tdtype[1:])

        op_kind = rng.random()
        if op_kind < 0.3:
            if tdtype.startswith("i"):
                val = rng.randint(-(2 ** (tw - 1)), 2 ** (tw - 1) - 1)
            else:
                val = rng.randint(0, 2**tw - 1)
            # Clamp to avoid values that exceed i64 range in JSON
            val = max(-(2**63), min(2**63 - 1, val))
            ops.append({"cop": "=", "returns": [tname], "args": [val]})
        elif op_kind < 0.6:
            src = rng.choice(vars_info)
            cop = rng.choice(FUZZ_COPS)
            rhs = rng.randint(0, 15)
            if cop in ("/", "%"):
                rhs = max(rhs, 1)
            if cop in (">>", "<<"):
                rhs = rhs % 16
            ops.append({"cop": "=", "returns": [tname], "args": [{"cop": cop, "args": [src[0], rhs]}]})
        elif op_kind < 0.8:
            bi = rng.randint(0, min(tsize - 1, 7))
            bv = rng.randint(0, 1)
            ops.append({"cop": "=", "returns": [[tname, bi]], "args": [bv]})
        else:
            targets = rng.sample(vars_info, min(len(vars_info), rng.randint(2, 3)))
            returns = [t[0] for t in targets]
            args = []
            for t in targets:
                ttw = int(t[1][1:])
                if t[1].startswith("i"):
                    args.append(rng.randint(max(-(2 ** (ttw - 1)), -(2**63)), min(2 ** (ttw - 1) - 1, 2**63 - 1)))
                else:
                    args.append(rng.randint(0, min(2**ttw - 1, 2**63 - 1)))
            ops.append({"cop": "=", "returns": returns, "args": args})

    return {"format": "PHIR/JSON", "version": "0.1.0", "ops": ops}


def _make_random_quantum_program(rng: random.Random) -> dict:
    """Generate a random quantum+classical PHIR program."""
    nq = rng.randint(1, 4)
    ops = [
        {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": nq},
        {"data": "cvar_define", "data_type": "u32", "variable": "m", "size": nq},
    ]

    for _ in range(rng.randint(1, 6)):
        q = rng.randint(0, nq - 1)
        gate = rng.choice(["H", "X", "Y", "Z"])
        ops.append({"qop": gate, "args": [["q", q]]})

    if nq >= 2 and rng.random() < 0.5:
        q1, q2 = rng.sample(range(nq), 2)
        ops.append({"qop": "CX", "args": [[["q", q1], ["q", q2]]]})

    ops.append(
        {
            "qop": "Measure",
            "args": [["q", i] for i in range(nq)],
            "returns": [["m", i] for i in range(nq)],
        },
    )

    if rng.random() < 0.5:
        q = rng.randint(0, nq - 1)
        b = rng.randint(0, nq - 1)
        gate = rng.choice(["X", "Y", "Z", "H"])
        ops.append(
            {
                "block": "if",
                "condition": {"cop": "==", "args": [["m", b], 1]},
                "true_branch": [{"qop": gate, "args": [["q", q]]}],
            },
        )
        ops.append(
            {
                "qop": "Measure",
                "args": [["q", i] for i in range(nq)],
                "returns": [["m", i] for i in range(nq)],
            },
        )

    return {"format": "PHIR/JSON", "version": "0.1.0", "ops": ops}


def test_fuzz_classical_programs() -> None:
    """Fuzz test: random classical programs must produce identical results."""
    rng = random.Random(2026)
    for i in range(200):
        phir = _make_random_classical_program(rng)
        py_r, rs_r = run_both(phir, seed=i, qsim="stabilizer")
        assert py_r == rs_r, f"Classical fuzz program {i} produced different results"


def test_fuzz_quantum_programs() -> None:
    """Fuzz test: random quantum programs must produce identical results."""
    rng = random.Random(2026)
    identical = 0

    for i in range(200):
        phir = _make_random_quantum_program(rng)
        py_r, rs_r = run_both(phir, shots=20, seed=i + 10000)
        assert py_r == rs_r, f"Quantum fuzz program {i} produced different results"
        identical += 1

    assert identical == 200


# ── Targeted classical edge cases ──────────────────────────────────
#
# These test the specific behaviors we changed in the interpreters.
# Each test checks BOTH parity (Python == Rust) AND the expected value
# (so we catch the case where both are wrong the same way).


def _run_classical(phir: dict) -> tuple[dict, dict]:
    """Run a classical-only program through both interpreters.

    Returns dicts mapping variable name -> int value (single shot).
    """
    py_r, rs_r = run_both(phir, qsim="stabilizer", return_int=True)
    # Results are {var: [shot0_val, shot1_val, ...]}. Extract single shot.
    py_vals = {k: int(v[0]) for k, v in py_r.items()}
    rs_vals = {k: int(v[0]) for k, v in rs_r.items()}
    return py_vals, rs_vals


def _make_classical_program(var_defs: list[tuple[str, str, int]], ops: list[dict]) -> dict:
    """Build a classical-only PHIR program.

    var_defs: list of (name, dtype, size) tuples
    ops: list of cop operations
    """
    phir_ops = []
    for name, dtype, size in var_defs:
        phir_ops.append({"data": "cvar_define", "data_type": dtype, "variable": name, "size": size})
    phir_ops.extend(ops)
    return {"format": "PHIR/JSON", "version": "0.1.0", "ops": phir_ops}


# ── Signed narrow register masking ──────────────────────────────────


@pytest.mark.parametrize(
    ("size", "val", "expected"),
    [
        (2, 3, 3),  # 3 fits in 2 bits -> 3
        (2, 5, 1),  # 5 = 0b101, masked to 2 bits -> 1
        (2, 4, 0),  # 4 = 0b100, masked to 2 bits -> 0
        (2, 7, 3),  # 7 = 0b111, masked to 2 bits -> 3
        (3, 10, 2),  # 10 = 0b1010, masked to 3 bits -> 2
        (4, 255, 15),  # 255 = 0xFF, masked to 4 bits -> 15
        (1, 1, 1),  # 1 fits in 1 bit -> 1
        (1, 2, 0),  # 2 = 0b10, masked to 1 bit -> 0
        (1, 3, 1),  # 3 = 0b11, masked to 1 bit -> 1
    ],
)
def test_signed_narrow_register_masking(size: int, val: int, expected: int) -> None:
    """i64 with size < 64 should mask to size bits on assignment."""
    phir = _make_classical_program(
        [("v", "i64", size)],
        [{"cop": "=", "returns": ["v"], "args": [val]}],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r, f"Parity failure: py={py_r}, rs={rs_r}"
    assert int(py_r["v"]) == expected, f"i64 size={size}, assign {val}: expected {expected}, got {int(py_r['v'])}"


@pytest.mark.parametrize(
    ("dtype", "size", "val", "expected"),
    [
        ("i32", 2, 5, 1),  # i32 size=2, assign 5 -> mask to 2 bits = 1
        ("i32", 4, 255, 15),  # i32 size=4, assign 255 -> mask to 4 bits = 15
        ("u32", 2, 5, 1),  # u32 size=2, assign 5 -> mask to 2 bits = 1
        ("u64", 3, 10, 2),  # u64 size=3, assign 10 -> mask to 3 bits = 2
        ("i64", 4, -1, 15),  # -1 in twos complement, masked to 4 bits = 0b1111 = 15
    ],
)
def test_narrow_register_masking_all_types(dtype: str, size: int, val: int, expected: int) -> None:
    """All integer types should mask to size bits on assignment."""
    phir = _make_classical_program(
        [("v", dtype, size)],
        [{"cop": "=", "returns": ["v"], "args": [val]}],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r, f"Parity failure: py={py_r}, rs={rs_r}"
    assert int(py_r["v"]) == expected, f"{dtype} size={size}, assign {val}: expected {expected}, got {int(py_r['v'])}"


# ── Unary NOT (~) ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("size", "val", "expected"),
    [
        # ~val at full 64-bit width, then masked to size bits
        (4, 5, 10),  # ~5 = ...1111010, masked to 4 bits = 0b1010 = 10
        (4, 0, 15),  # ~0 = all ones, masked to 4 bits = 15
        (4, 15, 0),  # ~15 = ...10000, masked to 4 bits = 0
        (8, 0xAA, 0x55),  # ~0xAA = 0x55...55, masked to 8 bits = 0x55
        (2, 0, 3),  # ~0 -> all ones, masked to 2 bits = 3
        (1, 0, 1),  # ~0 -> all ones, masked to 1 bit = 1
        (1, 1, 0),  # ~1 -> ...1110, masked to 1 bit = 0
    ],
)
def test_not_narrow_register(size: int, val: int, expected: int) -> None:
    """~ evaluates at full width, result masked to register size."""
    phir = _make_classical_program(
        [("v", "i64", size)],
        [
            {"cop": "=", "returns": ["v"], "args": [val]},
            {"cop": "=", "returns": ["v"], "args": [{"cop": "~", "args": ["v"]}]},
        ],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r, f"Parity failure: py={py_r}, rs={rs_r}"
    assert int(py_r["v"]) == expected, f"~{val} in i64 size={size}: expected {expected}, got {int(py_r['v'])}"


# ── Shifts ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("size", "val", "shift", "expected"),
    [
        (4, 1, 10, 0),  # 1 << 10 = 1024, masked to 4 bits = 0
        (4, 1, 3, 8),  # 1 << 3 = 8, masked to 4 bits = 8
        (4, 1, 4, 0),  # 1 << 4 = 16, masked to 4 bits = 0
        (8, 1, 7, 128),  # 1 << 7 = 128, fits in 8 bits
        (8, 1, 8, 0),  # 1 << 8 = 256, masked to 8 bits = 0
        (2, 1, 1, 2),  # 1 << 1 = 2, fits in 2 bits
        (2, 1, 2, 0),  # 1 << 2 = 4, masked to 2 bits = 0
    ],
)
def test_left_shift_with_masking(size: int, val: int, shift: int, expected: int) -> None:
    """Left shift evaluates at full width, result masked to register size."""
    phir = _make_classical_program(
        [("v", "i64", size)],
        [
            {"cop": "=", "returns": ["v"], "args": [val]},
            {"cop": "=", "returns": ["v"], "args": [{"cop": "<<", "args": ["v", shift]}]},
        ],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r, f"Parity failure: py={py_r}, rs={rs_r}"
    assert int(py_r["v"]) == expected, f"{val} << {shift} in i64 size={size}: expected {expected}, got {int(py_r['v'])}"


@pytest.mark.parametrize(
    ("size", "val", "shift", "expected"),
    [
        (8, 128, 1, 64),  # 128 >> 1 = 64
        (8, 128, 7, 1),  # 128 >> 7 = 1
        (8, 128, 8, 0),  # 128 >> 8 = 0
        (4, 8, 1, 4),  # 8 >> 1 = 4
        (4, 15, 2, 3),  # 15 >> 2 = 3
    ],
)
def test_right_shift_with_masking(size: int, val: int, shift: int, expected: int) -> None:
    """Right shift evaluates at full width, result masked to register size."""
    phir = _make_classical_program(
        [("v", "i64", size)],
        [
            {"cop": "=", "returns": ["v"], "args": [val]},
            {"cop": "=", "returns": ["v"], "args": [{"cop": ">>", "args": ["v", shift]}]},
        ],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r, f"Parity failure: py={py_r}, rs={rs_r}"
    assert int(py_r["v"]) == expected, f"{val} >> {shift} in i64 size={size}: expected {expected}, got {int(py_r['v'])}"


# ── Expression overflow into narrow register ────────────────────────


@pytest.mark.parametrize(
    ("size", "a_val", "b_val", "cop", "expected"),
    [
        # Subtraction underflow: 0 - 1 at full width is huge negative, masked to size bits
        (3, 0, 1, "-", 7),  # 0 - 1 = -1 = ...1111, masked to 3 bits = 7
        (4, 0, 1, "-", 15),  # 0 - 1 = -1, masked to 4 bits = 15
        (8, 0, 1, "-", 255),  # 0 - 1 = -1, masked to 8 bits = 255
        # Addition overflow
        (4, 15, 1, "+", 0),  # 15 + 1 = 16 = 0b10000, masked to 4 bits = 0
        (4, 15, 2, "+", 1),  # 15 + 2 = 17 = 0b10001, masked to 4 bits = 1
        (3, 7, 1, "+", 0),  # 7 + 1 = 8 = 0b1000, masked to 3 bits = 0
        # Multiplication overflow
        (4, 4, 5, "*", 4),  # 4 * 5 = 20 = 0b10100, masked to 4 bits = 4
    ],
)
def test_expression_overflow_narrow(size: int, a_val: int, b_val: int, cop: str, expected: int) -> None:
    """Arithmetic at full width, overflow masked to register size."""
    phir = _make_classical_program(
        [("v", "i64", size)],
        [
            {"cop": "=", "returns": ["v"], "args": [a_val]},
            {"cop": "=", "returns": ["v"], "args": [{"cop": cop, "args": ["v", b_val]}]},
        ],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r, f"Parity failure: py={py_r}, rs={rs_r}"
    assert (
        int(py_r["v"]) == expected
    ), f"{a_val} {cop} {b_val} in i64 size={size}: expected {expected}, got {int(py_r['v'])}"


# ── Division and modulo edge cases ──────────────────────────────────


@pytest.mark.parametrize(
    ("a_val", "b_val", "cop", "expected"),
    [
        (7, 2, "/", 3),  # 7 / 2 = 3 (truncated)
        (7, 2, "%", 1),  # 7 % 2 = 1
        (10, 3, "/", 3),  # 10 / 3 = 3 (truncated)
        (10, 3, "%", 1),  # 10 % 3 = 1
        (1, 2, "/", 0),  # 1 / 2 = 0
        (0, 5, "/", 0),  # 0 / 5 = 0
        (0, 5, "%", 0),  # 0 % 5 = 0
    ],
)
def test_division_modulo_positive(a_val: int, b_val: int, cop: str, expected: int) -> None:
    """Division and modulo with positive values."""
    phir = _make_classical_program(
        [("a", "i64", 32), ("r", "i64", 32)],
        [
            {"cop": "=", "returns": ["a"], "args": [a_val]},
            {"cop": "=", "returns": ["r"], "args": [{"cop": cop, "args": ["a", b_val]}]},
        ],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r, f"Parity failure: py={py_r}, rs={rs_r}"
    assert int(py_r["r"]) == expected, f"{a_val} {cop} {b_val}: expected {expected}, got {int(py_r['r'])}"


@pytest.mark.parametrize("cop", ["/", "%"])
def test_division_by_zero(cop: str) -> None:
    """Division and modulo by zero should raise an error in both interpreters."""
    phir = _make_classical_program(
        [("a", "i64", 32), ("r", "i64", 32)],
        [
            {"cop": "=", "returns": ["a"], "args": [42]},
            {"cop": "=", "returns": ["r"], "args": [{"cop": cop, "args": ["a", 0]}]},
        ],
    )
    with pytest.raises((ZeroDivisionError, RuntimeError)):
        run_both(phir, qsim="stabilizer")


def test_signed_division_min_by_neg_one() -> None:
    """i64::MIN / -1 wraps to i64::MIN in both interpreters.

    This is a fragile parity: Rust uses wrapping_div (returns i64::MIN),
    Python evaluates at arbitrary precision (gives 2**63) then truncates
    on storage (back to i64::MIN). Lock it down with a regression test.
    """
    min_i64 = -(2**63)
    phir = _make_classical_program(
        [("a", "i64", 63), ("r", "i64", 63)],
        [
            {"cop": "=", "returns": ["a"], "args": [min_i64]},
            {"cop": "=", "returns": ["r"], "args": [{"cop": "/", "args": ["a", -1]}]},
        ],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r, f"Parity failure on i64::MIN / -1: py={py_r}, rs={rs_r}"


# ── Nested expressions with narrow registers ────────────────────────


def test_nested_expression_full_width() -> None:
    """Nested expressions should evaluate at full width before storing masked."""
    # (a | b) + c where a=3, b=12, c=1 -> (3|12)+1 = 16, masked to 4 bits = 0
    phir = _make_classical_program(
        [("a", "i64", 4), ("b", "i64", 4), ("c", "i64", 4), ("r", "i64", 4)],
        [
            {"cop": "=", "returns": ["a"], "args": [3]},
            {"cop": "=", "returns": ["b"], "args": [12]},
            {"cop": "=", "returns": ["c"], "args": [1]},
            {
                "cop": "=",
                "returns": ["r"],
                "args": [
                    {
                        "cop": "+",
                        "args": [
                            {"cop": "|", "args": ["a", "b"]},
                            "c",
                        ],
                    },
                ],
            },
        ],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r
    assert int(py_r["r"]) == 0, f"(3|12)+1 masked to 4 bits: expected 0, got {int(py_r['r'])}"


def test_chained_not_narrow() -> None:
    """~~x in narrow register should round-trip through full width."""
    # x = 5 (0101), ~x = ...1010, ~~x = ...0101, masked to 4 bits = 5
    phir = _make_classical_program(
        [("v", "i64", 4)],
        [
            {"cop": "=", "returns": ["v"], "args": [5]},
            {"cop": "=", "returns": ["v"], "args": [{"cop": "~", "args": [{"cop": "~", "args": ["v"]}]}]},
        ],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r
    assert int(py_r["v"]) == 5, f"~~5 in 4-bit register: expected 5, got {int(py_r['v'])}"


# ── Cross-register-size operations ──────────────────────────────────


def test_wide_to_narrow_assignment() -> None:
    """Assigning a wide-register value to a narrow register should mask."""
    phir = _make_classical_program(
        [("wide", "i64", 32), ("narrow", "i64", 4)],
        [
            {"cop": "=", "returns": ["wide"], "args": [255]},
            {"cop": "=", "returns": ["narrow"], "args": ["wide"]},
        ],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r
    assert int(py_r["narrow"]) == 15, f"255 into 4-bit register: expected 15, got {int(py_r['narrow'])}"


def test_narrow_to_wide_assignment() -> None:
    """Assigning a narrow-register value to a wide register should not lose data."""
    phir = _make_classical_program(
        [("narrow", "i64", 4), ("wide", "i64", 32)],
        [
            {"cop": "=", "returns": ["narrow"], "args": [15]},
            {"cop": "=", "returns": ["wide"], "args": ["narrow"]},
        ],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r
    assert int(py_r["wide"]) == 15, f"15 from 4-bit to 32-bit: expected 15, got {int(py_r['wide'])}"


def test_cross_size_expression() -> None:
    """Expression with registers of different sizes should evaluate correctly."""
    # narrow = 3 (4 bits), wide = 100 (32 bits), result = narrow + wide = 103, stored in 8-bit reg
    phir = _make_classical_program(
        [("narrow", "i64", 4), ("wide", "i64", 32), ("result", "i64", 8)],
        [
            {"cop": "=", "returns": ["narrow"], "args": [3]},
            {"cop": "=", "returns": ["wide"], "args": [100]},
            {"cop": "=", "returns": ["result"], "args": [{"cop": "+", "args": ["narrow", "wide"]}]},
        ],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r
    assert int(py_r["result"]) == 103, f"3 + 100 in 8-bit register: expected 103, got {int(py_r['result'])}"


def test_cross_size_expression_overflow() -> None:
    """Cross-size expression that overflows the destination register."""
    # narrow = 15 (4 bits), wide = 250 (32 bits), result = 265, masked to 8 bits = 9
    phir = _make_classical_program(
        [("narrow", "i64", 4), ("wide", "i64", 32), ("result", "i64", 8)],
        [
            {"cop": "=", "returns": ["narrow"], "args": [15]},
            {"cop": "=", "returns": ["wide"], "args": [250]},
            {"cop": "=", "returns": ["result"], "args": [{"cop": "+", "args": ["narrow", "wide"]}]},
        ],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r
    assert int(py_r["result"]) == 9, f"15 + 250 in 8-bit register: expected 9, got {int(py_r['result'])}"


# ── Bit-level operations on narrow registers ────────────────────────


def test_set_bit_in_narrow_register() -> None:
    """Setting individual bits in a narrow register."""
    phir = _make_classical_program(
        [("v", "i64", 4)],
        [
            {"cop": "=", "returns": ["v"], "args": [0]},
            {"cop": "=", "returns": [["v", 0]], "args": [1]},  # set bit 0
            {"cop": "=", "returns": [["v", 2]], "args": [1]},  # set bit 2
            # v should be 0b0101 = 5
        ],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r
    assert int(py_r["v"]) == 5, f"Set bits 0,2 in 4-bit register: expected 5, got {int(py_r['v'])}"


def test_clear_bit_in_narrow_register() -> None:
    """Clearing individual bits in a narrow register."""
    phir = _make_classical_program(
        [("v", "i64", 4)],
        [
            {"cop": "=", "returns": ["v"], "args": [15]},  # all bits set
            {"cop": "=", "returns": [["v", 1]], "args": [0]},  # clear bit 1
            {"cop": "=", "returns": [["v", 3]], "args": [0]},  # clear bit 3
            # v should be 0b0101 = 5
        ],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r
    assert int(py_r["v"]) == 5, f"Clear bits 1,3 from 15 in 4-bit register: expected 5, got {int(py_r['v'])}"


def test_bit_read_in_condition() -> None:
    """Reading individual bits from a narrow register in conditions."""
    phir = _make_classical_program(
        [("v", "i64", 4), ("r", "i64", 4)],
        [
            {"cop": "=", "returns": ["v"], "args": [5]},  # 0b0101
            {"cop": "=", "returns": ["r"], "args": [0]},
            # if v[0] == 1, set r = 10
            {
                "block": "if",
                "condition": {"cop": "==", "args": [["v", 0], 1]},
                "true_branch": [{"cop": "=", "returns": ["r"], "args": [10]}],
            },
        ],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r
    assert int(py_r["r"]) == 10, f"if v[0]==1 should trigger for v=5, expected r=10, got {int(py_r['r'])}"


def test_bit_read_zero() -> None:
    """Reading a zero bit should not trigger condition."""
    phir = _make_classical_program(
        [("v", "i64", 4), ("r", "i64", 4)],
        [
            {"cop": "=", "returns": ["v"], "args": [5]},  # 0b0101
            {"cop": "=", "returns": ["r"], "args": [0]},
            # if v[1] == 1, set r = 10 (should NOT trigger, bit 1 is 0)
            {
                "block": "if",
                "condition": {"cop": "==", "args": [["v", 1], 1]},
                "true_branch": [{"cop": "=", "returns": ["r"], "args": [10]}],
            },
        ],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r
    assert int(py_r["r"]) == 0, f"if v[1]==1 should NOT trigger for v=5, expected r=0, got {int(py_r['r'])}"


# ── Bitwise ops on narrow registers ─────────────────────────────────


@pytest.mark.parametrize(
    ("cop", "a", "b", "size", "expected"),
    [
        ("&", 0b1010, 0b1100, 4, 0b1000),  # 10 & 12 = 8
        ("|", 0b1010, 0b0101, 4, 0b1111),  # 10 | 5 = 15
        ("^", 0b1010, 0b1111, 4, 0b0101),  # 10 ^ 15 = 5
        ("&", 0xFF, 0x0F, 4, 0x0F),  # 255 & 15, but stored in 4-bit = 15
        ("|", 0b1010, 0b0101, 2, 0b11),  # 10 | 5 = 15, masked to 2 bits = 3
    ],
)
def test_bitwise_ops_narrow(cop: str, a: int, b: int, size: int, expected: int) -> None:
    """Bitwise ops evaluate at full width, result masked to register size."""
    phir = _make_classical_program(
        [("a", "i64", size), ("b", "i64", size), ("r", "i64", size)],
        [
            {"cop": "=", "returns": ["a"], "args": [a]},
            {"cop": "=", "returns": ["b"], "args": [b]},
            {"cop": "=", "returns": ["r"], "args": [{"cop": cop, "args": ["a", "b"]}]},
        ],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r
    assert int(py_r["r"]) == expected, f"{a} {cop} {b} in i64 size={size}: expected {expected}, got {int(py_r['r'])}"


# ── All-zeros and all-ones ──────────────────────────────────────────


@pytest.mark.parametrize("size", [1, 2, 3, 4, 8, 16, 32, 63])
def test_all_zeros_register(size: int) -> None:
    """Register initialized to 0 should stay 0."""
    phir = _make_classical_program(
        [("v", "i64", size)],
        [{"cop": "=", "returns": ["v"], "args": [0]}],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r
    assert int(py_r["v"]) == 0


@pytest.mark.parametrize("size", [1, 2, 3, 4, 8, 16, 32, 63])
def test_all_ones_register(size: int) -> None:
    """Assigning max value for the register size."""
    max_val = (1 << size) - 1
    phir = _make_classical_program(
        [("v", "i64", size)],
        [{"cop": "=", "returns": ["v"], "args": [max_val]}],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r
    assert int(py_r["v"]) == max_val, f"All ones in size={size}: expected {max_val}, got {int(py_r['v'])}"


@pytest.mark.parametrize("size", [1, 2, 3, 4, 8, 16, 32, 63])
def test_overflow_to_all_ones(size: int) -> None:
    """0 - 1 in narrow register should give all ones (max value for that size)."""
    max_val = (1 << size) - 1
    phir = _make_classical_program(
        [("v", "i64", size)],
        [
            {"cop": "=", "returns": ["v"], "args": [0]},
            {"cop": "=", "returns": ["v"], "args": [{"cop": "-", "args": ["v", 1]}]},
        ],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r
    assert int(py_r["v"]) == max_val, f"0-1 in size={size}: expected {max_val}, got {int(py_r['v'])}"


@pytest.mark.parametrize("size", [1, 2, 3, 4, 8, 16, 32, 63])
def test_overflow_wraps_to_zero(size: int) -> None:
    """Max value + 1 in narrow register should wrap to 0."""
    max_val = (1 << size) - 1
    phir = _make_classical_program(
        [("v", "i64", size)],
        [
            {"cop": "=", "returns": ["v"], "args": [max_val]},
            {"cop": "=", "returns": ["v"], "args": [{"cop": "+", "args": ["v", 1]}]},
        ],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r
    assert int(py_r["v"]) == 0, f"max+1 in size={size}: expected 0, got {int(py_r['v'])}"
