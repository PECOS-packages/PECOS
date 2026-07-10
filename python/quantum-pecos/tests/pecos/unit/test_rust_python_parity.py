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
from pecos_rslib import RustPhirClassicalInterpreter, qasm_to_phir_json_py

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


@pytest.mark.parametrize("filename", ["spec_example.phir.json", "example1.phir.json"])
def test_wasm_parity(filename: str) -> None:
    """PHIR programs with WASM foreign calls must produce identical results."""
    from pecos import WasmForeignObject

    phir = json.loads((PHIR_DIR / filename).read_text())
    py_r, rs_r = run_both(
        phir,
        shots=20,
        seed=42,
        foreign_object=WasmForeignObject(WAT_DIR / "math.wat"),
    )
    assert py_r == rs_r


# ── Real-generator differential: QASM → PHIR ─────────────────────────

QASM_VALIDATION_DIR = (
    Path(__file__).resolve().parents[5] / "crates" / "pecos-qasm" / "tests" / "fixtures" / "qasm_validation"
)


@pytest.mark.parametrize(
    "qasm_file",
    sorted(QASM_VALIDATION_DIR.glob("*.qasm")),
    ids=lambda p: p.stem,
)
def test_qasm_generated_phir_parity(qasm_file: Path) -> None:
    """Real generated PHIR must be interpreted identically by both interpreters.

    Converts each QASM fixture to PHIR through the real Rust ``qasm_to_phir_json``
    generator (conditionals, classical registers, ...) and runs the result
    through both interpreters -- a differential over actual generator output,
    complementing the synthetic fuzz.
    """
    phir = qasm_to_phir_json_py(qasm_file.read_text())
    py_r, rs_r = run_both(phir, shots=10, seed=42)
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
            {"data": "cvar_define", "data_type": "i32", "variable": "x", "size": 31},
            {"data": "cvar_define", "data_type": "i32", "variable": "r", "size": 31},
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
        # A signed size-S register is an i(S+1) integer, so the full range of an
        # iN backing type is declared with size N-1 (N-1 data bits + sign bit).
        ("i8", 7, 127),
        ("i8", 7, -128),
        ("u8", 8, 255),
        ("u8", 8, 0),
        ("i16", 15, 32767),
        ("i16", 15, -32768),
        ("u16", 16, 65535),
        ("i32", 31, 2**31 - 1),
        ("i32", 31, -(2**31)),
        ("u32", 32, 2**32 - 1),
        ("i64", 63, 2**63 - 1),
        ("i64", 63, -(2**63)),
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
            {"data": "cvar_define", "data_type": "i32", "variable": "a", "size": 31},
            {"data": "cvar_define", "data_type": "i32", "variable": "b", "size": 31},
            {"data": "cvar_define", "data_type": "i32", "variable": "c", "size": 31},
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
            {"data": "cvar_define", "data_type": "i32", "variable": "a", "size": 31},
            {"data": "cvar_define", "data_type": "u32", "variable": "b", "size": 32},
            {"cop": "=", "returns": ["a"], "args": [42]},
            {"cop": "=", "returns": ["b"], "args": [7]},
        ],
    }
    py_r, rs_r = run_both(phir, qsim="stabilizer", return_int=True)
    assert py_r == rs_r


# ── Fuzz testing ─────────────────────────────────────────────────────

FUZZ_DTYPES = ["i32", "u32", "i64", "u64"]
# Includes / and % so signed (negative) division and modulo are fuzzed for parity.
FUZZ_COPS = ["+", "-", "*", "&", "|", "^", ">>", "<<", "/", "%", "==", "!=", "<", ">", "<=", ">="]


def _random_expr(rng: random.Random, vars_info: list, depth: int):
    """Generate a random (possibly nested) classical expression.

    Nests binary ops and the unary ``~`` so the interpreters' expression
    recursion is exercised, not just flat one-level ops. Division/modulo use a
    nonzero literal divisor and shifts a bounded literal amount, so no generated
    program divides by zero or shifts wildly (both interpreters would agree on
    those, but they'd raise/short-circuit and stop the differential).
    """
    if depth <= 0 or rng.random() < 0.4:
        # Leaf: a variable reference or a small literal.
        if rng.random() < 0.6:
            return rng.choice(vars_info)[0]
        return rng.randint(0, 15)
    if rng.random() < 0.2:
        return {"cop": "~", "args": [_random_expr(rng, vars_info, depth - 1)]}
    cop = rng.choice(FUZZ_COPS)
    lhs = _random_expr(rng, vars_info, depth - 1)
    if cop in ("/", "%"):
        rhs = rng.randint(1, 15)
    elif cop in (">>", "<<"):
        rhs = rng.randint(0, 15)
    else:
        rhs = _random_expr(rng, vars_info, depth - 1)
    return {"cop": cop, "args": [lhs, rhs]}


def _make_random_classical_program(rng: random.Random) -> dict:
    """Generate a random classical-only PHIR program."""
    nvars = rng.randint(2, 5)
    ops = []
    vars_info = []
    for i in range(nvars):
        dtype = rng.choice(FUZZ_DTYPES)
        tw = int(dtype[1:])
        # A signed size-S register is an i(S+1), so S+1 must fit the backing
        # width: signed sizes top out at tw-1, unsigned at tw.
        max_size = tw - 1 if dtype.startswith("i") else tw
        size = rng.randint(1, max_size)
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
            expr = _random_expr(rng, vars_info, depth=rng.randint(1, 3))
            ops.append({"cop": "=", "returns": [tname], "args": [expr]})
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
    for i in range(2000):
        phir = _make_random_classical_program(rng)
        py_r, rs_r = run_both(phir, seed=i, qsim="stabilizer")
        assert py_r == rs_r, f"Classical fuzz program {i} produced different results"


def test_fuzz_quantum_programs() -> None:
    """Fuzz test: random quantum programs must produce identical results."""
    rng = random.Random(2026)
    identical = 0

    for i in range(1000):
        phir = _make_random_quantum_program(rng)
        py_r, rs_r = run_both(phir, shots=20, seed=i + 10000)
        assert py_r == rs_r, f"Quantum fuzz program {i} produced different results"
        identical += 1

    assert identical == 1000


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
        # A signed size-S register is an i(S+1) integer: wrap to S+1 bits and
        # sign-extend, so a value whose sign bit (bit S) is set is negative.
        (2, 3, 3),  # i3: 3 = 0b011, sign bit clear -> 3
        (2, 5, -3),  # i3: 5 = 0b101, sign bit set -> 5 - 8 = -3
        (2, 4, -4),  # i3: 4 = 0b100 -> 4 - 8 = -4
        (2, 7, -1),  # i3: 7 = 0b111 -> 7 - 8 = -1
        (3, 10, -6),  # i4: 10 = 0b1010 -> 10 - 16 = -6
        (4, 255, -1),  # i5: 255 & 31 = 31 = 0b11111 -> -1
        (1, 1, 1),  # i2: 1 = 0b01 -> 1
        (1, 2, -2),  # i2: 2 = 0b10 -> 2 - 4 = -2
        (1, 3, -1),  # i2: 3 = 0b11 -> 3 - 4 = -1
    ],
)
def test_signed_narrow_register_masking(size: int, val: int, expected: int) -> None:
    """A signed i64 size-S register is an i(S+1): assignment wraps and sign-extends."""
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
        ("i32", 2, 5, -3),  # i3: assign 5 -> 0b101 -> -3
        ("i32", 4, 255, -1),  # i5: assign 255 -> 0b11111 -> -1
        ("u32", 2, 5, 1),  # u2: assign 5 -> mask to 2 bits = 1
        ("u64", 3, 10, 2),  # u3: assign 10 -> mask to 3 bits = 2
        ("i64", 4, -1, -1),  # i5: -1 stays -1 (0b11111)
    ],
)
def test_narrow_register_masking_all_types(dtype: str, size: int, val: int, expected: int) -> None:
    """Signed types wrap to i(S+1); unsigned types mask to S bits on assignment."""
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
        # ~val at full width, then wrapped/sign-extended into i(S+1).
        (4, 5, -6),  # ~5 = -6, fits i5 -> -6
        (4, 0, -1),  # ~0 = -1 -> i5 -> -1
        (4, 15, -16),  # ~15 = -16 -> i5 -> -16
        (8, 0xAA, -171),  # ~0xAA = -171 -> i9 -> -171
        (2, 0, -1),  # ~0 = -1 -> i3 -> -1
        (1, 0, -1),  # ~0 = -1 -> i2 -> -1
        (1, 1, -2),  # ~1 = -2 -> i2 -> -2
    ],
)
def test_not_narrow_register(size: int, val: int, expected: int) -> None:
    """~ evaluates at full width, result wrapped/sign-extended into i(S+1)."""
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
        (4, 1, 10, 0),  # 1 << 10 = 1024 -> i5 -> 0
        (4, 1, 3, 8),  # 1 << 3 = 8 -> i5 -> 8
        (4, 1, 4, -16),  # 1 << 4 = 16 = 0b10000 -> i5 sign bit -> -16
        (8, 1, 7, 128),  # 1 << 7 = 128 -> i9 -> 128
        (8, 1, 8, -256),  # 1 << 8 = 256 = sign bit of i9 -> -256
        (2, 1, 1, 2),  # 1 << 1 = 2 -> i3 -> 2
        (2, 1, 2, -4),  # 1 << 2 = 4 = sign bit of i3 -> -4
    ],
)
def test_left_shift_with_masking(size: int, val: int, shift: int, expected: int) -> None:
    """Left shift evaluates at full width, result wrapped/sign-extended into i(S+1)."""
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
        # Subtraction underflow: 0 - 1 = -1 fits every i(S+1) -> -1
        (3, 0, 1, "-", -1),  # i4: 0 - 1 = -1
        (4, 0, 1, "-", -1),  # i5: 0 - 1 = -1
        (8, 0, 1, "-", -1),  # i9: 0 - 1 = -1
        # Addition overflow past the sign bit wraps negative
        (4, 15, 1, "+", -16),  # i5: 15 + 1 = 16 = sign bit -> -16
        (4, 15, 2, "+", -15),  # i5: 15 + 2 = 17 = 0b10001 -> -15
        (3, 7, 1, "+", -8),  # i4: 7 + 1 = 8 = sign bit -> -8
        # Multiplication overflow
        (4, 4, 5, "*", -12),  # i5: 4 * 5 = 20 = 0b10100 -> -12
    ],
)
def test_expression_overflow_narrow(size: int, a_val: int, b_val: int, cop: str, expected: int) -> None:
    """Arithmetic at full width, overflow wrapped/sign-extended into i(S+1)."""
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


# ── Signed division / modulo with negative operands ─────────────────


@pytest.mark.parametrize(
    ("a_val", "b_val", "cop", "expected"),
    [
        # Truncate toward zero (C/Rust), NOT Python floor. Remainder sign
        # follows the dividend.
        (-7, 2, "/", -3),
        (-7, 3, "%", -1),
        (7, -2, "/", -3),
        (7, -3, "%", 1),
        (-7, -2, "/", 3),
        (-8, 3, "%", -2),
        (-1, 2, "/", 0),
        (-10, 3, "/", -3),
        (-10, 3, "%", -1),
    ],
)
def test_signed_division_modulo_negative(a_val: int, b_val: int, cop: str, expected: int) -> None:
    """Signed division/modulo truncate toward zero and match Rust for negatives."""
    phir = _make_classical_program(
        [("a", "i64", 63), ("r", "i64", 63)],
        [
            {"cop": "=", "returns": ["a"], "args": [a_val]},
            {"cop": "=", "returns": ["r"], "args": [{"cop": cop, "args": ["a", b_val]}]},
        ],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r, f"Parity failure: py={py_r}, rs={rs_r}"
    assert int(py_r["r"]) == expected, f"{a_val} {cop} {b_val}: expected {expected}, got {int(py_r['r'])}"


# ── Full-backing (64-bit) evaluation of intermediate overflow ────────


def test_intermediate_overflow_matches_backing_width() -> None:
    """Intermediate arithmetic wraps at 64 bits before a comparison (not arbitrary precision).

    ``(2**62 * 4)`` overflows 64 bits to 0, so ``== 0`` is true in both
    interpreters -- Python must evaluate at the backing width, not arbitrary
    precision.
    """
    phir = _make_classical_program(
        [("a", "i64", 63), ("r", "i64", 63)],
        [
            {"cop": "=", "returns": ["a"], "args": [2**62]},
            {"cop": "=", "returns": ["r"], "args": [{"cop": "==", "args": [{"cop": "*", "args": ["a", 4]}, 0]}]},
        ],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r
    assert int(py_r["r"]) == 1, f"(2^62*4)==0 should be 1, got {int(py_r['r'])}"


def test_unsigned_large_value_comparison() -> None:
    """A u64 value above 2**63 compares as a large positive (unsigned), matching Rust.

    This is the case a naive "wrap intermediates to i64" would get wrong: the
    value must stay unsigned, so ``> 10`` is true.
    """
    phir = _make_classical_program(
        [("u", "u64", 64), ("r", "u64", 64)],
        [
            {"cop": "=", "returns": ["u"], "args": [2**63 + 5]},
            {"cop": "=", "returns": ["r"], "args": [{"cop": ">", "args": ["u", 10]}]},
        ],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r
    assert int(py_r["r"]) == 1, f"u64(2^63+5) > 10 should be 1, got {int(py_r['r'])}"


@pytest.mark.parametrize(
    ("cop", "b", "expected"),
    [
        # A literal above i64::MAX is an unsigned operand (Rust ArgItem::UInteger),
        # so the shift is logical and the comparison is unsigned.
        (">>", 1, (2**64 - 1) >> 1),
        ("<", 0, 0),
        (">", 0, 1),
    ],
)
def test_large_literal_is_unsigned(cop: str, b: int, expected: int) -> None:
    """Literals in [2**63, 2**64) evaluate as unsigned, matching Rust."""
    phir = _make_classical_program(
        [("r", "u64", 64)],
        [{"cop": "=", "returns": ["r"], "args": [{"cop": cop, "args": [2**64 - 1, b]}]}],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r, f"Parity failure: py={py_r}, rs={rs_r}"
    assert int(py_r["r"]) == expected


@pytest.mark.parametrize("bit", [0, 1])
def test_not_of_single_bit_is_boolean(bit: int) -> None:
    """~ of a single bit is a boolean NOT (0<->1), not a full-width bit flip."""
    phir = _make_classical_program(
        [("m", "u64", 2), ("r", "u64", 64)],
        [
            {"cop": "=", "returns": [["m", 0]], "args": [bit]},
            {"cop": "=", "returns": ["r"], "args": [{"cop": "~", "args": [["m", 0]]}]},
        ],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r, f"Parity failure: py={py_r}, rs={rs_r}"
    assert int(py_r["r"]) == (0 if bit else 1)


@pytest.mark.parametrize("bit", [0, 1])
def test_double_not_of_bit_roundtrips(bit: int) -> None:
    """~~ of a bit is the bit again (boolean NOT is involutive), matching Rust."""
    phir = _make_classical_program(
        [("m", "u64", 2), ("r", "u64", 64)],
        [
            {"cop": "=", "returns": [["m", 0]], "args": [bit]},
            {"cop": "=", "returns": ["r"], "args": [{"cop": "~", "args": [{"cop": "~", "args": [["m", 0]]}]}]},
        ],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r, f"Parity failure: py={py_r}, rs={rs_r}"
    assert int(py_r["r"]) == bit


def test_not_of_bit_expression_is_bitwise() -> None:
    """~ of an arithmetic expression on a bit is a bitwise NOT, not boolean."""
    phir = _make_classical_program(
        [("m", "u64", 2), ("r", "u64", 64)],
        [
            {"cop": "=", "returns": [["m", 0]], "args": [1]},
            {"cop": "=", "returns": ["r"], "args": [{"cop": "~", "args": [{"cop": "&", "args": [["m", 0], 5]}]}]},
        ],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r, f"Parity failure: py={py_r}, rs={rs_r}"
    assert int(py_r["r"]) == ((1 & 5) ^ (2**64 - 1))  # ~(1) at 64 bits


# ── Nested expressions with narrow registers ────────────────────────


def test_nested_expression_full_width() -> None:
    """Nested expressions should evaluate at full width before storing."""
    # (a | b) + c where a=3, b=12, c=1 -> (3|12)+1 = 16 = sign bit of i5 -> -16
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
    assert int(py_r["r"]) == -16, f"(3|12)+1 in i5: expected -16, got {int(py_r['r'])}"


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
    """Assigning a wide-register value to a narrow register wraps into i(S+1)."""
    phir = _make_classical_program(
        [("wide", "i64", 32), ("narrow", "i64", 4)],
        [
            {"cop": "=", "returns": ["wide"], "args": [255]},
            {"cop": "=", "returns": ["narrow"], "args": ["wide"]},
        ],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r
    assert int(py_r["narrow"]) == -1, f"255 into i5 register: expected -1, got {int(py_r['narrow'])}"


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
    # narrow = 15 (i5 -> +15), wide = 250 (i33 -> +250), sum = 265; i9: 265 - 512 = -247
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
    assert int(py_r["result"]) == -247, f"15 + 250 in i9 register: expected -247, got {int(py_r['result'])}"


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
        # a=10 -> i3 +2, b=5 -> i3 -3; 2 | -3 = -1; stored i3 -> -1
        ("|", 0b1010, 0b0101, 2, -1),
    ],
)
def test_bitwise_ops_narrow(cop: str, a: int, b: int, size: int, expected: int) -> None:
    """Bitwise ops evaluate at full width on the signed operands, then wrap into i(S+1)."""
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
def test_overflow_to_negative_one(size: int) -> None:
    """0 - 1 in a signed i(S+1) register is -1 (all S+1 bits set)."""
    phir = _make_classical_program(
        [("v", "i64", size)],
        [
            {"cop": "=", "returns": ["v"], "args": [0]},
            {"cop": "=", "returns": ["v"], "args": [{"cop": "-", "args": ["v", 1]}]},
        ],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r
    assert int(py_r["v"]) == -1, f"0-1 in size={size}: expected -1, got {int(py_r['v'])}"


@pytest.mark.parametrize("size", [1, 2, 3, 4, 8, 16, 32, 63])
def test_overflow_wraps_to_min(size: int) -> None:
    """Max value + 1 in a signed i(S+1) register wraps to its minimum, -(2**S)."""
    max_val = (1 << size) - 1  # i(S+1) max: bit S clear
    expected = -(1 << size)  # i(S+1) min
    phir = _make_classical_program(
        [("v", "i64", size)],
        [
            {"cop": "=", "returns": ["v"], "args": [max_val]},
            {"cop": "=", "returns": ["v"], "args": [{"cop": "+", "args": ["v", 1]}]},
        ],
    )
    py_r, rs_r = _run_classical(phir)
    assert py_r == rs_r
    assert int(py_r["v"]) == expected, f"max+1 in size={size}: expected {expected}, got {int(py_r['v'])}"
