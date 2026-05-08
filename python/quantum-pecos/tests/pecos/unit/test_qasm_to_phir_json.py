# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Tests for the Rust QASM-to-PHIR-JSON converter.

Validates that converter output conforms to the PHIR spec using the
official `phir` validator package.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pecos_rslib import qasm_to_phir_json_py
from phir.model import PHIRModel

FIXTURE_DIR = Path(__file__).resolve().parents[5] / "crates" / "pecos-qasm" / "tests" / "fixtures" / "qasm_validation"


def all_qasm_fixtures() -> list[Path]:
    """Collect all .qasm files from the hardware validation fixtures."""
    return sorted(FIXTURE_DIR.glob("*.qasm"))


@pytest.fixture(params=all_qasm_fixtures(), ids=lambda p: p.stem)
def qasm_fixture(request: pytest.FixtureRequest) -> Path:
    """Parametrized fixture yielding each .qasm file."""
    return request.param


def test_converter_output_validates(qasm_fixture: Path) -> None:
    """Each QASM fixture should convert to valid PHIR-JSON."""
    qasm_str = qasm_fixture.read_text()
    phir = qasm_to_phir_json_py(qasm_str)

    # Structural checks
    assert phir["format"] == "PHIR/JSON"
    assert phir["version"] == "0.1.0"
    assert isinstance(phir["ops"], list)

    # Official PHIR validator
    PHIRModel.model_validate(phir)


def test_bell_state_validates() -> None:
    """A simple Bell state QASM program should produce valid PHIR-JSON."""
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    """
    phir = qasm_to_phir_json_py(qasm)
    PHIRModel.model_validate(phir)


def test_all_cregs_are_i64() -> None:
    """All classical registers should be emitted as i64."""
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[4];
    creg a[1];
    creg b[4];
    creg c[32];
    """
    phir = qasm_to_phir_json_py(qasm)
    cvars = [op for op in phir["ops"] if op.get("data") == "cvar_define"]
    assert len(cvars) == 3
    for cvar in cvars:
        assert cvar["data_type"] == "i64"


def test_conditional_produces_if_block() -> None:
    """QASM if-statements should become PHIR if-blocks."""
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    creg c[1];
    measure q[0] -> c[0];
    if(c==1) x q[0];
    """
    phir = qasm_to_phir_json_py(qasm)
    PHIRModel.model_validate(phir)

    if_blocks = [op for op in phir["ops"] if op.get("block") == "if"]
    assert len(if_blocks) == 1
    assert if_blocks[0]["condition"] == {"cop": "==", "args": ["c", 1]}
    assert len(if_blocks[0]["true_branch"]) == 1
    assert if_blocks[0]["true_branch"][0]["qop"] == "X"


def test_measurements_are_inline() -> None:
    """Measurements should appear inline (not deferred to end)."""
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    creg m[1];
    creg r[1];
    x q[0];
    measure q[0] -> m[0];
    if(m==1) x q[0];
    measure q[0] -> r[0];
    """
    phir = qasm_to_phir_json_py(qasm)
    PHIRModel.model_validate(phir)

    # Extract operation sequence (skip definitions and export)
    op_types = []
    for op in phir["ops"]:
        if "qop" in op:
            op_types.append(op["qop"])
        elif "block" in op:
            op_types.append("if")

    # Should be: X, Measure, if, Measure (inline, not deferred)
    assert op_types == ["X", "Measure", "if", "Measure"]


def test_register_sizes_preserved() -> None:
    """Register sizes from QASM should be preserved in PHIR-JSON."""
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[8];
    creg narrow[2];
    creg wide[32];
    """
    phir = qasm_to_phir_json_py(qasm)
    cvars = {op["variable"]: op["size"] for op in phir["ops"] if op.get("data") == "cvar_define"}
    assert cvars["narrow"] == 2
    assert cvars["wide"] == 32


# ── Round-trip tests: QASM -> PHIR-JSON -> interpreter -> check values ──


def _run_phir_json(phir: dict, *, shots: int = 1, seed: int = 42) -> dict:
    """Run a PHIR-JSON program through both interpreters, return single-shot int results."""
    from pecos.classical_interpreters.phir_classical_interpreter import PhirClassicalInterpreter
    from pecos.engines.hybrid_engine import HybridEngine
    from pecos_rslib import RustPhirClassicalInterpreter

    py_i = PhirClassicalInterpreter()
    py_r = HybridEngine(cinterp=py_i).run(phir, shots=shots, seed=seed, return_int=True)

    rs_i = RustPhirClassicalInterpreter()
    rs_r = HybridEngine(cinterp=rs_i).run(phir, shots=shots, seed=seed, return_int=True)

    py_vals = {k: int(v[0]) for k, v in py_r.items()}
    rs_vals = {k: int(v[0]) for k, v in rs_r.items()}
    assert py_vals == rs_vals, f"Parity failure: py={py_vals}, rs={rs_vals}"
    return py_vals


def test_roundtrip_deterministic_measure() -> None:
    """QASM -> PHIR-JSON -> interpreter: X gates produce deterministic measurement results."""
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg m[2];
    x q[0];
    x q[1];
    measure q[0] -> m[0];
    measure q[1] -> m[1];
    """
    phir = qasm_to_phir_json_py(qasm)
    results = _run_phir_json(phir)
    assert results["m"] == 3, f"Both qubits X'd, expected m=3, got {results['m']}"


def test_roundtrip_conditional_feedback() -> None:
    """QASM -> PHIR-JSON -> interpreter: measure-conditional-correct loop."""
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    creg m[1];
    creg r[1];
    x q[0];
    measure q[0] -> m[0];
    if(m==1) x q[0];
    measure q[0] -> r[0];
    """
    phir = qasm_to_phir_json_py(qasm)
    results = _run_phir_json(phir)
    assert results["m"] == 1, f"First measure after X: expected m=1, got {results['m']}"
    assert results["r"] == 0, f"After correction: expected r=0, got {results['r']}"


def test_roundtrip_conditional_no_trigger() -> None:
    """QASM -> PHIR-JSON -> interpreter: conditional that should not fire."""
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    creg m[1];
    creg r[1];
    x q[0];
    measure q[0] -> m[0];
    if(m==0) x q[0];
    measure q[0] -> r[0];
    """
    phir = qasm_to_phir_json_py(qasm)
    results = _run_phir_json(phir)
    assert results["m"] == 1, f"Expected m=1, got {results['m']}"
    assert results["r"] == 1, f"Condition should not trigger, expected r=1, got {results['r']}"


def test_roundtrip_multi_bit_conditional() -> None:
    """QASM -> PHIR-JSON -> interpreter: if(c==3) on 2-bit register."""
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg m[2];
    creg r[1];
    x q[0];
    x q[1];
    measure q[0] -> m[0];
    measure q[1] -> m[1];
    if(m==3) x q[0];
    measure q[0] -> r[0];
    """
    phir = qasm_to_phir_json_py(qasm)
    results = _run_phir_json(phir)
    assert results["m"] == 3, f"Expected m=3, got {results['m']}"
    assert results["r"] == 0, f"Conditional should trigger, expected r=0, got {results['r']}"


def test_roundtrip_wide_register() -> None:
    """QASM -> PHIR-JSON -> interpreter: 8-bit register with alternating pattern."""
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[8];
    creg m[8];
    x q[1];
    x q[3];
    x q[5];
    x q[7];
    measure q[0] -> m[0];
    measure q[1] -> m[1];
    measure q[2] -> m[2];
    measure q[3] -> m[3];
    measure q[4] -> m[4];
    measure q[5] -> m[5];
    measure q[6] -> m[6];
    measure q[7] -> m[7];
    """
    phir = qasm_to_phir_json_py(qasm)
    results = _run_phir_json(phir)
    assert results["m"] == 170, f"Alternating pattern 10101010 = 170, got {results['m']}"


def test_roundtrip_conditional_chain() -> None:
    """QASM -> PHIR-JSON -> interpreter: sequential measure-correct chain."""
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    creg m0[1];
    creg m1[1];
    creg m2[1];
    x q[0];
    measure q[0] -> m0[0];
    if(m0==1) x q[0];
    x q[0];
    measure q[0] -> m1[0];
    if(m1==1) x q[0];
    measure q[0] -> m2[0];
    """
    phir = qasm_to_phir_json_py(qasm)
    results = _run_phir_json(phir)
    assert results["m0"] == 1, f"Expected m0=1, got {results['m0']}"
    assert results["m1"] == 1, f"Expected m1=1, got {results['m1']}"
    assert results["m2"] == 0, f"Expected m2=0, got {results['m2']}"


def test_roundtrip_multi_register() -> None:
    """QASM -> PHIR-JSON -> interpreter: multiple registers of different sizes."""
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[4];
    creg a[1];
    creg b[1];
    creg c[2];
    x q[0];
    x q[2];
    x q[3];
    measure q[0] -> a[0];
    measure q[1] -> b[0];
    measure q[2] -> c[0];
    measure q[3] -> c[1];
    """
    phir = qasm_to_phir_json_py(qasm)
    results = _run_phir_json(phir)
    assert results["a"] == 1, f"Expected a=1, got {results['a']}"
    assert results["b"] == 0, f"Expected b=0, got {results['b']}"
    assert results["c"] == 3, f"Expected c=3, got {results['c']}"


def test_roundtrip_conditional_cx() -> None:
    """QASM -> PHIR-JSON -> interpreter: conditional two-qubit gate."""
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg m[1];
    creg r[2];
    x q[0];
    measure q[0] -> m[0];
    if(m==1) cx q[0], q[1];
    measure q[0] -> r[0];
    measure q[1] -> r[1];
    """
    phir = qasm_to_phir_json_py(qasm)
    results = _run_phir_json(phir)
    assert results["m"] == 1, f"Expected m=1, got {results['m']}"
    assert results["r"] == 3, f"CX should trigger, both qubits |1>, expected r=3, got {results['r']}"
