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

"""Resolution and execution coverage for every gate emitted by SLR QASM codegen."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest
from pecos.slr import CReg, Main, QReg, Return, SlrConverter, rad
from pecos.slr.ast.codegen.qasm import GATE_TO_QASM, AstToQasm
from pecos.slr.ast.nodes import AllocatorDecl, GateKind, GateOp, LiteralExpr, Program, SlotRef
from pecos.slr.qeclib import qubit
from pecos_rslib import Qasm, sim, state_vector

GATE_DEFINITION = re.compile(r"^\s*gate\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)


def _included_macro_names(includes: list[str]) -> set[str]:
    repository = next(parent for parent in Path(__file__).parents if (parent / "crates/pecos-qasm/includes").is_dir())
    include_directory = repository / "crates/pecos-qasm/includes"
    names = set()
    for include in includes:
        names.update(GATE_DEFINITION.findall((include_directory / include).read_text(encoding="utf-8")))
    return names


def _emitted_names(gate: GateKind, qasm: str) -> set[str]:
    mapped_call = GATE_TO_QASM[gate]
    if mapped_call is not None:
        return {mapped_call.split("(", maxsplit=1)[0]}

    names = set()
    for line in qasm.splitlines():
        operation = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)", line)
        if operation and operation.group(1) not in {"OPENQASM", "include", "qreg"}:
            names.add(operation.group(1))
    return names


def _single_gate_program(gate: GateKind) -> Program:
    targets = tuple(SlotRef(allocator="q", index=index) for index in range(gate.arity))
    parameter = 0.37 if gate.name.startswith("CR") else rad(0.37)
    params = (LiteralExpr(value=parameter),) if gate.is_parameterized else ()
    return Program(
        name=f"resolve_{gate.name}",
        allocator=AllocatorDecl(name="q", capacity=2),
        body=(GateOp(gate=gate, targets=targets, params=params),),
    )


@pytest.mark.parametrize("gate", GateKind, ids=lambda gate: gate.name)
def test_every_emitted_gate_resolves_under_default_includes(gate: GateKind) -> None:
    """The Rust parser must resolve every direct and specially lowered codegen spelling."""
    generator = AstToQasm()
    qasm = "\n".join(generator.generate(_single_gate_program(gate)))
    emitted_names = _emitted_names(gate, qasm)
    macro_names = _included_macro_names(generator.includes)

    for emitted_name in emitted_names:
        if emitted_name[0].islower() and emitted_name not in macro_names:
            pytest.fail(
                f"emitted gate {emitted_name!r} did not resolve; includes searched: {generator.includes!r}",
                pytrace=False,
            )

    try:
        sim(Qasm.from_string(qasm)).quantum(state_vector()).run(1)
    except Exception as error:
        pytest.fail(
            f"emitted gates {sorted(emitted_names)!r} did not resolve; "
            f"includes searched: {generator.includes!r}; {error}",
            pytrace=False,
        )


def _run_slr_qasm(program: Main, shots: int = 8) -> list[int]:
    qasm = SlrConverter(program).qasm()
    results = sim(Qasm.from_string(qasm)).quantum(state_vector()).seed(42).run(shots)
    values = results.to_dict()["c"]
    assert isinstance(values, list)
    return values


def test_sxx_and_syy_have_distinct_end_to_end_signatures() -> None:
    """Native SXX and SYY retain their distinguishing actions."""
    sxx_twice = Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        qubit.SXX(q[0], q[1]),
        qubit.SXX(q[0], q[1]),
        qubit.Measure(q[0]) > c[0],
        qubit.Measure(q[1]) > c[1],
        Return(c),
    )
    syy_twice = Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        qubit.SYY(q[0], q[1]),
        qubit.SYY(q[0], q[1]),
        qubit.Measure(q[0]) > c[0],
        qubit.Measure(q[1]) > c[1],
        Return(c),
    )
    sxx_then_syy = Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        qubit.SXX(q[0], q[1]),
        qubit.SYY(q[0], q[1]),
        qubit.Measure(q[0]) > c[0],
        qubit.Measure(q[1]) > c[1],
        Return(c),
    )
    odd_parity = Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        qubit.X(q[1]),
        qubit.SXX(q[0], q[1]),
        qubit.SYY(q[0], q[1]),
        qubit.Measure(q[0]) > c[0],
        qubit.Measure(q[1]) > c[1],
        Return(c),
    )
    assert _run_slr_qasm(sxx_twice) == [3] * 8
    assert _run_slr_qasm(syy_twice) == [3] * 8
    assert _run_slr_qasm(sxx_then_syy) == [0] * 8
    assert _run_slr_qasm(odd_parity) == [1] * 8


def test_szz_native_end_to_end_signature() -> None:
    """Native SZZ reaches execution and has its squared Pauli action."""
    szz_twice = Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        qubit.H(q[0]),
        qubit.H(q[1]),
        qubit.SZZ(q[0], q[1]),
        qubit.SZZ(q[0], q[1]),
        qubit.H(q[0]),
        qubit.H(q[1]),
        qubit.Measure(q[0]) > c[0],
        qubit.Measure(q[1]) > c[1],
        Return(c),
    )

    szz_qasm = SlrConverter(szz_twice).qasm()
    assert "\nSZZ " in szz_qasm
    assert "\nZZ " not in szz_qasm
    assert _run_slr_qasm(szz_twice) == [3] * 8


def test_hqslib_opt_in_executes_supported_native_program() -> None:
    """The explicit Quantinuum include remains usable with native SZZ emission."""
    program = Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        qubit.H(q[0]),
        qubit.H(q[1]),
        qubit.SZZ(q[0], q[1]),
        qubit.SZZ(q[0], q[1]),
        qubit.H(q[0]),
        qubit.H(q[1]),
        qubit.Measure(q[0]) > c[0],
        qubit.Measure(q[1]) > c[1],
        Return(c),
    )

    qasm = SlrConverter(program, includes=["hqslib1.inc"]).qasm()
    assert 'include "hqslib1.inc";' in qasm
    assert 'include "qelib1.inc";' not in qasm
    results = sim(Qasm.from_string(qasm)).quantum(state_vector()).run(1)
    assert results.to_dict()["c"] == [3]


def test_ch_exercises_both_control_branches_end_to_end() -> None:
    """CH preserves both control branches and their relative phase."""
    active_control = Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        qubit.X(q[0]),
        qubit.CH(q[0], q[1]),
        qubit.H(q[1]),
        qubit.Measure(q[0]) > c[0],
        qubit.Measure(q[1]) > c[1],
        Return(c),
    )
    inactive_control = Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        qubit.X(q[1]),
        qubit.CH(q[0], q[1]),
        qubit.Measure(q[0]) > c[0],
        qubit.Measure(q[1]) > c[1],
        Return(c),
    )
    phase_sensitive_control = Main(
        q := QReg("q", 2),
        c := CReg("c", 1),
        qubit.H(q[0]),
        qubit.RY(rad(-3 * math.pi / 4), q[1]),
        qubit.CH(q[0], q[1]),
        qubit.H(q[0]),
        qubit.Measure(q[0]) > c[0],
        Return(c),
    )

    assert _run_slr_qasm(active_control, shots=32) == [1] * 32
    assert _run_slr_qasm(inactive_control) == [2] * 8
    assert _run_slr_qasm(phase_sensitive_control, shots=32) == [1] * 32


def test_controlled_rotations_have_distinct_end_to_end_signatures() -> None:
    """The SLR QASM path distinguishes CRX, CRY, and CRZ exactly."""
    direct_programs = []
    conjugated_programs = []
    control_superposition_programs = []
    sign_sensitive_programs = []
    for gate in (qubit.CRX, qubit.CRY, qubit.CRZ):
        direct_programs.append(
            Main(
                q := QReg("q", 2),
                c := CReg("c", 2),
                qubit.X(q[0]),
                gate(math.pi, q[0], q[1]),
                qubit.Measure(q[0]) > c[0],
                qubit.Measure(q[1]) > c[1],
                Return(c),
            ),
        )
        conjugated_programs.append(
            Main(
                q := QReg("q", 2),
                c := CReg("c", 2),
                qubit.X(q[0]),
                qubit.H(q[1]),
                gate(math.pi, q[0], q[1]),
                qubit.H(q[1]),
                qubit.Measure(q[0]) > c[0],
                qubit.Measure(q[1]) > c[1],
                Return(c),
            ),
        )
        control_superposition_programs.append(
            Main(
                q := QReg("q", 2),
                c := CReg("c", 1),
                qubit.H(q[0]),
                gate(math.tau, q[0], q[1]),
                qubit.H(q[0]),
                qubit.Measure(q[0]) > c[0],
                Return(c),
            ),
        )
        q = QReg("q", 2)
        c = CReg("c", 1)
        target_preparation = []
        if gate is qubit.CRX:
            target_preparation.append(qubit.H(q[1]))
        elif gate is qubit.CRY:
            target_preparation.extend((qubit.H(q[1]), qubit.SZ(q[1])))
        sign_sensitive_programs.append(
            Main(
                q,
                c,
                qubit.H(q[0]),
                *target_preparation,
                gate(math.pi, q[0], q[1]),
                qubit.SZdg(q[0]),
                qubit.H(q[0]),
                qubit.Measure(q[0]) > c[0],
                Return(c),
            ),
        )

    assert [_run_slr_qasm(program) for program in direct_programs] == [[3] * 8, [3] * 8, [1] * 8]
    assert [_run_slr_qasm(program) for program in conjugated_programs] == [[1] * 8, [3] * 8, [3] * 8]
    assert [_run_slr_qasm(program) for program in control_superposition_programs] == [[1] * 8] * 3
    assert [_run_slr_qasm(program) for program in sign_sensitive_programs] == [[1] * 8] * 3
