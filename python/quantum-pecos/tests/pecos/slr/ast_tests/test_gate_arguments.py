# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0

"""Gate argument validation and dependency regressions for issue #1476."""

import pytest
from pecos.slr import CReg, LoopVar, Main, Parallel, QAlloc, QReg, SlrConverter
from pecos.slr.ast import GateKind, GateOp, SlotRef, slr_to_ast
from pecos.slr.gen_codes import QASMGenerator, QIRGenerator, QuantumCircuitGenerator, StimGenerator
from pecos.slr.qeclib import qubit as qb


@pytest.mark.parametrize("shape", ["register", "register_tuple", "pair", "tuple"])
def test_repeated_qubits(shape: str) -> None:
    q = QReg("q", 2)
    ops = {
        "register": qb.CX(q, q),
        "register_tuple": qb.CX((q, q)),
        "pair": qb.CX(q[0], q[0]),
        "tuple": qb.CX((q[0], q[0])),
    }
    with pytest.raises(ValueError, match=r"CX.*q\[0\]"):
        slr_to_ast(Main(q, ops[shape]))


@pytest.mark.parametrize("kind", [kind for kind in GateKind if kind.arity > 1])
def test_gate_op_repeated_targets(kind: GateKind) -> None:
    with pytest.raises(ValueError, match=rf"{kind.name}.*q\[0\]"):
        GateOp(gate=kind, targets=(SlotRef(allocator="q", index=0), SlotRef(allocator="q", index=0)))


@pytest.mark.parametrize("shape", ["register", "tuple", "register_tuple", "explicit", "allocator"])
def test_parallel_dependencies(shape: str) -> None:
    q = QAlloc(2, name="q") if shape == "allocator" else QReg("q", 2)
    r = QReg("r", 2)
    pairs = [qb.CX(q[i], r[i]) for i in range(2)]
    ops = {
        "register": [qb.CX(q, r)],
        "tuple": [qb.CX((q[0], r[0]), (q[1], r[1]))],
        "register_tuple": [qb.CX((q, r))],
        "explicit": pairs,
        "allocator": pairs,
    }[shape]
    expected = SlrConverter(Main(q, r, *pairs, qb.H(q[0])), optimize_parallel=False).qasm(skip_headers=True)
    for optimize in (False, True):
        actual = SlrConverter(Main(q, r, Parallel(*ops, qb.H(q[0]))), optimize_parallel=optimize).qasm(
            skip_headers=True,
        )
        for ref in ("q[0]", "q[1]", "r[0]", "r[1]"):
            assert [line.strip() for line in actual.splitlines() if ref in line] == [
                line.strip() for line in expected.splitlines() if ref in line
            ]


def test_parallel_single_qubit_register_dependency() -> None:
    q = QReg("q", 2)
    actual = SlrConverter(Main(q, Parallel(qb.X(q), qb.H(q[0])))).qasm(skip_headers=True)
    assert [line for line in actual.splitlines() if "q[0]" in line] == ["x q[0];", "h q[0];"]


def test_single_qubit_allocator() -> None:
    a = QAlloc(2, name="a")
    assert slr_to_ast(Main(a, qb.H(a))).body == tuple(
        GateOp(gate=GateKind.H, targets=(SlotRef(allocator="a", index=i),)) for i in range(2)
    )


@pytest.mark.parametrize("shape", ["register", "bit", "string", "number", "nested"])
def test_single_qubit_invalid_argument(shape: str) -> None:
    q, c = QReg("q", 2), CReg("c", 2)
    arg = {"register": c, "bit": c[0], "string": "bad", "number": 42, "nested": [c[0]]}[shape]
    with pytest.raises(ValueError, match="H") as exc:
        slr_to_ast(Main(q, c, qb.H(q[0], arg)))
    assert str(c[0] if shape == "nested" else arg) in str(exc.value)


@pytest.mark.parametrize("shape", ["none", "register", "allocator", "list"])
def test_single_qubit_empty_arguments(shape: str) -> None:
    args = {"none": (), "register": (QReg("q", 0),), "allocator": (QAlloc(0, name="a"),), "list": ([],)}[shape]
    with pytest.raises(ValueError, match="H"):
        slr_to_ast(Main(qb.H(*args)))


@pytest.mark.parametrize("gate_name", ["H", "CX"])
@pytest.mark.parametrize("nested", [False, True])
def test_symbolic_qubit_rejected(gate_name: str, nested: bool) -> None:
    q, r, i = QReg("q", 2), QReg("r", 2), LoopVar("i")
    args = (q[i],) if gate_name == "H" else (q[i], r[i])
    if nested:
        args = (list(args),) if gate_name == "H" else (args,)
    with pytest.raises(ValueError, match=rf"{gate_name}.*loop-variable qubit indices.*not supported"):
        slr_to_ast(Main(q, r, getattr(qb, gate_name)(*args)))


def _legacy_output(backend: str, program: Main):
    if backend == "gate_qasm":
        return "\n".join(op.gen("qasm") for op in program.ops)
    if backend == "qasm":
        generator = QASMGenerator(skip_headers=True, _internal=True)
        generator.generate_block(program)
        return "\n".join(generator.output)
    if backend == "qir":
        generator = QIRGenerator(_internal=True)
        generator.generate_block(program)
        return [line.strip() for line in generator.get_output().splitlines() if "call void" in line]
    if backend == "stim":
        generator = StimGenerator(_internal=True)
        generator.generate_block(program)
        return generator.get_circuit().to_tableau()
    generator = QuantumCircuitGenerator(_internal=True)
    generator.generate_block(program)
    return [
        (symbol, location) for symbol, locations, _ in generator.get_circuit().items() for location in sorted(locations)
    ]


@pytest.mark.parametrize("backend", ["stim", "quantum_circuit", "qir", "qasm", "gate_qasm"])
@pytest.mark.parametrize("tuple_form", [False, True])
def test_legacy_register_pairing(backend: str, tuple_form: bool) -> None:
    q, r = QReg("q", 3), QReg("r", 3)
    gate = qb.CX((q, r)) if tuple_form else qb.CX(q, r)
    actual = _legacy_output(backend, Main(q, r, gate))
    if backend in ("qasm", "gate_qasm") and actual == "cx q, r;":
        # OpenQASM 2 whole-register broadcast is also a valid lowering.
        return
    assert actual == _legacy_output(
        backend,
        Main(q, r, *(qb.CX(q[i], r[i]) for i in range(3))),
    )


@pytest.mark.parametrize("backend", ["stim", "quantum_circuit", "qir", "qasm", "gate_qasm"])
@pytest.mark.parametrize("shape", ["mismatch", "mixed", "repeat_register", "repeat_qubit", "too_many", "list"])
@pytest.mark.parametrize("tuple_form", [False, True])
def test_legacy_invalid_arguments(backend: str, shape: str, tuple_form: bool) -> None:
    q, r, s = QReg("q", 2), QReg("r", 2), QReg("s", 3)
    args = {
        "mismatch": (q, s),
        "mixed": (q[0], r),
        "repeat_register": (q, q),
        "repeat_qubit": (q[0], q[0]),
        "too_many": (q[0], q[1], r[0], r[1]),
        "list": ([q[0], r[0]],),
    }[shape]
    if tuple_form:
        args = (args,)
    with pytest.raises(ValueError, match="CX") as exc:
        _legacy_output(backend, Main(q, r, s, qb.CX(*args)))
    if shape == "mismatch":
        assert "q (size 2)" in str(exc.value)
        assert "s (size 3)" in str(exc.value)
    elif shape.startswith("repeat"):
        assert "q[0]" in str(exc.value)


@pytest.mark.parametrize("backend", ["stim", "quantum_circuit", "qir", "qasm", "gate_qasm"])
def test_legacy_explicit_tuple_applications(backend: str) -> None:
    q = QReg("q", 3)
    assert _legacy_output(backend, Main(q, qb.CX((q[0], q[1]), (q[1], q[2])))) == _legacy_output(
        backend,
        Main(q, qb.CX(q[0], q[1]), qb.CX(q[1], q[2])),
    )
