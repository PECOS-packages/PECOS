"""Tests for the SLR v2 typed-angle API (`rad(...)` / `turns(...)`).

Covers: construction + conversions (delegating to `pecos.angle64`),
pretty-print unit provenance, canonicalization semantics, AST JSON
round-trip with the unit preserved, and backend lowering (QASM radians,
Guppy half-turns).
"""

from __future__ import annotations

import math

import pytest
from pecos.slr import Angle, Main, QReg, generate, rad, turns
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.nodes import GateKind, GateOp, LiteralExpr
from pecos.slr.ast.pretty_print import pretty_print
from pecos.slr.ast.serialize import ast_to_json, json_to_ast
from pecos.slr.qeclib import qubit as qb


def test_rad_turns_construct_angle_over_angle64() -> None:
    a = rad(0.5)
    assert isinstance(a, Angle)
    assert a.source_unit == "rad"
    assert a.value.to_radians() == pytest.approx(0.5)

    t = turns(0.25)
    assert t.source_unit == "turns"
    assert t.value.to_radians() == pytest.approx(math.pi / 2)
    assert t.value.to_half_turns() == pytest.approx(0.5)


def test_rad_and_turns_agree_for_equivalent_angles() -> None:
    # turns(0.25) == rad(pi/2): same underlying fixed-point fraction.
    assert turns(0.25).value == rad(math.pi / 2).value


def test_pretty_print_round_trips_source_unit() -> None:
    prog = Main(q := QReg("q", 1), qb.RX(rad(0.5), q[0]), qb.RZ(turns(0.25), q[0]))
    text = pretty_print(slr_to_ast(prog))
    assert "qb.RX(rad(0.5), q[0])" in text
    assert "qb.RZ(turns(0.25), q[0])" in text


def test_canonicalization_full_turn_is_identity() -> None:
    # angle64 wraps mod a full turn.
    assert turns(1.0).value == turns(0.0).value
    assert rad(2 * math.pi).value == rad(0.0).value


def test_negative_angle_signed_round_trip() -> None:
    # Signed display keeps ordinary negatives readable.
    assert rad(-0.5).slr_repr() == "rad(-0.5)"
    assert rad(-0.5).value.to_radians_signed() == pytest.approx(-0.5)


def test_ast_json_round_trip_preserves_unit_and_value() -> None:
    prog = Main(q := QReg("q", 1), qb.RZ(turns(0.125), q[0]))
    ast = slr_to_ast(prog)
    restored = json_to_ast(ast_to_json(ast))
    # JSON is stable and the angle literal survives with its unit.
    assert ast_to_json(restored) == ast_to_json(ast)
    gate = next(s for s in restored.body if isinstance(s, GateOp) and s.gate == GateKind.RZ)
    angle = gate.params[0]
    assert isinstance(angle, LiteralExpr)
    assert isinstance(angle.value, Angle)
    assert angle.value == turns(0.125)
    assert "_angle" in ast_to_json(ast)


def test_qasm_lowers_to_signed_radians() -> None:
    prog = Main(q := QReg("q", 1), qb.RX(rad(0.5), q[0]), qb.RZ(turns(0.25), q[0]))
    qasm = generate(prog, "qasm")
    assert "rx(0.5) q[0];" in qasm
    # turns(0.25) == pi/2 radians.
    assert f"rz({math.pi / 2}) q[0];" in qasm


def test_guppy_lowers_to_half_turns() -> None:
    prog = Main(q := QReg("q", 1), qb.RZ(turns(0.25), q[0]))
    guppy = generate(prog, "guppy")
    # turns(0.25) == 0.5 half-turns; Guppy `angle` is half-turn based.
    assert "rz(q_0, angle(0.5))" in guppy


def test_bare_float_rejected_but_typed_angle_accepted() -> None:
    q = QReg("q", 1)
    with pytest.raises(TypeError, match="bare numeric angle"):
        qb.RX(0.5, q[0])
    # The typed form constructs fine.
    g = qb.RX(rad(0.5), q[0])
    assert isinstance(g.params[0], Angle)


def test_construction_arity_guard() -> None:
    # A parameterized call with too few qubits must fail
    # at construction, not survive to codegen.
    q = QReg("q", 2)
    with pytest.raises(TypeError, match="needs at least one qubit"):
        qb.RX(rad(0.5))  # no qubit
    with pytest.raises(TypeError, match="needs at least 2 qubit"):
        qb.RZZ(rad(0.5), q[0])  # one short for a 2q gate
    with pytest.raises(TypeError, match="needs at least 2 qubit"):
        qb.CRZ(rad(0.5), q[0])
    # Valid forms still construct: whole-register broadcast + parallel.
    assert qb.RZZ(rad(0.5), q) is not None
    assert qb.RX(rad(0.5), q[0], q[1]) is not None
    assert qb.RZZ(rad(0.5), q[0], q[1]) is not None


def test_direct_ast_float_param_rejected_uniformly_across_backends() -> None:
    # A malformed direct-AST GateOp carrying a bare-float
    # angle (bypassing the SLR call guard) must fail loud in EVERY backend,
    # not just Guppy -- the backends must not diverge.
    from dataclasses import replace

    from pecos.slr.ast.codegen import generate as ast_generate
    from pecos.slr.ast.codegen.guppy import GuppyCodegenError

    prog = Main(q := QReg("q", 1), qb.RX(rad(0.5), q[0]))
    good = slr_to_ast(prog)
    rx = good.body[0]
    assert isinstance(rx, GateOp)
    assert rx.gate == GateKind.RX
    bad = replace(good, body=(replace(rx, params=(LiteralExpr(value=0.5),)),))

    for target in ("qir", "qasm", "quantum_circuit"):
        with pytest.raises(NotImplementedError, match="requires typed `Angle`"):
            ast_generate(bad, target)
    with pytest.raises(GuppyCodegenError, match="requires a typed `Angle`"):
        ast_generate(bad, "guppy")
