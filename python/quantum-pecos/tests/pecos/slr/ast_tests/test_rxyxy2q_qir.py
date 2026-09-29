# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0

"""Pin the unsupported QIR boundary until qir-qis defines an RPP entry point."""

from dataclasses import dataclass

import pytest
from pecos.slr import Main, QReg, rad
from pecos.slr.ast.codegen import ast_to_qir
from pecos.slr.ast.nodes import AllocatorDecl, GateOp, LiteralExpr, Program, SlotRef
from pecos.slr.gen_codes.gen_qir import QIRGenerator
from pecos.slr.qeclib.qubit.qgate_base import TQGate


class RXYXY2QGate(TQGate):
    """Represent the pending gate without extending out-of-scope SLR gate lists."""

    has_parameters = True
    num_params = 2


@dataclass(frozen=True)
class PendingAstGate:
    """Gate metadata until the separate SLR gate inventory gains RXYXY2Q."""

    name: str = "RXYXY2Q"
    arity: int = 2
    is_parameterized: bool = True


@pytest.mark.parametrize("generator", ["ast", "legacy"])
def test_rxyxy2q_qir_fails_loudly(generator: str) -> None:
    """Flip this test when an upstream QIR RPP entry point is available."""
    if generator == "ast":
        program = Program(
            name="pending_rxyxy2q",
            declarations=(AllocatorDecl(name="q", capacity=2),),
            body=(
                GateOp(
                    gate=PendingAstGate(),
                    targets=(SlotRef(allocator="q", index=0), SlotRef(allocator="q", index=1)),
                    params=(LiteralExpr(value=rad(0.73)), LiteralExpr(value=rad(-0.41))),
                ),
            ),
        )
        with pytest.raises(NotImplementedError, match=r"RXYXY2Q.*no QIR lowering"):
            ast_to_qir(program)
        return
    qubits = QReg("q", 2)
    gate = RXYXY2QGate()(rad(0.73), rad(-0.41), qubits[0], qubits[1])
    program = Main(qubits, gate)
    with pytest.raises(KeyError, match="RXYXY2Q"):
        QIRGenerator(_internal=True).generate_block(program)
