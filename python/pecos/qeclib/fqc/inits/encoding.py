from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.qeclib.fqc.gates_sq.rotate_states import RotateMinus
from pecos.qeclib.fqc.inits.ft_minus_encoding import EncodeFTMinus
from pecos.qeclib.fqc.inits.nonft_minus_encoding import EncodeNonFTMinus
from pecos.slr import Assign, Barrier, Block, Comment, Repeat

if TYPE_CHECKING:
    from pecos.slr import CReg, QReg


class BasisAssign(Block):
    def __init__(self, state: str, basis: CReg) -> None:
        super().__init__()

        # 0th bit represents X, 1st bit represents Z
        if state in ["|0>", "|1>", "|+i>", "|-i>"]:
            self.extend(
                Assign(basis[1], 1),
            )

        if state in ["|+>", "|->", "|+i>", "|-i>"]:
            self.extend(
                Assign(basis[0], 1),
            )


class FTInitFQC(Block):
    def __init__(
        self,
        state: str,
        q: QReg,
        a: QReg,
        basis: CReg,
        init_syn: CReg,
        init_done: CReg,
        init_out: CReg,
        *,
        rus: int = 1,
        reset_cregs: bool = True,
    ) -> None:
        super().__init__()

        self.extend(
            Comment("\n"),
            Comment("======= Begin Init ======="),
        )

        if reset_cregs:
            self.extend(
                Assign(init_syn, 0),
                Assign(init_done, 0),
            )
        self.extend(
            Repeat(rus).block(
                EncodeFTMinus(q, a, init_syn, init_out, init_done),
            ),
            RotateMinus(q, state),
            BasisAssign(state, basis),
            Barrier(q),
            Comment("======= End Init ======="),
            Comment("\n"),
        )


class NonFTInitFQC(Block):
    def __init__(
        self,
        state: str,
        q: QReg,
        basis: CReg | None = None,
    ):
        super().__init__()

        self.extend(
            Comment("\n"),
            Comment("======= Begin Init ======="),
            Barrier(q),
            EncodeNonFTMinus(q),
            RotateMinus(q, state),
        )

        if basis:
            self.extend(
                BasisAssign(state, basis),
            )

        self.extend(
            Barrier(q),
            Comment("======= End Init ======="),
            Comment("\n"),
        )
