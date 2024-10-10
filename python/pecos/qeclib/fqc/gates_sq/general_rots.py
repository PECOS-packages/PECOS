from pecos.qeclib import phys
from pecos.slr import Block, Comment, QReg


class RZ(Block):
    """
    Non-FT rotation about Z

    see Fig. 4 in arXiv:1603.03948
    """

    def __init__(self, q: QReg, angle: float):
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 5"
            raise Exception(msg)

        super().__init__(
            Comment("Logical RZ"),
            phys.Fdg(q[0], q[4]),
            phys.Y(q[2]),
            phys.CX((q[4]), q[2]),
            phys.CX((q[0]), q[2]),
            phys.RZ[angle](q[2]),
            phys.CX((q[0]), q[2]),
            phys.CX((q[4]), q[2]),
            phys.Y(q[2]),
            phys.F(q[0], q[4]),
        )
