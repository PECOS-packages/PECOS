from pecos.qeclib import phys
from pecos.slr import Block, Comment, QReg


class RoundRobinSZZPiece1(Block):
    """Does the first half of the round-robin logical SZZ gate."""

    def __init__(self, q1: QReg, q2: QReg):
        super().__init__()
        self.extend(
            Comment("======= Begin CX part 1 ======="),
            phys.Fdg(q1[0], q1[4], q2[0], q2[4]),
            phys.Y(q1[2], q2[2]),
            phys.SZZ((q1[0], q2[0]), (q1[2], q2[2]), (q1[4], q2[4])),
            phys.SZZ((q1[0], q2[4]), (q1[2], q2[0]), (q1[4], q2[2])),
            Comment("======= End CX part 1 ======="),
        )


class RoundRobinSZZPiece2(Block):
    """Does the second half of the round-robin logical SZZ gate."""

    def __init__(self, q1: QReg, q2: QReg):
        super().__init__()
        self.extend(
            Comment("======= Begin CX part 2 ======="),
            phys.SZZ((q1[0], q2[2]), (q1[2], q2[4]), (q1[4], q2[0])),
            phys.Y(q1[2], q2[2]),
            phys.F(q1[0], q1[4], q2[0], q2[4]),
            Comment("======= End CX part 2 ======="),
        )
