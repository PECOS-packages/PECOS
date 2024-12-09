from pecos.qeclib.fqc.gates_tq.cy_round_robin_pieces import RoundRobinCYPiece1, RoundRobinCYPiece2
from pecos.slr import Block, QReg


class RoundRobinCY(Block):
    def __init__(self, q1: QReg, q2: QReg) -> None:
        """A Round Robin CY without any correction in the middle."""

        super().__init__(
            RoundRobinCYPiece1(q1, q2),
            RoundRobinCYPiece2(q1, q2),
        )
