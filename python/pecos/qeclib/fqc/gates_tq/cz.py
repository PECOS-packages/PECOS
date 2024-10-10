from pecos.qeclib.fqc.gates_tq.cz_round_robin_pieces import RoundRobinCZPiece1, RoundRobinCZPiece2
from pecos.slr import Block, QReg


class RoundRobinCZ(Block):
    def __init__(self, q1: QReg, q2: QReg) -> None:
        """A Round Robin CZ without any correction in the middle."""

        super().__init__(
            RoundRobinCZPiece1(q1, q2),
            RoundRobinCZPiece2(q1, q2),
        )
