from pecos.qeclib.fqc.gates_tq.szz_round_robin_pieces import RoundRobinSZZPiece1, RoundRobinSZZPiece2
from pecos.slr import Block, QReg


class NonFTSZZ(Block):
    """Does the second half of the round-robin logical SZZ gate."""

    def __init__(self, q1: QReg, q2: QReg):
        super().__init__()
        self.extend(
            RoundRobinSZZPiece1(q1, q2),
            RoundRobinSZZPiece2(q1, q2),
        )
