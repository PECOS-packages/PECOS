from pecos.qeclib.phys import CZ, H, Reset
from pecos.slr import Block, Comment, QReg


class EncodeNonFTMinus(Block):
    """Create a logical minus state using the encoding circuit in a non-FT manner."""

    def __init__(self, q: QReg):
        super().__init__(
            Comment("Init |->"),
            Reset(q),
            H(q),
            CZ(q[0], q[1]),
            CZ(q[2], q[3]),
            CZ(q[1], q[2]),
            CZ(q[3], q[4]),
            CZ(q[0], q[4]),
        )
