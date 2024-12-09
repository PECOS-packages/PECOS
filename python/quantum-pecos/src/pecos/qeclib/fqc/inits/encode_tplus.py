from numpy import pi

from pecos.qeclib.fqc.gates_sq.general_rots import RZ
from pecos.qeclib.fqc.inits.nonft_minus_encoding import EncodeNonFTMinus
from pecos.slr import Block, Comment, QReg


class EncodeNonFTTPlus(Block):
    """Create a logical minus state using the encoding circuit in a non-FT manner."""

    def __init__(self, q: QReg):
        super().__init__(
            Comment("Init T|+>"),
            EncodeNonFTMinus(q),
            RZ(q, angle=-pi / 4),
        )
