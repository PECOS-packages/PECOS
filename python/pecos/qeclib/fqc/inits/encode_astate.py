# Create a logical |A><A| = (1/2)(I + (1/sqrt(3) (X + Y + Z)
# |A> = cos beta |0> + exp(i pi/ 4) sin beta |1>, where cos(2beta) = 1/sqrt(3)
# |A> = RZ(theta) RY(phi) |0>, where theta = pi/4 and phi = arccos(1/sqrt(3))

from pecos.qeclib import phys
from pecos.qeclib.fqc.inits.encoding_circ import EncodingCircuit
from pecos.slr import Block, Comment, QReg


class EncodeNonFTAState(Block):
    def __init__(self, q: QReg):
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 5"
            raise Exception(msg)

        super().__init__()
        self.extend(
            Comment("===== non-FT prep of |A> ====="),
            # q[4] is input
            phys.Reset(q[4]),
            phys.RY[2.0 * 0.4776583090622547](q[4]),
            phys.RZ[2.0 * -2.748893571891069](q[4]),
            EncodingCircuit(q),
        )
