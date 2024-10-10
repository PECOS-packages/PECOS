from pecos.qeclib import phys
from pecos.slr import Block, Comment, QReg


class X(Block):
    """
    Transversal logical Z.

    X -> X
    Z -> -Z

    Y -> -Y
    """

    def __init__(self, q: QReg):
        super().__init__(
            Comment("Transversal logical X = -ZXZII"),
            # phys.Z(q[0], q[2]),
            # phys.X(q[1]),
            phys.X(q),
        )


class Y(Block):
    """
    Transversal logical Z.

    X -> -X
    Z -> -Z

    Y -> Y
    """

    def __init__(self, q):
        super().__init__(
            Comment("Transversal logical Y = -XYXII"),
            # phys.X(q[0], q[2]),
            # phys.Y(q[1]),
            phys.Y(q),
        )


class Z(Block):
    """
    Transversal logical Z.

    X -> -X
    Z -> Z

    Y -> -Y
    """

    def __init__(self, q):
        super().__init__(
            Comment("Transversal logical Z = -YZYII"),
            # phys.Y(q[0], q[2]),
            # phys.Z(q[1]),
            phys.Z(q),
        )
