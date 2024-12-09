from pecos.qeclib import phys
from pecos.slr import Block, Comment


class F(Block):
    """Logical face  rotation.

    Sdg; H; = SX; SZ; = SY; SX; = SZ; SY;

    X -> Y -> Z -> X

    X -> Y
    Z -> X
    Y -> Z
    """

    def __init__(self, q):
        super().__init__(
            Comment("Transversal logical face rotation"),
            phys.F(q),
        )


class Fdg(Block):
    """Logical face  rotations.

    H; S; = SXdg; SYdg = SYdg; SZdg; = SZdg; SXdg;

    X -> Z -> Y -> X

    X -> Z
    Z -> Y
    Y -> X
    """

    def __init__(self, q):
        super().__init__(
            Comment("Transversal logical adjoint of face rotation"),
            phys.Fdg(q),  # TODO: replace this with the weight 3 operator
        )
