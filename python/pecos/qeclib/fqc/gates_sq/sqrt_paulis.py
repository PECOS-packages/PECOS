from pecos.qeclib import phys
from pecos.slr import Block, Comment, QReg


class SX(Block):
    """
    Square root of X.

    X -> X
    Z -> -Y

    Y -> Z
    """

    def __init__(self, q: QReg):
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 5"
            raise Exception(msg)

        super().__init__(
            Comment("Transversal logical sqrt of X"),
            phys.SX(q),
        )


class SXdg(Block):
    """
    Hermitian adjoint of the square root of X.

    X -> X
    Z -> Y

    Y -> -Z
    """

    def __init__(self, q: QReg):
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 5"
            raise Exception(msg)

        super().__init__(
            Comment("Transversal logical adjoint of sqrt of X"),
            phys.SXdg,
        )


class SY(Block):
    """
    Square root of Y.

    X -> -Z
    Z -> X

    Y -> Y
    """

    def __init__(self, q: QReg):
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 5"
            raise Exception(msg)

        super().__init__(
            Comment("Transversal logical sqrt of Y"),
            phys.SY(q),
        )


class SYdg(Block):
    """
    Square root of X.

    X -> Z
    Z -> -X

    Y -> Y
    """

    def __init__(self, q: QReg):
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 5"
            raise Exception(msg)

        super().__init__(
            Comment("Transversal logical adjoint of sqrt of Y"),
            phys.SYdg(q),
        )


class SZ(Block):
    """
    Square root of Z. Also known as the S gate.
    diag(1, i)

    X -> Y
    Z -> Z

    Y -> -X
    """

    def __init__(self, q: QReg):
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 5"
            raise Exception(msg)

        super().__init__(
            Comment("Transversal logical sqrt of Z"),
            phys.SZ(q),
        )


class SZdg(Block):
    """
    Hermitian adjoint of the square root of Z. Also known as the Sdg gate.
    diag(1, -i)

    X -> -Y
    Z -> Z

    Y -> X
    """

    def __init__(self, q: QReg):
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 5"
            raise Exception(msg)

        super().__init__(
            Comment("Transversal logical adjoint of sqrt of Z"),
            phys.SZdg(q),
        )
