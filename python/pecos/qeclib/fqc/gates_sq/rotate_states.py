from pecos.qeclib.fqc.gates_sq.face_rots import F, Fdg
from pecos.qeclib.fqc.gates_sq.paulis import Z
from pecos.slr import Block, Comment, QReg


class RotateMinus(Block):
    """Rotates the logical |-X> state to another Pauli basis state."""

    def __init__(self, q: QReg, state: str) -> None:
        super().__init__()

        """Rotates FQC minus state to other Pauli basis states."""
        if state == "|->":
            self.extend(
                Comment("\nLogical |->: -X"),
            )
        elif state == "|+>":
            self.extend(
                Comment("\nLogical |+>: -X -> +X"),
                Z(q),
            )
        elif state == "|0>":
            self.extend(
                Comment("\nLogical |0>: -X -> +X -> +Z"),
                Z(q),
                Fdg(q),
            )
        elif state == "|1>":
            self.extend(
                Comment("\nLogical |1>: -X -> -Z"),
                Fdg(q),
            )
        elif state == "|+i>":
            self.extend(
                Comment("\nLogical |+i>: -X -> +X -> +Y"),
                Z(q),
                F(q),
            )
        elif state == "|-i>":
            self.extend(
                Comment("\nLogical |-i>: -X -> -Y"),
                F(q),
            )
        else:
            msg = f'The state "{state}" is not implemented!'
            raise NotImplementedError(msg)
