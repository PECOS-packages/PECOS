from pecos.qeclib.fqc.frame_rots.face_frame_rots import FdgFrameRot, FFrameRot
from pecos.qeclib.fqc.gates_sq.face_rots import F, Fdg
from pecos.slr import Block, Comment, CReg, QReg

# ruff: noqa: E501


class RotMeasBasisY(Block):
    """Rotates state and syndromes to Y basis."""

    def __init__(self, q: QReg, scratch: CReg, syn: CReg) -> None:
        super().__init__(
            Comment(
                "Measurement basis: Y\n\nFdg: X -> Z -> Y -> X\nRotate: Y -> X\ncorr: X -> Z, Z -> Y (correction anticom preserved)",
            ),
            Fdg(q),
            FdgFrameRot(scratch, syn),
        )


class RotMeasBasisZ(Block):
    """Rotates state and syndromes to Z basis."""

    def __init__(self, q: QReg, scratch: CReg, syn: CReg) -> None:
        super().__init__(
            Comment(
                "Measurement basis: Z\n\nF: X -> Y -> Z -> X\nRotate: Z -> X\ncorr: X -> Y (correction anticom preserved))",
            ),
            F(q),
            FFrameRot(scratch, syn),
        )


class RotMeasureBasis(Block):
    """Rotates the measurement basis to the appropriate Pauli basis."""

    def __init__(self, q: QReg, meas_basis: str, scratch: CReg, syn: CReg) -> None:
        super().__init__()

        self.extend(
            Comment("\n======= Begin measure basis rotation ======="),
        )

        if meas_basis == "X":
            self.extend(
                Comment("Already measuring in X basis"),
            )
        elif meas_basis == "Y":
            self.extend(
                RotMeasBasisY(q, scratch, syn),
            )
        elif meas_basis == "Z":
            self.extend(
                RotMeasBasisZ(q, scratch, syn),
            )
        else:
            msg = f'Measurement basis "{meas_basis}" not implemented!'
            raise NotImplementedError(msg)

        self.extend(
            Comment("======= End measure basis rotation =======\n"),
        )
