from pecos.qeclib import phys
from pecos.slr import QASM, Bit, Block, Comment, CReg, QReg


class NonFTMeasX(Block):
    def __init__(self, q: QReg, out: CReg, log_out: Bit) -> None:
        super().__init__(
            Comment("Measure logical X = -YIXIY"),
            phys.SX(q[0], q[4]),
            phys.SYdg(q[2]),
            phys.Measure(q[0]) > out[0],
            phys.Measure(q[2]) > out[1],
            phys.Measure(q[4]) > out[2],
            # Assign(log_out, out[0] ^ out[1] ^ out[2] ^ 1),
            QASM(f"{log_out} = {out[0]} ^ {out[1]} ^ {out[2]} ^ 1;"),
        )


class NonFTMeasY(Block):
    def __init__(self, q: QReg, out: CReg, log_out: Bit) -> None:
        super().__init__(
            Comment("Measure logical Y = -ZIYIZ"),
            phys.SX(q[2]),
            phys.Measure(q[0]) > out[0],
            phys.Measure(q[2]) > out[1],
            phys.Measure(q[4]) > out[2],
            # Assign(log_out, out[0] ^ out[1] ^ out[2] ^ 1),
            QASM(f"{log_out} = {out[0]} ^ {out[1]} ^ {out[2]} ^ 1;"),
        )


class NonFTMeasZ(Block):
    def __init__(self, q: QReg, out: CReg, log_out: Bit) -> None:
        super().__init__(
            Comment("Measure logical Z = -XIZIX"),
            phys.SYdg(q[0], q[4]),
            phys.Measure(q[0]) > out[0],
            phys.Measure(q[2]) > out[1],
            phys.Measure(q[4]) > out[2],
            # Assign(log_out, out[0] ^ out[1] ^ out[2] ^ 1),
            QASM(f"{log_out} = {out[0]} ^ {out[1]} ^ {out[2]} ^ 1;"),
        )


class NonFTMeas(Block):
    def __init__(self, q: QReg, out: CReg, log_out: Bit, meas_basis: str) -> None:
        super().__init__()

        if meas_basis == "X":
            self.extend(NonFTMeasX(q, out, log_out))
        elif meas_basis == "Y":
            self.extend(NonFTMeasY(q, out, log_out))
        elif meas_basis == "Z":
            self.extend(NonFTMeasZ(q, out, log_out))
        else:
            msg = f"Measurement basis '{meas_basis}' not supported!"
            raise NotImplementedError(msg)


class NonFTMeasXWith2Logicals(Block):
    def __init__(self, q: QReg, out: CReg, log_out: Bit, log_out2: Bit) -> None:
        if len(out.elems) != 5:
            msg = "Expected a CReg of width 5."
            raise Exception(msg)

        # -YIXIY
        # -IXIYY => YXXYY stab: YXXYI
        # -IZXZI => YZXZY stab: YZIZY
        # -YYIXI => YYXXY stab: IYXXY

        super().__init__(
            Comment("Measure logical X = -YIXIY"),
            Comment("Measure logical X = -IZXZI"),
            phys.SX(q[0], q[4]),
            phys.SYdg(q[2]),
            phys.Measure(q[0]) > out[0],
            phys.Measure(q[1]) > out[1],
            phys.Measure(q[2]) > out[2],
            phys.Measure(q[3]) > out[3],
            phys.Measure(q[4]) > out[4],
            QASM(f"{log_out} = {out[0]} ^ {out[2]} ^ {out[4]} ^ 1;"),
            QASM(f"{log_out2} = {out[1]} ^ {out[2]} ^ {out[3]} ^ 1;"),
        )


class NonFTMeasYWith2Logicals(Block):
    def __init__(self, q: QReg, out: CReg, log_out: Bit, log_out2: Bit) -> None:
        if len(out.elems) != 5:
            msg = "Expected a CReg of width 5."
            raise Exception(msg)

        # -ZIYIZ
        # -IXYXI

        super().__init__(
            Comment("Measure logical Y = -ZIYIZ"),
            Comment("Measure logical Y = -IXYXI"),
            phys.SX(q[2]),  # Y -> Z (X->X, Z->-Y)
            phys.SYdg(q[1], q[3]),  # X -> Z (Y->Y, Z->-X)
            phys.Measure(q[0]) > out[0],
            phys.Measure(q[1]) > out[1],
            phys.Measure(q[2]) > out[2],
            phys.Measure(q[3]) > out[3],
            phys.Measure(q[4]) > out[4],
            QASM(f"{log_out} = {out[0]} ^ {out[2]} ^ {out[4]} ^ 1;"),
            QASM(f"{log_out2} = {out[1]} ^ {out[2]} ^ {out[3]} ^ 1;"),
        )


class NonFTMeasZWith2Logicals(Block):
    def __init__(self, q: QReg, out: CReg, log_out: Bit, log_out2: Bit) -> None:
        # -XIZIX
        #  IXZZX 0100
        #  XYIYX 1100
        #  XZZXI 1000

        # log
        # -XIZIX 1101
        # -IYZYI 0001 stab XYIYX 1100
        # -IZIXX 0101 stab XZZXI 1000
        # -XXIZI 1001 stab IXZZX 0100

        # -IYZYI => XYZYX
        # -IZIXX => XZZXX
        # -XXIZI => XXZZX

        # measure -XIZIX & -IYZYI

        if len(out.elems) != 5:
            msg = "Expected a CReg of width 5."
            raise Exception(msg)

        super().__init__(
            Comment("Measure logical Z = -XIZIX"),
            Comment("Measure logical Z = -IYZYI"),
            phys.SYdg(q[0], q[4]),  # X -> Z (Y->Y, Z->-X)
            phys.SX(q[1], q[3]),  # Y -> Z (X->X, Z->-Y)
            phys.Measure(q[0]) > out[0],
            phys.Measure(q[1]) > out[1],
            phys.Measure(q[2]) > out[2],
            phys.Measure(q[3]) > out[3],
            phys.Measure(q[4]) > out[2],
            # Assign(log_out, out[0] ^ out[1] ^ out[2] ^ 1),
            QASM(f"{log_out} = {out[0]} ^ {out[2]} ^ {out[4]} ^ 1;"),
            QASM(f"{log_out2} = {out[1]} ^ {out[2]} ^ {out[3]} ^ 1;"),
        )


class NonFTMeasWith2Logicals(Block):
    def __init__(self, q: QReg, out: CReg, log_out: Bit, log_out2: Bit, meas_basis: str) -> None:
        super().__init__()

        if meas_basis == "X":
            self.extend(NonFTMeasXWith2Logicals(q, out, log_out, log_out2))
        elif meas_basis == "Y":
            self.extend(NonFTMeasYWith2Logicals(q, out, log_out, log_out2))
        elif meas_basis == "Z":
            self.extend(NonFTMeasZWith2Logicals(q, out, log_out, log_out2))
        else:
            msg = f"Measurement basis '{meas_basis}' not supported!"
            raise NotImplementedError(msg)
