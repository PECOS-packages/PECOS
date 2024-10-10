from pecos.qeclib.fqc.decoders.cx_decoding import CXWasmDecoding, RotateCXSyndromes
from pecos.qeclib.fqc.gates_tq.cx_round_robin_pieces import RoundRobinCXPiece1, RoundRobinCXPiece2
from pecos.qeclib.fqc.syn_extract.flagged_qec_10code import FlaggedQEC10QubitCode
from pecos.qeclib.fqc.syn_extract.unflagged_syn_extract_10code import SynExtract10QubitCodeUnflaggedPart
from pecos.slr import Assign, Barrier, Block, CReg, QReg


class PieceableFTCX(Block):
    """Pieceable FT CX gate with QEC gadget in the middle of the two pieces."""

    def __init__(
        self,
        q1: QReg,
        q2: QReg,
        a: QReg,
        check_scratch1: CReg,
        s: CReg,
        f: CReg,
        last_syn_tq: CReg,
        syn_tq: CReg,
        last_syn1: CReg,  # size 4
        last_syn2: CReg,  # size 4
    ):
        super().__init__(
            Barrier(q1, q2, a),
            RoundRobinCXPiece1(q1, q2),
            Barrier(q1, q2, a),
            FlaggedQEC10QubitCode(q1, q2, a, check_scratch1, s, f, last_syn_tq, syn_tq),
            Barrier(q1, q2, a),
            RoundRobinCXPiece2(q1, q2),
            Barrier(q1, q2, a),
            # Process syndrome information for further decoding
            RotateCXSyndromes(last_syn1, last_syn2, syn_tq),
            CXWasmDecoding(syn_tq, f, check_scratch1),
        )


class PieceableFTUnflaggedSynCX(Block):
    """Pieceable FT CX gate with a single round of unflagged syndrome extraction in the middle of the two pieces."""

    def __init__(
        self,
        q1: QReg,
        q2: QReg,
        a: QReg,
        check_scratch1: CReg,
        syn_tq: CReg,
    ):
        super().__init__(
            Barrier(q1, q2, a),
            RoundRobinCXPiece1(q1, q2),
            Barrier(q1, q2, a),
            Assign(check_scratch1, 1),
            SynExtract10QubitCodeUnflaggedPart(q1, q2, a, check_scratch1, syn_tq),
            Barrier(q1, q2, a),
            RoundRobinCXPiece2(q1, q2),
            Barrier(q1, q2, a),
            # TODO: Add processing methods
        )


class RoundRobinCX(Block):
    def __init__(self, q1: QReg, q2: QReg) -> None:
        """A Round Robin CX without any correction in the middle."""

        super().__init__(
            RoundRobinCXPiece1(q1, q2),
            RoundRobinCXPiece2(q1, q2),
        )
