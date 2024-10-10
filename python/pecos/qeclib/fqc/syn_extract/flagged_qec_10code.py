from pecos.qeclib.fqc.syn_extract.flagged_syn_extract_10code import SynExtract10QubitCodeFlaggedPart
from pecos.qeclib.fqc.syn_extract.unflagged_syn_extract_10code import SynExtract10QubitCodeUnflaggedPart
from pecos.slr import Barrier, Block, CReg, QReg


class FlaggedQEC10QubitCode(Block):
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
    ):
        super().__init__(
            Barrier(q1, q2, a),
            SynExtract10QubitCodeFlaggedPart(q1, q2, a, check_scratch1, s, f, last_syn_tq),
            Barrier(q1, q2, a),
            SynExtract10QubitCodeUnflaggedPart(
                q1,
                q2,
                a,
                check_scratch1,
                syn_tq,
            ),
            Barrier(q1, q2, a),
        )
