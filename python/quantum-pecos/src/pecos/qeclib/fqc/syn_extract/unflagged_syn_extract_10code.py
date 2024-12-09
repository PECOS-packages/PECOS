from pecos.slr import CReg, QReg, util


class SynExtract10QubitCodeUnflaggedPart:
    """Does the unflagged 10 qubit code QEC."""

    def __init__(
        self,
        q1: QReg,
        q2: QReg,
        a: QReg,
        check_scratch1: CReg,
        syn_tq: CReg,
    ):
        self.q1 = q1
        self.q2 = q2
        self.a = a
        self.check_scratch1 = check_scratch1
        self.syn_tq = syn_tq

    def qasm(self):
        qasm = f"""
        // ======= Begin unflagged 10 qubit code QEC =======
        barrier {self.q1}, {self.q2}, {self.a};

        // -YZXIZ IIXIX
        if({self.check_scratch1} > 0) barrier {self.q1}, {self.q2}, {self.a};
        if({self.check_scratch1} > 0) reset {self.a[0]};
        if({self.check_scratch1} > 0) h {self.a[0]};
        if({self.check_scratch1} > 0) cy {self.a[0]}, {self.q1[0]};
        if({self.check_scratch1} > 0) cz {self.a[0]}, {self.q1[1]};
        if({self.check_scratch1} > 0) cx {self.a[0]}, {self.q1[2]};
        if({self.check_scratch1} > 0) cz {self.a[0]}, {self.q1[4]};
        if({self.check_scratch1} > 0) cx {self.a[0]}, {self.q2[2]};
        if({self.check_scratch1} > 0) cx {self.a[0]}, {self.q2[4]};
        if({self.check_scratch1} > 0) h {self.a[0]};
        if({self.check_scratch1} > 0) x {self.a[0]};  // flip the outcome to 0 by default
        if({self.check_scratch1} > 0) measure {self.a[0]} -> {self.syn_tq[0]};

        // -ZZZXI IIIII
        if({self.check_scratch1} > 0) barrier {self.q1}, {self.a};
        if({self.check_scratch1} > 0) reset {self.a[1]};
        if({self.check_scratch1} > 0) h {self.a[1]};
        if({self.check_scratch1} > 0) cz {self.a[1]}, {self.q1[0]};
        if({self.check_scratch1} > 0) cz {self.a[1]}, {self.q1[1]};
        if({self.check_scratch1} > 0) cz {self.a[1]}, {self.q1[2]};
        if({self.check_scratch1} > 0) cx {self.a[1]}, {self.q1[3]};
        if({self.check_scratch1} > 0) h {self.a[1]};
        if({self.check_scratch1} > 0) x {self.a[1]};  // flip the outcome to 0 by default
        if({self.check_scratch1} > 0) measure {self.a[1]} -> {self.syn_tq[1]};

        // -IXZZZ IIIII
        if({self.check_scratch1} > 0) reset {self.a[0]};
        if({self.check_scratch1} > 0) h {self.a[0]};
        if({self.check_scratch1} > 0) cx {self.a[0]}, {self.q1[1]};
        if({self.check_scratch1} > 0) cz {self.a[0]}, {self.q1[2]};
        if({self.check_scratch1} > 0) cz {self.a[0]}, {self.q1[3]};
        if({self.check_scratch1} > 0) cz {self.a[0]}, {self.q1[4]};
        if({self.check_scratch1} > 0) h {self.a[0]};
        if({self.check_scratch1} > 0) x {self.a[0]};  // flip the outcome to 0 by default
        if({self.check_scratch1} > 0) measure {self.a[0]} -> {self.syn_tq[2]};

        // -ZIXZY XIIIX
        if({self.check_scratch1} > 0) barrier {self.q1}, {self.q2}, {self.a};
        if({self.check_scratch1} > 0) reset {self.a[1]};
        if({self.check_scratch1} > 0) h {self.a[1]};
        if({self.check_scratch1} > 0) cz {self.a[1]}, {self.q1[0]};
        if({self.check_scratch1} > 0) cx {self.a[1]}, {self.q1[2]};
        if({self.check_scratch1} > 0) cz {self.a[1]}, {self.q1[3]};
        if({self.check_scratch1} > 0) cy {self.a[1]}, {self.q1[4]};
        if({self.check_scratch1} > 0) cx {self.a[1]}, {self.q2[0]};
        if({self.check_scratch1} > 0) cx {self.a[1]}, {self.q2[4]};
        if({self.check_scratch1} > 0) h {self.a[1]};
        if({self.check_scratch1} > 0) x {self.a[1]};  // flip the outcome to 0 by default
        if({self.check_scratch1} > 0) measure {self.a[1]} -> {self.syn_tq[3]};

        // ....................................................

        // -IIZIZ YZXIZ -> -IIIII XXXYI
        if({self.check_scratch1} > 0) barrier {self.q2}, {self.a};
        if({self.check_scratch1} > 0) reset {self.a[0]};
        if({self.check_scratch1} > 0) h {self.a[0]};
        if({self.check_scratch1} > 0) cx {self.a[0]}, {self.q2[0]};
        if({self.check_scratch1} > 0) cx {self.a[0]}, {self.q2[1]};
        if({self.check_scratch1} > 0) cx {self.a[0]}, {self.q2[2]};
        if({self.check_scratch1} > 0) cy {self.a[0]}, {self.q2[3]};
        if({self.check_scratch1} > 0) h {self.a[0]};
        if({self.check_scratch1} > 0) x {self.a[0]};  // flip the outcome to 0 by default
        if({self.check_scratch1} > 0) measure {self.a[0]} -> {self.syn_tq[4]};

        // -ZIIIZ ZZZXI
        if({self.check_scratch1} > 0) barrier {self.q1}, {self.q2}, {self.a};
        if({self.check_scratch1} > 0) reset {self.a[1]};
        if({self.check_scratch1} > 0) h {self.a[1]};
        if({self.check_scratch1} > 0) cz {self.a[1]}, {self.q1[0]};
        if({self.check_scratch1} > 0) cz {self.a[1]}, {self.q1[4]};
        if({self.check_scratch1} > 0) cz {self.a[1]}, {self.q2[0]};
        if({self.check_scratch1} > 0) cz {self.a[1]}, {self.q2[1]};
        if({self.check_scratch1} > 0) cz {self.a[1]}, {self.q2[2]};
        if({self.check_scratch1} > 0) cx {self.a[1]}, {self.q2[3]};
        if({self.check_scratch1} > 0) h {self.a[1]};
        if({self.check_scratch1} > 0) x {self.a[1]};  // flip the outcome to 0 by default
        if({self.check_scratch1} > 0) measure {self.a[1]} -> {self.syn_tq[5]};

        // -ZIZII IXZZZ
        if({self.check_scratch1} > 0) barrier {self.q1}, {self.q2}, {self.a};
        if({self.check_scratch1} > 0) reset {self.a[0]};
        if({self.check_scratch1} > 0) h {self.a[0]};
        if({self.check_scratch1} > 0) cz {self.a[0]}, {self.q1[0]};
        if({self.check_scratch1} > 0) cz {self.a[0]}, {self.q1[2]};
        if({self.check_scratch1} > 0) cx {self.a[0]}, {self.q2[1]};
        if({self.check_scratch1} > 0) cz {self.a[0]}, {self.q2[2]};
        if({self.check_scratch1} > 0) cz {self.a[0]}, {self.q2[3]};
        if({self.check_scratch1} > 0) cz {self.a[0]}, {self.q2[4]};
        if({self.check_scratch1} > 0) h {self.a[0]};
        if({self.check_scratch1} > 0) x {self.a[0]};  // flip the outcome to 0 by default
        if({self.check_scratch1} > 0) measure {self.a[0]} -> {self.syn_tq[6]};

        // -IIIII IYXXX
        if({self.check_scratch1} > 0) barrier {self.q2}, {self.a};
        if({self.check_scratch1} > 0) reset {self.a[1]};
        if({self.check_scratch1} > 0) h {self.a[1]};
        if({self.check_scratch1} > 0) cy {self.a[1]}, {self.q2[1]};
        if({self.check_scratch1} > 0) cx {self.a[1]}, {self.q2[2]};
        if({self.check_scratch1} > 0) cx {self.a[1]}, {self.q2[3]};
        if({self.check_scratch1} > 0) cx {self.a[1]}, {self.q2[4]};
        if({self.check_scratch1} > 0) h {self.a[1]};
        if({self.check_scratch1} > 0) x {self.a[1]};  // flip the outcome to 0 by default
        if({self.check_scratch1} > 0) measure {self.a[1]} -> {self.syn_tq[7]};

        barrier {self.q1}, {self.q2}, {self.a};
        // ======= End unflagged 10 qubit code QEC =======
        """
        return util.rm_white_space(qasm)
