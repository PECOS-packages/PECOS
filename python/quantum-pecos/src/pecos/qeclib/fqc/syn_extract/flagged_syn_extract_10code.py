from pecos.slr import CReg, QReg, util

# ruff: noqa: E501


class SynExtract10QubitCodeFlaggedPart:
    """Does the flagged 10 qubit code QEC."""

    def __init__(
        self,
        q1: QReg,
        q2: QReg,
        a: QReg,
        check_scratch1: CReg,
        s: CReg,
        f: CReg,
        last_syn_tq: CReg,
    ):
        self.q1 = q1
        self.q2 = q2
        self.a = a
        self.check_scratch1 = check_scratch1
        self.s = s
        self.f = f
        self.last_syn_tq = last_syn_tq

    def qasm(self):
        qasm = f"""
        // ======= Begin flagged 10 qubit code QEC =======
        {self.check_scratch1} = 0;

        // -YZXIZ IIXIX
        barrier {self.q1}, {self.q2}, {self.a};
        reset {self.a};
        h {self.a};
        cy {self.a[0]}, {self.q1[0]};
        barrier {self.a[0]}, {self.a[1]};
        cz {self.a[0]}, {self.a[1]};
        barrier {self.a[0]}, {self.a[1]};
        cz {self.a[0]}, {self.q1[1]};  // fa0
        barrier {self.a[0]}, {self.a[1]};
        cx {self.a[0]}, {self.q1[2]};  // fa1
        barrier {self.a[0]}, {self.a[1]};
        cz {self.a[0]}, {self.q1[4]};  // fa2
        barrier {self.a[0]}, {self.a[1]};
        cx {self.a[0]}, {self.q2[2]};  // fa3
        barrier {self.a[0]}, {self.a[1]};
        cz {self.a[0]}, {self.a[1]};
        barrier {self.a[0]}, {self.a[1]};
        cx {self.a[0]}, {self.q2[4]};
        h {self.a};
        x {self.a[0]};  // flip the outcome to 0 by default
        measure {self.a[0]} -> {self.s[0]};
        measure {self.a[1]} -> {self.f[0]};
        {self.s[0]} = {self.s[0]} ^ {self.last_syn_tq[0]};  // correction for previous syndromes
        {self.check_scratch1[0]} = {self.s[0]} | {self.f[0]};

        // -ZZZXI IIIII
        if({self.check_scratch1} == 0) barrier {self.q1}, {self.a};
        if({self.check_scratch1} == 0) reset {self.a};
        if({self.check_scratch1} == 0) h {self.a};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.q1[0]};
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.q1[1]};  // fb0
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.q1[2]};  // fb1
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cx {self.a[0]}, {self.q1[3]};
        if({self.check_scratch1} == 0) h {self.a};
        if({self.check_scratch1} == 0) x {self.a[0]};  // flip the outcome to 0 by default
        if({self.check_scratch1} == 0) measure {self.a[0]} -> {self.s[0]};
        if({self.check_scratch1} == 0) measure {self.a[1]} -> {self.f[1]};
        if({self.check_scratch1} == 0) {self.s[0]} = {self.s[0]} ^ {self.last_syn_tq[1]};  // correction for previous syndromes
        if({self.check_scratch1} == 0) {self.check_scratch1[1]} = {self.s[0]} | {self.f[1]};

        // -IXZZZ IIIII
        if({self.check_scratch1} == 0) barrier {self.q1}, {self.a};
        if({self.check_scratch1} == 0) reset {self.a};
        if({self.check_scratch1} == 0) h {self.a};
        if({self.check_scratch1} == 0) cx {self.a[0]}, {self.q1[1]};
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.q1[2]};  // fc0
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.q1[3]};  // fc1
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.q1[4]};
        if({self.check_scratch1} == 0) h {self.a};
        if({self.check_scratch1} == 0) x {self.a[0]};  // flip the outcome to 0 by default
        if({self.check_scratch1} == 0) measure {self.a[0]} -> {self.s[0]};
        if({self.check_scratch1} == 0) measure {self.a[1]} -> {self.f[2]};
        if({self.check_scratch1} == 0) {self.s[0]} = {self.s[0]} ^ {self.last_syn_tq[2]};  // correction for previous syndromes
        if({self.check_scratch1} == 0) {self.check_scratch1[2]} = {self.s[0]} | {self.f[2]};

        // -ZIXZY XIIIX
        if({self.check_scratch1} == 0) barrier {self.q1}, {self.q2}, {self.a};
        if({self.check_scratch1} == 0) reset {self.a};
        if({self.check_scratch1} == 0) h {self.a};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.q1[0]};
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cx {self.a[0]}, {self.q1[2]};  // fd0
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.q1[3]};  // fd1
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cy {self.a[0]}, {self.q1[4]};  // fd2
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cx {self.a[0]}, {self.q2[0]};  // fd3
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cx {self.a[0]}, {self.q2[4]};
        if({self.check_scratch1} == 0) h {self.a};
        if({self.check_scratch1} == 0) x {self.a[0]};  // flip the outcome to 0 by default
        if({self.check_scratch1} == 0) measure {self.a[0]} -> {self.s[0]};
        if({self.check_scratch1} == 0) measure {self.a[1]} -> {self.f[3]};
        if({self.check_scratch1} == 0) {self.s[0]} = {self.s[0]} ^ {self.last_syn_tq[3]};  // correction for previous syndromes
        if({self.check_scratch1} == 0) {self.check_scratch1[3]} = {self.s[0]} | {self.f[3]};

        // ....................................................

        // -IIZIZ YZXIZ -> -IIIII XXXYI
        if({self.check_scratch1} == 0) barrier {self.q2}, {self.a};
        if({self.check_scratch1} == 0) reset {self.a};
        if({self.check_scratch1} == 0) h {self.a};
        if({self.check_scratch1} == 0) cx {self.a[0]}, {self.q2[0]};
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cx {self.a[0]}, {self.q2[1]};  // fe0
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cx {self.a[0]}, {self.q2[2]};  // fe1
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cy {self.a[0]}, {self.q2[3]};
        if({self.check_scratch1} == 0) h {self.a};
        if({self.check_scratch1} == 0) x {self.a[0]};  // flip the outcome to 0 by default
        if({self.check_scratch1} == 0) measure {self.a[0]} -> {self.s[0]};
        if({self.check_scratch1} == 0) measure {self.a[1]} -> {self.f[4]};
        if({self.check_scratch1} == 0) {self.s[0]} = {self.s[0]} ^ {self.last_syn_tq[4]};  // correction for previous syndromes
        if({self.check_scratch1} == 0) {self.check_scratch1[4]} = {self.s[0]} | {self.f[4]};

        // -ZIIIZ ZZZXI
        if({self.check_scratch1} == 0) barrier {self.q1}, {self.q2}, {self.a};
        if({self.check_scratch1} == 0) reset {self.a};
        if({self.check_scratch1} == 0) h {self.a};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.q1[0]};
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.q1[4]};  // ff0
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.q2[0]};  // ff1
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.q2[1]};  // ff2
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.q2[2]};  // ff3
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cx {self.a[0]}, {self.q2[3]};
        if({self.check_scratch1} == 0) h {self.a};
        if({self.check_scratch1} == 0) x {self.a[0]};  // flip the outcome to 0 by default
        if({self.check_scratch1} == 0) measure {self.a[0]} -> {self.s[0]};
        if({self.check_scratch1} == 0) measure {self.a[1]} -> {self.f[5]};
        if({self.check_scratch1} == 0) {self.s[0]} = {self.s[0]} ^ {self.last_syn_tq[5]};  // correction for previous syndromes
        if({self.check_scratch1} == 0) {self.check_scratch1[5]} = {self.s[0]} | {self.f[5]};

        // -ZIZII IXZZZ
        if({self.check_scratch1} == 0) barrier {self.q1}, {self.q2}, {self.a};
        if({self.check_scratch1} == 0) reset {self.a};
        if({self.check_scratch1} == 0) h {self.a};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.q1[0]};
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.q1[2]};  // fg0
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cx {self.a[0]}, {self.q2[1]};  // fg1
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.q2[2]};  // fg2
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.q2[3]};  // fg3
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.q2[4]};
        if({self.check_scratch1} == 0) h {self.a};
        if({self.check_scratch1} == 0) x {self.a[0]};  // flip the outcome to 0 by default
        if({self.check_scratch1} == 0) measure {self.a[0]} -> {self.s[0]};
        if({self.check_scratch1} == 0) measure {self.a[1]} -> {self.f[6]};
        if({self.check_scratch1} == 0) {self.s[0]} = {self.s[0]} ^ {self.last_syn_tq[6]};  // correction for previous syndromes
        if({self.check_scratch1} == 0) {self.check_scratch1[6]} = {self.s[0]} | {self.f[6]};

        // -IIIII IYXXX
        if({self.check_scratch1} == 0) barrier {self.q2}, {self.a};
        if({self.check_scratch1} == 0) reset {self.a};
        if({self.check_scratch1} == 0) h {self.a};
        if({self.check_scratch1} == 0) cy {self.a[0]}, {self.q2[1]};
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cx {self.a[0]}, {self.q2[2]};  // fh0
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cx {self.a[0]}, {self.q2[3]};  // fh1
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cz {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) barrier {self.a[0]}, {self.a[1]};
        if({self.check_scratch1} == 0) cx {self.a[0]}, {self.q2[4]};
        if({self.check_scratch1} == 0) h {self.a};
        if({self.check_scratch1} == 0) x {self.a[0]};  // flip the outcome to 0 by default
        if({self.check_scratch1} == 0) measure {self.a[0]} -> {self.s[0]};
        if({self.check_scratch1} == 0) measure {self.a[1]} -> {self.f[7]};
        if({self.check_scratch1} == 0) {self.s[0]} = {self.s[0]} ^ {self.last_syn_tq[7]};  // correction for previous syndromes
        if({self.check_scratch1} == 0) {self.check_scratch1[7]} = {self.s[0]} | {self.f[7]};

        barrier {self.q1}, {self.q2}, {self.a};

        // ======= End flagged 10 qubit code QEC =======
        """
        return util.rm_white_space(qasm)
