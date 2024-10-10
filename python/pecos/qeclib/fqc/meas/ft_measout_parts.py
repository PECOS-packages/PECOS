from pecos.slr import Bit, CReg, QReg, util

# ruff: noqa: E501


class FTMeasOutPart1:
    def __init__(
        self,
        q: QReg,
        a: QReg,
        check_scratch: CReg,
        ms: list[Bit],
        ms_c: list[Bit],
        last_syn: CReg,
        flip_bit: Bit,
        stop_bit: Bit,
        destruct_meas_bit: Bit,
    ) -> None:
        self.q = q
        self.a = a
        self.check_scratch = check_scratch
        self.ms = ms
        self.ms_c = ms_c
        self.last_syn = last_syn
        self.flip_bit = flip_bit
        self.stop_bit = stop_bit
        self.destruct_meas_bit = destruct_meas_bit

    def qasm(self):
        qasm = f"""// ==== Part 1 ===

                {self.check_scratch} = 0;
                barrier {self.q}, {self.a};

                reset {self.a};
                h {self.a[0]};
                cz {self.a[0]}, {self.q[4]};

                barrier {self.q}, {self.a};
                cx {self.a[0]}, {self.a[1]};  // flagging
                barrier {self.q}, {self.a};

                cx {self.a[0]}, {self.q[0]};

                barrier {self.q}, {self.a};
                cx {self.a[0]}, {self.a[1]};  // flagging
                barrier {self.q}, {self.a};

                cz {self.a[0]}, {self.q[1]};
                h {self.a[0]};
                x {self.a[0]};

                // q1 syn 0, 1, 2  q2 syn 3, 4, 5
                // q1 flg 6, 7, 8  q2 flg 9, 10, 11
                measure {self.a[0]} -> {self.ms[0]};
                measure {self.a[1]} -> {self.ms[1]};
                {self.ms_c[0]} = {self.ms[0]} ^ {self.last_syn[0]} ^ {self.last_syn[1]} ^ {self.last_syn[2]} ^ {self.last_syn[3]} ^ {self.flip_bit};
                barrier {self.a}, {self.q};

                // jump to destructive measurement for q1 if flagged
                if({self.ms[1]}==1) {self.stop_bit} = 1;
                if({self.ms[1]}==1) {self.destruct_meas_bit} = 1;
                """

        return util.rm_white_space(qasm)


class FTMeasOutPart2:
    def __init__(
        self,
        q: QReg,
        a: QReg,
        check_scratch: CReg,
        ms: list[Bit],
        ms_c: list[Bit],
        last_syn: CReg,
        flip_bit: Bit,
        stop_bit: Bit,
        destruct_meas_bit: Bit,
        log_out: CReg,
    ) -> None:
        self.q = q
        self.a = a
        self.check_scratch = check_scratch
        self.ms = ms
        self.ms_c = ms_c
        self.last_syn = last_syn
        self.flip_bit = flip_bit
        self.stop_bit = stop_bit
        self.destruct_meas_bit = destruct_meas_bit
        self.log_out = log_out

    def qasm(self):
        qasm = f"""// ==== Part 2 ===

                if({self.stop_bit}==0) barrier {self.a}, {self.q};

                if({self.stop_bit}==0) reset {self.a};
                if({self.stop_bit}==0) h {self.a[0]};
                if({self.stop_bit}==0) cz {self.a[0]}, {self.q[0]};

                if({self.stop_bit}==0) barrier {self.q}, {self.a};
                if({self.stop_bit}==0) cx {self.a[0]}, {self.a[1]};  // flagging
                if({self.stop_bit}==0) barrier {self.q}, {self.a};

                if({self.stop_bit}==0) cx {self.a[0]}, {self.q[1]};

                if({self.stop_bit}==0) barrier {self.q}, {self.a};
                if({self.stop_bit}==0) cx {self.a[0]}, {self.a[1]};  // flagging
                if({self.stop_bit}==0) barrier {self.q}, {self.a};

                if({self.stop_bit}==0) cz {self.a[0]}, {self.q[2]};
                if({self.stop_bit}==0) h {self.a[0]};
                if({self.stop_bit}==0) x {self.a[0]};

                // q1 0, 1, 2  q2 3, 4, 5
                // ms 1, 7, 6
                if({self.stop_bit}==0) measure {self.a[0]} -> {self.ms[0]};
                if({self.stop_bit}==0) measure {self.a[1]} -> {self.ms[1]};
                if({self.stop_bit}==0) {self.ms_c[1]} = {self.ms[0]} ^ {self.last_syn[1]} ^ {self.last_syn[2]} ^ {self.flip_bit};
                if({self.stop_bit}==0) barrier {self.a}, {self.q};

                // if flagged, return previous result
                if({self.ms[1]}==1) {self.log_out[0]} = {self.ms_c[0]};
                if({self.ms[1]}==1) {self.stop_bit} = 1;
                if({self.ms[1]}==1) {self.destruct_meas_bit} = 0;

                // otherwise, if s1 != s2, data error, remeasure
                // not previously flagged
                if({self.ms[2]}==0) {self.check_scratch[0]} = {self.ms_c[0]} ^ {self.ms_c[1]};

                if({self.ms[1]}==1) {self.check_scratch[0]} = 0; // currently flagged
                if({self.check_scratch[0]}==1) {self.stop_bit} = 1;
                if({self.check_scratch[0]}==1) {self.destruct_meas_bit} = 1;
                """

        return util.rm_white_space(qasm)


class FTMeasOutPart3:
    def __init__(
        self,
        q: QReg,
        a: QReg,
        check_scratch: CReg,
        ms: list[Bit],
        ms_c: list[Bit],
        last_syn: CReg,
        flip_bit: Bit,
        stop_bit: Bit,
        destruct_meas_bit: Bit,
        log_out: CReg,
    ) -> None:
        self.q = q
        self.a = a
        self.check_scratch = check_scratch
        self.ms = ms
        self.ms_c = ms_c
        self.last_syn = last_syn
        self.flip_bit = flip_bit
        self.stop_bit = stop_bit
        self.destruct_meas_bit = destruct_meas_bit
        self.log_out = log_out

    def qasm(self):
        qasm = f"""
                // ==== Part 3 ===

                if({self.stop_bit}==0) barrier {self.a}, {self.q};

                if({self.stop_bit}==0) reset {self.a};
                if({self.stop_bit}==0) h {self.a[0]};
                if({self.stop_bit}==0) cz {self.a[0]}, {self.q[2]};

                if({self.stop_bit}==0) barrier {self.q}, {self.a};
                if({self.stop_bit}==0) cx {self.a[0]}, {self.a[1]};  // flagging
                if({self.stop_bit}==0) barrier {self.q}, {self.a};

                if({self.stop_bit}==0) cx {self.a[0]}, {self.q[3]};

                if({self.stop_bit}==0) barrier {self.q}, {self.a};
                if({self.stop_bit}==0) cx {self.a[0]}, {self.a[1]};  // flagging
                if({self.stop_bit}==0) barrier {self.q}, {self.a};

                if({self.stop_bit}==0) cz {self.a[0]}, {self.q[4]};
                if({self.stop_bit}==0) h {self.a[0]};
                if({self.stop_bit}==0) x {self.a[0]};

                // q1 0, 1, 2  q2 3, 4, 5
                if({self.stop_bit}==0) measure {self.a[0]} -> {self.ms[0]};
                if({self.stop_bit}==0) measure {self.a[1]} -> {self.ms[1]};
                if({self.stop_bit}==0) {self.ms_c[0]} = {self.ms[0]} ^ {self.last_syn[1]} ^ {self.last_syn[2]} ^ {self.last_syn[3]} ^ {self.flip_bit};
                if({self.stop_bit}==0) barrier {self.a}, {self.q};

                // if flagged, return previous result
                // if flag: return log:= s1 == s2
                if({self.ms[1]}==1) {self.log_out[0]} = {self.ms_c[1]};
                if({self.ms[1]}==1) {self.destruct_meas_bit} = 0;

                {self.check_scratch} = 0;

                // determine (s1==s2)==s3
                if({self.stop_bit}==0) {self.check_scratch[0]} = 1 ^ {self.ms[1]};  // only 1, if no stop & no flag
                if({self.check_scratch[0]}==1) {self.check_scratch[1]} = {self.ms_c[1]} ^ {self.ms_c[0]};

                // if no flag & s1==s2!=s3: destructive meas
                if({self.check_scratch}==3) {self.destruct_meas_bit} = 1;

                // if no flag & s1==s2==s3: log:= s1==s2==s3
                if({self.check_scratch}==1) {self.log_out[0]} = {self.ms_c[0]};
                if({self.check_scratch}==1) {self.destruct_meas_bit} = 0;
                """

        return util.rm_white_space(qasm)


class FTMeasOutPart4:
    def __init__(self, q: QReg, destruct_meas_bit: Bit, meas: CReg) -> None:
        self.q = q
        self.destruct_meas_bit = destruct_meas_bit
        self.meas = meas

    def qasm(self):
        qasm = f"""
                // ==== Part 4 ===
                if({self.destruct_meas_bit}==1) barrier {self.q};
                if({self.destruct_meas_bit}==1) cz {self.q[0]}, {self.q[4]};
                if({self.destruct_meas_bit}==1) barrier {self.q};
                if({self.destruct_meas_bit}==1) cz {self.q[1]}, {self.q[2]};
                if({self.destruct_meas_bit}==1) cz {self.q[3]}, {self.q[4]};
                if({self.destruct_meas_bit}==1) barrier {self.q};
                if({self.destruct_meas_bit}==1) cz {self.q[0]}, {self.q[1]};
                if({self.destruct_meas_bit}==1) cz {self.q[2]}, {self.q[3]};
                if({self.destruct_meas_bit}==1) h {self.q};
                if({self.destruct_meas_bit}==1) measure {self.q} -> {self.meas};
                if({self.destruct_meas_bit}==1) barrier {self.q};
                """

        return util.rm_white_space(qasm)
