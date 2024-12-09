from pecos.slr import QReg, util


class RoundRobinCZPiece1:
    """Does the first half of the round-robin logical CZ gate."""

    def __init__(self, q1: QReg, q2: QReg):
        self.q1 = q1
        self.q2 = q2

    def qasm(self):
        qasm = f"""
        // ======= Begin CZ part 1 =======
        h {self.q1[0]};
        s {self.q1[0]};

        y {self.q1[2]};

        h {self.q1[4]};
        s {self.q1[4]};


        h {self.q2[0]};
        s {self.q2[0]};

        y {self.q2[2]};

        h {self.q2[4]};
        s {self.q2[4]};

        cz {self.q1[0]}, {self.q2[0]};
        cz {self.q1[2]}, {self.q2[2]};
        cz {self.q1[4]}, {self.q2[4]};
        cz {self.q1[0]}, {self.q2[4]};
        cz {self.q1[2]}, {self.q2[0]};
        cz {self.q1[4]}, {self.q2[2]};
        // ======= End CY part 1 =======
        """
        return util.rm_white_space(qasm)


class RoundRobinCZPiece2:
    """Does the second half of the round-robin logical CZ gate."""

    def __init__(self, q1: QReg, q2: QReg):
        self.q1 = q1
        self.q2 = q2

    def qasm(self):
        qasm = f"""
        // ======= Begin CZ part 2 =======
        cz {self.q1[0]}, {self.q2[2]};
        cz {self.q1[2]}, {self.q2[4]};
        cz {self.q1[4]}, {self.q2[0]};

        sdg {self.q1[0]};
        h {self.q1[0]};

        y {self.q1[2]};

        sdg {self.q1[4]};
        h {self.q1[4]};


        sdg {self.q2[0]};
        h {self.q2[0]};

        y {self.q2[2]};

        sdg {self.q2[4]};
        h {self.q2[4]};
        // ======= End CZ part 2 =======
        """
        return util.rm_white_space(qasm)
