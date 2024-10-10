from pecos.slr import CReg, util


class FdgFrameRot:
    """Syndromes and Pauli Frame for Fdg.

    Fdg = H; S; = SXdg; SYdg = SYdg; SZdg; = SZdg; SXdg;

    X -> Z -> Y -> X
    """

    def __init__(
        self,
        scratch: CReg,
        syn: CReg,
    ):
        self.scratch = scratch
        self.syn = syn

    def qasm(self):
        qasm = f"""
        // Frame rotations for F
        {self.scratch} = 0;
        if({self.syn[0]}==1) {self.scratch} = {self.scratch} ^ 10;
        if({self.syn[1]}==1) {self.scratch} = {self.scratch} ^ 14;
        if({self.syn[2]}==1) {self.scratch} = {self.scratch} ^ 7;
        if({self.syn[3]}==1) {self.scratch} = {self.scratch} ^ 5;
        {self.syn} = {self.scratch};
        """

        return util.rm_white_space(qasm)


class FFrameRot:
    """
    F = Sdg; H; = SX; SZ; = SY; SX; = SZ; SY;

    X -> Y -> Z -> X
    """

    def __init__(
        self,
        scratch: CReg,
        syn: CReg,
    ):
        self.scratch = scratch
        self.syn = syn

    def qasm(self):
        qasm = f"""
        // Frame rotations for F
        {self.scratch} = 0;
        if({self.syn[0]}==1) {self.scratch} = {self.scratch} ^ 11;
        if({self.syn[1]}==1) {self.scratch} = {self.scratch} ^ 12;
        if({self.syn[2]}==1) {self.scratch} = {self.scratch} ^ 3;
        if({self.syn[3]}==1) {self.scratch} = {self.scratch} ^ 13;
        {self.syn} = {self.scratch};
        """

        return util.rm_white_space(qasm)
