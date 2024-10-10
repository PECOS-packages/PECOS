from pecos.slr import Bit, CReg, util

# ruff: noqa: E501


class FTMeasOutWasm:
    def __init__(
        self,
        check_scratch: CReg,
        ms_bit: Bit,
        destruct_meas_bit: Bit,
        xflip: CReg,
        zflip: CReg,
        basis: CReg,
        log_out: CReg,
        meas: CReg,
        last_syn: CReg,
        num_q: CReg,
    ) -> None:
        self.check_scratch = check_scratch
        self.ms_bit = ms_bit
        self.destruct_meas_bit = destruct_meas_bit
        self.xflip = xflip
        self.zflip = zflip
        self.basis = basis
        self.log_out = log_out
        self.meas = meas
        self.last_syn = last_syn
        self.num_q = num_q

    def qasm(self):
        qasm = f"""
                // WASM CHECK -----------------------------------

                {self.check_scratch} = 0;
                {self.check_scratch[0]} = {self.ms_bit};
                {self.check_scratch[1]} = {self.destruct_meas_bit};

                {self.zflip} = {self.basis[0]};
                {self.xflip} = {self.basis[1]};
                if({self.check_scratch}==2) {self.log_out} = meas_decoder({self.meas}, {self.last_syn}, {self.num_q}, {self.xflip}, {self.zflip});
                if({self.check_scratch}==3) {self.log_out} = meas_decoder_flag({self.meas}, {self.last_syn}, {self.num_q}, {self.xflip}, {self.zflip});
                """

        return util.rm_white_space(qasm)
