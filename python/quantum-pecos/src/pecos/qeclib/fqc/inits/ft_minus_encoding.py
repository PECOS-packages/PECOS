from pecos.qeclib.fqc.inits.nonft_minus_encoding import EncodeNonFTMinus
from pecos.qeclib.phys import CX, CZ, H, Measure, Reset
from pecos.slr import Assign, Barrier, Block, CReg, If, QReg


class EncodeFTMinus(Block):
    """
    Fault tolerantly encode a logical minus state using a scheme based on Chao and Reichardt's work: arXiv:1705.02329
    """

    def __init__(self, q: QReg, a: QReg, init_syn: CReg, init_out: CReg, init_done: CReg):
        # fmt: off
        super().__init__(

            If(init_syn == 0).Then(

                Barrier(q, a),

                EncodeNonFTMinus(q),

                Barrier(q, a),

                Reset(a),

                H(a[0]),  # prep ancilla
                CX(a[0], q[0]),  # First syn check
                Barrier(a[0], a[1]),
                CX(a[0], a[1]),
                Barrier(a[0], a[1]),
                CZ(a[0], q[1]),
                Barrier(a[0], a[1]),
                CX(a[0], a[1]),
                Barrier(a[0], a[1]),

                CZ(a[0], q[4]),
                H(a[0]),
                Measure(a) > init_out,
                Assign(init_syn, init_out),
                Barrier(q, a),
            ),

            # Second syndromes
            If(init_syn == 0).Then(

                Barrier(q, a),

                Reset(a),
                H(a[0]),
                CZ(a[0], q[0]),

                Barrier(a[0], a[1]),

                CX(a[0], a[1]),

                Barrier(a[0], a[1]),

                CX(a[0], q[1]),

                Barrier(a[0], a[1]),

                CX(a[0], a[1]),

                Barrier(a[0], a[1]),

                CZ(a[0], q[2]),
                H(a[0]),
                Measure(a) > init_out,
                Assign(init_syn, init_out),
                Barrier(q, a),
            ),

            # Third Syndromes
            If(init_syn == 0).Then(

                Barrier(q, a),

                Reset(a),
                H(a[0]),
                CX(a[0], q[3]),

                Barrier(a[0], a[1]),

                CX(a[0], a[1]),

                Barrier(a[0], a[1]),

                CZ(a[0], q[2]),

                Barrier(a[0], a[1]),

                CX(a[0], a[1]),

                Barrier(a[0], a[1]),

                CZ(a[0], q[4]),
                H(a[0]),
                Measure(a) > init_out,
                Assign(init_syn, init_out),
                Barrier(q, a),
            ),

            If(init_syn == 0).Then(Assign(init_done, 1)),

            # If we made it to this point and init_syn is still 0, we are done
            If(init_done == 1).Then(Assign(init_syn, 1)),
            # If we are done, set init_syn != 0 so we don't run init any more

            # Otherwise... syn != 0 and init_done == 0
            If(init_done == 0).Then(Assign(init_syn, 0)),  # If we aren't done, reset init_syn to do another round
            # ^ equiv. to: if(syn==0){ init_syn = 1; } else { init_syn=0; } # => if trivial don't run any more,
            # otherwise keep running

            If(init_done == 0).Then(Reset(q)),  # Reset to start over if failure

            Barrier(q, a),
        )
        # fmt: on
