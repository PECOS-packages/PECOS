from pecos.slr.std.phys.metaclasses import SingleQubitUnitary


class TGate(SingleQubitUnitary): ...


T = TGate(qasm_sym="rz(pi/4)")
FZ = T


class TdgGate(SingleQubitUnitary): ...


Tdg = TdgGate(qasm_sym="rz(-pi/4)")
FZdg = Tdg

# TODO: SWAP, iSWAP, Toffoli (CCNOT, CCX, TOFF)
