from pypecos.slr.std.phys.metaclasses import SQCliffordGate


class HGate(SQCliffordGate): ...


H = HGate(qasm_sym="h")
