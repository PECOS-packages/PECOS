from pecos.slr.std.phys.metaclasses import SQCliffordGate


class SXGate(SQCliffordGate):
    """
    X -> X
    Z -> -Y
    Y -> Z
    """


SX = SXGate(qasm_sym="rx(pi/2)")


class SYGate(SQCliffordGate): ...


SY = SYGate(qasm_sym="ry(pi/2)")


class SZGate(SQCliffordGate): ...


S = SZ = SZGate(qasm_sym="rz(pi/2)")


class SXdgGate(SQCliffordGate): ...


SXdg = SXdgGate(qasm_sym="rx(-pi/2)")


class SYdgGate(SQCliffordGate): ...


SYdg = SYdgGate(qasm_sym="ry(-pi/2)")


class SZdgGate(SQCliffordGate): ...


Sdg = SZdg = SZdgGate(qasm_sym="rz(-pi/2)")
