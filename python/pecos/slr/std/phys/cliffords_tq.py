from pecos.slr.std.phys.metaclasses import TQCliffordGate


class CXGate(TQCliffordGate): ...


CNOT = CX = CXGate(qasm_sym="cx")


class CYGate(TQCliffordGate): ...


CY = CYGate(qasm_sym="cy")


class CZGate(TQCliffordGate): ...


CZ = CZGate(qasm_sym="cz")


class CHGate(TQCliffordGate):
    # TODO: Is this actually a Clifford?
    ...


CH = CHGate(qasm_sym="ch")


class SXXGate(TQCliffordGate): ...


SXX = SXXGate()


class SYYGate(TQCliffordGate): ...


SYY = SYYGate()


class SZZGate(TQCliffordGate): ...


SZZ = SZZGate(qasm_sym="ZZ")


class SXXdgGate(TQCliffordGate): ...


SXXdg = SXXdgGate()


class SYYdgGate(TQCliffordGate): ...


SYYdg = SYYdgGate()


class SZZdgGate(TQCliffordGate): ...


SZZdg = SZZdgGate()
