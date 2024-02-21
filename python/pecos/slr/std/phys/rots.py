from pecos.slr.std.phys.metaclasses import SingleQubitUnitary, TwoQubitUnitary


class RXGate(SingleQubitUnitary): ...


RX = RXGate(qasm_sym="rx")


class RYGate(SingleQubitUnitary): ...


RY = RYGate(qasm_sym="ry")


class RZGate(SingleQubitUnitary): ...


RZ = RZGate(qasm_sym="rz")


class RZZGate(TwoQubitUnitary): ...


RZZ = RZZGate(qasm_sym="rzz")
