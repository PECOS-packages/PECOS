from pecos.slr.std.phys.metaclasses import SQCliffordGate


class FGate(SQCliffordGate):
    """Face  rotation.

    Sdg; H; = SX; SZ; = SY; SX; = SZ; SY;

    X -> Y -> Z -> X

    X -> Y
    Z -> X
    Y -> Z
    """

    def qasm(self):
        str_list = []
        for q in self.qargs:
            str_list.append(f"rx(pi/2) {str(q)};\nrz(pi/2) {str(q)};")

        return " ".join(str_list)


F = FGate("F")


class FdgGate(SQCliffordGate):
    """Adjoint of the face  rotations.

    H; S; = SXdg; SYdg = SYdg; SZdg; = SZdg; SXdg;

    X -> Z -> Y -> X

    X -> Z
    Z -> Y
    Y -> X
    """

    def qasm(self):
        str_list = []
        for q in self.qargs:
            str_list.append(f"ry(-pi/2) {str(q)};\nrz(-pi/2) {str(q)};")

        return " ".join(str_list)


Fdg = FdgGate("Fdg")


class F4Gate(SQCliffordGate):
    """Face  4 rotation.

    H; Sdg = SX; SYdg = SYdg; SZ; = SZ; SX;

    X -> Z -> -Y -> X

    X -> Z
    Z -> -Y
    Y -> -X
    """

    def qasm(self):
        str_list = []
        for q in self.qargs:
            str_list.append(f"ry(-pi/2) {str(q)};\nrz(pi/2) {str(q)};")

        return " ".join(str_list)


F4 = FdgGate("F4")


class F4dgGate(SQCliffordGate):
    """Adjoint of the face 4  rotation.

    S; H; = SY; SXdg = SZdg; SY; = SXdg; SZdg;

    X -> -Y -> Z -> X

    X -> -Y
    Z -> X

    Y -> -Z
    """

    def qasm(self):
        str_list = []
        for q in self.qargs:
            str_list.append(f"rx(-pi/2) {str(q)};\nrz(-pi/2) {str(q)};")

        return " ".join(str_list)


F4dg = F4dgGate("F4dg")
