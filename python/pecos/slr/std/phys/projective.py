from pecos.slr.std.phys.metaclasses import NoParamsQGate


class ResetGate(NoParamsQGate):
    """Resetting a qubit to the zero state."""


Reset = ResetGate(qasm_sym="reset")


class MeasureGate(NoParamsQGate):
    """A measurement of a qubit in the Z basis."""

    def __init__(self):
        super().__init__("Measure Z", qasm_sym="measure")
        self.cout = None
        self.qasm_sym = "measure"

    def __gt__(self, cout):
        g = self.copy()

        if isinstance(cout, tuple):
            g.cout = cout
        else:
            g.cout = (cout,)

        return g

    def qasm(self):
        sym = self.qasm_sym

        str_list = []
        for q, c in zip(self.qargs, self.cout, strict=True):
            str_list.append(f"{sym} {str(q)} -> {c};")

        return " ".join(str_list)


Measure = MeasureGate()
