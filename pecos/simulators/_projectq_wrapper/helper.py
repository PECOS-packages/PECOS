# TODO: Make a "MakeFunc" for measurements...


class MakeFunc:
    """
    Converts ProjectQ gate to a function.
    """
    def __init__(self, gate, angle=False):
        """

        Args:
            gate:
        """

        self.gate = gate
        self.angle = angle

    def func(self, state, qubits, **params):

        if isinstance(qubits, int):
            qs = state.qids[qubits]
        else:
            qs = []
            for q in qubits:
                qs.append(state.qids[q])

        if self.angle:
            self.gate(params['angle']) | qs
        else:
            self.gate | qs
