from ..sim_class_types import StateVector
from ...circuits import QuantumCircuit

class CUQuantumSim(StateVector):
    """
    Args:
        num_qubits (int): Number of qubits being represented.

    Returns:

    """

    def __init__(self, num_qubits):

        if not isinstance(num_qubits, int):
            raise Exception('``num_qubits`` should be of type ``int.``')

        super().__init__()

        self.bindings = bindings.gate_dict
        self.num_qubits = num_qubits
        self.eng = MainEngine()

        self.qureg = self.eng.allocate_qureg(num_qubits)
        self.qs = list(self.qureg)
        self.qids = {i: q for i, q in enumerate(self.qs)}

    def logical_sign(self, logical_op: QuantumCircuit) -> int:
        """

        Args:
            logical_op:

        Returns:

        """
        return find_logical_signs(self, logical_op,)

    def add_gate(self,
                 symbol: str,
                 gate_obj,
                 make_func: bool = True):
        """
        Adds a new gate on the fly to the this Simulator.

        Args:
            symbol:
            gate_obj:
            make_func:

        Returns:

        """

        if symbol in self.gate_dict:
            print('WARNING: Can not add gate as the symbol has already been taken.')
        else:
            if make_func:
                self.gate_dict[symbol] = MakeFunc(gate_obj).func
            else:
                self.gate_dict[symbol] = gate_obj

    def __del__(self):
        self.eng.flush()
        All(Measure) | self.qureg  # Requirement by ProjectQ...

        try:
            self.eng.flush(deallocate_qubits=True)
        except KeyError:
            pass