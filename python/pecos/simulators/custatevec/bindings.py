# TODO: Include license information?

from pecos.simulators.custatevec.gates_meas import meas_z
import pecos.simulators.custatevec.gates_one_qubit as one_qubit_gates

# Supporting gates from table:
#   https://github.com/CQCL/phir/blob/main/phir_spec_qasm.md#table-ii---quantum-operations
gate_dict = {
#    'Init': ignore_gate,
    'Measure': meas_z,

    'I': one_qubit_gates.I,
    'X': one_qubit_gates.X,
    'Y': one_qubit_gates.Y,
    'Z': one_qubit_gates.Z,

#    'RX': ignore_gate,
#    'RY': ignore_gate,
#    'RZ': ignore_gate,
#    'R1XY': ignore_gate,

#    'SX': ignore_gate,
#    'SXdg': ignore_gate,
#    'SY': ignore_gate,
#    'SYdg': ignore_gate,
#    'SZ': ignore_gate,
#    'SZdg': ignore_gate,

    'H': one_qubit_gates.H,
#    'F': ignore_gate,
#    'Fdg': ignore_gate,

#    'T': ignore_gate,
#    'Tdg': ignore_gate,

#    'CX': ignore_gate,
#    'CY': ignore_gate,
#    'CZ': ignore_gate,

#    'RXX': ignore_gate,
#    'RYY': ignore_gate,
#    'RZZ': ignore_gate,
#    'R2XXYYZZ': ignore_gate,

#    'SXX': ignore_gate,
#    'SXXdg': ignore_gate,
#    'SYY': ignore_gate,
#    'SYYdg': ignore_gate,
#    'SZZ': ignore_gate,
#    'SZZdg': ignore_gate,

#    'SWAP': ignore_gate,
}