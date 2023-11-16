# TODO: Include license information?

from pecos.simulators.cointoss.gates import ignore_gate, measure

# Supporting gates from table:
#   https://github.com/CQCL/phir/blob/main/phir_spec_qasm.md#table-ii---quantum-operations
gate_dict = {
    "Init": ignore_gate,
    "Measure": measure,
    "I": ignore_gate,
    "X": ignore_gate,
    "Y": ignore_gate,
    "Z": ignore_gate,
    "RX": ignore_gate,
    "RY": ignore_gate,
    "RZ": ignore_gate,
    "R1XY": ignore_gate,
    "SX": ignore_gate,
    "SXdg": ignore_gate,
    "SY": ignore_gate,
    "SYdg": ignore_gate,
    "SZ": ignore_gate,
    "SZdg": ignore_gate,
    "H": ignore_gate,
    "F": ignore_gate,
    "Fdg": ignore_gate,
    "T": ignore_gate,
    "Tdg": ignore_gate,
    "CX": ignore_gate,
    "CY": ignore_gate,
    "CZ": ignore_gate,
    "RXX": ignore_gate,
    "RYY": ignore_gate,
    "RZZ": ignore_gate,
    "R2XXYYZZ": ignore_gate,
    "SXX": ignore_gate,
    "SXXdg": ignore_gate,
    "SYY": ignore_gate,
    "SYYdg": ignore_gate,
    "SZZ": ignore_gate,
    "SZZdg": ignore_gate,
    "SWAP": ignore_gate,
}
