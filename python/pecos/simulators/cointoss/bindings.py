# TODO: Include license information?

from pecos.simulators.cointoss.gates import ignore_gate, measure, force_output

gate_dict = {
    'U1q': ignore_gate,
    'RXY1Q': ignore_gate,

    'RX': ignore_gate,
    'RY': ignore_gate,
    'RZ': ignore_gate,

    'H': ignore_gate,

    'I': ignore_gate,
    'X': ignore_gate,
    'Y': ignore_gate,
    'Z': ignore_gate,

    'SqrtZZ': ignore_gate,
    'RZZ': ignore_gate,
    'CNOT': ignore_gate,
    'CX': ignore_gate,

    'measure Z': measure,
    'measure X': measure,
    'measure Y': measure,
    'force output': force_output,

    'init |0>': ignore_gate,
    'init |1>': ignore_gate,

    'leak': ignore_gate,
    'unleak |0>': ignore_gate,
    'unleak |1>': ignore_gate,

    # Square root of Paulis
    'Q': ignore_gate,   # +x-y  sqrt of X
    'Qd': ignore_gate,  # +x+y sqrt of X dagger
    'R': ignore_gate,   # -z+x sqrt of Y
    'Rd': ignore_gate,  # +z-x sqrt of Y dagger
    'S': ignore_gate,   # +y+z sqrt of Z
    'Sd': ignore_gate,  # -y+z sqrt of Z dagger

    'SqrtX': ignore_gate,   # +x-y  sqrt of X
    'SqrtXd': ignore_gate,  # +x+y sqrt of X dagger
    'SqrtY': ignore_gate,   # -z+x sqrt of Y
    'SqrtYd': ignore_gate,  # +z-x sqrt of Y dagger
    'SqrtZ': ignore_gate,   # +y+z sqrt of Z
    'SqrtZd': ignore_gate,  # -y+z sqrt of Z dagger
}