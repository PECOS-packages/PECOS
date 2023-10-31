# TODO: Include license information?

from typing import Any
import random

from cuquantum import custatevec as cusv

from pecos.simulators.custatevec.gates_meas import meas_z
from pecos.simulators.custatevec.gates_one_qubit import X

def init_zero(state, qubit: int, **params: Any) -> None:
    """Initialise or reset the qubit to state |0>

    Args:
        state: An instance of CuStateVec
        qubit: The index of the qubit to be initialised
    """
    result = meas_z(state, qubit)

    if result:
        X(state, qubit)
