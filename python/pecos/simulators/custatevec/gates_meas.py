# TODO: Include license information?

from typing import Any
import random

from cuquantum import custatevec as cusv

def meas_z(state, qubit: int, **params: Any) -> int:
    """Measure in the Z-basis, collapse and normalise.

    Notes:
        The number of qubits in the state remains the same.

    Args:
        state: An instance of CuStateVec
        qubit: The index of the qubit to be measured

    Returns:
        The outcome of the measurement, either 0 or 1.
    """
    if qubit >= state.num_qubits or qubit < 0:
        raise ValueError(f"Qubit {qubit} out of range.")
    # CuStateVec uses smaller qubit index as least significant
    target = state.num_qubits - 1 - qubit

    return cusv.measure_on_z_basis(
        handle=state.libhandle,
        sv=state.vector.data.ptr,
        sv_data_type=state.cuda_type,
        n_index_bits=state.num_qubits,  # Number of qubits in the statevector
        basis_bits=[target],  # The index of the qubit being measured
        n_basis_bits=1,  # Number of qubits being measured
        rand_num=random.random(),  # Source of randomness for the measurement
        collapse=cusv.Collapse.NORMALIZE_AND_ZERO,  # Collapse and normalise
    )
    state.stream.synchronize()
