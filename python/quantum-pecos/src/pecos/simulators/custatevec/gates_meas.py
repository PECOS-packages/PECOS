# Copyright 2023 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Quantum measurement operations for cuStateVec simulator.

This module provides GPU-accelerated quantum measurement operations for the NVIDIA cuStateVec simulator, including
projective measurements with proper state collapse using CUDA acceleration for high-performance simulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pecos as pc

if TYPE_CHECKING:
    from pecos.simulators.custatevec.state import CuStateVec
    from pecos.typing import SimulatorGateParams
from cuquantum.bindings import custatevec as cusv


def meas_z(state: CuStateVec, qubit: int, **_params: SimulatorGateParams) -> int:
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
        msg = f"Qubit {qubit} out of range."
        raise ValueError(msg)
    # CuStateVec uses smaller qubit index as least significant
    target = state.num_qubits - 1 - qubit

    result = cusv.measure_on_z_basis(
        handle=state.libhandle,
        sv=state.cupy_vector.data.ptr,
        sv_data_type=state.cuda_type,
        n_index_bits=state.num_qubits,  # Number of qubits in the statevector
        basis_bits=[target],  # The index of the qubit being measured
        n_basis_bits=1,  # Number of qubits being measured
        randnum=pc.random.random(1)[0],  # Source of randomness for the measurement
        collapse=cusv.Collapse.NORMALIZE_AND_ZERO,  # Collapse and normalise
    )
    state.stream.synchronize()

    return result
