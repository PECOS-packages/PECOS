# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Gate bindings for CUDA stabilizer simulator (pecos-rslib-cuda).

Note: This simulator only supports Clifford gates. Non-Clifford gates
(T, arbitrary rotations) are not available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.simulators.cuda_stabilizer.state import CudaStabilizer
    from pecos.typing import SimulatorGateParams


# =============================================================================
# Initialization gates
# =============================================================================


def init_zero(
    state: CudaStabilizer,
    qubit: int,
    **_params: SimulatorGateParams,
) -> None:
    """Initialize qubit to |0> state."""
    result = meas_z(state, qubit)
    if result:
        X(state, qubit)


def init_one(state: CudaStabilizer, qubit: int, **_params: SimulatorGateParams) -> None:
    """Initialize qubit to |1> state."""
    result = meas_z(state, qubit)
    if not result:
        X(state, qubit)


# =============================================================================
# Measurement gates
# =============================================================================


def meas_z(state: CudaStabilizer, qubit: int, **_params: SimulatorGateParams) -> int:
    """Measure in the Z-basis."""
    results = state.backend.mz([qubit])
    return results[0]


def meas_x(state: CudaStabilizer, qubit: int, **_params: SimulatorGateParams) -> int:
    """Measure in the X-basis."""
    results = state.backend.mx([qubit])
    return results[0]


def meas_y(state: CudaStabilizer, qubit: int, **_params: SimulatorGateParams) -> int:
    """Measure in the Y-basis."""
    results = state.backend.my([qubit])
    return results[0]


# =============================================================================
# Single-qubit Pauli gates
# =============================================================================


def identity(state: CudaStabilizer, qubit: int, **_params: SimulatorGateParams) -> None:
    """Identity gate (no-op)."""


def X(state: CudaStabilizer, qubit: int, **_params: SimulatorGateParams) -> None:
    """Pauli X gate."""
    state.backend.x([qubit])


def Y(state: CudaStabilizer, qubit: int, **_params: SimulatorGateParams) -> None:
    """Pauli Y gate."""
    state.backend.y([qubit])


def Z(state: CudaStabilizer, qubit: int, **_params: SimulatorGateParams) -> None:
    """Pauli Z gate."""
    state.backend.z([qubit])


# =============================================================================
# Square root gates
# =============================================================================


def SX(state: CudaStabilizer, qubit: int, **_params: SimulatorGateParams) -> None:
    """sqrt(X) gate."""
    state.backend.sx([qubit])


def SXdg(state: CudaStabilizer, qubit: int, **_params: SimulatorGateParams) -> None:
    """sqrt(X)-dagger gate."""
    state.backend.sxdg([qubit])


def SY(state: CudaStabilizer, qubit: int, **_params: SimulatorGateParams) -> None:
    """sqrt(Y) gate."""
    state.backend.sy([qubit])


def SYdg(state: CudaStabilizer, qubit: int, **_params: SimulatorGateParams) -> None:
    """sqrt(Y)-dagger gate."""
    state.backend.sydg([qubit])


def SZ(state: CudaStabilizer, qubit: int, **_params: SimulatorGateParams) -> None:
    """sqrt(Z) gate (S gate)."""
    state.backend.s([qubit])


def SZdg(state: CudaStabilizer, qubit: int, **_params: SimulatorGateParams) -> None:
    """sqrt(Z)-dagger gate (S-dagger)."""
    state.backend.sdg([qubit])


# =============================================================================
# Hadamard variants
# =============================================================================


def H(state: CudaStabilizer, qubit: int, **_params: SimulatorGateParams) -> None:
    """Hadamard gate."""
    state.backend.h([qubit])


def H2(state: CudaStabilizer, qubit: int, **_params: SimulatorGateParams) -> None:
    """H2 gate (Z*H*Z)."""
    state.backend.h2([qubit])


def H3(state: CudaStabilizer, qubit: int, **_params: SimulatorGateParams) -> None:
    """H3 gate."""
    state.backend.h3([qubit])


def H4(state: CudaStabilizer, qubit: int, **_params: SimulatorGateParams) -> None:
    """H4 gate."""
    state.backend.h4([qubit])


def H5(state: CudaStabilizer, qubit: int, **_params: SimulatorGateParams) -> None:
    """H5 gate."""
    state.backend.h5([qubit])


def H6(state: CudaStabilizer, qubit: int, **_params: SimulatorGateParams) -> None:
    """H6 gate."""
    state.backend.h6([qubit])


# =============================================================================
# Face rotation gates
# =============================================================================


def F(state: CudaStabilizer, qubit: int, **_params: SimulatorGateParams) -> None:
    """F gate (face rotation X->Y->Z->X)."""
    state.backend.f([qubit])


def Fdg(state: CudaStabilizer, qubit: int, **_params: SimulatorGateParams) -> None:
    """F-dagger gate."""
    state.backend.fdg([qubit])


# =============================================================================
# Two-qubit gates
# =============================================================================


def CX(
    state: CudaStabilizer,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """CNOT gate."""
    state.backend.cx(list(qubits))


def CY(
    state: CudaStabilizer,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """CY gate."""
    state.backend.cy(list(qubits))


def CZ(
    state: CudaStabilizer,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """CZ gate."""
    state.backend.cz(list(qubits))


def SWAP(
    state: CudaStabilizer,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """SWAP gate."""
    state.backend.swap(list(qubits))


def ISWAP(
    state: CudaStabilizer,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """ISWAP gate."""
    state.backend.iswap(list(qubits))


def G(
    state: CudaStabilizer,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """G gate (Quantinuum native two-qubit gate)."""
    state.backend.g(list(qubits))


# =============================================================================
# Two-qubit sqrt gates
# =============================================================================


def SXX(
    state: CudaStabilizer,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """sqrt(XX) gate."""
    state.backend.sxx(list(qubits))


def SXXdg(
    state: CudaStabilizer,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """sqrt(XX)-dagger gate."""
    state.backend.sxxdg(list(qubits))


def SYY(
    state: CudaStabilizer,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """sqrt(YY) gate."""
    state.backend.syy(list(qubits))


def SYYdg(
    state: CudaStabilizer,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """sqrt(YY)-dagger gate."""
    state.backend.syydg(list(qubits))


def SZZ(
    state: CudaStabilizer,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """sqrt(ZZ) gate."""
    state.backend.szz(list(qubits))


def SZZdg(
    state: CudaStabilizer,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """sqrt(ZZ)-dagger gate."""
    state.backend.szzdg(list(qubits))


# =============================================================================
# Gate dictionary (Clifford gates only)
# =============================================================================


gate_dict = {
    # Initialization
    "Init": init_zero,
    "Init +Z": init_zero,
    "Init -Z": init_one,
    "init |0>": init_zero,
    "init |1>": init_one,
    "leak": init_zero,
    "leak |0>": init_zero,
    "leak |1>": init_one,
    "unleak |0>": init_zero,
    "unleak |1>": init_one,
    # Measurement
    "Measure": meas_z,
    "measure Z": meas_z,
    "measure X": meas_x,
    "measure Y": meas_y,
    # Identity
    "I": identity,
    "II": identity,
    # Paulis
    "X": X,
    "Y": Y,
    "Z": Z,
    # Square roots
    "SX": SX,
    "SXdg": SXdg,
    "SqrtX": SX,
    "SqrtXd": SXdg,
    "Q": SX,
    "Qd": SXdg,
    "SY": SY,
    "SYdg": SYdg,
    "SqrtY": SY,
    "SqrtYd": SYdg,
    "R": SY,
    "Rd": SYdg,
    "SZ": SZ,
    "SZdg": SZdg,
    "SqrtZ": SZ,
    "SqrtZd": SZdg,
    "S": SZ,
    "Sd": SZdg,
    # Hadamard
    "H": H,
    "H1": H,
    "H2": H2,
    "H3": H3,
    "H4": H4,
    "H5": H5,
    "H6": H6,
    "H+z+x": H,
    "H-z-x": H2,
    "H+y-z": H3,
    "H-y-z": H4,
    "H-x+y": H5,
    "H-x-y": H6,
    # Face rotations
    "F": F,
    "F1": F,
    "Fdg": Fdg,
    "F1d": Fdg,
    # Two-qubit
    "CX": CX,
    "CNOT": CX,
    "CY": CY,
    "CZ": CZ,
    "SWAP": SWAP,
    "ISWAP": ISWAP,
    "G": G,
    # Two-qubit sqrt
    "SXX": SXX,
    "SXXdg": SXXdg,
    "SYY": SYY,
    "SYYdg": SYYdg,
    "SZZ": SZZ,
    "SqrtZZ": SZZ,
    "SZZdg": SZZdg,
}
