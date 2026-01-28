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

"""Gate bindings for CUDA state vector simulator (pecos-rslib-cuda)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.simulators.cuda_statevec.state import CudaStateVec
    from pecos.typing import SimulatorGateParams


def _to_list(q: int | tuple[int, ...]) -> list[int]:
    """Convert qubit location to list."""
    if isinstance(q, int):
        return [q]
    return list(q)


# =============================================================================
# Initialization gates
# =============================================================================


def init_zero(state: CudaStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """Initialize qubit to |0> state."""
    result = meas_z(state, qubit)
    if result:
        X(state, qubit)


def init_one(state: CudaStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """Initialize qubit to |1> state."""
    result = meas_z(state, qubit)
    if not result:
        X(state, qubit)


# =============================================================================
# Measurement gates
# =============================================================================


def meas_z(state: CudaStateVec, qubit: int, **_params: SimulatorGateParams) -> int:
    """Measure in the Z-basis, collapse and normalize."""
    results = state.backend.mz([qubit])
    return results[0]


def meas_x(state: CudaStateVec, qubit: int, **_params: SimulatorGateParams) -> int:
    """Measure in the X-basis."""
    results = state.backend.mx([qubit])
    return results[0]


def meas_y(state: CudaStateVec, qubit: int, **_params: SimulatorGateParams) -> int:
    """Measure in the Y-basis."""
    results = state.backend.my([qubit])
    return results[0]


# =============================================================================
# Single-qubit Pauli gates
# =============================================================================


def identity(state: CudaStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """Identity gate (no-op)."""


def X(state: CudaStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """Pauli X gate."""
    state.backend.x([qubit])


def Y(state: CudaStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """Pauli Y gate."""
    state.backend.y([qubit])


def Z(state: CudaStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """Pauli Z gate."""
    state.backend.z([qubit])


# =============================================================================
# Square root gates
# =============================================================================


def SX(state: CudaStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """sqrt(X) gate."""
    state.backend.sx([qubit])


def SXdg(state: CudaStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """sqrt(X)-dagger gate."""
    state.backend.sxdg([qubit])


def SY(state: CudaStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """sqrt(Y) gate."""
    state.backend.sy([qubit])


def SYdg(state: CudaStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """sqrt(Y)-dagger gate."""
    state.backend.sydg([qubit])


def SZ(state: CudaStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """sqrt(Z) gate (S gate)."""
    state.backend.s([qubit])


def SZdg(state: CudaStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """sqrt(Z)-dagger gate (S-dagger)."""
    state.backend.sdg([qubit])


# =============================================================================
# Hadamard variants
# =============================================================================


def H(state: CudaStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """Hadamard gate."""
    state.backend.h([qubit])


def H2(state: CudaStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """H2 gate (Z*H*Z)."""
    state.backend.h2([qubit])


def H3(state: CudaStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """H3 gate."""
    state.backend.h3([qubit])


def H4(state: CudaStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """H4 gate."""
    state.backend.h4([qubit])


def H5(state: CudaStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """H5 gate."""
    state.backend.h5([qubit])


def H6(state: CudaStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """H6 gate."""
    state.backend.h6([qubit])


# =============================================================================
# Face rotation gates
# =============================================================================


def F(state: CudaStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """F gate (face rotation X->Y->Z->X)."""
    state.backend.f([qubit])


def Fdg(state: CudaStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """F-dagger gate."""
    state.backend.fdg([qubit])


# =============================================================================
# T gate
# =============================================================================


def T(state: CudaStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """T gate (pi/8 gate)."""
    state.backend.t([qubit])


def Tdg(state: CudaStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """T-dagger gate."""
    state.backend.tdg([qubit])


# =============================================================================
# Rotation gates
# =============================================================================


def RX(
    state: CudaStateVec,
    qubit: int,
    angles: tuple[float],
    **_params: SimulatorGateParams,
) -> None:
    """RX rotation gate."""
    if len(angles) != 1:
        msg = "RX gate requires exactly 1 angle parameter."
        raise ValueError(msg)
    state.backend.rx(angles[0], [qubit])


def RY(
    state: CudaStateVec,
    qubit: int,
    angles: tuple[float],
    **_params: SimulatorGateParams,
) -> None:
    """RY rotation gate."""
    if len(angles) != 1:
        msg = "RY gate requires exactly 1 angle parameter."
        raise ValueError(msg)
    state.backend.ry(angles[0], [qubit])


def RZ(
    state: CudaStateVec,
    qubit: int,
    angles: tuple[float],
    **_params: SimulatorGateParams,
) -> None:
    """RZ rotation gate."""
    if len(angles) != 1:
        msg = "RZ gate requires exactly 1 angle parameter."
        raise ValueError(msg)
    state.backend.rz(angles[0], [qubit])


def R1XY(
    state: CudaStateVec,
    qubit: int,
    angles: tuple[float, float],
    **_params: SimulatorGateParams,
) -> None:
    """R1XY gate (rotation in XY plane)."""
    if len(angles) != 2:
        msg = "R1XY gate requires exactly 2 angle parameters."
        raise ValueError(msg)
    state.backend.r1xy(angles[0], angles[1], [qubit])


# =============================================================================
# Two-qubit gates
# =============================================================================


def CX(
    state: CudaStateVec,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """CNOT gate."""
    state.backend.cx(list(qubits))


def CY(
    state: CudaStateVec,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """CY gate."""
    state.backend.cy(list(qubits))


def CZ(
    state: CudaStateVec,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """CZ gate."""
    state.backend.cz(list(qubits))


def SWAP(
    state: CudaStateVec,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """SWAP gate."""
    state.backend.swap(list(qubits))


def ISWAP(
    state: CudaStateVec,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """ISWAP gate."""
    state.backend.iswap(list(qubits))


def G(
    state: CudaStateVec,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """G gate (Quantinuum native two-qubit gate)."""
    state.backend.g(list(qubits))


# =============================================================================
# Two-qubit sqrt gates
# =============================================================================


def SXX(
    state: CudaStateVec,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """sqrt(XX) gate."""
    state.backend.sxx(list(qubits))


def SXXdg(
    state: CudaStateVec,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """sqrt(XX)-dagger gate."""
    state.backend.sxxdg(list(qubits))


def SYY(
    state: CudaStateVec,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """sqrt(YY) gate."""
    state.backend.syy(list(qubits))


def SYYdg(
    state: CudaStateVec,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """sqrt(YY)-dagger gate."""
    state.backend.syydg(list(qubits))


def SZZ(
    state: CudaStateVec,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """sqrt(ZZ) gate."""
    state.backend.szz(list(qubits))


def SZZdg(
    state: CudaStateVec,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """sqrt(ZZ)-dagger gate."""
    state.backend.szzdg(list(qubits))


# =============================================================================
# Two-qubit rotation gates
# =============================================================================


def RXX(
    state: CudaStateVec,
    qubits: tuple[int, int],
    angles: tuple[float],
    **_params: SimulatorGateParams,
) -> None:
    """RXX rotation gate."""
    if len(angles) != 1:
        msg = "RXX gate requires exactly 1 angle parameter."
        raise ValueError(msg)
    state.backend.rxx(angles[0], list(qubits))


def RYY(
    state: CudaStateVec,
    qubits: tuple[int, int],
    angles: tuple[float],
    **_params: SimulatorGateParams,
) -> None:
    """RYY rotation gate."""
    if len(angles) != 1:
        msg = "RYY gate requires exactly 1 angle parameter."
        raise ValueError(msg)
    state.backend.ryy(angles[0], list(qubits))


def RZZ(
    state: CudaStateVec,
    qubits: tuple[int, int],
    angles: tuple[float],
    **_params: SimulatorGateParams,
) -> None:
    """RZZ rotation gate."""
    if len(angles) != 1:
        msg = "RZZ gate requires exactly 1 angle parameter."
        raise ValueError(msg)
    state.backend.rzz(angles[0], list(qubits))


# =============================================================================
# Gate dictionary
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
    # T gate
    "T": T,
    "Tdg": Tdg,
    # Rotations
    "RX": RX,
    "RY": RY,
    "RZ": RZ,
    "R1XY": R1XY,
    "RXY1Q": R1XY,
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
    # Two-qubit rotations
    "RXX": RXX,
    "RYY": RYY,
    "RZZ": RZZ,
}
