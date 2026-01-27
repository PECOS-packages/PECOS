"""Type stubs for pecos_rslib_cuda - CUDA/cuQuantum Python bindings for PECOS."""

from typing import List

__version__: str

def is_cuquantum_available() -> bool:
    """Check if cuQuantum is available on this system.

    Returns:
        True if cuQuantum SDK is installed and accessible.
    """
    ...

class CuStateVec:
    """GPU-accelerated state vector quantum simulator using cuQuantum.

    This simulator can handle up to approximately 30 qubits (limited by GPU memory).
    It supports all quantum gates including arbitrary rotations.

    Args:
        num_qubits: Number of qubits to simulate.

    Example:
        >>> sim = CuStateVec(4)
        >>> sim.h([0])
        >>> sim.cx([0, 1])
        >>> results = sim.mz([0, 1])
    """

    def __init__(self, num_qubits: int) -> None: ...
    @staticmethod
    def with_seed(num_qubits: int, seed: int) -> "CuStateVec":
        """Create a new state vector simulator with a specific random seed."""
        ...

    @property
    def num_qubits(self) -> int:
        """Get the number of qubits in this simulator."""
        ...

    def reset(self) -> None:
        """Reset the simulator to the |0...0> state."""
        ...
    # =========================================================================
    # Pauli gates
    # =========================================================================

    def x(self, qubits: List[int]) -> None:
        """Apply Pauli X gate to the specified qubits."""
        ...

    def y(self, qubits: List[int]) -> None:
        """Apply Pauli Y gate to the specified qubits."""
        ...

    def z(self, qubits: List[int]) -> None:
        """Apply Pauli Z gate to the specified qubits."""
        ...
    # =========================================================================
    # Hadamard and variants
    # =========================================================================

    def h(self, qubits: List[int]) -> None:
        """Apply Hadamard gate to the specified qubits."""
        ...

    def h2(self, qubits: List[int]) -> None:
        """Apply H2 gate (Z*H*Z) to the specified qubits."""
        ...

    def h3(self, qubits: List[int]) -> None:
        """Apply H3 gate to the specified qubits."""
        ...

    def h4(self, qubits: List[int]) -> None:
        """Apply H4 gate to the specified qubits."""
        ...

    def h5(self, qubits: List[int]) -> None:
        """Apply H5 gate to the specified qubits."""
        ...

    def h6(self, qubits: List[int]) -> None:
        """Apply H6 gate to the specified qubits."""
        ...
    # =========================================================================
    # Square root gates (Clifford)
    # =========================================================================

    def s(self, qubits: List[int]) -> None:
        """Apply S gate (sqrt(Z)) to the specified qubits."""
        ...

    def sdg(self, qubits: List[int]) -> None:
        """Apply S-dagger gate to the specified qubits."""
        ...

    def sx(self, qubits: List[int]) -> None:
        """Apply sqrt(X) gate to the specified qubits."""
        ...

    def sxdg(self, qubits: List[int]) -> None:
        """Apply sqrt(X)-dagger gate to the specified qubits."""
        ...

    def sy(self, qubits: List[int]) -> None:
        """Apply sqrt(Y) gate to the specified qubits."""
        ...

    def sydg(self, qubits: List[int]) -> None:
        """Apply sqrt(Y)-dagger gate to the specified qubits."""
        ...

    def sz(self, qubits: List[int]) -> None:
        """Apply sqrt(Z) gate (same as S) to the specified qubits."""
        ...

    def szdg(self, qubits: List[int]) -> None:
        """Apply sqrt(Z)-dagger gate (same as Sdg) to the specified qubits."""
        ...
    # =========================================================================
    # Face rotation gates
    # =========================================================================

    def f(self, qubits: List[int]) -> None:
        """Apply F gate (face rotation X->Y->Z->X) to the specified qubits."""
        ...

    def fdg(self, qubits: List[int]) -> None:
        """Apply F-dagger gate to the specified qubits."""
        ...
    # =========================================================================
    # Two-qubit Clifford gates
    # =========================================================================

    def cx(self, qubits: List[int]) -> None:
        """Apply CNOT (CX) gate. First qubit is control, second is target."""
        ...

    def cy(self, qubits: List[int]) -> None:
        """Apply CY gate. First qubit is control, second is target."""
        ...

    def cz(self, qubits: List[int]) -> None:
        """Apply CZ gate."""
        ...

    def swap(self, qubits: List[int]) -> None:
        """Apply SWAP gate."""
        ...

    def iswap(self, qubits: List[int]) -> None:
        """Apply iSWAP gate."""
        ...

    def g(self, qubits: List[int]) -> None:
        """Apply G gate (Quantinuum native two-qubit gate)."""
        ...

    def sxx(self, qubits: List[int]) -> None:
        """Apply sqrt(XX) gate."""
        ...

    def sxxdg(self, qubits: List[int]) -> None:
        """Apply sqrt(XX)-dagger gate."""
        ...

    def syy(self, qubits: List[int]) -> None:
        """Apply sqrt(YY) gate."""
        ...

    def syydg(self, qubits: List[int]) -> None:
        """Apply sqrt(YY)-dagger gate."""
        ...

    def szz(self, qubits: List[int]) -> None:
        """Apply sqrt(ZZ) gate."""
        ...

    def szzdg(self, qubits: List[int]) -> None:
        """Apply sqrt(ZZ)-dagger gate."""
        ...
    # =========================================================================
    # Non-Clifford single-qubit gates
    # =========================================================================

    def t(self, qubits: List[int]) -> None:
        """Apply T gate (pi/8 gate) to the specified qubits."""
        ...

    def tdg(self, qubits: List[int]) -> None:
        """Apply T-dagger gate to the specified qubits."""
        ...
    # =========================================================================
    # Rotation gates
    # =========================================================================

    def rx(self, angle: float, qubits: List[int]) -> None:
        """Apply RX rotation gate."""
        ...

    def ry(self, angle: float, qubits: List[int]) -> None:
        """Apply RY rotation gate."""
        ...

    def rz(self, angle: float, qubits: List[int]) -> None:
        """Apply RZ rotation gate."""
        ...

    def rxx(self, angle: float, qubits: List[int]) -> None:
        """Apply RXX rotation gate."""
        ...

    def ryy(self, angle: float, qubits: List[int]) -> None:
        """Apply RYY rotation gate."""
        ...

    def rzz(self, angle: float, qubits: List[int]) -> None:
        """Apply RZZ rotation gate."""
        ...

    def u(self, theta: float, phi: float, lam: float, qubits: List[int]) -> None:
        """Apply U gate (general single-qubit rotation)."""
        ...

    def r1xy(self, theta: float, phi: float, qubits: List[int]) -> None:
        """Apply R1XY gate (rotation in XY plane)."""
        ...
    # =========================================================================
    # Measurement
    # =========================================================================

    def mx(self, qubits: List[int]) -> List[int]:
        """Measure qubits in the X basis."""
        ...

    def mnx(self, qubits: List[int]) -> List[int]:
        """Measure qubits in the -X basis."""
        ...

    def my(self, qubits: List[int]) -> List[int]:
        """Measure qubits in the Y basis."""
        ...

    def mny(self, qubits: List[int]) -> List[int]:
        """Measure qubits in the -Y basis."""
        ...

    def mz(self, qubits: List[int]) -> List[int]:
        """Measure qubits in the Z basis.

        Returns:
            A list of measurement results (0 or 1) for each qubit.
        """
        ...

    def sample(self, num_samples: int) -> List[int]:
        """Sample measurement outcomes from the current state without collapsing it.

        Args:
            num_samples: Number of samples to draw.

        Returns:
            List of bitstrings as integers. Each integer represents a measurement outcome
            where bit i corresponds to qubit i.
        """
        ...

class CuStabilizer:
    """GPU-accelerated stabilizer quantum simulator using cuQuantum.

    This simulator can handle thousands of qubits efficiently, but only supports
    Clifford gates (no T gates or arbitrary rotations).

    Args:
        num_qubits: Number of qubits to simulate.

    Example:
        >>> sim = CuStabilizer(100)
        >>> sim.h([0])
        >>> sim.cx([0, 1])
        >>> results = sim.mz([0, 1])
    """

    def __init__(self, num_qubits: int) -> None: ...
    @staticmethod
    def with_seed(num_qubits: int, seed: int) -> "CuStabilizer":
        """Create a new stabilizer simulator with a specific random seed."""
        ...

    @property
    def num_qubits(self) -> int:
        """Get the number of qubits in this simulator."""
        ...

    def reset(self) -> None:
        """Reset the simulator to the |0...0> state."""
        ...
    # =========================================================================
    # Pauli gates
    # =========================================================================

    def x(self, qubits: List[int]) -> None:
        """Apply Pauli X gate to the specified qubits."""
        ...

    def y(self, qubits: List[int]) -> None:
        """Apply Pauli Y gate to the specified qubits."""
        ...

    def z(self, qubits: List[int]) -> None:
        """Apply Pauli Z gate to the specified qubits."""
        ...
    # =========================================================================
    # Hadamard and variants
    # =========================================================================

    def h(self, qubits: List[int]) -> None:
        """Apply Hadamard gate to the specified qubits."""
        ...

    def h2(self, qubits: List[int]) -> None:
        """Apply H2 gate (Z*H*Z) to the specified qubits."""
        ...

    def h3(self, qubits: List[int]) -> None:
        """Apply H3 gate to the specified qubits."""
        ...

    def h4(self, qubits: List[int]) -> None:
        """Apply H4 gate to the specified qubits."""
        ...

    def h5(self, qubits: List[int]) -> None:
        """Apply H5 gate to the specified qubits."""
        ...

    def h6(self, qubits: List[int]) -> None:
        """Apply H6 gate to the specified qubits."""
        ...
    # =========================================================================
    # Square root gates
    # =========================================================================

    def s(self, qubits: List[int]) -> None:
        """Apply S gate (sqrt(Z)) to the specified qubits."""
        ...

    def sdg(self, qubits: List[int]) -> None:
        """Apply S-dagger gate to the specified qubits."""
        ...

    def sx(self, qubits: List[int]) -> None:
        """Apply sqrt(X) gate to the specified qubits."""
        ...

    def sxdg(self, qubits: List[int]) -> None:
        """Apply sqrt(X)-dagger gate to the specified qubits."""
        ...

    def sy(self, qubits: List[int]) -> None:
        """Apply sqrt(Y) gate to the specified qubits."""
        ...

    def sydg(self, qubits: List[int]) -> None:
        """Apply sqrt(Y)-dagger gate to the specified qubits."""
        ...

    def sz(self, qubits: List[int]) -> None:
        """Apply sqrt(Z) gate (same as S) to the specified qubits."""
        ...

    def szdg(self, qubits: List[int]) -> None:
        """Apply sqrt(Z)-dagger gate (same as Sdg) to the specified qubits."""
        ...
    # =========================================================================
    # Face rotation gates
    # =========================================================================

    def f(self, qubits: List[int]) -> None:
        """Apply F gate (face rotation X->Y->Z->X) to the specified qubits."""
        ...

    def fdg(self, qubits: List[int]) -> None:
        """Apply F-dagger gate to the specified qubits."""
        ...
    # =========================================================================
    # Two-qubit Clifford gates
    # =========================================================================

    def cx(self, qubits: List[int]) -> None:
        """Apply CNOT (CX) gate. First qubit is control, second is target."""
        ...

    def cy(self, qubits: List[int]) -> None:
        """Apply CY gate. First qubit is control, second is target."""
        ...

    def cz(self, qubits: List[int]) -> None:
        """Apply CZ gate."""
        ...

    def swap(self, qubits: List[int]) -> None:
        """Apply SWAP gate."""
        ...

    def iswap(self, qubits: List[int]) -> None:
        """Apply iSWAP gate."""
        ...

    def g(self, qubits: List[int]) -> None:
        """Apply G gate (Quantinuum native two-qubit gate)."""
        ...

    def sxx(self, qubits: List[int]) -> None:
        """Apply sqrt(XX) gate."""
        ...

    def sxxdg(self, qubits: List[int]) -> None:
        """Apply sqrt(XX)-dagger gate."""
        ...

    def syy(self, qubits: List[int]) -> None:
        """Apply sqrt(YY) gate."""
        ...

    def syydg(self, qubits: List[int]) -> None:
        """Apply sqrt(YY)-dagger gate."""
        ...

    def szz(self, qubits: List[int]) -> None:
        """Apply sqrt(ZZ) gate."""
        ...

    def szzdg(self, qubits: List[int]) -> None:
        """Apply sqrt(ZZ)-dagger gate."""
        ...
    # =========================================================================
    # Measurement
    # =========================================================================

    def mx(self, qubits: List[int]) -> List[int]:
        """Measure qubits in the X basis."""
        ...

    def mnx(self, qubits: List[int]) -> List[int]:
        """Measure qubits in the -X basis."""
        ...

    def my(self, qubits: List[int]) -> List[int]:
        """Measure qubits in the Y basis."""
        ...

    def mny(self, qubits: List[int]) -> List[int]:
        """Measure qubits in the -Y basis."""
        ...

    def mz(self, qubits: List[int]) -> List[int]:
        """Measure qubits in the Z basis.

        Returns:
            A list of measurement results (0 or 1) for each qubit.
        """
        ...

class CuTensorNet:
    """Tensor network simulator using NVIDIA cuTensorNet.

    This class manages a cuTensorNet handle for tensor network contractions.
    Tensor network methods can be used for simulating quantum circuits by
    contracting tensor networks representing the circuit.

    Use Cases:
        - Simulating quantum circuits with many qubits but shallow depth
        - Calculating expectation values
        - Approximate simulation of larger circuits

    Example:
        >>> net = CuTensorNet()
        >>> print(f"cuTensorNet version: {CuTensorNet.version()}")
    """

    def __init__(self) -> None:
        """Create a new tensor network handle."""
        ...

    @staticmethod
    def version() -> int:
        """Get the cuTensorNet version.

        Returns:
            The version as a single integer (e.g., 20000 for version 2.0.0).
        """
        ...

class CuDensityMat:
    """Density matrix simulator using NVIDIA cuDensityMat.

    This simulator manages a cuDensityMat handle and state, providing methods for
    density matrix operations including noisy quantum simulation.

    Advantages over State Vector:
        - Can represent mixed states (statistical mixtures)
        - Natural representation for noise and decoherence
        - Essential for open quantum system simulation

    Memory Requirements:
        Density matrices require O(4^n) memory vs O(2^n) for state vectors,
        limiting practical simulation to fewer qubits.

    Args:
        num_qubits: Number of qubits to simulate.

    Example:
        >>> sim = CuDensityMat(4)  # 4-qubit density matrix
        >>> print(f"cuDensityMat version: {CuDensityMat.version()}")
    """

    def __init__(self, num_qubits: int) -> None:
        """Create a new density matrix simulator.

        Initializes the state to the pure state |0...0><0...0|.

        Args:
            num_qubits: Number of qubits to simulate.
        """
        ...

    @property
    def num_qubits(self) -> int:
        """Get the number of qubits in this simulator."""
        ...

    @staticmethod
    def version() -> int:
        """Get the cuDensityMat version.

        Returns:
            The version as a single integer.
        """
        ...
