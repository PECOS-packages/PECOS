"""Python interface for QASM simulation with enhanced API.

This module provides a clean Python interface for running quantum circuit simulations
using OpenQASM 2.0. It supports various noise models, quantum engines, and parallel execution.

For detailed usage examples, see the PECOS documentation:
https://github.com/CQCL/PECOS/blob/master/docs/user-guide/qasm-simulation.md
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
from pecos_rslib._pecos_rslib import (
    NoiseModel,
    QuantumEngine,
    QasmSimulation,
    QasmSimulationBuilder,
    GeneralNoiseModelBuilder,
    run_qasm as _run_qasm,
    qasm_sim as _qasm_sim,
    get_noise_models as _get_noise_models,
    get_quantum_engines as _get_quantum_engines,
)

__all__ = [
    "NoiseModel",
    "QuantumEngine",
    "QasmSimulation",
    "QasmSimulationBuilder",
    "get_noise_models",
    "get_quantum_engines",
    # Noise model dataclasses
    "PassThroughNoise",
    "DepolarizingNoise",
    "DepolarizingCustomNoise",
    "BiasedDepolarizingNoise",
    "GeneralNoise",
    # Builder classes
    "GeneralNoiseModelBuilder",  # Rust-native builder
    # Main interface
    "run_qasm",
    "qasm_sim",
]


# Noise model dataclasses


@dataclass
class PassThroughNoise:
    """No noise - ideal quantum simulation."""

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PassThroughNoise":
        """Create PassThroughNoise from configuration dictionary."""
        return cls()


@dataclass
class DepolarizingNoise:
    """Standard depolarizing noise with uniform probability.

    Args:
        p: Uniform error probability for all operations
    """

    p: float = 0.001

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DepolarizingNoise":
        """Create DepolarizingNoise from configuration dictionary."""
        return cls(p=config.get("p", 0.001))


@dataclass
class DepolarizingCustomNoise:
    """Depolarizing noise with custom probabilities for different operations.

    Args:
        p_prep: State preparation error probability
        p_meas: Measurement error probability
        p1: Single-qubit gate error probability
        p2: Two-qubit gate error probability
    """

    p_prep: float = 0.001
    p_meas: float = 0.001
    p1: float = 0.001
    p2: float = 0.002

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DepolarizingCustomNoise":
        """Create DepolarizingCustomNoise from configuration dictionary."""
        return cls(
            p_prep=config.get("p_prep", 0.001),
            p_meas=config.get("p_meas", 0.001),
            p1=config.get("p1", 0.001),
            p2=config.get("p2", 0.002),
        )


@dataclass
class BiasedDepolarizingNoise:
    """Biased depolarizing noise model.

    Args:
        p: Uniform probability for all operations
    """

    p: float = 0.001

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BiasedDepolarizingNoise":
        """Create BiasedDepolarizingNoise from configuration dictionary."""
        return cls(p=config.get("p", 0.001))


@dataclass
class GeneralNoise:
    """General noise model with full parameter configuration.

    This noise model supports detailed configuration of various error types including:
    - Idle/memory errors with coherent and incoherent noise
    - State preparation errors with leakage and crosstalk
    - Single-qubit gate errors with emission and Pauli models
    - Two-qubit gate errors with angle-dependent noise
    - Measurement errors with asymmetric bit-flip probabilities

    All parameters are optional. If not specified, default values from the
    GeneralNoiseModel will be used.
    """

    # Global parameters
    noiseless_gates: Optional[List[str]] = None
    seed: Optional[int] = None
    scale: Optional[float] = None
    leakage_scale: Optional[float] = None
    emission_scale: Optional[float] = None

    # Idle noise parameters
    p_idle_coherent: Optional[bool] = None
    p_idle_linear_rate: Optional[float] = None
    p_idle_linear_model: Optional[Dict[str, float]] = None
    p_idle_quadratic_rate: Optional[float] = None
    p_idle_coherent_to_incoherent_factor: Optional[float] = None
    idle_scale: Optional[float] = None

    # Preparation noise parameters
    p_prep: Optional[float] = None
    p_prep_leak_ratio: Optional[float] = None
    p_prep_crosstalk: Optional[float] = None
    prep_scale: Optional[float] = None
    p_prep_crosstalk_scale: Optional[float] = None

    # Single-qubit gate noise parameters
    p1: Optional[float] = None
    p1_emission_ratio: Optional[float] = None
    p1_emission_model: Optional[Dict[str, float]] = None
    p1_seepage_prob: Optional[float] = None
    p1_pauli_model: Optional[Dict[str, float]] = None
    p1_scale: Optional[float] = None

    # Two-qubit gate noise parameters
    p2: Optional[float] = None
    p2_angle_params: Optional[Tuple[float, float, float, float]] = None
    p2_angle_power: Optional[float] = None
    p2_emission_ratio: Optional[float] = None
    p2_emission_model: Optional[Dict[str, float]] = None
    p2_seepage_prob: Optional[float] = None
    p2_pauli_model: Optional[Dict[str, float]] = None
    p2_idle: Optional[float] = None
    p2_scale: Optional[float] = None

    # Measurement noise parameters
    p_meas_0: Optional[float] = None
    p_meas_1: Optional[float] = None
    p_meas_crosstalk: Optional[float] = None
    meas_scale: Optional[float] = None
    p_meas_crosstalk_scale: Optional[float] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GeneralNoise":
        """Create GeneralNoise from configuration dictionary."""
        # Filter out non-GeneralNoise fields
        filtered_config = {k: v for k, v in config.items() if k != "type"}
        return cls(**filtered_config)


def run_qasm(
    qasm: str,
    shots: int,
    noise_model: Optional[Any] = None,
    engine: Optional[QuantumEngine] = None,
    workers: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, List[int]]:
    """Run a QASM simulation with specified parameters.

    Args:
        qasm: QASM code as a string
        shots: Number of measurement shots to perform
        noise_model: Noise model instance (e.g., DepolarizingNoise(p=0.01)) or None for no noise
        engine: Quantum simulation engine (QuantumEngine.StateVector or QuantumEngine.SparseStabilizer)
        workers: Number of worker threads (None for default of 1)
        seed: Random seed for reproducibility (None for non-deterministic)

    Returns:
        Dict mapping register names to lists of measurement values (as integers).
        For example: {"c": [0, 3, 0, 3, ...]} for a Bell state measurement.

    Example:
        >>> from pecos_rslib.qasm_sim import run_qasm, DepolarizingNoise, QuantumEngine
        >>> qasm = '''
        ... OPENQASM 2.0;
        ... include "qelib1.inc";
        ... qreg q[2];
        ... creg c[2];
        ... h q[0];
        ... cx q[0], q[1];
        ... measure q -> c;
        ... '''
        >>> results = run_qasm(qasm, shots=1000, noise_model=DepolarizingNoise(p=0.01))
        >>> # Results are in columnar format
        >>> print(f"Got {len(results['c'])} measurements")
        >>> # Count occurrences of each measurement outcome
        >>> from collections import Counter
        >>> counts = Counter(results["c"])
        >>> print(counts)  # Should show roughly equal counts of 0 (00) and 3 (11)
    """
    return _run_qasm(qasm, shots, noise_model, engine, workers, seed)


def qasm_sim(qasm: str) -> QasmSimulationBuilder:
    """Create a QASM simulation builder for flexible configuration.

    This provides a builder pattern for QASM simulations, allowing you to
    build once and run multiple times with different shot counts.

    Args:
        qasm: QASM code as a string

    Returns:
        QasmSimulationBuilder that can be configured and run

    Example:
        >>> from pecos_rslib.qasm_sim import qasm_sim, DepolarizingNoise, QuantumEngine
        >>> qasm = '''
        ... OPENQASM 2.0;
        ... include "qelib1.inc";
        ... qreg q[2];
        ... creg c[2];
        ... h q[0];
        ... cx q[0], q[1];
        ... measure q -> c;
        ... '''
        >>> # Build once, run multiple times
        >>> sim = qasm_sim(qasm).seed(42).noise(DepolarizingNoise(p=0.01)).build()
        >>>
        >>> results_100 = sim.run(100)
        >>> results_1000 = sim.run(1000)
        >>>
        >>> # Or run directly without building
        >>> results = (
        ...     qasm_sim(qasm).noise(DepolarizingNoise(p=0.01)).workers(4).run(1000)
        ... )
        >>>
        >>> # Use Rust-native builder with fluent chaining
        >>> from pecos_rslib.qasm_sim import GeneralNoiseModelBuilder
        >>> builder = (
        ...     GeneralNoiseModelBuilder()
        ...     .with_seed(42)
        ...     .with_p1_probability(0.001)
        ...     .with_p2_probability(0.01)
        ... )
        >>>
        >>> # Direct configuration with method chaining (like Rust API)
        >>> sim = (
        ...     qasm_sim(qasm)
        ...     .seed(42)
        ...     .auto_workers()
        ...     .noise(builder)
        ...     .quantum_engine(QuantumEngine.StateVector)
        ...     .with_binary_string_format()
        ...     .build()
        ... )
        >>> results = sim.run(1000)
        >>>
        >>> # Using WebAssembly functions (requires wasm feature)
        >>> qasm_with_wasm = '''
        ... OPENQASM 2.0;
        ... creg a[10];
        ... creg b[10];
        ... creg result[10];
        ... a = 5;
        ... b = 3;
        ... result = add(a, b);  // Call WASM function
        ... '''
        >>> # Run with WASM module
        >>> results = qasm_sim(qasm_with_wasm).wasm("add.wasm").run(100)
    """
    return _qasm_sim(qasm)


def get_noise_models() -> List[str]:
    """Get a list of available noise model names.

    Returns:
        List of string names of available noise models, such as
        'PassThrough', 'Depolarizing', 'DepolarizingCustom', etc.

    Example:
        >>> from pecos_rslib.qasm_sim import get_noise_models
        >>> noise_models = get_noise_models()
        >>> print(noise_models)
        ['PassThrough', 'Depolarizing', 'DepolarizingCustom', ...]
    """
    return _get_noise_models()


def get_quantum_engines() -> List[str]:
    """Get a list of available quantum engine names.

    Returns:
        List of string names of available quantum engines, such as
        'StateVector', 'SparseStabilizer', etc.

    Example:
        >>> from pecos_rslib.qasm_sim import get_quantum_engines
        >>> engines = get_quantum_engines()
        >>> print(engines)
        ['StateVector', 'SparseStabilizer']
    """
    return _get_quantum_engines()
