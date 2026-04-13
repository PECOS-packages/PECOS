# Copyright 2024 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Integration tests for state vector quantum simulators using pure PECOS (no NumPy)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from pecos.simulators.sim_class_types import StateVector

import json
from pathlib import Path

import pecos as pc
import pytest
from pecos.circuits import QuantumCircuit
from pecos.engines.hybrid_engine import HybridEngine
from pecos.noise.generic_error_model import GenericErrorModel
from pecos.simulators import (
    MPS,
    CuStateVec,
    StateVec,
)
from pecos.testing import assert_allclose

str_to_sim = {
    "StateVec": StateVec,
    "CuStateVec": CuStateVec,
    "MPS": MPS,
}


def check_dependencies(
    simulator: str,
    **kwargs: object,
) -> Callable[[int], StateVector]:
    """Check if dependencies for a simulator are available and skip test if not.

    Args:
        simulator: Name of the simulator to check.
        **kwargs: Optional parameters to pass to the simulator constructor.

    Returns:
        A function that creates a simulator instance with the given parameters.
    """
    if simulator not in str_to_sim or str_to_sim[simulator] is None:
        pytest.skip(f"Requirements to test {simulator} are not met.")
    sim_class = str_to_sim[simulator]

    # Return a lambda that passes kwargs to the simulator constructor
    if kwargs:
        return lambda num_qubits: sim_class(num_qubits, **kwargs)
    return sim_class


def _compare_vectors(
    sim_vector: pc.Array,
    ref_vector: pc.Array,
    simulator: str,
) -> None:
    """Compare two state vectors, accounting for global phase differences."""
    sim_vector_normalized = sim_vector / (pc.linalg.norm(sim_vector) or 1)
    ref_vector_normalized = ref_vector / (pc.linalg.norm(ref_vector) or 1)

    phase = ref_vector_normalized[0] / sim_vector_normalized[0] if pc.abs(sim_vector_normalized[0]) > 1e-10 else 1

    sim_vector_adjusted = sim_vector_normalized * phase

    rtol = 1e-5
    _ = simulator  # reserved for per-backend tolerance tuning

    # Add absolute tolerance to handle near-zero values with numerical noise
    # MPS uses tensor network approximations that can introduce ~1e-15 errors
    # This prevents "inf" relative errors when comparing to exact 0
    atol = 1e-12

    assert_allclose(
        sim_vector_adjusted,
        ref_vector_normalized,
        rtol=rtol,
        atol=atol,
        err_msg="State vectors do not match.",
    )


def verify(simulator: str, qc: QuantumCircuit, final_vector: pc.Array) -> None:
    """Verify quantum circuit simulation results against expected state vector."""
    sim = check_dependencies(simulator)(len(qc.qudits))
    sim.run_circuit(qc)
    _compare_vectors(sim.vector, final_vector, simulator)


def check_measurement(
    simulator: str,
    qc: QuantumCircuit,
    final_results: dict[int, int] | None = None,
) -> None:
    """Check measurement results from quantum circuit simulation."""
    sim = check_dependencies(simulator)(len(qc.qudits))

    results = sim.run_circuit(qc)

    if final_results is not None:
        assert results == final_results

    state = 0
    for q, value in results.items():
        state += value * 2 ** (sim.num_qubits - 1 - q)
    final_vector = pc.zeros(shape=(2**sim.num_qubits,), dtype=pc.dtypes.complex128)
    final_vector[state] = 1

    abs_values_vector = [pc.abs(x) for x in sim.vector]

    assert pc.allclose(abs_values_vector, final_vector)


def compare_against_statevec(
    simulator: str,
    qc: QuantumCircuit,
    **sim_kwargs: object,
) -> None:
    """Compare simulator results against StateVec reference implementation.

    Args:
        simulator: Name of the simulator to test.
        qc: Quantum circuit to simulate.
        **sim_kwargs: Optional parameters passed to the simulator constructor.
            For MPS, use chi=32 or truncation_fidelity=0.999 for faster tests
            (cannot use both simultaneously).
    """
    statevec = StateVec(len(qc.qudits))
    statevec.run_circuit(qc)

    sim = check_dependencies(simulator, **sim_kwargs)(len(qc.qudits))
    sim.run_circuit(qc)

    _compare_vectors(sim.vector, statevec.vector, simulator)


def generate_random_state(seed: int | None = None) -> QuantumCircuit:
    """Generate a quantum circuit with random gates for testing."""
    pc.random.seed(seed)

    qc = QuantumCircuit()
    qc.append({"Init": {0, 1, 2, 3}})

    for _ in range(3):
        qc.append({"RZ": {0}}, angles=(pc.f64.pi * pc.random.random(1)[0],))
        qc.append({"RZ": {1}}, angles=(pc.f64.pi * pc.random.random(1)[0],))
        qc.append({"RZ": {2}}, angles=(pc.f64.pi * pc.random.random(1)[0],))
        qc.append({"RZ": {3}}, angles=(pc.f64.pi * pc.random.random(1)[0],))
        qc.append({"RXX": {(0, 1)}}, angles=(pc.f64.pi * pc.random.random(1)[0],))
        qc.append({"RXX": {(0, 2)}}, angles=(pc.f64.pi * pc.random.random(1)[0],))
        qc.append({"RXX": {(0, 3)}}, angles=(pc.f64.pi * pc.random.random(1)[0],))
        qc.append({"RXX": {(1, 2)}}, angles=(pc.f64.pi * pc.random.random(1)[0],))
        qc.append({"RXX": {(1, 3)}}, angles=(pc.f64.pi * pc.random.random(1)[0],))
        qc.append({"RXX": {(2, 3)}}, angles=(pc.f64.pi * pc.random.random(1)[0],))

    return qc


@pytest.mark.parametrize(
    "simulator",
    [
        "StateVec",
        "CuStateVec",
        "MPS",
    ],
)
def test_init(simulator: str) -> None:
    """Test quantum state initialization."""
    qc = QuantumCircuit()
    qc.append({"Init": {0, 1, 2, 3}})

    final_vector = pc.zeros(shape=(2**4,), dtype=pc.dtypes.complex128)
    final_vector[0] = 1

    verify(simulator, qc, final_vector)


@pytest.mark.parametrize(
    "simulator",
    [
        "StateVec",
        "CuStateVec",
        "MPS",
    ],
)
def test_H_measure(simulator: str) -> None:
    """Test Hadamard gate followed by measurement."""
    qc = QuantumCircuit()
    qc.append({"H": {0, 1, 2, 3}})
    qc.append({"Measure": {0, 1, 2, 3}})

    check_measurement(simulator, qc)


@pytest.mark.parametrize(
    "simulator",
    [
        "StateVec",
        "CuStateVec",
        "MPS",
    ],
)
def test_comp_basis_circ_and_measure(simulator: str) -> None:
    """Test computational basis circuit and measurement."""
    qc = QuantumCircuit()
    qc.append({"Init": {0, 1, 2, 3}})

    # Step 1
    qc.append({"X": {0, 2}})  # |0000> -> |1010>

    final_vector = pc.zeros(shape=(2**4,), dtype=pc.dtypes.complex128)
    final_vector[10] = 1  # |1010>

    # Run the circuit and compare results
    verify(simulator, qc, final_vector)

    # Insert detailed debug prints after verify
    sim_class = check_dependencies(simulator)
    sim_instance = sim_class(len(qc.qudits))
    sim_instance.run_circuit(qc)

    # Step 2
    qc.append({"CX": {(2, 1)}})  # |1010> -> |1110>

    final_vector = pc.zeros(shape=(2**4,), dtype=pc.dtypes.complex128)
    final_vector[14] = 1  # |1110>

    # Run the circuit and compare results for Step 2
    verify(simulator, qc, final_vector)
    sim_instance.run_circuit(qc)


def _apply_gate_and_compare(
    qc: QuantumCircuit,
    ref_sim: StateVector,
    test_sim: StateVector,
    simulator: str,
    gate: dict,
    **params: object,
) -> None:
    """Apply gate to circuit and both sims, then compare state vectors."""
    qc.append(gate, **params)
    symbol = next(iter(gate))
    locations = gate[symbol]
    ref_sim.run_gate(symbol, locations, **params)
    test_sim.run_gate(symbol, locations, **params)
    _compare_vectors(test_sim.vector, ref_sim.vector, simulator)


def _test_all_gates_incremental(
    simulator: str,
    qc: QuantumCircuit,
    sim_kwargs: dict,
) -> None:
    """Apply gates incrementally to persistent sims and compare after each."""
    num_qubits = len(qc.qudits)
    ref_sim = StateVec(num_qubits)
    ref_sim.run_circuit(qc)
    test_sim = check_dependencies(simulator, **sim_kwargs)(num_qubits)
    test_sim.run_circuit(qc)

    def _apply(gate: dict, **params: object) -> None:
        _apply_gate_and_compare(qc, ref_sim, test_sim, simulator, gate, **params)

    _apply({"SZZ": {(3, 2)}})
    _apply({"RX": {0, 2}}, angles=(pc.f64.frac_pi_4,))
    _apply({"SXXdg": {(0, 3)}})
    _apply({"RY": {0, 3}}, angles=(pc.f64.pi / 8,))
    _apply({"RZZ": {(0, 3)}}, angles=(pc.f64.pi / 16,))
    _apply({"RZ": {1, 3}}, angles=(pc.f64.pi / 16,))
    _apply({"R1XY": {2}}, angles=(pc.f64.pi / 16, pc.f64.frac_pi_2))
    _apply({"I": {0, 1, 3}})
    _apply({"X": {1, 2}})
    _apply({"Y": {2, 3}})
    _apply({"CY": {(2, 3), (0, 1)}})
    _apply({"SYY": {(1, 2)}})
    _apply({"Z": {2, 0}})
    _apply({"H": {3, 1}})
    _apply({"RYY": {(2, 1)}}, angles=(pc.f64.pi / 8,))
    _apply({"SZZdg": {(3, 1)}})
    _apply({"F": {0, 1, 2}})
    _apply({"CX": {(0, 1), (3, 2)}})
    _apply({"Fdg": {3, 1}})
    _apply({"SYYdg": {(1, 3)}})
    _apply({"SX": {1, 2}})
    _apply(
        {"RXXRYYRZZ": {(0, 3)}},
        angles=(pc.f64.frac_pi_4, pc.f64.pi / 16, pc.f64.frac_pi_2),
    )
    _apply({"SY": {2, 3}})
    _apply({"SZ": {2, 0}})
    _apply({"SZdg": {1, 2}})
    _apply({"CZ": {(1, 3)}})
    _apply({"SXdg": {2, 3}})
    _apply({"SYdg": {2, 0}})
    _apply({"T": {0, 1, 2}})
    _apply({"SXX": {(0, 2)}})
    _apply({"SWAP": {(3, 0)}})
    _apply({"Tdg": {3, 1}})
    _apply({"RXX": {(1, 3)}}, angles=(pc.f64.frac_pi_4,))
    _apply({"Q": {0, 1, 2}})
    _apply({"Qd": {0, 3}})
    _apply({"R": {0}})
    _apply({"Rd": {0, 1, 2}})
    _apply({"S": {0, 3}})
    _apply({"Sd": {0}})
    _apply({"H2": {2, 3}})
    _apply({"H3": {0, 1, 2}})
    _apply({"H4": {2, 3}})
    _apply({"H5": {0, 3}})
    _apply({"H6": {0, 1, 2}})
    _apply({"F2": {0, 1, 2}})
    _apply({"F2d": {0, 3}})
    _apply({"F3": {2, 3}})
    _apply({"F3d": {0, 1, 2}})
    _apply({"F4": {2, 3}})
    _apply({"F4d": {0, 3}})
    _apply({"CNOT": {(0, 1)}})
    _apply({"G": {(1, 3)}})
    _apply({"II": {(3, 2)}})

    # Measure
    qc.append({"Measure": {0, 1, 2, 3}})
    check_measurement(simulator, qc)


@pytest.mark.parametrize(
    "simulator",
    [
        "StateVec",
        "CuStateVec",
        "MPS",
    ],
)
def test_all_gate_circ(simulator: str) -> None:
    """Test circuit with all quantum gates.

    Maintains persistent simulator instances and applies gates incrementally
    rather than replaying the full circuit from scratch after each gate append.

    Note:
        For MPS simulator, uses reduced bond dimension (chi=32) to limit computational
        cost while maintaining reasonable accuracy.
    """
    # Use chi=32 for MPS to balance speed and accuracy
    # This limits bond dimension and speeds up the 4-qubit test
    sim_kwargs = {"chi": 32} if simulator == "MPS" else {}

    # Generate three different arbitrary states
    qcs: list[QuantumCircuit] = []
    qcs.append(generate_random_state(seed=1234))
    qcs.append(generate_random_state(seed=5555))
    qcs.append(generate_random_state(seed=42))

    # Verify that each of these states matches with StateVec
    for qc in qcs:
        compare_against_statevec(simulator, qc, **sim_kwargs)

    # Apply each gate on randomly generated states and compare after each.
    # Uses persistent simulators with incremental gate application to avoid
    # replaying the full circuit from scratch on every comparison.
    for qc in qcs:
        _test_all_gates_incremental(simulator, qc, sim_kwargs)


@pytest.mark.parametrize(
    "simulator",
    [
        "StateVec",
        "CuStateVec",
    ],
)
def test_hybrid_engine_no_noise(simulator: str) -> None:
    """Test that HybridEngine can use these simulators."""
    check_dependencies(simulator)

    n_shots = 1000
    phir_folder = Path(__file__).parent.parent / "phir"

    sim = HybridEngine(qsim=simulator)
    with (phir_folder / "bell_qparallel.phir.json").open() as f:
        program = json.load(f)
    results = sim.run(
        program=program,
        shots=n_shots,
        seed=42,
    )

    register = "c" if "c" in results else "m"
    result_values = results[register]
    assert pc.isclose(
        result_values.count("00") / n_shots,
        result_values.count("11") / n_shots,
        rtol=0.0,
        atol=0.1,
    )


@pytest.mark.parametrize(
    "simulator",
    [
        "StateVec",
        "CuStateVec",
    ],
)
def test_hybrid_engine_noisy(simulator: str) -> None:
    """Test that HybridEngine with noise can use these simulators."""
    check_dependencies(simulator)

    n_shots = 1000
    phir_folder = Path(__file__).parent.parent / "phir"

    generic_errors = GenericErrorModel(
        error_params={
            "p1": 2e-1,
            "p2": 2e-1,
            "p_meas": 2e-1,
            "p_init": 1e-1,
            "p1_error_model": {
                "X": 0.25,
                "Y": 0.25,
                "Z": 0.25,
                "L": 0.25,
            },
        },
    )
    sim = HybridEngine(qsim=simulator, error_model=generic_errors)
    with (phir_folder / "example1_no_wasm.phir.json").open() as f:
        program = json.load(f)
    sim.run(
        program=program,
        shots=n_shots,
    )
