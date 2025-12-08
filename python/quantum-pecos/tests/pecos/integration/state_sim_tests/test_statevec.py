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
from pecos.error_models.generic_error_model import GenericErrorModel
from pecos.simulators import (
    MPS,
    CuStateVec,
    QuestStateVec,
    Qulacs,
    StateVec,
)
from pecos.testing import assert_allclose

str_to_sim = {
    "StateVec": StateVec,
    "Qulacs": Qulacs,
    "CuStateVec": CuStateVec,
    "MPS": MPS,
    "QuestStateVec": QuestStateVec,
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


def verify(simulator: str, qc: QuantumCircuit, final_vector: pc.Array) -> None:
    """Verify quantum circuit simulation results against expected state vector."""
    sim = check_dependencies(simulator)(len(qc.qudits))
    sim.run_circuit(qc)

    # Normalize vectors
    sim_vector_normalized = sim.vector / (pc.linalg.norm(sim.vector) or 1)
    final_vector_normalized = final_vector / (pc.linalg.norm(final_vector) or 1)

    phase = (
        final_vector_normalized[0] / sim_vector_normalized[0]
        if pc.abs(sim_vector_normalized[0]) > 1e-10
        else 1
    )

    sim_vector_adjusted = sim_vector_normalized * phase

    # Use looser tolerance for simulators that use gate decompositions
    # QuestStateVec uses decompositions for RXX, RYY, RZZ which accumulate errors
    rtol = 1e-3 if simulator == "QuestStateVec" else 1e-5

    # Add absolute tolerance to handle near-zero values with numerical noise
    # MPS uses tensor network approximations that can introduce ~1e-15 errors
    # This prevents "inf" relative errors when comparing to exact 0
    atol = 1e-12

    assert_allclose(
        sim_vector_adjusted,
        final_vector_normalized,
        rtol=rtol,
        atol=atol,
        err_msg="State vectors do not match.",
    )


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

    # Use updated verify function
    verify(simulator, qc, statevec.vector)


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
        "Qulacs",
        "CuStateVec",
        "MPS",
        "QuestStateVec",
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
        "Qulacs",
        "CuStateVec",
        "MPS",
        "QuestStateVec",
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
        "Qulacs",
        "CuStateVec",
        "MPS",
        "QuestStateVec",
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


@pytest.mark.parametrize(
    "simulator",
    [
        "StateVec",
        "Qulacs",
        "CuStateVec",
        "MPS",
        "QuestStateVec",
    ],
)
def test_all_gate_circ(simulator: str) -> None:
    """Test circuit with all quantum gates.

    Note:
        For MPS simulator, uses reduced bond dimension (chi=32) to limit computational
        cost while maintaining reasonable accuracy. MPS tests take longer due to gate
        application overhead in the tensor network backend.
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

    # Apply each gate on randomly generated states and compare again
    for qc in qcs:
        qc.append({"SZZ": {(3, 2)}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"RX": {0, 2}}, angles=(pc.f64.frac_pi_4,))
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"SXXdg": {(0, 3)}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"RY": {0, 3}}, angles=(pc.f64.pi / 8,))
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"RZZ": {(0, 3)}}, angles=(pc.f64.pi / 16,))
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"RZ": {1, 3}}, angles=(pc.f64.pi / 16,))
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"R1XY": {2}}, angles=(pc.f64.pi / 16, pc.f64.frac_pi_2))
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"I": {0, 1, 3}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"X": {1, 2}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"Y": {2, 3}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"CY": {(2, 3), (0, 1)}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"SYY": {(1, 2)}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"Z": {2, 0}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"H": {3, 1}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"RYY": {(2, 1)}}, angles=(pc.f64.pi / 8,))
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"SZZdg": {(3, 1)}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"F": {0, 1, 2}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"CX": {(0, 1), (3, 2)}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"Fdg": {3, 1}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"SYYdg": {(1, 3)}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"SX": {1, 2}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append(
            {"R2XXYYZZ": {(0, 3)}},
            angles=(pc.f64.frac_pi_4, pc.f64.pi / 16, pc.f64.frac_pi_2),
        )
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"SY": {2, 3}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"SZ": {2, 0}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"SZdg": {1, 2}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"CZ": {(1, 3)}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"SXdg": {2, 3}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"SYdg": {2, 0}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"T": {0, 1, 2}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"SXX": {(0, 2)}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"SWAP": {(3, 0)}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"Tdg": {3, 1}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"RXX": {(1, 3)}}, angles=(pc.f64.frac_pi_4,))
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"Q": {0, 1, 2}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"Qd": {0, 3}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"R": {0}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"Rd": {0, 1, 2}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"S": {0, 3}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"Sd": {0}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"H2": {2, 3}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"H3": {0, 1, 2}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"H4": {2, 3}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"H5": {0, 3}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"H6": {0, 1, 2}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"F2": {0, 1, 2}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"F2d": {0, 3}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"F3": {2, 3}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"F3d": {0, 1, 2}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"F4": {2, 3}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"F4d": {0, 3}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"CNOT": {(0, 1)}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"G": {(1, 3)}})
        compare_against_statevec(simulator, qc, **sim_kwargs)
        qc.append({"II": {(3, 2)}})
        compare_against_statevec(simulator, qc, **sim_kwargs)

        # Measure
        qc.append({"Measure": {0, 1, 2, 3}})
        check_measurement(simulator, qc)


@pytest.mark.parametrize(
    "simulator",
    [
        "StateVec",
        "Qulacs",
        "CuStateVec",
        "QuestStateVec",
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
        "Qulacs",
        "CuStateVec",
        "QuestStateVec",
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
