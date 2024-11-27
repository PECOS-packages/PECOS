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

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable

    from pecos.simulators.sim_class_types import StateVector

import json
from pathlib import Path

import numpy as np
import pytest
from pecos.circuits import QuantumCircuit
from pecos.engines.hybrid_engine import HybridEngine
from pecos.error_models.generic_error_model import GenericErrorModel
from pecos.simulators import (
    MPS,
    BasicSV,
    CuStateVec,
    QuEST,
    Qulacs,
)

str_to_sim = {
    "BasicSV": BasicSV,
    "Qulacs": Qulacs,
    "QuEST": QuEST,
    "CuStateVec": CuStateVec,
    "MPS": MPS,
}


def check_dependencies(simulator) -> Callable[[int], StateVector]:
    if simulator not in str_to_sim or str_to_sim[simulator] is None:
        pytest.skip(f"Requirements to test {simulator} are not met.")
    return str_to_sim[simulator]


def verify(simulator, qc: QuantumCircuit, final_vector: np.ndarray) -> None:
    sim = check_dependencies(simulator)(len(qc.qudits))

    sim.run_circuit(qc)
    assert np.allclose(sim.vector, final_vector)


def check_measurement(
    simulator,
    qc: QuantumCircuit,
    final_results: dict[int, int] | None = None,
) -> None:
    sim = check_dependencies(simulator)(len(qc.qudits))

    results = sim.run_circuit(qc)

    if final_results is not None:
        assert results == final_results

    state = 0
    for q, value in results.items():
        state += value * 2 ** (sim.num_qubits - 1 - q)
    final_vector = np.zeros(shape=(2**sim.num_qubits,))
    final_vector[state] = 1

    abs_values_vector = [abs(x) for x in sim.vector]

    assert np.allclose(abs_values_vector, final_vector)


def compare_against_basicsv(simulator, qc: QuantumCircuit):
    basicsv = BasicSV(len(qc.qudits))
    basicsv.run_circuit(qc)
    verify(simulator, qc, basicsv.vector)


def generate_random_state(seed=None) -> QuantumCircuit:
    np.random.seed(seed)

    qc = QuantumCircuit()
    qc.append({"Init": {0, 1, 2, 3, 4}})

    for _ in range(3):
        qc.append({"RZ": {0}}, angles=(np.pi * np.random.random(),))
        qc.append({"RZ": {1}}, angles=(np.pi * np.random.random(),))
        qc.append({"RZ": {2}}, angles=(np.pi * np.random.random(),))
        qc.append({"RZ": {3}}, angles=(np.pi * np.random.random(),))
        qc.append({"RZ": {4}}, angles=(np.pi * np.random.random(),))
        qc.append({"RXX": {(0, 1)}}, angles=(np.pi * np.random.random(),))
        qc.append({"RXX": {(0, 2)}}, angles=(np.pi * np.random.random(),))
        qc.append({"RXX": {(0, 3)}}, angles=(np.pi * np.random.random(),))
        qc.append({"RXX": {(0, 4)}}, angles=(np.pi * np.random.random(),))
        qc.append({"RXX": {(1, 2)}}, angles=(np.pi * np.random.random(),))
        qc.append({"RXX": {(1, 3)}}, angles=(np.pi * np.random.random(),))
        qc.append({"RXX": {(1, 4)}}, angles=(np.pi * np.random.random(),))
        qc.append({"RXX": {(2, 3)}}, angles=(np.pi * np.random.random(),))
        qc.append({"RXX": {(2, 4)}}, angles=(np.pi * np.random.random(),))
        qc.append({"RXX": {(3, 4)}}, angles=(np.pi * np.random.random(),))

    return qc


@pytest.mark.parametrize(
    "simulator",
    [
        "BasicSV",
        "Qulacs",
        "QuEST",
        "CuStateVec",
        "MPS",
    ],
)
def test_init(simulator):
    qc = QuantumCircuit()
    qc.append({"Init": {0, 1, 2, 3}})

    final_vector = np.zeros(shape=(2**4,))
    final_vector[0] = 1

    verify(simulator, qc, final_vector)


@pytest.mark.parametrize(
    "simulator",
    [
        "BasicSV",
        "Qulacs",
        "QuEST",
        "CuStateVec",
        "MPS",
    ],
)
def test_H_measure(simulator):
    qc = QuantumCircuit()
    qc.append({"H": {0, 1, 2, 3, 4}})
    qc.append({"Measure": {0, 1, 2, 3, 4}})

    check_measurement(simulator, qc)


@pytest.mark.parametrize(
    "simulator",
    [
        "BasicSV",
        "Qulacs",
        "QuEST",
        "CuStateVec",
        "MPS",
    ],
)
def test_comp_basis_circ_and_measure(simulator):
    qc = QuantumCircuit()
    qc.append({"Init": {0, 1, 2, 3}})

    # Step 1
    qc.append({"X": {0, 2}})  # |0000> -> |1010>

    final_vector = np.zeros(shape=(2**4,))
    final_vector[10] = 1  # |1010>

    verify(simulator, qc, final_vector)

    # Step 2
    qc.append({"CX": {(2, 1)}})  # |1010> -> |1110>

    final_vector = np.zeros(shape=(2**4,))
    final_vector[14] = 1  # |1110>

    verify(simulator, qc, final_vector)

    # Step 3
    qc.append({"SWAP": {(0, 3)}})  # |1110> -> |0111>

    final_vector = np.zeros(shape=(2**4,))
    final_vector[7] = 1  # |0111>

    verify(simulator, qc, final_vector)

    # Step 4
    qc.append({"CX": {(0, 2)}})  # |0111> -> |0111>

    final_vector = np.zeros(shape=(2**4,))
    final_vector[7] = 1  # |0111>

    verify(simulator, qc, final_vector)

    # Step 5
    qc.append({"Init": {1}})  # |0111> -> |0011>

    final_vector = np.zeros(shape=(2**4,))
    final_vector[3] = 1  # |0011>

    verify(simulator, qc, final_vector)

    # Step 6
    qc.append({"SWAP": {(1, 2)}})  # |0011> -> |0101>

    final_vector = np.zeros(shape=(2**4,))
    final_vector[5] = 1  # |0011>

    verify(simulator, qc, final_vector)

    # Measure
    qc.append({"Measure": {0, 1, 2, 3}})

    final_results = {1: 1, 3: 1}  # |0101>

    check_measurement(simulator, qc, final_results)


@pytest.mark.parametrize(
    "simulator",
    [
        "Qulacs",
        "QuEST",
        "CuStateVec",
        "MPS",
    ],
)
def test_all_gate_circ(simulator):
    # Generate three different arbitrary states
    qcs: list[QuantumCircuit] = []
    qcs.append(generate_random_state(seed=1234))
    qcs.append(generate_random_state(seed=5555))
    qcs.append(generate_random_state(seed=42))

    # Verify that each of these states matches with BasicSV
    for qc in qcs:
        compare_against_basicsv(simulator, qc)

    # Apply each gate on randomly generated states and compare again
    for qc in qcs:
        qc.append({"SZZ": {(4, 2)}})
        compare_against_basicsv(simulator, qc)
        qc.append({"RX": {0, 2}}, angles=(np.pi / 4,))
        compare_against_basicsv(simulator, qc)
        qc.append({"SXXdg": {(0, 3)}})
        compare_against_basicsv(simulator, qc)
        qc.append({"RY": {0, 3}}, angles=(np.pi / 8,))
        compare_against_basicsv(simulator, qc)
        qc.append({"RZZ": {(0, 3)}}, angles=(np.pi / 16,))
        compare_against_basicsv(simulator, qc)
        qc.append({"RZ": {1, 4}}, angles=(np.pi / 16,))
        compare_against_basicsv(simulator, qc)
        qc.append({"R1XY": {2}}, angles=(np.pi / 16, np.pi / 2))
        compare_against_basicsv(simulator, qc)
        qc.append({"I": {0, 1, 3}})
        compare_against_basicsv(simulator, qc)
        qc.append({"X": {1, 2}})
        compare_against_basicsv(simulator, qc)
        qc.append({"Y": {3, 4}})
        compare_against_basicsv(simulator, qc)
        qc.append({"CY": {(2, 3), (4, 1)}})
        compare_against_basicsv(simulator, qc)
        qc.append({"SYY": {(1, 4)}})
        compare_against_basicsv(simulator, qc)
        qc.append({"Z": {2, 0}})
        compare_against_basicsv(simulator, qc)
        qc.append({"H": {3, 1}})
        compare_against_basicsv(simulator, qc)
        qc.append({"RYY": {(2, 1)}}, angles=(np.pi / 8,))
        compare_against_basicsv(simulator, qc)
        qc.append({"SZZdg": {(3, 1)}})
        compare_against_basicsv(simulator, qc)
        qc.append({"F": {0, 2, 4}})
        compare_against_basicsv(simulator, qc)
        qc.append({"CX": {(0, 1), (4, 2)}})
        compare_against_basicsv(simulator, qc)
        qc.append({"Fdg": {3, 1}})
        compare_against_basicsv(simulator, qc)
        qc.append({"SYYdg": {(1, 3)}})
        compare_against_basicsv(simulator, qc)
        qc.append({"SX": {1, 2}})
        compare_against_basicsv(simulator, qc)
        qc.append({"R2XXYYZZ": {(0, 4)}}, angles=(np.pi / 4, np.pi / 16, np.pi / 2))
        compare_against_basicsv(simulator, qc)
        qc.append({"SY": {3, 4}})
        compare_against_basicsv(simulator, qc)
        qc.append({"SZ": {2, 0}})
        compare_against_basicsv(simulator, qc)
        qc.append({"SZdg": {1, 2}})
        compare_against_basicsv(simulator, qc)
        qc.append({"CZ": {(1, 3)}})
        compare_against_basicsv(simulator, qc)
        qc.append({"SXdg": {3, 4}})
        compare_against_basicsv(simulator, qc)
        qc.append({"SYdg": {2, 0}})
        compare_against_basicsv(simulator, qc)
        qc.append({"T": {0, 2, 4}})
        compare_against_basicsv(simulator, qc)
        qc.append({"SXX": {(0, 2)}})
        compare_against_basicsv(simulator, qc)
        qc.append({"SWAP": {(4, 0)}})
        compare_against_basicsv(simulator, qc)
        qc.append({"Tdg": {3, 1}})
        compare_against_basicsv(simulator, qc)
        qc.append({"RXX": {(1, 3)}}, angles=(np.pi / 4,))
        compare_against_basicsv(simulator, qc)
        qc.append({"Q": {1, 4, 2}})
        compare_against_basicsv(simulator, qc)
        qc.append({"Qd": {0, 3}})
        compare_against_basicsv(simulator, qc)
        qc.append({"R": {0}})
        compare_against_basicsv(simulator, qc)
        qc.append({"Rd": {1, 4, 2}})
        compare_against_basicsv(simulator, qc)
        qc.append({"S": {0, 3}})
        compare_against_basicsv(simulator, qc)
        qc.append({"Sd": {0}})
        compare_against_basicsv(simulator, qc)
        qc.append({"H1": {0, 3}})
        compare_against_basicsv(simulator, qc)
        qc.append({"H2": {2, 3}})
        compare_against_basicsv(simulator, qc)
        qc.append({"H3": {1, 4, 2}})
        compare_against_basicsv(simulator, qc)
        qc.append({"H4": {2, 3}})
        compare_against_basicsv(simulator, qc)
        qc.append({"H5": {0, 3}})
        compare_against_basicsv(simulator, qc)
        qc.append({"H6": {1, 4, 2}})
        compare_against_basicsv(simulator, qc)
        qc.append({"H+z+x": {2, 3}})
        compare_against_basicsv(simulator, qc)
        qc.append({"H-z-x": {1, 4, 2}})
        compare_against_basicsv(simulator, qc)
        qc.append({"H+y-z": {0, 3}})
        compare_against_basicsv(simulator, qc)
        qc.append({"H-y-z": {2, 3}})
        compare_against_basicsv(simulator, qc)
        qc.append({"H-x+y": {0, 3}})
        compare_against_basicsv(simulator, qc)
        qc.append({"H-x-y": {1, 4, 2}})
        compare_against_basicsv(simulator, qc)
        qc.append({"F1": {0, 3}})
        compare_against_basicsv(simulator, qc)
        qc.append({"F1d": {2, 3}})
        compare_against_basicsv(simulator, qc)
        qc.append({"F2": {1, 4, 2}})
        compare_against_basicsv(simulator, qc)
        qc.append({"F2d": {0, 3}})
        compare_against_basicsv(simulator, qc)
        qc.append({"F3": {2, 3}})
        compare_against_basicsv(simulator, qc)
        qc.append({"F3d": {1, 4, 2}})
        compare_against_basicsv(simulator, qc)
        qc.append({"F4": {2, 3}})
        compare_against_basicsv(simulator, qc)
        qc.append({"F4d": {0, 3}})
        compare_against_basicsv(simulator, qc)
        qc.append({"CNOT": {(0, 1)}})
        compare_against_basicsv(simulator, qc)
        qc.append({"G": {(1, 3)}})
        compare_against_basicsv(simulator, qc)
        qc.append({"II": {(4, 2)}})
        compare_against_basicsv(simulator, qc)

        # Measure
        qc.append({"Measure": {0, 1, 2, 3, 4}})
        check_measurement(simulator, qc)


@pytest.mark.parametrize(
    "simulator",
    [
        "MPS",
        "Qulacs",
        "QuEST",
        "CuStateVec",
    ],
)
def test_hybrid_engine_no_noise(simulator):
    """Test that HybridEngine can use these simulators"""
    check_dependencies(simulator)

    n_shots = 1000
    phir_folder = Path(__file__).parent.parent / "phir"

    results = HybridEngine(qsim=simulator).run(
        program=json.load(Path.open(phir_folder / "bell_qparallel.json")),
        shots=n_shots,
    )

    m = results["m"]
    assert np.isclose(m.count("00") / n_shots, m.count("11") / n_shots, atol=0.1)


@pytest.mark.parametrize(
    "simulator",
    [
        "MPS",
        "Qulacs",
        "QuEST",
        "CuStateVec",
    ],
)
def test_hybrid_engine_noisy(simulator):
    """Test that HybridEngine with noise can use these simulators"""
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
    sim.run(
        program=json.load(Path.open(phir_folder / "example1_no_wasm.json")),
        shots=n_shots,
    )
