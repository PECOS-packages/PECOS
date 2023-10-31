# TODO: Include license information?

import pytest

from importlib.metadata import version
from packaging.version import parse as vparse

from pecos.circuits import QuantumCircuit
import numpy as np

try:
    import cuquantum
    imported_cuquantum = vparse(version("cuquantum")) >= vparse("23.6.0")
    import cupy as cp
    imported_cupy = vparse(version("cupy")) >= vparse("10.4.0")
    from pecos.simulators import CuStateVec
    custatevec_ready = imported_cuquantum and imported_cupy
except ImportError:
    custatevec_ready = False


# TODO: Add unit tests for all gates and larger circuits


def verify(simulator, qc: QuantumCircuit, final_vector: np.ndarray) -> None:

    if simulator == "CuStateVec" and custatevec_ready:
        sim = CuStateVec(len(qc.qudits))
        sim.run_circuit(qc)

        assert np.allclose(sim.vector, final_vector)

    else:
        pytest.skip(f"Requirements to test {simulator} are not met.")


def check_measurement(simulator, qc: QuantumCircuit, final_results: dict[int,int]=None) -> None:

    if simulator == "CuStateVec" and custatevec_ready:
        sim = CuStateVec(len(qc.qudits))
        results = sim.run_circuit(qc)

        if final_results is not None:
            assert results == final_results

        state = 0
        for q, value in results.items():
            state += value*2**(sim.num_qubits-1-q)
        final_vector = cp.zeros(shape=(2**sim.num_qubits,))
        final_vector[state] = 1

        assert np.allclose(sim.vector, final_vector)

    else:
        pytest.skip(f"Requirements to test {simulator} are not met.")


@pytest.mark.parametrize(
    "simulator",
    [
        "CuStateVec",
    ],
)
def test_init(simulator):
    qc = QuantumCircuit()
    qc.append({'Init': {0, 1, 2, 3}})

    final_vector = cp.zeros(shape=(2**4,))
    final_vector[0] = 1

    verify(simulator, qc, final_vector)


@pytest.mark.parametrize(
    "simulator",
    [
        "CuStateVec",
    ],
)
def test_H_measure(simulator):
    qc = QuantumCircuit()
    qc.append({'H': {0, 1, 2, 3, 4}})
    qc.append({'Measure': {0, 1, 2, 3, 4}})

    check_measurement(simulator, qc)


@pytest.mark.parametrize(
    "simulator",
    [
        "CuStateVec",
    ],
)
def test_comp_basis_circ_and_measure(simulator):
    qc = QuantumCircuit()
    qc.append({'Init': {0, 1, 2, 3}})

    # Step 1
    qc.append({'X': {0, 2}})  # |0000> -> |1010>

    final_vector = cp.zeros(shape=(2**4,))
    final_vector[10] = 1  # |1010>

    verify(simulator, qc, final_vector)

    # Step 2
    qc.append({'CX': {(2, 1)}})  # |1010> -> |1110>

    final_vector = cp.zeros(shape=(2**4,))
    final_vector[14] = 1  # |1110>

    verify(simulator, qc, final_vector)

    # Step 3
    qc.append({'SWAP': {(0, 3)}})  # |1110> -> |0111>

    final_vector = cp.zeros(shape=(2**4,))
    final_vector[7] = 1  # |0111>

    verify(simulator, qc, final_vector)

    # Step 4
    qc.append({'CX': {(0, 2)}})  # |0111> -> |0111>

    final_vector = cp.zeros(shape=(2**4,))
    final_vector[7] = 1  # |0111>

    verify(simulator, qc, final_vector)

    # Step 5
    qc.append({'Init': {1}})  # |0111> -> |0011>

    final_vector = cp.zeros(shape=(2**4,))
    final_vector[3] = 1  # |0011>

    verify(simulator, qc, final_vector)

    # Step 6
    qc.append({'SWAP': {(1, 2)}})  # |0011> -> |0101>

    final_vector = cp.zeros(shape=(2**4,))
    final_vector[5] = 1  # |0011>

    verify(simulator, qc, final_vector)

    # Measure
    qc.append({'Measure': {0, 1, 2, 3}})

    final_results = {1: 1, 3: 1}  # |0101>

    check_measurement(simulator, qc, final_results)
