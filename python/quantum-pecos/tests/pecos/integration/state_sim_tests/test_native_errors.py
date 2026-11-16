"""Native input errors propagate through the legacy Python simulator wrappers."""

from types import SimpleNamespace

import pytest
from pecos.simulators import PauliProp, SparseStabPy, StateVec
from pecos_rslib import AngleSource, GateRegistry


@pytest.mark.parametrize("simulator_type", [StateVec, PauliProp])
def test_wrapper_propagates_native_pair_value_error(simulator_type):
    """A single invalid pair raises ValueError rather than a native PanicException."""
    sim = simulator_type(num_qubits=3)
    with pytest.raises(ValueError, match="CX requires distinct qubits; repeated qubit 1"):
        sim.run_gate("CX", {(1, 1)})


def test_wrapper_propagates_registry_decompose_value_error():
    """The base decomposition path propagates fallible registry input errors."""
    registry = GateRegistry()
    registry.define("ROT", 1).angle_arity(1).step_with_angles("RZ", [0], [AngleSource.input(0)]).register_into(registry)
    circuit = SimpleNamespace(items=lambda: [("ROT", {0}, {})])
    with pytest.raises(ValueError, match="Gate ROT expects 1 qubits and 1 angles; got 1 qubits and 0 angles"):
        SparseStabPy(2).run_circuit(circuit, gate_registry=registry)
