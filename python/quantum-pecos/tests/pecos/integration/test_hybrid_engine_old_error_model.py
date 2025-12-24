"""Integration tests for hybrid engine with old error model."""

import pecos as pc
from pecos.error_models.error_depolar import DepolarizingErrorModel
from pecos.simulators import SparseSim


def test_simple_conditional() -> None:
    """Verify simulation and noise modeling works with conditional operations."""
    qc = pc.QuantumCircuit(cvar_spec={"m": 1, "a": 1}, num_qubits=1)
    qc.append("X", {0}, cond={"a": "a", "op": "==", "b": 0})
    qc.append("measure Z", {0}, var_output={0: ("m", 0)})

    eng = pc.HybridEngine()
    state = SparseSim(1)
    err = DepolarizingErrorModel()

    error_params = {
        "p1": 0.01,
        "p2": 0.01,
        "p_init": 0.01,
        "p_meas": 0.01,
        "p2_mem": 0.01,
    }

    eng.run(state, qc, error_gen=err, shot_id=0, error_params=error_params)
