"""Integration tests for hybrid engine with old error model."""

import pecos as pc
from pecos.noise.error_depolar import DepolarizingErrorModel
from pecos.simulators import SparseStab


def test_run_preserves_classical_register_metadata() -> None:
    """Runtime scratch and output overrides must not mutate the circuit's spec."""
    from copy import deepcopy

    from pecos.slr import CReg, Main, QReg, SlrConverter
    from pecos.slr.qeclib import qubit as qb

    q, c = QReg("data", 1), CReg("c", 1)
    circuit = SlrConverter(Main(q, c, qb.X(q[0]), qb.Measure(q[0]) > c[0])).quantum_circuit()
    original = deepcopy(circuit.metadata)
    pc.HybridEngine().run(SparseStab(1), circuit, shot_id=0, output_spec={"extra": 2})
    assert circuit.metadata == original


def test_simple_conditional() -> None:
    """Verify simulation and noise modeling works with conditional operations."""
    qc = pc.QuantumCircuit(cvar_spec={"m": 1, "a": 1}, num_qubits=1)
    qc.append("X", {0}, cond={"a": "a", "op": "==", "b": 0})
    qc.append("measure Z", {0}, var_output={0: ("m", 0)})

    eng = pc.HybridEngine()
    state = SparseStab(1)
    err = DepolarizingErrorModel()

    error_params = {
        "p1": 0.01,
        "p2": 0.01,
        "p_prep": 0.01,
        "p_meas": 0.01,
        "p2_mem": 0.01,
    }

    eng.run(state, qc, error_gen=err, shot_id=0, error_params=error_params)
