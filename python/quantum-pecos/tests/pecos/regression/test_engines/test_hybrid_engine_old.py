"""Tests for the hybrid engine."""

import pecos as pc
from pecos.simulators import SparseSim


def test_hybrid_engine() -> None:
    """Test hybrid engine functionality with a simple Bell state circuit."""
    qc = pc.QuantumCircuit(cvar_spec={"m": 2})
    qc.append("init |0>", {0, 1})
    qc.append("H", {0})
    qc.append("CNOT", {(0, 1)})
    qc.append("measure Z", {0}, var=("m", 0))
    qc.append("measure Z", {1}, var=("m", 1))

    state = SparseSim(2)
    runner = pc.HybridEngine()

    ms = []
    for i in range(10):
        state.reset()
        shot_output, _ = runner.run(state, qc, i)
        ms.append(str(shot_output["m"]))

    assert ms.count("00") + ms.count("11") == len(ms)
