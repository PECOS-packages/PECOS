"""PHIR scheduling markers reach the noise model and never reach the simulator."""

import pytest
from pecos.engines.hybrid_engine import HybridEngine
from pecos.noise.depolarizing_error_model import DepolarizingErrorModel
from pecos.noise.generic_error_model import GenericErrorModel
from pecos.reps.pyphir.op_types import MOp


@pytest.mark.parametrize("model_type", [GenericErrorModel, DepolarizingErrorModel])
@pytest.mark.parametrize("name", ["Idle", "Transport"])
@pytest.mark.parametrize("cinterp", ["python", "rust"])
def test_phir_timing_consumed(
    model_type: type,
    name: str,
    cinterp: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real PHIR passes scheduling metadata through the engine before consuming the marker."""
    program = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 1},
            {"data": "cvar_define", "data_type": "u32", "variable": "m", "size": 1},
            {"qop": "Init", "args": [["q", 0]], "metadata": {}},
            {"qop": "X", "args": [["q", 0]], "metadata": {}},
            {"mop": name, "args": [["q", 0]], "duration": [20, "ns"]},
            {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]], "metadata": {}},
        ],
    }
    model = model_type({"p1": 0, "p2": 0, "p_meas": 0, "p_prep": 0})
    process = model.process
    observed = []

    def record_process(ops: list) -> list:
        observed.extend((type(op), op.name, op.args, op.metadata["duration"]) for op in ops if op.name == name)
        return process(ops)

    monkeypatch.setattr(model, "process", record_process)
    result = HybridEngine(cinterp=cinterp, qsim="stabilizer", error_model=model).run(program, shots=2)
    assert result == {"m": ["1", "1"]}
    assert observed == [(MOp, name, [0], [20, "ns"])] * 2
