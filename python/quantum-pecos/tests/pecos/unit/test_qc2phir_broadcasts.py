"""QuantumCircuit compaction preserves sequential gates and measurement destinations."""

import json

import pecos as pc
import pytest


@pytest.mark.parametrize("symbol", ["Measure", "measure Z"])
def test_multi_qubit_measurement_rejects_single_var(symbol) -> None:
    """A single destination cannot describe several measurement results."""
    qc = pc.QuantumCircuit(cvar_spec={"c": 2})
    qc.append(symbol, {0, 1}, var=("c", 0))
    with pytest.raises(ValueError, match=rf"{symbol!r}.*qubits.*0.*1.*var.*c.*var_output"):
        qc.to_phir_json()


@pytest.mark.parametrize("symbol", ["Measure", "measure Z"])
def test_measurement_rejects_missing_var_output(symbol) -> None:
    """Partial per-gate metadata must explain which result is missing."""
    qc = pc.QuantumCircuit(cvar_spec={"c": 2})
    qc.append(symbol, {0, 1}, var_output={0: ("c", 1)})
    with pytest.raises(ValueError, match=rf"{symbol!r}.*qubits.*0.*1.*var_output.*c.*missing.*1"):
        qc.to_phir_json()


@pytest.mark.parametrize("case", ["repeated_h", "repeated_measure", "cx_ladder"])
def test_round_trip_matches_direct_circuit(case: str) -> None:
    """Overlapping gates in consecutive ticks remain executable in native PHIR."""
    qc = pc.QuantumCircuit(cvar_spec={"m": 3})
    if case == "repeated_h":
        qc.append("H", {0})
        qc.append("H", {0})
        qc.append("measure Z", {0}, var=("m", 0))
        expected = 0
    elif case == "repeated_measure":
        qc.append("X", {0})
        qc.append("measure Z", {0}, var=("m", 0))
        qc.append("measure Z", {0}, var=("m", 1))
        expected = 3
    else:
        qc.append("X", {0})
        qc.append("CX", {(0, 1)})
        qc.append("CX", {(1, 2)})
        qc.append("measure Z", {0, 1, 2}, var_output={q: ("m", 2 - q) for q in range(3)})
        expected = 7

    output, _ = pc.circuit_runners.Standard(seed=1314).run(pc.simulators.SparseStabPy(3), qc)
    direct = 0
    for tick, (name, qubits, metadata) in enumerate(qc.items()):
        if name == "measure Z":
            for qubit in qubits:
                destination = metadata["var"] if "var" in metadata else metadata["var_output"][qubit]
                direct |= output.get(tick, {}).get(qubit, 0) << destination[1]
    assert direct == expected
    results = pc.sim(pc.PhirJson(qc.to_phir_json())).quantum(pc.state_vector()).seed(1314).run(4).to_dict()
    assert results["m"] == [direct] * 4


@pytest.mark.parametrize(
    ("name", "first", "second", "third"),
    [("H", {0}, {1}, {0}), ("CX", {(0, 1)}, {(2, 3)}, {(3, 4)})],
)
def test_compaction_checks_every_qubit_in_the_accumulated_op(name, first, second, third) -> None:
    """Disjoint gates compact; overlap with any earlier target starts a new op."""
    qc = pc.QuantumCircuit()
    for targets in [first, second, third]:
        qc.append(name, targets)
    operations = [op for op in json.loads(qc.to_phir_json())["ops"] if "qop" in op]
    assert len(operations) == 2
    assert len(operations[0]["args"]) == 2
    assert len(operations[1]["args"]) == 1


def test_compacted_measurement_returns_stay_in_argument_order() -> None:
    """A later overlapping measurement keeps its own return destination."""
    qc = pc.QuantumCircuit(cvar_spec={"m": 3})
    for qubit, bit in [(0, 2), (1, 0), (0, 1)]:
        qc.append("measure Z", {qubit}, var=("m", bit))
    operations = [op for op in json.loads(qc.to_phir_json())["ops"] if "qop" in op]
    assert operations == [
        {"qop": "Measure", "args": [["q", 0], ["q", 1]], "returns": [["m", 2], ["m", 0]]},
        {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 1]]},
    ]


def test_partial_measurement_destinations_are_not_merged() -> None:
    """Named and unnamed measurements must keep their different return shapes."""
    qc = pc.QuantumCircuit(cvar_spec={"m": 1})
    qc.append("measure Z", {0})
    qc.append("measure Z", {1}, var=("m", 0))
    operations = [op for op in json.loads(qc.to_phir_json())["ops"] if "qop" in op]
    assert len(operations) == 2
    assert "returns" not in operations[0]
    assert operations[1]["returns"] == [["m", 0]]


def test_register_names_are_part_of_the_qubit_identity() -> None:
    """Equal indices in distinct quantum registers are disjoint targets."""
    qc = pc.QuantumCircuit(qvar_spec={"a": 1, "b": 1}, qvars={0: ("a", 0), 1: ("b", 0)})
    qc.append("H", {0})
    qc.append("H", {1})
    operations = [op for op in json.loads(qc.to_phir_json())["ops"] if "qop" in op]
    assert operations == [{"qop": "H", "args": [["a", 0], ["b", 0]]}]


def test_export_compaction_does_not_absorb_a_quantum_operation() -> None:
    """Only consecutive exports can share their variables list."""
    qc = pc.QuantumCircuit(cvar_spec={"a": 1, "b": 1})
    qc.append("cop", set(), cop_type="ExportCVar", export="a")
    qc.append("cop", set(), cop_type="ExportCVar", export="b")
    qc.append("H", {0})
    operations = json.loads(qc.to_phir_json())["ops"]
    assert operations[-2:] == [
        {"data": "cvar_export", "variables": ["a", "b"]},
        {"qop": "H", "args": [["q", 0]]},
    ]


def test_compaction_visits_each_qubit_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compacting many ticks must not rescan the accumulated operation."""
    from pecos.circuits import qc2phir
    from pecos.circuits.qc2phir import _qop_qubits as original

    visits = 0

    def count_qubits(op):
        nonlocal visits
        qubits = original(op)
        visits += len(qubits)
        return qubits

    monkeypatch.setattr(qc2phir, "_qop_qubits", count_qubits)
    count = 2000
    qc = pc.QuantumCircuit()
    for qubit in range(count):
        qc.append("H", {qubit})
    operations = [op for op in qc2phir.to_phir_dict(qc)["ops"] if "qop" in op]
    assert len(operations) == 1
    assert len(operations[0]["args"]) == count
    assert visits == count


def test_compaction_resets_qubits_when_a_new_operation_starts() -> None:
    """A different gate starts a fresh set of occupied targets."""
    qc = pc.QuantumCircuit()
    qc.append("H", {0})
    qc.append("X", {1})
    qc.append("X", {0})
    operations = [op for op in json.loads(qc.to_phir_json())["ops"] if "qop" in op]
    assert operations == [
        {"qop": "H", "args": [["q", 0]]},
        {"qop": "X", "args": [["q", 1], ["q", 0]]},
    ]


@pytest.mark.parametrize("cinterp", ["rust", "python"])
def test_zero_target_quantum_operations_are_omitted(cinterp: str) -> None:
    """Empty circuit operations do nothing on either HybridEngine path."""
    from pecos.engines.hybrid_engine import HybridEngine

    qc = pc.QuantumCircuit(cvar_spec={"m": 1})
    qc.append("H", set())
    qc.append("X", {0})
    qc.append("measure Z", set())
    qc.append("measure Z", {0}, var=("m", 0))
    qc.append("CX", set())
    source = qc.to_phir_dict()
    assert all(op["args"] for op in source["ops"] if "qop" in op)
    result = HybridEngine(cinterp=cinterp, qsim="stabilizer").run(qc, return_int=True)
    assert result["m"] == [1]
