"""Regression tests for noise eligibility and caller emission slots."""

import pecos as pc
import pytest
from pecos.machines.generic_machine import GenericMachine
from pecos.noise.depolarizing_error_model import DepolarizingErrorModel
from pecos.noise.generic_error_model import GenericErrorModel
from pecos.noise.noise_impl.noise_tq_depolarizing_leakage import (
    noise_tq_depolarizing_leakage,
    surviving_two_qubit_op,
)
from pecos.noise.noise_impl_old.tq_noise import (
    noise_depolarizing_two_qubit_gates,
    noise_two_qubit_gates_depolarizing_with_noiseless,
)
from pecos.noise.parent_class_error_gen import Generator
from pecos.reps.pyphir.op_types import QOp


@pytest.mark.parametrize("faults", [("X", "Z"), ("L", "Z"), ("X", "L")])
@pytest.mark.parametrize(
    ("leaked", "expected_targets"),
    [
        pytest.param({0}, {2, 3}, id="first-qubit-leaked"),
        pytest.param(set(), {0, 1, 2, 3}, id="none-leaked"),
        pytest.param({0, 1}, {2, 3}, id="both-qubits-leaked"),
        pytest.param({1}, {2, 3}, id="second-qubit-leaked"),
        pytest.param({0, 2}, set(), id="no-eligible-pairs"),
    ],
)
def test_leakage_noise_targets_healthy_pairs(
    leaked: set[int],
    expected_targets: set[int],
    faults: tuple[str, str],
) -> None:
    """Either leaked input excludes the entire pair from noise eligibility."""
    pc.random.seed(42)
    machine = GenericMachine(num_qubits=4)
    machine.leaked_qubits.update(leaked)
    op = QOp(name="CNOT", args=[(0, 1), (2, 3)], metadata={})

    model = GenericErrorModel(
        {"p1": 0, "p2": 1, "p_meas": 0, "p_prep": 0, "p2_error_model": {faults: 1.0}},
    )
    model.init(4, machine)
    result = model.process([op])
    gates = [item for item in result if item.name == "CNOT"]
    assert [gate.args for gate in gates] == (
        [[(2, 3)]] if leaked and expected_targets else [op.args] if expected_targets else []
    )
    noise = [item for item in result if item.name != "CNOT"]

    # GenericMachine.leak emits Init operations; include their targets too.
    targets = {qubit for error in noise for qubit in error.args} if noise is not None else set()
    assert targets == expected_targets
    newly_leaked = set()
    for pair in op.args:
        if set(pair) <= expected_targets:
            newly_leaked.update(qubit for qubit, fault in zip(pair, faults, strict=True) if fault == "L")
    assert machine.leaked_qubits == leaked | newly_leaked


@pytest.mark.parametrize(
    ("noiseless_qubits", "expected_targets"),
    [
        pytest.param({0}, {1}, id="first-qubit-noiseless"),
        pytest.param({1}, {0}, id="second-qubit-noiseless"),
        pytest.param({0, 1}, set(), id="both-qubits-noiseless"),
        pytest.param(
            set(),
            {0, 1},
            id="neither-qubit-noiseless",
        ),
    ],
)
def test_depolarizing_noise_targets_noisy_qubits(
    noiseless_qubits: set[int],
    expected_targets: set[int],
) -> None:
    """A noiseless input protects itself without protecting its noisy partner."""
    pc.random.seed(42)
    after = pc.QuantumCircuit()

    noise_two_qubit_gates_depolarizing_with_noiseless(
        {(0, 1)},
        after,
        p=1,
        noiseless_qubits=noiseless_qubits,
    )

    errors = list(after.items())
    assert {qubit for _, locations, _ in errors for qubit in locations} == expected_targets
    assert len(errors) == len(expected_targets)
    assert all(symbol in {"I", "X", "Y", "Z"} for symbol, _, _ in errors)


@pytest.mark.parametrize("model_type", [GenericErrorModel, DepolarizingErrorModel])
@pytest.mark.parametrize("gate_fault", ["XI", "II"])
@pytest.mark.parametrize("memory_fault", ["IZ", "II"])
def test_gate_and_memory_noise_merge(model_type: type, gate_fault: str, memory_fault: str) -> None:
    """Both draws survive merging, including either draw producing no operations (#893)."""
    model = model_type(
        {
            "p1": 0,
            "p2": 1,
            "p_meas": 0,
            "p_prep": 0,
            "p2_mem": 1,
            "p2_error_model": {gate_fault: 1.0},
            "p2_mem_error_model": {memory_fault: 1.0},
        },
    )
    model.init(2, GenericMachine(num_qubits=2))
    op = QOp(name="CNOT", args=[(0, 1)], metadata={})
    expected = [("CNOT", [(0, 1)])]
    if gate_fault != "II":
        expected.append(("X", [0]))
    if memory_fault != "II":
        expected.append(("Z", [1]))
    assert [(item.name, item.args) for item in model.process([op])] == expected


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        ("H", [0], [("H", [0]), ("X", [0])]),
        ("Measure", [0], [("Measure", [0])]),
        ("Init", [0], [("Init", [0]), ("X", [0])]),
        ("CNOT", [(0, 1)], [("CNOT", [(0, 1)]), ("X", [0])]),
    ],
)
def test_process_emission_slots(name: str, args: list, expected: list) -> None:
    """Single-qubit/measurement results replace; init/two-qubit noise follows the gate."""
    model = GenericErrorModel(
        {"p1": 1, "p2": 1, "p_meas": 1, "p_prep": 1, "p1_error_model": {"X": 1}, "p2_error_model": {"XI": 1}},
    )
    model.init(2, GenericMachine(num_qubits=2))
    op = QOp(name=name, args=args, returns=["result"], metadata={"tag": "preserved"})
    result = model.process([op])
    assert [(item.name, item.args) for item in result] == expected
    if name == "Measure":
        assert result[0] is not op
        assert result[0].metadata == {"tag": "preserved", "bitflips": [0]}
        assert result[0].returns == ["result"]
    else:
        assert result[0] is op


@pytest.mark.parametrize("leaked", [set(), {0}, {1}, {0, 2}])
def test_surviving_op_is_pure(leaked: set[int]) -> None:
    """Filtering preserves operation attributes, machine state, and the random stream."""
    machine = GenericMachine(num_qubits=4)
    machine.leaked_qubits.update(leaked)
    op = QOp("RZZ", [(0, 1), (2, 3)], returns=["result"], metadata={"tag": 1}, angles=(0.5,), sim_name="RZZ")
    pc.random.seed(42)
    expected_random = list(pc.random.random(4))
    pc.random.seed(42)
    result = surviving_two_qubit_op(op, machine)
    assert list(pc.random.random(4)) == expected_random
    assert machine.leaked_qubits == leaked
    assert op.args == [(0, 1), (2, 3)]
    if leaked == {0, 2}:
        assert result is None
    else:
        assert result.args == ([(2, 3)] if leaked else op.args)
        assert (result.name, result.returns, result.metadata, result.angles, result.sim_name) == (
            op.name,
            op.returns,
            op.metadata,
            op.angles,
            op.sim_name,
        )


@pytest.mark.parametrize("p2", [0.4, 1.0])
@pytest.mark.parametrize("seed", [0, 42, 1234])
@pytest.mark.parametrize("leaked", [set(), {0}, {0, 2}])
def test_seeded_gate_leakage_excludes_memory(seed: int, leaked: set[int], p2: float) -> None:
    """Each call sees the same pair count and random stream as legacy internal filtering."""
    op = QOp("CNOT", [(0, 1), (2, 3)])
    machine = GenericMachine(num_qubits=4)
    machine.leaked_qubits.update(leaked)
    model = GenericErrorModel(
        {
            "p1": 0,
            "p2": p2,
            "p_meas": 0,
            "p_prep": 0,
            "p2_mem": 0.7,
            "p2_error_model": {("L", "I"): 1},
            "p2_mem_error_model": {("X", "Z"): 1},
        },
    )
    model.init(4, machine)
    pc.random.seed(seed)
    result = model.process([op])
    next_random = list(pc.random.random(4))

    legacy_machine = GenericMachine(num_qubits=4)
    legacy_machine.leaked_qubits.update(leaked)
    pc.random.seed(seed)
    # Reproduce the old helper's filter immediately before each draw, even for zero pairs.
    gate_args = [[a, b] for a, b in op.args if a not in leaked and b not in leaked]
    expected = [("CNOT", gate_args)] if gate_args else []
    for probability, faults in [(p2 * 5 / 4, {("L", "I"): 1}), (0.7, {("X", "Z"): 1})]:
        args = [
            [a, b]
            for a, b in op.args
            if a not in legacy_machine.leaked_qubits and b not in legacy_machine.leaked_qubits
        ]
        noise = noise_tq_depolarizing_leakage(QOp("CNOT", args), probability, faults, legacy_machine)
        expected.extend((item.name, item.args) for item in noise or [])
    actual = [(item.name, [list(pair) for pair in item.args] if item.name == "CNOT" else item.args) for item in result]
    assert actual == expected
    assert machine.leaked_qubits == legacy_machine.leaked_qubits
    assert next_random == list(pc.random.random(4))


@pytest.mark.parametrize("with_noiseless", [False, True])
def test_integer_choice_legacy_two_qubit_entry_points(with_noiseless: bool) -> None:
    """Both legacy helpers sample the Pauli population through integer choice."""
    pc.random.seed(42)
    after = pc.QuantumCircuit()
    if with_noiseless:
        noise_two_qubit_gates_depolarizing_with_noiseless({(0, 1)}, after, 1, set())
    else:
        noise_depolarizing_two_qubit_gates({(0, 1)}, after, 1)
    assert [(symbol, locations) for symbol, locations, _ in after.items()] == [("I", {0}), ("X", {1})]


@pytest.mark.parametrize("after_gate", [False, True])
def test_integer_choice_multi_qudit_entry_points(after_gate: bool) -> None:
    """The configured error function samples before and after multi-qudit gates."""
    error = Generator.ErrorSetMultiQuditGate(
        [(pc.Pauli.X, pc.Pauli.Z), (pc.Pauli.Y, pc.Pauli.X)],
        after=after_gate,
    )
    after, before = pc.QuantumCircuit(), pc.QuantumCircuit()
    pc.random.seed(42)
    error.error_func(after, before, set(), (0, 1), {})
    actual = [[(symbol, locations) for symbol, locations, _ in circuit.items()] for circuit in (before, after)]
    expected = [("X", {0}), ("Z", {1})]
    assert actual == ([[], expected] if after_gate else [expected, []])
