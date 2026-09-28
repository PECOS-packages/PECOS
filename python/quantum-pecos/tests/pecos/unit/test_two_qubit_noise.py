"""Regression tests for two-qubit noise eligibility (#816 and #817)."""

import pecos as pc
import pytest
from pecos.machines.generic_machine import GenericMachine
from pecos.noise.noise_impl.noise_tq_depolarizing_leakage import noise_tq_depolarizing_leakage
from pecos.noise.noise_impl_old.tq_noise import noise_two_qubit_gates_depolarizing_with_noiseless
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

    noise = noise_tq_depolarizing_leakage(op, p=1, noise_dict={faults: 1.0}, machine=machine)

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
            marks=pytest.mark.xfail(strict=True, reason="pc.random.choice rejects an integer; see #889"),
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
    assert all(symbol in {"X", "Y", "Z"} for symbol, _, _ in errors)
