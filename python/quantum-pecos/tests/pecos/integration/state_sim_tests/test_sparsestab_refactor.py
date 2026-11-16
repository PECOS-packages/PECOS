"""Regression tests for signed stabilizer generator refactoring."""

from copy import deepcopy

import pytest
from pecos.simulators.sparsestab import SparseStabPy


@pytest.mark.parametrize("forced_outcome", [0, 1])
def test_refactor_graph_state_preserves_parity(forced_outcome: int) -> None:
    """Refactoring +YY into the graph state's basis preserves its parity."""
    original = SparseStabPy(2)
    original.run_gate("H", {0, 1})
    original.run_gate("CZ", {(0, 1)})
    refactored = deepcopy(original)

    found, new_stab = refactored.refactor({0, 1}, {0, 1}, choose=0)
    assert found
    assert refactored.stabs.row_x[new_stab] == {0, 1}
    assert refactored.stabs.row_z[new_stab] == {0, 1}

    outcomes = []
    for state in (original, refactored):
        state.run_gate("Sd", {0, 1})
        state.run_gate("H", {0, 1})
        outcomes.append(
            [state.run_gate("measure Z", {q}, forced_outcome=forced_outcome).get(q, 0) for q in (0, 1)],
        )

    assert outcomes[0] == [forced_outcome, forced_outcome]
    assert outcomes[1] == outcomes[0]


def test_refactor_with_default_selection() -> None:
    """Refactoring with no choose, prefer, or protected picks a source generator."""
    state = SparseStabPy(2)
    state.run_gate("H", {0, 1})
    state.run_gate("CZ", {(0, 1)})

    found, new_stab = state.refactor({0, 1}, {0, 1})
    assert found
    assert state.stabs.row_x[new_stab] == {0, 1}
    assert state.stabs.row_z[new_stab] == {0, 1}
