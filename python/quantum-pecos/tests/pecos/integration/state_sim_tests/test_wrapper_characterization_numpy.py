"""NumPy wrapper pins recorded on dev ad1e97642, before #1480 batch validation.

NumPy is installed by the workspace test dependency group. This module groups
NumPy scalar and array location cases separately from the main wrapper corpus.
"""

import numpy as np
import pytest
from pecos import simulators
from pecos.simulators import CoinToss, PauliProp


@pytest.fixture(params=["StateVec", "SparseStabPy", "PauliProp", "CoinToss"])
def state(request):
    params = {"num_qubits": 2}
    if request.param in {"StateVec", "CoinToss"}:
        params["seed"] = 42
    if request.param == "CoinToss":
        params["prob"] = 1
    return getattr(simulators, request.param)(**params)


def _prepare_one(state):
    if isinstance(state, PauliProp):
        state.faults = {"X": {1}, "Y": set(), "Z": set()}
    else:
        state.run_gate("X", {1})


def test_numpy_integer_location(state):
    _prepare_one(state)
    location = np.int64(1)
    result = state.run_gate("Measure", [location])
    if isinstance(state, CoinToss):
        # CoinToss run_gate assertions are input-independent no-op pins.
        assert result == {}
    else:
        assert result == {1: 1}
        assert next(iter(result)) is location


def test_numpy_bare_integer(state):
    # KNOWN-WRONG (#1515): Bare NumPy zero silently skips dispatch while nonzero raises.
    assert state.run_gate("X", np.int64(0)) == {}
    if isinstance(state, CoinToss):
        assert state.run_gate("X", np.int64(1)) == {}
    else:
        with pytest.raises(TypeError, match=r"^'numpy\.int64' object is not iterable$"):
            state.run_gate("X", np.int64(1))


def test_numpy_pair_tuple(state):
    if isinstance(state, PauliProp):
        state.faults = {"X": {0}, "Y": set(), "Z": set()}
    else:
        state.run_gate("X", {0})
    assert state.run_gate("CX", [(np.int64(0), np.int64(1))]) == {}
    assert state.run_gate("Measure", [0, 1]) == ({} if isinstance(state, CoinToss) else {0: 1, 1: 1})


def test_numpy_zero_dimensional_location(state):
    location = np.array(1)
    if isinstance(state, PauliProp):
        with pytest.raises(TypeError, match=r"^unhashable type: 'numpy\.ndarray'$"):
            state.run_gate("X", [location])
    else:
        # KNOWN-WRONG (#1480): An unhashable 0-d location is accepted for mutation, then fails on result insertion.
        assert state.run_gate("X", [location]) == {}
        if isinstance(state, CoinToss):
            assert state.run_gate("Measure", [location]) == {}
        else:
            with pytest.raises(TypeError, match=r"^unhashable type: 'numpy\.ndarray'$"):
                state.run_gate("Measure", [location])
            assert state.run_gate("Measure", {1}) == {1: 1}


def test_numpy_bare_zero_dimensional_array(state):
    # KNOWN-WRONG (#1515): Bare 0-d zero skips dispatch, but a nonzero 0-d array raises.
    assert state.run_gate("X", np.array(0)) == {}
    if isinstance(state, CoinToss):
        assert state.run_gate("X", np.array(1)) == {}
    else:
        with pytest.raises(TypeError, match=r"^iteration over a 0-d array$"):
            state.run_gate("X", np.array(1))
