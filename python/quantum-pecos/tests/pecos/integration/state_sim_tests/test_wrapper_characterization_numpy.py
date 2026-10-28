"""NumPy wrapper pins recorded on dev ad1e97642, updated for #1480 batch validation.

NumPy is not declared in the workspace test dependency group (quantum-pecos
inherits it), so this separate module alone uses a module-level importorskip.
"""

import pecos
import pytest
from pecos import simulators
from pecos.simulators import CoinToss, PauliProp

np = pytest.importorskip("numpy")


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
    # Deliberate #1493 flip: CoinToss now returns its measurement, preserving the key.
    assert result == {1: 1}
    assert next(iter(result)) is location


def test_numpy_bare_integer(state):
    # KNOWN-WRONG (#1480): Bare NumPy zero silently skips dispatch while nonzero raises.
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
    assert state.run_gate("Measure", [0, 1]) == {0: 1, 1: 1}


def test_numpy_zero_dimensional_location(state):
    location = np.array(1)
    if isinstance(state, CoinToss):
        assert state.run_gate("X", [location]) == {}
        # Deliberate #1493 flip: measurement validates before drawing coins.
        with pytest.raises(ValueError, match="Measure: expected a hashable qubit index"):
            state.run_gate("Measure", [location])
    else:
        # Unhashable qubits are rejected before either mutation or measurement.
        for symbol in ("X", "Measure"):
            with pytest.raises(ValueError, match=f"{symbol}: expected a hashable qubit index"):
                state.run_gate(symbol, [location])
        assert state.run_gate("Measure", {1}) == {}


def test_numpy_bare_zero_dimensional_array(state):
    # KNOWN-WRONG (#1480): Bare 0-d zero skips dispatch, but a nonzero 0-d array raises.
    assert state.run_gate("X", np.array(0)) == {}
    if isinstance(state, CoinToss):
        assert state.run_gate("X", np.array(1)) == {}
    else:
        with pytest.raises(TypeError, match=r"^iteration over a 0-d array$"):
            state.run_gate("X", np.array(1))


@pytest.mark.parametrize("name", ["StateVec", "SparseStabPy", "PauliProp"])
def test_numpy_unhashable_pair_tail_is_atomic(name):
    state = getattr(simulators, name)(num_qubits=4)
    if isinstance(state, PauliProp):
        state.faults = {"X": {0}, "Y": set(), "Z": set()}
    else:
        state.run_gate("X", [0])
    with pytest.raises(ValueError, match="CX: expected a hashable qubit index"):
        state.run_gate("CX", [(0, 1), (np.array(2), 3)])
    assert state.run_gate("Measure", [0, 1]) == {0: 1}
    state.run_gate("CX", [(0, 1)])
    assert state.run_gate("Measure", [0, 1]) == {0: 1, 1: 1}


def test_hybrid_numpy_zero_dimensional_zero_raises():
    with pytest.raises(TypeError, match=r"len\(\) of unsized object"):
        pecos.HybridEngine.run_gate(simulators.SparseStabPy(2), {}, "X", np.array(0))
