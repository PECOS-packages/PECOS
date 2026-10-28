# Copyright 2024 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Integration tests for coin toss quantum simulator."""

from __future__ import annotations

from types import SimpleNamespace

import pecos as pc
import pytest
from pecos.circuits import QuantumCircuit
from pecos.engines.circuit_runners.timing_runner import TimingRunner
from pecos.exceptions import NotSupportedGateError
from pecos.simulators import CoinToss
from pecos.simulators.batch_validation import GATE_ARITIES
from pecos_rslib import _gate_bindings_symbols

MEASURE_SYMBOLS = (
    "Measure",
    "measure Z",
    "Measure +Z",
    "Measure -Z",
    "Measure +X",
    "Measure -X",
    "Measure +Y",
    "Measure -Y",
    "MX",
    "MY",
    "MZ",
)


def _items(symbol, locations, **params):
    return SimpleNamespace(items=lambda: iter([(symbol, locations, params)]))


def _dispatch(state, route, symbol, locations, **params):
    if route == "gate":
        return state.run_gate(symbol, locations, **params)
    if route == "hybrid":
        output = {"m": pc.BitInt(4, 0)}
        pc.HybridEngine.run_gate(
            state,
            output,
            symbol,
            locations,
            var_output={q: ("m", q) for q in range(4)},
            **params,
        )
        return {q: 1 for q in range(4) if output["m"][q]}
    return state.run_circuit(_items(symbol, locations, **params))


@pytest.mark.parametrize("route", ["gate", "circuit", "hybrid"])
@pytest.mark.parametrize("symbol", [*MEASURE_SYMBOLS, "MEASURE custom"])
@pytest.mark.parametrize("container", [list, tuple, set, frozenset, iter])
@pytest.mark.parametrize("simulate_gate", [True, False])
def test_measurement_containers_follow_coin_order(route, symbol, container, simulate_gate):
    """Every measurement route consumes exactly the caller's ordered coin stream."""
    state = CoinToss(4, seed=42)
    fresh = CoinToss(4, seed=42)
    if route == "hybrid" and symbol not in state.bindings and simulate_gate:
        # HybridEngine dispatches through bindings: an unbound spelling is loud.
        with pytest.raises(NotSupportedGateError):
            _dispatch(state, route, symbol, container([3, 1, 2, 0]), simulate_gate=simulate_gate)
        return
    order = list(container([3, 1, 2, 0]))
    expected = {}
    if simulate_gate:
        for q in order:
            if fresh.mz(q):
                expected[q] = 1
    assert _dispatch(state, route, symbol, container(order), simulate_gate=simulate_gate) == expected
    assert [state.mz(0) for _ in range(32)] == [fresh.mz(0) for _ in range(32)]


@pytest.mark.parametrize("route", ["gate", "circuit"])
@pytest.mark.parametrize("container", [list, tuple, iter])
def test_repeated_measurements_draw_each_coin(route, container):
    """Repeated locations are not deduplicated, and recorded ones are retained."""
    state = CoinToss(4, seed=42)
    fresh = CoinToss(4, seed=42)
    locations = [3, 1, 2, 0, 1]
    expected = {}
    for q in locations:
        if fresh.mz(q):
            expected[q] = 1
    assert _dispatch(state, route, "MZ", container(locations)) == expected
    assert [state.mz(0) for _ in range(32)] == [fresh.mz(0) for _ in range(32)]


@pytest.mark.parametrize("route", ["gate", "circuit"])
@pytest.mark.parametrize("location", [0, 3])
@pytest.mark.parametrize("simulate_gate", [True, False])
def test_scalar_measurement_coin_stream(route, location, simulate_gate):
    state = CoinToss(4, seed=42)
    fresh = CoinToss(4, seed=42)
    expected = {location: 1} if simulate_gate and fresh.mz(location) else {}
    assert _dispatch(state, route, "MX", location, simulate_gate=simulate_gate) == expected
    assert [state.mz(0) for _ in range(32)] == [fresh.mz(0) for _ in range(32)]


def test_timing_runner_receives_measurement_results():
    """The gate-based runner now receives the wrapper's outcome-1 results."""
    state = CoinToss(4, seed=42)
    fresh = CoinToss(4, seed=42)
    expected = {}
    locations = {0, 1, 2, 3}
    for q in locations:
        if fresh.mz(q):
            expected[q] = 1
    assert TimingRunner(seed=42).run_gates(state, _items("MZ", locations)) == expected
    assert [state.mz(0) for _ in range(32)] == [fresh.mz(0) for _ in range(32)]


@pytest.mark.parametrize("route", ["gate", "circuit", "hybrid"])
@pytest.mark.parametrize(
    ("symbol", "binding", "measures"),
    [
        ("custom coin", "Measure", True),
        ("CX", "Measure", True),
        ("Measure", "X", False),
        ("MEASURE custom", None, True),
        ("MX", "X", False),
    ],
)
def test_measurement_rule_follows_bindings_or_prefix(route, symbol, binding, measures):
    """Rebindings, unknown aliases, and prefix fallback have the same dispatch rule."""
    state = CoinToss(4, seed=42)
    fresh = CoinToss(4, seed=42)
    if binding is not None:
        state.bindings[symbol] = state.bindings[binding]
    if route == "hybrid" and symbol not in state.bindings:
        # HybridEngine dispatches through bindings: an unbound spelling is loud.
        with pytest.raises(NotSupportedGateError):
            _dispatch(state, route, symbol, [3, 1, 2, 0])
        return
    expected = {}
    if measures:
        for q in [3, 1, 2, 0]:
            if fresh.mz(q):
                expected[q] = 1
    assert _dispatch(state, route, symbol, [3, 1, 2, 0]) == expected
    assert [state.mz(0) for _ in range(32)] == [fresh.mz(0) for _ in range(32)]
    if measures:
        with pytest.raises(ValueError, match="qubit 4"):
            _dispatch(state, route, symbol, [0, 4], simulate_gate=False)
        with pytest.raises(ValueError, match="expected 1 qubit"):
            _dispatch(state, route, symbol, [(0, 1)])
    else:
        if route == "hybrid":
            pc.HybridEngine.run_gate(state, {}, symbol, [0, 4])
        else:
            assert _dispatch(state, route, symbol, [0, 4]) == {}


@pytest.mark.parametrize("route", ["gate", "circuit"])
@pytest.mark.parametrize("symbol", ["MX", "MY", "MZ", "measure custom"])
@pytest.mark.parametrize("container", [list, tuple, set, frozenset, iter])
@pytest.mark.parametrize("simulate_gate", [True, False])
@pytest.mark.parametrize("prob", [0.5, 1.0])
def test_all_measurement_batches_are_atomic(route, symbol, container, simulate_gate, prob, monkeypatch):
    state = CoinToss(4, seed=42, prob=prob)
    fresh = CoinToss(4, seed=42, prob=prob)
    assert list(container([0, 4])) == [0, 4]
    calls = []
    original = state.mz

    def record_measurement(qubit):
        calls.append(qubit)
        return original(qubit)

    monkeypatch.setattr(state, "mz", record_measurement)
    with pytest.raises(ValueError, match="qubit 4"):
        _dispatch(state, route, symbol, container([0, 4]), simulate_gate=simulate_gate)
    assert calls == []
    state.prob = fresh.prob = 0.5
    assert [state.mz(0) for _ in range(32)] == [fresh.mz(0) for _ in range(32)]


@pytest.mark.parametrize("symbol", MEASURE_SYMBOLS)
@pytest.mark.parametrize("simulate_gate", [True, False])
@pytest.mark.parametrize("prob", [0.5, 1.0])
def test_measurement_batch_failure_preserves_results_and_rng(symbol, simulate_gate, prob, monkeypatch):
    """An invalid second location must not record a one or consume a coin toss."""
    state = CoinToss(4, prob=prob, seed=42)
    fresh = CoinToss(4, prob=prob, seed=42)
    locations = {0, 4}
    assert list(locations) == [0, 4]  # Ensure the invalid location really is second.

    calls = []
    original = state.mz

    def record_measurement(qubit):
        calls.append(qubit)
        return original(qubit)

    monkeypatch.setattr(state, "mz", record_measurement)
    # Seed a real result with an earlier, valid batch. The failing batch must
    # neither invoke a measurement nor change the accumulated result.
    items = [("MZForced", [3], {"forced_outcome": 1}), (symbol, locations, {"simulate_gate": simulate_gate})]
    circuit = SimpleNamespace(items=lambda: iter(items))
    with pytest.raises(ValueError, match="qubit 4") as error:
        state.run_circuit(circuit)
    assert calls == []
    frame = next(entry.frame for entry in error.traceback if entry.name == "run_circuit")
    assert frame.f_locals["measurement_results"] == {3: 1}
    state.prob = fresh.prob = 0.5
    probe = _items("Measure", {0, 1, 2, 3})
    assert [state.run_circuit(probe) for _ in range(32)] == [fresh.run_circuit(probe) for _ in range(32)]
    assert symbol in str(error.value)


@pytest.mark.parametrize("location", [-1, 4])
def test_bare_int_measurement_is_validated(location):
    """Scalar measurements get the same gate-named validation as set batches."""
    with pytest.raises(ValueError, match="Measure -Z"):
        CoinToss(4).run_circuit(_items("Measure -Z", location))


@pytest.mark.parametrize("symbol", [*MEASURE_SYMBOLS, "MEASURE Z", "measure custom"])
@pytest.mark.parametrize(
    ("locations", "expected"),
    [
        (
            {0, 1, 2, 3},
            [{1: 1, 2: 1, 3: 1}, {3: 1}, {1: 1}, {2: 1}, {0: 1, 1: 1, 2: 1, 3: 1}, {1: 1}, {}, {1: 1, 3: 1}],
        ),
        (0, [{}, {0: 1}, {0: 1}, {0: 1}, {}, {}, {}, {0: 1}]),
    ],
)
def test_valid_measurement_batches_match_seeded_coin_stream(symbol, locations, expected):
    """Enabled measurement spellings share the pinned output and RNG stream."""
    state = CoinToss(4, seed=42)
    circuit = _items(symbol, locations)
    assert [state.run_circuit(circuit) for _ in expected] == expected


@pytest.mark.parametrize("route", ["gate", "circuit"])
@pytest.mark.parametrize("symbol", ["X", "CX", "unknown"])
def test_nonmeasurement_batches_preserve_rng(route, symbol):
    """Only non-measurement symbols ignore bad locations without drawing coins."""
    state = CoinToss(4, seed=42)
    fresh = CoinToss(4, seed=42)
    assert _dispatch(state, route, symbol, [0, 4]) == {}
    assert [state.mz(0) for _ in range(32)] == [fresh.mz(0) for _ in range(32)]


def test_circuit_validation_uses_instance_measurement_binding():
    """A non-prefixed alias of a measurement binding is pre-checked."""
    state = CoinToss(4, seed=42)
    state.bindings["custom coin"] = state.bindings["MX"]
    with pytest.raises(ValueError, match="custom coin"):
        state.run_circuit(_items("custom coin", {0, 4}))


class IndexScalar:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


@pytest.mark.parametrize("route", ["gate", "circuit", "hybrid"])
@pytest.mark.parametrize("symbol", ["MZ", "X", "custom coin", "measure custom"])
def test_custom_binding_dispatches_and_returns_results(route, symbol):
    state = CoinToss(4, prob=1, seed=42)
    calls = []

    def measure(sim, qubit, **params):
        calls.append((qubit, params["tag"]))
        return sim.mz(qubit)

    state.bindings[symbol] = measure
    assert _dispatch(state, route, symbol, [2, 0, 2], tag="kept") == {0: 1, 2: 1}
    assert calls == [(2, "kept"), (0, "kept"), (2, "kept")]
    calls.clear()
    assert _dispatch(state, route, symbol, [0], tag="kept", simulate_gate=False) == {}
    assert calls == []


@pytest.mark.parametrize("route", ["gate", "circuit", "hybrid"])
@pytest.mark.parametrize("symbol", ["MZ", "X", "measure custom"])
@pytest.mark.parametrize("disabled", [False, True])
def test_custom_binding_known_arity_is_atomic(route, symbol, disabled):
    state = CoinToss(4, seed=42)
    calls = []

    def measure(sim, qubit, **_params):
        calls.append(qubit)
        return sim.mz(qubit)

    state.bindings[symbol] = measure
    with pytest.raises(ValueError, match=f"{symbol}: qubit 4"):
        _dispatch(state, route, symbol, [0, 4], simulate_gate=not disabled)
    assert calls == []


@pytest.mark.parametrize("route", ["gate", "circuit"])
def test_custom_binding_unknown_arity_and_known_pair(route):
    state = CoinToss(4)
    calls = []

    def pair_binding(_state, location, **_params):
        calls.append(location)
        return 7

    state.bindings["custom pair"] = pair_binding
    state.bindings["CX"] = pair_binding
    assert _dispatch(state, route, "custom pair", [(0, 1)]) == {(0, 1): 7}
    assert _dispatch(state, route, "CX", [(1, 2)]) == {(1, 2): 7}
    assert calls == [(0, 1), (1, 2)]
    with pytest.raises(ValueError, match="CX requires distinct qubits"):
        _dispatch(state, route, "CX", [(0, 1), (2, 2)])
    assert calls == [(0, 1), (1, 2)]


@pytest.mark.parametrize("route", ["gate", "circuit", "hybrid"])
@pytest.mark.parametrize("symbol", ["I", "X", "CX", "Measure", "measure custom"])
def test_builtin_noop_binding_keeps_permissive_locations(route, symbol):
    state = CoinToss(4)
    state.bindings[symbol] = state.bindings["X"]
    assert _dispatch(state, route, symbol, [(0, 0), -1, 99, [0], None], simulate_gate=False) == {}
    # HybridEngine's output lookup is irrelevant to a no-op; use no output mapping.
    if route == "hybrid":
        pc.HybridEngine.run_gate(state, {}, symbol, [(0, 0), -1, 99, [0], None])
    else:
        assert _dispatch(state, route, symbol, [(0, 0), -1, 99, [0], None]) == {}


NATIVE_MEASUREMENTS = [entry for entry in _gate_bindings_symbols() if entry[1]]


@pytest.mark.parametrize("route", ["gate", "circuit", "hybrid"])
@pytest.mark.parametrize(("symbol", "measurement", "mode", "arity"), NATIVE_MEASUREMENTS)
def test_every_native_measurement_has_a_working_binding(route, symbol, measurement, mode, arity):
    state = CoinToss(4, prob=1)
    assert measurement
    assert arity == GATE_ARITIES[symbol] == 1
    assert symbol in state.bindings
    params = {"forced_outcome": -1} if mode == "forced_outcome" else {}
    assert _dispatch(state, route, symbol, [0, 2], **params) == {0: 1, 2: 1}


@pytest.mark.parametrize("route", ["gate", "circuit", "hybrid"])
@pytest.mark.parametrize("symbol", [entry[0] for entry in NATIVE_MEASUREMENTS if "forced_outcome" in entry[2]])
@pytest.mark.parametrize("outcome", [-1, 0, 1, False, True, IndexScalar(1)])
def test_native_forced_outcomes_and_rng(route, symbol, outcome):
    state = CoinToss(4, seed=42)
    fresh = CoinToss(4, seed=42)
    value = outcome.value if isinstance(outcome, IndexScalar) else outcome
    expected = {}
    for q in [2, 0]:
        if fresh.mz(q) if value == -1 else value:
            expected[q] = 1
    assert _dispatch(state, route, symbol, [2, 0], forced_outcome=outcome) == expected
    assert [state.mz(0) for _ in range(32)] == [fresh.mz(0) for _ in range(32)]


@pytest.mark.parametrize("route", ["gate", "circuit", "hybrid"])
@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("params", [{}, {"forced_outcome": 2}, {"forced_outcome": None}, {"forced_outcome": 1.0}])
def test_forced_measurement_bad_params_precede_dispatch(route, disabled, params, monkeypatch):
    state = CoinToss(4)
    calls = []
    monkeypatch.setattr(state, "mz", calls.append)
    with pytest.raises(ValueError, match=r"MZForced requires.*forced_outcome"):
        _dispatch(state, route, "MZForced", [0, 1], simulate_gate=not disabled, **params)
    assert calls == []


@pytest.mark.parametrize("route", ["gate", "circuit"])
@pytest.mark.parametrize("value", [0, 2, -1, 4])
@pytest.mark.parametrize("scalar_type", [int, IndexScalar])
def test_integer_like_scalar_batch(route, value, scalar_type):
    state = CoinToss(4, prob=1)
    location = scalar_type(value)
    if value in (-1, 4):
        with pytest.raises(ValueError, match=f"MZ: qubit {value} is out of bounds"):
            _dispatch(state, route, "MZ", location)
    else:
        result = _dispatch(state, route, "MZ", location)
        assert result == {location: 1}
        assert next(iter(result)) is location


def test_fixed_prob() -> None:
    """Test coin toss simulator with fixed probability."""
    qc = QuantumCircuit()
    qc.append({"Init": {0, 1, 2, 3, 4}})
    qc.append({"Measure": {0, 1, 2, 3, 4}})

    # Probability = 0
    sim = CoinToss(len(qc.qudits), prob=0.0)
    results = sim.run_circuit(qc)
    assert len(results) == 0  # No measurement returned 1

    # Probability = 1
    sim = CoinToss(len(qc.qudits), prob=1.0)
    results = sim.run_circuit(qc)
    assert len(results) == len(qc.qudits)  # All measurements returned 1


def test_all_gate_circ() -> None:
    """Test coin toss simulator with all gate circuit."""
    qc = QuantumCircuit()

    # Apply each gate once
    qc.append({"Init": {0, 1, 2, 3, 4}})
    qc.append({"SZZ": {(4, 2)}})
    qc.append({"RX": {0, 2}}, angles=(pc.f64.frac_pi_4,))
    qc.append({"SXXdg": {(0, 3)}})
    qc.append({"RY": {0, 3}}, angles=(pc.f64.pi / 8,))
    qc.append({"RZZ": {(0, 3)}}, angles=(pc.f64.pi / 16,))
    qc.append({"RZ": {1, 4}}, angles=(pc.f64.pi / 16,))
    qc.append({"RXY1Q": {2}}, angles=(pc.f64.pi / 16, pc.f64.frac_pi_2))
    qc.append({"I": {0, 1, 3}})
    qc.append({"X": {1, 2}})
    qc.append({"Y": {3, 4}})
    qc.append({"CY": {(2, 3)}})
    qc.append({"SYY": {(1, 4)}})
    qc.append({"Z": {2, 0}})
    qc.append({"H": {3, 1}})
    qc.append({"RYY": {(2, 1)}}, angles=(pc.f64.pi / 8,))
    qc.append({"SZZdg": {(3, 1)}})
    qc.append({"F": {0, 2, 4}})
    qc.append({"CX": {(0, 1)}})
    qc.append({"Fdg": {3, 1}})
    qc.append({"SYYdg": {(1, 3)}})
    qc.append({"SX": {1, 2}})
    qc.append(
        {"RXXRYYRZZ": {(0, 4)}},
        angles=(pc.f64.frac_pi_4, pc.f64.pi / 16, pc.f64.frac_pi_2),
    )
    qc.append({"SY": {3, 4}})
    qc.append({"SZ": {2, 0}})
    qc.append({"SZdg": {1, 2}})
    qc.append({"CZ": {(1, 3)}})
    qc.append({"SXdg": {3, 4}})
    qc.append({"SYdg": {2, 0}})
    qc.append({"T": {0, 2, 4}})
    qc.append({"SXX": {(0, 2)}})
    qc.append({"SWAP": {(4, 0)}})
    qc.append({"Tdg": {3, 1}})
    qc.append({"RXX": {(1, 3)}}, angles=(pc.f64.frac_pi_4,))
    qc.append({"Q": {1, 4, 2}})
    qc.append({"Qd": {0, 3}})
    qc.append({"R": {0}})
    qc.append({"Rd": {1, 4, 2}})
    qc.append({"S": {0, 3}})
    qc.append({"Sd": {0}})
    qc.append({"H1": {0, 3}})
    qc.append({"H2": {2, 3}})
    qc.append({"H3": {1, 4, 2}})
    qc.append({"H4": {2, 3}})
    qc.append({"H5": {0, 3}})
    qc.append({"H6": {1, 4, 2}})
    qc.append({"H+z+x": {2, 3}})
    qc.append({"H-z-x": {1, 4, 2}})
    qc.append({"H+y-z": {0, 3}})
    qc.append({"H-y-z": {2, 3}})
    qc.append({"H-x+y": {0, 3}})
    qc.append({"H-x-y": {1, 4, 2}})
    qc.append({"F1": {0, 3}})
    qc.append({"F1d": {2, 3}})
    qc.append({"F2": {1, 4, 2}})
    qc.append({"F2d": {0, 3}})
    qc.append({"F3": {2, 3}})
    qc.append({"F3d": {1, 4, 2}})
    qc.append({"F4": {2, 3}})
    qc.append({"F4d": {0, 3}})
    qc.append({"CNOT": {(0, 1)}})
    qc.append({"G": {(1, 3)}})
    qc.append({"II": {(4, 2)}})

    # Measure
    qc.append({"Measure": {0, 1, 2, 3, 4}})

    # Run
    sim = CoinToss(len(qc.qudits))
    sim.run_circuit(qc)


def test_prefix_measurement_without_binding_is_checked_before_measuring():
    """run_circuit measures any "measure"-prefixed batch, so it must check them all."""
    state = CoinToss(4, prob=0.5, seed=42)
    fresh = CoinToss(4, prob=0.5, seed=42)
    with pytest.raises(ValueError, match="measure custom"):
        state.run_circuit(_items("measure custom", {0, 9}))
    assert state.run_circuit(_items("Measure", {0, 1, 2, 3})) == fresh.run_circuit(
        _items("Measure", {0, 1, 2, 3}),
    )
