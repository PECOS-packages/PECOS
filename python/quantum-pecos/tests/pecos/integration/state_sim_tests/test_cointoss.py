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
def test_all_measurement_batches_are_atomic(route, symbol, container, simulate_gate, prob):
    state = CoinToss(4, seed=42, prob=prob)
    fresh = CoinToss(4, seed=42, prob=prob)
    assert list(container([0, 4])) == [0, 4]
    with pytest.raises(ValueError, match="qubit 4") as error:
        _dispatch(state, route, symbol, container([0, 4]), simulate_gate=simulate_gate)
    for entry in error.traceback:
        if entry.name in {"run_gate", "run_circuit"}:
            assert entry.frame.f_locals.get("measurement_results", {}) == {}
    state.prob = fresh.prob = 0.5
    assert [state.mz(0) for _ in range(32)] == [fresh.mz(0) for _ in range(32)]


@pytest.mark.parametrize("symbol", MEASURE_SYMBOLS)
@pytest.mark.parametrize("simulate_gate", [True, False])
@pytest.mark.parametrize("prob", [0.5, 1.0])
def test_measurement_batch_failure_preserves_results_and_rng(symbol, simulate_gate, prob):
    """An invalid second location must not record a one or consume a coin toss."""
    state = CoinToss(4, prob=prob, seed=42)
    fresh = CoinToss(4, prob=prob, seed=42)
    locations = {0, 4}
    assert list(locations) == [0, 4]  # Ensure the invalid location really is second.

    with pytest.raises(ValueError, match="qubit 4") as error:
        state.run_circuit(_items(symbol, locations, simulate_gate=simulate_gate))

    # run_circuit owns the result dict locally and cannot return it on failure.
    # Inspect its exception frame to detect results recorded before rejection.
    frame = next(entry.frame for entry in error.traceback if entry.name == "run_circuit")
    assert frame.f_locals["measurement_results"] == fresh.run_circuit(_items(symbol, set())) == {}
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
def test_valid_measurement_batches_match_dev(symbol, locations, expected):
    """Enabled batches retain dev's prefix-measurement output and RNG draws."""
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
    """A known prefix spelling rebound to a measurement is pre-checked."""
    state = CoinToss(4, seed=42)
    state.bindings["measure X"] = state.bindings["MX"]
    with pytest.raises(ValueError, match="measure X"):
        state.run_circuit(_items("measure X", {0, 4}))


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
