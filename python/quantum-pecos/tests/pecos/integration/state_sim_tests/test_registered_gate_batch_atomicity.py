# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Registered batches validate all decomposed steps before touching state or RNG."""

from functools import partial
from math import pi
from types import SimpleNamespace

import pecos
import pytest
from pecos.simulators import SparseStabPy
from pecos_rslib import AngleSource, GateRegistry


def _circuit(symbol, locations, **params):
    return SimpleNamespace(items=lambda: iter([(symbol, locations, params)]))


def _make():
    state = SparseStabPy(4)
    # The registry emits MZ; the Python wrappers spell this binding Measure.
    # Install the alias explicitly, as callers could already do on dev.
    state.bindings = {**state.bindings, "MZ": state.bindings["Measure"]}
    return state


def _snapshot(state):
    return state.stabs.row_string(), state.destabs.row_string()


def _rng_probe(state):
    results = []
    for _ in range(32):
        state.run_gate("H", [0])
        results.append(state.run_gate("Measure", [0]))
    return results, list(pecos.random.random(16))


def _assert_untouched(state):
    snapshot = _snapshot(state)
    observed_rng = _rng_probe(state)
    pecos.random.seed(721)
    fresh = _make()
    assert snapshot == _snapshot(fresh)
    assert observed_rng == _rng_probe(fresh)


@pytest.mark.parametrize("measure", [False, True])
@pytest.mark.parametrize("simulate_gate", [False, True])
def test_unbound_step_preserves_state_and_rng(measure, simulate_gate):
    registry = GateRegistry()
    symbol = "HMT" if measure else "HT"
    builder = registry.define(symbol, 1).step("H", [0])
    if measure:
        builder.step("MZ", [0])
    builder.step("T", [0]).register_into(registry)
    pecos.random.seed(721)
    state = _make()
    assert "T" not in state.bindings
    with pytest.raises(ValueError, match=f"Gate {symbol} step T.*not supported"):
        state.run_circuit(_circuit(symbol, [0, 1], simulate_gate=simulate_gate), gate_registry=registry)
    _assert_untouched(state)


@pytest.mark.parametrize(("bad", "reason"), [((2, 4), "out of bounds"), ((2, 2), "distinct qubits")])
@pytest.mark.parametrize("container", [list, iter])
def test_second_location_bad_step_preserves_state_and_rng(bad, reason, container):
    registry = GateRegistry()
    (registry.define("HMCX", 2).step("H", [0]).step("MZ", [0]).step("CX", [0, 1]).register_into(registry))
    pecos.random.seed(721)
    state = _make()
    with pytest.raises(ValueError, match=reason) as error:
        state.run_circuit(_circuit("HMCX", container([(0, 1), bad])), gate_registry=registry)
    _assert_untouched(state)
    assert "Gate HMCX" in str(error.value)


@pytest.mark.parametrize("container", [list, iter])
def test_valid_multilocation_batch_matches_sequential_steps(container):
    registry = GateRegistry()
    (
        registry.define("HCMX", 2)
        .step("H", [0])
        .step("CX", [0, 1])
        .step("MZ", [1])
        .step("X", [0])
        .register_into(registry)
    )
    # Non-sorted, overlapping and repeated locations pin the original order.
    locations = [(2, 3), (0, 2), (2, 3), (1, 0)]
    pecos.random.seed(721)
    state = _make()
    result = state.run_circuit(_circuit("HCMX", container(locations)), gate_registry=registry)
    snapshot = _snapshot(state)
    observed_rng = _rng_probe(state)

    pecos.random.seed(721)
    fresh = _make()
    expected = {}
    for control, target in locations:
        fresh.run_gate("H", {control})
        fresh.run_gate("CX", {(control, target)})
        expected.update(fresh.run_gate("MZ", {target}))
        fresh.run_gate("X", {control})
    assert result == expected
    assert snapshot == _snapshot(fresh)
    assert observed_rng == _rng_probe(fresh)


def test_valid_batch_preserves_angles_and_metadata():
    registry = GateRegistry()
    (
        registry.define("ROTATIONS", 1)
        .angle_arity(1)
        .step("H", [0])
        .step_with_angles("RX", [0], [AngleSource.input(0)])
        .step_with_angles("RZ", [0], [AngleSource.fixed(0.25)])
        .step_with_metadata("RY", [0], [AngleSource.neg_input(0)], {"simulate_gate": False})
        .step("MZ", [0])
        .register_into(registry)
    )
    locations = [2, 0, 2, 1]
    pecos.random.seed(721)
    state = _make()
    # The step angles replace the outer angle; a leaked 99 rad is not Clifford.
    result = state.run_circuit(
        _circuit("ROTATIONS", locations, angles=(pi / 2,), angle=99),
        gate_registry=registry,
    )
    snapshot = _snapshot(state)
    observed_rng = _rng_probe(state)

    pecos.random.seed(721)
    fresh = _make()
    expected = {}
    for qubit in locations:
        fresh.run_gate("H", {qubit})
        fresh.run_gate("RX", {qubit}, angle=pi / 2)
        fresh.run_gate("RZ", {qubit}, angle=pi / 2)
        expected.update(fresh.run_gate("MZ", {qubit}))
    assert result == expected
    assert snapshot == _snapshot(fresh)
    assert observed_rng == _rng_probe(fresh)


_ROTATIONS = [
    ("RX", [0], (0.1,)),
    ("RY", [0], (0.1,)),
    ("RZ", [0], (0.1,)),
    ("RXX", [0, 1], (0.1,)),
    ("RYY", [0, 1], (0.1,)),
    ("RZZ", [0, 1], (0.1,)),
    ("RXY1Q", [0], (0.1, 0.0)),
    ("R1XY", [0], (0.1, 0.0)),
    ("RXYXY2Q", [0, 1], (0.1, 0.0)),
]


@pytest.mark.parametrize(("symbol", "qubits", "angles"), _ROTATIONS)
@pytest.mark.parametrize("measure", [False, True])
def test_registered_bad_angle_preserves_state_and_rng(symbol, qubits, angles, measure):
    registry = GateRegistry()
    builder = registry.define("BAD_ROTATION", 2).angle_arity(len(angles)).step("H", [0])
    if measure:
        builder.step("MZ", [0])
    builder.step_with_angles(symbol, qubits, [AngleSource.input(i) for i in range(len(angles))]).register_into(registry)
    pecos.random.seed(721)
    state = _make()
    with pytest.raises(ValueError, match="not a Clifford rotation") as error:
        state.run_circuit(_circuit("BAD_ROTATION", iter([(0, 1), (2, 3)]), angles=angles), gate_registry=registry)
    _assert_untouched(state)
    assert "Gate BAD_ROTATION step " in str(error.value)


@pytest.mark.parametrize(("symbol", "qubits", "angles"), [*_ROTATIONS, ("U1q", [0], (0.1, 0.0))])
@pytest.mark.parametrize("hybrid", [False, True])
def test_plain_disabled_rotation_ignores_params(symbol, qubits, angles, hybrid):
    pecos.random.seed(721)
    state = _make()
    locations = [0, 1] if len(qubits) == 1 else [(0, 1), (2, 3)]
    run = partial(pecos.HybridEngine.run_gate, state, {}) if hybrid else state.run_gate
    run(symbol, iter(locations), angles=angles, simulate_gate=False)
    _assert_untouched(state)
    with pytest.raises(ValueError, match="out of bounds"):
        run(symbol, [99] if len(qubits) == 1 else [(0, 99)], angles=angles, simulate_gate=False)


@pytest.mark.parametrize(
    ("arity", "steps", "bad", "reason"),
    [
        (2, [0, 1], (1, 1), "distinct qubits"),
        (2, [0], (0, 99), "out of bounds"),
        (2, [0], (0, -1), "out of bounds"),
        (2, [0], (0,), "expected 2 qubit"),
        (2, [0], (0, 1, 2), "expected 2 qubit"),
        (3, [0], (0, 1, 1), "distinct qubits"),
        (3, [0], (0, 1, 99), "out of bounds"),
        (0, [], (0,), "expected 0 qubit"),
    ],
)
@pytest.mark.parametrize("container", [list, iter])
@pytest.mark.parametrize("simulate_gate", [False, True])
def test_outer_location_preserves_state_and_rng(arity, steps, bad, reason, container, simulate_gate):
    registry = GateRegistry()
    builder = registry.define("OUTER", arity)
    for qubit in steps:
        builder.step("H", [qubit]).step("MZ", [qubit])
    builder.register_into(registry)
    pecos.random.seed(721)
    state = _make()
    locations = container([tuple(range(arity)), bad])
    with pytest.raises(ValueError, match=f"Gate OUTER.*{reason}"):
        state.run_circuit(_circuit("OUTER", locations, simulate_gate=simulate_gate), gate_registry=registry)
    _assert_untouched(state)


@pytest.mark.parametrize(
    "params",
    [
        {"angle": pi / 2},
        {"angles": (pi / 2,)},
        {"angle": 0.1, "angles": (pi / 2,)},
        {"angle": pi / 2, "angles": ()},
        {"angle": pi / 2, "angles": (0.1, 0.2)},
    ],
)
def test_plain_rotation_keeps_binding_parameter_precedence(params):
    state = _make()
    state.run_gate("RX", [0], **params)
    fresh = _make()
    fresh.bindings["RX"](fresh, 0, angle=pi / 2)
    assert _snapshot(state) == _snapshot(fresh)


def test_registered_overridden_rotation_keeps_its_parameter_contract():
    registry = GateRegistry()
    (
        registry.define("OVERRIDE", 1)
        .angle_arity(1)
        .step_with_angles("RZ", [0], [AngleSource.input(0)])
        .register_into(registry)
    )
    state = _make()
    calls = []
    state.bindings["RZ"] = lambda _state, qubit, **params: calls.append((qubit, params))
    state.run_circuit(_circuit("OVERRIDE", [0], angles=(pi / 4,)), gate_registry=registry)
    assert calls == [(0, {"angles": (pi / 4,), "angle": pi / 4})]


@pytest.mark.parametrize("outer_enabled", [False, True])
@pytest.mark.parametrize("step_enabled", [False, True])
def test_registered_step_params_only_checked_when_enabled(outer_enabled, step_enabled):
    registry = GateRegistry()
    registry.define("META", 1).step("H", [0]).step_with_metadata(
        "RZ",
        [0],
        [AngleSource.fixed(0.25)],
        {"angles": [0.1], "simulate_gate": step_enabled},
    ).register_into(registry)
    pecos.random.seed(721)
    state = _make()
    circuit = _circuit("META", [0], simulate_gate=outer_enabled)
    if outer_enabled and step_enabled:
        with pytest.raises(ValueError, match=r"Gate META step RZ.*not a Clifford rotation"):
            state.run_circuit(circuit, gate_registry=registry)
        _assert_untouched(state)
    else:
        assert state.run_circuit(circuit, gate_registry=registry) == {}
        if outer_enabled:
            fresh = _make()
            fresh.run_gate("H", [0])
            assert _snapshot(state) == _snapshot(fresh)
        else:
            _assert_untouched(state)


@pytest.mark.parametrize(
    ("source", "input_angles", "direct_angle"),
    [
        (AngleSource.input(0), (pi / 2,), pi / 2),
        (AngleSource.fixed(0.25), (), pi / 2),
        (AngleSource.neg_input(0), (pi / 2,), -pi / 2),
    ],
    ids=["input-radians", "fixed-quarter-turn", "negative-input"],
)
def test_registered_angles_match_direct_state(source, input_angles, direct_angle):
    registry = GateRegistry()
    (
        registry.define("ROTATE", 1)
        .angle_arity(len(input_angles))
        .step("H", [0])
        .step_with_angles("RZ", [0], [source])
        .step("H", [0])
        .register_into(registry)
    )
    state = _make()
    state.run_circuit(_circuit("ROTATE", [0], angles=input_angles), gate_registry=registry)
    fresh = _make()
    fresh.run_gate("H", [0])
    fresh.run_gate("RZ", [0], angle=direct_angle)
    fresh.run_gate("H", [0])
    # A unit error yields a non-Clifford angle, which SparseStabPy rejects; a
    # sign error flips the stabilizer sign.
    assert _snapshot(state) == _snapshot(fresh)


@pytest.mark.parametrize("simulate_gate", [False, True])
def test_empty_registered_decomposition_is_noop(simulate_gate):
    registry = GateRegistry()
    registry.define("EMPTY", 1).register_into(registry)
    pecos.random.seed(721)
    state = _make()
    assert state.run_circuit(_circuit("EMPTY", iter([0, 1]), simulate_gate=simulate_gate), gate_registry=registry) == {}
    with pytest.raises(ValueError, match=r"Gate EMPTY.*out of bounds"):
        state.run_circuit(_circuit("EMPTY", [0, 99], simulate_gate=simulate_gate), gate_registry=registry)
    _assert_untouched(state)
