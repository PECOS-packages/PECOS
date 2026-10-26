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

from types import SimpleNamespace

import pecos
import pytest
from pecos.simulators import DefaultSimulator, SparseStabPy, StateVec
from pecos_rslib import AngleSource, GateRegistry


def _circuit(symbol, locations, **params):
    return SimpleNamespace(items=lambda: iter([(symbol, locations, params)]))


def _make(simulator_type=SparseStabPy):
    state = simulator_type(4, seed=42) if simulator_type is StateVec else simulator_type(4)
    # The registry emits MZ; the Python wrappers spell this binding Measure.
    # Install the alias explicitly, as callers could already do on dev.
    state.bindings = {**state.bindings, "MZ": state.bindings["Measure"]}
    return state


def _snapshot(state):
    if isinstance(state, SparseStabPy):
        return state.stabs.row_string(), state.destabs.row_string()
    return list(state.vector)


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
def test_unbound_step_preserves_state_and_rng(measure):
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
        state.run_circuit(_circuit(symbol, [0, 1]), gate_registry=registry)
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
    assert "Gate HMCX step CX" in str(error.value)


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
    state = _make(StateVec)
    # StateVec overrides run_circuit without registry support; exercise the
    # base registry path with its real bindings and native state/RNG.
    result = DefaultSimulator._run_decomposed_gate(  # noqa: SLF001 - test the base path with a native RNG
        state,
        registry,
        "ROTATIONS",
        locations,
        angles=(0.125,),
        angle=99,
    )
    snapshot = _snapshot(state)
    observed_rng = _rng_probe(state)

    pecos.random.seed(721)
    fresh = _make(StateVec)
    expected = {}
    for qubit in locations:
        fresh.run_gate("H", {qubit})
        fresh.run_gate("RX", {qubit}, angles=(0.125,), angle=0.125)
        fresh.run_gate("RZ", {qubit}, angles=(0.25,), angle=0.25)
        expected.update(fresh.run_gate("MZ", {qubit}))
    assert result == expected
    assert snapshot == pytest.approx(_snapshot(fresh))
    assert observed_rng == _rng_probe(fresh)
