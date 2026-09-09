# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Exact symbol availability and measurement contracts for Rust gate bindings."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from pecos.circuits import QuantumCircuit
from pecos.engines.hybrid_engine_old import HybridEngine
from pecos.exceptions import NotSupportedGateError
from pecos_rslib import _gate_bindings_symbols
from pecos_rslib.simulators import PauliProp, SparseStab, Stabilizer, StabVec, StateVec

SimulatorFactory = Callable[[int], object]

SIMULATORS: tuple[tuple[str, SimulatorFactory], ...] = (
    ("SparseStab", SparseStab),
    ("Stabilizer", Stabilizer),
    ("StabVec", StabVec),
    ("StateVec", StateVec),
    ("PauliProp", PauliProp),
)


def _params(parameter_mode: str) -> dict[str, object]:
    if parameter_mode == "angle":
        return {"angle": math.pi / 2}
    if parameter_mode == "angles:2":
        return {"angles": (math.pi / 2, 0.0)}
    if parameter_mode == "angles:3":
        return {"angles": (0.0, 0.0, 0.0)}
    if parameter_mode in {"forced_outcome", "optional_forced_outcome"}:
        return {"forced_outcome": 1}
    assert parameter_mode == "none"
    return {}


def _prepare_z_one(state: object, simulator_name: str) -> None:
    if simulator_name == "PauliProp":
        state.track_x([0])
    else:
        state.run_gate("X", {0})


def _assert_forced_result(state: object, simulator_name: str, spelling: str, result: object) -> None:
    if spelling == "MZForced":
        expected = 0 if simulator_name == "PauliProp" else 1
        assert result == expected
    else:
        assert spelling == "PZForced"
        assert result is None
        assert state.bindings["MZ"](state, 0) == 0


@pytest.mark.parametrize(("simulator_name", "factory"), SIMULATORS)
def test_mapping_membership_and_unsupported_override(
    simulator_name: str,
    factory: SimulatorFactory,
) -> None:
    """Membership reflects the simulator surface while per-instance overrides win."""
    del simulator_name
    bindings = factory(2).bindings
    sentinel = object()

    assert "NOPE" not in bindings
    assert bindings.get("NOPE", sentinel) is sentinel
    with pytest.raises(KeyError, match="NOPE"):
        bindings["NOPE"]
    assert "SZZdg" in bindings

    override = object()
    bindings["NOPE"] = override
    assert "NOPE" in bindings
    assert bindings.get("NOPE", sentinel) is override
    assert bindings["NOPE"] is override


@pytest.mark.parametrize(("simulator_name", "factory"), SIMULATORS)
def test_table_and_dispatch_surface_agree(simulator_name: str, factory: SimulatorFactory) -> None:
    """Every table entry agrees with membership and the simulator's dispatcher."""
    entries = _gate_bindings_symbols()
    assert len(entries) == 134
    assert len({spelling for spelling, _, _, _ in entries}) == len(entries)

    for spelling, _, parameter_mode, qubit_count in entries:
        state = factory(2)
        bindings = state.bindings
        location = (0, 1) if qubit_count == 2 else 0
        locations = {location}
        params = _params(parameter_mode)

        if spelling in bindings:
            gate = bindings[spelling]
            if spelling == "PZForced":
                _prepare_z_one(state, simulator_name)
            elif spelling == "MZForced" and simulator_name != "PauliProp":
                state.run_gate("H", {0})
            result = gate(state, location, **params)
            if parameter_mode == "optional_forced_outcome":
                state_without_parameter = factory(2)
                state_without_parameter.bindings[spelling](state_without_parameter, location)
            elif parameter_mode == "forced_outcome":
                _assert_forced_result(state, simulator_name, spelling, result)
                state_without_parameter = factory(2)
                if spelling == "PZForced":
                    _prepare_z_one(state_without_parameter, simulator_name)
                try:
                    result_without_parameter = state_without_parameter.bindings[spelling](
                        state_without_parameter,
                        location,
                    )
                except ValueError as error:
                    assert "Unsupported" not in str(error)
                    assert "requires" in str(error)
                else:
                    _assert_forced_result(
                        state_without_parameter,
                        simulator_name,
                        spelling,
                        result_without_parameter,
                    )
        else:
            gate_kind = "two" if qubit_count == 2 else "single"
            with pytest.raises(ValueError, match=f"Unsupported {gate_kind}-qubit gate"):
                state.run_gate(spelling, locations, **params)


def test_legacy_engine_translates_unknown_binding_to_supported_error() -> None:
    """The legacy engine converts a missing binding into its public exception."""
    circuit = QuantumCircuit()
    circuit.append("NOPE", {0})

    with pytest.raises(NotSupportedGateError, match="NOPE"):
        HybridEngine().run(SparseStab(1), circuit, shot_id=0)


@pytest.mark.parametrize(("simulator_name", "factory"), SIMULATORS)
@pytest.mark.parametrize(
    ("location", "gate_kind"),
    [(0, "single"), ((0, 1), "two")],
)
def test_unknown_gate_is_rejected_when_simulation_is_disabled(
    simulator_name: str,
    factory: SimulatorFactory,
    location: int | tuple[int, int],
    gate_kind: str,
) -> None:
    """Skipping execution does not bypass symbol validation."""
    del simulator_name
    with pytest.raises(ValueError, match=f"Unsupported {gate_kind}-qubit gate: NOPE"):
        factory(2).run_gate("NOPE", {location}, simulate_gate=False)


@pytest.mark.parametrize(("simulator_name", "factory"), SIMULATORS)
def test_empty_locations_still_validate_the_symbol(
    simulator_name: str,
    factory: SimulatorFactory,
) -> None:
    """Empty work skips execution only after validating the symbol."""
    del simulator_name
    with pytest.raises(ValueError, match="Unsupported single-qubit gate: NOPE"):
        factory(1).run_gate("NOPE", set())
    assert factory(1).run_gate("I", set()) == {}


def test_numpy_integer_measurement_location_is_scalar() -> None:
    """Non-built-in integral scalar locations retain scalar behavior."""
    state = SparseStab(1)
    state.run_gate("X", {0})

    assert state.bindings["MZ"](state, np.int64(0)) == 1


@pytest.mark.parametrize(("simulator_name", "factory"), SIMULATORS)
def test_z_measurement_bindings_always_return_bits(
    simulator_name: str,
    factory: SimulatorFactory,
) -> None:
    """All ordinary Z-measurement aliases return an explicit zero or one."""
    for spelling in ("MZ", "Measure", "measure Z", "Measure +Z"):
        for location in (0, (0,), [0]):
            zero_state = factory(1)
            assert zero_state.bindings[spelling](zero_state, location) == 0

            one_state = factory(1)
            _prepare_z_one(one_state, simulator_name)
            assert one_state.bindings[spelling](one_state, location) == 1


@pytest.mark.parametrize(("simulator_name", "factory"), SIMULATORS)
def test_x_measurement_on_plus_state_returns_zero(
    simulator_name: str,
    factory: SimulatorFactory,
) -> None:
    """MX is tested on its deterministic +1 eigenstate, not on computational zero."""
    del simulator_name
    state = factory(1)
    state.run_gate("H", {0})
    assert state.bindings["MX"](state, 0) == 0


def test_batch_measurement_output_remains_sparse() -> None:
    """Aggregate measurement output continues to contain fired locations only."""
    state = SparseStab(2)
    assert state.run_gate("MZ", {(0,), (1,)}) == {}


@pytest.mark.parametrize(("simulator_name", "factory"), SIMULATORS)
def test_measurement_binding_reports_nothing_when_simulation_is_disabled(
    simulator_name: str,
    factory: SimulatorFactory,
) -> None:
    """A disabled measurement is not an outcome: the callable must not fabricate a zero."""
    state = factory(1)
    assert state.bindings["MZ"](state, 0, simulate_gate=False) is None, simulator_name
    assert state.bindings["MZ"](state, 0) == 0, simulator_name
