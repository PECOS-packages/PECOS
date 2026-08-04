# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The simulator noise builders name each setter after the field it sets.

The suffixed spellings (``with_p1_probability``, ``with_meas_probability``, ...) were
replaced outright rather than aliased, so this pins both halves: the field-name setters
work through the pyo3 surface, and the old spellings are gone.
"""

import math

import pytest
from guppylang import guppy
from guppylang.std.quantum import measure, qubit
from pecos import Qasm, qasm_engine, sim
from pecos_rslib import (
    biased_depolarizing_noise,
    depolarizing_noise,
    general_noise,
    state_vector,
)

# (builder factory, setters that builder is expected to expose)
BUILDER_SETTERS = [
    (general_noise, ("with_p1", "with_p2", "with_p_prep", "with_p_meas", "with_p_meas_0", "with_p_meas_1")),
    (depolarizing_noise, ("with_p1", "with_p2", "with_p_prep", "with_p_meas")),
    (biased_depolarizing_noise, ("with_p1", "with_p2", "with_p_prep", "with_p_meas_0", "with_p_meas_1")),
]

REMOVED_SETTERS = (
    "with_p1_probability",
    "with_p2_probability",
    "with_prep_probability",
    "with_meas_probability",
    "with_meas_0_probability",
    "with_meas_1_probability",
    "with_average_p1_probability",
    "with_average_p2_probability",
)

_AFTER_2Q_QASM = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
cx q[0], q[1];
measure q -> c;
"""


def _run_after_2q_noise(noise, shots: int = 1, seed: int = 424) -> list[int]:
    results = qasm_engine().program(Qasm.from_string(_AFTER_2Q_QASM)).to_sim().noise(noise).seed(seed).run(shots)
    return results.to_dict()["c"]


def _idle_family_noise(*, sine_rate: float, sine_model: dict[str, float], seed: int = 424):
    return (
        general_noise()
        .with_seed(seed)
        .with_p_prep(0.0)
        .with_p1(0.0)
        .with_p2(0.0)
        .with_p_meas(0.0)
        .with_p_idle_linear(0.0, {"Z": 1.0})
        .with_p_idle_sin_squared(sine_rate, sine_model)
        .with_idle_after_2q(1.0)
    )


@pytest.mark.parametrize(("factory", "setters"), BUILDER_SETTERS)
def test_field_name_setters_are_chainable(factory, setters) -> None:
    """Every field-name setter exists and returns a builder that keeps chaining."""
    builder = factory()
    for setter in setters:
        builder = getattr(builder, setter)(0.01)
    assert builder is not None


@pytest.mark.parametrize(("factory", "_setters"), BUILDER_SETTERS)
def test_suffixed_setters_are_gone(factory, _setters) -> None:
    """The replaced spellings must not linger as aliases."""
    builder = factory()
    for removed in REMOVED_SETTERS:
        assert not hasattr(builder, removed), f"{removed} should have been renamed away"


def test_average_setters_keep_their_conversion() -> None:
    """``with_average_p*`` survives the rename; it converts from average gate error."""
    builder = general_noise()
    assert callable(builder.with_average_p1)
    assert callable(builder.with_average_p2)
    assert builder.with_average_p1(0.01).with_average_p2(0.02) is not None


def test_idle_family_setters_are_chainable() -> None:
    """The structured linear and sine families are present on the pyo3 fluent builder."""
    builder = general_noise().with_p_idle_linear(0.01, {"X": 0.5, "L": 0.5})
    assert builder.with_p_idle_sin_squared(0.02, {"X": 1.0, "Z": 2.0, "L": 0.25}) is not None


def test_linear_idle_family_rejects_unnormalized_model() -> None:
    """The linear family reuses the normalized weighted-sampler contract."""
    with pytest.raises(BaseException, match=r"total weight 2.*deviates from 1.0"):
        general_noise().with_p_idle_linear(0.01, {"X": 1.0, "Z": 1.0})


def test_sine_idle_family_x_axis_reaches_runtime() -> None:
    """A certain X sine event after CX flips both measured qubits; a Z-only path would not."""
    noise = _idle_family_noise(sine_rate=math.pi / 2, sine_model={"X": 1.0})
    assert _run_after_2q_noise(noise, 10) == [3] * 10


def test_sine_idle_multipliers_are_not_normalized() -> None:
    """X=Z=1 keeps a certain X event at rate pi/2 instead of reducing it to probability 1/2."""
    noise = _idle_family_noise(sine_rate=math.pi / 2, sine_model={"X": 1.0, "Z": 1.0})
    assert _run_after_2q_noise(noise, 32) == [3] * 32


def test_legacy_quadratic_and_sine_family_conflict_at_build() -> None:
    """The pyo3 build route names both incompatible rate spellings and their units."""
    noise = general_noise().with_p_idle_quadratic_rate(0.01).with_p_idle_sin_squared(0.02, {"Z": 1.0})
    with pytest.raises(ValueError, match=r"with_p_idle_quadratic_rate.*radians.*cycles"):
        _run_after_2q_noise(noise)


def test_sine_family_and_coherent_legacy_path_conflict_at_build() -> None:
    """The stochastic family cannot silently ignore the legacy coherent switch."""
    noise = general_noise().with_p_idle_sin_squared(0.02, {"Z": 1.0}).with_p_idle_coherent(True)
    with pytest.raises(ValueError, match=r"with_p_idle_coherent\(true\).*stochastic by definition"):
        _run_after_2q_noise(noise)


def test_sine_idle_family_is_deterministic_for_same_seed() -> None:
    """The pyo3 surface preserves the Rust model's fixed-seed draw sequence."""
    first = _run_after_2q_noise(_idle_family_noise(sine_rate=0.6, sine_model={"X": 1.0}), 128)
    second = _run_after_2q_noise(_idle_family_noise(sine_rate=0.6, sine_model={"X": 1.0}), 128)
    assert first == second
    assert 0 in first
    assert 3 in first


def test_with_p_meas_actually_configures_measurement_noise() -> None:
    """A renamed setter still reaches the model: certain measurement flips flip every shot."""

    @guppy
    def prepare_and_measure() -> bool:
        q = qubit()
        return measure(q)

    noise = general_noise().with_p_prep(0.0).with_p1(0.0).with_p2(0.0).with_p_meas(1.0)
    results = sim(prepare_and_measure).qubits(1).quantum(state_vector()).noise(noise).seed(42).run(20).to_dict()

    raw = results["measurements"]
    measurements = [m[-1] if isinstance(m, list) else m for m in raw]
    assert all(m == 1 for m in measurements), "p_meas=1.0 should flip every |0> measurement"
