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
from guppylang.std.quantum import measure, qubit, x
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


def _coherent_idle_noise(*, rate: float, model: dict[str, float] | None = None):
    noise = (
        general_noise()
        .with_p_prep(0.0)
        .with_p1(0.0)
        .with_p2(0.0)
        .with_p_meas(0.0)
        .with_p_idle_linear(0.0, {"Z": 1.0})
        .with_idle_after_2q(1.0)
    )
    if model is None:
        return noise.with_p_idle_coherent(rate)
    return noise.with_p_idle_coherent(rate, model)


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


def test_auto_is_chainable_and_explicit_zeros_win_in_both_orders() -> None:
    """The pyo3 preset preserves explicit zero rates before and after ``auto``."""

    @guppy
    def deterministic_x() -> bool:
        q = qubit()
        x(q)
        return measure(q)

    auto_then_zeros = (
        general_noise().auto().with_p_prep(0.0).with_p1(0.0).with_p2(0.0).with_p_meas(0.0).with_p_idle_linear_rate(0.0)
    )
    zeros_then_auto = (
        general_noise().with_p_prep(0.0).with_p1(0.0).with_p2(0.0).with_p_meas(0.0).with_p_idle_linear_rate(0.0).auto()
    )

    for noise in (auto_then_zeros, zeros_then_auto):
        results = sim(deterministic_x).qubits(1).quantum(state_vector()).noise(noise).seed(42).run(20).to_dict()
        raw = results["measurements"]
        measurements = [m[-1] if isinstance(m, list) else m for m in raw]
        assert measurements == [1] * 20


def test_auto_matches_explicit_legacy_preset_at_python_surface() -> None:
    """The pyo3 ``auto`` method delegates to the complete Rust legacy preset."""

    @guppy
    def deterministic_x() -> bool:
        q = qubit()
        x(q)
        return measure(q)

    explicit = (
        general_noise()
        .with_p_prep(0.01)
        .with_p_meas_0(0.01)
        .with_p_meas_1(0.01)
        .with_p1(0.001)
        .with_p2(0.01)
        .with_p_idle_linear_rate(0.001)
        .with_p1_emission_ratio(0.5)
        .with_p2_emission_ratio(0.5)
        .with_prep_leak_ratio(0.5)
        .with_p1_seepage_prob(0.5)
        .with_p2_seepage_prob(0.5)
        .with_p_idle_coherent_to_incoherent_factor(1.5)
    )

    def run(noise) -> list[bool]:
        results = sim(deterministic_x).qubits(1).quantum(state_vector()).noise(noise).seed(424).run(512).to_dict()
        raw = results["measurements"]
        return [m[-1] if isinstance(m, list) else m for m in raw]

    auto_results = run(general_noise().auto())
    explicit_results = run(explicit)

    assert auto_results == explicit_results
    assert auto_results != [1] * 512


def test_idle_family_setters_are_chainable() -> None:
    """All structured idle families and the renamed quadratic switch are fluent."""
    builder = general_noise().with_p_idle_linear(0.01, {"X": 0.5, "L": 0.5})
    builder = builder.with_p_idle_sin_squared(0.02, {"X": 1.0, "Z": 2.0, "L": 0.25})
    builder = builder.with_p_idle_coherent(0.03, {"RX": 1.0, "RZ": 2.0})
    assert builder.with_p_idle_quadratic_coherent(False) is not None


def test_general_noise_linear_rate_setter_keeps_total_rate_family_semantics() -> None:
    """The live engines spelling remains a total rate split by its model."""

    uniform_model = {"X": 1.0 / 3.0, "Y": 1.0 / 3.0, "Z": 1.0 / 3.0}
    otherwise_noiseless = (
        general_noise().with_p_prep(0.0).with_p1(0.0).with_p2(0.0).with_p_meas(0.0).with_idle_after_2q(1.0)
    )
    legacy_spelling = otherwise_noiseless.with_p_idle_linear_rate(1.0)
    total_rate_family = otherwise_noiseless.with_p_idle_linear(1.0, uniform_model)
    z_only_family = otherwise_noiseless.with_p_idle_linear(1.0, {"Z": 1.0})

    legacy_results = _run_after_2q_noise(legacy_spelling, shots=64, seed=1234)
    assert legacy_results == _run_after_2q_noise(total_rate_family, shots=64, seed=1234)
    assert _run_after_2q_noise(z_only_family, shots=64, seed=1234) == [0] * 64
    assert legacy_results != [0] * 64


def test_retired_coherent_bool_switch_is_not_an_alias() -> None:
    """The old one-bool call cannot silently become a zero/one coherent-family rate."""
    with pytest.raises(TypeError, match=r"coherent idling rate.*not bool"):
        general_noise().with_p_idle_coherent(False)


def test_coherent_idle_default_model_is_available() -> None:
    """Omitting the pyo3 model selects the documented symmetric RX/RY/RZ multipliers."""
    implicit = _coherent_idle_noise(rate=math.pi)
    explicit = _coherent_idle_noise(rate=math.pi, model={"RX": 1.0, "RY": 1.0, "RZ": 1.0})
    assert _run_after_2q_noise(implicit, 64, seed=424) == _run_after_2q_noise(explicit, 64, seed=424)


def test_coherent_idle_rx_reaches_runtime_deterministically() -> None:
    """A pi RX idle rotation after CX flips both measured qubits for every seed."""
    noise = _coherent_idle_noise(rate=math.pi, model={"RX": 1.0})
    assert _run_after_2q_noise(noise, 10, seed=1) == [3] * 10
    assert _run_after_2q_noise(noise, 10, seed=999) == [3] * 10


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
    noise = general_noise().with_p_idle_sin_squared(0.02, {"Z": 1.0}).with_p_idle_quadratic_coherent(True)
    with pytest.raises(ValueError, match=r"with_p_idle_quadratic_coherent\(true\).*stochastic by definition"):
        _run_after_2q_noise(noise)


def test_coherent_family_and_quadratic_coherent_path_conflict_at_build() -> None:
    """The independent and legacy coherent paths cannot both emit rotations."""
    noise = general_noise().with_p_idle_coherent(0.02, {"RZ": 1.0}).with_p_idle_quadratic_coherent(True)
    with pytest.raises(ValueError, match=r"with_p_idle_coherent.*with_p_idle_quadratic_coherent\(true\)"):
        _run_after_2q_noise(noise)


@pytest.mark.parametrize("model", [{"L": 1.0}, {"A": 1.0}])
def test_coherent_idle_family_rejects_non_rotation_keys(model: dict[str, float]) -> None:
    """Leakage and unknown generators are rejected instead of being treated as rotations."""
    with pytest.raises(BaseException, match=r"invalid key.*expected RX, RY, or RZ"):
        general_noise().with_p_idle_coherent(0.02, model)


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
