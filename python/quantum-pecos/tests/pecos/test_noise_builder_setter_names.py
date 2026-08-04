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

import pytest
from guppylang import guppy
from guppylang.std.quantum import measure, qubit
from pecos import sim
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
