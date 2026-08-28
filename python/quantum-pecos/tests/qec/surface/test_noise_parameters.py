# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Contract tests for DEM-construction noise parameters."""

from __future__ import annotations

from dataclasses import fields

import pecos.qec.surface as surface
import pytest
from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, measure, qubit
from pecos import NoiseParameters
from pecos.qec import DetectorErrorModel


@guppy
def _two_qubit_program() -> None:
    q0 = qubit()
    q1 = qubit()
    cx(q0, q1)
    result("m0", measure(q0).read())
    result("m1", measure(q1).read())


def _dem_bytes(noise: NoiseParameters) -> bytes:
    build = (
        DetectorErrorModel.builder()
        .with_program(_two_qubit_program)
        .with_qubits(2)
        .with_detectors_json('[{"id":0,"result_tags":["m0"]}]')
        .with_observables_json('[{"id":0,"result_tags":["m1"]}]')
        .with_noise(noise)
        .build()
    )
    return build.dem.to_string().encode()


def test_fluent_chain_matches_constructor_and_dem() -> None:
    constructor = NoiseParameters(
        p1=0.001,
        p1_weights={"X": 0.2, "Y": 0.3, "Z": 0.5},
        p2=0.01,
        p2_weights={"IX": 1.0},
        p2_replacement_approximation="ignore_gate_removal",
        p_meas=0.002,
        p_prep=0.003,
    )
    fluent = (
        NoiseParameters()
        .with_p1(0.001)
        .with_p1_weights({"X": 0.2, "Y": 0.3, "Z": 0.5})
        .with_p2(0.01)
        .with_p2_weights({"IX": 1.0})
        .with_p2_replacement_approximation("ignore_gate_removal")
        .with_p_meas(0.002)
        .with_p_prep(0.003)
    )

    assert fluent == constructor
    assert _dem_bytes(fluent) == _dem_bytes(constructor)


# The idle-family model fields are the one deliberate exception to the
# mechanical rule: they are set through their family's rate setter, because a
# model without a rate is inert and the two cannot be set in separate calls.
_FAMILY_MODEL_FIELDS = {
    "p_idle_linear_model",
    "p_idle_sin_squared_model",
    "p_idle_coherent_model",
}
_INTERNAL_IDLE_FIELDS = {
    "_p_idle_linear_rate",
    "_p_idle_quadratic_rate",
    "_p_idle_x_linear_rate",
    "_p_idle_y_linear_rate",
    "_p_idle_z_linear_rate",
    "_p_idle_x_quadratic_rate",
    "_p_idle_y_quadratic_rate",
    "_p_idle_z_quadratic_rate",
    "_p_idle_quadratic_sine_rate",
    "_p_idle_x_quadratic_sine_rate",
    "_p_idle_y_quadratic_sine_rate",
    "_p_idle_z_quadratic_sine_rate",
}
_REMOVED_IDLE_SETTERS = tuple(f"with_{name.removeprefix('_')}" for name in sorted(_INTERNAL_IDLE_FIELDS))


def test_every_field_has_a_mechanical_fluent_setter() -> None:
    field_names = {field.name for field in fields(NoiseParameters)}
    public_field_names = field_names - _INTERNAL_IDLE_FIELDS

    assert len(field_names) == 30
    assert len(public_field_names) == 18
    for field_name in public_field_names - _FAMILY_MODEL_FIELDS:
        assert callable(getattr(NoiseParameters, f"with_{field_name}")), field_name


def test_per_axis_and_legacy_idle_fields_are_internal() -> None:
    noise = NoiseParameters()

    for internal_name in _INTERNAL_IDLE_FIELDS:
        assert hasattr(noise, internal_name), internal_name
        assert not hasattr(noise, internal_name.removeprefix("_")), internal_name


def test_per_axis_and_legacy_idle_setters_are_removed() -> None:
    noise = NoiseParameters()

    for setter_name in _REMOVED_IDLE_SETTERS:
        assert not hasattr(noise, setter_name), setter_name


def test_per_axis_idle_constructor_keyword_fails_loudly() -> None:
    kwargs = {"p_idle_z_linear_rate": 0.01}
    with pytest.raises(TypeError, match=r"unexpected keyword argument 'p_idle_z_linear_rate'"):
        NoiseParameters(**kwargs)


def test_family_models_are_set_through_their_rate_setter() -> None:
    import inspect

    for family in ("p_idle_linear", "p_idle_sin_squared", "p_idle_coherent"):
        signature = inspect.signature(getattr(NoiseParameters, f"with_{family}"))
        assert "model" in signature.parameters, family


def test_fluent_setter_returns_a_new_object() -> None:
    original = NoiseParameters(p1=0.001)

    updated = original.with_p1(0.002)

    assert updated is not original
    assert original.p1 == 0.001
    assert updated.p1 == 0.002


def test_structured_family_survives_fluent_chain_and_runtime_conversion() -> None:
    noise = NoiseParameters().with_p_idle_linear(0.3).with_p1(0.001).with_p_meas(0.002)

    converted = noise.for_runtime_idle_time_units(time_units_per_second=10.0)

    assert converted.idle_memory_rates[:3] == pytest.approx((0.01, 0.01, 0.01))
    assert converted.p_idle_linear is None
    assert converted.p_idle_linear_model is None


def test_idle_family_rate_and_model_set_together() -> None:
    # The family halves must be settable in ONE call: __post_init__ translates a
    # family into per-axis fields and clears it, so a separate model-setting call
    # would collide with the per-axis values the rate call just produced.
    noise = NoiseParameters().with_p_idle_linear(0.01, {"Z": 1.0}).with_p1(0.001)

    assert noise.idle_memory_rates[2] == pytest.approx(0.01)
    assert noise.idle_memory_rates[0] in (None, 0.0)
    assert noise.p_idle_linear is None
    assert noise.p1 == pytest.approx(0.001)


def test_idle_families_have_no_separate_model_setters() -> None:
    # A model without a rate is inert and rejected, so exposing a lone model
    # setter would only ever produce an error or a collision.
    for name in (
        "with_p_idle_linear_model",
        "with_p_idle_sin_squared_model",
        "with_p_idle_coherent_model",
    ):
        assert not hasattr(NoiseParameters, name), name


def test_each_idle_family_round_trips_through_runtime_conversion() -> None:
    linear = NoiseParameters().with_p_idle_linear(0.3, {"Z": 1.0})
    sine = NoiseParameters().with_p_idle_sin_squared(0.2, {"X": 1.0})

    assert linear.for_runtime_idle_time_units(time_units_per_second=10.0).idle_memory_rates[2] == pytest.approx(0.03)
    converted_sine = sine.for_runtime_idle_time_units(time_units_per_second=10.0)
    assert converted_sine.idle_memory_rates[6] == pytest.approx(0.02)


def test_deprecated_alias_warns_and_returns_noise_parameters() -> None:
    with pytest.warns(
        DeprecationWarning,
        match=r"NoiseModel.*NoiseParameters.*from pecos import NoiseParameters",
    ):
        legacy = surface.NoiseModel(p1=0.001)

    assert type(legacy) is NoiseParameters
    assert legacy == NoiseParameters(p1=0.001)


def test_public_import_paths_refer_to_the_same_class() -> None:
    from pecos import NoiseParameters as TopLevelNoiseParameters
    from pecos.qec.surface import NoiseParameters as SurfaceNoiseParameters

    assert TopLevelNoiseParameters is NoiseParameters
    assert SurfaceNoiseParameters is NoiseParameters


def test_chaining_order_does_not_matter() -> None:
    first = NoiseParameters().with_p1(0.001).with_p2(0.01).with_p_meas(0.002).with_p_prep(0.003)
    second = NoiseParameters().with_p_prep(0.003).with_p_meas(0.002).with_p2(0.01).with_p1(0.001)

    assert first == second
