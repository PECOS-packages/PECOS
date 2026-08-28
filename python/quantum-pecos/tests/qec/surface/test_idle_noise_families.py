from __future__ import annotations

from pathlib import Path

import pytest
from pecos.qec.surface import NoiseParameters, SurfacePatch, TwirlConfig
from pecos.qec.surface.decode import generate_circuit_level_dem_from_builder

_FIXTURES = Path(__file__).with_name("fixtures")


def _native_surface_dem(noise: NoiseParameters) -> str:
    patch = SurfacePatch.create(distance=3)
    return generate_circuit_level_dem_from_builder(
        patch,
        num_rounds=2,
        noise=noise,
        basis="Z",
        decompose_errors=True,
        twirl=TwirlConfig(),
    )


def _pre_change_dem_fixture(name: str) -> str:
    return (_FIXTURES / name).read_text().removesuffix("\n")


def test_z_linear_family_matches_removed_z_only_setter_dem_fixture() -> None:
    rate = 0.003

    actual = _native_surface_dem(
        NoiseParameters().with_p_idle_linear(rate, {"Z": 1.0}),
    )

    assert actual == _pre_change_dem_fixture("idle_z_linear.dem")


def test_z_sin_squared_family_matches_removed_z_only_setter_dem_fixture() -> None:
    rate = 0.03

    actual = _native_surface_dem(NoiseParameters().with_p_idle_sin_squared(rate, {"Z": 1.0}))

    assert actual == _pre_change_dem_fixture("idle_z_sin_squared.dem")


def test_linear_family_default_is_symmetric_end_to_end() -> None:
    rate = 0.003
    implicit = NoiseParameters().with_p_idle_linear(rate)
    explicit = NoiseParameters().with_p_idle_linear(
        rate,
        {"X": 1.0 / 3.0, "Y": 1.0 / 3.0, "Z": 1.0 / 3.0},
    )

    assert implicit.idle_memory_rates[:3] == pytest.approx((rate / 3.0,) * 3)
    assert _native_surface_dem(implicit) == _native_surface_dem(explicit)


def test_sin_squared_family_default_is_symmetric_end_to_end() -> None:
    rate = 0.03
    implicit = NoiseParameters().with_p_idle_sin_squared(rate)
    explicit = NoiseParameters().with_p_idle_sin_squared(rate, {"X": 1.0, "Y": 1.0, "Z": 1.0})

    assert implicit.idle_memory_rates[6:] == pytest.approx((rate,) * 3)
    assert _native_surface_dem(implicit) == _native_surface_dem(explicit)


def test_structured_families_survive_runtime_idle_unit_conversion() -> None:
    noise = NoiseParameters(
        p_idle_linear=0.3,
        p_idle_sin_squared=0.2,
        p_idle_sin_squared_model={"Z": 1.0},
    )

    converted = noise.for_runtime_idle_time_units(time_units_per_second=10.0)

    assert converted.idle_memory_rates[:3] == pytest.approx((0.01, 0.01, 0.01))
    assert converted.idle_memory_rates[6:] == (None, None, pytest.approx(0.02))
    assert converted.p_idle_linear is None
    assert converted.p_idle_linear_model is None
    assert converted.p_idle_sin_squared is None
    assert converted.p_idle_sin_squared_model is None
    assert converted.p_idle_coherent is None
    assert converted.p_idle_coherent_model is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"p_idle_linear": 0.01, "_p_idle_x_linear_rate": 0.02},
        {"p_idle_sin_squared": 0.01, "_p_idle_y_quadratic_sine_rate": 0.02},
    ],
)
def test_structured_family_conflicts_with_corresponding_primitive(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        NoiseParameters(**kwargs)


def test_idle_memory_rates_include_translated_family_values() -> None:
    noise = NoiseParameters(
        p_idle_linear=0.3,
        p_idle_sin_squared=0.2,
        p_idle_sin_squared_model={"Z": 1.0},
    )

    assert noise.idle_memory_rates[:3] == pytest.approx((0.1, 0.1, 0.1))
    assert noise.idle_memory_rates[3:8] == (None, None, None, None, None)
    assert noise.idle_memory_rates[8] == pytest.approx(0.2)


def test_nonzero_coherent_family_is_rejected_by_standard_dem_model() -> None:
    with pytest.raises(ValueError, match="cannot represent coherent idle noise"):
        NoiseParameters().with_p_idle_coherent(0.01)
