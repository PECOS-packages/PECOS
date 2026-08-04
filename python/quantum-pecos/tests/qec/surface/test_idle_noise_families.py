from __future__ import annotations

import pytest
from pecos.qec.surface import NoiseParameters, SurfacePatch, TwirlConfig
from pecos.qec.surface.decode import generate_circuit_level_dem_from_builder


def _native_surface_dem(noise: NoiseParameters) -> bytes:
    patch = SurfacePatch.create(distance=3)
    dem = generate_circuit_level_dem_from_builder(
        patch,
        num_rounds=2,
        noise=noise,
        basis="Z",
        decompose_errors=True,
        twirl=TwirlConfig(),
    )
    return dem.encode()


def test_linear_family_matches_per_axis_native_surface_dem() -> None:
    rate = 0.003

    structured = _native_surface_dem(NoiseParameters(p_idle_linear=rate))
    primitive = _native_surface_dem(
        NoiseParameters(
            p_idle_x_linear_rate=rate / 3.0,
            p_idle_y_linear_rate=rate / 3.0,
            p_idle_z_linear_rate=rate / 3.0,
        ),
    )

    assert structured == primitive


def test_sin_squared_family_matches_per_axis_native_surface_dem() -> None:
    rate = 0.03

    structured = _native_surface_dem(
        NoiseParameters(
            p_idle_sin_squared=rate,
            p_idle_sin_squared_model={"Z": 1.0},
        ),
    )
    primitive = _native_surface_dem(NoiseParameters(p_idle_z_quadratic_sine_rate=rate))

    assert structured == primitive


def test_structured_families_survive_runtime_idle_unit_conversion() -> None:
    noise = NoiseParameters(
        p_idle_linear=0.3,
        p_idle_sin_squared=0.2,
        p_idle_sin_squared_model={"Z": 1.0},
    )

    converted = noise.for_runtime_idle_time_units(time_units_per_second=10.0)

    assert converted.p_idle_x_linear_rate == pytest.approx(0.01)
    assert converted.p_idle_y_linear_rate == pytest.approx(0.01)
    assert converted.p_idle_z_linear_rate == pytest.approx(0.01)
    assert converted.p_idle_x_quadratic_sine_rate is None
    assert converted.p_idle_y_quadratic_sine_rate is None
    assert converted.p_idle_z_quadratic_sine_rate == pytest.approx(0.02)
    assert converted.p_idle_linear is None
    assert converted.p_idle_linear_model is None
    assert converted.p_idle_sin_squared is None
    assert converted.p_idle_sin_squared_model is None
    assert converted.p_idle_coherent is None
    assert converted.p_idle_coherent_model is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"p_idle_linear": 0.01, "p_idle_x_linear_rate": 0.02},
        {"p_idle_sin_squared": 0.01, "p_idle_y_quadratic_sine_rate": 0.02},
    ],
)
def test_structured_family_conflicts_with_corresponding_primitive(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        NoiseParameters(**kwargs)


@pytest.mark.parametrize(
    "field",
    [
        "p_idle_linear_rate",
        "p_idle_quadratic_rate",
        "p_idle_quadratic_sine_rate",
    ],
)
def test_bare_z_only_alias_warns_through_noise_model(field: str) -> None:
    with pytest.warns(DeprecationWarning, match=field):
        NoiseParameters(**{field: 0.01})


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
        NoiseParameters(p_idle_coherent=0.01)
