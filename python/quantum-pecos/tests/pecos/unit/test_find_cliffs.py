import importlib
import warnings

import pecos as pc
import pytest
from pecos.analysis import (
    cliff_str2matrix,
    m2cliff,
    r1xy2cliff,
    r1xy_ang2str,
    rz2cliff,
    rz_ang2str,
)


@pytest.mark.parametrize(
    ("theta", "phi"),
    [
        (-5e-11, 0.37),
        (5e-11, -0.81),
        (pc.f64.tau - 5e-11, 1.23),
        (pc.f64.tau + 5e-11, -1.42),
        (-pc.f64.tau - 5e-11, 0.19),
    ],
)
def test_r1xy_identity_recognition_is_symmetric_around_tau_multiples(theta, phi) -> None:
    assert r1xy2cliff(theta, phi, atol=1e-9) == "I"


@pytest.mark.parametrize(
    ("theta", "atol", "expected"),
    [
        (-5e-13, 1e-12, "I"),
        (-1e-5, 1e-12, False),
        (-5e-10, 1e-9, "I"),
        (-5e-8, 1e-9, False),
    ],
)
def test_identity_recognition_uses_only_requested_absolute_tolerance(
    theta,
    atol,
    expected,
) -> None:
    assert r1xy2cliff(theta, 0.37, atol=atol) == expected
    assert rz2cliff(theta, atol=atol) == expected


@pytest.mark.parametrize("atol", [1e-12, 1e-9])
@pytest.mark.parametrize("direction", [-1, 1])
def test_r1xy_table_matching_uses_only_requested_absolute_tolerance(
    atol,
    direction,
) -> None:
    theta = pc.f64.frac_pi_2
    phi = pc.f64.frac_pi_2
    inside_offset = direction * atol / 2
    outside_offset = direction * 1e-5

    assert r1xy2cliff(theta + inside_offset, phi, atol=atol) == "SY"
    assert r1xy2cliff(theta, phi + inside_offset, atol=atol) == "SY"
    assert r1xy2cliff(theta + outside_offset, phi, atol=atol) is False
    assert r1xy2cliff(theta, phi + outside_offset, atol=atol) is False


@pytest.mark.parametrize("atol", [1e-12, 1e-9])
@pytest.mark.parametrize("direction", [-1, 1])
def test_rz_table_matching_uses_only_requested_absolute_tolerance(
    atol,
    direction,
) -> None:
    inside_offset = direction * atol / 2
    outside_offset = direction * 1e-5

    assert rz2cliff(pc.f64.pi + inside_offset, atol=atol) == "Z"
    assert rz2cliff(pc.f64.pi + outside_offset, atol=atol) is False


@pytest.mark.parametrize(
    ("theta", "expected"),
    [
        (-5e-11, "I"),
        (5e-11, "I"),
        (pc.f64.tau - 5e-11, "I"),
        (pc.f64.tau + 5e-11, "I"),
        (-pc.f64.tau - 5e-11, "I"),
        (pc.f64.frac_pi_2 + 5e-11, "SZ"),
    ],
)
def test_rz_recognition_respects_requested_tolerance(theta, expected) -> None:
    assert rz2cliff(theta, atol=1e-9) == expected


@pytest.mark.parametrize(
    ("angles", "expected"),
    list(r1xy_ang2str.items()),
)
def test_r1xy_matrix_fallback_matches_conversion_table(angles, expected) -> None:
    theta, phi = angles
    assert (
        r1xy2cliff(
            theta,
            phi,
            atol=1e-9,
            use_conv_table=False,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("angles", "expected"),
    list(rz_ang2str.items()),
)
def test_rz_matrix_fallback_matches_conversion_table(angles, expected) -> None:
    assert (
        rz2cliff(
            angles[0],
            atol=1e-9,
            use_conv_table=False,
        )
        == expected
    )


def test_matrix_fallback_propagates_tolerance() -> None:
    theta = 3 * pc.f64.pi + 5e-11

    assert r1xy2cliff(theta, 0.0, atol=1e-9, use_conv_table=False) == "X"
    assert r1xy2cliff(theta, 0.0, atol=1e-12, use_conv_table=False) is False


def test_matrix_normalization_tolerance_is_configurable() -> None:
    noisy_x = pc.array(
        [
            [5e-11, 1.0],
            [1.0, 0.0],
        ],
        dtype="complex",
    )

    assert m2cliff(noisy_x, atol=1e-9, normalization_atol=1e-10) == "X"
    assert m2cliff(noisy_x, atol=1e-9, normalization_atol=1e-12) is False


@pytest.mark.parametrize(
    ("expected", "matrix"),
    list(cliff_str2matrix.items()),
)
def test_every_canonical_matrix_is_identified(expected, matrix) -> None:
    assert m2cliff(matrix) == expected


def test_tools_compatibility_module_reexports_dtype() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy_find_cliffs = importlib.import_module("pecos.tools.find_cliffs")

    assert legacy_find_cliffs.dtype == "complex"
    assert "dtype" in legacy_find_cliffs.__all__
