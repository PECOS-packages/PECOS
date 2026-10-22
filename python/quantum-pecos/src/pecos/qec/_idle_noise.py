# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Shared translation of structured idle-noise families to DEM primitives.

The linear family is a categorical Pauli channel. The Rust DEM builder first
groups its non-empty propagated flip signatures and only then converts distinct
signatures to independent mechanisms. An infeasible exact conversion uses a
non-negative boundary fit and exposes its quantified both-fire residual on the
DEM. Sine-squared axes are independent already and remain separate mechanisms
from the linear family.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping


def _translate_structured_idle_noise(
    *,
    p_idle_linear: float | None,
    p_idle_linear_model: Mapping[str, float] | None,
    p_idle_sin_squared: float | None,
    p_idle_sin_squared_model: Mapping[str, float] | None,
    p_idle_coherent: float | None,
    p_idle_coherent_model: Mapping[str, float] | None,
    p_idle_linear_rate: float | None,
    p_idle_quadratic_rate: float | None,
    p_idle_x_linear_rate: float | None,
    p_idle_y_linear_rate: float | None,
    p_idle_z_linear_rate: float | None,
    p_idle_quadratic_sine_rate: float | None,
    p_idle_x_quadratic_sine_rate: float | None,
    p_idle_y_quadratic_sine_rate: float | None,
    p_idle_z_quadratic_sine_rate: float | None,
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    """Convert Python objects and translate idle families using Rust."""
    from pecos_rslib.qec import translate_structured_idle_noise

    families = {}
    for family, rate, model in [
        ("coherent", p_idle_coherent, p_idle_coherent_model),
        ("linear", p_idle_linear, p_idle_linear_model),
        ("sin_squared", p_idle_sin_squared, p_idle_sin_squared_model),
    ]:
        rate_name = f"p_idle_{family}"
        model_name = f"{rate_name}_model"
        if isinstance(rate, bool):
            msg = f"{rate_name} must be a finite, non-negative float"
            raise TypeError(msg)
        if rate is not None:
            try:
                rate = float(rate)
            except (TypeError, ValueError) as exc:
                msg = f"{rate_name} must be a finite, non-negative float"
                raise ValueError(msg) from exc
        if model is not None:
            if not isinstance(model, Mapping):
                expected = "'RX', 'RY', 'RZ'" if family == "coherent" else "'L', 'X', 'Y', 'Z'"
                msg = f"{model_name} must be a mapping from {expected} to relative-rate multipliers"
                raise ValueError(msg)
            converted = {}
            for key, multiplier in model.items():
                if not isinstance(key, str):
                    expected = "'RX', 'RY', 'RZ'" if family == "coherent" else "'L', 'X', 'Y', 'Z'"
                    msg = f"invalid {model_name} key {key!r}; expected {expected}"
                    raise ValueError(msg)
                try:
                    converted[key] = float(multiplier)
                except (TypeError, ValueError) as exc:
                    msg = f"{model_name} multiplier for {key!r} must be a finite, non-negative float"
                    raise ValueError(msg) from exc
            model = converted
        families[rate_name] = rate
        families[model_name] = model

    axes = translate_structured_idle_noise(
        **families,
        p_idle_linear_rate=p_idle_linear_rate,
        p_idle_quadratic_rate=p_idle_quadratic_rate,
        p_idle_x_linear_rate=p_idle_x_linear_rate,
        p_idle_y_linear_rate=p_idle_y_linear_rate,
        p_idle_z_linear_rate=p_idle_z_linear_rate,
        p_idle_quadratic_sine_rate=p_idle_quadratic_sine_rate,
        p_idle_x_quadratic_sine_rate=p_idle_x_quadratic_sine_rate,
        p_idle_y_quadratic_sine_rate=p_idle_y_quadratic_sine_rate,
        p_idle_z_quadratic_sine_rate=p_idle_z_quadratic_sine_rate,
    )
    legacy_replacements = {
        "p_idle_linear_rate": (
            p_idle_linear_rate,
            (
                "p_idle_linear with p_idle_linear_model={'Z': 1.0} for the engines-consistent interface, "
                "or p_idle_z_linear_rate for literal Z-only behavior"
            ),
        ),
        "p_idle_quadratic_rate": (
            p_idle_quadratic_rate,
            (
                "p_idle_sin_squared for the engines-consistent dephasing interface, "
                "or p_idle_z_quadratic_rate for literal coefficient-style Z-only behavior"
            ),
        ),
        "p_idle_quadratic_sine_rate": (
            p_idle_quadratic_sine_rate,
            (
                "p_idle_sin_squared for the engines-consistent sine-law interface, "
                "or p_idle_z_quadratic_sine_rate for literal Z-only behavior"
            ),
        ),
    }
    for name, (value, replacement) in legacy_replacements.items():
        if value is not None:
            warnings.warn(
                f"{name} is deprecated; use {replacement}",
                DeprecationWarning,
                stacklevel=3,
            )

    return axes
