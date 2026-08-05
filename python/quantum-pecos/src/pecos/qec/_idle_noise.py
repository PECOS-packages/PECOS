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

import math
import warnings
from collections.abc import Mapping

_IDLE_MODEL_NORMALIZATION_TOLERANCE = 1.0e-5
_IDLE_MODEL_FLOAT_EPSILON = 1.0e-10


def _validate_idle_family_model(
    *,
    rate: float | None,
    rate_name: str,
    model: Mapping[str, float] | None,
    model_name: str,
    default_model: Mapping[str, float],
    accepted_keys: frozenset[str],
    require_normalized: bool,
    nonzero_rate_guidance: str | None = None,
    zero_only_key_guidance: Mapping[str, str] | None = None,
) -> tuple[float, dict[str, float]] | None:
    """Validate one structured idle family and return its rate and multipliers."""
    if model is not None and rate is None:
        msg = f"{model_name} requires {rate_name}; otherwise the model is inert"
        raise ValueError(msg)
    if rate is None:
        return None

    if isinstance(rate, bool):
        msg = f"{rate_name} must be a finite, non-negative float"
        raise TypeError(msg)
    try:
        numeric_rate = float(rate)
    except (TypeError, ValueError) as exc:
        msg = f"{rate_name} must be a finite, non-negative float"
        raise ValueError(msg) from exc
    if not math.isfinite(numeric_rate) or numeric_rate < 0.0:
        msg = f"{rate_name} must be a finite, non-negative float"
        raise ValueError(msg)
    if numeric_rate != 0.0 and nonzero_rate_guidance is not None:
        raise ValueError(nonzero_rate_guidance)

    if model is not None and not isinstance(model, Mapping):
        expected = ", ".join(repr(key) for key in sorted(accepted_keys))
        msg = f"{model_name} must be a mapping from {expected} to relative-rate multipliers"
        raise ValueError(msg)
    selected_model = model if model is not None else default_model
    validated_model: dict[str, float] = {}
    for key, multiplier in selected_model.items():
        if key not in accepted_keys:
            expected = ", ".join(repr(valid_key) for valid_key in sorted(accepted_keys))
            msg = f"invalid {model_name} key {key!r}; expected {expected}"
            raise ValueError(msg)
        try:
            numeric_multiplier = float(multiplier)
        except (TypeError, ValueError) as exc:
            msg = f"{model_name} multiplier for {key!r} must be a finite, non-negative float"
            raise ValueError(msg) from exc
        if not math.isfinite(numeric_multiplier) or numeric_multiplier < 0.0:
            msg = f"{model_name} multiplier for {key!r} must be a finite, non-negative float"
            raise ValueError(msg)
        validated_model[key] = numeric_multiplier

    if require_normalized:
        total_multiplier = sum(validated_model.values())
        if total_multiplier <= 0.0 or abs(total_multiplier - 1.0) > _IDLE_MODEL_NORMALIZATION_TOLERANCE:
            msg = (
                f"{model_name} multipliers must sum to 1.0 within tolerance "
                f"{_IDLE_MODEL_NORMALIZATION_TOLERANCE:g}; got {total_multiplier}"
            )
            raise ValueError(msg)
        if abs(total_multiplier - 1.0) > _IDLE_MODEL_FLOAT_EPSILON:
            validated_model = {key: multiplier / total_multiplier for key, multiplier in validated_model.items()}

    for key, guidance in (zero_only_key_guidance or {}).items():
        if validated_model.get(key, 0.0) != 0.0:
            msg = f"{model_name} key {key!r} has a nonzero multiplier; {guidance}"
            raise ValueError(msg)

    return numeric_rate, validated_model


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
    """Validate and translate engines-style idle noise to DEM primitives."""
    _validate_idle_family_model(
        rate=p_idle_coherent,
        rate_name="p_idle_coherent",
        model=p_idle_coherent_model,
        model_name="p_idle_coherent_model",
        default_model={"RX": 1.0, "RY": 1.0, "RZ": 1.0},
        accepted_keys=frozenset({"RX", "RY", "RZ"}),
        require_normalized=False,
        nonzero_rate_guidance=(
            "the standard DEM builder cannot represent coherent idle noise; its previous behavior silently stored "
            "the Pauli twirl, discarding exactly the coherence that was requested. The EEG coherent route in "
            "exp/pecos-eeg is the consumer that can represent it, and only with an RZ generator even there. The "
            "honest stochastic equivalent, which is the exact Pauli twirl of RZ(rate * t), is "
            "p_idle_sin_squared=rate/2 with p_idle_sin_squared_model={'Z': 1.0}"
        ),
    )

    linear_primitives = {
        "p_idle_linear_rate": p_idle_linear_rate,
        "p_idle_x_linear_rate": p_idle_x_linear_rate,
        "p_idle_y_linear_rate": p_idle_y_linear_rate,
        "p_idle_z_linear_rate": p_idle_z_linear_rate,
    }
    if (p_idle_linear is not None or p_idle_linear_model is not None) and any(
        value is not None for value in linear_primitives.values()
    ):
        conflicts = ", ".join(name for name, value in linear_primitives.items() if value is not None)
        msg = f"p_idle_linear/p_idle_linear_model cannot be combined with low-level idle rate(s): {conflicts}"
        raise ValueError(msg)
    sine_primitives = {
        "p_idle_quadratic_sine_rate": p_idle_quadratic_sine_rate,
        "p_idle_x_quadratic_sine_rate": p_idle_x_quadratic_sine_rate,
        "p_idle_y_quadratic_sine_rate": p_idle_y_quadratic_sine_rate,
        "p_idle_z_quadratic_sine_rate": p_idle_z_quadratic_sine_rate,
    }
    if (p_idle_sin_squared is not None or p_idle_sin_squared_model is not None) and any(
        value is not None for value in sine_primitives.values()
    ):
        conflicts = ", ".join(name for name, value in sine_primitives.items() if value is not None)
        msg = f"p_idle_sin_squared/p_idle_sin_squared_model cannot be combined with sine-law idle rate(s): {conflicts}"
        raise ValueError(msg)

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

    linear_family = _validate_idle_family_model(
        rate=p_idle_linear,
        rate_name="p_idle_linear",
        model=p_idle_linear_model,
        model_name="p_idle_linear_model",
        default_model={"X": 1.0 / 3.0, "Y": 1.0 / 3.0, "Z": 1.0 / 3.0},
        accepted_keys=frozenset({"X", "Y", "Z", "L"}),
        require_normalized=True,
        zero_only_key_guidance={
            "L": "DEM fault propagation is Pauli-only; the engines simulators support and consume leakage models",
        },
    )
    if linear_family is not None:
        linear_rate, linear_model = linear_family
        p_idle_x_linear_rate = linear_rate * linear_model.get("X", 0.0)
        p_idle_y_linear_rate = linear_rate * linear_model.get("Y", 0.0)
        p_idle_z_linear_rate = linear_rate * linear_model.get("Z", 0.0)

    sin_squared_family = _validate_idle_family_model(
        rate=p_idle_sin_squared,
        rate_name="p_idle_sin_squared",
        model=p_idle_sin_squared_model,
        model_name="p_idle_sin_squared_model",
        default_model={"X": 1.0, "Y": 1.0, "Z": 1.0},
        accepted_keys=frozenset({"X", "Y", "Z", "L"}),
        require_normalized=False,
        zero_only_key_guidance={
            "L": "DEM fault propagation is Pauli-only; the engines simulators support and consume leakage models",
        },
    )
    if sin_squared_family is not None:
        sin_squared_rate, sin_squared_model = sin_squared_family
        p_idle_x_quadratic_sine_rate = (
            sin_squared_rate * sin_squared_model["X"] if sin_squared_model.get("X", 0.0) != 0.0 else None
        )
        p_idle_y_quadratic_sine_rate = (
            sin_squared_rate * sin_squared_model["Y"] if sin_squared_model.get("Y", 0.0) != 0.0 else None
        )
        p_idle_z_quadratic_sine_rate = (
            sin_squared_rate * sin_squared_model["Z"] if sin_squared_model.get("Z", 0.0) != 0.0 else None
        )

    return (
        p_idle_x_linear_rate,
        p_idle_y_linear_rate,
        p_idle_z_linear_rate,
        p_idle_x_quadratic_sine_rate,
        p_idle_y_quadratic_sine_rate,
        p_idle_z_quadratic_sine_rate,
    )
