"""Immutable options shared by the Guppy/HUGR WASM entry points."""

from __future__ import annotations

import math
import reprlib
from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, fields
from functools import partial
from types import MappingProxyType


def _describe_value(value: object) -> str:
    """Return a short repr of a rejected option for an error message; never raise.

    reprlib bounds the length. Formatting can still fail (an int longer than
    ``sys.get_int_max_str_digits()`` before Python 3.14, or a hostile object), and that
    must not replace the validation error, so any failure yields a fixed placeholder.
    """
    try:
        return reprlib.repr(value)
    except Exception:  # noqa: BLE001 - a diagnostic must never replace the error it reports
        return "<unprintable value>"


@dataclass(frozen=True)
class WasmOptions:
    """Validated WASM modules and execution policy.

    ``modules`` maps exact HUGR module identifiers to explicit WAT or WASM bytes.
    The mapping is copied and frozen. All other arguments are keyword-only.
    Integer widths are 32, 64, or ``"auto"`` to follow native export signatures.
    The positive finite timeout applies separately to start, init, and each call.
    Disabling ``auto_init`` skips optional exported init, never WASM start.
    State is fresh per shot unless ``state_scope="worker"`` retains it across
    clean shots. ``async_workers`` is 0 for inline calls or 1..64 for a pool.
    """

    __hash__ = None

    modules: Mapping[str, bytes]
    _: KW_ONLY
    int_width: int | str = 64
    timeout_seconds: float = 1.0
    auto_init: bool = True
    state_scope: str = "shot"
    async_workers: int = 0

    def __post_init__(self) -> None:
        """Validate in declaration order and snapshot caller-owned modules."""
        if not isinstance(self.modules, Mapping):
            msg = (
                "WasmOptions.modules must be a Mapping; "
                f"got {type(self.modules).__name__}: {_describe_value(self.modules)}"
            )
            raise TypeError(msg)
        modules = dict(self.modules)
        for key, value in modules.items():
            if not isinstance(key, str):
                msg = f"WasmOptions.modules key {_describe_value(key)} must be a str; got {type(key).__name__}"
                raise TypeError(msg)
            if not isinstance(value, bytes):
                msg = (
                    f"WasmOptions.modules[{_describe_value(key)}] must be explicit bytes, not paths; "
                    f"got {type(value).__name__}: {_describe_value(value)}"
                )
                raise TypeError(msg)
        object.__setattr__(self, "modules", MappingProxyType(modules))
        if not (
            (type(self.int_width) is int and self.int_width in (32, 64))
            or (type(self.int_width) is str and self.int_width == "auto")
        ):
            msg = f"WasmOptions.int_width must be 32 or 64, or 'auto'; got {_describe_value(self.int_width)}"
            raise ValueError(msg)
        try:
            valid_timeout = (
                type(self.timeout_seconds) in (int, float)
                and math.isfinite(self.timeout_seconds)
                and self.timeout_seconds > 0
            )
        except OverflowError:
            valid_timeout = False
        if not valid_timeout:
            msg = (
                "WasmOptions.timeout_seconds must be a positive finite number of seconds; "
                f"got {_describe_value(self.timeout_seconds)}"
            )
            raise ValueError(msg)
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))
        if type(self.auto_init) is not bool:
            msg = f"WasmOptions.auto_init must be True or False; got {_describe_value(self.auto_init)}"
            raise ValueError(msg)
        if type(self.state_scope) is not str or self.state_scope not in ("shot", "worker"):
            msg = f"WasmOptions.state_scope must be 'shot' or 'worker'; got {_describe_value(self.state_scope)}"
            raise ValueError(msg)
        if type(self.async_workers) is not int or not 0 <= self.async_workers <= 64:
            msg = (
                "WasmOptions.async_workers must be an integer between 0 and 64; "
                f"got {_describe_value(self.async_workers)}"
            )
            raise ValueError(msg)

    def __repr__(self) -> str:
        """Show module identifiers and sizes without exposing payload bytes."""
        modules = ", ".join(f"{key!r}: <{len(value)} bytes>" for key, value in self.modules.items())
        scalars = ", ".join(f"{field.name}={getattr(self, field.name)!r}" for field in fields(self)[1:])
        return f"WasmOptions(modules={{{modules}}}, {scalars})"

    def __reduce__(self) -> tuple[object, tuple[dict[str, bytes]]]:
        """Rebuild through validation for pickle and deepcopy."""
        scalars = {field.name: getattr(self, field.name) for field in fields(self)[1:]}
        return partial(type(self), **scalars), (dict(self.modules),)
