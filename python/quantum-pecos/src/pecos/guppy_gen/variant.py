# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Variant-scoped Guppy program factories.

guppylang registers each ``@guppy`` function by its module-qualified name, and
the downstream Selene build cache keys compiled executables by the HUGR
entry-point name (that same function name). A factory that builds a ``@guppy``
program closing over compile-time parameters therefore hands every
parameterization the *same* name::

    def make_prog(num_rounds):
        @guppy
        def prog() -> None:          # __qualname__ == "make_prog.<locals>.prog"
            for _ in range(comptime(num_rounds)):
                ...
        return prog

Building two parameterizations in one process then collides: the second reuses
the first's cached compilation / Selene executable and silently runs the wrong
program. :func:`variant_scoped` gives each parameterization a distinct name so its
identity -- and thus its HUGR entry point and Selene build -- is unique.
"""

from collections.abc import Callable
from typing import TypeVar

F = TypeVar("F", bound=Callable)

__all__ = ["variant_scoped"]


def _name_fragment(value: object) -> str:
    """Render a variant value as an identifier-safe name fragment."""
    return "".join(ch if ch.isalnum() else "_" for ch in str(value))


def variant_scoped(func: F, *variant: object) -> F:
    """Rename ``func`` with a ``variant`` suffix for a parameterization-unique ``guppy()`` program.

    Define the program as a plain (undecorated) nested function and wrap it with
    ``guppy()`` at the call site, passing the values that distinguish this
    parameterization::

        from guppylang import guppy
        from pecos.guppy_gen import variant_scoped

        def make_prog(num_rounds):
            def prog() -> None:
                for _ in range(comptime(num_rounds)):
                    ...
            return guppy(variant_scoped(prog, num_rounds))

    ``variant_scoped`` only renames ``func`` (setting ``__name__``/``__qualname__``)
    and returns it; **you** apply ``guppy()`` to the result. This is required
    because guppylang resolves the program's names (gates, ``result``, ...) against
    the *caller's* module, so ``guppy()`` must be invoked where those names are in
    scope -- a helper that called ``guppy()`` itself would fail to resolve them.

    The variant suffix gives each parameterization a distinct module-qualified
    name (so guppylang does not reuse a stale compiled body) and a distinct HUGR
    entry-point name (so the Selene build cache does not reuse a stale
    executable). Without it, building two parameterizations in one process makes
    the second silently execute the first's program.

    Args:
        func: The undecorated program function to rename in place.
        variant: One or more values identifying this parameterization (e.g. the
            round count, distance). At least one is required; each is stringified
            and sanitized into the name suffix.

    Returns:
        ``func`` (renamed), ready to pass to ``guppy()`` in the caller's module.
    """
    if not variant:
        msg = "variant_scoped requires at least one distinguishing value"
        raise ValueError(msg)
    suffix = "_" + "_".join(_name_fragment(v) for v in variant)
    func.__name__ = f"{func.__name__}{suffix}"
    func.__qualname__ = f"{func.__qualname__}{suffix}"
    return func
