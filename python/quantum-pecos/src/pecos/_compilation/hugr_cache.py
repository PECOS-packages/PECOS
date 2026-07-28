# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Per-definition cache of compiled Guppy HUGR bytes.

Guppy -> HUGR compilation is deterministic per definition object but not
cached by guppylang, and one DEM build compiles the same program several
times (generator certificate, preflight digest check, trace execution).

``GuppyFunctionDefinition`` is weakref-able but unhashable, so a
``WeakKeyDictionary`` cannot hold it. Entries are keyed by ``id()`` and
validated against a stored weakref on lookup, with a finalizer evicting the
entry when the definition dies; a recycled ``id`` therefore never returns
another program's bytes. This module is deliberately dependency-free so both
compile entry points can share it cheaply.
"""

from __future__ import annotations

import inspect
import weakref
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

_hugr_bytes_cache: dict[int, tuple[weakref.ref, bytes]] = {}


def definition_takes_parameters(guppy_func: Callable) -> bool:
    """Best-effort check whether a Guppy definition takes entry-point parameters.

    The compile entry points diverge for parametric definitions
    (``compile_function()`` library form vs entry-point ``compile()``), so a
    parametric definition must never share a cache entry with the entry-point
    form. Guppy definitions do not expose ``__wrapped__``; the raw Python
    function sits at ``wrapped.python_func``. When no real signature is
    reachable, the definition is treated as parametric -- the conservative
    direction (parametric definitions are simply never cached).
    """
    target = getattr(getattr(guppy_func, "wrapped", None), "python_func", None)
    if target is None:
        target = getattr(guppy_func, "__wrapped__", guppy_func)
    try:
        return len(inspect.signature(target).parameters) > 0
    except (TypeError, ValueError):
        return True


def lookup_cached_hugr_bytes(guppy_func: Callable) -> bytes | None:
    """Return previously compiled HUGR bytes for this definition object."""
    entry = _hugr_bytes_cache.get(id(guppy_func))
    if entry is None:
        return None
    ref, hugr_bytes = entry
    return hugr_bytes if ref() is guppy_func else None


def store_cached_hugr_bytes(guppy_func: Callable, hugr_bytes: bytes) -> None:
    """Cache compiled HUGR bytes for this definition object's lifetime."""
    try:
        ref = weakref.ref(guppy_func)
        weakref.finalize(guppy_func, _hugr_bytes_cache.pop, id(guppy_func), None)
    except TypeError:  # not weakref-able: cache not applicable
        return
    _hugr_bytes_cache[id(guppy_func)] = (ref, hugr_bytes)
