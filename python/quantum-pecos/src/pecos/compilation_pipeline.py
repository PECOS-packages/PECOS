"""Clean API for the quantum compilation pipeline.

This module provides a structured interface for the compilation pipeline:
1. Guppy -> HUGR (Python)
2. HUGR -> LLVM/QIR (Selene's compiler package)
"""

from collections.abc import Callable

from pecos_rslib.hugr_lowering import compile_hugr_to_qis


# Step 1: Guppy -> HUGR
def compile_guppy_to_hugr(guppy_function: Callable) -> bytes:
    """Compile a Guppy function to HUGR bytes.

    Args:
        guppy_function: A function decorated with @guppy

    Returns:
        HUGR package as bytes

    Raises:
        ValueError: If function is not a Guppy function
        RuntimeError: If compilation fails
    """
    from guppylang.defs import GuppyFunctionDefinition

    is_guppy = isinstance(guppy_function, GuppyFunctionDefinition)
    if not is_guppy:
        msg = "Function must be decorated with @guppy"
        raise ValueError(msg)

    from pecos._compilation.hugr_cache import (
        definition_takes_parameters,
        lookup_cached_hugr_bytes,
        store_cached_hugr_bytes,
    )

    has_params = definition_takes_parameters(guppy_function)
    if not has_params:
        cached = lookup_cached_hugr_bytes(guppy_function)
        if cached is not None:
            return cached

    # guppylang's compile()/compile_function() both return a hugr `Package`.
    # Parametric functions must use compile_function() (compile() needs entry-point
    # arguments); non-parametric functions use compile() for the entry point.
    # Only the entry-point form is cached: the two forms are not
    # interchangeable, and guppy_to_hugr only ever produces the entry-point one.
    try:
        compiled = guppy_function.compile_function() if has_params else guppy_function.compile()
    except Exception as e:
        msg = f"Failed to compile Guppy to HUGR: {e}"
        raise RuntimeError(msg) from e

    # Use the binary HUGR envelope for cached packages. The lowering boundary
    # also accepts text envelopes supplied directly by callers.
    hugr_bytes = compiled.to_bytes()
    if not has_params:
        store_cached_hugr_bytes(guppy_function, hugr_bytes)
    return hugr_bytes


__all__ = ["compile_guppy_to_hugr", "compile_hugr_to_qis"]
