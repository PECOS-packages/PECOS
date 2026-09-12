"""Compile Guppy entry points to cached HUGR envelopes."""

from collections.abc import Callable

from guppylang import guppy

from pecos._compilation.hugr_cache import (
    definition_takes_parameters,
    lookup_cached_hugr_bytes,
    store_cached_hugr_bytes,
)


def guppy_to_hugr(guppy_func: Callable) -> bytes:
    """Convert a Guppy function to HUGR bytes.

    This function compiles a Guppy quantum program to HUGR format, which can then
    be executed by HUGR-compatible engines like Selene.

    Args:
        guppy_func: A function decorated with @guppy

    Returns:
        HUGR program as bytes

    Raises:
        ValueError: If the function is not a Guppy function
        RuntimeError: If compilation fails
    """
    # Check if this is a Guppy function
    is_guppy = (
        hasattr(guppy_func, "_guppy_compiled")
        or hasattr(guppy_func, "compile")
        or str(type(guppy_func)).find("GuppyDefinition") != -1
        or str(type(guppy_func)).find("GuppyFunctionDefinition") != -1
    )

    if not is_guppy:
        msg = "Function must be decorated with @guppy"
        raise ValueError(msg)

    # Parametric definitions never share cache entries: this entry point only
    # produces (and its callers only expect) the entry-point compile() form,
    # which rejects parameters.
    is_parametric = definition_takes_parameters(guppy_func)
    if not is_parametric:
        cached = lookup_cached_hugr_bytes(guppy_func)
        if cached is not None:
            return cached

    # Compile Guppy → HUGR
    try:
        compiled = guppy_func.compile() if hasattr(guppy_func, "compile") else guppy.compile(guppy_func)

        if hasattr(compiled, "to_bytes"):
            hugr_bytes = compiled.to_bytes()
        elif hasattr(compiled, "package"):
            hugr_bytes = compiled.package.to_bytes()
        elif hasattr(compiled, "to_package"):
            hugr_bytes = compiled.to_package().to_bytes()
        else:
            msg = "Cannot serialize HUGR to binary format"
            raise RuntimeError(msg)
    except Exception as e:
        msg = f"Failed to compile Guppy to HUGR: {e}"
        raise RuntimeError(msg) from e
    if not is_parametric:
        store_cached_hugr_bytes(guppy_func, hugr_bytes)
    return hugr_bytes
