"""Validate runnable Guppy entry points before canonical compilation."""

from collections.abc import Callable

from guppylang.defs import GuppyFunctionDefinition

from pecos._compilation.hugr_cache import definition_takes_parameters
from pecos.compilation_pipeline import compile_guppy_to_hugr


def guppy_to_hugr(guppy_func: Callable) -> bytes:
    """Compile a runnable entry point without parameters to HUGR bytes."""
    is_guppy = isinstance(guppy_func, GuppyFunctionDefinition)
    if not is_guppy:
        msg = "Function must be decorated with @guppy"
        raise ValueError(msg)
    if definition_takes_parameters(guppy_func):
        msg = "Failed to compile Guppy to HUGR: a runnable entry point without parameters is required"
        raise RuntimeError(msg)
    return compile_guppy_to_hugr(guppy_func)
