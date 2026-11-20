"""Validate runnable Guppy entry points before canonical compilation."""

from collections.abc import Callable

from guppylang.defs import GuppyFunctionDefinition

from pecos._compilation.hugr_cache import definition_takes_parameters
from pecos.compilation_pipeline import compile_guppy_to_hugr


def guppy_to_hugr(guppy_func: Callable, *, wasm: bool = False) -> bytes:
    """Compile a runnable entry point without parameters to HUGR bytes."""
    is_guppy = isinstance(guppy_func, GuppyFunctionDefinition)
    if not is_guppy:
        msg = "Function must be decorated with @guppy"
        raise ValueError(msg)
    if definition_takes_parameters(guppy_func):
        msg = "Failed to compile Guppy to HUGR: a runnable entry point without parameters is required"
        raise RuntimeError(msg)
    if wasm:
        # Keep the WASM extra optional for ordinary Guppy programs.
        try:
            from pecos_selene_wasm import compile_guppy  # noqa: PLC0415
        except ImportError as error:
            msg = "Guppy WASM lowering requires the pecos-selene-wasm package (quantum-pecos[wasm])"
            raise ImportError(msg) from error

        return compile_guppy(guppy_func).to_bytes()
    return compile_guppy_to_hugr(guppy_func)
