"""Clean API for the quantum compilation pipeline.

This module provides a structured interface for the compilation pipeline:
1. Guppy -> HUGR (Python)
2. HUGR -> LLVM/QIR (Rust via PyO3)
3. LLVM/QIR -> Execution (PECOS)
"""

import tempfile
from collections.abc import Callable
from pathlib import Path

from pecos.exceptions import HugrTypeError


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
    # Check if this is a Guppy function
    is_guppy = (
        hasattr(guppy_function, "_guppy_compiled")
        or hasattr(guppy_function, "name")
        or str(type(guppy_function)).find("GuppyDefinition") != -1
        or str(type(guppy_function)).find("GuppyFunctionDefinition") != -1
    )

    if not is_guppy:
        msg = "Function must be decorated with @guppy"
        raise ValueError(msg)

    from pecos._compilation.hugr_cache import (
        definition_takes_parameters,
        lookup_cached_hugr_bytes,
        store_cached_hugr_bytes,
    )

    cached = lookup_cached_hugr_bytes(guppy_function)
    if cached is not None:
        return cached

    # guppylang's compile()/compile_function() both return a hugr `Package`.
    # Parametric functions must use compile_function() (compile() needs entry-point
    # arguments); non-parametric functions use compile() for the entry point.
    # Only the entry-point form is cached: the two forms are not
    # interchangeable, and guppy_to_hugr only ever produces the entry-point one.
    has_params = definition_takes_parameters(guppy_function)
    try:
        compiled = guppy_function.compile_function() if has_params else guppy_function.compile()
    except Exception as e:
        msg = f"Failed to compile Guppy to HUGR: {e}"
        raise RuntimeError(msg) from e

    # Serialize the Package as the BINARY HUGR envelope (Model format). The Selene/QIS
    # engine's HUGR reader rejects hugr-py 0.16's S-expression *text* envelope
    # (`to_str`) with "Failed to read HUGR", whereas the binary Model form round-trips
    # cleanly, including CFG loops (while statements).
    hugr_bytes = compiled.to_bytes()
    if not has_params:
        store_cached_hugr_bytes(guppy_function, hugr_bytes)
    return hugr_bytes


# Step 2: HUGR -> LLVM/QIR
def compile_hugr_to_qis(
    hugr_bytes: bytes,
    *,
    _debug_info: bool = False,
) -> str:
    """Compile HUGR bytes to LLVM IR string.

    Args:
        hugr_bytes: HUGR package as bytes
        debug_info: Whether to include debug information

    Returns:
        LLVM IR as string (HUGR convention)

    Raises:
        ImportError: If no HUGR backend is available
        RuntimeError: If compilation fails
    """
    # Try to use PECOS's HUGR to LLVM compiler
    try:
        from pecos_rslib_llvm import compile_hugr_to_qis

        rust_backend_available = True
    except ImportError:
        rust_backend_available = False

    if rust_backend_available:
        try:
            return compile_hugr_to_qis(
                hugr_bytes,
                None,
            )
        except RuntimeError as e:
            error_msg = str(e)
            if "Unknown type:" in error_msg:
                raise HugrTypeError(error_msg) from e
            msg = f"Failed to compile HUGR to LLVM: {e}"
            raise RuntimeError(msg) from e
    else:
        # Try our execute_llvm module as fallback
        try:
            from pecos import execute_llvm

            return execute_llvm.compile_module_to_string(hugr_bytes)
        except Exception as e:
            msg = "No HUGR backend available. Build PECOS with HUGR support."
            raise ImportError(
                msg,
            ) from e


# Step 3: Execute LLVM/QIR
def execute_llvm(
    llvm_ir: str | Path,
    shots: int = 1000,
    config: dict | None = None,
) -> dict:
    """Execute LLVM IR/QIR code.

    Args:
        llvm_ir: LLVM IR as string or path to file
        shots: Number of shots to run
        config: Optional execution configuration

    Returns:
        Execution results dictionary

    Raises:
        ImportError: If execution backend is not available
        RuntimeError: If execution fails
    """
    try:
        from pecos_rslib import execute_llvm
    except ImportError as err:
        msg = "LLVM execution backend not available"
        raise ImportError(msg) from err

    # If llvm_ir is a string content, write to temporary file
    if isinstance(llvm_ir, str) and not Path(llvm_ir).exists():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ll", delete=False) as f:
            f.write(llvm_ir)
            temp_path = f.name
        try:
            result = execute_llvm(temp_path, shots, config)
        finally:
            temp_file = Path(temp_path)
            if temp_file.exists():
                temp_file.unlink()
    else:
        # It's a path
        result = execute_llvm(str(llvm_ir), shots, config)

    return {
        "results": result.get("results", []),
        "shots": shots,
        "backend": "pecos_llvm_runtime",
    }


# Convenience functions for common pipelines
def compile_guppy_to_llvm(
    guppy_function: Callable,
    *,
    debug_info: bool = False,
) -> str:
    """Compile a Guppy function directly to LLVM IR.

    Args:
        guppy_function: A function decorated with @guppy
        debug_info: Whether to include debug information

    Returns:
        LLVM IR as string (HUGR convention)
    """
    hugr_bytes = compile_guppy_to_hugr(guppy_function)
    return compile_hugr_to_qis(hugr_bytes, debug_info=debug_info)


def run_guppy_function(
    guppy_function: Callable,
    shots: int = 1000,
    *,
    debug_info: bool = False,
) -> dict:
    """Compile and execute a Guppy function.

    Args:
        guppy_function: A function decorated with @guppy
        shots: Number of shots to run
        debug_info: Whether to include debug information

    Returns:
        Execution results dictionary
    """
    llvm_ir = compile_guppy_to_llvm(
        guppy_function,
        debug_info=debug_info,
    )
    return execute_llvm(llvm_ir, shots)


# Export all functions
__all__ = [
    # Core pipeline functions
    "compile_guppy_to_hugr",
    "compile_guppy_to_llvm",
    "compile_hugr_to_qis",
    "execute_llvm",
    # Convenience functions
    "run_guppy_function",
]
