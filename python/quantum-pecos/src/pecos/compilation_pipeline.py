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
        ImportError: If guppylang is not available
        ValueError: If function is not a Guppy function
        RuntimeError: If compilation fails
    """
    try:
        from guppylang import guppy as guppy_module
    except ImportError as err:
        msg = (
            "guppylang is not available. Install with: pip install quantum-pecos[guppy]"
        )
        raise ImportError(
            msg,
        ) from err

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

    try:
        # Check if this is a parametric function (has arguments)
        import inspect

        sig = inspect.signature(
            (
                guppy_function.__wrapped__
                if hasattr(guppy_function, "__wrapped__")
                else guppy_function
            ),
        )
        has_params = len(sig.parameters) > 0

        if has_params:
            # For parametric functions, use compile_function() which allows parameters
            if hasattr(guppy_function, "compile_function"):
                compiled = guppy_function.compile_function()
            else:
                # Fall back to regular compile and let it handle the error
                compiled = guppy_function.compile()
        else:
            # For non-parametric functions, use compile() for entrypoint
            if hasattr(guppy_function, "compile"):
                # New API: function.compile()
                compiled = guppy_function.compile()
            else:
                # Old API: guppy.compile(function)
                compiled = guppy_module.compile(guppy_function)

        # Handle the return value - it might be a FuncDefnPointer or similar
        # Use the new HUGR envelope methods (to_str/to_bytes) instead of deprecated to_json
        if hasattr(compiled, "to_str"):
            # Use string format for JSON compatibility with HUGR 0.13 compiler
            return compiled.to_str().encode("utf-8")
        if hasattr(compiled, "to_json"):
            # Fallback to to_json for older versions (with deprecation warning)
            return compiled.to_json().encode("utf-8")

        if hasattr(compiled, "package"):
            if hasattr(compiled.package, "to_str"):
                return compiled.package.to_str().encode("utf-8")
            if hasattr(compiled.package, "to_json"):
                return compiled.package.to_json().encode("utf-8")
            return compiled.package.to_bytes()

        if hasattr(compiled, "to_package"):
            package = compiled.to_package()
            if hasattr(package, "to_str"):
                return package.to_str().encode("utf-8")
            if hasattr(package, "to_json"):
                return package.to_json().encode("utf-8")
            return package.to_bytes()

        # Try to serialize directly
        return compiled.to_bytes()
    except Exception as e:
        msg = f"Failed to compile Guppy to HUGR: {e}"
        raise RuntimeError(msg) from e


# Step 2: HUGR -> LLVM/QIR
def _update_tket_wasm_version(hugr_bytes: bytes) -> bytes:
    """Update tket.wasm version from 0.3.0 to 0.4.1 for compatibility.

    Args:
        hugr_bytes: HUGR package bytes

    Returns:
        Updated HUGR bytes with tket.wasm 0.4.1
    """
    import json

    hugr_str = hugr_bytes.decode("utf-8")

    # Check if it starts with the envelope header
    if hugr_str.startswith("HUGRiHJv"):
        # Find where the JSON starts
        json_start = hugr_str.find("{", 8)
        if json_start != -1:
            header = hugr_str[:json_start]
            json_part = hugr_str[json_start:]

            # Parse the JSON
            hugr_data = json.loads(json_part)

            # Update version in extensions
            if "extensions" in hugr_data:
                for ext in hugr_data["extensions"]:
                    if ext.get("name") == "tket.wasm" and ext.get("version") == "0.3.0":
                        ext["version"] = "0.4.1"

            # Update version in module metadata
            if hugr_data.get("modules"):
                module = hugr_data["modules"][0]
                if "metadata" in module:
                    for meta_item in module["metadata"]:
                        if (
                            isinstance(meta_item, dict)
                            and "core.used_extensions" in meta_item
                        ):
                            for ext in meta_item["core.used_extensions"]:
                                if (
                                    ext.get("name") == "tket.wasm"
                                    and ext.get("version") == "0.3.0"
                                ):
                                    ext["version"] = "0.4.1"

            # Reconstruct the HUGR envelope
            modified_json = json.dumps(hugr_data, separators=(",", ":"))
            modified_hugr = header + modified_json
            return modified_hugr.encode("utf-8")

    return hugr_bytes


def compile_hugr_to_llvm(
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
        from pecos_rslib import compile_hugr_to_llvm_rust

        rust_backend_available = True
    except ImportError:
        rust_backend_available = False

    if rust_backend_available:
        try:
            return compile_hugr_to_llvm_rust(
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
    return compile_hugr_to_llvm(hugr_bytes, debug_info=debug_info)


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
    "compile_hugr_to_llvm",
    "execute_llvm",
    # Convenience functions
    "run_guppy_function",
]
