"""Guppy Frontend for PECOS.

This module provides integration between Guppy quantum programming language
and PECOS execution infrastructure through the HUGR intermediate representation.
"""

import contextlib
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from guppylang import guppy
from pecos_rslib import compile_hugr_to_qis


def _raise_external_compiler_error() -> None:
    """Raise ImportError for missing external compiler.

    This is extracted as a separate function to satisfy TRY301.
    """
    msg = "External compiler not available"
    raise ImportError(msg) from None


class GuppyFrontend:
    """Frontend for compiling Guppy quantum programs to QIR for PECOS execution.

    This class handles the complete pipeline:
    1. Guppy function → HUGR (using guppylang)
    2. HUGR format conversion (for compatibility)
    3. HUGR → LLVM IR/QIR (using hugr-llvm with quantum extensions)
    4. QIR execution on PECOS
    """

    def __init__(
        self,
        hugr_to_llvm_binary: Path | None = None,
        format_converter: Path | None = None,
        use_rust_backend: bool | None = None,
    ) -> None:
        """Initialize the Guppy frontend.

        Args:
            hugr_to_llvm_binary: Path to the hugr-to-llvm compiler binary (for external mode)
            format_converter: Path to the HUGR format converter script (for external mode)
            use_rust_backend: Force use of Rust backend (True) or external tools (False).
                             If None, auto-detect best available option.
        """
        # Initialize attributes first to avoid AttributeError in cleanup
        self._temp_dir = None

        # Determine backend to use (Rust backend is always available)
        self.use_rust_backend = (
            use_rust_backend if use_rust_backend is not None else True
        )

        # External tools configuration (used when Rust backend not requested)
        self.hugr_to_llvm_binary = hugr_to_llvm_binary
        self.format_converter = format_converter

    def get_backend_info(self) -> dict:
        """Get information about the backend being used."""
        return {
            "backend": "rust" if self.use_rust_backend else "external",
            "rust_available": True,  # HUGR support is always available
            "guppy_available": True,  # guppylang is now a required dependency
            "external_tools": {
                "hugr_to_llvm_binary": (
                    str(self.hugr_to_llvm_binary) if self.hugr_to_llvm_binary else None
                ),
                "format_converter": (
                    str(self.format_converter) if self.format_converter else None
                ),
            },
        }

    def compile_function(self, func: Callable) -> Path:
        """Compile a Guppy function to QIR.

        Args:
            func: A function decorated with @guppy

        Returns:
            Path to the generated QIR/LLVM IR file

        Raises:
            RuntimeError: If compilation fails at any stage
        """
        # Check if this is a Guppy function
        # GuppyDefinition objects have different attributes than regular functions
        is_guppy = (
            hasattr(func, "_guppy_compiled")
            or hasattr(func, "name")
            or str(type(func)).find("GuppyDefinition") != -1
            or str(type(func)).find("GuppyFunctionDefinition") != -1
        )

        if not is_guppy:
            msg = "Function must be decorated with @guppy"
            raise ValueError(msg)

        # Step 1: Compile Guppy to HUGR
        hugr_bytes = None
        try:
            # Try both new and old API
            compiled = (
                func.compile() if hasattr(func, "compile") else guppy.compile(func)
            )

            # Handle the return value - it might be a FuncDefnPointer or similar
            # Both Rust backend and Selene now use binary envelope format
            if hasattr(compiled, "to_bytes"):
                hugr_bytes = compiled.to_bytes()
            elif hasattr(compiled, "package"):
                hugr_bytes = compiled.package.to_bytes()
            elif hasattr(compiled, "to_package"):
                package = compiled.to_package()
                hugr_bytes = package.to_bytes()
        except Exception as e:
            msg = f"Failed to compile Guppy to HUGR: {e}"
            raise RuntimeError(msg) from e

        if hugr_bytes is None:
            msg = "Cannot serialize HUGR to binary envelope format"
            raise RuntimeError(msg)

        if self.use_rust_backend:
            # Use Rust backend for compilation
            return self._compile_with_rust_backend(func, hugr_bytes)
        # Use external tools for compilation
        return self._compile_with_external_tools(func, hugr_bytes)

    def _compile_with_rust_backend(self, func: Callable, hugr_bytes: bytes) -> Path:
        """Compile using Rust backend."""
        try:
            # Create temp directory for output
            if self._temp_dir is None:
                self._temp_dir = tempfile.mkdtemp(prefix="pecos_guppy_rust_")

            temp_path = Path(self._temp_dir)
            func_name = getattr(func, "__name__", getattr(func, "name", "guppy_func"))
            qir_file = temp_path / f"{func_name}.ll"

            # Compile HUGR to QIR using Rust backend
            # Use the configured naming convention
            qir_content = compile_hugr_to_qis(
                hugr_bytes,
                None,  # output_path
            )

            # Write QIR to file
            with Path(qir_file).open("w") as f:
                f.write(qir_content)

        except Exception as e:
            msg = f"Rust backend compilation failed: {e}"
            raise RuntimeError(msg) from e
        else:
            return qir_file

    def _compile_with_external_tools(self, func: Callable, hugr_bytes: bytes) -> Path:
        """Compile using external tools."""
        # Create temp directory for intermediate files
        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="pecos_guppy_external_")

        temp_path = Path(self._temp_dir)

        # Get function name safely
        func_name = getattr(func, "__name__", getattr(func, "name", "guppy_func"))

        # Write HUGR to file
        hugr_file = temp_path / f"{func_name}.hugr"
        with Path(hugr_file).open("wb") as f:
            f.write(hugr_bytes)

        # Step 2: Convert HUGR format if converter is available
        if self.format_converter:
            converted_hugr = temp_path / f"{func_name}_converted.hugr"
            try:
                subprocess.run(
                    [
                        "python",
                        str(self.format_converter),
                        str(hugr_file),
                        str(converted_hugr),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                hugr_file = converted_hugr
            except subprocess.CalledProcessError as e:
                msg = f"HUGR format conversion failed: {e.stderr}"
                raise RuntimeError(msg) from e

        # Step 3: Compile HUGR to LLVM IR/QIR
        qir_file = temp_path / f"{func_name}.ll"

        if self.hugr_to_llvm_binary:
            try:
                subprocess.run(
                    [
                        str(self.hugr_to_llvm_binary),
                        str(hugr_file),
                        str(qir_file),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                msg = f"HUGR to LLVM compilation failed: {e.stderr}"
                raise RuntimeError(msg) from e
        else:
            # Use PECOS HUGR compiler for real HUGR→LLVM compilation
            try:
                # Try to import the hugr_llvm_compiler
                from pecos._compilation.hugr_llvm import HugrLlvmCompiler

                compiler = HugrLlvmCompiler()
                if compiler.is_available():
                    # Use the external hugr_quantum_llvm binary
                    llvm_ir = compiler.compile_hugr_to_qis(
                        hugr_bytes,
                    )

                    qir_file = temp_path / f"{func_name}.ll"
                    with Path(qir_file).open("w") as f:
                        f.write(llvm_ir)

                    return qir_file
                _raise_external_compiler_error()

            except ImportError:
                # Fall back to execute_llvm if available
                pass

            try:
                # First try PECOS's own execute_llvm
                try:
                    from pecos import execute_llvm
                except ImportError:
                    # Try external execute_llvm
                    import execute_llvm

                # Compile HUGR bytes to LLVM IR string
                llvm_ir = execute_llvm.compile_module_to_string(hugr_bytes)

                # Write LLVM IR to file
                qir_file = temp_path / f"{func_name}.ll"
                with Path(qir_file).open("w") as f:
                    f.write(llvm_ir)

            except ImportError as e:
                # No fallback - we only support proper HUGR->LLVM compilation
                msg = (
                    "HUGR to LLVM compilation failed: No working HUGR compiler available. "
                    "The Rust backend (compile_hugr_to_qis) failed and no external "
                    "compiler is available. We only support proper HUGR convention LLVM-IR "
                    "generated via hugr-llvm, not fallback QIR."
                )
                raise RuntimeError(msg) from e
            else:
                return qir_file

    def cleanup(self) -> None:
        """Clean up temporary files."""
        if (
            hasattr(self, "_temp_dir")
            and self._temp_dir
            and Path(self._temp_dir).exists()
        ):

            shutil.rmtree(self._temp_dir)
            self._temp_dir = None

    def __del__(self) -> None:
        """Cleanup on destruction."""
        with contextlib.suppress(Exception):
            self.cleanup()


def compile_guppy_to_qir(
    func: Callable,
    hugr_to_llvm_binary: Path | None = None,
    format_converter: Path | None = None,
) -> Path:
    """Convenience function to compile a Guppy function to QIR.

    Args:
        func: A function decorated with @guppy
        hugr_to_llvm_binary: Path to the hugr-to-llvm compiler binary
        format_converter: Path to the HUGR format converter script

    Returns:
        Path to the generated QIR file
    """
    frontend = GuppyFrontend(hugr_to_llvm_binary, format_converter)
    try:
        return frontend.compile_function(func)
    finally:
        frontend.cleanup()


def run_guppy_on_pecos(
    func: Callable,
    shots: int = 1000,
    hugr_to_llvm_binary: Path | None = None,
    format_converter: Path | None = None,
) -> dict:
    """Convenience function to compile and run a Guppy function on PECOS.

    Args:
        func: A function decorated with @guppy
        shots: Number of shots to execute
        hugr_to_llvm_binary: Path to the hugr-to-llvm compiler binary
        format_converter: Path to the HUGR format converter script

    Returns:
        Dictionary containing execution results
    """
    frontend = GuppyFrontend(hugr_to_llvm_binary, format_converter)
    try:
        return frontend.compile_and_run(func, shots)
    finally:
        frontend.cleanup()


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

    # Compile Guppy → HUGR
    try:
        compiled = (
            guppy_func.compile()
            if hasattr(guppy_func, "compile")
            else guppy.compile(guppy_func)
        )

        if hasattr(compiled, "to_bytes"):
            return compiled.to_bytes()
        if hasattr(compiled, "package"):
            return compiled.package.to_bytes()
        if hasattr(compiled, "to_package"):
            package = compiled.to_package()
            return package.to_bytes()
        msg = "Cannot serialize HUGR to binary format"
        raise RuntimeError(msg)
    except Exception as e:
        msg = f"Failed to compile Guppy to HUGR: {e}"
        raise RuntimeError(msg) from e
