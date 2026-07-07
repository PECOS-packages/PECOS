"""Zluppy: A Zig/SLR/NASA Power of 10 quantum programming language.

This module provides Python bindings for compiling and running Zluppy programs.

Example:
    >>> import zluppy
    >>> from pecos import hugr_engine
    >>>
    >>> # Compile and run a Bell state program
    >>> hugr_bytes = (
    ...     zluppy.ZluppyEngine()
    ...     .source(
    ...         '''
    ...     fn main() -> void {
    ...         var q = qalloc(2);
    ...         H(q[0]);
    ...         CX(q[0], q[1]);
    ...     }
    ... '''
    ...     )
    ...     .to_hugr_bytes()
    ... )
    >>>
    >>> result = hugr_engine().hugr_bytes(hugr_bytes).to_sim().run(shots=100)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Re-export everything from the Rust extension
from zluppy._zluppy import (
    ZluppyError,
    compile_to_slr,
    compile_to_slr_json,
    compile_to_hugr,
    compile_file,
    compile_file_json,
    compile_file_hugr,
    check,
    check_file,
    parse_debug,
    version,
    SlrProgram,
    ZlupProgram,
)

# Import the Rust ZluppyEngine for wrapping
from zluppy._zluppy import ZluppyEngine as _RustZluppyEngine

if TYPE_CHECKING:
    from pecos_rslib import ShotVec

__all__ = [
    # Exception
    "ZluppyError",
    # Source compilation
    "compile_to_slr",
    "compile_to_slr_json",
    "compile_to_hugr",
    "check",
    "parse_debug",
    # File compilation
    "compile_file",
    "compile_file_json",
    "compile_file_hugr",
    "check_file",
    # Utilities
    "version",
    # Classes
    "SlrProgram",
    "ZlupProgram",
    "ZluppyEngine",
]


class ZluppyEngine:
    """Engine for compiling and running Zluppy programs.

    Wraps the Rust ZluppyEngine and adds convenience methods for running
    through PECOS's hugr_engine.

    Example:
        >>> result = (
        ...     zluppy.ZluppyEngine()
        ...     .source(
        ...         '''
        ...     fn main() -> void {
        ...         var q = qalloc(2);
        ...         H(q[0]);
        ...         CX(q[0], q[1]);
        ...     }
        ... '''
        ...     )
        ...     .run(shots=100)
        ... )
        >>> print(result.to_dict())

    Or with explicit steps:
        >>> engine = zluppy.ZluppyEngine().file("bell.zlp")
        >>> hugr_bytes = engine.to_hugr_bytes()
        >>> result = hugr_engine().hugr_bytes(hugr_bytes).to_sim().run(shots=100)
    """

    def __init__(self, strict: bool = False) -> None:
        """Create a new ZluppyEngine.

        Args:
            strict: Enable strict mode (NASA Power of 10 checks).
        """
        self._rust_engine = _RustZluppyEngine(strict)

    def source(self, code: str) -> ZluppyEngine:
        """Compile Zluppy source code.

        Args:
            code: Zluppy source code as a string.

        Returns:
            self for method chaining.

        Raises:
            ZluppyError: If parsing, semantic analysis, or codegen fails.
        """
        self._rust_engine = self._rust_engine.source(code)
        return self

    def file(self, path: str) -> ZluppyEngine:
        """Compile a .zlp file.

        Args:
            path: Path to a .zlp file.

        Returns:
            self for method chaining.

        Raises:
            IOError: If the file cannot be read.
            ZluppyError: If parsing, semantic analysis, or codegen fails.
        """
        self._rust_engine = self._rust_engine.file(path)
        return self

    def to_hugr_bytes(self) -> bytes:
        """Return the compiled HUGR bytes.

        Returns:
            HUGR in binary envelope format, suitable for hugr_engine().

        Raises:
            ValueError: If no source has been compiled.
        """
        return self._rust_engine.to_hugr_bytes()

    def run(self, shots: int = 1) -> ShotVec:
        """Run the compiled program through the simulator.

        Convenience method that calls PECOS's hugr_engine with the compiled HUGR.

        Args:
            shots: Number of shots to run.

        Returns:
            The simulation result from PECOS.

        Raises:
            ValueError: If no source has been compiled.
            ImportError: If pecos is not installed.
        """
        from pecos import hugr_engine

        hugr_bytes = self.to_hugr_bytes()
        return hugr_engine().hugr_bytes(hugr_bytes).to_sim().run(shots=shots)

    def __repr__(self) -> str:
        return repr(self._rust_engine)
