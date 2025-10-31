"""Python wrapper for sim() that handles Guppy programs.

This module provides a Python-side sim() function that acts as a thin wrapper:
1. Detects Guppy programs and compiles them to HUGR format
2. Passes all programs to the Rust sim() which handles HUGR->QIS conversion internally

The HUGR to QIS conversion now happens in Rust, making the Python side a truly thin wrapper.
"""

import logging
from typing import TYPE_CHECKING, Protocol, Union

if TYPE_CHECKING:
    from pecos_rslib.programs import HugrProgram, QisProgram, QasmProgram

logger = logging.getLogger(__name__)


class GuppyFunction(Protocol):
    """Protocol for Guppy-decorated functions."""

    def compile(self) -> dict: ...


ProgramType = Union[
    GuppyFunction, "QasmProgram", "QisProgram", "HugrProgram", bytes, str
]


def sim(program: ProgramType) -> object:
    """Thin Python wrapper for sim() that handles Guppy programs.

    This wrapper:
    1. Detects Guppy functions and compiles them to HUGR format
    2. Passes all programs (including HugrProgram) to the Rust sim()
    3. Rust handles HUGR->QIS conversion internally

    Args:
        program: The program to simulate (Guppy function, HugrProgram, QasmProgram, etc.)

    Returns:
        SimBuilder instance
    """
    from . import _pecos_rslib

    # Check if this is a Guppy function
    def is_guppy_function(obj: object) -> bool:
        """Check if an object is a Guppy-decorated function."""
        return (
            hasattr(obj, "_guppy_compiled")
            or hasattr(obj, "compile")
            or str(type(obj)).find("GuppyFunctionDefinition") != -1
        )

    # Check if this is a HugrProgram - pass it directly to Rust
    if type(program).__name__ == "HugrProgram":
        logger.info(
            "Detected HugrProgram, passing directly to Rust for HUGR->QIS conversion"
        )
        # Keep program as HugrProgram - Rust will handle the conversion internally

    elif is_guppy_function(program):
        logger.info("Detected Guppy function, compiling to HUGR format")

        # Compile Guppy → HUGR
        hugr_package = program.compile()
        logger.info("Compiled Guppy function to HUGR package")

        # Convert HUGR package to binary format for Rust
        # to_bytes() is the standard binary encoding (uses envelope with format 0x02)
        hugr_bytes = hugr_package.to_bytes()

        # Create HugrProgram - Rust will handle HUGR->QIS conversion
        hugr_program = _pecos_rslib.HugrProgram.from_bytes(hugr_bytes)
        logger.info(
            "Created HugrProgram, passing to Rust sim() for HUGR->QIS conversion"
        )

        program = hugr_program

    # Pass to Rust sim() which handles all fallback logic
    logger.info("Using Rust sim() for program type: %s", type(program))
    result = _pecos_rslib.sim(program)

    # Force comprehensive cleanup after each simulation to prevent state pollution between tests
    try:
        _pecos_rslib.clear_jit_cache()
    except Exception as e:
        logger.debug("Cache clearing failed (this is non-critical): %s", e)

    # Force garbage collection to clean up any lingering engine resources
    import gc

    gc.collect()

    return result
