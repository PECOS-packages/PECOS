"""Execute LLVM module - HUGR to LLVM compilation interface.

This module provides HUGR to LLVM compilation using Selene's compiler package.
"""

from pathlib import Path


def compile_module_to_string(hugr_bytes: bytes) -> str:
    """Compile HUGR bytes to LLVM IR string.

    Args:
        hugr_bytes: HUGR module serialized as bytes

    Returns:
        LLVM IR as a string

    Raises:
        HugrReadError: If the HUGR envelope is invalid
    """
    from pecos.compilation_pipeline import compile_hugr_to_qis

    return compile_hugr_to_qis(hugr_bytes)


def compile_module_to_file(hugr_bytes: bytes, output_path: str | Path) -> None:
    """Compile HUGR bytes to LLVM IR file.

    Args:
        hugr_bytes: HUGR module serialized as bytes
        output_path: Path where the LLVM IR should be written
    """
    llvm_ir = compile_module_to_string(hugr_bytes)
    with Path(output_path).open("w") as f:
        f.write(llvm_ir)


def compile_hugr_file_to_string(hugr_path: str | Path) -> str:
    """Compile HUGR file to LLVM IR string.

    Args:
        hugr_path: Path to HUGR file

    Returns:
        LLVM IR as a string
    """
    with Path(hugr_path).open("rb") as f:
        hugr_bytes = f.read()
    return compile_module_to_string(hugr_bytes)


def compile_hugr_file_to_file(
    hugr_path: str | Path,
    output_path: str | Path,
) -> None:
    """Compile HUGR file to LLVM IR file.

    Args:
        hugr_path: Path to HUGR file
        output_path: Path where the LLVM IR should be written
    """
    llvm_ir = compile_hugr_file_to_string(hugr_path)
    with Path(output_path).open("w") as f:
        f.write(llvm_ir)


def is_available() -> bool:
    """Check if execute_llvm functionality is available.

    Returns:
        True if the Selene HUGR->LLVM compiler is available, False otherwise
    """
    import importlib.util

    return importlib.util.find_spec("selene_hugr_qis_compiler") is not None


# Additional metadata
__all__ = [
    "compile_hugr_file_to_file",
    "compile_hugr_file_to_string",
    "compile_module_to_file",
    "compile_module_to_string",
    "is_available",
]
