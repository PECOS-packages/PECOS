"""Lower HUGR with Selene and normalize PECOS runtime helper symbols.

Install ``pecos-rslib[hugr]`` to enable compilation. The base bindings wheel
remains usable without the optional ``selene-hugr-qis-compiler`` package.
Importing this module loads neither that compiler nor quantum-pecos; the
compiler is imported lazily when compile_hugr_to_qis is called.
"""

from pecos_rslib.qis import PECOS_HELPER_ABIS, normalize_pecos_helper_symbols

__all__ = ["PECOS_HELPER_ABIS", "compile_hugr_to_qis", "normalize_pecos_helper_symbols"]


def compile_hugr_to_qis(
    hugr_bytes: bytes,
    *,
    platform: str = "helios",
    opt_level: int = 2,
    emit_debug: bool = False,
) -> str:
    """Lower a HUGR envelope with Selene and normalize PECOS runtime helpers.

    Args:
        hugr_bytes: HUGR package in the text or binary envelope format.
        platform: Target QSystem platform ("helios" or "sol").
        opt_level: LLVM optimization level.
        emit_debug: Whether to emit debug information.

    Returns:
        LLVM IR with PECOS's public helper symbols.

    Raises:
        ImportError: If selene-hugr-qis-compiler is not installed.
        ValueError: If compiler options are invalid or helper validation fails.

    Compiler exceptions, including HugrReadError and TypeError, propagate unchanged.
    """
    if type(opt_level) is not int or opt_level not in (0, 1, 2, 3):
        msg = f"opt_level must be one of 0, 1, 2, 3; got {opt_level!r}"
        raise ValueError(msg)
    if platform not in ("helios", "sol"):
        msg = f"platform must be 'helios' or 'sol'; got {platform!r}"
        raise ValueError(msg)
    try:
        from selene_hugr_qis_compiler import compile_to_llvm_ir
    except ImportError as error:
        msg = "HUGR -> QIS lowering requires the selene-hugr-qis-compiler package"
        raise ImportError(msg) from error

    return normalize_pecos_helper_symbols(
        compile_to_llvm_ir(hugr_bytes, platform=platform, opt_level=opt_level, emit_debug=emit_debug),
    )
