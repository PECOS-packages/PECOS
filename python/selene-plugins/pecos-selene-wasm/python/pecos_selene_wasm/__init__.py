"""Public WASM compilation and utility plugin for standalone Selene."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from selene_core import Utility

from pecos_selene_wasm.lowering import compile_wasm

if TYPE_CHECKING:
    from collections.abc import Mapping

    from hugr.package import Package
    from selene_sim import SeleneInstance


class WasmPlugin(Utility):
    """Link the public scalar WASM ABI into a standalone Selene executable."""

    @property
    def library_file(self) -> Path:
        """Return the bundled library or an explicit development-build override."""
        override = os.environ.get("PECOS_WASM_LIBRARY")
        if override:
            path = Path(override).resolve()
        else:
            filename = {"darwin": "libpecos_selene_wasm.dylib", "win32": "pecos_selene_wasm.dll"}.get(
                sys.platform,
                "libpecos_selene_wasm.so",
            )
            path = Path(__file__).parent / "_dist" / "lib" / filename
        if not path.is_file():
            msg = f"Public WASM runtime library not found: {path}"
            raise FileNotFoundError(msg)
        return path

    @property
    def library_search_dirs(self) -> list[Path]:
        """Locate the native library when Selene links its executable."""
        return [self.library_file.parent]


def build(
    program: bytes | Package,
    modules: Mapping[str, bytes],
    *,
    platform: str = "helios",
    opt_level: int = 2,
    emit_debug: bool = False,
    **kwargs: object,
) -> SeleneInstance:
    """Build a standalone Selene runner with explicit WASM module bytes."""
    # Standalone Selene is optional when this package supplies PECOS lowering.
    from selene_sim import build as selene_build  # noqa: PLC0415

    utilities = [*kwargs.pop("utilities", []), WasmPlugin()]
    llvm = compile_wasm(program, modules, platform=platform, opt_level=opt_level, emit_debug=emit_debug)
    return selene_build(
        llvm,
        platform=platform,
        opt_level=opt_level,
        emit_debug=emit_debug,
        utilities=utilities,
        **kwargs,
    )


__all__ = ["WasmPlugin", "build", "compile_wasm"]
