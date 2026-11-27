"""Public WASM compilation and utility plugin for standalone Selene."""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING

from selene_core import Utility

from pecos_selene_wasm.lowering import compile_guppy, compile_wasm
from pecos_selene_wasm.options import WasmOptions

if TYPE_CHECKING:
    from guppylang.defs import GuppyFunctionDefinition
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
    program: bytes | Package | GuppyFunctionDefinition,
    options: WasmOptions,
    *,
    platform: str = "helios",
    opt_level: int = 2,
    emit_debug: bool = False,
    **kwargs: object,
) -> SeleneInstance:
    """Build Selene with explicit WASM bytes and safe compilation of Guppy source.

    Precompiled input should come directly from ``compile_guppy`` without
    additional optimization.
    """
    if not isinstance(options, WasmOptions):
        msg = f"options must be a WasmOptions; got {type(options).__name__}"
        raise TypeError(msg)
    names = {field.name for field in fields(WasmOptions)}
    misplaced = kwargs.keys() & (names | {f"wasm_{name}" for name in names})
    if misplaced:
        corrections = {name if name in names else name.removeprefix("wasm_") for name in misplaced}
        arguments = ", ".join(f"{name}=..." for name in sorted(corrections))
        msg = f"Pass {arguments} in WasmOptions(...), not as build keyword arguments"
        raise TypeError(msg)
    # Standalone Selene is optional when this package supplies PECOS lowering.
    from selene_sim import build as selene_build  # noqa: PLC0415

    utilities = [*kwargs.pop("utilities", []), WasmPlugin()]
    llvm = compile_wasm(
        program,
        options,
        platform=platform,
        opt_level=opt_level,
        emit_debug=emit_debug,
    )
    if options.async_workers:
        runtime = ctypes.CDLL(str(utilities[-1].library_file))
        if not hasattr(runtime, "pecos_wasm_v2_call"):
            msg = "Asynchronous WASM calls require an updated native runtime; rebuild or upgrade pecos-selene-wasm"
            raise RuntimeError(msg)
    if options.state_scope == "worker":
        runtime = ctypes.CDLL(str(utilities[-1].library_file))
        if not all(
            hasattr(runtime, name)
            for name in (
                "pecos_wasm_v2_begin_worker_shot",
                "pecos_wasm_v2_finish_worker_shot",
            )
        ):
            msg = "Worker-scoped WASM state requires an updated native runtime; rebuild or upgrade pecos-selene-wasm"
            raise RuntimeError(msg)
    if options.int_width == "auto":
        runtime = ctypes.CDLL(str(utilities[-1].library_file))
        for symbol in ("pecos_wasm_v3_lookup", "pecos_wasm_v3_lookup_by_id"):
            if f"call i64 @{symbol}(" in llvm and not hasattr(runtime, symbol):
                msg = (
                    "Automatic WASM integer widths require an updated native runtime; "
                    "rebuild or upgrade pecos-selene-wasm"
                )
                raise RuntimeError(msg)
    if not options.auto_init:
        runtime = ctypes.CDLL(str(utilities[-1].library_file))
        if not hasattr(runtime, "pecos_wasm_v3_register"):
            msg = (
                "Disabling WASM auto-init requires an updated native WASM runtime; rebuild or upgrade pecos-selene-wasm"
            )
            raise RuntimeError(msg)
    elif options.timeout_seconds != 1.0:
        runtime = ctypes.CDLL(str(utilities[-1].library_file))
        if not hasattr(runtime, "pecos_wasm_v2_register"):
            msg = "Custom WASM timeouts require an updated native WASM runtime; rebuild or upgrade pecos-selene-wasm"
            raise RuntimeError(msg)
    return selene_build(
        llvm,
        platform=platform,
        opt_level=opt_level,
        emit_debug=emit_debug,
        utilities=utilities,
        **kwargs,
    )


__all__ = ["WasmOptions", "WasmPlugin", "build", "compile_guppy", "compile_wasm"]
