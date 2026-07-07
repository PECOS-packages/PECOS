"""Python package wrapper for the native ``pecos_rslib_llvm`` extension."""

from __future__ import annotations

import os
from pathlib import Path

_DLL_DIRECTORY_HANDLES = []


def _add_dll_directory(path: Path) -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory") or not path.is_dir():
        return

    try:
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(path)))
    except OSError:
        pass


def _add_windows_llvm_dll_directories() -> None:
    if os.name != "nt":
        return

    seen: set[str] = set()
    candidates: list[Path] = []

    for env_name in ("PECOS_LLVM", "LLVM_SYS_211_PREFIX"):
        if raw_path := os.environ.get(env_name):
            prefix = Path(raw_path)
            candidates.extend((prefix / "bin", prefix))

    home = Path.home()
    candidates.extend(
        (
            home / ".pecos" / "deps" / "llvm-21.1" / "Library" / "bin",
            home / ".pecos" / "deps" / "llvm-21.1" / "bin",
        )
    )

    for candidate in candidates:
        key = os.path.normcase(os.path.normpath(str(candidate)))
        if key in seen:
            continue
        seen.add(key)
        _add_dll_directory(candidate)


_add_windows_llvm_dll_directories()

from . import pecos_rslib_llvm as _native  # noqa: E402
from .pecos_rslib_llvm import *  # noqa: E402,F403

__doc__ = _native.__doc__
if hasattr(_native, "__all__"):
    __all__ = _native.__all__
