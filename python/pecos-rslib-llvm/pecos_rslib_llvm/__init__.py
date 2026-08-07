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


def _preload_unix_llvm_runtime() -> None:
    """Preload ``libLLVM`` (global visibility) so the extension's ``NEEDED
    libLLVM.so.21.1`` resolves -- a fallback only, run when the extension
    cannot import on its own.

    A static Linux/Windows release wheel has no ``NEEDED`` entry for
    ``libLLVM`` at all, and a macOS release wheel bundles ``libLLVM`` with a
    relocatable load path (delocate) and imports directly -- so this never
    runs for a release wheel. That matters: preloading a *second*, differently-named ``libLLVM``
    alongside a bundled copy leaves two sets of LLVM global state that crash
    the interpreter at shutdown, so this must stay a fallback and never become
    an eager preload. A shared source/editable build, however, links
    ``libLLVM`` by soname and may carry no usable rpath, so without help it
    fails ``ImportError: libLLVM.so.21.1``. PECOS installs LLVM under
    ``~/.pecos/deps/llvm-21.1`` (honouring ``PECOS_LLVM`` /
    ``LLVM_SYS_211_PREFIX``); load it from there. Best-effort: if no candidate
    is found, the caller's retry re-raises the original ImportError.
    """
    if os.name == "nt":
        return

    import ctypes

    # Soname the extension is linked against, plus nearby fallbacks.
    lib_names = ("libLLVM.so.21.1", "libLLVM.so.21", "libLLVM.dylib", "libLLVM-21.dylib")

    lib_dirs: list[Path] = []
    for env_name in ("PECOS_LLVM", "LLVM_SYS_211_PREFIX"):
        if raw_path := os.environ.get(env_name):
            lib_dirs.append(Path(raw_path) / "lib")
    lib_dirs.append(Path.home() / ".pecos" / "deps" / "llvm-21.1" / "lib")

    for lib_dir in lib_dirs:
        for name in lib_names:
            candidate = lib_dir / name
            if not candidate.is_file():
                continue
            try:
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
            return


# Windows has no rpath, so a Windows source/editable build needs the managed
# LLVM ``bin`` on the DLL search path (search-path only -- never loads a second
# copy, so it is safe to run eagerly). A Unix shared source build normally
# resolves ``libLLVM`` via its own rpath; only if that fails do we fall back to
# preloading it below. Static release wheels do not need a runtime ``libLLVM``.
_add_windows_llvm_dll_directories()

try:
    from . import pecos_rslib_llvm as _native  # noqa: E402
except ImportError:
    _preload_unix_llvm_runtime()
    from . import pecos_rslib_llvm as _native  # noqa: E402

from .pecos_rslib_llvm import *  # noqa: E402,F403

__doc__ = _native.__doc__
if hasattr(_native, "__all__"):
    __all__ = _native.__all__
