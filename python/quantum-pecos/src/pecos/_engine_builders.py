# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Engine builder wrappers for PECOS.

This module provides Python wrappers around the Rust engine builders to accept
Python program wrappers from pecos.programs.
"""

from __future__ import annotations

from importlib import import_module
from os import PathLike, fspath
from pathlib import Path
from typing import TYPE_CHECKING

import pecos_rslib

if TYPE_CHECKING:
    import pecos_rslib as prs
    from typing_extensions import Self

    from pecos.programs import Hugr, PhirJson, Qasm, Qis
    from pecos.typing import CompiledPhirJson, CompiledQasm, CompiledQis


class QasmEngineBuilder:
    """Python wrapper for QASM engine builder.

    This wrapper accepts Python Qasm objects from pecos.programs.

    Example:
        >>> from pecos import qasm_engine, Qasm
        >>> results = qasm_engine().program(Qasm("OPENQASM 2.0; qreg q[2]; ...")).to_sim().run(1000)
    """

    def __init__(self) -> None:
        self._builder = pecos_rslib.qasm_engine()

    def program(self, program: Qasm | CompiledQasm) -> Self:
        """Set the program for this engine.

        Args:
            program: A Qasm object (from pecos.programs or pecos_rslib.programs)

        Returns:
            Self for method chaining.
        """
        # If it's a Python wrapper, extract the underlying Rust type
        if hasattr(program, "_to_program"):
            self._builder = self._builder.program(program._to_program())
        else:
            # It's already a Rust type
            self._builder = self._builder.program(program)
        return self

    def wasm(self, wasm_path: str) -> Self:
        """Set the WebAssembly module for foreign function calls.

        Returns:
            Self for method chaining.
        """
        self._builder = self._builder.wasm(wasm_path)
        return self

    def to_sim(self) -> prs.SimBuilder:
        """Convert to simulation builder.

        Returns:
            A SimBuilder that can be configured and run.
        """
        return self._builder.to_sim()


class PhirJsonEngineBuilder:
    """Python wrapper for PHIR JSON engine builder.

    This wrapper accepts Python PhirJson objects from pecos.programs.

    Example:
        >>> from pecos import phir_json_engine, PhirJson
        >>> results = phir_json_engine().program(PhirJson('{"format": "PHIR/JSON", ...}')).to_sim().run(1000)
    """

    def __init__(self) -> None:
        self._builder = pecos_rslib.phir_json_engine()

    def program(self, program: PhirJson | CompiledPhirJson) -> Self:
        """Set the program for this engine.

        Args:
            program: A PhirJson object (from pecos.programs or pecos_rslib.programs)

        Returns:
            Self for method chaining.
        """
        # If it's a Python wrapper, extract the underlying Rust type
        if hasattr(program, "_to_program"):
            self._builder = self._builder.program(program._to_program())
        else:
            # It's already a Rust type
            self._builder = self._builder.program(program)
        return self

    def wasm(self, wasm_path: str) -> Self:
        """Set the WebAssembly module for foreign function calls.

        Returns:
            Self for method chaining.
        """
        self._builder = self._builder.wasm(wasm_path)
        return self

    def to_sim(self) -> prs.SimBuilder:
        """Convert to simulation builder.

        Returns:
            A SimBuilder that can be configured and run.
        """
        return self._builder.to_sim()


class QisEngineBuilder:
    """Python wrapper for QIS engine builder.

    This wrapper accepts Python Qis or Hugr objects from pecos.programs.

    Example:
        >>> from pecos import qis_engine, Qis
        >>> results = qis_engine().program(Qis(llvm_ir_code)).to_sim().run(1000)
    """

    def __init__(self) -> None:
        self._builder = pecos_rslib.qis_engine()

    def program(self, program: Qis | Hugr | CompiledQis) -> Self:
        """Set the program for this engine.

        Args:
            program: A Qis or Hugr object (from pecos.programs or pecos_rslib.programs)

        Returns:
            Self for method chaining.
        """
        # If it's a Python wrapper, extract the underlying Rust type
        if hasattr(program, "_to_program"):
            self._builder = self._builder.program(program._to_program())
        else:
            # It's already a Rust type
            self._builder = self._builder.program(program)
        return self

    def selene_runtime(self, runtime: object | None = None) -> Self:
        """Use a Selene runtime.

        Args:
            runtime: Optional runtime selector. ``None`` selects the default
                ``selene_simple_runtime``. A string without path separators is
                treated as a built Selene runtime library name. A path-like
                value, or any Selene runtime plugin object exposing
                ``library_file``, ``get_init_args()``, and optional
                ``library_search_dirs``, is passed through generically.

        Returns:
            Self for method chaining.
        """
        self._builder = _configure_selene_runtime(self._builder, runtime)
        return self

    def interface(self, builder: object) -> Self:
        """Set the interface builder.

        Returns:
            Self for method chaining.
        """
        self._builder = self._builder.interface(builder)
        return self

    def to_sim(self) -> prs.SimBuilder:
        """Convert to simulation builder.

        Returns:
            A SimBuilder that can be configured and run.
        """
        return self._builder.to_sim()


def qasm_engine() -> QasmEngineBuilder:
    """Create a QASM engine builder.

    Returns:
        QasmEngineBuilder: A builder for QASM simulations.

    Example:
        >>> from pecos import qasm_engine, Qasm
        >>> results = qasm_engine().program(Qasm("OPENQASM 2.0; qreg q[2]; ...")).to_sim().run(1000)
    """
    return QasmEngineBuilder()


def phir_json_engine() -> PhirJsonEngineBuilder:
    """Create a PHIR JSON engine builder.

    Returns:
        PhirJsonEngineBuilder: A builder for PHIR JSON simulations.

    Example:
        >>> from pecos import phir_json_engine, PhirJson
        >>> results = phir_json_engine().program(PhirJson('{"format": "PHIR/JSON", ...}')).to_sim().run(1000)
    """
    return PhirJsonEngineBuilder()


def qis_engine() -> QisEngineBuilder:
    """Create a QIS engine builder.

    Returns:
        QisEngineBuilder: A builder for QIS/HUGR simulations.

    Example:
        >>> from pecos import qis_engine, Qis
        >>> results = qis_engine().program(Qis(llvm_ir_code)).to_sim().run(1000)
    """
    return QisEngineBuilder()


def _looks_like_library_path(value: str) -> bool:
    path = Path(value)
    return path.exists() or "/" in value or "\\" in value or path.suffix in {".so", ".dylib", ".dll"}


def _plugin_object_from_module(module: object) -> object:
    for exported_name in getattr(module, "__all__", ()):
        exported = getattr(module, exported_name, None)
        if isinstance(exported, type):
            return exported()
    return module


def _configure_selene_runtime(builder: object, runtime: object | None) -> object:
    if runtime is None:
        # Issue #365: freshly built Cargo artifacts win for dev iteration; the
        # installed plugin package is the stable cwd-independent fallback.
        try:
            return builder.selene_runtime()
        except RuntimeError as original_error:
            try:
                plugin_module = import_module("selene_simple_runtime_plugin")
            except ImportError:
                # The discovery error is the actionable one; a missing plugin
                # package must not mask it.
                raise original_error from None
            return _configure_selene_runtime(builder, _plugin_object_from_module(plugin_module))

    if isinstance(runtime, str):
        if _looks_like_library_path(runtime):
            return builder.selene_runtime_plugin(runtime)
        try:
            return builder.selene_runtime(runtime)
        except RuntimeError as original_error:
            try:
                plugin_module = import_module(f"{runtime}_plugin")
            except ImportError:
                raise original_error from None
            return _configure_selene_runtime(builder, _plugin_object_from_module(plugin_module))

    if isinstance(runtime, PathLike):
        return builder.selene_runtime_plugin(fspath(runtime))

    library_file = getattr(runtime, "library_file", None)
    if library_file is None:
        msg = (
            "Selene runtime must be None, a built runtime name, a shared-library path, "
            "or a Selene runtime plugin object with a 'library_file' property."
        )
        raise TypeError(msg)

    get_init_args = getattr(runtime, "get_init_args", None)
    init_args = list(get_init_args()) if callable(get_init_args) else []
    library_search_dirs = [fspath(path) for path in getattr(runtime, "library_search_dirs", [])]
    return builder.selene_runtime_plugin(
        fspath(library_file),
        [str(arg) for arg in init_args],
        library_search_dirs,
    )


def selene_engine(runtime: object | None = None) -> QisEngineBuilder:
    """Create a Selene-backed QIS engine builder.

    Args:
        runtime: Optional runtime selector. ``None`` selects the default
            ``selene_simple_runtime``. A built runtime name, shared-library
            path, or generic Selene runtime plugin object may also be supplied.

    Returns:
        QisEngineBuilder: A builder for Selene-backed QIS/HUGR simulations.
    """
    return QisEngineBuilder().selene_runtime(runtime).interface(pecos_rslib.qis_helios_interface())


__all__ = [
    # Builder classes
    "PhirJsonEngineBuilder",
    "QasmEngineBuilder",
    "QisEngineBuilder",
    # Factory functions
    "phir_json_engine",
    "qasm_engine",
    "qis_engine",
    "selene_engine",
]
