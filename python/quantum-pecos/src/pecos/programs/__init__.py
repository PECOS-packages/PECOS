# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""PECOS Program Types and Wrappers.

This module provides program wrapper classes for quantum programs that can be
simulated using PECOS's sim() API. The wrapper classes provide a clean, pythonic
interface for creating programs from strings, bytes, or files.

Wrapper classes (for use with sim()):
    - Guppy: Guppy-decorated functions
    - Hugr: HUGR binary format
    - Qasm: OpenQASM 2.0/3.0 programs
    - Qis: QIS/LLVM IR programs
    - PhirJson: PHIR JSON format programs
    - Wasm: WebAssembly binary programs
    - Wat: WebAssembly text format programs

Low-level program types (from pecos_rslib):
    - Hugr, Qasm, Qis, PhirJson, Wasm, Wat

Example:
    >>> from pecos import sim, Qasm, Guppy
    >>>
    >>> # QASM program
    >>> results = sim(
    ...     Qasm(
    ...         '''
    ...     OPENQASM 2.0;
    ...     qreg q[2];
    ...     creg c[2];
    ...     h q[0];
    ...     cx q[0], q[1];
    ...     measure q -> c;
    ... '''
    ...     )
    ... ).run(1000)
    >>>
    >>> # Guppy function
    >>> from guppylang import guppy
    >>> from guppylang.std.quantum import qubit, h, measure
    >>>
    >>> @guppy
    ... def my_circuit():
    ...     q = qubit()
    ...     h(q)
    ...     return measure(q)
    ...
    >>>
    >>> results = sim(Guppy(my_circuit)).run(1000)
"""

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import pecos_rslib

if TYPE_CHECKING:
    from pecos.typing import (
        CompiledHugr,
        CompiledPhirJson,
        CompiledQasm,
        CompiledQis,
        CompiledWasm,
        CompiledWat,
    )


# =============================================================================
# Protocol definitions for type checking
# =============================================================================


class GuppyFunction(Protocol):
    """Protocol for Guppy-decorated functions."""

    def compile(self) -> "HugrPackage":
        """Compile the Guppy function to HUGR format."""
        ...


class HugrPackage(Protocol):
    """Protocol for HUGR package objects."""

    def to_bytes(self) -> bytes:
        """Serialize the HUGR package to bytes."""
        ...


# =============================================================================
# Program wrapper classes
# =============================================================================


class Guppy:
    """Wrapper for Guppy functions.

    Converts Guppy-decorated functions to Hugr format for simulation.
    The conversion is cached, so multiple calls will not recompile.

    Example:
        >>> from guppylang import guppy
        >>> from guppylang.std.quantum import qubit, h, measure
        >>>
        >>> @guppy
        ... def bell_state():
        ...     q0, q1 = qubit(), qubit()
        ...     h(q0)
        ...     cx(q0, q1)
        ...     return measure(q0), measure(q1)
        ...
        >>>
        >>> from pecos import sim, Guppy
        >>> results = sim(Guppy(bell_state)).run(1000)
    """

    def __init__(self, func: GuppyFunction) -> None:
        """Initialize with a Guppy-decorated function."""
        self._func = func
        self._program = None

    def _to_program(self) -> "CompiledHugr":
        """Convert to the underlying Rust program type."""
        if self._program is None:
            hugr_package = self._func.compile()
            hugr_bytes = hugr_package.to_bytes()
            self._program = pecos_rslib.Hugr.from_bytes(hugr_bytes)
        return self._program


class Hugr:
    """Wrapper for HUGR (Higher-order Unified Graph Representation) programs.

    Accepts HUGR data as bytes or a file path.

    Example:
        >>> from pecos import sim, Hugr
        >>>
        >>> # From bytes
        >>> results = sim(Hugr(hugr_bytes)).run(1000)
        >>>
        >>> # From file
        >>> results = sim(Hugr.from_file("program.hugr")).run(1000)
    """

    def __init__(self, data: bytes) -> None:
        """Initialize with HUGR bytes."""
        self._data = data
        self._program = None

    @classmethod
    def from_file(cls, path: str | Path) -> "Hugr":
        """Create from a HUGR file."""
        with Path(path).open("rb") as f:
            return cls(f.read())

    @classmethod
    def from_bytes(cls, data: bytes) -> "Hugr":
        """Create from HUGR bytes.

        This is an alias for the constructor, provided for API consistency
        with the Rust Hugr type.
        """
        return cls(data)

    def _to_program(self) -> "CompiledHugr":
        """Convert to the underlying Rust program type."""
        if self._program is None:
            self._program = pecos_rslib.Hugr.from_bytes(self._data)
        return self._program


class Qasm:
    """Wrapper for OpenQASM programs.

    Accepts QASM code as a string or a file path.

    Example:
        >>> from pecos import sim, Qasm
        >>>
        >>> # From string
        >>> results = sim(
        ...     Qasm(
        ...         '''
        ...     OPENQASM 2.0;
        ...     qreg q[2];
        ...     creg c[2];
        ...     h q[0];
        ...     cx q[0], q[1];
        ...     measure q -> c;
        ... '''
        ...     )
        ... ).run(1000)
        >>>
        >>> # From file
        >>> results = sim(Qasm.from_file("program.qasm")).run(1000)
    """

    def __init__(self, code: str) -> None:
        """Initialize with QASM code string."""
        self._code = code
        self._program = None

    @classmethod
    def from_file(cls, path: str | Path) -> "Qasm":
        """Create from a QASM file."""
        with Path(path).open() as f:
            return cls(f.read())

    @classmethod
    def from_string(cls, code: str) -> "Qasm":
        """Create from a QASM string.

        This is an alias for the constructor, provided for API consistency
        with the Rust Qasm type.
        """
        return cls(code)

    def _to_program(self) -> "CompiledQasm":
        """Convert to the underlying Rust program type."""
        if self._program is None:
            self._program = pecos_rslib.Qasm.from_string(self._code)
        return self._program


class Qis:
    """Wrapper for QIS (LLVM-based) programs.

    Accepts QIS/LLVM IR code as a string or a file path.

    Example:
        >>> from pecos import sim, Qis
        >>>
        >>> # From string
        >>> results = sim(Qis(llvm_ir_code)).run(1000)
        >>>
        >>> # From file
        >>> results = sim(Qis.from_file("program.ll")).run(1000)
    """

    def __init__(self, code: str) -> None:
        """Initialize with QIS/LLVM IR code string."""
        self._code = code
        self._program = None

    @classmethod
    def from_file(cls, path: str | Path) -> "Qis":
        """Create from a QIS/LLVM file."""
        with Path(path).open() as f:
            return cls(f.read())

    @classmethod
    def from_string(cls, code: str) -> "Qis":
        """Create from a QIS/LLVM IR string.

        This is an alias for the constructor, provided for API consistency
        with the Rust Qis type.
        """
        return cls(code)

    def _to_program(self) -> "CompiledQis":
        """Convert to the underlying Rust program type."""
        if self._program is None:
            self._program = pecos_rslib.Qis.from_string(self._code)
        return self._program


class PhirJson:
    """Wrapper for PHIR JSON format programs.

    Accepts PHIR JSON as a string or a file path.

    Example:
        >>> from pecos import sim, PhirJson
        >>>
        >>> # From string
        >>> results = sim(PhirJson(phir_json)).run(1000)
        >>>
        >>> # From file
        >>> results = sim(PhirJson.from_file("program.json")).run(1000)
        >>>
        >>> # Alternative constructors
        >>> results = sim(PhirJson.from_json(json_str)).run(1000)
        >>> results = sim(PhirJson.from_string(json_str)).run(1000)
    """

    def __init__(self, json_str: str) -> None:
        """Initialize with PHIR JSON string."""
        self._json = json_str
        self._program = None

    @classmethod
    def from_file(cls, path: str | Path) -> "PhirJson":
        """Create from a PHIR JSON file."""
        with Path(path).open() as f:
            return cls(f.read())

    @classmethod
    def from_string(cls, json_str: str) -> "PhirJson":
        """Create from a PHIR JSON string.

        This is an alias for the constructor, provided for API consistency
        with the Rust PhirJson type.
        """
        return cls(json_str)

    @classmethod
    def from_json(cls, json_str: str) -> "PhirJson":
        """Create from a PHIR JSON string.

        This is an alias for the constructor, provided for API consistency
        with the Rust PhirJson type.
        """
        return cls(json_str)

    def _to_program(self) -> "CompiledPhirJson":
        """Convert to the underlying Rust program type."""
        if self._program is None:
            self._program = pecos_rslib.PhirJson.from_string(self._json)
        return self._program


class Wasm:
    """Wrapper for WebAssembly programs.

    Accepts WASM binary data or a file path.

    Example:
        >>> from pecos import sim, Wasm
        >>>
        >>> # From bytes
        >>> results = sim(Wasm(wasm_bytes)).run(1000)
        >>>
        >>> # From file
        >>> results = sim(Wasm.from_file("program.wasm")).run(1000)
    """

    def __init__(self, data: bytes) -> None:
        """Initialize with WASM binary data."""
        self._data = data
        self._program = None

    @classmethod
    def from_file(cls, path: str | Path) -> "Wasm":
        """Create from a WASM file."""
        with Path(path).open("rb") as f:
            return cls(f.read())

    @classmethod
    def from_bytes(cls, data: bytes) -> "Wasm":
        """Create from WASM binary bytes.

        This is an alias for the constructor, provided for API consistency
        with the Rust Wasm type.
        """
        return cls(data)

    def _to_program(self) -> "CompiledWasm":
        """Convert to the underlying Rust program type."""
        if self._program is None:
            self._program = pecos_rslib.Wasm.from_bytes(self._data)
        return self._program


class Wat:
    """Wrapper for WebAssembly text format programs.

    Accepts WAT code as a string or a file path.

    Example:
        >>> from pecos import sim, Wat
        >>>
        >>> # From string
        >>> results = sim(Wat(wat_code)).run(1000)
        >>>
        >>> # From file
        >>> results = sim(Wat.from_file("program.wat")).run(1000)
    """

    def __init__(self, code: str) -> None:
        """Initialize with WAT code string."""
        self._code = code
        self._program = None

    @classmethod
    def from_file(cls, path: str | Path) -> "Wat":
        """Create from a WAT file."""
        with Path(path).open() as f:
            return cls(f.read())

    @classmethod
    def from_string(cls, code: str) -> "Wat":
        """Create from a WAT code string.

        This is an alias for the constructor, provided for API consistency
        with the Rust Wat type.
        """
        return cls(code)

    def _to_program(self) -> "CompiledWat":
        """Convert to the underlying Rust program type."""
        if self._program is None:
            self._program = pecos_rslib.Wat.from_string(self._code)
        return self._program


# =============================================================================
# Program type unions
# =============================================================================

#: Type alias for Python program wrapper classes (primary user-facing types)
ProgramWrapper = Guppy | Hugr | Qasm | Qis | PhirJson | Wasm | Wat


__all__ = [
    # Program wrapper classes (primary API for sim())
    "Guppy",
    "GuppyFunction",
    "Hugr",
    "HugrPackage",
    "PhirJson",
    "ProgramWrapper",
    "Qasm",
    "Qis",
    "Wasm",
    "Wat",
]


def __dir__() -> list[str]:
    """Return a clean list of public names for tab completion and dir()."""
    return __all__
