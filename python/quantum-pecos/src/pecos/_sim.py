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

"""Simulation entry point for PECOS.

This module provides the primary entry point for running quantum simulations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pecos_rslib

if TYPE_CHECKING:
    import pecos_rslib as prs

    from pecos.programs import GuppyFunction, ProgramWrapper
    from pecos.typing import CompiledProgram


def sim(
    program: ProgramWrapper | CompiledProgram | GuppyFunction,
) -> prs.SimBuilder:
    """Create a simulation builder for a quantum program.

    This is the primary entry point for running quantum simulations in PECOS.

    Args:
        program: A wrapped quantum program (Guppy, Qasm, Qis, Hugr, PhirJson, Wasm, or Wat),
                 a raw Rust program type from pecos_rslib,
                 or a Guppy-decorated function (which will be auto-wrapped).

    Returns:
        A SimBuilder that can be configured and run.

    Example:
        >>> from pecos import sim, Qasm
        >>> results = sim(Qasm("OPENQASM 2.0; qreg q[2]; ...")).run(1000)

        >>> # Guppy functions are auto-wrapped
        >>> @guppy
        ... def my_circuit():
        ...     q = qubit()
        ...     return measure(q)
        ...
        >>> results = sim(my_circuit).run(100)
    """
    from pecos.programs import (  # noqa: PLC0415
        Guppy,
    )

    # Auto-wrap Guppy-decorated functions (they have a 'compile' method)
    if hasattr(program, "compile") and not hasattr(program, "_to_program"):
        program = Guppy(program)

    # If it's a Python wrapper, extract the underlying Rust type
    if hasattr(program, "_to_program"):
        return pecos_rslib.sim(program._to_program())  # noqa: SLF001
    # It's already a Rust type (from pecos_rslib), pass directly
    return pecos_rslib.sim(program)


def get_guppy_backends() -> dict:
    """Get available Guppy backends.

    Returns a dict with:
        - guppy_available: True if guppylang is installed
        - rust_backend: Always True (HUGR support is built into pecos-rslib)
    """
    result = {"guppy_available": False, "rust_backend": True}
    try:
        import guppylang  # noqa: F401, PLC0415

        result["guppy_available"] = True
    except ImportError:
        pass
    return result


__all__ = [
    "get_guppy_backends",
    "sim",
]
