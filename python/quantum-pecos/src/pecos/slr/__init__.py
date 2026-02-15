# Copyright 2023-2024 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""SLR (Structured Language Representation) for quantum circuits.

SLR is a Python-embedded DSL for defining quantum programs. It provides
a declarative way to specify quantum circuits with support for:

- Quantum registers and qubits
- Standard gates (H, X, Y, Z, CX, etc.) via :mod:`pecos.slr.qeclib`
- Control flow (If, For, Repeat, While)
- Parallel operations
- Barriers and comments

Code Generation
---------------
To generate code from SLR programs, use the :func:`generate` function:

.. code-block:: python

    from pecos.slr import Main, QReg, generate
    from pecos.slr.qeclib import qubit as qb

    prog = Main(
        q := QReg("q", 2),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
    )

    # Generate OpenQASM
    qasm = generate(prog, "qasm")

    # Generate Stim circuit
    stim_code = generate(prog, "stim")

    # Generate Guppy code
    guppy_code = generate(prog, "guppy")

Supported targets: "qasm", "guppy", "stim", "qir", "quantum_circuit"

AST-based Architecture
----------------------
The :func:`generate` function uses an AST (Abstract Syntax Tree) based
approach that provides:

- **Validation**: Type checking, bounds checking, allocation validation
- **Analysis**: T-count, circuit depth, connectivity, parallelism metrics
- **Optimization**: Gate cancellation, rotation merging, and more

For advanced use cases, access the AST directly via :mod:`pecos.slr.ast`:

.. code-block:: python

    from pecos.slr.ast import slr_to_ast
    from pecos.slr.ast.validation import validate
    from pecos.slr.ast.codegen import generate_with_validation

    ast = slr_to_ast(prog)
    result = generate_with_validation(ast, target="qasm", include_analysis=True)
    print(f"T-count: {result.t_count.t_count}")
    print(f"Depth: {result.depth.max_depth}")

See Also
--------
- :mod:`pecos.slr.ast` - AST representation and tools
- :mod:`pecos.slr.ast.codegen` - Code generators
- :mod:`pecos.slr.ast.validation` - Validation passes
- :mod:`pecos.slr.ast.analysis` - Analysis passes
- :mod:`pecos.slr.ast.optimization` - Optimization passes
- :mod:`pecos.slr.qeclib` - Gate library
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.slr import ast, qeclib
from pecos.slr.block import Block
from pecos.slr.cond_block import If, Repeat
from pecos.slr.gen_codes.guppy.qubit_state_validator import (
    QubitStateValidator,
    StateViolation,
    validate_qubit_states,
)
from pecos.slr.loop_block import For, While
from pecos.slr.main import Main
from pecos.slr.main import (
    Main as SLR,
)
from pecos.slr.misc import Barrier, Comment, Parallel, Permute, Return
from pecos.slr.qalloc import QAlloc, QubitRef, SlotState
from pecos.slr.slr_converter import SlrConverter
from pecos.slr.types import Array
from pecos.slr.types import Bit as BitType
from pecos.slr.types import Qubit as QubitType
from pecos.slr.vars import Bit, CReg, LoopVar, QReg, Qubit, Vars

if TYPE_CHECKING:
    from pecos.circuits import QuantumCircuit


def generate(
    program: Main,
    target: str = "qasm",
    *,
    validate: bool = True,
) -> str | QuantumCircuit:
    """Generate code from an SLR program using AST-based code generation.

    This is the recommended way to generate code from SLR programs. It converts
    the SLR program to an AST, optionally validates it, and then generates code
    for the specified target.

    Args:
        program: The SLR Main block to generate code from.
        target: The target format. One of:
            - "qasm": OpenQASM 2.0 (default)
            - "guppy": Guppy Python code
            - "stim": Stim circuit format
            - "qir": QIR/LLVM IR format
            - "quantum_circuit": PECOS QuantumCircuit object
        validate: Whether to validate the AST before generation (default: True).
            Raises ValueError if validation fails.

    Returns:
        Generated code as a string, or a QuantumCircuit object for "quantum_circuit" target.

    Raises:
        ValueError: If validation is enabled and the program is invalid.

    Example:
        >>> from pecos.slr import Main, QReg, generate
        >>> from pecos.slr.qeclib import qubit as qb
        >>>
        >>> prog = Main(
        ...     q := QReg("q", 2),
        ...     qb.H(q[0]),
        ...     qb.CX(q[0], q[1]),
        ... )
        >>> qasm = generate(prog, "qasm")
        >>> print(qasm)
    """
    # Lazy imports to avoid circular dependencies
    from pecos.slr.ast import slr_to_ast  # noqa: PLC0415
    from pecos.slr.ast.codegen import generate as ast_generate  # noqa: PLC0415
    from pecos.slr.ast.validation import validate as ast_validate  # noqa: PLC0415

    # Convert SLR to AST
    program_ast = slr_to_ast(program)

    # Validate if requested
    if validate:
        result = ast_validate(program_ast)
        if not result.valid:
            msg = f"Program validation failed:\n{result}"
            raise ValueError(msg)

    # Generate code
    return ast_generate(program_ast, target)


__all__ = [
    "SLR",
    "Array",
    "Barrier",
    "Bit",
    "BitType",
    "Block",
    "CReg",
    "Comment",
    "For",
    "If",
    "LoopVar",
    "Main",
    "Parallel",
    "Permute",
    # Qubit allocator (new)
    "QAlloc",
    # Legacy register (kept for compatibility)
    "QReg",
    "Qubit",
    "QubitRef",
    # State validation
    "QubitStateValidator",
    "QubitType",
    "Repeat",
    "Return",
    "SlotState",
    "SlrConverter",
    "StateViolation",
    "Vars",
    "While",
    # AST module
    "ast",
    # Code generation (recommended)
    "generate",
    # QEC library
    "qeclib",
    "validate_qubit_states",
]
