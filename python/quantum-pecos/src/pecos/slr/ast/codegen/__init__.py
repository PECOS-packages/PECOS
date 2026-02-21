# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Code generation from AST.

This package provides code generators that transform AST into target languages.
Supported targets include:
- Guppy (Python-based quantum language for HUGR)
- QASM (Extended OpenQASM 2.0)
- Stim (stabilizer circuit simulator)
- QuantumCircuit (PECOS internal format)
- QIR (Quantum Intermediate Representation via LLVM)

Example with validation:
    >>> from pecos.slr.ast.codegen import generate_with_validation
    >>> result = generate_with_validation(ast, target="qasm")
    >>> if result.valid:
    ...     print(result.code)
    ...
"""

from pecos.slr.ast.codegen.base import CodegenOptions, CodegenResult
from pecos.slr.ast.codegen.guppy import AstToGuppy, ast_to_guppy
from pecos.slr.ast.codegen.qasm import AstToQasm, ast_to_qasm
from pecos.slr.ast.codegen.qir import AstToQir, ast_to_qir
from pecos.slr.ast.codegen.quantum_circuit import (
    AstToQuantumCircuit,
    ast_to_quantum_circuit,
    ast_to_quantum_circuit_str,
)
from pecos.slr.ast.codegen.stim import AstToStim, ast_to_stim, ast_to_stim_str
from pecos.slr.ast.nodes import Program

# Mapping of target names to generator functions
_GENERATORS = {
    "qasm": ast_to_qasm,
    "guppy": ast_to_guppy,
    "stim": ast_to_stim_str,
    "qir": ast_to_qir,
    "quantum_circuit": ast_to_quantum_circuit,  # Returns actual QuantumCircuit object
}


def generate(
    program: Program,
    target: str = "qasm",
) -> str | object:
    """Generate code for a specific target.

    Args:
        program: The AST Program to generate code for.
        target: The target format ("qasm", "guppy", "stim", "qir", "quantum_circuit").

    Returns:
        Generated code. Most targets return a string, but "quantum_circuit"
        returns a QuantumCircuit object directly.

    Raises:
        ValueError: If target is not supported.
    """
    generator = _GENERATORS.get(target.lower())
    if generator is None:
        msg = f"Unknown target: {target}. Supported: {list(_GENERATORS.keys())}"
        raise ValueError(msg)
    return generator(program)


def generate_with_validation(
    program: Program,
    target: str = "qasm",
    *,
    include_analysis: bool = False,
) -> CodegenResult:
    """Generate code with validation and optional analysis.

    This function validates the program before generating code and
    optionally includes analysis results in the output.

    Args:
        program: The AST Program to generate code for.
        target: The target format ("qasm", "guppy", "stim", "qir", "quantum_circuit").
        include_analysis: If True, include all analysis passes.

    Returns:
        CodegenResult with code and metadata.

    Raises:
        ValueError: If target is not supported.

    Example:
        >>> result = generate_with_validation(ast, target="qasm", include_analysis=True)
        >>> if result.valid:
        ...     print(result.code)
        ...     print(f"T-count: {result.t_count.t_count}")
        ...
    """
    from pecos.slr.ast.analysis import (  # noqa: PLC0415
        analyze_connectivity,
        analyze_depth,
        analyze_parallelism,
        analyze_t_count,
        count_resources,
    )
    from pecos.slr.ast.validation import validate  # noqa: PLC0415

    # Validate
    validation_result = validate(program)

    # Generate code
    code = generate(program, target)

    # Create result
    result = CodegenResult(
        code=code,
        target=target,
        validation=validation_result,
    )

    # Add analysis if requested
    if include_analysis:
        result.resources = count_resources(program)
        result.t_count = analyze_t_count(program)
        result.depth = analyze_depth(program)
        result.connectivity = analyze_connectivity(program)
        result.parallelism = analyze_parallelism(program)

    return result


def generate_with_options(
    program: Program,
    target: str = "qasm",
    options: CodegenOptions | None = None,
) -> CodegenResult:
    """Generate code with fine-grained options.

    Args:
        program: The AST Program to generate code for.
        target: The target format.
        options: Options controlling validation and analysis.

    Returns:
        CodegenResult with code and requested metadata.
    """
    from pecos.slr.ast.analysis import (  # noqa: PLC0415
        analyze_connectivity,
        analyze_depth,
        analyze_parallelism,
        analyze_t_count,
        count_resources,
    )
    from pecos.slr.ast.validation import validate  # noqa: PLC0415

    if options is None:
        options = CodegenOptions()

    # Generate code
    code = generate(program, target)

    # Create result
    result = CodegenResult(code=code, target=target)

    # Add validation if requested
    if options.validate:
        result.validation = validate(program)

    # Add analysis passes based on options
    if options.should_include_resources():
        result.resources = count_resources(program)

    if options.should_include_t_count():
        result.t_count = analyze_t_count(program)

    if options.should_include_depth():
        result.depth = analyze_depth(program)

    if options.should_include_connectivity():
        result.connectivity = analyze_connectivity(program)

    if options.should_include_parallelism():
        result.parallelism = analyze_parallelism(program)

    return result


__all__ = [
    # Guppy code generation
    "AstToGuppy",
    # QASM code generation
    "AstToQasm",
    # QIR code generation
    "AstToQir",
    # QuantumCircuit code generation
    "AstToQuantumCircuit",
    # Stim code generation
    "AstToStim",
    # Base classes
    "CodegenOptions",
    "CodegenResult",
    "ast_to_guppy",
    "ast_to_qasm",
    "ast_to_qir",
    "ast_to_quantum_circuit",
    "ast_to_quantum_circuit_str",
    "ast_to_stim",
    "ast_to_stim_str",
    # Convenience functions
    "generate",
    "generate_with_options",
    "generate_with_validation",
]
