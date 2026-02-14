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

"""AST module for SLR quantum programs.

This module provides an Abstract Syntax Tree (AST) representation for SLR
quantum programs. The AST is the foundation for:

- **Code generation**: Convert to QASM, Guppy, Stim, QIR, QuantumCircuit
- **Validation**: Type checking, bounds validation, allocation tracking
- **Analysis**: T-count, circuit depth, connectivity metrics, parallelism
- **Optimization**: Gate cancellation, rotation merging, and more

Quick Start
-----------
The simplest way to use AST features is through :func:`pecos.slr.generate`:

.. code-block:: python

    from pecos.slr import Main, QReg, generate
    from pecos.slr.qeclib import qubit as qb

    prog = Main(
        q := QReg("q", 2),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
    )
    qasm = generate(prog, "qasm")  # Validates and generates

Direct AST Access
-----------------
For advanced use cases, access the AST directly:

.. code-block:: python

    from pecos.slr.ast import slr_to_ast
    from pecos.slr.ast.validation import validate
    from pecos.slr.ast.codegen import generate_with_validation

    # Convert SLR to AST
    ast = slr_to_ast(prog)

    # Validate
    result = validate(ast)
    assert result.valid

    # Generate with analysis
    codegen_result = generate_with_validation(ast, "qasm", include_analysis=True)
    print(f"T-count: {codegen_result.t_count.t_count}")

Creating AST Directly
---------------------
You can also construct AST nodes directly:

.. code-block:: python

    from pecos.slr.ast import Program, GateOp, GateKind, SlotRef, AllocatorDecl

    program = Program(
        name="example",
        declarations=(AllocatorDecl("q", 2),),
        body=(
            GateOp(GateKind.H, (SlotRef("q", 0),)),
            GateOp(GateKind.CX, (SlotRef("q", 0), SlotRef("q", 1))),
        ),
    )

Custom Visitors
---------------
Extend :class:`BaseVisitor` to create custom analysis passes:

.. code-block:: python

    from pecos.slr.ast import BaseVisitor, GateOp


    class GateCounter(BaseVisitor[int]):
        def default_result(self) -> int:
            return 0

        def combine_results(self, results: list[int]) -> int:
            return sum(results)

        def visit_gate(self, node: GateOp) -> int:
            return 1 + self.combine_results(self.visit_children(node))


    counter = GateCounter()
    gate_count = counter.visit(program)

Submodules
----------
- :mod:`pecos.slr.ast.codegen` - Code generators for various targets
- :mod:`pecos.slr.ast.validation` - Validation passes
- :mod:`pecos.slr.ast.analysis` - Analysis passes (T-count, depth, etc.)
- :mod:`pecos.slr.ast.optimization` - Optimization passes
- :mod:`pecos.slr.ast.serialize` - JSON serialization
- :mod:`pecos.slr.ast.pretty_print` - Human-readable output
- :mod:`pecos.slr.ast.compare` - AST comparison tools
"""

from pecos.slr.ast.analysis import (
    AstQubitStateValidator,
    DepthAnalyzer,
    DepthResult,
    QubitStateTracker,
    ResourceCount,
    ResourceCounter,
    StateViolation,
    ValidationSlotState,
    analyze_depth,
    count_resources,
    validate_ast_qubit_states,
)
from pecos.slr.ast.codegen import (
    AstToGuppy,
    AstToQasm,
    CodegenOptions,
    CodegenResult,
    ast_to_guppy,
    ast_to_qasm,
    generate,
    generate_with_options,
    generate_with_validation,
)
from pecos.slr.ast.compare import ast_equal, compare_ast, nodes_equal
from pecos.slr.ast.converter import SlrToAst, slr_to_ast
from pecos.slr.ast.nodes import (
    # Declarations
    AllocatorDecl,
    # Types
    AllocatorTypeExpr,
    ArrayTypeExpr,
    # Statements
    AssignOp,
    # Base
    AstNode,
    BarrierOp,
    # Expressions
    BinaryExpr,
    # Enums
    BinaryOp,
    BitExpr,
    # References
    BitRef,
    BitTypeExpr,
    CommentOp,
    Declaration,
    Expression,
    # Control flow
    ForStmt,
    GateKind,
    GateOp,
    IfStmt,
    LiteralExpr,
    MeasureOp,
    ParallelBlock,
    PermuteOp,
    PrepareOp,
    # Program
    Program,
    QubitTypeExpr,
    RegisterDecl,
    RepeatStmt,
    ReturnOp,
    SlotRef,
    SourceLocation,
    Statement,
    TypeExpr,
    UnaryExpr,
    UnaryOp,
    VarExpr,
    WhileStmt,
)
from pecos.slr.ast.pretty_print import format_expression, format_statement, pretty_print
from pecos.slr.ast.serialize import ast_to_dict, ast_to_json, dict_to_ast, json_to_ast
from pecos.slr.ast.visitor import (
    AstVisitor,
    BaseVisitor,
    CollectingVisitor,
    VoidVisitor,
)

__all__ = [
    "AllocatorDecl",
    "AllocatorTypeExpr",
    "ArrayTypeExpr",
    "AssignOp",
    # Base
    "AstNode",
    # Analysis
    "AstQubitStateValidator",
    # Code generation
    "AstToGuppy",
    "AstToQasm",
    # Visitors
    "AstVisitor",
    "BarrierOp",
    "BaseVisitor",
    "BinaryExpr",
    "BinaryOp",
    "BitExpr",
    "BitRef",
    "BitTypeExpr",
    "CodegenOptions",
    "CodegenResult",
    "CollectingVisitor",
    "CommentOp",
    # Declarations
    "Declaration",
    "DepthAnalyzer",
    "DepthResult",
    # Expressions
    "Expression",
    "ForStmt",
    # Enums
    "GateKind",
    "GateOp",
    # Control flow
    "IfStmt",
    "LiteralExpr",
    "MeasureOp",
    "ParallelBlock",
    "PermuteOp",
    "PrepareOp",
    # Program
    "Program",
    "QubitStateTracker",
    "QubitTypeExpr",
    "RegisterDecl",
    "RepeatStmt",
    "ResourceCount",
    "ResourceCounter",
    "ReturnOp",
    # References
    "SlotRef",
    # Converter
    "SlrToAst",
    "SourceLocation",
    "StateViolation",
    # Statements
    "Statement",
    # Types
    "TypeExpr",
    "UnaryExpr",
    "UnaryOp",
    "ValidationSlotState",
    "VarExpr",
    "VoidVisitor",
    "WhileStmt",
    "analyze_depth",
    # Comparison
    "ast_equal",
    # Serialization
    "ast_to_dict",
    "ast_to_guppy",
    "ast_to_json",
    "ast_to_qasm",
    "compare_ast",
    "count_resources",
    "dict_to_ast",
    # Pretty-printing
    "format_expression",
    "format_statement",
    "generate",
    "generate_with_options",
    "generate_with_validation",
    "json_to_ast",
    "nodes_equal",
    "pretty_print",
    "slr_to_ast",
    "validate_ast_qubit_states",
]
