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

"""Integration tests for AST infrastructure.

These tests verify that the various AST modules work correctly together:
- Serialization
- Pretty-printing
- Comparison
- Validation
- Analysis
- Code generation
"""

import math

import pytest
from pecos.slr import CReg, If, Main, QReg, Repeat
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.codegen import (
    CodegenOptions,
    generate,
    generate_with_options,
    generate_with_validation,
)
from pecos.slr.ast.compare import ast_equal
from pecos.slr.ast.nodes import (
    AllocatorDecl,
    GateKind,
    GateOp,
    LiteralExpr,
    Program,
    SlotRef,
)
from pecos.slr.ast.pretty_print import pretty_print
from pecos.slr.ast.serialize import ast_to_json, json_to_ast
from pecos.slr.qeclib import qubit as qb


class TestFullPipeline:
    """Test full pipeline: SLR -> AST -> validate -> analyze -> codegen."""

    def test_simple_circuit_pipeline(self) -> None:
        """Simple circuit through full pipeline."""
        # 1. Create SLR program
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        # 2. Convert to AST
        ast = slr_to_ast(prog)

        # 3. Generate with validation and analysis
        result = generate_with_validation(ast, target="qasm", include_analysis=True)

        # 4. Verify results
        assert result.valid
        assert result.code is not None
        assert "OPENQASM" in result.code
        assert result.resources is not None
        assert result.resources.total_gates == 2  # H + CX
        assert result.t_count is not None
        assert result.t_count.t_count == 0  # No T gates

    def test_ghz_state_pipeline(self) -> None:
        """GHZ state preparation through pipeline."""
        prog = Main(
            q := QReg("q", 4),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.CX(q[1], q[2]),
            qb.CX(q[2], q[3]),
        )

        ast = slr_to_ast(prog)
        result = generate_with_validation(ast, target="qasm", include_analysis=True)

        assert result.valid
        assert result.resources.total_gates == 4  # 1 H + 3 CX
        assert result.connectivity is not None
        assert result.connectivity.is_linear  # GHZ has linear connectivity
        assert len(result.connectivity.edges) == 3

    def test_t_gate_circuit_pipeline(self) -> None:
        """Circuit with T gates through pipeline."""
        prog = Program(
            name="t_circuit",
            allocator=AllocatorDecl(name="q", capacity=2),
            body=(
                GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),)),
                GateOp(gate=GateKind.T, targets=(SlotRef(allocator="q", index=0),)),
                GateOp(gate=GateKind.T, targets=(SlotRef(allocator="q", index=1),)),
                GateOp(
                    gate=GateKind.CX,
                    targets=(
                        SlotRef(allocator="q", index=0),
                        SlotRef(allocator="q", index=1),
                    ),
                ),
            ),
        )

        result = generate_with_validation(prog, target="qasm", include_analysis=True)

        assert result.valid
        assert result.t_count.t_count == 2


class TestSerializationRoundTrip:
    """Test serialization round-trips preserve equality."""

    def test_basic_roundtrip(self) -> None:
        """Basic serialization round-trip."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)

        # Round-trip through JSON
        json_str = ast_to_json(ast)
        restored = json_to_ast(json_str)

        # Verify equality
        assert ast_equal(ast, restored, ignore_name=True)

    def test_complex_circuit_roundtrip(self) -> None:
        """Complex circuit with control flow round-trips."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 1),
            qb.H(q[0]),
            If(c[0] == 1).Then(
                qb.X(q[0]),
            ),
            Repeat(cond=3).block(
                qb.CX(q[0], q[1]),
            ),
        )

        ast = slr_to_ast(prog)
        json_str = ast_to_json(ast)
        restored = json_to_ast(json_str)

        assert ast_equal(ast, restored, ignore_name=True)

    def test_rotation_gates_roundtrip(self) -> None:
        """Rotation gates with float params round-trip."""
        prog = Main(
            q := QReg("q", 1),
            qb.RZ[0.5](q[0]),
            qb.RX[math.pi / 4](q[0]),
            qb.RY[1.234567890123](q[0]),
        )

        ast = slr_to_ast(prog)
        json_str = ast_to_json(ast)
        restored = json_to_ast(json_str)

        assert ast_equal(ast, restored, ignore_name=True)

        # Verify float values preserved
        original_rz = next(s for s in ast.body if isinstance(s, GateOp) and s.gate == GateKind.RZ)
        restored_rz = next(s for s in restored.body if isinstance(s, GateOp) and s.gate == GateKind.RZ)
        assert original_rz.params[0].value == restored_rz.params[0].value


class TestValidationCodegen:
    """Test validation before code generation."""

    def test_valid_program_generates_code(self) -> None:
        """Valid program generates code successfully."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)
        result = generate_with_validation(ast, target="qasm")

        assert result.valid
        assert result.code is not None
        assert len(result.code) > 0

    def test_invalid_program_reports_errors(self) -> None:
        """Invalid program reports validation errors but still generates code."""
        # Create program with out-of-bounds access
        prog = Program(
            name="invalid",
            allocator=AllocatorDecl(name="q", capacity=2),
            body=(
                GateOp(
                    gate=GateKind.H,
                    targets=(SlotRef(allocator="q", index=10),),  # Out of bounds
                ),
            ),
        )

        result = generate_with_validation(prog, target="qasm")

        # Validation should fail
        assert not result.valid
        assert len(result.validation.errors) > 0

        # But code should still be generated
        assert result.code is not None

    def test_all_targets_work(self) -> None:
        """All code generation targets work with validation."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)

        targets = ["qasm", "guppy", "stim"]
        for target in targets:
            result = generate_with_validation(ast, target=target)
            assert result.valid, f"Failed for target: {target}"
            assert result.code is not None, f"No code for target: {target}"


class TestAnalysisIntegration:
    """Test analysis passes work together."""

    def test_multiple_analysis_passes(self) -> None:
        """Multiple analysis passes return consistent results."""
        prog = Main(
            q := QReg("q", 3),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.CX(q[1], q[2]),
        )

        ast = slr_to_ast(prog)
        result = generate_with_validation(ast, target="qasm", include_analysis=True)

        # Check all analysis results are present
        assert result.resources is not None
        assert result.t_count is not None
        assert result.depth is not None
        assert result.connectivity is not None
        assert result.parallelism is not None

        # Check consistency
        assert result.resources.total_gates == 3  # 1 H + 2 CX
        assert result.resources.qubit_count == 3
        assert result.connectivity.max_degree == 2  # Middle qubit

    def test_fine_grained_options(self) -> None:
        """Fine-grained options control which analyses run."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)

        # Only request T-count
        options = CodegenOptions(validate=True, include_t_count=True)
        result = generate_with_options(ast, target="qasm", options=options)

        assert result.validation is not None
        assert result.t_count is not None
        assert result.resources is None  # Not requested
        assert result.depth is None  # Not requested

    def test_no_validation_option(self) -> None:
        """Can skip validation with options."""
        prog = Main(q := QReg("q", 1), qb.H(q[0]))

        ast = slr_to_ast(prog)

        options = CodegenOptions(validate=False, include_resources=True)
        result = generate_with_options(ast, target="qasm", options=options)

        assert result.validation is None
        assert result.resources is not None


class TestPrettyPrintIntegration:
    """Test pretty-print integration."""

    def test_pretty_print_readable(self) -> None:
        """Pretty-printed output is readable."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)
        output = pretty_print(ast)

        # Should contain key elements
        assert "Main(" in output
        assert "QReg" in output
        assert "qb.H" in output
        assert "qb.CX" in output

    def test_pretty_print_with_control_flow(self) -> None:
        """Pretty-print handles control flow."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.H(q[0]),
            ),
        )

        ast = slr_to_ast(prog)
        output = pretty_print(ast)

        assert "If(" in output
        assert ".Then(" in output


class TestCodegenResult:
    """Test CodegenResult class."""

    def test_result_string_representation(self) -> None:
        """CodegenResult has useful string representation."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)
        result = generate_with_validation(ast, target="qasm", include_analysis=True)

        result_str = str(result)

        assert "qasm" in result_str
        assert "valid" in result_str.lower()
        assert "gates" in result_str.lower()

    def test_result_bool_valid(self) -> None:
        """CodegenResult.valid property works."""
        prog = Main(q := QReg("q", 1), qb.H(q[0]))
        ast = slr_to_ast(prog)

        result = generate_with_validation(ast, target="qasm")

        assert result.valid is True

    def test_generate_simple(self) -> None:
        """Simple generate function works."""
        prog = Main(q := QReg("q", 1), qb.H(q[0]))
        ast = slr_to_ast(prog)

        code = generate(ast, target="qasm")

        assert isinstance(code, str)
        assert "OPENQASM" in code

    def test_generate_unknown_target_raises(self) -> None:
        """Unknown target raises ValueError."""
        prog = Main(q := QReg("q", 1), qb.H(q[0]))
        ast = slr_to_ast(prog)

        with pytest.raises(ValueError, match="Unknown target"):
            generate(ast, target="unknown_target")


class TestRealWorldPatterns:
    """Test real-world quantum patterns."""

    def test_bell_measurement_pattern(self) -> None:
        """Bell state preparation pattern."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)
        result = generate_with_validation(ast, target="qasm", include_analysis=True)

        assert result.valid
        assert result.connectivity.is_linear

    def test_ghz_pattern(self) -> None:
        """GHZ state preparation pattern."""
        prog = Main(
            q := QReg("q", 4),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.CX(q[1], q[2]),
            qb.CX(q[2], q[3]),
        )

        ast = slr_to_ast(prog)
        result = generate_with_validation(ast, target="qasm", include_analysis=True)

        assert result.valid
        assert len(result.connectivity.edges) == 3
        assert result.connectivity.is_linear

    def test_parallel_operations_pattern(self) -> None:
        """Parallel single-qubit operations pattern."""
        prog = Main(
            q := QReg("q", 4),
            qb.H(q[0]),
            qb.H(q[1]),
            qb.H(q[2]),
            qb.H(q[3]),
        )

        ast = slr_to_ast(prog)
        result = generate_with_validation(ast, target="qasm", include_analysis=True)

        assert result.valid
        assert result.parallelism is not None
        # 4 H gates can run in parallel
        assert result.parallelism.max_parallel_gates >= 4

    def test_repeated_syndrome_pattern(self) -> None:
        """Repeated syndrome measurement pattern."""
        prog = Main(
            q := QReg("q", 2),
            Repeat(cond=3).block(
                qb.H(q[0]),
                qb.CX(q[0], q[1]),
                qb.H(q[0]),
            ),
        )

        ast = slr_to_ast(prog)
        result = generate_with_validation(ast, target="qasm", include_analysis=True)

        assert result.valid
