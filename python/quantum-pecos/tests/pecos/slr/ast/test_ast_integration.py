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

"""Integration tests for the full AST pipeline.

Tests the complete flow: SLR → AST → validate → optimize → analyze → codegen
"""

import pytest

from pecos.slr import CReg, If, Main, QAlloc, QReg, Repeat
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.analysis import (
    analyze_connectivity,
    analyze_depth,
    analyze_parallelism,
    analyze_t_count,
    count_resources,
)
from pecos.slr.ast.codegen import (
    generate,
    generate_with_options,
    generate_with_validation,
    CodegenOptions,
)
from pecos.slr.ast.compare import ast_equal
from pecos.slr.ast.optimizations import optimize
from pecos.slr.ast.pretty_print import pretty_print
from pecos.slr.ast.serialize import ast_to_json, json_to_ast
from pecos.slr.ast.validation import validate
from pecos.slr.qeclib import qubit as qb


class TestFullPipeline:
    """Test complete SLR → AST → validate → optimize → codegen pipeline."""

    def test_simple_circuit_pipeline(self):
        """Test basic pipeline with Bell state."""
        # SLR → AST
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        # Validate
        validation = validate(ast)
        assert validation.valid

        # Optimize (should not change anything here)
        opt_result = optimize(ast, level=1)
        assert opt_result.gates_removed == 0

        # Generate code
        qasm = generate(opt_result.program, "qasm")
        assert "h q[0]" in qasm.lower()
        assert "cx q[0], q[1]" in qasm.lower() or "cx q[0],q[1]" in qasm.lower()

    def test_optimize_then_generate(self):
        """Test optimization followed by code generation."""
        # Create circuit with redundant gates
        prog = Main(
            q := QReg("q", 1),
            qb.X(q[0]),
            qb.X(q[0]),  # Should cancel
            qb.H(q[0]),
        )
        ast = slr_to_ast(prog)

        # Optimize
        opt_result = optimize(ast, level=1)
        assert opt_result.gates_removed == 2  # Both X gates removed

        # Validate optimized
        validation = validate(opt_result.program)
        assert validation.valid

        # Generate code
        qasm = generate(opt_result.program, "qasm")
        # Should only have H gate
        assert "h q[0]" in qasm.lower()
        assert qasm.lower().count("x q[0]") == 0

    def test_pipeline_with_control_flow(self):
        """Test pipeline with If statement."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.H(q[0]),
            qb.Measure(q[0]) > c[0],
            If(c[0] == 1).Then(
                qb.X(q[0]),
            ),
        )
        ast = slr_to_ast(prog)

        # Validate
        validation = validate(ast)
        assert validation.valid

        # Analyze
        resources = count_resources(ast)
        assert resources.total_gates >= 2  # H and X
        assert resources.measurement_count == 1

        # Generate
        qasm = generate(ast, "qasm")
        assert "if" in qasm.lower() or "measure" in qasm.lower()

    def test_pipeline_with_hierarchical_allocators(self):
        """Test pipeline with hierarchical allocators."""
        all_qubits = QAlloc(4, name="all")
        data = QAlloc(2, name="data", parent=all_qubits)
        ancilla = QAlloc(2, name="ancilla", parent=all_qubits)

        prog = Main(
            all_qubits,
            data,
            ancilla,
            qb.H(data[0]),
            qb.CX(data[0], ancilla[0]),
        )
        ast = slr_to_ast(prog)

        # Validate
        validation = validate(ast)
        assert validation.valid

        # Generate for different targets
        qasm = generate(ast, "qasm")
        assert "qreg" in qasm.lower() or "qubit" in qasm.lower()


class TestAnalysisIntegration:
    """Test combining multiple analysis passes."""

    def test_all_analysis_passes(self):
        """Run all analysis passes on a single program."""
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 1),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.CX(q[1], q[2]),
            qb.Measure(q[2]) > c[0],
        )
        ast = slr_to_ast(prog)

        # Run all analyses
        resources = count_resources(ast)
        depth = analyze_depth(ast)
        t_count = analyze_t_count(ast)
        connectivity = analyze_connectivity(ast)
        parallelism = analyze_parallelism(ast)

        # Verify results
        assert resources.total_gates == 3  # H + 2 CX
        assert resources.measurement_count == 1
        assert depth.depth >= 3  # Sequential gates
        assert t_count.t_count == 0  # No T gates
        assert resources.qubit_count == 3

    def test_analysis_with_t_gates(self):
        """Test T-count analysis with T gates."""
        prog = Main(
            q := QReg("q", 1),
            qb.T(q[0]),
            qb.H(q[0]),
            qb.T(q[0]),
        )
        ast = slr_to_ast(prog)

        t_count = analyze_t_count(ast)
        assert t_count.t_count == 2

    def test_analysis_consistency(self):
        """Verify analysis results are consistent with each other."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.H(q[1]),  # Can run in parallel with above
            qb.CX(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        resources = count_resources(ast)
        connectivity = analyze_connectivity(ast)

        # Should have 3 gates total
        assert resources.total_gates == 3
        # Should have 2 qubits based on qubit count from allocators
        assert resources.qubit_count == 2


class TestValidationCodegen:
    """Test validation before code generation."""

    def test_generate_with_validation_valid(self):
        """Test generate_with_validation with valid program."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        result = generate_with_validation(ast, target="qasm")

        assert result.valid
        assert result.code is not None
        assert "h" in result.code.lower()

    def test_generate_with_validation_and_analysis(self):
        """Test generate_with_validation with analysis enabled."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.T(q[0]),
            qb.CX(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        result = generate_with_validation(
            ast, target="qasm", include_analysis=True
        )

        assert result.valid
        assert result.resources is not None
        assert result.t_count is not None
        assert result.t_count.t_count == 1
        assert result.depth is not None

    def test_generate_with_options(self):
        """Test generate_with_options with custom options."""
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
        )
        ast = slr_to_ast(prog)

        options = CodegenOptions(
            validate=True,
            include_resources=True,
            include_t_count=True,
        )
        result = generate_with_options(ast, target="qasm", options=options)

        assert result.validation is not None
        assert result.validation.valid
        assert result.resources is not None

    def test_all_codegen_targets(self):
        """Test code generation for all supported targets."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        # Test each target
        qasm = generate(ast, "qasm")
        assert isinstance(qasm, str)
        assert len(qasm) > 0

        guppy = generate(ast, "guppy")
        assert isinstance(guppy, str)
        assert len(guppy) > 0

        stim = generate(ast, "stim")
        assert isinstance(stim, str)
        # Stim uses different gate names

        qir = generate(ast, "qir")
        assert isinstance(qir, str)
        assert len(qir) > 0

        qc = generate(ast, "quantum_circuit")
        # quantum_circuit returns an object, not a string
        assert qc is not None


class TestSerializationRoundtrip:
    """Test AST serialization round-trips."""

    def test_serialize_validate_deserialize(self):
        """Test: serialize → validate → deserialize → compare."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 1),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.Measure(q[1]) > c[0],
        )
        ast = slr_to_ast(prog)

        # Serialize
        json_str = ast_to_json(ast)

        # Deserialize
        ast2 = json_to_ast(json_str)

        # Compare
        assert ast_equal(ast, ast2)

        # Both should validate
        assert validate(ast).valid
        assert validate(ast2).valid

    def test_serialize_optimize_compare(self):
        """Test: optimize → serialize → deserialize → compare."""
        prog = Main(
            q := QReg("q", 1),
            qb.X(q[0]),
            qb.X(q[0]),
            qb.H(q[0]),
        )
        ast = slr_to_ast(prog)

        # Optimize
        opt_result = optimize(ast, level=1)
        optimized = opt_result.program

        # Serialize round-trip
        json_str = ast_to_json(optimized)
        restored = json_to_ast(json_str)

        # Compare
        assert ast_equal(optimized, restored)

    def test_serialize_generate_compare(self):
        """Test: generate code from original and deserialized AST."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        # Serialize round-trip
        json_str = ast_to_json(ast)
        restored = json_to_ast(json_str)

        # Generate from both
        qasm1 = generate(ast, "qasm")
        qasm2 = generate(restored, "qasm")

        # Should produce identical code
        assert qasm1 == qasm2


class TestOptimizeAndGenerate:
    """Test optimization followed by code generation for each target."""

    def test_optimize_then_qasm(self):
        """Test optimize then QASM generation."""
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
            qb.H(q[0]),  # Cancels
            qb.X(q[0]),
        )
        ast = slr_to_ast(prog)

        opt = optimize(ast, level=1)
        assert opt.gates_removed == 2

        qasm = generate(opt.program, "qasm")
        assert "x q[0]" in qasm.lower()
        # H gates should be gone
        assert qasm.lower().count("h ") <= 1  # Allow for potential header

    def test_optimize_then_guppy(self):
        """Test optimize then Guppy generation."""
        prog = Main(
            q := QReg("q", 1),
            qb.X(q[0]),
            qb.X(q[0]),
        )
        ast = slr_to_ast(prog)

        opt = optimize(ast, level=1)
        assert opt.gates_removed == 2

        guppy = generate(opt.program, "guppy")
        # Should have minimal content
        assert "def" in guppy or "@guppy" in guppy

    def test_optimize_then_stim(self):
        """Test optimize then Stim generation."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        opt = optimize(ast, level=1)
        assert opt.gates_removed == 2

        stim = generate(opt.program, "stim")
        # Stim uses different naming but should have CX
        assert len(stim) > 0


class TestQECPatterns:
    """Test QEC-specific patterns through the pipeline."""

    def test_syndrome_extraction_pipeline(self):
        """Test syndrome extraction pattern."""
        prog = Main(
            data := QReg("data", 2),
            ancilla := QReg("ancilla", 1),
            syndrome := CReg("syndrome", 1),
            # ZZ stabilizer measurement
            qb.CX(data[0], ancilla[0]),
            qb.CX(data[1], ancilla[0]),
            qb.Measure(ancilla[0]) > syndrome[0],
        )
        ast = slr_to_ast(prog)

        # Validate
        validation = validate(ast)
        assert validation.valid

        # Analyze
        resources = count_resources(ast)
        assert resources.two_qubit_gates == 2
        assert resources.measurement_count == 1

        # Generate
        result = generate_with_validation(ast, target="qasm", include_analysis=True)
        assert result.valid

    def test_ghz_state_preparation(self):
        """Test GHZ state preparation."""
        prog = Main(
            q := QReg("q", 4),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.CX(q[0], q[2]),
            qb.CX(q[0], q[3]),
        )
        ast = slr_to_ast(prog)

        # Analyze
        resources = count_resources(ast)
        assert resources.single_qubit_gates == 1  # H
        assert resources.two_qubit_gates == 3  # 3 CX

        depth = analyze_depth(ast)
        assert depth.depth >= 4  # H + 3 sequential CX

        # Optimize (nothing to optimize here)
        opt = optimize(ast, level=1)
        assert opt.gates_removed == 0

    def test_repeat_stabilizer_rounds(self):
        """Test repeated stabilizer measurement."""
        prog = Main(
            data := QReg("data", 2),
            ancilla := QReg("ancilla", 1),
            c := CReg("c", 1),
            Repeat(cond=3).block(
                qb.CX(data[0], ancilla[0]),
                qb.CX(data[1], ancilla[0]),
                qb.Measure(ancilla[0]) > c[0],
            ),
        )
        ast = slr_to_ast(prog)

        # Validate
        validation = validate(ast)
        assert validation.valid

        # Generate
        qasm = generate(ast, "qasm")
        assert len(qasm) > 0


class TestPrettyPrintIntegration:
    """Test pretty-printing with other operations."""

    def test_pretty_print_optimized(self):
        """Test pretty-printing optimized AST."""
        prog = Main(
            q := QReg("q", 1),
            qb.X(q[0]),
            qb.X(q[0]),
            qb.H(q[0]),
        )
        ast = slr_to_ast(prog)

        # Optimize
        opt = optimize(ast, level=1)

        # Pretty-print
        output = pretty_print(opt.program)
        assert "Main(" in output
        assert "qb.H" in output
        # X gates should be gone
        assert output.count("qb.X") == 0

    def test_pretty_print_deserialized(self):
        """Test pretty-printing after deserialization."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        # Serialize round-trip
        json_str = ast_to_json(ast)
        restored = json_to_ast(json_str)

        # Pretty-print both
        output1 = pretty_print(ast)
        output2 = pretty_print(restored)

        # Should be identical
        assert output1 == output2


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_program(self):
        """Test pipeline with empty program."""
        prog = Main()
        ast = slr_to_ast(prog)

        # Should validate
        validation = validate(ast)
        assert validation.valid

        # Should generate (minimal output)
        qasm = generate(ast, "qasm")
        assert isinstance(qasm, str)

    def test_invalid_target_raises_error(self):
        """Test that invalid target raises ValueError."""
        prog = Main(q := QReg("q", 1), qb.H(q[0]))
        ast = slr_to_ast(prog)

        with pytest.raises(ValueError, match="Unknown target"):
            generate(ast, "invalid_target")

    def test_multiple_optimization_levels(self):
        """Test different optimization levels."""
        prog = Main(
            q := QReg("q", 1),
            qb.X(q[0]),
            qb.X(q[0]),
        )
        ast = slr_to_ast(prog)

        # Level 0 - no optimization
        opt0 = optimize(ast, level=0)
        assert opt0.gates_removed == 0

        # Level 1 - gate cancellation
        opt1 = optimize(ast, level=1)
        assert opt1.gates_removed == 2

        # Level 2 - includes rotation merging
        opt2 = optimize(ast, level=2)
        assert opt2.gates_removed >= 2

    def test_deep_nesting(self):
        """Test pipeline with deeply nested control flow."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 2),
            If(c[0] == 1).Then(
                If(c[1] == 1).Then(
                    qb.X(q[0]),
                ).Else(
                    qb.Y(q[0]),
                ),
            ).Else(
                qb.Z(q[0]),
            ),
        )
        ast = slr_to_ast(prog)

        # Should validate
        validation = validate(ast)
        assert validation.valid

        # Should generate
        qasm = generate(ast, "qasm")
        assert len(qasm) > 0


class TestMultipleCodegenTargets:
    """Test that all codegen targets work with various patterns."""

    @pytest.mark.parametrize("target", ["qasm", "guppy", "stim", "qir"])
    def test_bell_state_all_targets(self, target):
        """Test Bell state generation for all targets."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        code = generate(ast, target)
        assert isinstance(code, str)
        assert len(code) > 0

    @pytest.mark.parametrize("target", ["qasm", "guppy", "stim", "qir"])
    def test_with_measurement_all_targets(self, target):
        """Test measurement generation for all targets."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.H(q[0]),
            qb.Measure(q[0]) > c[0],
        )
        ast = slr_to_ast(prog)

        code = generate(ast, target)
        assert isinstance(code, str)
        assert len(code) > 0
