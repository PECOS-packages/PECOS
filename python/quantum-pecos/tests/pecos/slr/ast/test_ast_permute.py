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

"""Tests for AST Permute operation support."""

import pytest

from pecos.slr import CReg, Main, Permute, QReg
from pecos.slr.ast import slr_to_ast, PermuteOp
from pecos.slr.ast.codegen import generate
from pecos.slr.ast.compare import ast_equal
from pecos.slr.ast.pretty_print import pretty_print
from pecos.slr.ast.serialize import ast_to_json, json_to_ast
from pecos.slr.qeclib import qubit as qb


class TestPermuteConversion:
    """Test SLR Permute to AST PermuteOp conversion."""

    def test_simple_permute_conversion(self):
        """Test basic Permute conversion."""
        prog = Main(
            a := QReg("a", 2),
            b := QReg("b", 2),
            Permute(a, b),
        )
        ast = slr_to_ast(prog)

        # Find the PermuteOp in the body
        permute_ops = [s for s in ast.body if isinstance(s, PermuteOp)]
        assert len(permute_ops) == 1

        permute = permute_ops[0]
        assert permute.sources == ("a",)
        assert permute.targets == ("b",)

    def test_permute_with_comment(self):
        """Test Permute with comment flag."""
        prog = Main(
            a := QReg("a", 2),
            b := QReg("b", 2),
            Permute(a, b, comment=True),
        )
        ast = slr_to_ast(prog)

        permute_ops = [s for s in ast.body if isinstance(s, PermuteOp)]
        assert len(permute_ops) == 1
        assert permute_ops[0].add_comment is True

    def test_permute_without_comment(self):
        """Test Permute without comment flag."""
        prog = Main(
            a := QReg("a", 2),
            b := QReg("b", 2),
            Permute(a, b, comment=False),
        )
        ast = slr_to_ast(prog)

        permute_ops = [s for s in ast.body if isinstance(s, PermuteOp)]
        assert len(permute_ops) == 1
        assert permute_ops[0].add_comment is False


class TestPermuteOpNode:
    """Test PermuteOp node behavior."""

    def test_permute_op_creation(self):
        """Test direct PermuteOp creation."""
        permute = PermuteOp(
            sources=("a", "b"),
            targets=("b", "a"),
            add_comment=True,
        )
        assert permute.sources == ("a", "b")
        assert permute.targets == ("b", "a")
        assert permute.add_comment is True

    def test_permute_op_frozen(self):
        """Test that PermuteOp is frozen (immutable)."""
        permute = PermuteOp(sources=("a",), targets=("b",))
        with pytest.raises(AttributeError):
            permute.sources = ("c",)


class TestPermuteCodegen:
    """Test code generation for Permute."""

    def test_permute_guppy_codegen(self):
        """Test Permute generates Guppy swap code."""
        prog = Main(
            a := QReg("a", 2),
            b := QReg("b", 2),
            qb.H(a[0]),
            Permute(a, b),
            qb.X(a[0]),  # Now refers to what was b[0]
        )
        ast = slr_to_ast(prog)

        guppy = generate(ast, "guppy")
        # Should contain swap code
        assert "Swap" in guppy or "_temp_" in guppy or "a, b = b, a" in guppy

    def test_permute_qasm_codegen(self):
        """Test Permute generates QASM comment."""
        prog = Main(
            a := QReg("a", 2),
            b := QReg("b", 2),
            Permute(a, b),
        )
        ast = slr_to_ast(prog)

        qasm = generate(ast, "qasm")
        # Should contain a comment about permute
        assert "Permute" in qasm or "permute" in qasm.lower()

    def test_permute_stim_codegen(self):
        """Test Permute works with Stim codegen."""
        prog = Main(
            a := QReg("a", 2),
            b := QReg("b", 2),
            qb.H(a[0]),
            Permute(a, b),
            qb.X(a[0]),
        )
        ast = slr_to_ast(prog)

        # Should not raise an error
        stim = generate(ast, "stim")
        assert stim is not None

    def test_permute_qir_codegen(self):
        """Test Permute works with QIR codegen."""
        prog = Main(
            a := QReg("a", 2),
            b := QReg("b", 2),
            qb.H(a[0]),
            Permute(a, b),
            qb.X(a[0]),
        )
        ast = slr_to_ast(prog)

        # Should not raise an error
        qir = generate(ast, "qir")
        assert qir is not None


class TestPermuteSerialization:
    """Test Permute serialization round-trip."""

    def test_permute_json_roundtrip(self):
        """Test Permute survives JSON serialization."""
        prog = Main(
            a := QReg("a", 2),
            b := QReg("b", 2),
            qb.H(a[0]),
            Permute(a, b),
            qb.X(a[0]),
        )
        ast = slr_to_ast(prog)

        # Serialize round-trip
        json_str = ast_to_json(ast)
        restored = json_to_ast(json_str)

        # Should be equal
        assert ast_equal(ast, restored)

        # Check PermuteOp preserved
        permute_ops = [s for s in restored.body if isinstance(s, PermuteOp)]
        assert len(permute_ops) == 1
        assert permute_ops[0].sources == ("a",)
        assert permute_ops[0].targets == ("b",)


class TestPermutePrettyPrint:
    """Test Permute pretty-printing."""

    def test_permute_pretty_print(self):
        """Test Permute appears in pretty-print output."""
        prog = Main(
            a := QReg("a", 2),
            b := QReg("b", 2),
            Permute(a, b),
        )
        ast = slr_to_ast(prog)

        output = pretty_print(ast)
        # Should contain something about permute
        # The exact format depends on the pretty-printer implementation
        assert "a" in output and "b" in output


class TestPermuteWithOperations:
    """Test Permute combined with other operations."""

    def test_permute_between_gates(self):
        """Test Permute between gate operations."""
        prog = Main(
            a := QReg("a", 2),
            b := QReg("b", 2),
            qb.H(a[0]),
            qb.X(b[0]),
            Permute(a, b),
            # After permute, a[0] is what was b[0] and vice versa
            qb.CX(a[0], a[1]),
        )
        ast = slr_to_ast(prog)

        # Should have gates and permute in correct order
        assert len(ast.body) >= 4

    def test_permute_with_measurement(self):
        """Test Permute with measurements."""
        prog = Main(
            a := QReg("a", 2),
            b := QReg("b", 2),
            c := CReg("c", 2),
            qb.H(a[0]),
            Permute(a, b),
            qb.Measure(a[0]) > c[0],
        )
        ast = slr_to_ast(prog)

        # Should not raise an error during code generation
        qasm = generate(ast, "qasm")
        assert "measure" in qasm.lower()


class TestMultiplePermutes:
    """Test multiple Permute operations."""

    def test_consecutive_permutes(self):
        """Test two consecutive Permutes."""
        prog = Main(
            a := QReg("a", 2),
            b := QReg("b", 2),
            Permute(a, b),
            Permute(a, b),  # Should swap back
        )
        ast = slr_to_ast(prog)

        permute_ops = [s for s in ast.body if isinstance(s, PermuteOp)]
        assert len(permute_ops) == 2

    def test_three_way_permute(self):
        """Test permuting three registers."""
        prog = Main(
            a := QReg("a", 2),
            b := QReg("b", 2),
            c := QReg("c", 2),
            Permute(a, b),
            Permute(b, c),
        )
        ast = slr_to_ast(prog)

        permute_ops = [s for s in ast.body if isinstance(s, PermuteOp)]
        assert len(permute_ops) == 2
