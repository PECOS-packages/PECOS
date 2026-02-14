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

"""Tests for AST resource counter."""

import pytest

from pecos.slr import CReg, Main, QReg, Repeat
from pecos.slr.ast import GateKind, slr_to_ast
from pecos.slr.ast.analysis import ResourceCounter, count_resources
from pecos.slr.qeclib import qubit as qb


class TestResourceCounterBasic:
    """Basic resource counting tests."""

    def test_empty_program(self):
        """Empty program has no resources."""
        prog = Main()
        ast = slr_to_ast(prog)

        result = count_resources(ast)

        assert result.qubit_count == 0
        assert result.classical_bit_count == 0
        assert result.total_gates == 0

    def test_qubits_counted(self):
        """Qubits in allocators are counted."""
        prog = Main(
            q := QReg("q", 5),
        )
        ast = slr_to_ast(prog)

        result = count_resources(ast)

        assert result.qubit_count == 5
        assert result.qubits_by_allocator["q"] == 5

    def test_classical_bits_counted(self):
        """Classical bits in registers are counted."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 3),
        )
        ast = slr_to_ast(prog)

        result = count_resources(ast)

        assert result.classical_bit_count == 3
        assert result.bits_by_register["c"] == 3

    def test_multiple_registers(self):
        """Multiple registers are counted separately."""
        prog = Main(
            q1 := QReg("q1", 2),
            q2 := QReg("q2", 3),
            c1 := CReg("c1", 1),
            c2 := CReg("c2", 2),
        )
        ast = slr_to_ast(prog)

        result = count_resources(ast)

        assert result.qubit_count == 5
        assert result.qubits_by_allocator["q1"] == 2
        assert result.qubits_by_allocator["q2"] == 3
        assert result.classical_bit_count == 3
        assert result.bits_by_register["c1"] == 1
        assert result.bits_by_register["c2"] == 2


class TestResourceCounterGates:
    """Gate counting tests."""

    def test_single_qubit_gates(self):
        """Single-qubit gates are counted."""
        prog = Main(
            q := QReg("q", 1),
            qb.Prep(q[0]),
            qb.H(q[0]),
            qb.X(q[0]),
            qb.Z(q[0]),
        )
        ast = slr_to_ast(prog)

        result = count_resources(ast)

        assert result.total_gates == 3  # H, X, Z (Prep is not a gate)
        assert result.single_qubit_gates == 3
        assert result.two_qubit_gates == 0

    def test_two_qubit_gates(self):
        """Two-qubit gates are counted."""
        prog = Main(
            q := QReg("q", 2),
            qb.Prep(q[0]),
            qb.Prep(q[1]),
            qb.CX(q[0], q[1]),
            qb.CZ(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        result = count_resources(ast)

        assert result.total_gates == 2
        assert result.single_qubit_gates == 0
        assert result.two_qubit_gates == 2

    def test_mixed_gates(self):
        """Mixed single and two-qubit gates are counted correctly."""
        prog = Main(
            q := QReg("q", 2),
            qb.Prep(q[0]),
            qb.Prep(q[1]),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.X(q[1]),
        )
        ast = slr_to_ast(prog)

        result = count_resources(ast)

        assert result.total_gates == 3
        assert result.single_qubit_gates == 2  # H, X
        assert result.two_qubit_gates == 1  # CX

    def test_gate_counts_by_type(self):
        """Gates are counted by type."""
        prog = Main(
            q := QReg("q", 2),
            qb.Prep(q[0]),
            qb.Prep(q[1]),
            qb.H(q[0]),
            qb.H(q[1]),
            qb.CX(q[0], q[1]),
            qb.H(q[0]),
        )
        ast = slr_to_ast(prog)

        result = count_resources(ast)

        assert result.gate_counts[GateKind.H] == 3
        assert result.gate_counts[GateKind.CX] == 1


class TestResourceCounterOperations:
    """Measurement and preparation counting tests."""

    def test_measurements_counted(self):
        """Measurements are counted."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qb.Prep(q[0]),
            qb.Prep(q[1]),
            qb.Measure(q[0]) > c[0],
            qb.Measure(q[1]) > c[1],
        )
        ast = slr_to_ast(prog)

        result = count_resources(ast)

        assert result.measurement_count == 2

    def test_preparations_counted(self):
        """Preparations are counted."""
        prog = Main(
            q := QReg("q", 2),
            qb.Prep(q[0]),
            qb.Prep(q[1]),
        )
        ast = slr_to_ast(prog)

        result = count_resources(ast)

        assert result.preparation_count == 2


class TestResourceCounterControlFlow:
    """Control flow resource counting tests."""

    def test_repeat_multiplies_resources(self):
        """Repeat loop multiplies gate counts."""
        prog = Main(
            q := QReg("q", 1),
            qb.Prep(q[0]),
            Repeat(cond=5).block(
                qb.H(q[0]),
                qb.X(q[0]),
            ),
        )
        ast = slr_to_ast(prog)

        result = count_resources(ast)

        assert result.gate_counts[GateKind.H] == 5
        assert result.gate_counts[GateKind.X] == 5
        assert result.total_gates == 10


class TestResourceCounterQEC:
    """QEC pattern resource counting tests."""

    def test_syndrome_extraction_resources(self):
        """Syndrome extraction resources are counted correctly."""
        prog = Main(
            data := QReg("data", 2),
            ancilla := QReg("ancilla", 1),
            c := CReg("c", 1),
            qb.Prep(data[0]),
            qb.Prep(data[1]),
            qb.Prep(ancilla[0]),
            qb.CX(data[0], ancilla[0]),
            qb.CX(data[1], ancilla[0]),
            qb.Measure(ancilla[0]) > c[0],
        )
        ast = slr_to_ast(prog)

        result = count_resources(ast)

        assert result.qubit_count == 3
        assert result.classical_bit_count == 1
        assert result.two_qubit_gates == 2
        assert result.measurement_count == 1
        assert result.preparation_count == 3


class TestResourceCounterClass:
    """Tests for the ResourceCounter class."""

    def test_counter_reusable(self):
        """Counter can be reused for multiple programs."""
        counter = ResourceCounter()

        prog1 = Main(q := QReg("q", 2))
        prog2 = Main(q := QReg("q", 5))

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        result1 = counter.count(ast1)
        result2 = counter.count(ast2)

        assert result1.qubit_count == 2
        assert result2.qubit_count == 5

    def test_result_string_representation(self):
        """ResourceCount has useful string representation."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 1),
            qb.Prep(q[0]),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        result = count_resources(ast)
        result_str = str(result)

        assert "Qubits: 2" in result_str
        assert "Classical bits: 1" in result_str
        assert "Total gates: 2" in result_str
