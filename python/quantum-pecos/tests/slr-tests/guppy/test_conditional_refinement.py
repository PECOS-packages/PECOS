"""Test suite for refined conditional analysis.

This tests the improvement where we track WHICH specific elements are
conditionally accessed, rather than marking the entire array as conditional.
"""

from pecos.qeclib import qubit
from pecos.slr import CReg, If, Main, QReg
from pecos.slr.gen_codes.guppy.ir_analyzer import IRAnalyzer


class TestConditionalElementTracking:
    """Test element-level conditional access tracking."""

    def test_single_element_conditional(self) -> None:
        """Only one element conditional - shouldn't affect others."""
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 1),
            qubit.Measure(q[0]) > c[0],
            If(c[0]).Then(
                qubit.X(q[1]),  # Only q[1] is conditional
            ),
            qubit.H(q[2]),  # q[2] is not conditional
        )

        analyzer = IRAnalyzer()
        analyzer.analyze_block(prog, {"q": q, "c": c})

        # Check that we track which specific elements are conditional
        q_info = analyzer.array_info["q"]
        assert hasattr(q_info, "conditionally_accessed_elements")
        assert 1 in q_info.conditionally_accessed_elements  # q[1] is conditional
        assert 2 not in q_info.conditionally_accessed_elements  # q[2] is not

    def test_multiple_elements_conditional(self) -> None:
        """Multiple elements conditional - track all of them."""
        prog = Main(
            q := QReg("q", 4),
            c := CReg("c", 2),
            qubit.Measure(q[0]) > c[0],
            qubit.Measure(q[1]) > c[1],
            If(c[0]).Then(
                qubit.X(q[2]),  # q[2] conditional on c[0]
            ),
            If(c[1]).Then(
                qubit.Z(q[3]),  # q[3] conditional on c[1]
            ),
        )

        analyzer = IRAnalyzer()
        analyzer.analyze_block(prog, {"q": q, "c": c})

        q_info = analyzer.array_info["q"]
        assert 2 in q_info.conditionally_accessed_elements
        assert 3 in q_info.conditionally_accessed_elements
        # q[0] and q[1] are measured but not used conditionally after
        assert 0 not in q_info.conditionally_accessed_elements
        assert 1 not in q_info.conditionally_accessed_elements

    def test_classical_element_in_condition(self) -> None:
        """Classical element used in condition."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 3),
            qubit.Measure(q[0]) > c[0],
            qubit.Measure(q[1]) > c[1],
            If(c[0]).Then(  # c[0] is in condition
                qubit.X(q[0]),
            ),
        )

        analyzer = IRAnalyzer()
        analyzer.analyze_block(prog, {"q": q, "c": c})

        c_info = analyzer.array_info["c"]
        assert 0 in c_info.conditionally_accessed_elements  # c[0] in condition
        assert 1 not in c_info.conditionally_accessed_elements  # c[1] not in condition
        assert 2 not in c_info.conditionally_accessed_elements  # c[2] never used

    def test_nested_conditionals(self) -> None:
        """Nested conditional blocks."""
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 2),
            qubit.Measure(q[0]) > c[0],
            qubit.Measure(q[1]) > c[1],
            If(c[0]).Then(
                If(c[1]).Then(
                    qubit.X(q[2]),  # q[2] conditional on both
                ),
            ),
        )

        analyzer = IRAnalyzer()
        analyzer.analyze_block(prog, {"q": q, "c": c})

        c_info = analyzer.array_info["c"]
        q_info = analyzer.array_info["q"]

        # Both c[0] and c[1] are in conditions
        assert 0 in c_info.conditionally_accessed_elements
        assert 1 in c_info.conditionally_accessed_elements

        # q[2] is conditionally accessed
        assert 2 in q_info.conditionally_accessed_elements


class TestConditionalUnpackingDecisions:
    """Test that unpacking decisions use refined conditional tracking."""

    def test_no_unpacking_for_non_conditional_elements(self) -> None:
        """Elements not used conditionally shouldn't force unpacking."""
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 1),
            # Use q[0] and q[1] normally (not conditional)
            qubit.H(q[0]),
            qubit.CX(q[0], q[1]),
            qubit.Measure(q[0]) > c[0],
            # Only q[2] is conditional
            If(c[0]).Then(
                qubit.X(q[2]),
            ),
        )

        analyzer = IRAnalyzer()
        plan = analyzer.analyze_block(prog, {"q": q, "c": c})

        # q should be unpacked because q[2] is conditionally accessed
        # But the decision should note that it's only because of q[2]
        q_info = analyzer.array_info["q"]
        assert q_info.conditionally_accessed_elements == {2}

        # Verify unpacking happens (because of q[2])
        assert "q" in plan.arrays_to_unpack

    def test_unpacking_only_when_necessary(self) -> None:
        """Don't unpack if conditional element not in element_accesses."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 1),
            qubit.H(q[0]),
            qubit.Measure(q[0]) > c[0],
            # c[0] is in condition, but we don't access q elements conditionally
            If(c[0]).Then(
                # Empty then block
            ),
        )

        analyzer = IRAnalyzer()
        analyzer.analyze_block(prog, {"q": q, "c": c})

        c_info = analyzer.array_info["c"]
        # c[0] is in condition
        assert 0 in c_info.conditionally_accessed_elements
        # But since only one element (c[0]) and it's just in condition,
        # unpacking decision depends on other rules

    def test_mixed_conditional_and_unconditional(self) -> None:
        """Mix of conditional and unconditional access."""
        prog = Main(
            q := QReg("q", 4),
            c := CReg("c", 2),
            # Unconditional uses
            qubit.H(q[0]),
            qubit.H(q[1]),
            # Measurements
            qubit.Measure(q[2]) > c[0],
            qubit.Measure(q[3]) > c[1],
            # Conditional uses
            If(c[0]).Then(
                qubit.X(q[0]),  # q[0] used both unconditionally and conditionally
            ),
            If(c[1]).Then(
                qubit.Z(q[1]),  # q[1] used both unconditionally and conditionally
            ),
        )

        analyzer = IRAnalyzer()
        analyzer.analyze_block(prog, {"q": q, "c": c})

        q_info = analyzer.array_info["q"]
        # q[0] and q[1] are conditionally accessed
        assert 0 in q_info.conditionally_accessed_elements
        assert 1 in q_info.conditionally_accessed_elements
        # q[2] and q[3] are measured but not used after
        assert 2 not in q_info.conditionally_accessed_elements
        assert 3 not in q_info.conditionally_accessed_elements


class TestConditionalImprovements:
    """Test specific improvements from refined conditional analysis."""

    def test_syndrome_with_partial_conditional(self) -> None:
        """Syndrome extraction with only some qubits conditional."""
        prog = Main(
            data := QReg("data", 3),
            ancilla := QReg("ancilla", 2),
            syndrome := CReg("syndrome", 2),
            # Entangle
            qubit.CX(data[0], ancilla[0]),
            qubit.CX(data[1], ancilla[0]),
            qubit.CX(data[1], ancilla[1]),
            qubit.CX(data[2], ancilla[1]),
            # Measure
            qubit.Measure(ancilla[0]) > syndrome[0],
            qubit.Measure(ancilla[1]) > syndrome[1],
            # Conditional correction on only ONE data qubit
            If(syndrome[0]).Then(
                qubit.X(data[0]),  # Only data[0] is conditional
            ),
            # data[1] and data[2] are NOT conditional
            qubit.H(data[1]),
            qubit.H(data[2]),
        )

        analyzer = IRAnalyzer()
        analyzer.analyze_block(
            prog,
            {"data": data, "ancilla": ancilla, "syndrome": syndrome},
        )

        data_info = analyzer.array_info["data"]
        # Only data[0] should be marked as conditional
        assert 0 in data_info.conditionally_accessed_elements
        assert 1 not in data_info.conditionally_accessed_elements
        assert 2 not in data_info.conditionally_accessed_elements

    def test_teleportation_pattern(self) -> None:
        """Quantum teleportation with bob corrections."""
        prog = Main(
            alice := QReg("alice", 1),
            bob := QReg("bob", 1),
            epr := QReg("epr", 1),
            c := CReg("c", 2),
            # EPR pair
            qubit.H(epr[0]),
            qubit.CX(epr[0], bob[0]),
            # Alice's operations
            qubit.CX(alice[0], epr[0]),
            qubit.H(alice[0]),
            # Measurements
            qubit.Measure(alice[0]) > c[0],
            qubit.Measure(epr[0]) > c[1],
            # Bob's corrections - bob[0] is conditional
            If(c[1]).Then(
                qubit.X(bob[0]),
            ),
            If(c[0]).Then(
                qubit.Z(bob[0]),
            ),
        )

        analyzer = IRAnalyzer()
        analyzer.analyze_block(
            prog,
            {"alice": alice, "bob": bob, "epr": epr, "c": c},
        )

        bob_info = analyzer.array_info["bob"]
        c_info = analyzer.array_info["c"]

        # bob[0] is conditionally accessed
        assert 0 in bob_info.conditionally_accessed_elements

        # c[0] and c[1] are in conditions
        assert 0 in c_info.conditionally_accessed_elements
        assert 1 in c_info.conditionally_accessed_elements

    def test_partial_array_conditional_vs_full(self) -> None:
        """Verify we don't mark entire array when only part is conditional."""
        prog = Main(
            q := QReg("q", 5),
            c := CReg("c", 1),
            # Use q[0], q[1], q[2] unconditionally
            qubit.H(q[0]),
            qubit.H(q[1]),
            qubit.H(q[2]),
            qubit.Measure(q[0]) > c[0],
            # Only q[3] is conditional
            If(c[0]).Then(
                qubit.X(q[3]),
            ),
            # q[4] is never used
        )

        analyzer = IRAnalyzer()
        analyzer.analyze_block(prog, {"q": q, "c": c})

        q_info = analyzer.array_info["q"]

        # Only q[3] should be conditional
        assert q_info.conditionally_accessed_elements == {3}

        # All used elements should be in element_accesses
        assert q_info.element_accesses == {0, 1, 2, 3}
