"""Test suite for data flow analysis."""

from pecos.qeclib import qubit
from pecos.slr import CReg, If, Main, QReg
from pecos.slr.gen_codes.guppy.data_flow import DataFlowAnalyzer


class TestDataFlowBasics:
    """Test basic data flow tracking."""

    def test_simple_gate_no_measurement(self) -> None:
        """Gates without measurement don't require unpacking."""
        prog = Main(
            q := QReg("q", 3),
            qubit.H(q[0]),
            qubit.CX(q[0], q[1]),
            qubit.H(q[2]),
        )

        analyzer = DataFlowAnalyzer()
        analysis = analyzer.analyze(prog, {"q": q})

        # No measurements, so no unpacking needed
        assert not analysis.array_requires_unpacking("q")
        assert len(analysis.elements_requiring_unpacking()) == 0

    def test_measurement_only_no_reuse(self) -> None:
        """Measurement without reuse doesn't require unpacking."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qubit.H(q[0]),
            qubit.Measure(q[0]) > c[0],
            qubit.Measure(q[1]) > c[1],
        )

        analyzer = DataFlowAnalyzer()
        analysis = analyzer.analyze(prog, {"q": q, "c": c})

        # Measurements but no reuse after measurement
        assert not analysis.array_requires_unpacking("q")
        assert len(analysis.elements_requiring_unpacking()) == 0

    def test_measure_then_use_same_qubit(self) -> None:
        """Measuring then using the same qubit REQUIRES unpacking."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qubit.H(q[0]),
            qubit.Measure(q[0]) > c[0],
            qubit.X(q[0]),  # Use after measurement - requires unpacking!
        )

        analyzer = DataFlowAnalyzer()
        analysis = analyzer.analyze(prog, {"q": q, "c": c})

        # q[0] is used after measurement - requires unpacking
        assert analysis.array_requires_unpacking("q")
        requiring = analysis.elements_requiring_unpacking()
        assert ("q", 0) in requiring
        assert ("q", 1) not in requiring

    def test_measure_then_use_different_qubit(self) -> None:
        """Measuring one qubit then using a different qubit is fine."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qubit.H(q[0]),
            qubit.Measure(q[0]) > c[0],
            qubit.X(q[1]),  # Different qubit - no problem!
        )

        analyzer = DataFlowAnalyzer()
        analysis = analyzer.analyze(prog, {"q": q, "c": c})

        # q[0] measured, q[1] used - no unpacking needed
        assert not analysis.array_requires_unpacking("q")
        assert len(analysis.elements_requiring_unpacking()) == 0

    def test_measure_prep_then_use(self) -> None:
        """Measure, Prep (replacement), then use is OK."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qubit.H(q[0]),
            qubit.Measure(q[0]) > c[0],
            qubit.Prep(q[0]),  # Replacement
            qubit.H(q[0]),  # Use after replacement - OK!
        )

        analyzer = DataFlowAnalyzer()
        analysis = analyzer.analyze(prog, {"q": q, "c": c})

        # Replacement between measurement and use - no unpacking needed
        assert not analysis.array_requires_unpacking("q")
        assert len(analysis.elements_requiring_unpacking()) == 0


class TestDataFlowConditionals:
    """Test data flow with conditional operations."""

    def test_conditional_gate(self) -> None:
        """Conditional gates are tracked."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 1),
            qubit.Measure(q[0]) > c[0],
            If(c[0]).Then(
                qubit.X(q[1]),  # Conditional on measurement
            ),
        )

        analyzer = DataFlowAnalyzer()
        analysis = analyzer.analyze(prog, {"q": q, "c": c})

        # Conditional access is tracked
        assert ("c", 0) in analysis.conditional_accesses
        assert ("q", 1) in analysis.conditional_accesses

    def test_conditional_reset_pattern(self) -> None:
        """Common error correction pattern: measure, conditionally reset."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qubit.H(q[0]),
            qubit.Measure(q[0]) > c[0],
            If(c[0]).Then(
                qubit.X(q[0]),  # Conditional flip based on measurement
            ),
        )

        analyzer = DataFlowAnalyzer()
        analysis = analyzer.analyze(prog, {"q": q, "c": c})

        # q[0] is used conditionally after measurement
        assert analysis.array_requires_unpacking("q")
        assert ("q", 0) in analysis.elements_requiring_unpacking()


class TestDataFlowComplexPatterns:
    """Test complex data flow patterns."""

    def test_multiple_measurements_different_qubits(self) -> None:
        """Multiple measurements on different qubits."""
        prog = Main(
            q := QReg("q", 4),
            c := CReg("c", 4),
            # Measure qubits 0 and 1
            qubit.Measure(q[0]) > c[0],
            qubit.Measure(q[1]) > c[1],
            # Use qubits 2 and 3 (not measured)
            qubit.H(q[2]),
            qubit.CX(q[2], q[3]),
        )

        analyzer = DataFlowAnalyzer()
        analysis = analyzer.analyze(prog, {"q": q, "c": c})

        # No qubits used after their own measurement
        assert not analysis.array_requires_unpacking("q")
        assert len(analysis.elements_requiring_unpacking()) == 0

    def test_syndrome_extraction_pattern(self) -> None:
        """Syndrome extraction: measure ancillas, use data qubits."""
        prog = Main(
            data := QReg("data", 3),
            ancilla := QReg("ancilla", 2),
            syndrome := CReg("syndrome", 2),
            # Entangle
            qubit.CX(data[0], ancilla[0]),
            qubit.CX(data[1], ancilla[0]),
            qubit.CX(data[1], ancilla[1]),
            qubit.CX(data[2], ancilla[1]),
            # Measure ancillas
            qubit.Measure(ancilla[0]) > syndrome[0],
            qubit.Measure(ancilla[1]) > syndrome[1],
            # Continue using data qubits
            qubit.H(data[0]),
            qubit.H(data[1]),
            qubit.H(data[2]),
        )

        analyzer = DataFlowAnalyzer()
        analysis = analyzer.analyze(
            prog,
            {"data": data, "ancilla": ancilla, "syndrome": syndrome},
        )

        # Ancillas measured but not reused - no unpacking
        assert not analysis.array_requires_unpacking("ancilla")
        # Data qubits never measured - no unpacking
        assert not analysis.array_requires_unpacking("data")
        assert len(analysis.elements_requiring_unpacking()) == 0

    def test_repeated_measurement_cycle(self) -> None:
        """Repeated measurement cycles with replacement."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 3),
            # Cycle 1
            qubit.H(q[0]),
            qubit.Measure(q[0]) > c[0],
            qubit.Prep(q[0]),
            # Cycle 2
            qubit.H(q[0]),
            qubit.Measure(q[0]) > c[1],
            qubit.Prep(q[0]),
            # Cycle 3
            qubit.H(q[0]),
            qubit.Measure(q[0]) > c[2],
        )

        analyzer = DataFlowAnalyzer()
        analysis = analyzer.analyze(prog, {"q": q, "c": c})

        # Each measurement is followed by Prep before next use
        # No unpacking needed
        assert not analysis.array_requires_unpacking("q")
        assert len(analysis.elements_requiring_unpacking()) == 0

    def test_partial_qubit_reuse(self) -> None:
        """Some qubits reused after measurement, others not."""
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 3),
            # Measure all
            qubit.Measure(q[0]) > c[0],
            qubit.Measure(q[1]) > c[1],
            qubit.Measure(q[2]) > c[2],
            # Reuse only q[1]
            qubit.X(q[1]),  # This one needs unpacking!
        )

        analyzer = DataFlowAnalyzer()
        analysis = analyzer.analyze(prog, {"q": q, "c": c})

        # Only q[1] requires unpacking
        assert analysis.array_requires_unpacking("q")
        requiring = analysis.elements_requiring_unpacking()
        assert ("q", 0) not in requiring
        assert ("q", 1) in requiring
        assert ("q", 2) not in requiring


class TestDataFlowEdgeCases:
    """Test edge cases in data flow analysis."""

    def test_empty_program(self) -> None:
        """Empty program."""
        prog = Main(
            q := QReg("q", 2),
        )

        analyzer = DataFlowAnalyzer()
        analysis = analyzer.analyze(prog, {"q": q})

        assert not analysis.array_requires_unpacking("q")
        assert len(analysis.elements_requiring_unpacking()) == 0

    def test_measurement_without_storage(self) -> None:
        """Measurement without storing result."""
        prog = Main(
            q := QReg("q", 1),
            qubit.H(q[0]),
            qubit.Measure(q[0]),  # No classical storage
        )

        analyzer = DataFlowAnalyzer()
        analysis = analyzer.analyze(prog, {"q": q})

        # Measurement tracked even without classical storage
        flow = analysis.element_flows.get(("q", 0))
        assert flow is not None
        assert len(flow.consumed_at) == 1

    def test_use_before_and_after_measurement(self) -> None:
        """Use before and after measurement."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qubit.H(q[0]),  # Before
            qubit.X(q[0]),  # Before
            qubit.Measure(q[0]) > c[0],
            qubit.X(q[0]),  # After - requires unpacking!
        )

        analyzer = DataFlowAnalyzer()
        analysis = analyzer.analyze(prog, {"q": q, "c": c})

        # Uses before measurement are fine, but use after requires unpacking
        assert analysis.array_requires_unpacking("q")
        assert ("q", 0) in analysis.elements_requiring_unpacking()

        # Check that all uses are tracked
        flow = analysis.element_flows[("q", 0)]
        assert len(flow.uses) == 4  # H, X, Measure, X
