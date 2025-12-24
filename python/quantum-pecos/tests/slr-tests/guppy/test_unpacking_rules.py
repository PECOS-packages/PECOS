"""Test suite for array unpacking decision rules."""

from _pytest.capture import CaptureFixture
from pecos.slr.gen_codes.guppy.ir_analyzer import ArrayAccessInfo
from pecos.slr.gen_codes.guppy.unpacking_rules import (
    UnpackingDecision,
    UnpackingDecisionTree,
    UnpackingReason,
    should_unpack_array,
)


class TestUnpackingDecisionTree:
    """Test the rule-based decision tree for array unpacking."""

    def test_full_array_measurement_prevents_unpacking(self) -> None:
        """Full array measurements should prevent unpacking."""
        # Quantum array with full measurement
        info = ArrayAccessInfo(
            array_name="q",
            size=5,
            is_classical=False,
        )
        info.full_array_accesses.append(10)
        info.element_accesses.add(0)  # Also has individual access
        info.element_accesses.add(1)

        result = UnpackingDecisionTree().decide(info)
        assert result.decision == UnpackingDecision.MUST_NOT_UNPACK
        assert result.reason == UnpackingReason.FULL_ARRAY_ONLY
        assert not result.should_unpack

    def test_no_individual_access_no_unpacking(self) -> None:
        """Arrays with no individual element access should not be unpacked."""
        info = ArrayAccessInfo(
            array_name="q",
            size=5,
            is_classical=False,
        )
        # No element accesses

        result = UnpackingDecisionTree().decide(info)
        assert result.decision == UnpackingDecision.SHOULD_NOT_UNPACK
        assert result.reason == UnpackingReason.NO_INDIVIDUAL_ACCESS
        assert not result.should_unpack

    def test_operations_after_measurement_requires_unpacking(self) -> None:
        """Quantum operations after measurement require unpacking."""
        info = ArrayAccessInfo(
            array_name="q",
            size=3,
            is_classical=False,
        )
        info.element_accesses.add(0)
        info.elements_consumed.add(0)  # Measured
        info.has_operations_between = True  # Then used again

        result = UnpackingDecisionTree().decide(info)
        assert result.decision == UnpackingDecision.MUST_UNPACK
        assert result.reason == UnpackingReason.OPERATIONS_AFTER_MEASUREMENT
        assert result.should_unpack

    def test_individual_quantum_measurement_requires_unpacking(self) -> None:
        """Individual quantum measurements require unpacking."""
        info = ArrayAccessInfo(
            array_name="q",
            size=5,
            is_classical=False,
        )
        info.element_accesses.add(0)
        info.element_accesses.add(2)
        info.elements_consumed.add(0)
        info.elements_consumed.add(2)

        result = UnpackingDecisionTree().decide(info)
        assert result.decision == UnpackingDecision.MUST_UNPACK
        assert result.reason == UnpackingReason.INDIVIDUAL_QUANTUM_MEASUREMENT
        assert result.should_unpack

    def test_conditional_access_requires_unpacking(self) -> None:
        """Conditional element access requires unpacking."""
        info = ArrayAccessInfo(
            array_name="c",
            size=4,
            is_classical=True,
        )
        info.element_accesses.add(0)
        info.element_accesses.add(1)
        info.has_conditionals_between = True

        result = UnpackingDecisionTree().decide(info)
        assert result.decision == UnpackingDecision.MUST_UNPACK
        assert result.reason == UnpackingReason.CONDITIONAL_ELEMENT_ACCESS
        assert result.should_unpack

    def test_single_element_access_no_unpacking(self) -> None:
        """Single element access should use direct indexing, not unpacking."""
        info = ArrayAccessInfo(
            array_name="q",
            size=5,
            is_classical=False,
        )
        info.element_accesses.add(2)  # Only one element

        result = UnpackingDecisionTree().decide(info)
        assert result.decision == UnpackingDecision.SHOULD_NOT_UNPACK
        assert result.reason == UnpackingReason.SINGLE_ELEMENT_ONLY
        assert not result.should_unpack

    def test_classical_multiple_accesses_should_unpack(self) -> None:
        """Classical arrays with multiple individual accesses should unpack for clarity."""
        info = ArrayAccessInfo(
            array_name="c",
            size=4,
            is_classical=True,
        )
        info.element_accesses.add(0)
        info.element_accesses.add(1)
        info.element_accesses.add(2)

        result = UnpackingDecisionTree().decide(info)
        assert result.decision == UnpackingDecision.SHOULD_UNPACK
        assert result.reason == UnpackingReason.MULTIPLE_INDIVIDUAL_ACCESSES
        assert result.should_unpack

    def test_partial_array_high_ratio_should_unpack(self) -> None:
        """Partial array usage with high access ratio should unpack."""
        info = ArrayAccessInfo(
            array_name="q",
            size=5,
            is_classical=False,
        )
        # Access 3 of 5 elements (60%)
        info.element_accesses.add(0)
        info.element_accesses.add(2)
        info.element_accesses.add(4)

        # Note: This won't trigger quantum measurement rule since no elements consumed
        result = UnpackingDecisionTree().decide(info)
        assert result.decision == UnpackingDecision.SHOULD_UNPACK
        assert result.reason == UnpackingReason.PARTIAL_ARRAY_USAGE
        assert result.should_unpack

    def test_partial_array_low_ratio_should_not_unpack(self) -> None:
        """Partial array usage with low access ratio should not unpack."""
        info = ArrayAccessInfo(
            array_name="q",
            size=10,
            is_classical=False,
        )
        # Access only 2 of 10 elements (20%)
        info.element_accesses.add(0)
        info.element_accesses.add(5)

        result = UnpackingDecisionTree().decide(info)
        # Could be SINGLE_ELEMENT_ONLY or PARTIAL_ARRAY_USAGE depending on rule order
        assert result.decision == UnpackingDecision.SHOULD_NOT_UNPACK
        assert not result.should_unpack

    def test_convenience_function_verbose(self, capsys: CaptureFixture[str]) -> None:
        """Test the convenience function with verbose output."""
        info = ArrayAccessInfo(
            array_name="test_array",
            size=3,
            is_classical=True,
        )
        info.element_accesses.add(0)
        info.element_accesses.add(1)

        result = should_unpack_array(info, verbose=True)
        assert result is True

        captured = capsys.readouterr()
        assert "test_array" in captured.out
        assert "SHOULD_UNPACK" in captured.out
        assert "MULTIPLE_INDIVIDUAL_ACCESSES" in captured.out


class TestRealWorldScenarios:
    """Test realistic scenarios from actual SLR code."""

    def test_simple_quantum_circuit_no_measurement(self) -> None:
        """Simple circuit with gates only, no measurements - should not unpack."""
        # Example: H(q[0]); CX(q[0], q[1]); H(q[1])
        info = ArrayAccessInfo(
            array_name="q",
            size=2,
            is_classical=False,
        )
        info.element_accesses.add(0)
        info.element_accesses.add(1)
        # No measurements, no consumption
        # All elements accessed (100%), no special conditions

        result = UnpackingDecisionTree().decide(info)
        # Should NOT unpack - all elements accessed, can use array operations
        # The default behavior prefers simpler code (no unpacking)
        assert not result.should_unpack

    def test_measure_all_qubits_into_classical(self) -> None:
        """Measure entire quantum register into classical register."""
        # Example: Measure(q) > c
        q_info = ArrayAccessInfo(
            array_name="q",
            size=5,
            is_classical=False,
        )
        q_info.full_array_accesses.append(10)

        c_info = ArrayAccessInfo(
            array_name="c",
            size=5,
            is_classical=True,
        )
        # Classical register receives results but no individual access

        q_result = UnpackingDecisionTree().decide(q_info)
        c_result = UnpackingDecisionTree().decide(c_info)

        assert not q_result.should_unpack  # Full array measurement
        assert not c_result.should_unpack  # No individual access

    def test_measure_individual_qubits(self) -> None:
        """Measure individual qubits separately."""
        # Example: Measure(q[0]) > c[0]; Measure(q[1]) > c[1]
        q_info = ArrayAccessInfo(
            array_name="q",
            size=3,
            is_classical=False,
        )
        q_info.element_accesses.add(0)
        q_info.element_accesses.add(1)
        q_info.elements_consumed.add(0)
        q_info.elements_consumed.add(1)

        c_info = ArrayAccessInfo(
            array_name="c",
            size=3,
            is_classical=True,
        )
        c_info.element_accesses.add(0)
        c_info.element_accesses.add(1)

        q_result = UnpackingDecisionTree().decide(q_info)
        c_result = UnpackingDecisionTree().decide(c_info)

        assert q_result.should_unpack  # Individual quantum measurements
        assert c_result.should_unpack  # Multiple classical accesses

    def test_conditional_reset_pattern(self) -> None:
        """Conditional reset based on measurement - common error correction pattern."""
        # Example: m = Measure(q[0]) > c[0]; if c[0]: X(q[0])
        info = ArrayAccessInfo(
            array_name="c",
            size=1,
            is_classical=True,
        )
        info.element_accesses.add(0)
        info.has_conditionals_between = True

        result = UnpackingDecisionTree().decide(info)
        assert result.should_unpack  # Conditional access requires unpacking

    def test_measure_then_replace_pattern(self) -> None:
        """Measure qubit, then replace with fresh qubit - needs unpacking."""
        # Example: Measure(q[0]); Prep(q[0]); H(q[0])
        info = ArrayAccessInfo(
            array_name="q",
            size=3,
            is_classical=False,
        )
        info.element_accesses.add(0)
        info.elements_consumed.add(0)
        info.has_operations_between = True

        result = UnpackingDecisionTree().decide(info)
        assert result.should_unpack  # Operations after measurement

    def test_partial_measurement_syndrome_extraction(self) -> None:
        """Measure ancilla qubits for syndrome extraction, keep data qubits."""
        # Example: ancilla = q[5:10]; Measure(ancilla) > syndrome
        # Data qubits q[0:5] still used
        ancilla_info = ArrayAccessInfo(
            array_name="ancilla",
            size=5,
            is_classical=False,
        )
        ancilla_info.full_array_accesses.append(20)

        syndrome_info = ArrayAccessInfo(
            array_name="syndrome",
            size=5,
            is_classical=True,
        )
        # No individual access

        ancilla_result = UnpackingDecisionTree().decide(ancilla_info)
        syndrome_result = UnpackingDecisionTree().decide(syndrome_info)

        assert not ancilla_result.should_unpack  # Full array measurement
        assert not syndrome_result.should_unpack  # No individual access

    def test_steane_code_pattern(self) -> None:
        """Steane code with data and ancilla qubits."""
        # Data qubits: individual gates
        data_info = ArrayAccessInfo(
            array_name="data",
            size=7,
            is_classical=False,
        )
        for i in range(7):
            data_info.element_accesses.add(i)

        # Ancilla qubits: measured individually
        ancilla_info = ArrayAccessInfo(
            array_name="ancilla",
            size=6,
            is_classical=False,
        )
        for i in range(6):
            ancilla_info.element_accesses.add(i)
            ancilla_info.elements_consumed.add(i)

        data_result = UnpackingDecisionTree().decide(data_info)
        ancilla_result = UnpackingDecisionTree().decide(ancilla_info)

        # Data: all elements accessed with no special conditions - default is no unpack
        # (can use array operations efficiently)
        assert not data_result.should_unpack
        # Ancilla: individual measurements REQUIRE unpacking
        assert ancilla_result.should_unpack


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_array(self) -> None:
        """Array of size 0."""
        info = ArrayAccessInfo(
            array_name="empty",
            size=0,
            is_classical=False,
        )

        result = UnpackingDecisionTree().decide(info)
        assert not result.should_unpack

    def test_size_one_array(self) -> None:
        """Array of size 1."""
        info = ArrayAccessInfo(
            array_name="single",
            size=1,
            is_classical=False,
        )
        info.element_accesses.add(0)

        result = UnpackingDecisionTree().decide(info)
        # Single element access should not unpack
        assert not result.should_unpack
        assert result.reason == UnpackingReason.SINGLE_ELEMENT_ONLY

    def test_conflicting_indicators(self) -> None:
        """Array with both full access and individual access."""
        info = ArrayAccessInfo(
            array_name="conflict",
            size=3,
            is_classical=False,
        )
        info.full_array_accesses.append(5)  # Full access at position 5
        info.element_accesses.add(0)  # Individual access
        info.element_accesses.add(1)

        result = UnpackingDecisionTree().decide(info)
        # Full array access takes precedence (MUST_NOT_UNPACK)
        assert not result.should_unpack
        assert result.reason == UnpackingReason.FULL_ARRAY_ONLY

    def test_all_elements_individually_accessed(self) -> None:
        """All array elements accessed individually."""
        info = ArrayAccessInfo(
            array_name="all",
            size=4,
            is_classical=True,
        )
        for i in range(4):
            info.element_accesses.add(i)

        result = UnpackingDecisionTree().decide(info)
        # Multiple individual accesses on classical array
        assert result.should_unpack
        assert result.reason == UnpackingReason.MULTIPLE_INDIVIDUAL_ACCESSES

    def test_exactly_50_percent_access_ratio(self) -> None:
        """Test boundary at 50% access ratio."""
        info = ArrayAccessInfo(
            array_name="half",
            size=4,
            is_classical=False,
        )
        info.element_accesses.add(0)
        info.element_accesses.add(1)
        # 2 of 4 = 50%

        result = UnpackingDecisionTree().decide(info)
        # Should not unpack at exactly 50% (threshold is > 0.5)
        assert not result.should_unpack
