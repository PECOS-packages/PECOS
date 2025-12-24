"""Test multi-qubit measurement support in Guppy IR builder."""

from pecos.qeclib import qubit
from pecos.slr import Block, CReg, QReg
from pecos.slr.slr_converter import SlrConverter


class MultiQubitMeasureWithOutputs(Block):
    """Test block with multi-qubit measurement and classical outputs."""

    def __init__(self, q: QReg, c: CReg) -> None:
        """Measure multiple qubits into classical bits.

        Args:
            q: Quantum register with 3 qubits
            c: Classical register with 3 bits
        """
        super().__init__()
        self.extend(
            # Multi-qubit measurement with classical outputs
            qubit.Measure(q[0], q[1], q[2])
            > (c[0], c[1], c[2]),
        )


class MultiQubitMeasureWithoutOutputs(Block):
    """Test block with multi-qubit measurement but no classical outputs."""

    def __init__(self, q: QReg) -> None:
        """Measure multiple qubits without storing results.

        Args:
            q: Quantum register with 3 qubits
        """
        super().__init__()
        self.extend(
            # Multi-qubit measurement without classical outputs
            qubit.Measure(q[0], q[1], q[2]),
        )


class MixedMeasurements(Block):
    """Test block with both single and multi-qubit measurements."""

    def __init__(self, q: QReg, c: CReg) -> None:
        """Mix of single and multi-qubit measurements.

        Args:
            q: Quantum register with 5 qubits
            c: Classical register with 5 bits
        """
        super().__init__()
        self.extend(
            # Single qubit measurement
            qubit.Measure(q[0]) > c[0],
            # Multi-qubit measurement
            qubit.Measure(q[1], q[2], q[3]) > (c[1], c[2], c[3]),
            # Another single measurement
            qubit.Measure(q[4]) > c[4],
        )


class MismatchedMeasurement(Block):
    """Test block with mismatched qubit/output counts (should generate error comment)."""

    def __init__(self, q: QReg, c: CReg) -> None:
        """Intentionally mismatched measurement.

        Args:
            q: Quantum register with 3 qubits
            c: Classical register with 2 bits (intentional mismatch)
        """
        super().__init__()
        # This creates a measurement with 3 qubits but only 2 outputs
        # In practice, this might not be possible due to PECOS validation,
        # but we test the IR builder's handling
        meas = qubit.Measure(q[0], q[1], q[2])
        meas.cout = (c[0], c[1])  # Manually set mismatched outputs
        self.extend(meas)


class TestMultiQubitMeasurements:
    """Test multi-qubit measurement IR generation."""

    def test_multi_qubit_with_outputs(self) -> None:
        """Test that multi-qubit measurements with outputs generate multiple IR measurement nodes."""
        q = QReg("q", 3)
        c = CReg("c", 3)
        block = MultiQubitMeasureWithOutputs(q, c)

        # Convert to Guppy
        guppy_code = SlrConverter(block).guppy()

        # Should generate three separate measurement statements
        assert "quantum.measure(" in guppy_code or "measure(" in guppy_code

        # Should have three measurements total
        assert guppy_code.count("measure(") >= 3

        # Should have array subscript references for qubits
        assert "q[0]" in guppy_code
        assert "q[1]" in guppy_code
        assert "q[2]" in guppy_code

        # Should reference classical bits
        assert "c[0]" in guppy_code
        assert "c[1]" in guppy_code
        assert "c[2]" in guppy_code

        # Should not have TODO or error comments
        assert "TODO" not in guppy_code
        assert "ERROR" not in guppy_code

    def test_multi_qubit_without_outputs(self) -> None:
        """Test that multi-qubit measurements without outputs are handled."""
        q = QReg("q", 3)
        block = MultiQubitMeasureWithoutOutputs(q)

        # Convert to Guppy
        guppy_code = SlrConverter(block).guppy()

        # Should generate measurement statements
        assert "measure(" in guppy_code

        # Should have three measurements
        assert guppy_code.count("measure(") >= 3

        # Should reference qubits
        assert "q[0]" in guppy_code
        assert "q[1]" in guppy_code
        assert "q[2]" in guppy_code

        # Should not have TODO or error comments
        assert "TODO" not in guppy_code
        assert "ERROR" not in guppy_code

    def test_mixed_measurements(self) -> None:
        """Test that single and multi-qubit measurements can coexist."""
        q = QReg("q", 5)
        c = CReg("c", 5)
        block = MixedMeasurements(q, c)

        # Convert to Guppy
        guppy_code = SlrConverter(block).guppy()

        # Should generate measurement statements
        assert "measure(" in guppy_code

        # Should have 5 measurements (1 single + 3 multi + 1 single)
        assert guppy_code.count("measure(") >= 5

        # Should reference all qubits
        for i in range(5):
            assert f"q[{i}]" in guppy_code

        # Should reference all classical bits
        for i in range(5):
            assert f"c[{i}]" in guppy_code

        # Should not have TODO or error comments
        assert "TODO" not in guppy_code
        assert "ERROR" not in guppy_code

    def test_resource_consumption(self) -> None:
        """Test that multi-qubit measurements properly track consumed qubits."""
        q = QReg("q", 3)
        c = CReg("c", 3)
        block = MultiQubitMeasureWithOutputs(q, c)

        # Convert to Guppy - should succeed without linearity errors
        guppy_code = SlrConverter(block).guppy()

        # Should not have error messages about unconsumed resources
        assert "ERROR" not in guppy_code
        assert "not all variables consumed" not in guppy_code.lower()


class TestMultiQubitMeasurementEdgeCases:
    """Test edge cases in multi-qubit measurement handling."""

    def test_two_qubit_measurement(self) -> None:
        """Test measurement with exactly two qubits."""

        class TwoQubitMeasure(Block):
            def __init__(self, q: QReg, c: CReg) -> None:
                super().__init__()
                self.extend(qubit.Measure(q[0], q[1]) > (c[0], c[1]))

        q = QReg("q", 2)
        c = CReg("c", 2)
        block = TwoQubitMeasure(q, c)

        guppy_code = SlrConverter(block).guppy()

        # Should handle 2-qubit case correctly
        assert "measure(" in guppy_code
        assert guppy_code.count("measure(") >= 2
        assert "TODO" not in guppy_code
        assert "ERROR" not in guppy_code

    def test_many_qubit_measurement(self) -> None:
        """Test measurement with many qubits (stress test)."""

        class ManyQubitMeasure(Block):
            def __init__(self, q: QReg, c: CReg) -> None:
                super().__init__()
                # Measure 7 qubits
                self.extend(
                    qubit.Measure(q[0], q[1], q[2], q[3], q[4], q[5], q[6])
                    > (c[0], c[1], c[2], c[3], c[4], c[5], c[6]),
                )

        q = QReg("q", 7)
        c = CReg("c", 7)
        block = ManyQubitMeasure(q, c)

        guppy_code = SlrConverter(block).guppy()

        # Should handle many qubits correctly
        assert "measure(" in guppy_code
        assert guppy_code.count("measure(") >= 7
        assert "TODO" not in guppy_code
        assert "ERROR" not in guppy_code

        # Should reference all 7 qubits
        for i in range(7):
            assert f"q[{i}]" in guppy_code


class TestSingleQubitMeasurementRegression:
    """Ensure single-qubit measurements still work correctly."""

    def test_single_qubit_with_output(self) -> None:
        """Test that single-qubit measurement with output still works."""

        class SingleMeasure(Block):
            def __init__(self, q: QReg, c: CReg) -> None:
                super().__init__()
                self.extend(qubit.Measure(q[0]) > c[0])

        q = QReg("q", 1)
        c = CReg("c", 1)
        block = SingleMeasure(q, c)

        guppy_code = SlrConverter(block).guppy()

        # Should generate measurement
        assert "measure(" in guppy_code
        assert "TODO" not in guppy_code
        assert "ERROR" not in guppy_code

    def test_single_qubit_without_output(self) -> None:
        """Test that single-qubit measurement without output still works."""

        class SingleMeasureNoOutput(Block):
            def __init__(self, q: QReg) -> None:
                super().__init__()
                self.extend(qubit.Measure(q[0]))

        q = QReg("q", 1)
        block = SingleMeasureNoOutput(q)

        guppy_code = SlrConverter(block).guppy()

        # Should generate measurement
        assert "measure(" in guppy_code
        assert "TODO" not in guppy_code
        assert "ERROR" not in guppy_code
