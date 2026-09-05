"""Test suite for advanced type support (futures, collections, etc)."""

import re

import pecos as pc
import pecos_rslib_llvm
from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import h, measure, qubit, x


class TestAdvancedTypes:
    """Test advanced type support."""

    def test_basic_measurement_future(self) -> None:
        """Test that measurement operations work (which use futures internally)."""

        @guppy
        def test_measure_future() -> bool:
            q = qubit()
            h(q)
            # Measurement returns a future internally in the HUGR
            return measure(q).read()

        hugr = test_measure_future.compile()
        output = pecos_rslib_llvm.compile_hugr_to_qis(hugr.to_bytes())

        # Should compile successfully
        assert "___lazy_measure" in output
        assert "qmain" in output

    def test_multiple_measurements(self) -> None:
        """Test multiple measurements (multiple futures)."""

        @guppy
        def test_multi_measure() -> tuple[bool, bool]:
            q1 = qubit()
            q2 = qubit()
            h(q1)
            h(q2)
            result1 = measure(q1).read()
            result2 = measure(q2).read()
            return result1, result2

        hugr = test_multi_measure.compile()
        output = pecos_rslib_llvm.compile_hugr_to_qis(hugr.to_bytes())

        # Should handle multiple futures correctly
        measure_calls = re.findall(r"\bcall\b[^\n]*@___lazy_measure\(", output)
        assert len(measure_calls) == 2, f"Expected 2 measurement calls, got {len(measure_calls)}"

    def test_advanced_types_compilation(self) -> None:
        """Test that advanced types don't break compilation."""

        @guppy
        def test_advanced() -> bool:
            q = qubit()
            h(q)
            # This will involve futures and potentially other advanced types
            return measure(q).read()

        hugr = test_advanced.compile()
        pecos_out = pecos_rslib_llvm.compile_hugr_to_qis(hugr.to_bytes())

        # Should compile successfully
        assert len(pecos_out) > 100
        # The return type could be i32 (for bool) or i64 depending on compiler version
        assert "define i32 @qmain" in pecos_out or "define i64 @qmain" in pecos_out

    def test_measurement_futures_read_out_of_order_on_selene(self) -> None:
        """Delayed reads retain the outcome of their own measurement."""

        @guppy
        def delayed_reads() -> None:
            q0 = qubit()
            q1 = qubit()
            x(q1)
            first = measure(q0)
            second = measure(q1)
            result("second", second.read())
            result("first", first.read())

        results = (
            pc.sim(delayed_reads)
            .classical(pc.selene_engine())
            .quantum(pc.state_vector())
            .qubits(2)
            .seed(42)
            .run(8)
            .to_dict()
        )
        assert results["first"] == [0] * 8
        assert results["second"] == [1] * 8

    def test_complex_quantum_program(self) -> None:
        """Test complex program that might use advanced types."""

        @guppy
        def test_complex() -> tuple[bool, bool, bool]:
            # Create a more complex program that might use advanced types
            q1 = qubit()
            q2 = qubit()
            q3 = qubit()

            h(q1)
            h(q2)
            h(q3)

            # Multiple measurements create multiple futures
            r1 = measure(q1).read()
            r2 = measure(q2).read()
            r3 = measure(q3).read()

            return r1, r2, r3

        hugr = test_complex.compile()
        output = pecos_rslib_llvm.compile_hugr_to_qis(hugr.to_bytes())

        # Should handle the complex program correctly
        assert "___qalloc" in output
        assert "___lazy_measure" in output
        assert "___qfree" in output
