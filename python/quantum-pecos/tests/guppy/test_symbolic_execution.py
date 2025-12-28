# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Tests for the Guppy -> HUGR -> symbolic execution -> sampling pipeline."""

from __future__ import annotations

import pytest
from guppylang import guppy
from guppylang.std.quantum import cx, cy, cz, h, measure, qubit, s, x, y, z
from pecos.experimental import (
    SymbolicExecutionResult,
    execute_dag_circuit_symbolic,
    execute_hugr_symbolic,
)
from pecos_rslib import hugr_to_dag_circuit


def outcome_to_tuple(outcome: bytes) -> tuple[bool, ...]:
    """Convert bytes outcome to tuple of bools for easier assertion."""
    return tuple(bool(b) for b in outcome)


class TestDebuggingWindowsCrash:
    """Debugging tests to isolate Windows heap corruption.

    These tests break down the pipeline into smaller steps to identify
    exactly where the crash occurs.
    """

    def test_step1_hugr_execution_only(self) -> None:
        """Test just HUGR execution without any sampling."""

        @guppy
        def no_gate() -> bool:
            q = qubit()
            return measure(q)

        result = execute_hugr_symbolic(no_gate.compile().to_bytes())
        print(f"Step 1: HUGR execution succeeded: {result}")
        assert result.num_measurements == 1
        assert result.num_deterministic == 1

    def test_step2_sample_single_shot(self) -> None:
        """Test sample() with just 1 shot."""

        @guppy
        def no_gate() -> bool:
            q = qubit()
            return measure(q)

        result = execute_hugr_symbolic(no_gate.compile().to_bytes())
        print(f"Step 2a: Result created: {result}")

        samples = result.sample(1)
        print(f"Step 2b: sample(1) returned: {samples}")
        assert len(samples) == 1

    def test_step3_sample_counts_single_shot(self) -> None:
        """Test sample_counts() with just 1 shot."""

        @guppy
        def no_gate() -> bool:
            q = qubit()
            return measure(q)

        result = execute_hugr_symbolic(no_gate.compile().to_bytes())
        print(f"Step 3a: Result created: {result}")

        counts = result.sample_counts(1)
        print(f"Step 3b: sample_counts(1) returned: {counts}")
        assert len(counts) == 1

    def test_step4_sample_counts_64_shots(self) -> None:
        """Test sample_counts() with 64 shots (one u64 word)."""

        @guppy
        def no_gate() -> bool:
            q = qubit()
            return measure(q)

        result = execute_hugr_symbolic(no_gate.compile().to_bytes())
        counts = result.sample_counts(64)
        print(f"Step 4: sample_counts(64) returned: {counts}")
        assert counts[b"\x00"] == 64

    def test_step5_sample_counts_256_shots(self) -> None:
        """Test sample_counts() with 256 shots (one u64x4 SIMD word)."""

        @guppy
        def no_gate() -> bool:
            q = qubit()
            return measure(q)

        result = execute_hugr_symbolic(no_gate.compile().to_bytes())
        counts = result.sample_counts(256)
        print(f"Step 5: sample_counts(256) returned: {counts}")
        assert counts[b"\x00"] == 256

    def test_step6_sample_counts_1000_shots(self) -> None:
        """Test sample_counts() with 1000 shots (the failing case)."""

        @guppy
        def no_gate() -> bool:
            q = qubit()
            return measure(q)

        result = execute_hugr_symbolic(no_gate.compile().to_bytes())
        counts = result.sample_counts(1000)
        print(f"Step 6: sample_counts(1000) returned: {counts}")
        assert counts[b"\x00"] == 1000


class TestBasicSymbolicExecution:
    """Tests for basic Guppy -> HUGR -> symbolic execution."""

    def test_single_qubit_h_measure(self) -> None:
        """Test single qubit with H gate - should be random."""

        @guppy
        def single_h() -> bool:
            q = qubit()
            h(q)
            return measure(q)

        result = execute_hugr_symbolic(single_h.compile().to_bytes())

        assert isinstance(result, SymbolicExecutionResult)
        assert result.num_measurements == 1
        assert result.num_nondeterministic == 1
        assert result.num_deterministic == 0

    def test_single_qubit_no_gate(self) -> None:
        """Test single qubit with no gates - should be deterministic 0."""

        @guppy
        def no_gate() -> bool:
            q = qubit()
            return measure(q)

        result = execute_hugr_symbolic(no_gate.compile().to_bytes())

        assert result.num_measurements == 1
        assert result.num_deterministic == 1
        assert result.num_nondeterministic == 0

        # All samples should be False (0)
        counts = result.sample_counts(1000)
        assert len(counts) == 1
        assert b"\x00" in counts
        assert counts[b"\x00"] == 1000

    def test_bell_state(self) -> None:
        """Test Bell state - two correlated measurements."""

        @guppy
        def bell_state() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            return (measure(q0), measure(q1))

        result = execute_hugr_symbolic(bell_state.compile().to_bytes())

        assert result.num_measurements == 2
        assert result.num_nondeterministic == 1  # Only one random bit
        assert result.num_deterministic == 1  # Second is correlated to first

        # Sample and verify correlations
        counts = result.sample_counts(10000)
        assert len(counts) == 2  # Only |00> and |11>
        assert b"\x00\x00" in counts
        assert b"\x01\x01" in counts
        assert b"\x00\x01" not in counts
        assert b"\x01\x00" not in counts

    def test_ghz_state(self) -> None:
        """Test 3-qubit GHZ state - all measurements correlated."""

        @guppy
        def ghz_state() -> tuple[bool, bool, bool]:
            q0 = qubit()
            q1 = qubit()
            q2 = qubit()
            h(q0)
            cx(q0, q1)
            cx(q1, q2)
            return (measure(q0), measure(q1), measure(q2))

        result = execute_hugr_symbolic(ghz_state.compile().to_bytes())

        assert result.num_measurements == 3
        assert result.num_nondeterministic == 1  # Only one random bit
        assert result.num_deterministic == 2  # Two correlated

        # Sample and verify only |000> and |111>
        counts = result.sample_counts(10000)
        assert len(counts) == 2
        assert b"\x00\x00\x00" in counts
        assert b"\x01\x01\x01" in counts


class TestTwoQubitGates:
    """Tests for two-qubit gates in symbolic execution."""

    def test_cx_gate(self) -> None:
        """Test CX gate creates proper correlations."""

        @guppy
        def cx_circuit() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            return (measure(q0), measure(q1))

        result = execute_hugr_symbolic(cx_circuit.compile().to_bytes())
        counts = result.sample_counts(10000)

        # Bell state: only |00> and |11>
        assert len(counts) == 2
        assert b"\x00\x00" in counts
        assert b"\x01\x01" in counts

    def test_cz_gate(self) -> None:
        """Test CZ gate works correctly.

        CZ = H_target . CX . H_target, so we test:
        H(q0) -> CZ(q0,q1) is equivalent to H(q0) -> H(q1) -> CX(q0,q1) -> H(q1)
        which creates a Bell state.
        """

        @guppy
        def cz_bell() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            h(q1)  # Prepare target in |+>
            cz(q0, q1)  # CZ creates phase correlation
            h(q1)  # Convert phase to amplitude correlation
            return (measure(q0), measure(q1))

        result = execute_hugr_symbolic(cz_bell.compile().to_bytes())

        # This creates Bell-like correlations
        counts = result.sample_counts(10000)
        assert len(counts) == 2
        assert b"\x00\x00" in counts
        assert b"\x01\x01" in counts

    def test_cy_gate(self) -> None:
        """Test CY gate works correctly."""

        @guppy
        def cy_circuit() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cy(q0, q1)
            return (measure(q0), measure(q1))

        result = execute_hugr_symbolic(cy_circuit.compile().to_bytes())

        # CY also creates correlations like CX
        counts = result.sample_counts(10000)
        assert len(counts) == 2


class TestSingleQubitGates:
    """Tests for single-qubit Clifford gates."""

    def test_pauli_gates(self) -> None:
        """Test X, Y, Z gates."""

        @guppy
        def x_gate() -> bool:
            q = qubit()
            x(q)
            return measure(q)

        result = execute_hugr_symbolic(x_gate.compile().to_bytes())
        counts = result.sample_counts(100)
        # X flips |0> to |1>
        assert counts == {b"\x01": 100}

        @guppy
        def z_gate() -> bool:
            q = qubit()
            z(q)
            return measure(q)

        result = execute_hugr_symbolic(z_gate.compile().to_bytes())
        counts = result.sample_counts(100)
        # Z on |0> is still |0>
        assert counts == {b"\x00": 100}

        @guppy
        def y_gate() -> bool:
            q = qubit()
            y(q)
            return measure(q)

        result = execute_hugr_symbolic(y_gate.compile().to_bytes())
        counts = result.sample_counts(100)
        # Y flips |0> to i|1>
        assert counts == {b"\x01": 100}

    def test_s_gate(self) -> None:
        """Test S gate (phase gate)."""

        @guppy
        def s_gate() -> bool:
            q = qubit()
            h(q)
            s(q)
            h(q)
            return measure(q)

        result = execute_hugr_symbolic(s_gate.compile().to_bytes())
        # H-S-H is equivalent to sqrt(X), deterministic
        assert result.num_measurements == 1


class TestSamplingMethods:
    """Tests for sampling methods on SymbolicExecutionResult."""

    def test_sample_returns_list(self) -> None:
        """Test that sample() returns a list of lists."""

        @guppy
        def bell() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            return (measure(q0), measure(q1))

        result = execute_hugr_symbolic(bell.compile().to_bytes())
        samples = result.sample(10)

        assert isinstance(samples, list)
        assert len(samples) == 10
        for sample in samples:
            assert isinstance(sample, list)
            assert len(sample) == 2
            assert all(isinstance(b, bool) for b in sample)

    def test_sample_counts_returns_dict(self) -> None:
        """Test that sample_counts() returns a dict."""

        @guppy
        def bell() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            return (measure(q0), measure(q1))

        result = execute_hugr_symbolic(bell.compile().to_bytes())
        counts = result.sample_counts(1000)

        assert isinstance(counts, dict)
        total = sum(counts.values())
        assert total == 1000

    def test_large_sample_count(self) -> None:
        """Test that large sample counts work efficiently."""

        @guppy
        def ghz() -> tuple[bool, bool, bool]:
            q0 = qubit()
            q1 = qubit()
            q2 = qubit()
            h(q0)
            cx(q0, q1)
            cx(q1, q2)
            return (measure(q0), measure(q1), measure(q2))

        result = execute_hugr_symbolic(ghz.compile().to_bytes())

        # Should handle 1M samples without issue
        counts = result.sample_counts(1_000_000)
        total = sum(counts.values())
        assert total == 1_000_000
        assert len(counts) == 2  # Only |000> and |111>


class TestMeasurementStructure:
    """Tests for measurement structure properties."""

    def test_deterministic_count(self) -> None:
        """Test that deterministic measurement count is correct."""

        @guppy
        def all_deterministic() -> tuple[bool, bool, bool]:
            q0 = qubit()
            q1 = qubit()
            q2 = qubit()
            x(q0)  # Flip to |1>
            # q1 stays |0>
            x(q2)  # Flip to |1>
            return (measure(q0), measure(q1), measure(q2))

        result = execute_hugr_symbolic(all_deterministic.compile().to_bytes())

        assert result.num_measurements == 3
        assert result.num_deterministic == 3
        assert result.num_nondeterministic == 0

        counts = result.sample_counts(100)
        assert counts == {b"\x01\x00\x01": 100}

    def test_nondeterministic_count(self) -> None:
        """Test that non-deterministic measurement count is correct."""

        @guppy
        def all_random() -> tuple[bool, bool, bool]:
            q0 = qubit()
            q1 = qubit()
            q2 = qubit()
            h(q0)
            h(q1)
            h(q2)
            return (measure(q0), measure(q1), measure(q2))

        result = execute_hugr_symbolic(all_random.compile().to_bytes())

        assert result.num_measurements == 3
        assert result.num_nondeterministic == 3
        assert result.num_deterministic == 0

        # Should have all 8 outcomes
        counts = result.sample_counts(10000)
        assert len(counts) == 8

    def test_mixed_deterministic_nondeterministic(self) -> None:
        """Test circuit with mix of deterministic and random measurements."""

        @guppy
        def mixed() -> tuple[bool, bool, bool]:
            q0 = qubit()
            q1 = qubit()
            q2 = qubit()
            h(q0)  # Random
            x(q1)  # Deterministic |1>
            # q2 stays |0> - deterministic
            return (measure(q0), measure(q1), measure(q2))

        result = execute_hugr_symbolic(mixed.compile().to_bytes())

        assert result.num_measurements == 3
        assert result.num_nondeterministic == 1
        assert result.num_deterministic == 2

        counts = result.sample_counts(1000)
        assert len(counts) == 2
        # q1=True, q2=False always; q0 varies
        assert b"\x00\x01\x00" in counts
        assert b"\x01\x01\x00" in counts


class TestRepetitionCode:
    """Tests for repetition code syndrome extraction."""

    def test_repetition_code_no_errors(self) -> None:
        """Test 3-qubit repetition code with CX-based syndrome extraction."""

        @guppy
        def repetition_code() -> tuple[bool, bool, bool, bool, bool]:
            d0 = qubit()
            d1 = qubit()
            d2 = qubit()
            a0 = qubit()
            a1 = qubit()

            # Encode logical |+_L>
            h(d0)
            cx(d0, d1)
            cx(d0, d2)

            # Syndrome Z0Z1 using CX gates
            cx(d0, a0)
            cx(d1, a0)
            s0 = measure(a0)

            # Syndrome Z1Z2 using CX gates
            cx(d1, a1)
            cx(d2, a1)
            s1 = measure(a1)

            return (s0, s1, measure(d0), measure(d1), measure(d2))

        result = execute_hugr_symbolic(repetition_code.compile().to_bytes())

        assert result.num_measurements == 5

        counts = result.sample_counts(10000)

        # With no errors, syndromes should be 00
        # Data qubits should all be same (000 or 111)
        for outcome in counts:
            s0, s1, d0, d1, d2 = outcome  # bytes unpack to ints
            assert s0 == 0, f"Expected s0=0, got {outcome}"
            assert s1 == 0, f"Expected s1=0, got {outcome}"
            assert d0 == d1 == d2, f"Data qubits should match: {outcome}"


class TestDagCircuitSymbolicExecution:
    """Tests for execute_dag_circuit_symbolic."""

    def test_dag_circuit_bell_state(self) -> None:
        """Test symbolic execution via DagCircuit."""

        @guppy
        def bell() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            return (measure(q0), measure(q1))

        hugr_bytes = bell.compile().to_bytes()
        dag = hugr_to_dag_circuit(hugr_bytes)

        result = execute_dag_circuit_symbolic(dag)

        assert result.num_measurements == 2
        counts = result.sample_counts(1000)
        assert len(counts) == 2
        assert b"\x00\x00" in counts
        assert b"\x01\x01" in counts

    def test_dag_circuit_matches_hugr(self) -> None:
        """Test that DagCircuit execution matches direct HUGR execution."""

        @guppy
        def ghz() -> tuple[bool, bool, bool]:
            q0 = qubit()
            q1 = qubit()
            q2 = qubit()
            h(q0)
            cx(q0, q1)
            cx(q1, q2)
            return (measure(q0), measure(q1), measure(q2))

        hugr_bytes = ghz.compile().to_bytes()

        # Execute via HUGR
        result_hugr = execute_hugr_symbolic(hugr_bytes)

        # Execute via DagCircuit
        dag = hugr_to_dag_circuit(hugr_bytes)
        result_dag = execute_dag_circuit_symbolic(dag)

        # Should have same structure
        assert result_hugr.num_measurements == result_dag.num_measurements
        assert result_hugr.num_deterministic == result_dag.num_deterministic
        assert result_hugr.num_nondeterministic == result_dag.num_nondeterministic


class TestResultStringRepresentation:
    """Tests for string representation of results."""

    def test_str_representation(self) -> None:
        """Test that str() on result gives meaningful output."""

        @guppy
        def bell() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            return (measure(q0), measure(q1))

        result = execute_hugr_symbolic(bell.compile().to_bytes())

        str_repr = str(result)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0
        # Should contain measurement info like "m0" or similar
        assert "m" in str_repr.lower() or "[" in str_repr
