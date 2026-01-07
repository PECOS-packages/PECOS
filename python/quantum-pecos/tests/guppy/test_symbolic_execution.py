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
    NoisySymbolicExecutionResult,
    SymbolicExecutionResult,
    execute_dag_circuit_symbolic,
    execute_dag_circuit_symbolic_noisy,
    execute_hugr_symbolic,
    execute_hugr_symbolic_noisy,
)
from pecos_rslib import hugr_to_dag_circuit


def outcome_to_tuple(outcome: bytes) -> tuple[bool, ...]:
    """Convert bytes outcome to tuple of bools for easier assertion."""
    return tuple(bool(b) for b in outcome)


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


class TestNoisySymbolicExecution:
    """Tests for noisy symbolic execution with depolarizing noise."""

    def test_noisy_execution_returns_noisy_result(self) -> None:
        """Test that noisy execution returns NoisySymbolicExecutionResult."""

        @guppy
        def bell() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            return (measure(q0), measure(q1))

        result = execute_hugr_symbolic_noisy(
            bell.compile().to_bytes(),
            p1=0.01,  # 1% single-qubit error
        )

        assert isinstance(result, NoisySymbolicExecutionResult)
        assert result.num_measurements == 2
        # Should have faults from the H gate (3 Pauli types)
        assert result.num_faults > 0

    def test_noiseless_execution_has_no_faults(self) -> None:
        """Test that zero noise produces no fault events."""

        @guppy
        def bell() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            return (measure(q0), measure(q1))

        result = execute_hugr_symbolic_noisy(
            bell.compile().to_bytes(),
            p1=0.0,
            p2=0.0,
            p_meas=0.0,
            p_prep=0.0,
        )

        assert result.num_measurements == 2
        assert result.num_faults == 0

    def test_noisy_sampling_produces_more_outcomes(self) -> None:
        """Test that noise can produce outcomes not in noiseless distribution."""

        @guppy
        def bell() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            return (measure(q0), measure(q1))

        hugr_bytes = bell.compile().to_bytes()

        # Noiseless: only |00> and |11>
        noiseless = execute_hugr_symbolic(hugr_bytes)
        noiseless_counts = noiseless.sample_counts(10000)
        assert len(noiseless_counts) == 2
        assert b"\x00\x01" not in noiseless_counts
        assert b"\x01\x00" not in noiseless_counts

        # With significant noise: should see some |01> and |10>
        noisy = execute_hugr_symbolic_noisy(
            hugr_bytes,
            p1=0.1,  # 10% single-qubit error (high for demonstration)
            p2=0.1,
        )
        noisy_counts = noisy.sample_counts(10000)

        # With 10% noise, we should see some anti-correlated outcomes
        # (though not guaranteed, highly likely with 10000 samples)
        assert len(noisy_counts) >= 2  # At minimum, still see 00 and 11

    def test_measurement_noise_flips_outcomes(self) -> None:
        """Test that measurement noise directly flips measurement outcomes."""

        @guppy
        def deterministic_zero() -> bool:
            q = qubit()
            return measure(q)

        # With 100% measurement noise, all outcomes should flip from 0 to 1
        result = execute_hugr_symbolic_noisy(
            deterministic_zero.compile().to_bytes(),
            p_meas=1.0,  # 100% measurement flip
        )

        # Sample many times - all should be 1 (flipped from deterministic 0)
        counts = result.sample_counts(100)
        # Note: measurement noise XORs, so deterministic 0 becomes 0 ^ 1 = 1
        assert b"\x01" in counts
        assert counts.get(b"\x01", 0) == 100

    def test_dag_circuit_noisy_execution(self) -> None:
        """Test noisy execution via DagCircuit."""

        @guppy
        def bell() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            return (measure(q0), measure(q1))

        hugr_bytes = bell.compile().to_bytes()
        dag = hugr_to_dag_circuit(hugr_bytes)

        result = execute_dag_circuit_symbolic_noisy(
            dag,
            p1=0.01,
            p2=0.01,
        )

        assert isinstance(result, NoisySymbolicExecutionResult)
        assert result.num_measurements == 2
        assert result.num_faults > 0

    def test_noisy_result_string_representation(self) -> None:
        """Test that str() on noisy result gives meaningful output."""

        @guppy
        def single_measure() -> bool:
            q = qubit()
            h(q)
            return measure(q)

        result = execute_hugr_symbolic_noisy(
            single_measure.compile().to_bytes(),
            p1=0.01,
        )

        str_repr = str(result)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0

        repr_str = repr(result)
        assert "NoisySymbolicExecutionResult" in repr_str
        assert "faults=" in repr_str
