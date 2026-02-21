"""Tests for the QASM simulation interface using sim()."""

from collections import Counter

from pecos_rslib import (
    biased_depolarizing_noise,
    depolarizing_noise,
    general_noise,
    sparse_stabilizer,
    state_vector,
)
from pecos_rslib.programs import Qasm
from pecos_rslib import sim


class TestPythonicInterface:
    """Test the QASM simulation interface with sim()."""

    def test_simple_sim_qasm(self) -> None:
        """Test basic sim() functionality with QASM."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        x q[0];
        x q[1];
        measure q -> c;
        """

        # Run with minimal parameters
        prog = Qasm.from_string(qasm)
        results = sim(prog).run(10).to_dict()
        assert "c" in results
        assert len(results["c"]) == 10

        # All shots should measure 11 (both qubits in |1>)
        assert all(val == 3 for val in results["c"])  # 0b11 = 3

    def test_sim_qasm_with_engine(self) -> None:
        """Test sim() with different quantum engines."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        h q[0];
        measure q[0] -> c[0];
        """

        prog = Qasm.from_string(qasm)

        # Test with StateVector engine
        results_sv = sim(prog).quantum(state_vector()).seed(42).run(100).to_dict()
        assert "c" in results_sv
        assert len(results_sv["c"]) == 100

        # Test with SparseStabilizer engine
        results_stab = sim(prog).quantum(sparse_stabilizer()).seed(42).run(100).to_dict()
        assert "c" in results_stab
        assert len(results_stab["c"]) == 100

    def test_sim_qasm_with_noise_models(self) -> None:
        """Test sim() with noise models."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        x q[0];
        measure q[0] -> c[0];
        """

        prog = Qasm.from_string(qasm)

        # Test with no noise (default)
        results = sim(prog).run(100).to_dict()
        assert all(val == 1 for val in results["c"])

        # Test with DepolarizingNoise (using builder for control)
        noise = depolarizing_noise().with_seed(42).with_uniform_probability(0.3)
        results = sim(prog).noise(noise).run(1000).to_dict()
        # With strong noise, should see some errors
        zeros = sum(1 for val in results["c"] if val == 0)
        assert 100 < zeros < 500  # Should see some bit flips

        # Test with BiasedDepolarizingNoise (using builder for control)
        noise = biased_depolarizing_noise().with_seed(42).with_uniform_probability(0.2)
        results = sim(prog).noise(noise).run(1000).to_dict()
        # With seed=42 and p=0.2, we should see errors
        zeros = sum(1 for val in results["c"] if val == 0)
        assert zeros > 0  # Should see some errors

    def test_sim_qasm_with_custom_noise_builder(self) -> None:
        """Test sim() with custom noise builder."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
        """

        prog = Qasm.from_string(qasm)

        # Custom depolarizing with different error rates
        noise_builder = (
            general_noise()
            .with_seed(42)
            .with_p1_probability(0.001)  # Low single-qubit error
            .with_p2_probability(0.1)  # High two-qubit error
            .with_meas_0_probability(0.02)
            .with_meas_1_probability(0.02)
        )

        results = sim(prog).noise(noise_builder).run(1000).to_dict()
        assert "c" in results
        assert len(results["c"]) == 1000

        # With CX error, should see some non-Bell states (01 and 10)
        counts = Counter(results["c"])

        # Should see some errors due to high CX error rate
        assert 1 in counts or 2 in counts  # 01 or 10 states

    def test_sim_qasm_deterministic(self) -> None:
        """Test deterministic behavior with seed."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        h q[0];
        measure q[0] -> c[0];
        """

        prog = Qasm.from_string(qasm)

        # Run twice with same seed
        results1 = sim(prog).seed(123).run(100).to_dict()
        results2 = sim(prog).seed(123).run(100).to_dict()

        # Results should be identical
        assert results1["c"] == results2["c"]

        # Different seed should give different results
        results3 = sim(prog).seed(456).run(100).to_dict()
        assert results1["c"] != results3["c"]  # With high probability

    def test_sim_qasm_multi_register(self) -> None:
        """Test sim() with multiple classical registers."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        creg c1[2];
        creg c2[2];

        x q[0];
        x q[2];

        measure q[0] -> c1[0];
        measure q[1] -> c1[1];
        measure q[2] -> c2[0];
        measure q[3] -> c2[1];
        """

        prog = Qasm.from_string(qasm)
        results = sim(prog).run(10).to_dict()

        # Check both registers exist
        assert "c1" in results
        assert "c2" in results

        # c1 should be 01 (q[0]=1, q[1]=0) = 1
        assert all(val == 1 for val in results["c1"])

        # c2 should be 01 (q[2]=1, q[3]=0) = 1
        assert all(val == 1 for val in results["c2"])
