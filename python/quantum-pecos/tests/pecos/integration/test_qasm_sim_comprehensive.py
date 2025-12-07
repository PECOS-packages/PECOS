"""Comprehensive tests for QASM simulations covering all features and edge cases."""

from collections import Counter

import pytest


class TestQasmSimComprehensive:
    """Comprehensive tests for all qasm_engine features."""

    def test_no_noise_deterministic(self) -> None:
        """Test no noise produces deterministic results."""
        from pecos import Qasm, qasm_engine

        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        x q[0];
        x q[1];
        measure q -> c;
        """

        # Without noise, results should be deterministic
        results = qasm_engine().program(Qasm.from_string(qasm)).to_sim().run(100)
        results_dict = results.to_dict()

        # Should always measure |11> = 3
        assert all(val == 3 for val in results_dict["c"])

    def test_general_noise(self) -> None:
        """Test GeneralNoise model."""
        from pecos import Qasm, general_noise, qasm_engine

        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
        """

        # GeneralNoise uses default configuration
        results = (
            qasm_engine()
            .program(Qasm.from_string(qasm))
            .to_sim()
            .seed(42)
            .noise(general_noise())
            .run(1000)
        )

        results_dict = results.to_dict()
        assert isinstance(results_dict, dict)
        assert "c" in results_dict
        assert len(results_dict["c"]) == 1000

    def test_state_vector_engine(self) -> None:
        """Test StateVector engine explicitly."""
        from pecos import Qasm, qasm_engine, state_vector

        # Use a circuit with T gate (non-Clifford)
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        t q[0];
        cx q[0], q[1];
        measure q -> c;
        """

        results = (
            qasm_engine()
            .program(Qasm.from_string(qasm))
            .to_sim()
            .quantum(state_vector())
            .seed(42)
            .run(1000)
        )

        results_dict = results.to_dict()
        assert len(results_dict["c"]) == 1000
        # Results should be probabilistic due to T gate
        counts = Counter(results_dict["c"])
        assert len(counts) > 1  # Should see multiple outcomes

    def test_sparse_stabilizer_engine(self) -> None:
        """Test SparseStabilizer engine explicitly with Clifford circuit."""
        from pecos import Qasm, qasm_engine, sparse_stabilizer

        # Pure Clifford circuit (using only H and CX which are natively supported)
        qasm = """
        OPENQASM 2.0;
        qreg q[3];
        creg c[3];
        H q[0];
        CX q[0], q[1];
        CX q[1], q[2];
        H q[2];
        measure q -> c;
        """

        results = (
            qasm_engine()
            .program(Qasm.from_string(qasm))
            .to_sim()
            .quantum(sparse_stabilizer())
            .seed(42)
            .run(1000)
        )

        results_dict = results.to_dict()
        assert len(results_dict["c"]) == 1000

    def test_multiple_registers(self) -> None:
        """Test circuits with multiple classical registers."""
        from pecos import Qasm, qasm_engine

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

        results = qasm_engine().program(Qasm.from_string(qasm)).to_sim().run(10)
        results_dict = results.to_dict()

        assert "c1" in results_dict
        assert "c2" in results_dict
        assert len(results_dict["c1"]) == 10
        assert len(results_dict["c2"]) == 10
        # c1 should always be |10> = 1
        assert all(val == 1 for val in results_dict["c1"])
        # c2 should always be |10> = 1
        assert all(val == 1 for val in results_dict["c2"])

    def test_empty_circuit(self) -> None:
        """Test empty circuit (no gates, just measurements)."""
        from pecos import Qasm, qasm_engine

        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        measure q -> c;
        """

        results = qasm_engine().program(Qasm.from_string(qasm)).to_sim().run(100)

        results_dict = results.to_dict()
        # Should always measure |00> = 0
        assert all(val == 0 for val in results_dict["c"])

    def test_no_measurements(self) -> None:
        """Test circuit with no measurements."""
        from pecos import Qasm, qasm_engine

        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        h q[0];
        cx q[0], q[1];
        """

        results = qasm_engine().program(Qasm.from_string(qasm)).to_sim().run(100)

        # Should return empty dict when no measurements
        assert results.to_dict() == {}

    def test_partial_measurements(self) -> None:
        """Test measuring only some qubits."""
        from pecos import Qasm, qasm_engine

        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        creg c[2];
        x q[0];
        x q[1];
        x q[2];
        x q[3];
        measure q[0] -> c[0];
        measure q[2] -> c[1];
        """

        results = qasm_engine().program(Qasm.from_string(qasm)).to_sim().run(50)

        results_dict = results.to_dict()
        assert len(results_dict["c"]) == 50
        # Should measure |11> = 3 (only q[0] and q[2])
        assert all(val == 3 for val in results_dict["c"])

    def test_one_shot(self) -> None:
        """Test running with just 1 shot."""
        from pecos import Qasm, qasm_engine

        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        x q[0];
        x q[1];
        measure q -> c;
        """

        results = qasm_engine().program(Qasm.from_string(qasm)).to_sim().run(1)

        results_dict = results.to_dict()
        assert "c" in results_dict
        assert len(results_dict["c"]) == 1
        assert results_dict["c"][0] == 3  # Should measure |11>

    def test_high_noise_probability(self) -> None:
        """Test with very high noise probability."""
        from pecos import Qasm, depolarizing_noise, qasm_engine

        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        x q[0];
        measure q[0] -> c[0];
        """

        # With 50% depolarizing noise
        results = (
            qasm_engine()
            .program(Qasm.from_string(qasm))
            .to_sim()
            .seed(42)
            .noise(depolarizing_noise().with_uniform_probability(0.5))
            .run(1000)
        )

        results_dict = results.to_dict()
        zeros = sum(1 for val in results_dict["c"] if val == 0)
        # Should see significant errors, roughly 50/50 distribution
        assert 300 < zeros < 700

    def test_all_noise_models_builder(self) -> None:
        """Test all noise models through builder pattern."""
        from pecos import (
            GeneralNoiseModelBuilder,
            Qasm,
            biased_depolarizing_noise,
            depolarizing_noise,
            qasm_engine,
        )

        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        x q[0];
        measure q[0] -> c[0];
        """

        noise_builders = [
            None,  # No noise
            GeneralNoiseModelBuilder(),
            depolarizing_noise().with_uniform_probability(0.1),
            biased_depolarizing_noise().with_uniform_probability(0.033),
            depolarizing_noise()
            .with_prep_probability(0.1)
            .with_meas_probability(0.1)
            .with_p1_probability(0.1)
            .with_p2_probability(0.1),
        ]

        for noise_builder in noise_builders:
            sim_builder = (
                qasm_engine().program(Qasm.from_string(qasm)).to_sim().seed(42)
            )
            if noise_builder is not None:
                sim_builder = sim_builder.noise(noise_builder)
            sim = sim_builder.build()
            results = sim.run(100)
            results_dict = results.to_dict()
            assert len(results_dict["c"]) == 100

    def test_binary_string_format_empty_register(self) -> None:
        """Test binary string format with empty measurements."""
        from pecos import Qasm, qasm_engine

        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        h q[0];
        """

        results = qasm_engine().program(Qasm.from_string(qasm)).to_sim().run(10)
        results_dict = results.to_dict()
        assert results_dict == {}  # No measurements

    def test_deterministic_with_seed(self) -> None:
        """Test that same seed produces same results."""
        from pecos import Qasm, depolarizing_noise, qasm_engine

        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
        """

        # Build and run simulations with same seed
        noise1 = depolarizing_noise().with_uniform_probability(0.01)
        noise2 = depolarizing_noise().with_uniform_probability(0.01)

        sim1 = (
            qasm_engine()
            .program(Qasm.from_string(qasm))
            .to_sim()
            .seed(123)
            .noise(noise1)
            .build()
        )
        sim2 = (
            qasm_engine()
            .program(Qasm.from_string(qasm))
            .to_sim()
            .seed(123)
            .noise(noise2)
            .build()
        )

        results1 = sim1.run(1000)
        results2 = sim2.run(1000)

        # Should produce identical results with same seed
        assert results1.to_dict()["c"] == results2.to_dict()["c"]

        # Run with different seed
        sim3 = (
            qasm_engine()
            .program(Qasm.from_string(qasm))
            .to_sim()
            .seed(456)
            .noise(depolarizing_noise().with_uniform_probability(0.01))
            .build()
        )
        results3 = sim3.run(1000)

        # Should produce different results (with very high probability)
        # Count occurrences to verify they're different
        from collections import Counter

        counts1 = Counter(results1.to_dict()["c"])
        counts3 = Counter(results3.to_dict()["c"])

        # With 1000 shots and noise, the exact counts should differ
        assert counts1 != counts3

    def test_no_noise_config(self) -> None:
        """Test building without noise."""
        from pecos import Qasm, qasm_engine

        qasm = """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            x q[0];
            measure q[0] -> c[0];
            """

        sim = qasm_engine().program(Qasm.from_string(qasm)).to_sim().build()
        results = sim.run(10)

        results_dict = results.to_dict()
        # Should work without noise
        assert all(val == 1 for val in results_dict["c"])

    def test_invalid_qasm_syntax(self) -> None:
        """Test handling of invalid QASM syntax."""
        from pecos import Qasm, qasm_engine

        invalid_qasm = """
        OPENQASM 2.0;
        invalid syntax here
        """

        with pytest.raises(RuntimeError):
            qasm_engine().program(Qasm.from_string(invalid_qasm)).to_sim().run(
                10,
            )
