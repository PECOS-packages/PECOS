"""Test QASM simulation structured configuration functionality."""

from collections import Counter


class TestQasmSimStructuredConfig:
    """Test qasm_engine structured configuration functionality."""

    def test_basic_config(self) -> None:
        """Test basic configuration without noise."""
        from pecos import Qasm, qasm_engine

        qasm = """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            h q[0];
            cx q[0], q[1];
            measure q -> c;
            """

        sim = qasm_engine().program(Qasm.from_string(qasm)).to_sim().seed(42).build()
        results = sim.run(1000)

        # Convert ShotVec to dict
        results_dict = results.to_dict()
        assert isinstance(results_dict, dict)
        assert "c" in results_dict
        assert len(results_dict["c"]) == 1000

        # Check Bell state results
        counts = Counter(results_dict["c"])
        assert set(counts.keys()) <= {0, 3}  # Only |00> and |11>

    def test_config_with_noise(self) -> None:
        """Test configuration with noise model."""
        from pecos import Qasm, depolarizing_noise, qasm_engine

        qasm = """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            x q[0];
            measure q[0] -> c[0];
            """

        sim = (
            qasm_engine()
            .program(Qasm.from_string(qasm))
            .to_sim()
            .seed(42)
            .noise(depolarizing_noise().with_uniform_probability(0.1))
            .build()
        )
        results = sim.run(1000)

        # Should see some errors due to noise
        results_dict = results.to_dict()
        zeros = sum(1 for val in results_dict["c"] if val == 0)
        assert 50 < zeros < 200  # Some bit flips due to noise

    def test_full_config(self) -> None:
        """Test configuration with all options."""
        from pecos import (
            Qasm,
            biased_depolarizing_noise,
            qasm_engine,
            sparse_stabilizer,
        )

        qasm = """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
            creg c[3];
            h q[0];
            cx q[0], q[1];
            cx q[1], q[2];
            measure q -> c;
            """

        sim = (
            qasm_engine()
            .program(Qasm.from_string(qasm))
            .to_sim()
            .seed(42)
            .workers(2)
            .noise(biased_depolarizing_noise().with_uniform_probability(0.003))
            .quantum(sparse_stabilizer())
            .build()
        )
        results = sim.run(100)

        results_dict = results.to_binary_dict()
        assert isinstance(results_dict, dict)
        assert "c" in results_dict
        assert len(results_dict["c"]) == 100

        # Check binary string format
        assert all(isinstance(val, str) for val in results_dict["c"])
        assert all(len(val) == 3 for val in results_dict["c"])
        assert all(set(val) <= {"0", "1"} for val in results_dict["c"])

    def test_auto_workers(self) -> None:
        """Test configuration with auto workers."""
        from pecos import Qasm, qasm_engine

        qasm = """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            h q[0];
            cx q[0], q[1];
            measure q -> c;
            """

        sim = (
            qasm_engine()
            .program(Qasm.from_string(qasm))
            .to_sim()
            .auto_workers()
            .build()
        )
        results = sim.run(100)

        results_dict = results.to_dict()
        assert len(results_dict["c"]) == 100

    def test_custom_noise_config(self) -> None:
        """Test configuration with custom noise parameters."""
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

        sim = (
            qasm_engine()
            .program(Qasm.from_string(qasm))
            .to_sim()
            .seed(42)
            .noise(
                depolarizing_noise()
                .with_prep_probability(0.001)
                .with_meas_probability(0.002)
                .with_p1_probability(0.003)
                .with_p2_probability(0.004),
            )
            .build()
        )
        results = sim.run(100)

        results_dict = results.to_dict()
        assert len(results_dict["c"]) == 100

    def test_missing_qasm_raises_error(self) -> None:
        """Test that missing QASM code raises error."""
        # This test is no longer relevant since QASM is now a required parameter
        # QASM is now a required parameter to sim(), not part of the config

    def test_invalid_noise_type_raises_error(self) -> None:
        """Test that invalid noise type raises error."""
        # In the new API, invalid noise types are caught at the type level
        # This test is no longer relevant as we use builder methods

    def test_invalid_engine_raises_error(self) -> None:
        """Test that invalid quantum engine raises error."""
        # In the new API, invalid engines are caught at the type level
        # This test is no longer relevant as we use builder methods

    def test_builder_pattern_serialization(self) -> None:
        """Test the new builder pattern approach."""
        from pecos import (
            Qasm,
            depolarizing_noise,
            qasm_engine,
            sparse_stabilizer,
        )

        qasm = """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            h q[0];
            cx q[0], q[1];
            measure q -> c;
            """

        # Builder pattern is the new approach
        sim = (
            qasm_engine()
            .program(Qasm.from_string(qasm))
            .to_sim()
            .seed(42)
            .workers(4)
            .noise(depolarizing_noise().with_uniform_probability(0.01))
            .quantum(sparse_stabilizer())
            .build()
        )
        results = sim.run(100)

        results_dict = results.to_dict()
        assert len(results_dict["c"]) == 100

    def test_structured_config(self) -> None:
        """Test new structured configuration approach."""
        from pecos import Qasm, general_noise, qasm_engine, state_vector

        qasm = """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            h q[0];
            cx q[0], q[1];
            measure q -> c;
            """

        # Create noise using functional API - pass it directly to noise() method
        noise_builder = (
            general_noise()
            .with_seed(42)
            .with_p1_probability(0.001)
            .with_p2_probability(0.01)
        )

        # Use builder pattern instead of config dict
        sim = (
            qasm_engine()
            .program(Qasm.from_string(qasm))
            .to_sim()
            .seed(42)
            .auto_workers()
            .noise(noise_builder)
            .quantum(state_vector())
            .build()
        )
        results = sim.run(100)

        results_dict = results.to_binary_dict()
        assert isinstance(results_dict, dict)
        assert "c" in results_dict
        assert len(results_dict["c"]) == 100

        # Check binary string format
        assert all(isinstance(val, str) for val in results_dict["c"])
        assert all(len(val) == 2 for val in results_dict["c"])

    def test_general_noise_config(self) -> None:
        """Test GeneralNoise configuration with functional API."""
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

        # Use functional API for GeneralNoise
        noise_builder = (
            general_noise()
            .with_seed(42)
            .with_p1_probability(0.001)
            .with_p2_probability(0.01)
            .with_prep_probability(0.001)
            .with_meas_0_probability(0.002)
            .with_meas_1_probability(0.002)
            # TODO: Add these methods to Python bindings:
            # .with_noiseless_gates(["H"])
            # .with_p1_pauli_model(x=0.5, y=0.3, z=0.2)
        )

        sim = (
            qasm_engine()
            .program(Qasm.from_string(qasm))
            .to_sim()
            .seed(42)
            .noise(noise_builder)
            .build()
        )
        results = sim.run(100)

        results_dict = results.to_dict()
        assert len(results_dict["c"]) == 100
