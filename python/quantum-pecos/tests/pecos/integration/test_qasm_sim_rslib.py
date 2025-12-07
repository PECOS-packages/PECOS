"""Integration tests for QASM simulations using pecos API."""

from collections import Counter

from pecos import (
    Qasm,
    biased_depolarizing_noise,
    depolarizing_noise,
    qasm_engine,
    sparse_stabilizer,
    state_vector,
)


class TestQasmSimRslib:
    """Test QASM simulation functionality using pecos imports."""

    def test_import_qasm_engine(self) -> None:
        """Test that we can import qasm_engine from pecos."""
        from pecos import qasm_engine

        assert callable(qasm_engine)

    def test_import_noise_models(self) -> None:
        """Test that we can import noise models from pecos."""
        from pecos import (
            biased_depolarizing_noise,
            depolarizing_noise,
            general_noise,
        )

        # Test that we can create noise builders
        assert depolarizing_noise() is not None
        assert biased_depolarizing_noise() is not None
        assert general_noise() is not None

    def test_import_utilities(self) -> None:
        """Test that we can import utility functions from pecos."""
        from pecos import sparse_stabilizer, state_vector

        # Test quantum engine builders
        assert callable(state_vector)
        assert callable(sparse_stabilizer)

    def test_basic_simulation(self) -> None:
        """Test basic QASM simulation using pecos imports."""
        qasm_code = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
        """

        results = qasm_engine().program(Qasm(qasm_code)).to_sim().seed(42).run(1000)
        results_dict = results.to_dict()

        assert isinstance(results_dict, dict)
        assert "c" in results_dict
        assert len(results_dict["c"]) == 1000

        # Check Bell state results
        counts = Counter(results_dict["c"])
        assert set(counts.keys()) <= {0, 3}  # Only |00> and |11>
        assert all(count > 400 for count in counts.values())  # Roughly equal

    def test_simulation_with_noise(self) -> None:
        """Test QASM simulation with noise using pecos imports."""
        qasm_code = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        x q[0];
        measure q[0] -> c[0];
        """

        # With noise
        results = (
            qasm_engine()
            .program(Qasm(qasm_code))
            .to_sim()
            .seed(42)
            .noise(depolarizing_noise().with_uniform_probability(0.1))
            .run(1000)
        )
        results_dict = results.to_dict()

        assert isinstance(results_dict, dict)
        assert "c" in results_dict
        assert len(results_dict["c"]) == 1000

        # Should see some errors due to noise
        zeros = sum(1 for val in results_dict["c"] if val == 0)
        assert 50 < zeros < 200  # Some bit flips due to noise

    def test_builder_pattern(self) -> None:
        """Test the builder pattern using pecos imports."""
        qasm_code = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c[3];
        h q[0];
        cx q[0], q[1];
        cx q[1], q[2];
        measure q -> c;
        """

        # Build once
        sim = (
            qasm_engine()
            .program(Qasm(qasm_code))
            .to_sim()
            .seed(42)
            .workers(2)
            .noise(biased_depolarizing_noise().with_uniform_probability(0.003))
            .quantum(sparse_stabilizer())
            .build()
        )

        # Run multiple times
        results1 = sim.run(100)
        results2 = sim.run(200)

        results1_dict = results1.to_dict()
        results2_dict = results2.to_dict()

        assert len(results1_dict["c"]) == 100
        assert len(results2_dict["c"]) == 200

        # Both should have the same types of results (GHZ state)
        counts1 = Counter(results1_dict["c"])
        counts2 = Counter(results2_dict["c"])

        # With low noise, should mostly see |000> and |111>
        assert 0 in counts1
        assert 7 in counts1
        assert 0 in counts2
        assert 7 in counts2

    def test_binary_string_format(self) -> None:
        """Test binary string format output using pecos imports."""
        qasm_code = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c[3];
        x q[0];
        x q[2];
        measure q -> c;
        """

        # Test binary string format
        results = qasm_engine().program(Qasm(qasm_code)).to_sim().run(10)
        results_dict = results.to_binary_dict()

        assert isinstance(results_dict, dict)
        assert "c" in results_dict
        assert len(results_dict["c"]) == 10

        # Check that all results are binary strings
        assert all(isinstance(val, str) for val in results_dict["c"])
        assert all(len(val) == 3 for val in results_dict["c"])
        assert all(set(val) <= {"0", "1"} for val in results_dict["c"])

        # Should always measure |101>
        assert all(val == "101" for val in results_dict["c"])

    def test_auto_workers(self) -> None:
        """Test auto_workers functionality using pecos imports."""
        qasm_code = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
        """

        # This should use all available CPU cores
        results = (
            qasm_engine().program(Qasm(qasm_code)).to_sim().auto_workers().run(1000)
        )
        results_dict = results.to_dict()

        assert isinstance(results_dict, dict)
        assert "c" in results_dict
        assert len(results_dict["c"]) == 1000

    def test_run_direct_pattern(self) -> None:
        """Test running simulations directly using pecos imports."""
        qasm_code = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
        """

        # Simple usage
        results = qasm_engine().program(Qasm(qasm_code)).to_sim().run(100)
        results_dict = results.to_dict()
        assert len(results_dict["c"]) == 100

        # With all parameters
        results = (
            qasm_engine()
            .program(Qasm(qasm_code))
            .to_sim()
            .noise(depolarizing_noise().with_uniform_probability(0.01))
            .quantum(state_vector())
            .workers(2)
            .seed(42)
            .run(100)
        )
        results_dict = results.to_dict()
        assert len(results_dict["c"]) == 100

    def test_large_register(self) -> None:
        """Test simulation with large quantum registers using pecos imports."""
        qasm_code = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[100];
        creg c[100];
        x q[0];
        x q[50];
        x q[99];
        measure q -> c;
        """

        # Test with default format (should handle big integers)
        results = qasm_engine().program(Qasm(qasm_code)).to_sim().run(5)
        results_dict = results.to_dict()

        assert "c" in results_dict
        assert len(results_dict["c"]) == 5

        # The result should have bits set at positions 0, 50, and 99
        # In integer form, this is 2^0 + 2^50 + 2^99
        expected = (1 << 0) + (1 << 50) + (1 << 99)
        assert all(val == expected for val in results_dict["c"])

        # Test with binary string format
        results_binary = qasm_engine().program(Qasm(qasm_code)).to_sim().run(5)
        results_binary_dict = results_binary.to_binary_dict()

        assert all(len(val) == 100 for val in results_binary_dict["c"])
        # Check specific bit positions (remember: MSB first in string)
        for binary_str in results_binary_dict["c"]:
            assert binary_str[99] == "1"  # q[0] -> position 99
            assert binary_str[49] == "1"  # q[50] -> position 49
            assert binary_str[0] == "1"  # q[99] -> position 0
            assert binary_str.count("1") == 3
