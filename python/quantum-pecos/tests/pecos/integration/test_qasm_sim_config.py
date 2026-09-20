"""Test QASM simulation structured configuration functionality."""

from collections import Counter

import pytest


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
        assert set(counts.keys()) == {0, 3}  # Only |00> and |11>

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
            sparse_stab,
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
            .quantum(sparse_stab())
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

        sim = qasm_engine().program(Qasm.from_string(qasm)).to_sim().auto_workers().build()
        results = sim.run(100)

        results_dict = results.to_dict()
        assert len(results_dict["c"]) == 100

    @pytest.mark.parametrize("probability", [0.0, 0.5, 1.0], ids=["disabled", "half", "enabled"])
    @pytest.mark.parametrize(
        ("setting", "instructions", "noisy_distribution"),
        [
            pytest.param("p_prep", "reset q; cx q[0], q[1];", {1: 1.0}, id="preparation"),
            pytest.param("p_meas", "", {3: 1.0}, id="measurement"),
            pytest.param("p1", "z q[0];", {0: 1 / 3, 1: 2 / 3}, id="one-qubit"),
            pytest.param("p2", "cx q[0], q[1];", {0: 3 / 15, 1: 4 / 15, 2: 4 / 15, 3: 4 / 15}, id="two-qubit"),
        ],
    )
    def test_custom_noise_config(self, probability, setting, instructions, noisy_distribution) -> None:
        """Each probability controls its own channel, with all other channels disabled."""
        from pecos import Qasm, depolarizing_noise, qasm_engine

        program = Qasm.from_string(f"""
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            {instructions}
            measure q -> c;
        """)
        noise = getattr(depolarizing_noise().with_uniform_probability(0.0), f"with_{setting}")(probability)
        shots = 4096
        values = qasm_engine().program(program).to_sim().noise(noise).seed(42).run(shots).to_dict()["c"]
        assert len(values) == shots
        p = probability
        if setting == "p_prep":
            # Independent preparation flips precede CX: 11 becomes 01,
            # distinguishing preparation faults from measurement faults.
            expected = {0: (1 - p) ** 2, 1: p**2, 2: p * (1 - p), 3: p * (1 - p)}
        elif setting == "p_meas":
            # Each of the two measured bits flips independently.
            expected = {0: (1 - p) ** 2, 1: p * (1 - p), 2: p * (1 - p), 3: p**2}
        else:
            # A single noisy gate either preserves 00 or applies a Pauli fault.
            expected = {outcome: p * rate for outcome, rate in noisy_distribution.items()}
            expected[0] += 1 - p
        expected = {outcome: rate for outcome, rate in expected.items() if rate > 0}
        counts = Counter(values)
        assert set(counts) == set(expected)
        # One-qubit Pauli faults flip a Z measurement for X and Y (2/3).
        # Of the 15 nonidentity two-qubit Paulis, three preserve 00 and
        # four produce each other outcome. Use six binomial standard errors;
        # deterministic channels must match exactly.
        for outcome, expected_rate in expected.items():
            tolerance = 6 * (expected_rate * (1 - expected_rate) / shots) ** 0.5
            assert counts[outcome] / shots == pytest.approx(expected_rate, abs=tolerance, rel=0)

    def test_missing_qasm_raises_error(self) -> None:
        from pecos import qasm_engine

        with pytest.raises(RuntimeError, match="No QASM source specified"):
            qasm_engine().to_sim().build()

    @pytest.mark.parametrize("invalid", ["invalid", object()], ids=["string", "python-object"])
    def test_invalid_noise_type_raises_error(self, invalid) -> None:
        from pecos import Qasm, qasm_engine

        program = Qasm.from_string('OPENQASM 2.0; include "qelib1.inc"; qreg q[1];')
        with pytest.raises(TypeError, match="Unrecognized noise builder type"):
            qasm_engine().program(program).to_sim().noise(invalid).build()

    @pytest.mark.parametrize("invalid", ["invalid", object()], ids=["string", "python-object"])
    def test_invalid_engine_raises_error(self, invalid) -> None:
        from pecos import Qasm, qasm_engine

        program = Qasm.from_string('OPENQASM 2.0; include "qelib1.inc"; qreg q[1];')
        with pytest.raises(TypeError, match="Unrecognized quantum engine builder type"):
            qasm_engine().program(program).to_sim().quantum(invalid).build()

    def test_combined_builder_options_smoke(self) -> None:
        """Combined worker, engine, and noise settings produce the requested shots."""
        from pecos import (
            Qasm,
            depolarizing_noise,
            qasm_engine,
            sparse_stab,
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
            .quantum(sparse_stab())
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
        noise_builder = general_noise().with_seed(42).with_p1(0.001).with_p2(0.01)

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
            .with_p1(0.001)
            .with_p2(0.01)
            .with_p_prep(0.001)
            .with_p_meas_0(0.002)
            .with_p_meas_1(0.002)
            # TODO: Add these methods to Python bindings:
            # .with_noiseless_gates(["H"])
            # .with_p1_pauli_model(x=0.5, y=0.3, z=0.2)
        )

        sim = qasm_engine().program(Qasm.from_string(qasm)).to_sim().seed(42).noise(noise_builder).build()
        results = sim.run(100)

        results_dict = results.to_dict()
        assert len(results_dict["c"]) == 100
