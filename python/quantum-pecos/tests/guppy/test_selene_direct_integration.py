"""Test running Guppy programs directly with Selene (without PECOS integration).

This test helps us understand how Selene works in isolation before integrating
it with PECOS's ClassicalControlEngine infrastructure.
"""

import re
import tempfile
from pathlib import Path
from typing import Any

from guppylang import guppy
from guppylang.std.quantum import cx, h, measure, qubit
from pecos.compilation_pipeline import compile_guppy_to_hugr
from selene_sim import build
from selene_sim.backends import Coinflip, SimpleRuntime
from selene_sim.backends import IdealErrorModel as IdealNoiseModel


class TestSeleneDirectIntegration:
    """Test Selene running Guppy programs directly."""

    def test_simple_bell_state_with_selene(self) -> None:
        """Test running a Bell state Guppy program through Selene's complete pipeline."""

        # Step 1: Define a Guppy quantum program
        @guppy
        def bell_state() -> tuple[bool, bool]:
            """Create a Bell state and measure both qubits."""
            q0, q1 = qubit(), qubit()
            h(q0)
            cx(q0, q1)
            return measure(q0).read(), measure(q1).read()

        # Step 2: Compile Guppy to HUGR
        hugr_bytes = compile_guppy_to_hugr(bell_state)
        assert hugr_bytes is not None, "HUGR compilation should succeed"
        assert len(hugr_bytes) > 0, "HUGR bytes should not be empty"

        # Step 3: Use Selene to build an executable from HUGR
        with tempfile.TemporaryDirectory() as tmpdir:
            build_dir = Path(tmpdir) / "selene_build"
            build_dir.mkdir()

            # Write HUGR to file for Selene to process
            hugr_file = build_dir / "program.hugr"
            hugr_file.write_bytes(hugr_bytes)
            assert hugr_file.exists(), "HUGR file should be created"

            # Use Selene's build API
            # Build the program using Selene (pass bytes directly)
            instance = build(hugr_bytes)
            assert instance is not None, "Build should create an instance"

            runtime = SimpleRuntime()  # Selene's simple runtime
            simulator = Coinflip()  # Simple 50/50 simulator
            noise_model = IdealNoiseModel()  # No noise

            # Step 5: Run the program and collect results
            n_shots = 10
            n_qubits = 2

            results: list[dict[str, Any]] = []
            for shot_results in instance.run_shots(
                simulator=simulator,
                n_qubits=n_qubits,
                runtime=runtime,
                error_model=noise_model,
                n_shots=n_shots,
                verbose=False,
            ):
                # Collect all results from this shot
                shot_data = dict(shot_results)
                results.append(shot_data)

            # Verify we got results
            assert len(results) == n_shots, f"Expected {n_shots} shots, got {len(results)}"


class TestGuppyToHUGRCompilation:
    """Test just the Guppy to HUGR compilation step."""

    def test_simple_h_gate_compilation(self) -> None:
        """Test compiling a simple H gate program."""

        @guppy
        def simple_h_gate() -> bool:
            """Apply H gate and measure."""
            q = qubit()
            h(q)
            return measure(q).read()

        hugr_bytes = compile_guppy_to_hugr(simple_h_gate)
        assert hugr_bytes is not None, "Should produce HUGR bytes"
        assert len(hugr_bytes) > 0, "HUGR bytes should not be empty"

        # Binary HUGR envelope (Model format); verify it is a valid, loadable HUGR.
        import pecos_rslib
        from pecos_rslib.hugr_lowering import compile_hugr_to_qis

        llvm_ir = compile_hugr_to_qis(hugr_bytes)
        assert re.search(r"define\b[^\n]*@qmain\(", llvm_ir)
        pecos_rslib.Qis(llvm_ir)

    def test_multi_qubit_compilation(self) -> None:
        """Test compiling a multi-qubit program."""

        @guppy
        def three_qubit_ghz() -> tuple[bool, bool, bool]:
            """Create a 3-qubit GHZ state."""
            q0, q1, q2 = qubit(), qubit(), qubit()
            h(q0)
            cx(q0, q1)
            cx(q1, q2)
            return measure(q0).read(), measure(q1).read(), measure(q2).read()

        hugr_bytes = compile_guppy_to_hugr(three_qubit_ghz)
        assert hugr_bytes is not None, "Should produce HUGR bytes"
        assert len(hugr_bytes) > 100, "Multi-qubit HUGR should be substantial"

        # Binary HUGR envelope (Model format); verify it is a valid, loadable HUGR.
        import pecos_rslib
        from pecos_rslib.hugr_lowering import compile_hugr_to_qis

        llvm_ir = compile_hugr_to_qis(hugr_bytes)
        assert re.search(r"define\b[^\n]*@qmain\(", llvm_ir)
        pecos_rslib.Qis(llvm_ir)

    def test_conditional_compilation(self) -> None:
        """Test compiling a program with conditional logic."""

        @guppy
        def conditional_circuit() -> int:
            """Circuit with measurement and conditional logic."""
            q = qubit()
            h(q)
            result = measure(q).read()
            if result:
                return 1
            return 0

        hugr_bytes = compile_guppy_to_hugr(conditional_circuit)
        assert hugr_bytes is not None, "Should produce HUGR bytes"
        assert len(hugr_bytes) > 0, "HUGR bytes should not be empty"

        # Binary HUGR envelope (Model format); verify it is a valid, loadable HUGR.
        import pecos_rslib
        from pecos_rslib.hugr_lowering import compile_hugr_to_qis

        llvm_ir = compile_hugr_to_qis(hugr_bytes)
        assert re.search(r"define\b[^\n]*@qmain\(", llvm_ir)
        pecos_rslib.Qis(llvm_ir)
