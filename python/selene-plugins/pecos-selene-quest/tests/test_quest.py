# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""Tests for the PECOS Quest Selene plugin."""

import functools
import os

import pytest
from guppylang import guppy
from guppylang.std.angles import pi
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, discard, h, measure, qubit, rz
from pecos_selene_quest import QuestPlugin, SimulatorMode
from selene_sim.build import build
from selene_sim.exceptions import SeleneRuntimeError


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available for Quest.

    This attempts to create a GPU-enabled Quest plugin and checks if it succeeds.
    Returns True if GPU is available, False otherwise.
    """
    try:
        # Try to build and run a minimal circuit with GPU
        @guppy
        def gpu_test() -> None:
            q = qubit()
            discard(q)

        runner = build(gpu_test.compile())
        simulator = QuestPlugin(use_gpu=True, random_seed=42)

        # This will fail during run if GPU is not available
        list(runner.run(simulator, n_qubits=1))
        return True
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def gpu_available() -> bool:
    """Cached check for GPU availability."""
    return is_gpu_available()


def should_skip_gpu_tests() -> bool:
    """Determine if GPU tests should be skipped.

    By default, GPU tests are skipped if GPU is not available.
    Set PECOS_TEST_GPU=1 to force GPU tests to run (and fail if GPU unavailable).
    """
    force_gpu = os.environ.get("PECOS_TEST_GPU", "").lower() in ("1", "true", "yes")
    if force_gpu:
        # User explicitly wants GPU tests - don't skip, let them fail if GPU unavailable
        return False
    # Default behavior: skip if GPU not available
    return not gpu_available()


requires_gpu = pytest.mark.skipif(
    should_skip_gpu_tests(),
    reason="GPU/CUDA not available (set PECOS_TEST_GPU=1 to force)",
)


class TestQuestBasic:
    """Basic functionality tests for the Quest plugin."""

    def test_single_qubit_discard(self) -> None:
        """Test that a qubit can be created and discarded."""

        @guppy
        def main() -> None:
            q = qubit()
            discard(q)

        runner = build(main.compile())
        simulator = QuestPlugin(random_seed=42)

        # Run a single shot
        results = list(runner.run(simulator, n_qubits=1))
        assert len(results) == 0  # No results expected since no measurements

    def test_single_qubit_identity(self) -> None:
        """Test that a qubit without operations measures to 0."""

        @guppy
        def main() -> None:
            q = qubit()
            bit = measure(q)
            result("outcome", bit)

        runner = build(main.compile())
        simulator = QuestPlugin(random_seed=42)

        # Run a single shot
        results = list(runner.run(simulator, n_qubits=1))
        assert dict(results)["outcome"] == 0

    def test_hadamard_measurement(self) -> None:
        """Test that H gate creates superposition."""

        @guppy
        def main() -> None:
            q = qubit()
            h(q)
            bit = measure(q)
            result("outcome", bit)

        runner = build(main.compile())
        simulator = QuestPlugin(random_seed=123)

        # Run a single shot
        results = list(runner.run(simulator, n_qubits=1))
        # The result should be either 0 or 1
        assert dict(results)["outcome"] in [0, 1]


class TestQuestBellState:
    """Tests involving Bell states and entanglement."""

    def test_bell_state_correlation(self) -> None:
        """Test that Bell state measurements are correlated."""

        @guppy
        def main() -> None:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            b0 = measure(q0)
            b1 = measure(q1)
            result("q0", b0)
            result("q1", b1)

        runner = build(main.compile())
        simulator = QuestPlugin(random_seed=999)

        # Run a single shot
        results = list(runner.run(simulator, n_qubits=2))
        d = dict(results)
        # Both qubits should always have the same outcome in a Bell state
        assert d["q0"] == d["q1"], f"Bell state correlation failed: {d}"


class TestQuestArbitraryRotations:
    """Tests for arbitrary rotation angles (non-Clifford)."""

    def test_t_gate_like_rotation(self) -> None:
        """Test that a T-gate-like rotation (pi/4) works."""

        @guppy
        def main() -> None:
            q = qubit()
            h(q)
            # T gate is Rz(pi/4)
            rz(q, pi / 4)
            h(q)
            bit = measure(q)
            result("outcome", bit)

        runner = build(main.compile())
        simulator = QuestPlugin(random_seed=42)

        # Run multiple shots to verify it works
        for _ in range(5):
            results = list(runner.run(simulator, n_qubits=1))
            # Just check it doesn't crash - the rotation is valid
            assert dict(results)["outcome"] in [0, 1]

    def test_arbitrary_rz_angle(self) -> None:
        """Test an arbitrary Rz rotation angle."""

        @guppy
        def main() -> None:
            q = qubit()
            h(q)
            # Non-Clifford angle (pi/8 is a common non-Clifford angle)
            rz(q, pi / 8)
            h(q)
            bit = measure(q)
            result("outcome", bit)

        runner = build(main.compile())
        simulator = QuestPlugin(random_seed=42)

        # This should work without error (unlike stabilizer simulators)
        results = list(runner.run(simulator, n_qubits=1))
        assert dict(results)["outcome"] in [0, 1]


class TestQuestPlugin:
    """Tests for the plugin interface."""

    def test_library_file_exists(self) -> None:
        """Test that the library file property returns a valid path."""
        plugin = QuestPlugin()
        lib_path = plugin.library_file

        # The path should be a Path object pointing to the expected location
        assert lib_path.name.startswith(
            "libpecos_selene_quest",
        ) or lib_path.name.startswith(
            "pecos_selene_quest",
        )

    def test_init_args_default(self) -> None:
        """Test that default init args include mode."""
        plugin = QuestPlugin()
        args = plugin.get_init_args()

        assert "--mode=state_vector" in args
        assert "--use-gpu" not in args

    def test_init_args_density_matrix(self) -> None:
        """Test init args for density matrix mode."""
        plugin = QuestPlugin(mode=SimulatorMode.DENSITY_MATRIX)
        args = plugin.get_init_args()

        assert "--mode=density_matrix" in args
        assert "--use-gpu" not in args

    def test_init_args_with_gpu(self) -> None:
        """Test init args with GPU enabled."""
        plugin = QuestPlugin(use_gpu=True)
        args = plugin.get_init_args()

        assert "--mode=state_vector" in args
        assert "--use-gpu" in args

    def test_init_args_density_matrix_gpu(self) -> None:
        """Test init args for density matrix mode with GPU."""
        plugin = QuestPlugin(mode=SimulatorMode.DENSITY_MATRIX, use_gpu=True)
        args = plugin.get_init_args()

        assert "--mode=density_matrix" in args
        assert "--use-gpu" in args


class TestQuestDensityMatrix:
    """Tests for density matrix simulation mode."""

    def test_density_matrix_single_qubit(self) -> None:
        """Test basic density matrix simulation."""

        @guppy
        def main() -> None:
            q = qubit()
            bit = measure(q)
            result("outcome", bit)

        runner = build(main.compile())
        simulator = QuestPlugin(mode=SimulatorMode.DENSITY_MATRIX, random_seed=42)

        # Run a single shot
        results = list(runner.run(simulator, n_qubits=1))
        assert dict(results)["outcome"] == 0

    def test_density_matrix_bell_state(self) -> None:
        """Test Bell state with density matrix simulation."""

        @guppy
        def main() -> None:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            b0 = measure(q0)
            b1 = measure(q1)
            result("q0", b0)
            result("q1", b1)

        runner = build(main.compile())
        simulator = QuestPlugin(mode=SimulatorMode.DENSITY_MATRIX, random_seed=999)

        # Run a single shot
        results = list(runner.run(simulator, n_qubits=2))
        d = dict(results)
        # Both qubits should always have the same outcome in a Bell state
        assert d["q0"] == d["q1"], f"Bell state correlation (density matrix) failed: {d}"

    def test_density_matrix_arbitrary_rotation(self) -> None:
        """Test arbitrary rotation with density matrix simulation."""

        @guppy
        def main() -> None:
            q = qubit()
            h(q)
            rz(q, pi / 8)
            h(q)
            bit = measure(q)
            result("outcome", bit)

        runner = build(main.compile())
        simulator = QuestPlugin(mode=SimulatorMode.DENSITY_MATRIX, random_seed=42)

        # This should work without error
        results = list(runner.run(simulator, n_qubits=1))
        assert dict(results)["outcome"] in [0, 1]


class TestQuestGPU:
    """Tests for GPU acceleration. Skipped if GPU/CUDA is not available."""

    @requires_gpu
    def test_gpu_single_qubit(self) -> None:
        """Test basic GPU simulation."""

        @guppy
        def main() -> None:
            q = qubit()
            bit = measure(q)
            result("outcome", bit)

        runner = build(main.compile())
        simulator = QuestPlugin(use_gpu=True, random_seed=42)

        # Run a single shot
        results = list(runner.run(simulator, n_qubits=1))
        assert dict(results)["outcome"] == 0

    @requires_gpu
    def test_gpu_hadamard(self) -> None:
        """Test Hadamard gate with GPU."""

        @guppy
        def main() -> None:
            q = qubit()
            h(q)
            bit = measure(q)
            result("outcome", bit)

        runner = build(main.compile())
        simulator = QuestPlugin(use_gpu=True, random_seed=123)

        # Run a single shot
        results = list(runner.run(simulator, n_qubits=1))
        assert dict(results)["outcome"] in [0, 1]

    @requires_gpu
    def test_gpu_bell_state(self) -> None:
        """Test Bell state with GPU acceleration."""

        @guppy
        def main() -> None:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            b0 = measure(q0)
            b1 = measure(q1)
            result("q0", b0)
            result("q1", b1)

        runner = build(main.compile())
        simulator = QuestPlugin(use_gpu=True, random_seed=999)

        # Run a single shot
        results = list(runner.run(simulator, n_qubits=2))
        d = dict(results)
        # Both qubits should always have the same outcome in a Bell state
        assert d["q0"] == d["q1"], f"Bell state correlation (GPU) failed: {d}"

    @requires_gpu
    def test_gpu_arbitrary_rotation(self) -> None:
        """Test arbitrary rotation with GPU."""

        @guppy
        def main() -> None:
            q = qubit()
            h(q)
            rz(q, pi / 8)
            h(q)
            bit = measure(q)
            result("outcome", bit)

        runner = build(main.compile())
        simulator = QuestPlugin(use_gpu=True, random_seed=42)

        # This should work without error
        results = list(runner.run(simulator, n_qubits=1))
        assert dict(results)["outcome"] in [0, 1]

    @requires_gpu
    def test_gpu_density_matrix(self) -> None:
        """Test density matrix simulation with GPU."""

        @guppy
        def main() -> None:
            q = qubit()
            h(q)
            bit = measure(q)
            result("outcome", bit)

        runner = build(main.compile())
        simulator = QuestPlugin(
            mode=SimulatorMode.DENSITY_MATRIX,
            use_gpu=True,
            random_seed=42,
        )

        # Run a single shot
        results = list(runner.run(simulator, n_qubits=1))
        assert dict(results)["outcome"] in [0, 1]

    def test_gpu_unavailable_error_message(self) -> None:
        """Test that requesting GPU when unavailable gives a clear error."""
        if gpu_available():
            pytest.skip("GPU is available, cannot test unavailable error")

        @guppy
        def main() -> None:
            q = qubit()
            discard(q)

        runner = build(main.compile())
        simulator = QuestPlugin(use_gpu=True, random_seed=42)

        with pytest.raises(SeleneRuntimeError, match=r"[Gg][Pp][Uu]"):
            list(runner.run(simulator, n_qubits=1))
