# Copyright 2026 The PECOS Developers
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

"""Tests for the PECOS StabActive Selene plugin."""

import math

from guppylang import guppy
from guppylang.std.angles import pi
from guppylang.std.builtins import result
from guppylang.std.quantum import crz, cx, discard, h, measure, qubit, reset, rx, ry, rz, x
from pecos_selene_stab_active import StabActivePlugin
from selene_sim.build import build


class TestStabActiveBasic:
    """Basic functionality tests for the StabActive plugin."""

    def test_single_qubit_discard(self) -> None:
        """Test that a qubit can be created and discarded."""

        @guppy
        def main() -> None:
            q = qubit()
            discard(q)

        runner = build(main.compile())
        simulator = StabActivePlugin(random_seed=42)

        results = list(runner.run(simulator, n_qubits=1))
        assert len(results) == 0

    def test_single_qubit_identity(self) -> None:
        """Test that a qubit without operations measures to 0."""

        @guppy
        def main() -> None:
            q = qubit()
            bit = measure(q).read()
            result("outcome", bit)

        runner = build(main.compile())
        simulator = StabActivePlugin(random_seed=42)

        results = list(runner.run(simulator, n_qubits=1))
        assert dict(results)["outcome"] == 0

    def test_hadamard_measurement(self) -> None:
        """Test that H gate creates superposition."""

        @guppy
        def main() -> None:
            q = qubit()
            h(q)
            bit = measure(q).read()
            result("outcome", bit)

        runner = build(main.compile())
        simulator = StabActivePlugin(random_seed=123)

        results = list(runner.run(simulator, n_qubits=1))
        assert dict(results)["outcome"] in [0, 1]


class TestStabActiveBellState:
    """Tests involving Bell states and entanglement."""

    def test_bell_state_correlation(self) -> None:
        """Test that Bell state measurements are correlated."""

        @guppy
        def main() -> None:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            b0 = measure(q0).read()
            b1 = measure(q1).read()
            result("q0", b0)
            result("q1", b1)

        runner = build(main.compile())
        simulator = StabActivePlugin(random_seed=999)

        results = list(runner.run(simulator, n_qubits=2))
        d = dict(results)
        assert d["q0"] == d["q1"], f"Bell state correlation failed: {d}"


class TestStabActiveArbitraryRotations:
    """Tests for arbitrary rotation angles (non-Clifford)."""

    def test_t_gate_like_rotation(self) -> None:
        """Test that a T-gate-like rotation (pi/4) works."""

        @guppy
        def main() -> None:
            q = qubit()
            h(q)
            rz(q, pi / 4)
            h(q)
            bit = measure(q).read()
            result("outcome", bit)

        runner = build(main.compile())
        simulator = StabActivePlugin(random_seed=42)

        results = list(runner.run(simulator, n_qubits=1))
        assert dict(results)["outcome"] in [0, 1]

    def test_arbitrary_rz_angle(self) -> None:
        """Test an arbitrary Rz rotation angle."""

        @guppy
        def main() -> None:
            q = qubit()
            h(q)
            rz(q, pi / 8)
            h(q)
            bit = measure(q).read()
            result("outcome", bit)

        runner = build(main.compile())
        simulator = StabActivePlugin(random_seed=42)

        results = list(runner.run(simulator, n_qubits=1))
        assert dict(results)["outcome"] in [0, 1]

    def test_rx_rotation(self) -> None:
        """Test RX rotation (exercises the RXY Selene interface)."""

        @guppy
        def main() -> None:
            q = qubit()
            rx(q, pi / 3)
            bit = measure(q).read()
            result("outcome", bit)

        runner = build(main.compile())
        simulator = StabActivePlugin(random_seed=42)

        results = list(runner.run(simulator, n_qubits=1))
        assert dict(results)["outcome"] in [0, 1]

    def test_ry_rotation(self) -> None:
        """Test RY rotation (exercises the RXY Selene interface with phi=pi/2)."""

        @guppy
        def main() -> None:
            q = qubit()
            ry(q, pi / 3)
            bit = measure(q).read()
            result("outcome", bit)

        runner = build(main.compile())
        simulator = StabActivePlugin(random_seed=42)

        results = list(runner.run(simulator, n_qubits=1))
        assert dict(results)["outcome"] in [0, 1]

    def test_crz_two_qubit(self) -> None:
        """Test controlled-RZ rotation (exercises the RZZ Selene interface)."""

        @guppy
        def main() -> None:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            crz(q0, q1, pi / 4)
            b0 = measure(q0).read()
            b1 = measure(q1).read()
            result("q0", b0)
            result("q1", b1)

        runner = build(main.compile())
        simulator = StabActivePlugin(random_seed=42)

        results = list(runner.run(simulator, n_qubits=2))
        d = dict(results)
        assert d["q0"] in [0, 1]
        assert d["q1"] in [0, 1]


class TestStabActiveReset:
    """Tests for qubit reset."""

    def test_reset_after_x(self) -> None:
        """Test that reset brings a |1> qubit back to |0>."""

        @guppy
        def main() -> None:
            q = qubit()
            x(q)
            reset(q)
            bit = measure(q).read()
            result("outcome", bit)

        runner = build(main.compile())
        simulator = StabActivePlugin(random_seed=42)

        results = list(runner.run(simulator, n_qubits=1))
        assert dict(results)["outcome"] == 0

    def test_reset_in_circuit(self) -> None:
        """Test reset mid-circuit followed by further operations."""

        @guppy
        def main() -> None:
            q = qubit()
            h(q)
            reset(q)
            h(q)
            bit = measure(q).read()
            result("outcome", bit)

        runner = build(main.compile())
        simulator = StabActivePlugin(random_seed=42)

        results = list(runner.run(simulator, n_qubits=1))
        assert dict(results)["outcome"] in [0, 1]


class TestStabActivePlugin:
    """Tests for the plugin interface."""

    def test_library_file_exists(self) -> None:
        """Test that the library file property returns a valid path."""
        plugin = StabActivePlugin()
        lib_path = plugin.library_file

        assert lib_path.name.startswith(
            "libpecos_selene_stab_active",
        ) or lib_path.name.startswith(
            "pecos_selene_stab_active",
        )

    def test_init_args_empty(self) -> None:
        """Test that init args are empty (no special parameters)."""
        plugin = StabActivePlugin()
        args = plugin.get_init_args()

        assert len(args) == 0


def _binomial_tail_probabilities(k: int, n: int, p: float) -> tuple[float, float]:
    """Return P(K <= k) and P(K >= k) for K ~ Binomial(n, p), using log-space terms."""

    def pmf(i: int) -> float:
        log_choose = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        return math.exp(log_choose + i * math.log(p) + (n - i) * math.log1p(-p))

    lower = math.fsum(pmf(i) for i in range(k + 1))
    upper = math.fsum(pmf(i) for i in range(k, n + 1))
    return lower, upper


def test_h_t_h_matches_exact_binomial() -> None:
    """H T H gives P(1) = sin^2(pi/8): accept unless either exact tail is below 0.05%."""

    @guppy
    def main() -> None:
        q = qubit()
        h(q)
        rz(q, pi / 4)  # T up to an irrelevant global phase.
        h(q)
        result("outcome", measure(q).read())

    shots = 4000
    runner = build(main.compile())
    ones = sum(
        int(dict(shot)["outcome"])
        for shot in runner.run_shots(
            simulator=StabActivePlugin(random_seed=42),
            n_qubits=1,
            n_shots=shots,
            n_processes=1,
        )
    )
    # The probability is known exactly, so this is an exact two-sided binomial
    # test at a 0.1% false-failure rate, with no third-party statistics package.
    lower, upper = _binomial_tail_probabilities(ones, shots, math.sin(math.pi / 8) ** 2)
    assert min(lower, upper) > 0.0005, (ones, lower, upper)


def test_rz_sign_through_rx_matches_exact_binomial() -> None:
    """H RZ(pi/4) RX(pi/2) gives P(1) = (1 - sin(pi/4))/2; a flipped RZ or RX sign gives 0.854."""

    @guppy
    def main() -> None:
        q = qubit()
        h(q)
        rz(q, pi / 4)
        rx(q, pi / 2)
        result("outcome", measure(q).read())

    shots = 4000
    runner = build(main.compile())
    ones = sum(
        int(dict(shot)["outcome"])
        for shot in runner.run_shots(
            simulator=StabActivePlugin(random_seed=7),
            n_qubits=1,
            n_shots=shots,
            n_processes=1,
        )
    )
    # Unlike sin^2(theta/2), this probability is odd in the RZ angle, so it
    # also pins the rotation sign and the RXY decomposition used for RX.
    expected = (1 - math.sin(math.pi / 4)) / 2
    lower, upper = _binomial_tail_probabilities(ones, shots, expected)
    assert min(lower, upper) > 0.0005, (ones, lower, upper)
