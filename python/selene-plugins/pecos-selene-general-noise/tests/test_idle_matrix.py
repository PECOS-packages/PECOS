"""Device-neutral duration sweeps for every current PECOS idle family."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
from general_noise_conformance import ConformanceExperiment, ExpectedDistribution
from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, measure, qubit
from pecos_selene_general_noise import GeneralNoiseParameters
from selene_sim import Stim
from selene_sim.build import build

pytestmark = pytest.mark.slow


@guppy
def idle_probe() -> None:
    """Create one two-qubit idle site without assuming a device schedule."""
    q0 = qubit()
    q1 = qubit()
    cx(q0, q1)
    result("q0", measure(q0).read())
    result("q1", measure(q1).read())


IDLE_RUNNER = build(idle_probe.compile())


@dataclass(frozen=True)
class IdleFamily:
    """One current general-noise idle law and its independent probability rule."""

    name: str
    rate: float

    def parameters(self, duration: float) -> GeneralNoiseParameters:
        base = GeneralNoiseParameters().with_idle_after_2q(duration)
        if self.name == "linear":
            return base.with_p_idle_linear(self.rate, {"X": 1.0})
        if self.name == "sine-squared":
            return base.with_p_idle_sin_squared(self.rate, {"X": 1.0})
        return base.with_p_idle_coherent(self.rate, {"RX": 1.0})

    def excitation_probability(self, duration: float) -> float:
        if self.name == "linear":
            return min(1.0, self.rate * duration)
        if self.name == "sine-squared":
            return math.sin(self.rate * duration) ** 2
        # RX(theta)|0> has population sin^2(theta/2) in |1>.
        return math.sin(self.rate * duration / 2.0) ** 2


IDLE_FAMILIES = (
    pytest.param(
        IdleFamily("linear", 0.4),
        id="linear",
        marks=pytest.mark.noise_channel("idle-linear", oracle="analytic"),
    ),
    pytest.param(
        IdleFamily("sine-squared", 1.5),
        id="sine-squared",
        marks=pytest.mark.noise_channel("idle-sine-squared", oracle="analytic"),
    ),
    pytest.param(
        IdleFamily("coherent", 2.0),
        id="coherent",
        marks=pytest.mark.noise_channel("idle-coherent", oracle="analytic"),
    ),
)

# These normalized durations play the same role as the original harness's
# depth-100 through depth-500 schedules without assuming a device clock cycle.
IDLE_DURATIONS = tuple(pytest.param(depth / 400.0, id=f"depth-{depth}") for depth in (100, 200, 300, 400, 500))


def _independent_distribution(excitation: float) -> ExpectedDistribution:
    stay = 1.0 - excitation
    return ExpectedDistribution(
        {
            (0, 0): stay * stay,
            (0, 1): stay * excitation,
            (1, 0): excitation * stay,
            (1, 1): excitation * excitation,
        },
    )


@pytest.mark.parametrize("family", IDLE_FAMILIES)
@pytest.mark.parametrize("duration", IDLE_DURATIONS)
def test_idle_family_duration_sweep(family: IdleFamily, duration: float) -> None:
    """Five schedule depths follow each idle family's independent analytic law."""
    excitation = family.excitation_probability(duration)
    if family.name == "coherent":
        statevec_module = pytest.importorskip("pecos_selene_statevec")
        simulator = statevec_module.StateVecPlugin(random_seed=900 + round(duration * 100))
    else:
        simulator = Stim(random_seed=900 + round(duration * 100))
    experiment = ConformanceExperiment(
        runner=IDLE_RUNNER,
        n_qubits=2,
        result_tags=("q0", "q1"),
        parameters=family.parameters(duration),
        expected=_independent_distribution(excitation),
        comparison=ExpectedDistribution({(0, 0): 1.0}),
        shots=4096,
        seed=800 + round(duration * 100) + len(family.name),
    )
    experiment.assert_conforms(simulator, n_processes=2)
