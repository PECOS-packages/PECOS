"""Device-neutral statistical conformance helpers for Selene noise plugins."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pecos_selene_general_noise import GeneralNoisePlugin

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pecos_selene_general_noise import GeneralNoiseParameters
    from selene_core import Simulator
    from selene_sim.instance import SeleneInstance

type Outcome = tuple[int, ...]


def _log_choose(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _fisher_exact_p_value(left: int, pooled: int, left_shots: int, total_shots: int) -> float:
    """Return the doubled smaller tail of a hypergeometric count."""
    low = max(0, left_shots - (total_shots - pooled))
    high = min(left_shots, pooled)
    log_norm = _log_choose(total_shots, left_shots)
    pmf = [
        math.exp(_log_choose(pooled, k) + _log_choose(total_shots - pooled, left_shots - k) - log_norm)
        for k in range(low, high + 1)
    ]
    lower_tail = math.fsum(pmf[: left - low + 1])
    upper_tail = math.fsum(pmf[left - low :])
    return min(1.0, 2.0 * min(lower_tail, upper_tail))


@dataclass(frozen=True)
class OutcomeSample:
    """Ordered outcomes from one reproducible Selene experiment."""

    outcomes: tuple[Outcome, ...]

    @property
    def counts(self) -> Counter[Outcome]:
        """Return the observed multinomial counts."""
        return Counter(self.outcomes)

    def assert_statistically_equivalent(
        self,
        other: OutcomeSample,
        *,
        failure_probability: float = 1e-6,
    ) -> None:
        """Compare two independent samples of Boolean outcome tuples.

        Each outcome gets Fisher's exact test: if both samples come from one
        distribution, an outcome's count in this sample, given its pooled count, is
        hypergeometric. A Bonferroni correction over all ``2**width`` Boolean outcomes
        bounds the probability of rejecting equal distributions by
        ``failure_probability``. The exact test adapts to each outcome's variance, so
        it resolves much smaller differences than a distribution-free Hoeffding bound.
        """
        left_shots = len(self.outcomes)
        right_shots = len(other.outcomes)
        if left_shots <= 0 or right_shots <= 0:
            message = "statistical equivalence requires two non-empty samples"
            raise ValueError(message)
        if not 0.0 < failure_probability < 1.0:
            message = "failure_probability must be between zero and one"
            raise ValueError(message)

        width = len(self.outcomes[0])
        left_counts = self.counts
        right_counts = other.counts
        if any(
            len(outcome) != width or not set(outcome) <= {0, 1} for outcome in left_counts.keys() | right_counts.keys()
        ):
            message = "statistical equivalence requires Boolean outcomes of one width"
            raise ValueError(message)

        threshold = failure_probability / 2**width
        mismatches = []
        for outcome in sorted(left_counts.keys() | right_counts.keys()):
            p_value = _fisher_exact_p_value(
                left_counts[outcome],
                left_counts[outcome] + right_counts[outcome],
                left_shots,
                left_shots + right_shots,
            )
            if p_value <= threshold:
                mismatches.append(
                    f"{outcome}: left={left_counts[outcome] / left_shots:.4f}, "
                    f"right={right_counts[outcome] / right_shots:.4f}, p={p_value:.2e}",
                )
        details = "; ".join(mismatches)
        message = f"empirical distributions differ at per-outcome threshold {threshold:.2e}: {details}"
        assert not mismatches, message


@dataclass(frozen=True)
class ExpectedDistribution:
    """Small exact reference distribution for a conformance circuit."""

    probabilities: Mapping[Outcome, float]

    def __post_init__(self) -> None:
        if not self.probabilities:
            message = "an expected distribution cannot be empty"
            raise ValueError(message)
        if any(probability < 0.0 or probability > 1.0 for probability in self.probabilities.values()):
            message = "expected probabilities must be between zero and one"
            raise ValueError(message)
        if not math.isclose(sum(self.probabilities.values()), 1.0, abs_tol=1e-12):
            message = "expected probabilities must sum to one"
            raise ValueError(message)

    def hoeffding_tolerance(self, shots: int, failure_probability: float = 1e-6) -> float:
        """Return a simultaneous absolute-frequency bound for every outcome.

        The union bound makes the probability that any expected category exceeds this
        tolerance at most ``failure_probability``. This avoids a SciPy dependency and,
        unlike a fixed tolerance, becomes stricter as the sample size grows.
        """
        if shots <= 0:
            message = "shots must be positive"
            raise ValueError(message)
        if not 0.0 < failure_probability < 1.0:
            message = "failure_probability must be between zero and one"
            raise ValueError(message)
        categories = max(1, len(self.probabilities))
        return math.sqrt(math.log(2.0 * categories / failure_probability) / (2.0 * shots))

    def total_variation_distance(self, other: ExpectedDistribution) -> float:
        """Return total variation distance from another exact distribution."""
        outcomes = self.probabilities.keys() | other.probabilities.keys()
        return 0.5 * sum(
            abs(self.probabilities.get(outcome, 0.0) - other.probabilities.get(outcome, 0.0)) for outcome in outcomes
        )

    def assert_sensitive_to(
        self,
        baseline: ExpectedDistribution,
        *,
        shots: int,
        failure_probability: float = 1e-6,
    ) -> None:
        """Reject a circuit whose expected signal is too close to its baseline."""
        tolerance = self.hoeffding_tolerance(shots, failure_probability)
        separation = self.total_variation_distance(baseline)
        assert separation > 2.0 * tolerance, (
            f"reference distribution is not sensitive enough: total variation {separation:.4f} "
            f"does not exceed twice the sampling tolerance {2.0 * tolerance:.4f}"
        )

    def assert_matches(
        self,
        sample: OutcomeSample,
        *,
        failure_probability: float = 1e-6,
    ) -> None:
        """Check every empirical frequency against its exact reference probability."""
        shots = len(sample.outcomes)
        tolerance = self.hoeffding_tolerance(shots, failure_probability)
        counts = sample.counts
        outcomes = self.probabilities.keys() | counts.keys()
        mismatches = []
        for outcome in sorted(outcomes):
            observed = counts[outcome] / shots
            expected = self.probabilities.get(outcome, 0.0)
            if abs(observed - expected) > tolerance:
                mismatches.append(
                    f"{outcome}: observed={observed:.4f}, expected={expected:.4f}",
                )
        details = "; ".join(mismatches)
        message = f"empirical distribution exceeds Hoeffding tolerance {tolerance:.4f}: {details}"
        assert not mismatches, message


@dataclass(frozen=True)
class ConformanceExperiment:
    """A simulator-independent Selene experiment with an exact reference model."""

    runner: SeleneInstance
    n_qubits: int
    result_tags: tuple[str, ...]
    parameters: GeneralNoiseParameters
    expected: ExpectedDistribution
    comparison: ExpectedDistribution
    shots: int = 512
    seed: int = 17

    def sample(self, simulator: Simulator, *, n_processes: int = 1) -> OutcomeSample:
        """Sample ordered bit outcomes through the actual native error-model plugin."""
        error_model = GeneralNoisePlugin(parameters=self.parameters, random_seed=self.seed)
        outcomes = []
        for shot in self.runner.run_shots(
            simulator=simulator,
            error_model=error_model,
            n_qubits=self.n_qubits,
            n_shots=self.shots,
            n_processes=n_processes,
        ):
            tagged = dict(shot)
            values = tuple(tagged[tag] for tag in self.result_tags)
            if not all(isinstance(value, (bool, int)) for value in values):
                message = f"conformance results must be scalar bits, got {values!r}"
                raise TypeError(message)
            outcomes.append(tuple(int(value) for value in values))
        return OutcomeSample(tuple(outcomes))

    def assert_conforms(self, simulator: Simulator, *, n_processes: int = 1) -> OutcomeSample:
        """Verify sensitivity first, then compare the native sample with the reference."""
        self.expected.assert_sensitive_to(self.comparison, shots=self.shots)
        sample = self.sample(simulator, n_processes=n_processes)
        self.expected.assert_matches(sample)
        return sample
