"""Explicit count conventions for reproducible QEC memory comparisons.

The equal-weight estimator follows the released analysis for Shutty,
arXiv:2609.21376v3, DOI 10.5281/zenodo.22820614. It is not a universal
normalization: other memory experiments may use different exposure conventions.
The pooled Wilson estimator separately implements the linear exposure convention
in Lavasani et al., arXiv:2610.24077v1.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, expm1, isfinite, log1p, sqrt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class MemoryCounts:
    """One independent basis/phase stratum; failure means any logical mismatch."""

    shots: int
    failures: int
    unresolved: int = 0

    def __post_init__(self) -> None:
        """Validate counts without interpreting unresolved outcomes as successes."""
        if (
            type(self.shots) is not int
            or type(self.failures) is not int
            or type(self.unresolved) is not int
            or self.shots <= 0
            or not 0 <= self.failures <= self.shots
            or not 0 <= self.unresolved <= self.shots
        ):
            msg = "Counts require positive integer shots and integer failures/unresolved in [0, shots]"
            raise ValueError(msg)


@dataclass(frozen=True)
class MemoryEstimate:
    """Equal-stratum block estimate and transformed simultaneous interval."""

    shots: int
    failures: int
    strata: int
    block_error_rate: float
    rate_per_round_per_logical: float
    interval: tuple[float, float]
    confidence: float
    num_logicals: int
    rounds: int


@dataclass(frozen=True)
class PooledMemoryEstimate:
    """Pooled trial counts with a Wilson interval and linear exposure normalization."""

    shots: int
    failures: int
    pooled_groups: int
    block_error_rate: float
    rate_per_cycle: float
    interval: tuple[float, float]
    confidence: float
    z_score: float
    rounds: int
    initialization_rounds: float


def pooled_wilson_memory_estimate(
    counts: Sequence[MemoryCounts],
    *,
    rounds: int,
    initialization_rounds: float = 0.5,
    z_score: float = 1.0,
) -> PooledMemoryEstimate:
    """Pool failure/shot counts, then divide the Wilson estimate by exposure.

    Defaults follow Lavasani et al., arXiv:2610.24077v1, Table I and the
    supplementary neural-decoder analysis: LER=(failures/shots)/(r+1/2),
    with one-sigma (z=1, approximately 68.27%) Wilson intervals. This linear
    convention differs from the equal-stratum transform used by Shutty.

    Counts are logical-memory trials, already including any pooled patches,
    signs, or bases. Do not multiply by the number of parallel physical patches
    again or combine cumulative/overlapping records. The interval assumes
    independent trials and does not quantify device/emulator mismatch or
    between-training-run variability. Unresolved outcomes prevent an estimate.
    """
    if (
        not counts
        or type(rounds) is not int
        or rounds < 1
        or type(initialization_rounds) not in (int, float)
        or not isfinite(initialization_rounds)
        or initialization_rounds < 0
        or type(z_score) not in (int, float)
        or not isfinite(z_score)
        or z_score <= 0
        or not isfinite(z_score * z_score)
    ):
        msg = "Pooled estimate requires counts, positive integer rounds, finite nonnegative overhead, and positive z"
        raise ValueError(msg)
    if any(c.unresolved for c in counts):
        msg = "Unresolved decoder outcomes cannot be included in an error-rate estimate"
        raise ValueError(msg)
    shots = sum(c.shots for c in counts)
    failures = sum(c.failures for c in counts)
    rate = failures / shots
    z2 = z_score * z_score
    denominator = 1 + z2 / shots
    center = (rate + z2 / (2 * shots)) / denominator
    half_width = z_score * sqrt((rate * (1 - rate) + z2 / (4 * shots)) / shots) / denominator
    lower = 0.0 if failures == 0 else max(0.0, center - half_width)
    upper = 1.0 if failures == shots else min(1.0, center + half_width)
    exposure = rounds + initialization_rounds
    return PooledMemoryEstimate(
        shots,
        failures,
        len(counts),
        rate,
        rate / exposure,
        (lower / exposure, upper / exposure),
        erf(z_score / sqrt(2)),
        z_score,
        rounds,
        initialization_rounds,
    )


def equal_weight_memory_estimate(
    counts: Sequence[MemoryCounts],
    *,
    num_logicals: int,
    rounds: int,
    confidence: float = 0.95,
) -> MemoryEstimate:
    """Weight basis/phase strata equally, even when their shot counts differ.

    Use Bonferroni-adjusted Clopper-Pearson intervals per stratum, average their
    endpoints, then apply 1-(1-P)**(1/(k*r)). Do not merge cumulative duplicate
    records into ``counts`` or omit required basis/phase strata. Unresolved decoder
    outcomes prevent a report. For adaptively allocated samples these intervals
    are descriptive, not confidence sequences. Requires SciPy.
    """
    from scipy.stats import beta  # noqa: PLC0415 - optional analysis dependency

    if (
        not counts
        or type(num_logicals) is not int
        or num_logicals < 1
        or type(rounds) is not int
        or rounds < 1
        or not isfinite(confidence)
        or not 0 < confidence < 1
    ):
        msg = "Estimate requires sampled strata, positive integer dimensions, and confidence in (0, 1)"
        raise ValueError(msg)
    if any(c.unresolved for c in counts):
        msg = "Unresolved decoder outcomes cannot be included in an error-rate estimate"
        raise ValueError(msg)
    tail = (1 - confidence) / (2 * len(counts))
    intervals = [
        (
            0.0 if c.failures == 0 else float(beta.ppf(tail, c.failures, c.shots - c.failures + 1)),
            1.0 if c.failures == c.shots else float(beta.ppf(1 - tail, c.failures + 1, c.shots - c.failures)),
        )
        for c in counts
    ]

    def transform(p: float) -> float:
        return 1.0 if p == 1.0 else -expm1(log1p(-p) / (num_logicals * rounds))

    mean = sum(c.failures / c.shots for c in counts) / len(counts)
    return MemoryEstimate(
        shots=sum(c.shots for c in counts),
        failures=sum(c.failures for c in counts),
        strata=len(counts),
        block_error_rate=mean,
        rate_per_round_per_logical=transform(mean),
        interval=tuple(transform(sum(pair[i] for pair in intervals) / len(counts)) for i in (0, 1)),
        confidence=confidence,
        num_logicals=num_logicals,
        rounds=rounds,
    )
