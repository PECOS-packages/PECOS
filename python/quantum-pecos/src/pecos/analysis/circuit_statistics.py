# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Condition-aware circuit execution statistics."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import TYPE_CHECKING, Any

from pecos.circuits.quantum_circuit import Location
from pecos.engines.cvm import DefaultClassicalSemantics

if TYPE_CHECKING:
    from pecos.circuits.quantum_circuit import TickView
    from pecos.engines.cvm import ClassicalSemantics


@dataclass(frozen=True, slots=True)
class OperationStatistic:
    """One count and/or duration contribution produced by a classifier.

    A classifier may return multiple contributions for one circuit operation.
    For example, a two-qubit gate can contribute to both a gate count and a
    parallel-layer count.
    """

    key: str
    count: int | None = None
    duration: float | None = None


OperationClassifier = Callable[
    [str, set[Location], Mapping[str, Any]],
    Iterable[OperationStatistic],
]


def classify_operations(
    symbol: str,
    locations: set[Location],
    metadata: Mapping[str, Any],
) -> tuple[OperationStatistic, ...]:
    """Classify an operation under its circuit symbol.

    The default policy counts each location, or one operation when there are
    no locations, and records an optional numeric ``duration`` metadata value.
    More specialized vocabularies can supply a custom classifier to
    :class:`CircuitStatistics`.
    """
    duration = metadata.get("duration")
    if duration is not None:
        try:
            duration = float(duration)
        except (TypeError, ValueError) as exc:
            msg = f"Operation duration must be numeric, got {duration!r}."
            raise TypeError(msg) from exc

    return (
        OperationStatistic(
            key=symbol,
            count=max(1, len(locations)),
            duration=duration,
        ),
    )


class CircuitStatistics:
    """Collect condition-aware counts and durations across circuit runs.

    Each operation contributes to three views:

    * ``max`` assumes every conditional operation executes.
    * ``min`` includes only unconditional operations.
    * ``runtime`` evaluates conditions against the current classical output.

    Call :meth:`new` before each run, :meth:`analyze` for every executed tick,
    and :meth:`finalize` after the run. The legacy hybrid engine calls
    :meth:`analyze`; execution wrappers remain responsible for the per-run
    lifecycle.
    """

    def __init__(
        self,
        classical_semantics: ClassicalSemantics | None = None,
        *,
        regwidth: int | None = None,
        classifier: OperationClassifier = classify_operations,
    ) -> None:
        """Create a statistics collector.

        Args:
            classical_semantics: Policy used to evaluate conditional
                operations. Defaults to PECOS's standard signed semantics.
            regwidth: Classical word width used for conditions. Defaults to the
                policy's configured width when available, otherwise 32.
            classifier: Converts circuit operations into one or more count or
                duration contributions.
        """
        resolved_semantics = classical_semantics if classical_semantics is not None else DefaultClassicalSemantics()
        resolved_width = regwidth if regwidth is not None else getattr(resolved_semantics, "width", 32)
        self.set_classical_semantics(
            resolved_semantics,
            regwidth=resolved_width,
        )
        if not callable(classifier):
            msg = "classifier must be callable."
            raise TypeError(msg)
        self.classifier = classifier
        self.data: dict[str, Any] = {"runs": []}
        self._current_data: dict[str, Any] | None = None

    def set_classical_semantics(
        self,
        classical_semantics: ClassicalSemantics,
        *,
        regwidth: int,
    ) -> None:
        """Set the policy and word width used for condition evaluation."""
        self.classical_semantics = classical_semantics
        self.regwidth = regwidth

    def eval_condition(
        self,
        condition: dict[str, Any] | tuple[Any, ...] | list[Any] | None,
        output: dict[str, Any],
    ) -> bool:
        """Evaluate an operation condition with the configured policy."""
        return self.classical_semantics.eval_condition(
            condition,
            output,
            width=self.regwidth,
        )

    def new(self) -> None:
        """Start collecting statistics for a new circuit run."""
        current_data = {
            "count": {
                "max": {},
                "min": {},
                "runtime": {},
            },
            "duration": {
                "max": {},
                "min": {},
                "runtime": {},
            },
        }
        self.data["runs"].append(current_data)
        self._current_data = current_data

    def _current_metric(self, metric: str) -> dict[str, dict[str, float | int]]:
        if self._current_data is None:
            msg = "Call new() before collecting circuit statistics."
            raise RuntimeError(msg)
        return self._current_data[metric]

    def add_count(
        self,
        key: str,
        count: int,
        condition: dict[str, Any] | tuple[Any, ...] | list[Any] | None,
        output: dict[str, Any],
    ) -> None:
        """Record one condition-aware count contribution."""
        counts = self._current_metric("count")
        counts["max"][key] = counts["max"].get(key, 0) + count
        if condition is None:
            counts["min"][key] = counts["min"].get(key, 0) + count
        if self.eval_condition(condition, output):
            counts["runtime"][key] = counts["runtime"].get(key, 0) + count

    def add_duration(
        self,
        key: str,
        duration: float,
        condition: dict[str, Any] | tuple[Any, ...] | list[Any] | None,
        output: dict[str, Any],
    ) -> None:
        """Record one condition-aware duration contribution."""
        durations = self._current_metric("duration")
        durations["max"][key] = durations["max"].get(key, 0.0) + duration
        if condition is None:
            durations["min"][key] = durations["min"].get(key, 0.0) + duration
        if self.eval_condition(condition, output):
            durations["runtime"][key] = durations["runtime"].get(key, 0.0) + duration

    def analyze(
        self,
        tick_circuit: TickView,
        time: int | tuple[int, ...],
        output: dict[str, Any],
    ) -> None:
        """Analyze every operation in an executed circuit tick."""
        del time
        self._current_metric("count")
        for symbol, locations, metadata in tick_circuit.items():
            condition = metadata.get("cond")
            for contribution in self.classifier(
                symbol,
                locations,
                metadata,
            ):
                if not isinstance(contribution, OperationStatistic):
                    msg = "classifier must return OperationStatistic objects."
                    raise TypeError(msg)
                if contribution.count is not None:
                    self.add_count(
                        contribution.key,
                        contribution.count,
                        condition,
                        output,
                    )
                if contribution.duration is not None:
                    self.add_duration(
                        contribution.key,
                        contribution.duration,
                        condition,
                        output,
                    )

    def finalize(self) -> None:
        """Update aggregate runtime-duration statistics across all runs."""
        runtimes = [sum(run["duration"]["runtime"].values()) for run in self.data["runs"]]
        if runtimes:
            average = fmean(runtimes)
            deviation = pstdev(runtimes)
        else:
            average = deviation = float("nan")
        self.data["total"] = {
            "runtime": runtimes,
            "avg_runtime": (average, deviation),
        }


__all__ = [
    "CircuitStatistics",
    "OperationClassifier",
    "OperationStatistic",
    "classify_operations",
]
