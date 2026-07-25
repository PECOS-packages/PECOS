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
from copy import deepcopy
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import TYPE_CHECKING, Any, Literal, TypedDict

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


# Numeric value stored for a count or duration.
StatisticValue = int | float


class ExecutionViews(TypedDict):
    """Maximum, minimum, and actual runtime values for one metric."""

    max: dict[str, StatisticValue]
    min: dict[str, StatisticValue]
    runtime: dict[str, StatisticValue]


class CircuitRunStatistics(TypedDict):
    """Counts and durations collected for one circuit execution."""

    count: ExecutionViews
    duration: ExecutionViews


class RuntimeSummary(TypedDict):
    """Runtime-duration summary across completed runs."""

    runtime: list[StatisticValue]
    avg_runtime: tuple[float, float]


class CircuitStatisticsData(TypedDict):
    """Complete circuit-statistics result schema."""

    runs: list[CircuitRunStatistics]
    total: RuntimeSummary


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

    :class:`~pecos.HybridEngine` manages the per-run lifecycle and passes actual
    operation execution decisions to :meth:`analyze_operation`. Manual callers
    can use :meth:`start_run`, :meth:`analyze`, and :meth:`finish_run`.
    """

    supports_operation_events = True

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
        self.data: CircuitStatisticsData = {
            "runs": [],
            "total": {
                "runtime": [],
                "avg_runtime": (float("nan"), float("nan")),
            },
        }
        self._current_data: CircuitRunStatistics | None = None

    @property
    def run_active(self) -> bool:
        """Whether a run is currently accepting operation statistics."""
        return self._current_data is not None

    @property
    def results(self) -> CircuitStatisticsData:
        """Return a defensive copy of the collected result."""
        return deepcopy(self.data)

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

    def start_run(self) -> None:
        """Start collecting a new run.

        Raises:
            RuntimeError: If the previous run has not been finished.
        """
        if self.run_active:
            msg = "Finish the active run before starting another."
            raise RuntimeError(msg)
        current_data: CircuitRunStatistics = {
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

    def new(self) -> None:
        """Alias for :meth:`start_run`."""
        self.start_run()

    def _current_metric(self, metric: Literal["count", "duration"]) -> ExecutionViews:
        if self._current_data is None:
            msg = "Call start_run() before collecting circuit statistics."
            raise RuntimeError(msg)
        return self._current_data[metric]

    @staticmethod
    def _record(
        values: ExecutionViews,
        key: str,
        value: StatisticValue,
        *,
        conditional: bool,
        executed: bool,
    ) -> None:
        values["max"][key] = values["max"].get(key, 0) + value
        if not conditional:
            values["min"][key] = values["min"].get(key, 0) + value
        if executed:
            values["runtime"][key] = values["runtime"].get(key, 0) + value

    def add_count(
        self,
        key: str,
        count: int,
        condition: dict[str, Any] | tuple[Any, ...] | list[Any] | None,
        output: dict[str, Any],
    ) -> None:
        """Record one condition-aware count contribution."""
        self._record(
            self._current_metric("count"),
            key,
            count,
            conditional=bool(condition),
            executed=self.eval_condition(condition, output),
        )

    def add_duration(
        self,
        key: str,
        duration: float,
        condition: dict[str, Any] | tuple[Any, ...] | list[Any] | None,
        output: dict[str, Any],
    ) -> None:
        """Record one condition-aware duration contribution."""
        self._record(
            self._current_metric("duration"),
            key,
            duration,
            conditional=bool(condition),
            executed=self.eval_condition(condition, output),
        )

    def analyze_operation(
        self,
        symbol: str,
        locations: set[Location],
        metadata: Mapping[str, Any],
        *,
        executed: bool,
    ) -> None:
        """Record one non-skipped operation using the engine's decision."""
        self._current_metric("count")
        conditional = bool(metadata.get("cond")) or bool(metadata.get("cond2"))
        for contribution in self.classifier(symbol, locations, metadata):
            if not isinstance(contribution, OperationStatistic):
                msg = "classifier must return OperationStatistic objects."
                raise TypeError(msg)
            if contribution.count is not None:
                self._record(
                    self._current_metric("count"),
                    contribution.key,
                    contribution.count,
                    conditional=conditional,
                    executed=executed,
                )
            if contribution.duration is not None:
                self._record(
                    self._current_metric("duration"),
                    contribution.key,
                    contribution.duration,
                    conditional=conditional,
                    executed=executed,
                )

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
            if metadata.get("skip"):
                continue
            condition = metadata.get("cond")
            condition2 = metadata.get("cond2")
            executed = self.eval_condition(condition, output)
            if condition2:
                executed = executed and self.eval_condition(condition2, output)
            self.analyze_operation(
                symbol,
                locations,
                metadata,
                executed=executed,
            )

    def _update_summary(self) -> None:
        runtimes = [sum(run["duration"]["runtime"].values()) for run in self.data["runs"]]
        average = (fmean(runtimes), pstdev(runtimes)) if runtimes else (float("nan"), float("nan"))
        self.data["total"] = {
            "runtime": runtimes,
            "avg_runtime": average,
        }

    def finish_run(self) -> None:
        """Finish the active run and update aggregate runtime statistics."""
        if not self.run_active:
            msg = "Call start_run() before finishing circuit statistics."
            raise RuntimeError(msg)
        self._update_summary()
        self._current_data = None

    def abort_run(self) -> None:
        """Discard the active run after an unsuccessful engine execution."""
        if not self.run_active:
            msg = "Call start_run() before aborting circuit statistics."
            raise RuntimeError(msg)
        if not self.data["runs"] or self.data["runs"][-1] is not self._current_data:
            msg = "The active circuit-statistics run is not the latest run."
            raise RuntimeError(msg)
        self.data["runs"].pop()
        self._current_data = None
        self._update_summary()

    def finalize(self) -> None:
        """Alias for :meth:`finish_run`."""
        self.finish_run()


__all__ = [
    "CircuitRunStatistics",
    "CircuitStatistics",
    "CircuitStatisticsData",
    "ExecutionViews",
    "OperationClassifier",
    "OperationStatistic",
    "RuntimeSummary",
    "StatisticValue",
    "classify_operations",
]
