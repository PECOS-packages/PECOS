# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Concrete, static QEC gadget contracts shared by circuit backends.

These are construction adapters, not the proposed Rust InstrProgram graph.
Qubit indices name roles; RELABEL exchanges their physical carriers without
introducing a quantum gate. Parities reference immutable measurement ordinals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING

from pecos.qec.generic.guppy_ordering import GuppyOrdering, plan_guppy_ordering

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import stim

    from pecos.qec.dem_spec import Detector, Observable, ResultRef
    from pecos.qec.generic.guppy_audit import GuppyImplementationAudit
    from pecos.qec.generic.guppy_comparison import GuppyScheduleComparison
    from pecos.qec.generic.guppy_ordering import GuppyOrderingPlan
    from pecos.qec.generic.guppy_schedule import GuppyScheduleAudit
    from pecos.qec.generic.parity_audit import ScheduledParityAudit
    from pecos.quantum import TickCircuit


class SchedulePolicy(Enum):
    """Concrete scheduling choices, independent of the extraction construction."""

    REFERENCE = "reference"
    ASAP_UNITARIES = "asap_unitaries"
    COMPACT_UNITARIES = "compact_unitaries"


@dataclass(frozen=True)
class OperationOrigin:
    """Stable authoring location, retained when an operation is rescheduled."""

    call: str
    gadget: str
    moment: int
    operation: int

    @property
    def id(self) -> str:
        """Unique operation identity within its gadget program."""
        return f"{self.call}/moment[{self.moment}]/op[{self.operation}]"


@dataclass(frozen=True)
class ScheduleRow:
    """One inspectable placement with dependencies and reference carriers.

    Carriers are those used by the static reference allocation, not runtime Guppy
    site IDs. For RELABEL they identify the carriers immediately before exchange.
    """

    id: str
    moment: int
    gate: str
    roles: tuple[str, ...]
    carriers: tuple[int, ...]
    source: OperationOrigin | None
    measurement: str | None
    dependencies: tuple[str, ...]
    placement: str


@dataclass(frozen=True)
class ScheduleResources:
    """Resource counts for the selected static schedule, before Guppy lowering."""

    allocated_roles: int
    peak_live: int
    moments: int
    physical_moments: int
    physical_operations: int
    two_qubit_operations: int
    measurements: int
    relabels: int


@dataclass(frozen=True)
class CircuitOp:
    """A physical primitive, or an explicit zero-gate carrier relabel."""

    name: str
    qubits: tuple[int, ...]
    origin: OperationOrigin | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        """Validate operation vocabulary and operands."""
        arity = {
            "R": 1,
            "H": 1,
            "S": 1,
            "S_DAG": 1,
            "X": 1,
            "Z": 1,
            "CX": 2,
            "CZ": 2,
            "M": 1,
            "RELABEL": 2,
            "C_XYZ": 1,
            "C_ZYX": 1,
            "SQRT_X": 1,
        }
        if self.name not in arity or len(self.qubits) != arity[self.name]:
            msg = f"Unsupported operation or arity: {self}"
            raise ValueError(msg)
        if len(set(self.qubits)) != len(self.qubits) or any(type(q) is not int or q < 0 for q in self.qubits):
            msg = f"Invalid operation operands: {self.qubits}"
            raise ValueError(msg)


def _detector_coordinates(coords: Sequence[float] | None) -> tuple[float, ...] | None:
    """Snapshot finite numerical annotation coordinates without guessing axes."""
    if coords is None:
        return None
    values = tuple(coords)
    if any(isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) for value in values):
        msg = "detector coordinates must be finite real numbers"
        raise ValueError(msg)
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class MeasurementParity:
    """An XOR of raw events, an explicit constant, and optional detector coordinates.

    Coordinates are annotation metadata. They retain the author's axis meanings
    when a circuit is rescheduled; they do not prescribe physical gate timing.
    """

    records: tuple[int, ...]
    label: str
    constant: bool = False
    coords: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        """Require an explicit Boolean XOR constant."""
        if type(self.constant) is not bool:
            msg = "parity constant must be bool"
            raise TypeError(msg)
        object.__setattr__(self, "coords", _detector_coordinates(self.coords))

    def __xor__(self, value: bool | int) -> MeasurementParity:
        """Toggle an affine constant without changing the measurement support."""
        if type(value) not in (bool, int) or value not in (0, 1):
            msg = "XOR constant must be bool or integer 0/1"
            raise ValueError(msg)
        return MeasurementParity(self.records, self.label, self.constant ^ bool(value), self.coords)

    __rxor__ = __xor__

    def evaluate(self, record: Sequence[bool]) -> bool:
        """Evaluate this parity against canonical measurement events."""
        value = self.constant
        for index in self.records:
            value ^= bool(record[index])
        return value


@dataclass(frozen=True)
class ScheduledCircuit:
    """A complete static experiment with explicit reference moments and events.

    The selected schedule is authoritative for Stim and TickCircuit. Guppy export is
    functional: its compiler/runtime may reschedule and allocate physical sites.
    """

    num_qubits: int
    moments: tuple[tuple[CircuitOp, ...], ...]
    detectors: tuple[MeasurementParity, ...]
    observables: tuple[MeasurementParity, ...]
    citations: tuple[str, ...] = ()
    schedule: SchedulePolicy = SchedulePolicy.REFERENCE
    role_names: tuple[str, ...] = ()
    measurement_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Check disjoint moments, qubit lifetimes, and record references."""
        if not isinstance(self.schedule, SchedulePolicy):
            msg = "schedule must be a SchedulePolicy"
            raise TypeError(msg)
        if type(self.num_qubits) is not int or self.num_qubits < 1:
            msg = "num_qubits must be a positive integer"
            raise ValueError(msg)
        live: set[int] = set()
        count = 0
        for moment in self.moments:
            touched: set[int] = set()
            for op in moment:
                if any(q >= self.num_qubits for q in op.qubits) or touched.intersection(op.qubits):
                    msg = "A moment has out-of-range or overlapping operands"
                    raise ValueError(msg)
                touched.update(op.qubits)
                if op.name == "R":
                    if op.qubits[0] in live:
                        msg = "Reset must begin a fresh lifetime after measurement"
                        raise ValueError(msg)
                    live.add(op.qubits[0])
                else:
                    if not set(op.qubits) <= live:
                        msg = f"Operation uses a non-live qubit: {op}"
                        raise ValueError(msg)
                    if op.name == "M":
                        live.remove(op.qubits[0])
                        count += 1
        if live:
            msg = "A complete experiment must consume all qubits"
            raise ValueError(msg)
        for names, size in ((self.role_names, self.num_qubits), (self.measurement_names, count)):
            if names and (len(names) != size or len(set(names)) != size or any(not n for n in names)):
                msg = "Named roles/events must be unique and match the complete canonical allocation"
                raise ValueError(msg)
        for parity in (*self.detectors, *self.observables):
            if len(set(parity.records)) != len(parity.records):
                msg = "Parity records must be XOR-normalized"
                raise ValueError(msg)
            if any(type(index) is not int or not 0 <= index < count for index in parity.records):
                msg = "Parity references a nonexistent measurement"
                raise ValueError(msg)
        if any(parity.coords is not None for parity in self.observables):
            msg = "Coordinates belong to detector annotations, not observables"
            raise ValueError(msg)

    @property
    def num_measurements(self) -> int:
        """Number of raw measurement events in the canonical program."""
        return sum(op.name == "M" for moment in self.moments for op in moment)

    def audit_parities(self) -> ScheduledParityAudit:
        """Check every authored detector/observable is identically zero ideally.

        Affine constants are checked, not removed. Reports distinguish an always
        one expression from a nondeterministic expression. This closed-circuit
        check does not certify arbitrary-input gadget action or Guppy lowering.
        """
        from pecos.qec.generic.parity_audit import audit_scheduled_parities  # noqa: PLC0415

        return audit_scheduled_parities(self)

    @property
    def output_carriers(self) -> tuple[int, ...]:
        """Final role-to-reference-carrier permutation, including retired roles."""
        carriers = list(range(self.num_qubits))
        for moment in self.moments:
            for op in moment:
                if op.name == "RELABEL":
                    a, b = op.qubits
                    carriers[a], carriers[b] = carriers[b], carriers[a]
        return tuple(carriers)

    def rescheduled(self, policy: SchedulePolicy) -> ScheduledCircuit:
        """Pack unitary regions while preserving reset/measurement/relabel barriers.

        Per-wire order, measurement order, and carrier transitions are retained.
        COMPACT_UNITARIES also preserves the complete flattened source order:
        it packs consecutive disjoint operations without allowing overtaking.
        This changes idle locations and is not the paper's noise benchmark.
        Reconstruct the experiment to recover the original reference schedule.
        """
        if not isinstance(policy, SchedulePolicy):
            msg = "policy must be a SchedulePolicy"
            raise TypeError(msg)
        if policy == self.schedule:
            return self
        if policy == SchedulePolicy.REFERENCE:
            msg = "The reference schedule cannot be recovered from an optimized plan"
            raise ValueError(msg)
        packed = []
        region: list[list[CircuitOp]] = []
        last: dict[int, int] = {}
        for moment in self.moments:
            if any(op.name in {"R", "M", "RELABEL"} for op in moment):
                packed.extend(map(tuple, region))
                packed.append(moment)
                region, last = [], {}
            else:
                for op in moment:
                    depth = max(last.get(q, -1) for q in op.qubits) + 1
                    if policy == SchedulePolicy.COMPACT_UNITARIES:
                        depth = max(depth, len(region) - 1)
                    while len(region) <= depth:
                        region.append([])
                    region[depth].append(op)
                    for q in op.qubits:
                        last[q] = depth
        packed.extend(map(tuple, region))
        return ScheduledCircuit(
            self.num_qubits,
            tuple(packed),
            self.detectors,
            self.observables,
            self.citations,
            policy,
            self.role_names,
            self.measurement_names,
        )

    @property
    def measurement_aliases(self) -> dict[str, str]:
        """Map human-readable measurement names to stable raw Guppy result tags."""
        names = self.measurement_names or tuple(f"measurement[{i}]" for i in range(self.num_measurements))
        return {name: f"qec:m:{i}" for i, name in enumerate(names)}

    def measurement_manifest(self) -> dict:
        """Export JSON-ready raw-event metadata and separate signed QEC definitions.

        This is a source-layout description. Use audit_guppy() separately for
        compiler-bound measurement evidence. Each call returns a detached dict.
        """
        from pecos.qec.generic.measurement_manifest import measurement_manifest  # noqa: PLC0415

        return measurement_manifest(self)

    @property
    def resources(self) -> ScheduleResources:
        """Count live roles and physical work without counting relabels as gates."""
        live = peak = physical = two_qubit = measurements = relabels = ticks = 0
        for moment in self.moments:
            live += sum(op.name == "R" for op in moment)
            peak = max(peak, live)
            live -= sum(op.name == "M" for op in moment)
            physical += sum(op.name != "RELABEL" for op in moment)
            two_qubit += sum(op.name in {"CX", "CZ"} for op in moment)
            measurements += sum(op.name == "M" for op in moment)
            relabels += sum(op.name == "RELABEL" for op in moment)
            ticks += not moment or any(op.name != "RELABEL" for op in moment)
        return ScheduleResources(
            self.num_qubits,
            peak,
            len(self.moments),
            ticks,
            physical,
            two_qubit,
            measurements,
            relabels,
        )

    def inspect(self) -> tuple[ScheduleRow, ...]:
        """Expose placements, per-role dependencies, and reset/readout boundaries.

        Reference mode retains authored moments. ASAP mode uses per-role dependencies
        within unitary regions; any reset, measurement or relabel moment bounds the
        whole region. Pure relabel moments carry no physical duration in Stim export.
        """
        names = self.role_names or tuple(f"q[{i}]" for i in range(self.num_qubits))
        events = tuple(self.measurement_aliases)
        carriers = list(range(self.num_qubits))
        last: dict[int, str] = {}
        barrier: tuple[str, ...] = ()
        region: list[str] = []
        rows = []
        measurement = 0
        for layer, moment in enumerate(self.moments):
            boundary = any(op.name in {"R", "M", "RELABEL"} for op in moment)
            ids = tuple(op.origin.id if op.origin else f"moment[{layer}]/op[{i}]" for i, op in enumerate(moment))
            for op, identity in zip(moment, ids, strict=True):
                deps = set(barrier)
                deps.update(last[q] for q in op.qubits if q in last)
                if boundary:
                    deps.update(region)
                placement = {
                    SchedulePolicy.REFERENCE: "authored reference moment",
                    SchedulePolicy.ASAP_UNITARIES: "ASAP per-role order",
                    SchedulePolicy.COMPACT_UNITARIES: "compact consecutive operations; preserve source order",
                }[self.schedule]
                if boundary:
                    placement += "; reset/measurement/relabel boundary"
                event = events[measurement] if op.name == "M" else None
                rows.append(
                    ScheduleRow(
                        identity,
                        layer,
                        op.name,
                        tuple(names[q] for q in op.qubits),
                        tuple(carriers[q] for q in op.qubits),
                        op.origin,
                        event,
                        tuple(sorted(deps)),
                        placement,
                    ),
                )
                if op.name == "M":
                    measurement += 1
                if op.name == "RELABEL":
                    a, b = op.qubits
                    carriers[a], carriers[b] = carriers[b], carriers[a]
                for q in op.qubits:
                    last[q] = identity
            if boundary:
                barrier, region = ids, []
            else:
                region.extend(ids)
        return tuple(rows)

    def describe(self) -> str:
        """Render an inspectable schedule with named roles, events, and origins."""
        lines = [
            f"{self.schedule.value}: {len(self.moments)} moments, {self.num_qubits} allocated roles, "
            f"{self.num_measurements} measurements",
        ]
        for row in self.inspect():
            source = f" [{row.source.id}]" if row.source else ""
            outcome = f" -> {row.measurement}" if row.measurement else ""
            operands = ", ".join(f"{r}@{q}" for r, q in zip(row.roles, row.carriers, strict=True))
            lines.append(f"  {row.moment:3}: {row.gate} {operands}{outcome}{source}")
        return "\n".join(lines)

    def explain_detector(self, label: str) -> str:
        """Trace a detector to named measurement events and their gadget sources."""
        for detector in self.detectors:
            if detector.label == label:
                events = tuple(self.measurement_aliases)
                terms = [events[i] for i in detector.records]
                if detector.constant:
                    terms.append("1")
                return f"{label} = " + (" ^ ".join(terms) or "0")
        raise KeyError(label)

    def explain_operation(self, identity: str) -> str:
        """Explain one placement using its stable authoring identity."""
        for row in self.inspect():
            if row.id == identity:
                return (
                    f"{identity}: {row.gate} at moment {row.moment}\n"
                    f"  placement: {row.placement}\n"
                    f"  predecessors: {', '.join(row.dependencies) or '(none)'}\n"
                    f"  roles: {', '.join(row.roles)}; reference carriers: {row.carriers}"
                )
        raise KeyError(identity)

    def to_stim(self) -> stim.Circuit:
        """Export physical moments and parities, using a final MPAD for constants.

        A signed export appends one synthetic record containing 1, without a
        physical gate or tick. Use evaluate_stim_record for these padded samples.
        Stim's detector sampler reports deviations from its own reference sample,
        which is distinct from evaluating the authored affine expression.
        """
        import stim  # noqa: PLC0415 - optional backend

        circuit = stim.Circuit()
        carriers = list(range(self.num_qubits))
        for moment in self.moments:
            physical = False
            for op in moment:
                if op.name == "RELABEL":
                    a, b = op.qubits
                    carriers[a], carriers[b] = carriers[b], carriers[a]
                else:
                    circuit.append(op.name, [carriers[q] for q in op.qubits])
                    physical = True
            if physical or not moment:
                circuit.append("TICK")
        count = self.num_measurements
        if any(p.constant for p in (*self.detectors, *self.observables)):
            circuit.append("MPAD", [1])
            count += 1
        for kind, parities in (("DETECTOR", self.detectors), ("OBSERVABLE_INCLUDE", self.observables)):
            for index, parity in enumerate(parities):
                targets = [stim.target_rec(i - count) for i in parity.records]
                if parity.constant:
                    targets.append(stim.target_rec(-1))
                circuit.append(kind, targets, [index] if kind == "OBSERVABLE_INCLUDE" else (parity.coords or ()))
        return circuit

    def guppy_ordering_plan(self, ordering: GuppyOrdering = GuppyOrdering.FUNCTIONAL) -> GuppyOrderingPlan:
        """Inspect compiler-ordering coverage and runtime helper cost before lowering."""
        return plan_guppy_ordering(self, ordering)

    def guppy_source(
        self,
        *,
        function_name: str = "memory",
        ordering: GuppyOrdering = GuppyOrdering.FUNCTIONAL,
        allocation_provenance: bool = False,
    ) -> str:
        """Render executable Guppy with stable scalar event names.

        allocation_provenance adds required reset trace labels under ALL_MOMENTS.
        These may introduce local runtime barriers/drains without adding gates.
        """
        from pecos.guppy_gen.scheduled import render_scheduled_circuit  # noqa: PLC0415 - lazy backend

        return render_scheduled_circuit(
            self,
            function_name=function_name,
            ordering=ordering,
            allocation_provenance=allocation_provenance,
        )

    def to_tick_circuit(self) -> TickCircuit:
        """Lower to native PECOS ticks with stable measurement IDs and parities.

        RELABEL changes carriers without a SWAP or an extra tick. Explicit empty
        moments are retained. Source roles, event names, and relabels remain in
        ``qec_*`` metadata. Measurement IDs are canonical event ordinals; detector
        offsets are resolved against the backend's actual emission order.
        Affine constants are retained in parity metadata for evaluate_tick_circuit.
        Native detector and observable definitions still require nonempty supports.
        """
        from pecos.qec.generic.tick_lowering import lower_tick_circuit  # noqa: PLC0415 - lazy backend

        return lower_tick_circuit(self)

    def guppy_program(
        self,
        *,
        ordering: GuppyOrdering = GuppyOrdering.FUNCTIONAL,
        allocation_provenance: bool = False,
    ) -> object:
        """Compile/load Guppy with an explicit, independently inspectable ordering contract."""
        from pecos.guppy_gen._module_loader import load_cached_guppy_source  # noqa: PLC0415 - lazy compiler

        return load_cached_guppy_source(
            "scheduled_qec",
            self.guppy_source(
                ordering=ordering,
                allocation_provenance=allocation_provenance,
            ),
        )["memory"]

    def audit_guppy(
        self,
        *,
        runtime: object | None = None,
        seed: int = 0,
        ordering: GuppyOrdering = GuppyOrdering.FUNCTIONAL,
    ) -> GuppyImplementationAudit:
        """Inspect runtime lowering and measurement bindings without assuming timing equivalence."""
        from pecos.qec.generic.guppy_audit import audit_guppy_implementation  # noqa: PLC0415 - optional backend

        return audit_guppy_implementation(self, runtime=runtime, seed=seed, ordering=ordering)

    def audit_guppy_schedule(
        self,
        *,
        runtime: object,
        seed: int = 0,
        ordering: GuppyOrdering = GuppyOrdering.ALL_MOMENTS,
        batch_adapter_factory: object | None = None,
        include_lifetimes: bool = False,
    ) -> GuppyScheduleAudit:
        """Inspect an explicit runtime plugin's batch clocks without assuming calibrated timing.

        include_lifetimes tags preparations to bind source endpoints to site
        windows. Its clocks describe tagged execution, which may schedule differently.
        """
        from pecos.qec.generic.guppy_schedule import audit_guppy_schedule  # noqa: PLC0415 - optional backend

        return audit_guppy_schedule(
            self,
            runtime=runtime,
            seed=seed,
            ordering=ordering,
            batch_adapter_factory=batch_adapter_factory,
            include_lifetimes=include_lifetimes,
        )

    def compare_guppy_schedules(
        self,
        *,
        runtime: object,
        seed: int = 0,
        candidate_policy: SchedulePolicy = SchedulePolicy.ASAP_UNITARIES,
    ) -> GuppyScheduleComparison:
        """Compare this reference plan with a selected policy under one runtime profile."""
        from pecos.qec.generic.guppy_comparison import compare_guppy_schedules  # noqa: PLC0415

        return compare_guppy_schedules(self, runtime=runtime, seed=seed, candidate_policy=candidate_policy)

    def dem_spec(self) -> tuple[list[Detector], list[Observable]]:
        """Return typed PECOS detector/observable specifications for raw events."""
        from pecos.qec.dem_spec import Detector, Observable, result_ref  # noqa: PLC0415 - lazy backend

        def refs(parity: MeasurementParity) -> list[ResultRef]:
            if not parity.records:
                msg = "The current DEM schema requires nonempty parities"
                raise ValueError(msg)
            return [result_ref(f"qec:m:{i}") for i in parity.records]

        return (
            [Detector(*refs(p), id=i, constant=p.constant, coords=p.coords) for i, p in enumerate(self.detectors)],
            [Observable(*refs(p), id=i, constant=p.constant) for i, p in enumerate(self.observables)],
        )

    def evaluate_stim_record(self, record: Sequence[bool]) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
        """Evaluate a raw to_stim() sample, validating any synthetic constant bit."""
        if any(p.constant for p in (*self.detectors, *self.observables)):
            if len(record) != self.num_measurements + 1 or record[-1] != 1:
                msg = "Signed Stim sample must end with the synthetic MPAD 1 record"
                raise ValueError(msg)
            record = record[:-1]
        return self.evaluate(record)

    def evaluate(self, record: Sequence[bool]) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
        """Evaluate raw canonical records, without conflating raw bits and checks."""
        if len(record) != self.num_measurements:
            msg = "Incorrect measurement record length"
            raise ValueError(msg)
        return tuple(p.evaluate(record) for p in self.detectors), tuple(p.evaluate(record) for p in self.observables)

    def evaluate_named(self, results: Mapping[str, bool]) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
        """Evaluate one Guppy shot by event name, independent of runtime order."""
        return self.evaluate([results[f"qec:m:{i}"] for i in range(self.num_measurements)])
