"""Typed measurement references and audited Guppy DEM build results."""

# Public ``id=`` mirrors DEM terminology; Rust-backed circuits are necessarily
# dynamic at this Python boundary. Error messages remain local and actionable.
# ruff: noqa: A002, ANN401, EM101, EM102, TRY003

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class RecordRef:
    """A Stim-style offset in the canonical Guppy measurement stream."""

    offset: int

    def __post_init__(self) -> None:
        """Validate the Stim-compatible offset."""
        if isinstance(self.offset, bool) or not isinstance(self.offset, int) or self.offset >= 0:
            msg = f"measurement record offsets must be negative integers, got {self.offset!r}"
            raise ValueError(msg)


class _RecordLookup:
    """Support the familiar ``rec[-k]`` spelling without carrying Stim."""

    __slots__ = ()

    def __getitem__(self, offset: int) -> RecordRef:
        return RecordRef(offset)


rec = _RecordLookup()


@dataclass(frozen=True, slots=True)
class ResultRef:
    """A measurement exposed through a Guppy ``result()`` call."""

    tag: str
    occurrence: int = 0
    element: int | None = None

    def __post_init__(self) -> None:
        """Validate the source result selector."""
        if not isinstance(self.tag, str) or not self.tag:
            raise ValueError("result_ref tag must be a non-empty string")
        if isinstance(self.occurrence, bool) or not isinstance(self.occurrence, int) or self.occurrence < 0:
            raise ValueError("result_ref occurrence must be a non-negative integer")
        if self.element is not None and (
            isinstance(self.element, bool) or not isinstance(self.element, int) or self.element < 0
        ):
            raise ValueError("result_ref element must be a non-negative integer")


def result_ref(tag: str, *, occurrence: int = 0, element: int | None = None) -> ResultRef:
    """Reference a direct scalar measurement emitted by Guppy ``result()``.

    ``element`` is reserved for future compiler-certified array provenance;
    nonzero/array-valued references currently fail closed during resolution.
    """
    return ResultRef(tag, occurrence=occurrence, element=element)


MeasurementRef = RecordRef | ResultRef


def _coerce_refs(refs: tuple[MeasurementRef | str, ...]) -> tuple[MeasurementRef, ...]:
    """Accept a bare tag string as shorthand for ``result_ref(tag)``."""
    if not refs:
        raise ValueError("detectors and observables must reference at least one measurement")
    coerced: list[MeasurementRef] = []
    for ref in refs:
        if isinstance(ref, str):
            coerced.append(ResultRef(ref))
        elif isinstance(ref, (RecordRef, ResultRef)):
            coerced.append(ref)
        else:
            msg = 'measurement references must be rec[...], result_ref(...), or a "tag" string'
            raise TypeError(msg)
    return tuple(coerced)


@dataclass(frozen=True, slots=True, init=False)
class Detector:
    """A detector parity over canonical measurement references."""

    refs: tuple[MeasurementRef, ...]
    id: int | None
    coords: tuple[float, ...] | None
    metadata: Mapping[str, Any] | None

    def __init__(
        self,
        *refs: MeasurementRef | str,
        id: int | None = None,
        coords: Sequence[float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Create a detector from measurement references.

        Each reference is ``rec[-k]``, ``result_ref(...)``, or a bare tag
        string, which is shorthand for ``result_ref(tag)``.
        """
        refs_tuple = _coerce_refs(tuple(refs))
        object.__setattr__(self, "refs", refs_tuple)
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "coords", tuple(float(value) for value in coords) if coords is not None else None)
        object.__setattr__(self, "metadata", dict(metadata) if metadata is not None else None)


@dataclass(frozen=True, slots=True, init=False)
class Observable:
    """A logical-observable parity over canonical measurement references."""

    refs: tuple[MeasurementRef, ...]
    id: int | None
    metadata: Mapping[str, Any] | None

    def __init__(
        self,
        *refs: MeasurementRef | str,
        id: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Create an observable from measurement references.

        Each reference is ``rec[-k]``, ``result_ref(...)``, or a bare tag
        string, which is shorthand for ``result_ref(tag)``.
        """
        refs_tuple = _coerce_refs(tuple(refs))
        object.__setattr__(self, "refs", refs_tuple)
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "metadata", dict(metadata) if metadata is not None else None)


def surface_memory_dem_spec(
    distance: int,
    num_rounds: int,
    basis: str,
    *,
    ancilla_budget: int | None = None,
    interaction_basis: str | None = None,
    check_plan: str | None = None,
    clifford_frame_policy: str | None = None,
) -> tuple[list[Detector], list[Observable]]:
    """Return typed detector and observable specs for ``make_surface_code``."""
    from pecos.qec.surface import SurfacePatch  # noqa: PLC0415
    from pecos.qec.surface.circuit_builder import generate_tick_circuit_from_patch  # noqa: PLC0415

    circuit = generate_tick_circuit_from_patch(
        SurfacePatch.create(distance=distance),
        num_rounds,
        basis,
        ancilla_budget=ancilla_budget,
        add_typed_annotations=False,
        interaction_basis=interaction_basis,
        check_plan=check_plan,
        clifford_frame_policy=clifford_frame_policy,
    )
    detector_entries = json.loads(circuit.get_meta("detectors") or "[]")
    observable_entries = json.loads(circuit.get_meta("observables") or "[]")

    def records_of(entry: Mapping[str, Any], kind: str) -> list[int]:
        records = entry.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError(
                f"surface builder emitted a {kind} entry without positional 'records' "
                f"({entry.get('id')!r}); surface_memory_dem_spec cannot convert it",
            )
        return [int(record) for record in records]

    detectors = []
    for entry in detector_entries:
        metadata = {key: value for key, value in entry.items() if key not in {"id", "records", "coords"}}
        detectors.append(
            Detector(
                *(rec[record] for record in records_of(entry, "detector")),
                id=int(entry["id"]),
                coords=entry.get("coords"),
                metadata=metadata or None,
            ),
        )
    observables = []
    for entry in observable_entries:
        metadata = {key: value for key, value in entry.items() if key not in {"id", "records"}}
        observables.append(
            Observable(
                *(rec[record] for record in records_of(entry, "observable")),
                id=int(entry["id"]),
                metadata=metadata or None,
            ),
        )
    return detectors, observables


@dataclass(frozen=True, slots=True)
class MeasurementLedgerEntry:
    """Auditable identity transport for one runtime measurement."""

    meas_id: int
    runtime_record_index: int
    canonical_record_index: int | None
    result_refs: tuple[ResultRef, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible audit record."""
        return {
            "meas_id": self.meas_id,
            "runtime_record_index": self.runtime_record_index,
            "canonical_record_index": self.canonical_record_index,
            "result_refs": [
                {"tag": ref.tag, "occurrence": ref.occurrence, "element": ref.element} for ref in self.result_refs
            ],
        }


@dataclass(frozen=True, slots=True)
class _ResolvedSchema:
    detectors_json: str
    observables_json: str
    detector_meas_ids: tuple[tuple[int, ...], ...]
    observable_meas_ids: tuple[tuple[int, tuple[int, ...]], ...]
    ledger: tuple[MeasurementLedgerEntry, ...]
    result_ids_by_tag: tuple[tuple[str, tuple[int | None, ...]], ...]
    schema_fingerprint: str


def _measurement_bit(value: Any, *, context: str) -> int:
    """Validate and normalize one externally supplied measurement bit."""
    if isinstance(value, Integral) and value in (0, 1):
        return int(value)
    raise ValueError(f"{context} must be bool, 0, or 1; got {value!r}")


@dataclass(frozen=True, slots=True)
class GuppyDemBuild:
    """A DEM and the single runtime trace from which it was constructed."""

    dem: Any
    circuit: Any
    detectors_json: str
    observables_json: str
    measurement_ledger: tuple[MeasurementLedgerEntry, ...]
    schema_fingerprint: str
    named_result_binding: str
    _detector_meas_ids: tuple[tuple[int, ...], ...]
    _observable_meas_ids: tuple[tuple[int, tuple[int, ...]], ...]
    _result_ids_by_tag: tuple[tuple[str, tuple[int | None, ...]], ...]

    @property
    def audit(self) -> dict[str, Any]:
        """Return trace and measurement-identity audit metadata."""
        runtime_order = [entry.meas_id for entry in self.measurement_ledger]
        return {
            "schema_fingerprint": self.schema_fingerprint,
            "soundness_scope": "static_schedule_only",
            "named_result_binding": self.named_result_binding,
            "measurement_count": len(runtime_order),
            "runtime_measurement_order": runtime_order,
            "runtime_order_is_canonical": runtime_order == list(range(len(runtime_order))),
            "runtime_order_mismatch_count": sum(index != meas_id for index, meas_id in enumerate(runtime_order)),
            "measurement_ledger": [entry.to_dict() for entry in self.measurement_ledger],
        }

    def evaluate_runtime_record(self, values: Sequence[int | bool]) -> tuple[list[int], int]:
        """Evaluate detectors/observables from values in runtime execution order."""
        if len(values) != len(self.measurement_ledger):
            msg = f"runtime measurement record has {len(values)} values; expected {len(self.measurement_ledger)}"
            raise ValueError(msg)
        by_id = {
            entry.meas_id: _measurement_bit(value, context=f"runtime measurement {entry.runtime_record_index}")
            for entry, value in zip(self.measurement_ledger, values, strict=True)
        }
        return self.evaluate_measurements(by_id)

    def evaluate_results(self, results: Mapping[str, Any]) -> tuple[list[int], int]:
        """Evaluate detectors/observables from one shot's named Guppy results."""
        by_id: dict[int, int] = {}
        for tag, result_ids in self._result_ids_by_tag:
            if tag not in results:
                continue
            raw_values = results[tag]
            if isinstance(raw_values, Sequence) and not isinstance(raw_values, (str, bytes)):
                values = list(raw_values)
            else:
                try:
                    values = list(raw_values)
                except TypeError:
                    values = [raw_values]
            if len(values) != len(result_ids):
                msg = f"result {tag!r} has {len(values)} values; trace provenance expects {len(result_ids)}"
                raise ValueError(msg)
            for meas_id, value in zip(result_ids, values, strict=True):
                if meas_id is None:
                    continue
                bit = _measurement_bit(value, context=f"result {tag!r}")
                if meas_id in by_id and by_id[meas_id] != bit:
                    raise ValueError(f"result outputs disagree about measurement MeasId {meas_id}")
                by_id[meas_id] = bit
        return self.evaluate_measurements(by_id)

    def evaluate_result_columns(self, results: Mapping[str, Sequence[Any]]) -> list[tuple[list[int], int]]:
        """Evaluate PECOS ``ShotMap.to_dict()`` column-oriented results.

        The column mapping is a trusted carrier: shot ``i`` is whatever sits at
        row ``i`` of every column. Columns spliced from different runs, or one
        independently permuted column, are not detectable at this boundary --
        pass columns exactly as one simulation produced them.
        """
        available = [tag for tag, _ in self._result_ids_by_tag if tag in results]
        if not available:
            raise ValueError("result columns do not contain any compiler- or generator-certified measurement tags")
        shot_count = len(results[available[0]])
        mismatched = {tag: len(results[tag]) for tag in available if len(results[tag]) != shot_count}
        if mismatched:
            raise ValueError(f"result columns have inconsistent shot counts: {mismatched}")
        return [self.evaluate_results({tag: results[tag][shot] for tag in available}) for shot in range(shot_count)]

    def evaluate_measurements(self, values_by_meas_id: Mapping[int, int | bool]) -> tuple[list[int], int]:
        """Evaluate the canonical detector vector and packed observable mask.

        The mask sets bit ``observable_id`` per observable. It is an
        arbitrary-precision Python int: with sparse or >=64 observable ids it
        exceeds 64 bits, so route it only to consumers that accept wide masks
        (narrow u64 decoders truncate silently).
        """

        def parity(meas_ids: tuple[int, ...]) -> int:
            missing = [meas_id for meas_id in meas_ids if meas_id not in values_by_meas_id]
            if missing:
                raise ValueError(f"measurement values are missing referenced MeasId(s): {missing[:8]}")
            return (
                sum(
                    _measurement_bit(values_by_meas_id[meas_id], context=f"measurement MeasId {meas_id}")
                    for meas_id in meas_ids
                )
                & 1
            )

        events = [parity(meas_ids) for meas_ids in self._detector_meas_ids]
        observable_mask = 0
        for observable_id, meas_ids in self._observable_meas_ids:
            observable_mask |= parity(meas_ids) << observable_id
        return events, observable_mask


def _measurement_ids_in_runtime_order(circuit: Any) -> list[int]:
    ids: list[int] = []
    for tick_index in range(circuit.num_ticks()):
        tick = circuit.get_tick(tick_index)
        if tick is None:
            continue
        for gate in tick.gate_batches():
            gate_type = str(gate.gate_type).rsplit(".", maxsplit=1)[-1]
            if gate_type not in {"MZ", "MeasureFree"}:
                continue
            qubits = list(gate.qubits)
            meas_ids = [int(meas_id) for meas_id in gate.meas_ids]
            if len(qubits) != len(meas_ids):
                raise ValueError(
                    f"traced measurement has {len(qubits)} qubit(s) but {len(meas_ids)} MeasId(s)",
                )
            ids.extend(meas_ids)
    duplicates = sorted(meas_id for meas_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"traced circuit contains duplicate MeasId(s): {duplicates[:8]}")
    return ids


def _index_result_traces(
    traces: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[int, tuple[int | None, ...]]], dict[int, list[ResultRef]]]:
    calls: dict[str, dict[int, tuple[int | None, ...]]] = defaultdict(dict)
    next_occurrence: dict[str, int] = defaultdict(int)
    refs_by_id: dict[int, list[ResultRef]] = defaultdict(list)
    for trace in traces:
        tag = trace.get("name")
        if not isinstance(tag, str):
            continue
        raw_occurrence = trace.get("occurrence")
        if raw_occurrence is None:
            occurrence = next_occurrence[tag]
        elif isinstance(raw_occurrence, bool) or not isinstance(raw_occurrence, int) or raw_occurrence < 0:
            raise ValueError(f"result trace {tag!r} has invalid occurrence {raw_occurrence!r}")
        else:
            occurrence = raw_occurrence
        next_occurrence[tag] = max(next_occurrence[tag], occurrence + 1)
        if occurrence in calls[tag]:
            raise ValueError(f"result trace {tag!r} repeats occurrence {occurrence}")

        values = trace.get("values")
        raw_ids = trace.get("result_ids")
        if not isinstance(values, list) or not isinstance(raw_ids, list) or len(values) != len(raw_ids) or not raw_ids:
            arity = max(len(values), 1) if isinstance(values, list) else 1
            calls[tag][occurrence] = (None,) * arity
            continue
        result_ids = tuple(int(result_id) for result_id in raw_ids)
        calls[tag][occurrence] = result_ids
        for element, meas_id in enumerate(result_ids):
            refs_by_id[meas_id].append(
                ResultRef(tag, occurrence=occurrence, element=element if len(result_ids) > 1 else None),
            )
    return calls, refs_by_id


def _resolve_ref(
    ref: MeasurementRef,
    *,
    num_measurements: int,
    runtime_meas_ids: set[int],
    result_calls: Mapping[str, Mapping[int, tuple[int | None, ...]]],
) -> int:
    if isinstance(ref, RecordRef):
        if sorted(runtime_meas_ids) != list(range(num_measurements)):
            raise ValueError(
                "rec[...] requires runtime MeasIds to preserve the canonical dense Guppy "
                "measurement identity range; use result_ref(...) for non-dense runtimes",
            )
        logical_index = num_measurements + ref.offset
        if logical_index < 0:
            raise ValueError(
                f"measurement record offset {ref.offset} is out of range for {num_measurements} measurements",
            )
        return logical_index

    calls = result_calls.get(ref.tag)
    if calls is None or ref.occurrence not in calls:
        raise ValueError(f"result_ref {ref.tag!r} occurrence {ref.occurrence} is absent from the runtime trace")
    result_ids = calls[ref.occurrence]
    if any(result_id is None for result_id in result_ids):
        raise ValueError(
            f"result_ref {ref.tag!r} occurrence {ref.occurrence} is not a direct scalar measurement result",
        )
    if len(result_ids) != 1:
        raise ValueError(
            "array-valued result_ref provenance is not yet certified; expose "
            "scalar result() tags or use canonical rec[...] references",
        )
    if ref.element is None:
        return int(result_ids[0])
    if ref.element not in (None, 0):
        raise ValueError(
            f"scalar result_ref {ref.tag!r} only accepts element=None or element=0",
        )
    return int(result_ids[0])


def _xor_normalize(meas_ids: Sequence[int], *, context: str) -> tuple[int, ...]:
    parity = Counter(meas_ids)
    normalized = tuple(meas_id for meas_id in dict.fromkeys(meas_ids) if parity[meas_id] & 1)
    if not normalized:
        # Every reference cancelled (even multiplicity throughout): the parity
        # is identically zero, which is almost certainly a spec mistake, not a
        # deliberately dead detector.
        raise ValueError(
            f"{context} references every measurement with even multiplicity; its parity is identically zero",
        )
    return normalized


def _metadata_entry(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    entry = dict(metadata or {})
    reserved = {"id", "records", "meas_ids", "result_tags"}.intersection(entry)
    if reserved:
        raise ValueError(f"detector/observable metadata uses reserved keys: {sorted(reserved)}")
    try:
        json.dumps(entry, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "detector/observable metadata must be JSON-serializable "
            f"(it is embedded in the DEM schema and its fingerprint): {exc}",
        ) from exc
    return entry


def _resolve_dem_specs(
    detectors: Sequence[Detector],
    observables: Sequence[Observable],
    *,
    circuit: Any,
    result_traces: Sequence[Mapping[str, Any]],
) -> _ResolvedSchema:
    runtime_order = _measurement_ids_in_runtime_order(circuit)
    num_measurements = len(runtime_order)
    runtime_ids = set(runtime_order)
    result_calls, refs_by_id = _index_result_traces(result_traces)

    resolved_detectors: list[tuple[int, dict[str, Any], tuple[int, ...]]] = []
    for position, detector in enumerate(detectors):
        detector_id = position if detector.id is None else detector.id
        if isinstance(detector_id, bool) or not isinstance(detector_id, int) or detector_id < 0:
            raise ValueError(f"detector id must be a non-negative integer, got {detector_id!r}")
        meas_ids = _xor_normalize(
            [
                _resolve_ref(
                    ref,
                    num_measurements=num_measurements,
                    runtime_meas_ids=runtime_ids,
                    result_calls=result_calls,
                )
                for ref in detector.refs
            ],
            context=f"detector {detector_id}",
        )
        entry = _metadata_entry(detector.metadata)
        entry.update({"id": detector_id, "meas_ids": list(meas_ids)})
        if detector.coords is not None:
            entry["coords"] = list(detector.coords)
        resolved_detectors.append((detector_id, entry, meas_ids))

    detector_ids = [item[0] for item in resolved_detectors]
    if sorted(detector_ids) != list(range(len(detector_ids))) or len(set(detector_ids)) != len(detector_ids):
        raise ValueError("detector ids must be unique and dense from 0 to len(detectors)-1")
    resolved_detectors.sort(key=lambda item: item[0])

    resolved_observables: list[tuple[int, dict[str, Any], tuple[int, ...]]] = []
    for position, observable in enumerate(observables):
        observable_id = position if observable.id is None else observable.id
        if isinstance(observable_id, bool) or not isinstance(observable_id, int) or observable_id < 0:
            raise ValueError(f"observable id must be a non-negative integer, got {observable_id!r}")
        meas_ids = _xor_normalize(
            [
                _resolve_ref(
                    ref,
                    num_measurements=num_measurements,
                    runtime_meas_ids=runtime_ids,
                    result_calls=result_calls,
                )
                for ref in observable.refs
            ],
            context=f"observable {observable_id}",
        )
        entry = _metadata_entry(observable.metadata)
        entry.update({"id": observable_id, "meas_ids": list(meas_ids)})
        resolved_observables.append((observable_id, entry, meas_ids))
    observable_ids = [item[0] for item in resolved_observables]
    if len(set(observable_ids)) != len(observable_ids):
        raise ValueError("observable ids must be unique")
    resolved_observables.sort(key=lambda item: item[0])

    detector_entries = [item[1] for item in resolved_detectors]
    observable_entries = [item[1] for item in resolved_observables]
    # Callers supply only compiler-certified direct scalar traces. Aggregate
    # array element order is not certified and must never become a shot binding.
    result_ids_by_tag = tuple(
        (
            tag,
            tuple(result_id for occurrence in range(len(calls)) for result_id in calls[occurrence]),
        )
        for tag, calls in sorted(result_calls.items())
        if calls
        and sorted(calls) == list(range(len(calls)))
        and all(len(call) == 1 or all(result_id is None for result_id in call) for call in calls.values())
        and all(result_id is None or isinstance(result_id, int) for call in calls.values() for result_id in call)
    )
    fingerprint_payload = {
        "detectors": detector_entries,
        "observables": observable_entries,
        "runtime_measurement_order": runtime_order,
        "named_result_measurements": result_ids_by_tag,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode(),
    ).hexdigest()

    dense_ids = sorted(runtime_ids) == list(range(num_measurements))
    ledger = tuple(
        MeasurementLedgerEntry(
            meas_id=meas_id,
            runtime_record_index=runtime_index,
            canonical_record_index=meas_id if dense_ids else None,
            result_refs=tuple(refs_by_id.get(meas_id, ())),
        )
        for runtime_index, meas_id in enumerate(runtime_order)
    )
    return _ResolvedSchema(
        detectors_json=json.dumps(detector_entries, separators=(",", ":")),
        observables_json=json.dumps(observable_entries, separators=(",", ":")),
        detector_meas_ids=tuple(item[2] for item in resolved_detectors),
        observable_meas_ids=tuple((item[0], item[2]) for item in resolved_observables),
        ledger=ledger,
        result_ids_by_tag=result_ids_by_tag,
        schema_fingerprint=fingerprint,
    )
