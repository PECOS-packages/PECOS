"""Typed measurement references and audited Guppy DEM build results."""

# Public ``id=`` mirrors DEM terminology; Rust-backed circuits are necessarily
# dynamic at this Python boundary. Error messages remain local and actionable.
# ruff: noqa: A002, ANN401, EM101, EM102, TRY003

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pecos.qec.guppy_output_dem import InferredGuppyDemAnnotations


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


@dataclass(frozen=True, slots=True)
class InferredOutput:
    """An element of a tag's flattened per-shot output to infer as a parity."""

    tag: str
    element: int = 0

    def __post_init__(self) -> None:
        """Validate the selector before executing a program."""
        if not isinstance(self.tag, str) or not self.tag:
            raise ValueError("inferred output tag must be a non-empty string")
        if isinstance(self.element, bool) or not isinstance(self.element, int) or self.element < 0:
            raise ValueError("inferred output element must be a non-negative integer")


@dataclass(frozen=True, slots=True, init=False)
class Detector:
    """A detector parity over canonical measurement references."""

    refs: tuple[MeasurementRef, ...]
    inferred_output: InferredOutput | None
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
        object.__setattr__(self, "inferred_output", None)
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "coords", tuple(float(value) for value in coords) if coords is not None else None)
        object.__setattr__(self, "metadata", dict(metadata) if metadata is not None else None)

    @classmethod
    def inferred(
        cls,
        tag: str,
        *,
        element: int = 0,
        id: int | None = None,
        coords: Sequence[float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Self:
        """Infer this definition from one flattened output element.

        The builder supplies default inference settings when none are given.
        Unselected outputs are not fitted or required.
        """
        selector = InferredOutput(tag, element)
        instance = object.__new__(cls)
        object.__setattr__(instance, "refs", ())
        object.__setattr__(instance, "inferred_output", selector)
        object.__setattr__(instance, "id", id)
        object.__setattr__(instance, "coords", tuple(float(value) for value in coords) if coords is not None else None)
        object.__setattr__(instance, "metadata", dict(metadata) if metadata is not None else None)
        return instance


@dataclass(frozen=True, slots=True, init=False)
class Observable:
    """A logical-observable parity over canonical measurement references."""

    refs: tuple[MeasurementRef, ...]
    inferred_output: InferredOutput | None
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
        object.__setattr__(self, "inferred_output", None)
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "metadata", dict(metadata) if metadata is not None else None)

    @classmethod
    def inferred(
        cls,
        tag: str,
        *,
        element: int = 0,
        id: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Self:
        """Infer this definition from one flattened output element.

        The builder supplies default inference settings when none are given.
        Unselected outputs are not fitted or required.
        """
        selector = InferredOutput(tag, element)
        instance = object.__new__(cls)
        object.__setattr__(instance, "refs", ())
        object.__setattr__(instance, "inferred_output", selector)
        object.__setattr__(instance, "id", id)
        object.__setattr__(instance, "metadata", dict(metadata) if metadata is not None else None)
        return instance


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


def _measurement_bit(value: Any, *, context: str) -> int:
    """Validate and normalize one externally supplied measurement bit."""
    if isinstance(value, Integral) and value in (0, 1):
        return int(value)
    raise ValueError(f"{context} must be bool, 0, or 1; got {value!r}")


def _result_values(raw: Any, *, tag: str, ids: tuple[int | None, ...]) -> list[int]:
    """Convert one named result while preserving unbound occurrence holes."""
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = list(raw)
    else:
        try:
            values = list(raw)
        except TypeError:
            values = [raw]
    return [
        0 if index < len(ids) and ids[index] is None else _measurement_bit(value, context=f"result {tag!r}")
        for index, value in enumerate(values)
    ]


def _definition_record(spec: Any, expected_type: type, role: str) -> tuple[Any, ...]:
    """Convert typed Python objects to the Rust definition value record."""
    if not isinstance(spec, expected_type):
        msg = f"with_{role}() requires {expected_type.__name__} definitions"
        raise TypeError(msg)
    if spec.id is not None and (isinstance(spec.id, bool) or not isinstance(spec.id, int) or spec.id < 0):
        msg = f"{role} ids must be non-negative integers"
        raise ValueError(msg)
    try:
        metadata = json.dumps(dict(spec.metadata or {}), allow_nan=False)
    except (TypeError, ValueError) as exc:
        msg = (
            "detector/observable metadata must be JSON-serializable "
            f"(it is embedded in the DEM schema and its fingerprint): {exc}"
        )
        raise ValueError(msg) from exc
    selector = spec.inferred_output
    references = (
        None
        if selector is not None
        else [
            -ref.offset if isinstance(ref, RecordRef) else (ref.tag, ref.occurrence, ref.element) for ref in spec.refs
        ]
    )
    return (
        spec.id,
        None,
        references,
        None if selector is None else (selector.tag, selector.element),
        list(spec.coords) if isinstance(spec, Detector) and spec.coords is not None else None,
        metadata,
    )


@dataclass(frozen=True, slots=True)
class GuppyDemBuild:
    """A DEM and its reusable Rust preparation, with detached circuit snapshots."""

    _build: Any
    _inference: InferredGuppyDemAnnotations | None = None
    _prepared: Any = field(init=False, repr=False)
    _dem: Any = field(init=False, repr=False)
    _detector_meas_ids: tuple[tuple[int, ...], ...] = field(init=False, repr=False)
    _observable_meas_ids: tuple[tuple[int, tuple[int, ...]], ...] = field(init=False, repr=False)
    _result_ids_by_tag: tuple[tuple[str, tuple[int | None, ...]], ...] = field(init=False, repr=False)
    _referenced_ids: frozenset[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Cache Rust objects and the value views needed for Python bit checks."""
        prepared = self._build.prepared
        ids = prepared.measurement_ids
        detectors = tuple(tuple(ids[position] for position in entry["measurements"]) for entry in prepared.detectors)
        observables = tuple(
            (entry["id"], tuple(ids[position] for position in entry["measurements"])) for entry in prepared.observables
        )
        referenced = frozenset(identity for support in detectors for identity in support) | frozenset(
            identity for _, support in observables for identity in support
        )
        for name, value in (
            ("_prepared", prepared),
            ("_dem", self._build.dem),
            ("_detector_meas_ids", detectors),
            ("_observable_meas_ids", observables),
            ("_result_ids_by_tag", tuple(prepared.named_result_measurements.items())),
            ("_referenced_ids", referenced),
        ):
            object.__setattr__(self, name, value)

    @property
    def dem(self) -> Any:
        """Return the Rust detector error model."""
        return self._dem

    @property
    def circuit(self) -> Any:
        """Return a detached snapshot of the prepared circuit."""
        return self._prepared.circuit

    @property
    def detectors_json(self) -> str:
        """Return detector metadata with computed references."""
        return self._prepared.detectors_json

    @property
    def observables_json(self) -> str:
        """Return observable metadata with computed references."""
        return self._prepared.observables_json

    @property
    def measurement_ledger(self) -> tuple[MeasurementLedgerEntry, ...]:
        """Return measurement identities and their certified result references."""
        return tuple(
            MeasurementLedgerEntry(
                meas_id=entry["meas_id"],
                runtime_record_index=entry["runtime_record_index"],
                canonical_record_index=entry["canonical_record_index"],
                result_refs=tuple(ResultRef(**ref) for ref in entry["result_refs"]),
            )
            for entry in self._prepared.measurement_ledger
        )

    @property
    def schema_fingerprint(self) -> str:
        """Return the prepared schema fingerprint."""
        return self._prepared.schema_fingerprint

    @property
    def named_result_binding(self) -> str:
        """Return the Rust classification of named measurement evidence."""
        return self._prepared.audit["named_result_binding"]

    @property
    def inference(self) -> InferredGuppyDemAnnotations | None:
        """Return empirical inference evidence when this build used inference."""
        return self._inference

    @property
    def definition_origins(self) -> tuple[Mapping[str, Any], ...]:
        """Return each definition's explicit or inferred origin."""
        return tuple(self._prepared.definition_origins)

    @property
    def audit(self) -> dict[str, Any]:
        """Return trace, measurement identity, and noise residual audit metadata."""
        audit = self._prepared.audit
        audit["idle_noise_residuals"] = self._build.idle_noise_residuals
        return audit

    def _runtime_bits(self, values: Sequence[int | bool]) -> list[int]:
        return [_measurement_bit(value, context=f"runtime measurement {index}") for index, value in enumerate(values)]

    def _measurement_bits(self, values: Mapping[int, int | bool]) -> dict[int, int]:
        return {
            identity: _measurement_bit(value, context=f"measurement MeasId {identity}")
            for identity, value in values.items()
            if identity in self._referenced_ids
        }

    def _result_bits(self, results: Mapping[str, Any]) -> dict[str, list[int]]:
        return {
            tag: _result_values(results[tag], tag=tag, ids=ids)
            for tag, ids in self._result_ids_by_tag
            if tag in results
        }

    def _result_columns(self, results: Mapping[str, Sequence[Any]]) -> dict[str, list[list[int]]]:
        return {
            tag: [_result_values(row, tag=tag, ids=ids) for row in results[tag]]
            for tag, ids in self._result_ids_by_tag
            if tag in results
        }

    def evaluate_runtime_record(self, values: Sequence[int | bool]) -> tuple[list[int], int]:
        """Evaluate detection events and observable flips from runtime record bits."""
        return self._prepared.evaluate_runtime_record(self._runtime_bits(values))

    def runtime_record_parities(self, values: Sequence[int | bool]) -> tuple[list[int], int]:
        """Evaluate raw parities from runtime record bits."""
        return self._prepared.runtime_record_parities(self._runtime_bits(values))

    def evaluate_measurements(self, values_by_meas_id: Mapping[int, int | bool]) -> tuple[list[int], int]:
        """Evaluate detection events and an arbitrary-precision observable flip mask."""
        return self._prepared.evaluate_measurements(self._measurement_bits(values_by_meas_id))

    def measurement_parities(self, values_by_meas_id: Mapping[int, int | bool]) -> tuple[list[int], int]:
        """Evaluate raw parities and an arbitrary-precision observable mask."""
        return self._prepared.measurement_parities(self._measurement_bits(values_by_meas_id))

    def evaluate_results(self, results: Mapping[str, Any]) -> tuple[list[int], int]:
        """Evaluate detection events and observable flips from one shot's results."""
        return self._prepared.evaluate_results(self._result_bits(results))

    def result_parities(self, results: Mapping[str, Any]) -> tuple[list[int], int]:
        """Evaluate raw parities from one shot's named results."""
        return self._prepared.result_parities(self._result_bits(results))

    def evaluate_result_columns(self, results: Mapping[str, Sequence[Any]]) -> list[tuple[list[int], int]]:
        """Evaluate PECOS ``ShotMap.to_dict()`` column-oriented results.

        The column mapping is a trusted carrier: shot ``i`` is whatever sits at
        row ``i`` of every column. Columns spliced from different runs, or one
        independently permuted column, are not detectable at this boundary --
        pass columns exactly as one simulation produced them.
        """
        return self._prepared.evaluate_result_columns(self._result_columns(results))

    def result_column_parities(self, results: Mapping[str, Sequence[Any]]) -> list[tuple[list[int], int]]:
        """Evaluate raw parities from trusted, row-aligned result columns."""
        return self._prepared.result_column_parities(self._result_columns(results))
