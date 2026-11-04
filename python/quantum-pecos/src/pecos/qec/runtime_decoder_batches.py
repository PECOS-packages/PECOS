"""Outcome-free snapshots of the runtime-emitted prefix at a live decode.

These describe runtime-emitted work, not a DEM, a compiler certificate or a
promise that unforced or future work is available. Source IDs identify measured
occurrences; no native future handles or measurement values are exposed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from pecos.qec.runtime_decode_context import _array, _object, _reject_constant, _uint, _unique_object


@dataclass(frozen=True, kw_only=True)
class RuntimeDecodeOperation:
    """A runtime-lowered operation; Idle parameters use seconds, angles radians.

    ``source_id`` is a shot-local source measurement occurrence, or None for a
    runtime-owned measurement with no program producer. This is not a static
    compiler site or fault identity. Nonmeasurement operations also use None.
    """

    kind: str
    qubits: tuple[int, ...]
    parameters: tuple[float, ...] = ()
    source_id: int | None = None

    def __post_init__(self) -> None:
        """Freeze nested sequences even for explicitly constructed records."""
        object.__setattr__(self, "qubits", tuple(self.qubits))
        object.__setattr__(self, "parameters", tuple(self.parameters))


@dataclass(frozen=True, kw_only=True)
class RuntimeDecodeBatch:
    """One emitted batch, retaining order and integer nanosecond batch timing.

    Operations include the existing lowering's inserted idle gaps, which can
    begin before this batch's start. They exclude physical noise realizations.
    ``normalization_applied`` records use of a runtime batch adapter; opaque
    original custom payloads and decorative source metadata are not retained.
    """

    batch_index: int
    start_time_nanos: int
    duration_nanos: int
    normalization_applied: bool
    operations: tuple[RuntimeDecodeOperation, ...]

    def __post_init__(self) -> None:
        """Retain an immutable operation sequence."""
        object.__setattr__(self, "operations", tuple(self.operations))


@dataclass(frozen=True, kw_only=True)
class RuntimeDecodeBatchPrefix:
    """A bounded interval since the prior decode, with an exclusive watermark.

    Shot IDs are local to the engine, not globally unique worker identities.
    Empty intervals are meaningful. The host publishes a prefix after its ready
    commands and feedback have executed, without flushing unread futures.
    """

    runtime_shot_id: int
    first_batch_index: int
    next_batch_index: int
    batches: tuple[RuntimeDecodeBatch, ...]

    def __post_init__(self) -> None:
        """Retain an immutable batch sequence."""
        object.__setattr__(self, "batches", tuple(self.batches))

    @classmethod
    def from_json(cls, data: bytes) -> RuntimeDecodeBatchPrefix:
        """Validate the native bridge's bounded version-one payload."""
        if type(data) is not bytes or not 0 < len(data) <= 32 * 1024 * 1024:
            msg = "decoder batch prefix must contain 1..33554432 bytes"
            raise ValueError(msg)
        try:
            raw = _object(json.loads(data, object_pairs_hook=_unique_object, parse_constant=_reject_constant))
            return _parse_prefix(raw)
        except (KeyError, TypeError, ValueError, RecursionError) as error:
            msg = f"invalid decoder batch prefix: {error}"
            raise ValueError(msg) from error


def _parse_prefix(raw: dict) -> RuntimeDecodeBatchPrefix:
    if set(raw) != {"version", "runtime_shot_id", "first_batch_index", "next_batch_index", "batches"}:
        msg = "unexpected prefix fields"
        raise ValueError(msg)
    version = raw.pop("version")
    if type(version) is not int or version != 1:
        msg = "unsupported prefix version"
        raise ValueError(msg)
    for key in ("runtime_shot_id", "first_batch_index", "next_batch_index"):
        _uint(raw[key], key)
    batches = _array(raw["batches"])
    if len(batches) > 4096 or raw["next_batch_index"] - raw["first_batch_index"] != len(batches):
        msg = "decoder batch prefix sequence or batch limit invalid"
        raise ValueError(msg)
    parsed = []
    count = 0
    for index, item in enumerate(batches, raw["first_batch_index"]):
        batch = _object(item)
        if set(batch) != {
            "batch_index",
            "start_time_nanos",
            "duration_nanos",
            "normalization_applied",
            "operations",
        }:
            msg = "unexpected batch fields"
            raise ValueError(msg)
        for key in ("batch_index", "start_time_nanos", "duration_nanos"):
            _uint(batch[key], key)
        if batch["batch_index"] != index or type(batch["normalization_applied"]) is not bool:
            msg = "invalid batch identity or normalization flag"
            raise ValueError(msg)
        _uint(batch["start_time_nanos"] + batch["duration_nanos"], "batch end time")
        operations = _array(batch["operations"])
        count += len(operations)
        if count > 65536:
            msg = "decoder batch operation limit exceeded"
            raise ValueError(msg)
        batch["operations"] = tuple(_operation(op) for op in operations)
        parsed.append(RuntimeDecodeBatch(**batch))
    raw["batches"] = tuple(parsed)
    return RuntimeDecodeBatchPrefix(**raw)


def _operation(raw: object) -> RuntimeDecodeOperation:
    value = _object(raw)
    if len(value) != 1:
        msg = "expected one operation variant"
        raise ValueError(msg)
    kind, args = next(iter(value.items()))
    # Exactly the native batch conversion's output vocabulary, not arbitrary
    # source QIS operations. Keep runtime result handles out of this contract.
    shape = {"RXY": (2, 1), "RZ": (1, 1), "RZZ": (1, 2), "RXYXY2Q": (2, 2), "Idle": (1, 1)}
    source_id = None
    if kind in shape:
        angles, qubits = shape[kind]
        args = _array(args)
        if len(args) != angles + qubits:
            msg = "invalid operation arity"
            raise ValueError(msg)
        params = args[:angles]
        targets = args[angles:]
    elif kind in ("Measure", "MeasureLeaked"):
        args = _array(args)
        if len(args) != 2:
            msg = "invalid measurement arity"
            raise ValueError(msg)
        target, source = args
        _uint(source, "source_id")
        if source == 2**64 - 2:
            msg = "reserved decoder response identity"
            raise ValueError(msg)
        source_id = None if source == 2**64 - 1 else source
        params, targets = [], [target]
    elif kind == "Reset":
        params, targets = [], [args]
    else:
        msg = f"unsupported lowered batch operation {kind}"
        raise ValueError(msg)
    for q in targets:
        _uint(q, "qubit")
    if len(set(targets)) != len(targets):
        msg = "duplicate operation qubit"
        raise ValueError(msg)
    if any(type(p) not in (int, float) or not math.isfinite(p) for p in params) or (kind == "Idle" and params[0] < 0):
        msg = "invalid operation parameter"
        raise ValueError(msg)
    return RuntimeDecodeOperation(kind=kind, qubits=tuple(targets), parameters=tuple(params), source_id=source_id)
