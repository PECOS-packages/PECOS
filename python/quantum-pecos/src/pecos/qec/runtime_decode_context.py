"""Version-one caller declarations for live decoder providers.

These records bind a supplied model to a program schema. They are not evidence
that the program used the declared reads, or a certificate of adaptive physics.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

MAX_CONTEXT_BYTES = 65536


def _identity(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
        msg = f"{name} must be a nonempty identity without control characters"
        raise ValueError(msg)
    # Reject unpaired surrogates even in escaped JSON strings.
    value.encode("utf-8")


def _uint(value: int, name: str) -> None:
    if type(value) is not int or not 0 <= value < 2**64:
        msg = f"{name} must be an unsigned 64-bit integer"
        raise ValueError(msg)


def _site(value: str) -> None:
    _identity(value, "site_id")
    if not value.startswith(("qis:", "hugr:")) or not value.split(":", 1)[1]:
        msg = "source sites must be qualified by qis: or hugr:"
        raise ValueError(msg)


def _tuple_of(owner: object, name: str, cls: type) -> tuple:
    values = tuple(getattr(owner, name))
    if any(type(value) is not cls for value in values):
        msg = f"{name} must contain {cls.__name__} values"
        raise TypeError(msg)
    object.__setattr__(owner, name, values)
    return values


def _unique(values: tuple, name: str) -> None:
    if len(set(values)) != len(values):
        msg = f"{name} must be unique"
        raise ValueError(msg)


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


@dataclass(frozen=True, kw_only=True)
class RuntimeRecordRef:
    """Named read occurrence within the current independent decode region.

    Occurrences start at zero for each name and region, in program read order.
    These are semantic read labels, not runtime future handles or QIR slots.
    """

    name: str
    occurrence: int

    def __post_init__(self) -> None:
        """Check a read identity."""
        _identity(self.name, "record name")
        _uint(self.occurrence, "record occurrence")


@dataclass(frozen=True, kw_only=True)
class RuntimeDetectorBinding:
    """One program byte: XOR of named reads and a Boolean reference sign.

    Terms are sorted canonically; duplicate terms are rejected, so the producer
    must perform GF(2) cancellation explicitly before declaring the schema.
    """

    detector_id: str
    records: tuple[RuntimeRecordRef, ...]
    reference: bool = False

    def __post_init__(self) -> None:
        """Normalize and validate the parity declaration."""
        _identity(self.detector_id, "detector_id")
        values = _tuple_of(self, "records", RuntimeRecordRef)
        _unique(values, "parity records")
        object.__setattr__(self, "records", tuple(sorted(values, key=lambda r: (r.name, r.occurrence))))
        if type(self.reference) is not bool:
            msg = "detector reference must be Boolean"
            raise ValueError(msg)


@dataclass(frozen=True, kw_only=True)
class RuntimeDecodeSchema:
    """Ordered program bytes and destination frame bits, independent of a DEM.

    ``schema_id`` is SHA-256 of ASCII JSON containing version=1, detectors and
    frame_ids. Keys are sorted; separators are ',' and ':'; non-ASCII characters
    use JSON escapes. Detector/frame order is significant; parity term order is
    normalized. No whitespace, NaN, or infinity is included.
    """

    detectors: tuple[RuntimeDetectorBinding, ...]
    frame_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Freeze names and check schema identities."""
        detectors = _tuple_of(self, "detectors", RuntimeDetectorBinding)
        frames = _tuple_of(self, "frame_ids", str)
        for name in frames:
            _identity(name, "frame identity")
        _unique(tuple(d.detector_id for d in detectors), "schema detector identities")
        _unique(frames, "frame identities")
        if len(frames) > 64:
            msg = "a runtime schema supports at most 64 frame bits"
            raise ValueError(msg)

    @property
    def schema_id(self) -> str:
        """Return the version-one content fingerprint, as lowercase SHA-256 hex."""
        return hashlib.sha256(_json({"version": 1, **asdict(self)})).hexdigest()


@dataclass(frozen=True, kw_only=True)
class RuntimeDecodeBinding:
    """Immutable program/compiler/site/schema identity expected by a window."""

    program_revision: str
    compiler_revision: str
    decode_site: str
    schema: RuntimeDecodeSchema

    def __post_init__(self) -> None:
        """Check that the binding names a complete schema and stable source site."""
        _identity(self.program_revision, "program_revision")
        _identity(self.compiler_revision, "compiler_revision")
        _site(self.decode_site)
        if type(self.schema) is not RuntimeDecodeSchema:
            msg = "binding requires a RuntimeDecodeSchema"
            raise TypeError(msg)

    def matches(self, context: RuntimeDecodeContext) -> bool:
        """Compare static binding identity, excluding dynamic execution context."""
        return (
            self.program_revision == context.program_revision
            and self.compiler_revision == context.compiler_revision
            and self.decode_site == context.decode_site
            and self.schema.schema_id == context.schema_id
        )


@dataclass(frozen=True, kw_only=True)
class RuntimeBranchRecord:
    """One declared branch event since the preceding decode (or shot start).

    Occurrence is zero-based per site within this interval. Order in branches is
    execution order. Neither outcome nor site is inferred from simulator truth.
    """

    site_id: str
    occurrence: int
    outcome: bool

    def __post_init__(self) -> None:
        """Check a branch occurrence."""
        _site(self.site_id)
        _uint(self.occurrence, "branch occurrence")
        if type(self.outcome) is not bool:
            msg = "branch outcome must be Boolean"
            raise ValueError(msg)


@dataclass(frozen=True, kw_only=True)
class RuntimeLoopRecord:
    """Active loop stack entry, ordered outermost first at the decode site.

    Occurrence numbers loop entries per site from shot start; iteration counts
    completed iterations of that entry. Future total trip counts are not assumed.
    """

    site_id: str
    occurrence: int
    iteration: int

    def __post_init__(self) -> None:
        """Check loop entry and completed-iteration counters."""
        _site(self.site_id)
        _uint(self.occurrence, "loop occurrence")
        _uint(self.iteration, "loop iteration")


@dataclass(frozen=True, kw_only=True)
class RuntimeInputDependency:
    """A declared source of a relevant classical input.

    For source='record', name/occurrence use RuntimeRecordRef's region-local
    read convention. For source='decoder', name is a declared frame identity and
    occurrence is a prior shot-local decode instance (strictly before this call).
    Neither declaration is verified against the running program's data flow.
    """

    source: str
    name: str
    occurrence: int

    def __post_init__(self) -> None:
        """Distinguish observed read and previous decoder-result dependencies."""
        if self.source not in ("record", "decoder"):
            msg = "input dependency source must be record or decoder"
            raise ValueError(msg)
        _identity(self.name, "dependency name")
        _uint(self.occurrence, "dependency occurrence")


@dataclass(frozen=True, kw_only=True)
class RuntimeClassicalInput:
    """Relevant caller-observed Boolean or 64-bit integer and its dependencies."""

    name: str
    kind: str
    value: bool | int
    dependencies: tuple[RuntimeInputDependency, ...]

    def __post_init__(self) -> None:
        """Reject implicit type coercions and unsupported classical types."""
        _identity(self.name, "input name")
        values = _tuple_of(self, "dependencies", RuntimeInputDependency)
        _unique(values, "input dependencies")
        valid = (self.kind == "bool" and type(self.value) is bool) or (
            self.kind in ("i64", "u64")
            and type(self.value) is int
            and (-(2**63) <= self.value < 2**63 if self.kind == "i64" else 0 <= self.value < 2**64)
        )
        if not valid:
            msg = "classical inputs require exact bool, i64 or u64 values"
            raise ValueError(msg)


@dataclass(frozen=True, kw_only=True)
class RuntimeDecodeContext:
    """Bounded caller-supplied context for the v1 intrinsic; not a cache key.

    Arrays must be supplied explicitly, including empty arrays. They are a local
    provenance declaration, not an automatic or complete execution certificate.
    No runtime batch/timing/noise or hidden simulator state is synthesized here.
    """

    version: int
    program_revision: str
    compiler_revision: str
    decode_site: str
    schema_id: str
    branches: tuple[RuntimeBranchRecord, ...]
    loops: tuple[RuntimeLoopRecord, ...]
    inputs: tuple[RuntimeClassicalInput, ...]

    def __post_init__(self) -> None:
        """Validate the version and freeze all nested provenance records."""
        if type(self.version) is not int or self.version != 1:
            msg = "unsupported runtime decode context version"
            raise ValueError(msg)
        _identity(self.program_revision, "program_revision")
        _identity(self.compiler_revision, "compiler_revision")
        _site(self.decode_site)
        if (
            not isinstance(self.schema_id, str)
            or len(self.schema_id) != 64
            or any(c not in "0123456789abcdef" for c in self.schema_id)
        ):
            msg = "schema_id must be a lowercase SHA-256 fingerprint"
            raise ValueError(msg)
        branches = _tuple_of(self, "branches", RuntimeBranchRecord)
        loops = _tuple_of(self, "loops", RuntimeLoopRecord)
        inputs = _tuple_of(self, "inputs", RuntimeClassicalInput)
        _unique(tuple((b.site_id, b.occurrence) for b in branches), "branch occurrences")
        _unique(tuple((loop.site_id, loop.occurrence) for loop in loops), "loop entries")
        _unique(tuple(value.name for value in inputs), "classical input names")
        next_occurrence = {}
        for branch in branches:
            if branch.occurrence != next_occurrence.get(branch.site_id, 0):
                msg = "branch occurrences must be contiguous per site within a context"
                raise ValueError(msg)
            next_occurrence[branch.site_id] = branch.occurrence + 1

    def to_json(self) -> bytes:
        """Encode the exact v1 envelope for a native call, with its byte bound."""
        data = _json(asdict(self))
        if len(data) > MAX_CONTEXT_BYTES:
            msg = "runtime decoder context exceeds 65536 bytes"
            raise ValueError(msg)
        return data

    @classmethod
    def from_json(cls, data: bytes) -> RuntimeDecodeContext:
        """Decode strict UTF-8 JSON; reject duplicate/unknown keys and coercions."""
        if type(data) is not bytes or not 0 < len(data) <= MAX_CONTEXT_BYTES:
            msg = "runtime decoder context must contain 1..65536 bytes"
            raise ValueError(msg)
        try:
            raw = _object(
                json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant),
            )
            # Explicit conversion leaves missing and unknown keys to constructors.
            raw["branches"] = tuple(RuntimeBranchRecord(**entry) for entry in _array(raw["branches"]))
            raw["loops"] = tuple(RuntimeLoopRecord(**entry) for entry in _array(raw["loops"]))
            inputs = []
            for item in _array(raw["inputs"]):
                entry = _object(item)
                entry["dependencies"] = tuple(RuntimeInputDependency(**dep) for dep in _array(entry["dependencies"]))
                inputs.append(RuntimeClassicalInput(**entry))
            raw["inputs"] = tuple(inputs)
            return cls(**raw)
        except (KeyError, TypeError, ValueError, RecursionError) as error:
            msg = f"invalid runtime decode context: {error}"
            raise ValueError(msg) from error


def _array(value: object) -> list:
    if type(value) is not list:
        msg = "context sequences must be JSON arrays"
        raise TypeError(msg)
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            msg = f"duplicate context key: {key}"
            raise ValueError(msg)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    msg = f"nonfinite JSON value: {value}"
    raise ValueError(msg)


def _object(value: object) -> dict:
    if type(value) is not dict:
        msg = "context objects must be JSON objects"
        raise TypeError(msg)
    return dict(value)
