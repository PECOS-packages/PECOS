"""Expected measurement occurrences and checks on native Boolean getter observations.

Source IDs are shot-local dense measurement occurrences assigned by the QIS FFI,
not future handles or static compiler sites. Expectations remain caller supplied.
"""

from dataclasses import dataclass

from pecos.qec.runtime_decode_context import RuntimeDecodeBinding, RuntimeRecordRef, _uint

MAX_READS = 65536
MAX_NAME_BYTES = 256
MAX_TOTAL_NAME_BYTES = 1048576


@dataclass(frozen=True, kw_only=True)
class RuntimeReadSource:
    """Bind one region-local named read to an expected shot-local measurement ID."""

    record: RuntimeRecordRef
    source_id: int

    def __post_init__(self) -> None:
        """Reject unsupported read labels and reserved source identities."""
        if type(self.record) is not RuntimeRecordRef:
            msg = "read source requires a RuntimeRecordRef"
            raise TypeError(msg)
        if len(self.record.name.encode("utf-8")) > MAX_NAME_BYTES:
            msg = "observed read name exceeds 256 UTF-8 bytes"
            raise ValueError(msg)
        _uint(self.source_id, "source_id")
        if self.source_id >= 2**64 - 2:
            msg = "reserved measurement source identity"
            raise ValueError(msg)


def validate_read_sources(binding: RuntimeDecodeBinding | None, sources: tuple[RuntimeReadSource, ...]) -> None:
    """Require an exact, bounded source mapping for the schema's read terms."""
    if binding is None:
        msg = "read_sources requires a schema binding"
        raise ValueError(msg)
    if any(type(source) is not RuntimeReadSource for source in sources):
        msg = "read_sources must contain RuntimeReadSource values"
        raise TypeError(msg)
    records = [source.record for source in sources]
    expected = {record for detector in binding.schema.detectors for record in detector.records}
    if len(set(records)) != len(records) or set(records) != expected:
        msg = "read_sources must cover exactly the unique schema records"
        raise ValueError(msg)
    if len(records) > MAX_READS or sum(len(record.name.encode("utf-8")) for record in records) > MAX_TOTAL_NAME_BYTES:
        msg = "read_sources exceeds observation limits"
        raise ValueError(msg)
    by_name = {}
    for record in records:
        by_name.setdefault(record.name, []).append(record.occurrence)
    if any(sorted(values) != list(range(len(values))) for values in by_name.values()):
        msg = "expected read occurrences must be contiguous per name from zero"
        raise ValueError(msg)


def check_observed_reads(
    binding: RuntimeDecodeBinding,
    sources: tuple[RuntimeReadSource, ...],
    syndrome: list[int] | bytes,
    observations: list[tuple[str, int, int, bool]],
) -> None:
    """Check the separate runtime channel before constructing any model decoder."""
    if type(observations) is not list or len(observations) > MAX_READS:
        msg = "observations must be a bounded list of native read tuples"
        raise ValueError(msg)
    values = {}
    actual_sources = {}
    next_occurrence = {}
    name_bytes = 0
    for observation in observations:
        if type(observation) is not tuple or len(observation) != 4:
            msg = "invalid observed read tuple"
            raise ValueError(msg)
        name, occurrence, source_id, value = observation
        record = RuntimeRecordRef(name=name, occurrence=occurrence)
        RuntimeReadSource(record=record, source_id=source_id)
        name_bytes += len(name.encode("utf-8"))
        if name_bytes > MAX_TOTAL_NAME_BYTES or occurrence != next_occurrence.get(name, 0):
            msg = "observed read occurrences or name byte limit invalid"
            raise ValueError(msg)
        if type(value) is not bool:
            msg = "observed read value must be Boolean"
            raise ValueError(msg)
        next_occurrence[name] = occurrence + 1
        values[record] = value
        actual_sources[record] = source_id
    if actual_sources != {source.record: source.source_id for source in sources}:
        msg = "observed read sources do not match the window's expected records and measurement identities"
        raise ValueError(msg)
    for byte, detector in zip(syndrome, binding.schema.detectors, strict=True):
        parity = detector.reference
        for record in detector.records:
            parity ^= values[record]
        if byte != parity:
            msg = f"observed read parity disagrees with detector {detector.detector_id}"
            raise ValueError(msg)
