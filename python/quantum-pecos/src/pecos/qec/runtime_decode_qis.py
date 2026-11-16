"""Schema-driven lowering of observed decoder sites into native QIS fragments.

This is a helper for frontend authors, not a QIS parser or physical-model
compiler. Assemble the fragments into a program and use compiler control
capture to check the complete module before executing it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from pecos.qec.runtime_decode_context import RuntimeDecodeBinding, RuntimeDecodeContext, RuntimeRecordRef, _uint
from pecos.qec.runtime_decoder import RuntimeDecodeWindow
from pecos.qec.runtime_observed_reads import MAX_NAME_BYTES, RuntimeReadSource, validate_read_sources

if TYPE_CHECKING:
    from collections.abc import Mapping


def _ssa(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"%[a-zA-Z$._][-a-zA-Z$._0-9]*", value) is None:
        msg = "operand must be a named LLVM SSA value"
        raise ValueError(msg)
    return value


def _constant(name: str, data: bytes) -> str:
    escaped = "".join(f"\\{byte:02X}" for byte in data)
    return f'@{name} = private constant [{len(data)} x i8] c"{escaped}"'


@dataclass(frozen=True, kw_only=True)
class RuntimeDecodeQisSite:
    """Lower one lexical decoder site from its existing immutable schema binding.

    ``name`` reserves ``pecos.decode.<name>.`` in global and local LLVM names;
    it must be unique in the assembled module. Place ``entry_ir`` once in the
    named entry block and ``read_ir`` once per record at the desired read point.
    Place ``decode_ir`` once after those reads. A loop may execute these static
    sites repeatedly. Reads of the same name must execute in occurrence order
    within each decode interval. LLVM dominance and the runtime source/parity
    checks remain authoritative; fragments alone do not certify a program.
    """

    binding: RuntimeDecodeBinding
    window_id: int
    name: str
    _records: tuple[RuntimeRecordRef, ...] = field(init=False, repr=False)
    _context_bytes: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Bound generated storage and reject unsupported identities/schemas."""
        if type(self.binding) is not RuntimeDecodeBinding:
            msg = "site requires a RuntimeDecodeBinding"
            raise TypeError(msg)
        _uint(self.window_id, "window_id")
        if not isinstance(self.name, str) or re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,63}", self.name) is None:
            msg = "site name must be an ASCII identifier of 1..64 characters"
            raise ValueError(msg)
        detectors = self.binding.schema.detectors
        if not 1 <= len(detectors) <= 4096 or sum(len(d.records) for d in detectors) > 4096:
            msg = "site requires 1..4096 detectors and at most 4096 parity terms"
            raise ValueError(msg)
        object.__setattr__(
            self,
            "_records",
            tuple(sorted({r for d in detectors for r in d.records}, key=lambda r: (r.name, r.occurrence))),
        )
        occurrences: dict[str, int] = {}
        for record in self.records:
            if len(record.name.encode("utf-8")) > MAX_NAME_BYTES:
                msg = "observed read name exceeds 256 UTF-8 bytes"
                raise ValueError(msg)
            if record.occurrence != occurrences.get(record.name, 0):
                msg = "site read occurrences must be contiguous per name from zero"
                raise ValueError(msg)
            occurrences[record.name] = record.occurrence + 1
        object.__setattr__(self, "_context_bytes", self._context())

    @staticmethod
    def declarations_ir() -> str:
        """Emit the two shared native imports, once per assembled module."""
        return (
            "declare i1 @pecos_runtime_read_bool_v1(ptr, i64, i64)\n"
            "declare i64 @pecos_runtime_decode_observed_v1(i64, i64, ptr, i64, ptr, i64)\n"
        )

    @property
    def records(self) -> tuple[RuntimeRecordRef, ...]:
        """Unique required semantic reads, sorted by name and occurrence."""
        return self._records

    @property
    def _prefix(self) -> str:
        return f"pecos.decode.{self.name}"

    def _context(self) -> bytes:
        return RuntimeDecodeContext(
            version=1,
            program_revision=self.binding.program_revision,
            compiler_revision=self.binding.compiler_revision,
            decode_site=self.binding.decode_site,
            schema_id=self.binding.schema.schema_id,
            branches=(),
            loops=(),
            inputs=(),
        ).to_json()

    def globals_ir(self) -> str:
        """Emit exact UTF-8 name lengths and the shared schema's constant context."""
        lines = [_constant(f"{self._prefix}.context", self._context_bytes)]
        lines.extend(_constant(f"{self._prefix}.name{i}", r.name.encode("utf-8")) for i, r in enumerate(self.records))
        return "\n".join(lines) + "\n"

    def entry_ir(self) -> str:
        """Emit bounded entry storage and direct constant byte pointers."""
        count = len(self.binding.schema.detectors)
        lines = [f"%{self._prefix}.data = alloca [{count} x i8]"]
        lines.extend(
            f"%{self._prefix}.slot{i} = getelementptr inbounds [{count} x i8], "
            f"ptr %{self._prefix}.data, i64 0, i64 {i}"
            for i in range(1, count)
        )
        return "\n".join(lines) + "\n"

    def read_value(self, record: RuntimeRecordRef) -> str:
        """Return the Boolean SSA name defined by this record's read fragment."""
        if type(record) is not RuntimeRecordRef or record not in self.records:
            msg = "read record is not in the site schema"
            raise ValueError(msg)
        return f"%{self._prefix}.read{self.records.index(record)}"

    def read_ir(self, record: RuntimeRecordRef, *, future: str) -> str:
        """Emit one observed getter at the caller-selected program position.

        ``future`` must be a native i64 future SSA value, not a measurement ID.
        The complete module checks its resource type; the runtime owns forcing.
        """
        result = self.read_value(record)
        index = self.records.index(record)
        return (
            f"{result} = call i1 @pecos_runtime_read_bool_v1(ptr @{self._prefix}.name{index}, "
            f"i64 {len(record.name.encode('utf-8'))}, i64 {_ssa(future)})\n"
        )

    @property
    def frame_value(self) -> str:
        """Return the i64 SSA name holding the decoder's program frame update."""
        return f"%{self._prefix}.frame"

    def decode_ir(self, *, instance: str | int) -> str:
        """Emit schema parities, fresh consecutive stores and the observed call.

        ``instance`` is an i64 SSA value or unsigned integer. It must advance
        contiguously across all decode sites in a shot, as checked by the host.
        This fragment adds no physical operation, delay or measurement read.
        """
        if isinstance(instance, str):
            instance = _ssa(instance)
        else:
            _uint(instance, "instance")
        calculations, stores = [], []
        for i, detector in enumerate(self.binding.schema.detectors):
            value = str(detector.reference).lower()
            for j, record in enumerate(detector.records):
                result = f"%{self._prefix}.parity{i}_{j}"
                calculations.append(f"{result} = xor i1 {value}, {self.read_value(record)}")
                value = result
            byte = f"%{self._prefix}.byte{i}"
            calculations.append(f"{byte} = zext i1 {value} to i8")
            pointer = f"%{self._prefix}.slot{i}" if i else f"%{self._prefix}.data"
            stores.append(f"store i8 {byte}, ptr {pointer}")
        call = (
            f"{self.frame_value} = call i64 @pecos_runtime_decode_observed_v1(i64 {self.window_id}, "
            f"i64 {instance}, ptr %{self._prefix}.data, i64 {len(stores)}, "
            f"ptr @{self._prefix}.context, i64 {len(self._context_bytes)})"
        )
        return "\n".join([*calculations, *stores, call]) + "\n"

    def bind_window(
        self,
        window: RuntimeDecodeWindow,
        *,
        source_ids: Mapping[RuntimeRecordRef, int],
    ) -> RuntimeDecodeWindow:
        """Bind an existing supplied model using explicit expected measurement IDs.

        Derive detector permutation from identities. Preserve output routing,
        fault ownership and independence declarations; never infer physical
        source IDs from future handles, decoder instances or hidden state.
        """
        if type(window) is not RuntimeDecodeWindow or window.window_id != self.window_id:
            msg = "window identity must match the QIS site"
            raise ValueError(msg)
        if window.binding is not None and window.binding != self.binding:
            msg = "window already has a different binding"
            raise ValueError(msg)
        if set(source_ids) != set(self.records) or any(type(r) is not RuntimeRecordRef for r in source_ids):
            msg = "source_ids must cover exactly the site's semantic reads"
            raise ValueError(msg)
        sources = tuple(RuntimeReadSource(record=r, source_id=source_ids[r]) for r in self.records)
        validate_read_sources(self.binding, sources)
        if window.read_sources is not None and {s.record: s.source_id for s in window.read_sources} != source_ids:
            msg = "window already has different read sources"
            raise ValueError(msg)
        ids = tuple(d.detector_id for d in self.binding.schema.detectors)
        if set(window.detector_ids) != set(ids):
            msg = "window detector identities must match the site schema"
            raise ValueError(msg)
        return replace(
            window,
            binding=self.binding,
            detector_order=tuple(ids.index(name) for name in window.detector_ids),
            read_sources=sources,
        )
