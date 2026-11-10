"""Hosted-operation metadata utilities.

A hosted operation is a source-local operation whose semantic role is tied to a
later host operation.  For example, an SZZ surface-code data-prefix pulse can
declare ``host_id=<source SZZ id>`` and ``local_role=basis_prefix`` while the
lowered SZZ/SZZdg host carries the same ``host_id``.  Runtimes are free to
lower and schedule the gates, but traced circuits can then fail loudly if the
source-host relationship was lost or separated too far.
"""

from __future__ import annotations

import operator
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pecos_rslib import _validate_hosted_operations

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


HOST_ID_META_KEY = "host_id"
LOCAL_ROLE_META_KEY = "local_role"
HOSTED_PROVENANCE_META_KEYS = (
    "source_kind",
    "source_label",
    "source_gate",
    "szz_host_label",
    "source_lowering_required",
    "label",
)


@dataclass(frozen=True)
class HostedGateRecord:
    """A traced gate carrying hosted-operation metadata."""

    tick_index: int
    gate_index: int
    gate_name: str
    qubits: tuple[int, ...]
    host_id: str
    local_role: str
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class HostedOperationBinding:
    """A concrete local-to-host relationship recovered from traced metadata."""

    host_id: str
    local_role: str
    local: HostedGateRecord
    host: HostedGateRecord
    tick_separation: int


def validate_hosted_operations(
    tick_circuit: object,
    *,
    host_id_key: str = HOST_ID_META_KEY,
    local_role_key: str = LOCAL_ROLE_META_KEY,
    max_tick_separation: int | None = None,
    require_shared_qubit: bool = True,
    require_host_after_local: bool = True,
    require_unique_host_id: bool = False,
    context: str = "hosted operation validation",
) -> tuple[HostedOperationBinding, ...]:
    """Validate and bind hosted-operation metadata in a traced ``TickCircuit``.

    A gate with ``local_role_key`` is a hosted local operation and must carry a
    non-empty ``host_id_key``.  By default it binds to the first later gate with
    the same host id and no local role.  When ``require_shared_qubit`` is true,
    the bound host must touch at least one of the local gate's qubits.

    Args:
        tick_circuit: PECOS ``TickCircuit``-like object with ``num_ticks()``,
            ``get_tick()``, tick ``gate_batches()``, and optional
            ``get_gate_meta(tick, gate, key)`` support.
        host_id_key: Metadata key naming the source host relationship.
        local_role_key: Metadata key identifying local operations.
        max_tick_separation: Optional maximum absolute signed tick separation
            between the local and selected host operation, in the unsigned 64-bit range.
        require_shared_qubit: Require the selected host to share a qubit with
            the local operation.
        require_host_after_local: Require host tick/gate order to follow local
            tick/gate order. Disable only for metadata-shape audits that need
            to report ordering drift instead of rejecting it immediately.
        require_unique_host_id: Require each host id to appear on at most one
            host gate. Enable this for strict validation because repeated host
            records make first-later-host binding ambiguous across repeated
            helper invocations.
        context: Human-readable context included in failures.

    Returns:
        Tuple of recovered local-to-host bindings.

    Raises:
        ValueError: If a local operation is missing a host id, has no later
            compatible host, exceeds ``max_tick_separation``, or has a qubit or limit
            outside the supported unsigned integer range.
        TypeError: If the supplied object is not TickCircuit-like.
    """
    max_tick_separation = _hosted_tick_separation_limit(max_tick_separation, context=context)
    records = _hosted_gate_records(
        tick_circuit,
        host_id_key=host_id_key,
        local_role_key=local_role_key,
        context=context,
    )
    bindings = _validate_hosted_operations(
        [(r.tick_index, r.gate_index, r.gate_name, r.qubits, r.host_id, r.local_role) for r in records],
        max_tick_separation,
        require_shared_qubit,
        require_host_after_local,
        require_unique_host_id,
        host_id_key,
        context,
    )
    return tuple(
        HostedOperationBinding(
            host_id=records[local].host_id,
            local_role=records[local].local_role,
            local=records[local],
            host=records[host],
            tick_separation=separation,
        )
        for local, host, separation in bindings
    )


def _hosted_tick_separation_limit(value: int | None, *, context: str) -> int | None:
    if value is None:
        return None
    try:
        maximum = operator.index(value)
    except TypeError as exc:
        msg = f"{context}: max_tick_separation must be an integer, got {value!r}"
        raise ValueError(msg) from exc
    if maximum < 0:
        msg = f"{context}: max_tick_separation must be non-negative, got {value}"
        raise ValueError(msg)
    if maximum > 2**64 - 1:
        msg = f"{context}: max_tick_separation exceeds the unsigned 64-bit maximum (18446744073709551615)"
        raise ValueError(msg)
    return maximum


def _hosted_qubits(gate: object, *, location: str, context: str) -> tuple[int, ...]:
    qubits: list[int] = []
    for value in getattr(gate, "qubits", ()):
        try:
            qubit = operator.index(value)
        except TypeError as exc:
            msg = f"{context}: hosted gate {location} has non-integer qubit {value!r}"
            raise ValueError(msg) from exc
        if not 0 <= qubit <= 2 * sys.maxsize + 1:
            msg = (
                f"{context}: hosted gate {location} has a qubit outside the usize range "
                f"(0 through {2 * sys.maxsize + 1})"
            )
            raise ValueError(msg)
        qubits.append(qubit)
    return tuple(qubits)


def _hosted_gate_records(
    tick_circuit: object,
    *,
    host_id_key: str,
    local_role_key: str,
    context: str,
) -> tuple[HostedGateRecord, ...]:
    records: list[HostedGateRecord] = []
    for tick_index, gate_index, gate in _iter_tick_gates(tick_circuit, context=context):
        metadata = _gate_metadata(
            tick_circuit,
            tick_index,
            gate_index,
            keys=(host_id_key, local_role_key, *HOSTED_PROVENANCE_META_KEYS),
        )
        host_id = _metadata_text(metadata, host_id_key)
        local_role = _metadata_text(metadata, local_role_key)
        if not host_id and not local_role:
            continue
        gate_name = _gate_name(gate)
        records.append(
            HostedGateRecord(
                tick_index=tick_index,
                gate_index=gate_index,
                gate_name=gate_name,
                qubits=_hosted_qubits(
                    gate,
                    location=f"{gate_name}@t{tick_index}/g{gate_index}",
                    context=context,
                ),
                host_id=host_id,
                local_role=local_role,
                metadata=metadata,
            ),
        )
    return tuple(records)


def _iter_tick_gates(
    tick_circuit: object,
    *,
    context: str,
) -> tuple[tuple[int, int, object], ...]:
    try:
        num_ticks = int(tick_circuit.num_ticks())
    except AttributeError as exc:
        msg = f"{context}: expected a TickCircuit with num_ticks()."
        raise TypeError(msg) from exc

    gate_locations: list[tuple[int, int, object]] = []
    for tick_index in range(num_ticks):
        try:
            tick = tick_circuit.get_tick(tick_index)
        except AttributeError as exc:
            msg = f"{context}: expected a TickCircuit with get_tick()."
            raise TypeError(msg) from exc
        if tick is None:
            continue
        try:
            gate_batches = tick.gate_batches()
        except AttributeError as exc:
            msg = f"{context}: expected TickCircuit ticks with gate_batches()."
            raise TypeError(msg) from exc
        gate_locations.extend((tick_index, gate_index, gate) for gate_index, gate in enumerate(gate_batches))
    return tuple(gate_locations)


def _gate_metadata(
    tick_circuit: object,
    tick_index: int,
    gate_index: int,
    *,
    keys: Sequence[str],
) -> dict[str, object]:
    getter = getattr(tick_circuit, "get_gate_meta", None)
    if getter is None:
        return {}
    metadata: dict[str, object] = {}
    for key in keys:
        try:
            value = getter(tick_index, gate_index, key)
        except (AttributeError, IndexError, KeyError, TypeError):
            continue
        if value is not None:
            metadata[key] = value
    return metadata


def _metadata_text(metadata: Mapping[str, object], key: str) -> str:
    value = metadata.get(key)
    return "" if value is None else str(value)


def _gate_name(gate: object) -> str:
    gate_type = getattr(gate, "gate_type", None)
    name = getattr(gate_type, "name", None)
    if name is not None:
        return str(name)
    if gate_type is not None:
        return str(gate_type)
    return type(gate).__name__
