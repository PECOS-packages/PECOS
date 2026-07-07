"""Hosted-operation metadata utilities.

A hosted operation is a source-local operation whose semantic role is tied to a
later host operation.  For example, an SZZ surface-code data-prefix pulse can
declare ``host_id=<source SZZ id>`` and ``local_role=basis_prefix`` while the
lowered SZZ/SZZdg host carries the same ``host_id``.  Runtimes are free to
lower and schedule the gates, but traced circuits can then fail loudly if the
source-host relationship was lost or separated too far.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
            between the local and selected host operation.
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
            compatible host, or exceeds ``max_tick_separation``.
        TypeError: If the supplied object is not TickCircuit-like.
    """
    if max_tick_separation is not None and max_tick_separation < 0:
        msg = f"{context}: max_tick_separation must be non-negative, got {max_tick_separation}"
        raise ValueError(msg)

    records = _hosted_gate_records(
        tick_circuit,
        host_id_key=host_id_key,
        local_role_key=local_role_key,
        context=context,
    )
    if require_unique_host_id:
        _raise_if_repeated_host_records(
            records,
            host_id_key=host_id_key,
            context=context,
        )
    bindings: list[HostedOperationBinding] = []
    for local in records:
        if not local.local_role:
            continue
        if not local.host_id:
            msg = (
                f"{context}: hosted local gate {local.gate_name}@t{local.tick_index}/"
                f"g{local.gate_index} on qubits {local.qubits} has local_role "
                f"{local.local_role!r} but no non-empty {host_id_key!r} metadata."
            )
            raise ValueError(msg)
        host_candidates = _matching_host_records(
            records,
            local,
            require_shared_qubit=require_shared_qubit,
        )
        if not host_candidates:
            shared_clause = " sharing a qubit" if require_shared_qubit else ""
            msg = (
                f"{context}: hosted local gate {local.gate_name}@t{local.tick_index}/"
                f"g{local.gate_index} on qubits {local.qubits} with host_id "
                f"{local.host_id!r} and local_role {local.local_role!r} has no "
                f"host gate{shared_clause} carrying the same host_id."
            )
            raise ValueError(msg)
        later_hosts = [candidate for candidate in host_candidates if _gate_order(candidate) > _gate_order(local)]
        if require_host_after_local and not later_hosts:
            nearest_host = host_candidates[-1]
            msg = (
                f"{context}: hosted local gate {local.gate_name}@t{local.tick_index}/"
                f"g{local.gate_index} on qubits {local.qubits} with host_id "
                f"{local.host_id!r} and local_role {local.local_role!r} has matching "
                f"host metadata only before it; nearest host is {nearest_host.gate_name}"
                f"@t{nearest_host.tick_index}/g{nearest_host.gate_index}. This "
                "indicates source-host ordering drift in the traced runtime schedule."
            )
            raise ValueError(msg)
        host = later_hosts[0] if later_hosts else host_candidates[-1]
        tick_separation = host.tick_index - local.tick_index
        if max_tick_separation is not None and abs(tick_separation) > max_tick_separation:
            msg = (
                f"{context}: hosted local gate {local.gate_name}@t{local.tick_index}/"
                f"g{local.gate_index} on qubits {local.qubits} with host_id "
                f"{local.host_id!r} binds to {host.gate_name}@t{host.tick_index}/"
                f"g{host.gate_index} with signed tick separation {tick_separation}, "
                f"exceeding max_tick_separation={max_tick_separation}."
            )
            raise ValueError(msg)
        bindings.append(
            HostedOperationBinding(
                host_id=local.host_id,
                local_role=local.local_role,
                local=local,
                host=host,
                tick_separation=tick_separation,
            ),
        )
    return tuple(bindings)


def _matching_host_records(
    records: Sequence[HostedGateRecord],
    local: HostedGateRecord,
    *,
    require_shared_qubit: bool,
) -> tuple[HostedGateRecord, ...]:
    local_qubits = set(local.qubits)
    candidates: list[HostedGateRecord] = []
    for candidate in records:
        if candidate.host_id != local.host_id or candidate.local_role:
            continue
        if require_shared_qubit and local_qubits.isdisjoint(candidate.qubits):
            continue
        candidates.append(candidate)
    return tuple(candidates)


def _raise_if_repeated_host_records(
    records: Sequence[HostedGateRecord],
    *,
    host_id_key: str,
    context: str,
) -> None:
    host_records_by_id: dict[str, list[HostedGateRecord]] = {}
    for record in records:
        if record.local_role or not record.host_id:
            continue
        host_records_by_id.setdefault(record.host_id, []).append(record)
    repeated = {host_id: host_records for host_id, host_records in host_records_by_id.items() if len(host_records) > 1}
    if not repeated:
        return
    host_id, host_records = next(iter(repeated.items()))
    first_locations = ", ".join(
        f"{record.gate_name}@t{record.tick_index}/g{record.gate_index}" for record in host_records[:4]
    )
    extra_count = len(host_records) - 4
    if extra_count > 0:
        first_locations = f"{first_locations}, ... (+{extra_count} more)"
    msg = (
        f"{context}: hosted metadata key {host_id_key!r} is ambiguous because "
        f"host_id {host_id!r} appears on {len(host_records)} host gates "
        f"({first_locations}). Strict hosted-operation validation requires "
        "host ids to identify one source host gate; include invocation-scoped "
        "metadata before validating ordering or tick separation."
    )
    raise ValueError(msg)


def _gate_order(record: HostedGateRecord) -> tuple[int, int]:
    return (record.tick_index, record.gate_index)


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
        records.append(
            HostedGateRecord(
                tick_index=tick_index,
                gate_index=gate_index,
                gate_name=_gate_name(gate),
                qubits=tuple(int(qubit) for qubit in getattr(gate, "qubits", ())),
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
