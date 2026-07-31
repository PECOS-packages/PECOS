# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Validate and replay framed QIS operation traces into PECOS circuits.

This module owns the generic trace-format and TickCircuit replay machinery.
It intentionally has no dependency on QEC or surface-code packages.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Mapping, Sequence
from typing import Any

from pecos_rslib import TimeUnits
from pecos_rslib.quantum import TickCircuit

from pecos.quantum import validate_hosted_operations


def _runtime_idle_seconds_to_time_units(duration_seconds: float) -> TimeUnits:
    """Convert runtime idle seconds into PECOS nanosecond time units."""
    if not math.isfinite(duration_seconds) or duration_seconds < 0.0:
        msg = f"Idle duration must be finite and non-negative, got {duration_seconds!r}"
        raise ValueError(msg)

    duration_nanoseconds = duration_seconds * 1_000_000_000.0
    if not math.isfinite(duration_nanoseconds):
        msg = f"Idle duration is too large to represent in nanoseconds, got {duration_seconds!r}"
        raise ValueError(msg)
    units = round(duration_nanoseconds)
    if duration_seconds > 0.0:
        units = max(1, units)
    return TimeUnits(units)


def _validate_measurement_crosstalk_topology(
    measurement_crosstalk_topology: str | None,
) -> str | None:
    if measurement_crosstalk_topology in (None, "none", "runtime_payloads"):
        return None
    if measurement_crosstalk_topology == "global_from_measurements":
        return measurement_crosstalk_topology
    msg = "measurement_crosstalk_topology must be None, 'runtime_payloads', or 'global_from_measurements'"
    raise ValueError(msg)


def _should_add_global_measurement_crosstalk_payload(
    measurement_crosstalk_topology: str | None,
) -> bool:
    return _validate_measurement_crosstalk_topology(measurement_crosstalk_topology) == "global_from_measurements"


def _replay_qis_trace_into_tick_circuit(
    operations: list[dict[str, Any]],
    *,
    measurement_crosstalk_topology: str | None = None,
) -> TickCircuit:
    """Replay traced QIS operations into a PECOS TickCircuit."""
    measurement_crosstalk_topology = _validate_measurement_crosstalk_topology(
        measurement_crosstalk_topology,
    )
    tick_circuit = TickCircuit()
    active_slots: dict[int, int] = {}
    free_slots: list[int] = []
    next_slot = 0

    def allocate_slot(program_id: int) -> int:
        nonlocal next_slot
        if program_id in active_slots:
            return active_slots[program_id]
        if free_slots:
            slot = heapq.heappop(free_slots)
        else:
            slot = next_slot
            next_slot += 1
        active_slots[program_id] = slot
        return slot

    def release_slot(program_id: int) -> None:
        slot = active_slots.pop(program_id, None)
        if slot is not None:
            heapq.heappush(free_slots, slot)

    def mapped_slot(program_id: int, op_name: str) -> int:
        if program_id not in active_slots:
            msg = f"Traced QIS op {op_name!r} referenced unmapped program qubit {program_id}"
            raise ValueError(msg)
        return active_slots[program_id]

    def scalar_arg(payload: object, op_name: str) -> int:
        if isinstance(payload, list):
            msg = f"Expected scalar payload for {op_name}, got {payload!r}"
            raise TypeError(msg)
        return int(payload)

    def tuple_args(payload: object, op_name: str, arity: int) -> tuple[Any, ...]:
        if not isinstance(payload, list) or len(payload) != arity:
            msg = f"Expected {arity} arguments for {op_name}, got {payload!r}"
            raise ValueError(msg)
        return tuple(payload)

    for operation in operations:
        if "AllocateQubit" in operation:
            program_id = int(operation["AllocateQubit"]["id"])
            slot = allocate_slot(program_id)
            tick_circuit.tick().pz([slot])
            continue

        if "ReleaseQubit" in operation:
            release_slot(int(operation["ReleaseQubit"]["id"]))
            continue

        if "AllocateResult" in operation or "RecordOutput" in operation or "Barrier" in operation:
            continue

        quantum = operation.get("Quantum")
        if quantum is None or len(quantum) != 1:
            msg = f"Unsupported traced operation payload: {operation!r}"
            raise ValueError(msg)

        op_name, payload = next(iter(quantum.items()))
        tick = tick_circuit.tick()

        if op_name == "H":
            tick.h([mapped_slot(scalar_arg(payload, op_name), op_name)])
        elif op_name == "X":
            tick.x([mapped_slot(scalar_arg(payload, op_name), op_name)])
        elif op_name == "Y":
            tick.y([mapped_slot(scalar_arg(payload, op_name), op_name)])
        elif op_name == "Z":
            tick.z([mapped_slot(scalar_arg(payload, op_name), op_name)])
        elif op_name == "S":
            tick.sz([mapped_slot(scalar_arg(payload, op_name), op_name)])
        elif op_name == "Sdg":
            tick.szdg([mapped_slot(scalar_arg(payload, op_name), op_name)])
        elif op_name == "T":
            tick.t([mapped_slot(scalar_arg(payload, op_name), op_name)])
        elif op_name == "Tdg":
            tick.tdg([mapped_slot(scalar_arg(payload, op_name), op_name)])
        elif op_name == "RX":
            theta, program_id = tuple_args(payload, op_name, 2)
            tick.rx(float(theta), [mapped_slot(int(program_id), op_name)])
        elif op_name == "RY":
            theta, program_id = tuple_args(payload, op_name, 2)
            tick.ry(float(theta), [mapped_slot(int(program_id), op_name)])
        elif op_name == "RZ":
            theta, program_id = tuple_args(payload, op_name, 2)
            tick.rz(float(theta), [mapped_slot(int(program_id), op_name)])
        elif op_name == "RXY":
            theta, phi, program_id = tuple_args(payload, op_name, 3)
            tick.r1xy(float(theta), float(phi), [mapped_slot(int(program_id), op_name)])
        elif op_name == "Idle":
            duration, program_id = tuple_args(payload, op_name, 2)
            tick.idle(
                _runtime_idle_seconds_to_time_units(float(duration)),
                [mapped_slot(int(program_id), op_name)],
            )
        elif op_name == "CX":
            control, target = tuple_args(payload, op_name, 2)
            tick.cx([(mapped_slot(int(control), op_name), mapped_slot(int(target), op_name))])
        elif op_name == "CY":
            control, target = tuple_args(payload, op_name, 2)
            tick.cy([(mapped_slot(int(control), op_name), mapped_slot(int(target), op_name))])
        elif op_name == "CZ":
            control, target = tuple_args(payload, op_name, 2)
            tick.cz([(mapped_slot(int(control), op_name), mapped_slot(int(target), op_name))])
        elif op_name == "CH":
            control, target = tuple_args(payload, op_name, 2)
            tick.ch([(mapped_slot(int(control), op_name), mapped_slot(int(target), op_name))])
        elif op_name == "CRZ":
            theta, control, target = tuple_args(payload, op_name, 3)
            tick.crz(
                float(theta),
                [(mapped_slot(int(control), op_name), mapped_slot(int(target), op_name))],
            )
        elif op_name == "CCX":
            control_a, control_b, target = tuple_args(payload, op_name, 3)
            tick.ccx(
                [
                    (
                        mapped_slot(int(control_a), op_name),
                        mapped_slot(int(control_b), op_name),
                        mapped_slot(int(target), op_name),
                    ),
                ],
            )
        elif op_name == "ZZ":
            qubit_a, qubit_b = tuple_args(payload, op_name, 2)
            tick.szz([(mapped_slot(int(qubit_a), op_name), mapped_slot(int(qubit_b), op_name))])
        elif op_name == "RZZ":
            theta, qubit_a, qubit_b = tuple_args(payload, op_name, 3)
            tick.rzz(
                float(theta),
                [(mapped_slot(int(qubit_a), op_name), mapped_slot(int(qubit_b), op_name))],
            )
        elif op_name == "Measure":
            program_id, result_id = tuple_args(payload, op_name, 2)
            measurement_qubit = mapped_slot(int(program_id), op_name)
            if _should_add_global_measurement_crosstalk_payload(
                measurement_crosstalk_topology,
            ):
                # Global crosstalk payload qubits are guaranteed not to be
                # affected; for measurement-induced global crosstalk this is
                # exactly the measured payload.
                tick_circuit.tick().add_gate(
                    "MeasCrosstalkGlobalPayload",
                    [measurement_qubit],
                )
            # Stamp the QIS-provided result_id as the MeasId rather than
            # discarding it and letting assign_missing_meas_ids() invent
            # sequential ids (which would be wrong for non-sequential ids).
            tick.mz_with_ids(
                [measurement_qubit],
                [int(result_id)],
            )
        elif op_name == "Reset":
            tick.pz([mapped_slot(scalar_arg(payload, op_name), op_name)])
        else:
            msg = f"Unsupported traced QIS quantum op {op_name!r}"
            raise ValueError(msg)

    # Compact: ASAP-schedule gates into minimal ticks
    tick_circuit.compact_ticks()

    return tick_circuit


def _gate_pairs(qubits: list[int], gate_type: str) -> list[tuple[int, int]]:
    """Convert a flattened qubit list into disjoint qubit pairs."""
    if len(qubits) % 2 != 0:
        msg = f"Lowered gate {gate_type!r} expected an even number of qubits, got {qubits!r}"
        raise ValueError(msg)
    return list(zip(qubits[::2], qubits[1::2], strict=True))


def _gate_triples(qubits: list[int], gate_type: str) -> list[tuple[int, int, int]]:
    """Convert a flattened qubit list into disjoint qubit triples."""
    if len(qubits) % 3 != 0:
        msg = f"Lowered gate {gate_type!r} expected qubits in triples, got {qubits!r}"
        raise ValueError(msg)
    return [(qubits[i], qubits[i + 1], qubits[i + 2]) for i in range(0, len(qubits), 3)]


def _require_gate_angles(angles: list[float], gate_type: str, arity: int) -> tuple[float, ...]:
    """Return a gate's angles after validating its trace-format arity."""
    if len(angles) != arity:
        msg = f"Lowered gate {gate_type!r} expected {arity} angle(s), got {angles!r}"
        raise ValueError(msg)
    return tuple(angles)


def _trace_gate_float_list(gate: Mapping[str, Any], field: str, gate_type: str) -> list[float]:
    """Return a finite floating-point list from a lowered trace gate."""
    values = gate.get(field, [])
    if not isinstance(values, list):
        msg = f"Lowered gate {gate_type!r} field {field!r} must be a list, got {values!r}"
        raise TypeError(msg)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        msg = f"Lowered gate {gate_type!r} field {field!r} must contain numbers, got {values!r}"
        raise TypeError(msg)
    try:
        converted = [float(value) for value in values]
    except OverflowError as exc:
        msg = f"Lowered gate {gate_type!r} field {field!r} must contain finite values, got {values!r}"
        raise ValueError(msg) from exc
    if not all(math.isfinite(value) for value in converted):
        msg = f"Lowered gate {gate_type!r} field {field!r} must contain finite values, got {values!r}"
        raise ValueError(msg)
    return converted


def _trace_gate_nonnegative_int_list(
    gate: Mapping[str, Any],
    field: str,
    gate_type: str,
) -> list[int]:
    """Return a non-negative integer list from a lowered trace gate."""
    values = gate.get(field, [])
    if not isinstance(values, list):
        msg = f"Lowered gate {gate_type!r} field {field!r} must be a list, got {values!r}"
        raise TypeError(msg)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        msg = f"Lowered gate {gate_type!r} field {field!r} must contain integers, got {values!r}"
        raise TypeError(msg)
    if any(value < 0 for value in values):
        msg = f"Lowered gate {gate_type!r} field {field!r} must contain non-negative values, got {values!r}"
        raise ValueError(msg)
    return values


def _lowered_gate_metadata(gate: Mapping[str, Any]) -> dict[str, Any]:
    """Return validated runtime/source metadata for a lowered trace gate."""
    metadata = gate.get("metadata")
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        msg = f"Lowered gate metadata must be an object, got {metadata!r}"
        raise TypeError(msg)
    return {str(key): value for key, value in metadata.items()}


def _set_lowered_gate_metadata(tick: object, metadata: Mapping[str, Any]) -> None:
    """Attach lowered trace metadata to the gate most recently added to ``tick``."""
    if not metadata:
        return
    tick.metas(metadata)


def _replay_lowered_qis_trace_into_tick_circuit(
    chunks: list[dict[str, Any]],
    *,
    measurement_crosstalk_topology: str | None = None,
) -> TickCircuit:
    """Replay lowered post-runtime ByteMessage gate batches into a TickCircuit.

    The lowered trace emits gates one at a time. We replay each into its own
    tick, then compact (ASAP schedule) so that gates on disjoint qubits share
    a tick --- matching the parallel structure of the abstract circuit.

    MeasIds flow from runtime-lowered measurement provenance:
    ``lowered_quantum_ops`` MZ entries must carry ``measurement_result_ids``.
    This avoids inferring lowered measurement IDs from raw QIS operation order,
    which is not stable under runtime scheduling or transport.
    """
    measurement_crosstalk_topology = _validate_measurement_crosstalk_topology(
        measurement_crosstalk_topology,
    )
    tick_circuit = TickCircuit()

    for chunk_index, chunk in enumerate(chunks):
        lowered_quantum_ops = chunk.get("lowered_quantum_ops")
        if not isinstance(lowered_quantum_ops, list):
            msg = f"Traced chunk {chunk_index} lowered_quantum_ops must be a list"
            raise TypeError(msg)
        for gate_index, gate in enumerate(lowered_quantum_ops):
            if not isinstance(gate, Mapping):
                msg = f"Lowered gate {gate_index} in traced chunk {chunk_index} must be an object"
                raise TypeError(msg)
            gate_type = gate.get("gate_type")
            if not isinstance(gate_type, str) or not gate_type:
                msg = f"Lowered gate {gate_index} in traced chunk {chunk_index} is missing a valid gate_type"
                raise TypeError(msg)
            qubits = _trace_gate_nonnegative_int_list(gate, "qubits", gate_type)
            angles = _trace_gate_float_list(gate, "angles", gate_type)
            params = _trace_gate_float_list(gate, "params", gate_type)
            metadata = _lowered_gate_metadata(gate)
            tick = tick_circuit.tick()

            if gate_type == "H":
                tick.h(qubits)
            elif gate_type == "X":
                tick.x(qubits)
            elif gate_type == "Y":
                tick.y(qubits)
            elif gate_type == "Z":
                tick.z(qubits)
            elif gate_type == "SZ":
                tick.sz(qubits)
            elif gate_type == "SZdg":
                tick.szdg(qubits)
            elif gate_type == "T":
                tick.t(qubits)
            elif gate_type == "Tdg":
                tick.tdg(qubits)
            elif gate_type == "PZ":
                tick.pz(qubits)
            elif gate_type == "Idle":
                if len(params) != 1:
                    msg = f"Lowered Idle gate expected one duration param, got {params!r}"
                    raise ValueError(msg)
                tick.idle(_runtime_idle_seconds_to_time_units(params[0]), qubits)
            elif gate_type == "MZ":
                if not isinstance(gate.get("measurement_result_ids"), list):
                    msg = (
                        "Lowered MZ trace is missing measurement_result_ids; "
                        "rebuild PECOS so runtime-lowered measurements carry "
                        "their result-id provenance instead of relying on "
                        "operation-order inference."
                    )
                    raise ValueError(msg)
                meas_ids = _trace_gate_nonnegative_int_list(
                    gate,
                    "measurement_result_ids",
                    gate_type,
                )
                if len(meas_ids) != len(qubits):
                    msg = f"Lowered MZ gate carries {len(meas_ids)} measurement_result_ids for {len(qubits)} qubit(s)"
                    raise ValueError(msg)
                if _should_add_global_measurement_crosstalk_payload(
                    measurement_crosstalk_topology,
                ):
                    # Global crosstalk payload qubits are guaranteed not to be
                    # affected; for measurement-induced global crosstalk this is
                    # exactly the measured payload.
                    tick_circuit.tick().add_gate(
                        "MeasCrosstalkGlobalPayload",
                        qubits,
                    )
                tick.mz_with_ids(qubits, [int(meas_id) for meas_id in meas_ids])
            elif gate_type == "MeasCrosstalkGlobalPayload":
                tick.add_gate("MeasCrosstalkGlobalPayload", qubits)
            elif gate_type == "MeasCrosstalkLocalPayload":
                tick.add_gate("MeasCrosstalkLocalPayload", qubits)
            elif gate_type == "RX":
                (theta,) = _require_gate_angles(angles, gate_type, 1)
                tick.rx(theta, qubits)
            elif gate_type == "RY":
                (theta,) = _require_gate_angles(angles, gate_type, 1)
                tick.ry(theta, qubits)
            elif gate_type == "RZ":
                (theta,) = _require_gate_angles(angles, gate_type, 1)
                tick.rz(theta, qubits)
            elif gate_type == "R1XY":
                theta, phi = _require_gate_angles(angles, gate_type, 2)
                tick.r1xy(theta, phi, qubits)
            elif gate_type == "CX":
                tick.cx(_gate_pairs(qubits, gate_type))
            elif gate_type == "CY":
                tick.cy(_gate_pairs(qubits, gate_type))
            elif gate_type == "CZ":
                tick.cz(_gate_pairs(qubits, gate_type))
            elif gate_type == "CH":
                tick.ch(_gate_pairs(qubits, gate_type))
            elif gate_type == "CRZ":
                (theta,) = _require_gate_angles(angles, gate_type, 1)
                tick.crz(theta, _gate_pairs(qubits, gate_type))
            elif gate_type == "SZZ":
                tick.szz(_gate_pairs(qubits, gate_type))
            elif gate_type == "SZZdg":
                tick.szzdg(_gate_pairs(qubits, gate_type))
            elif gate_type == "RZZ":
                (theta,) = _require_gate_angles(angles, gate_type, 1)
                tick.rzz(theta, _gate_pairs(qubits, gate_type))
            elif gate_type == "CCX":
                tick.ccx(_gate_triples(qubits, gate_type))
            else:
                msg = f"Unsupported lowered traced gate {gate_type!r}"
                raise ValueError(msg)
            _set_lowered_gate_metadata(tick, metadata)

    # Compact: ASAP-schedule gates into minimal ticks
    tick_circuit.compact_ticks()

    return tick_circuit


def _chunk_has_lowerable_op(chunk: dict[str, Any]) -> bool:
    """True if a chunk carries an operation that lowers to a TickCircuit gate.

    A raw ``Quantum`` op (gate / measure / reset) lowers to a gate, and an
    ``AllocateQubit`` lowers to a prep (``PZ``) -- both appear in
    ``lowered_quantum_ops`` after Selene lowering, and both are emitted as
    gates by the raw replay (see :func:`_replay_qis_trace_into_tick_circuit`).
    ``AllocateResult``, ``RecordOutput``, ``Barrier``, and ``ReleaseQubit``
    emit no gate and are pass-through bookkeeping, so a chunk containing only
    those legitimately has no lowered ops.
    """
    return any(
        isinstance(op, dict) and ("Quantum" in op or "AllocateQubit" in op) for op in (chunk.get("operations") or [])
    )


def _reject_partially_lowered_trace(chunks: list[dict[str, Any]]) -> None:
    """Fail loud on a mixed/partially-lowered trace.

    The lowered replay consumes a chunk's gates from ``lowered_quantum_ops``
    only (it reads ``operations`` solely for measurement result ids). So once
    *any* chunk is lowered, a chunk that carries a lowerable operation (a raw
    ``Quantum`` gate/measure/reset, or an ``AllocateQubit`` prep) but an empty
    ``lowered_quantum_ops`` would have those gates silently dropped -- the
    resulting TickCircuit would be missing operations with no error. A dropped
    *measurement* is already caught downstream by the meas-count guard in
    :func:`_replay_lowered_qis_trace_into_tick_circuit`, but a dropped prep or
    non-measurement gate (H, CX, ...) would pass silently. Reject the
    incomplete trace here instead of building from a partial gate stream.

    This is the explicit trace-format contract for live
    ``capture_operation_trace()`` output: lowered and raw forms must not be
    mixed across chunks. Per-chunk completeness of lowering is exercised by
    the trace replay regressions.
    """
    for idx, chunk in enumerate(chunks):
        if _chunk_has_lowerable_op(chunk) and chunk.get("lowered_quantum_ops_complete") is not True:
            msg = (
                f"Traced chunk {idx} does not attest a complete lowered gate stream. "
                "Audited trace replay requires lowered_quantum_ops_complete=true "
                "from the QIS trace producer; refusing to infer completeness from "
                "a non-empty lowered_quantum_ops list."
            )
            raise ValueError(msg)
        if _chunk_has_lowerable_op(chunk) and not chunk.get("lowered_quantum_ops"):
            msg = (
                f"Traced chunk {idx} carries lowerable operations (a quantum "
                "gate/measure/reset or an AllocateQubit prep) but no "
                "lowered_quantum_ops while other chunks are lowered. This "
                "mixed/partially-lowered trace would silently drop the chunk's "
                "gates in the lowered replay; refusing to build from an "
                "incomplete gate stream."
            )
            raise ValueError(msg)


def _replay_qis_trace_chunks_into_tick_circuit(
    chunks: list[dict[str, Any]],
    *,
    measurement_crosstalk_topology: str | None = None,
    allow_raw_measurement_id_fallback: bool = False,
) -> TickCircuit:
    """Replay captured QIS operation trace chunks into a ``TickCircuit``."""
    if not isinstance(chunks, list):
        msg = f"QIS operation trace must be a list of chunks, got {type(chunks).__name__}"
        raise TypeError(msg)
    if any(not isinstance(chunk, Mapping) for chunk in chunks):
        msg = "QIS operation trace chunks must be objects"
        raise TypeError(msg)
    measurement_crosstalk_topology = _validate_measurement_crosstalk_topology(
        measurement_crosstalk_topology,
    )
    has_lowered_operations = any(chunk.get("lowered_quantum_ops") for chunk in chunks)
    if (
        not allow_raw_measurement_id_fallback
        and not has_lowered_operations
        and any(_chunk_has_lowerable_op(chunk) for chunk in chunks)
    ):
        msg = (
            "runtime trace does not contain lowered_quantum_ops; refusing to "
            "replay an audited circuit from the raw pre-runtime QIS operation order"
        )
        raise ValueError(msg)
    if not allow_raw_measurement_id_fallback:
        _validate_audited_trace_stream(chunks)
    if has_lowered_operations:
        _reject_partially_lowered_trace(chunks)
        try:
            return _replay_lowered_qis_trace_into_tick_circuit(
                chunks,
                measurement_crosstalk_topology=measurement_crosstalk_topology,
            )
        except ValueError as exc:
            if "missing measurement_result_ids" not in str(exc):
                raise
            if not allow_raw_measurement_id_fallback:
                msg = (
                    "runtime-lowered trace is missing measurement_result_ids; "
                    "refusing to fall back to the raw QIS stream for an audited build"
                )
                raise ValueError(msg) from exc
            # Older local Selene/qis-compiler builds can emit lowered gates
            # without measurement_result_ids while still carrying the raw QIS
            # operations, whose Measure payloads include the stable result ids.
            # Replay the raw operations in that compatibility case instead of
            # losing provenance.

    operations: list[dict[str, Any]] = []
    for chunk in chunks:
        operations.extend(list(chunk.get("operations", [])))
    return _replay_qis_trace_into_tick_circuit(
        operations,
        measurement_crosstalk_topology=measurement_crosstalk_topology,
    )


def _validate_audited_trace_stream(chunks: list[dict[str, Any]]) -> None:
    """Validate framing and completeness across an audited QIS trace stream."""
    if not chunks:
        msg = "audited runtime trace is empty"
        raise ValueError(msg)
    engine_trace_id = chunks[0].get("engine_trace_id")
    shot_index = chunks[0].get("shot_index")
    if isinstance(engine_trace_id, bool) or not isinstance(engine_trace_id, int):
        msg = "audited runtime trace is missing a valid engine_trace_id"
        raise TypeError(msg)
    if engine_trace_id < 0:
        msg = "audited runtime trace engine_trace_id must be non-negative"
        raise ValueError(msg)
    if isinstance(shot_index, bool) or not isinstance(shot_index, int):
        msg = "audited runtime trace is missing a valid shot_index"
        raise TypeError(msg)
    if shot_index < 0:
        msg = "audited runtime trace shot_index must be non-negative"
        raise ValueError(msg)
    for expected_index, chunk in enumerate(chunks):
        if chunk.get("format") != "pecos_qis_operation_trace_v1":
            msg = f"audited runtime trace chunk {expected_index} has an unsupported format"
            raise ValueError(msg)
        if chunk.get("engine_trace_id") != engine_trace_id or chunk.get("shot_index") != shot_index:
            msg = "audited runtime trace mixes engine or shot identities"
            raise ValueError(msg)
        chunk_index = chunk.get("chunk_index")
        if isinstance(chunk_index, bool) or not isinstance(chunk_index, int):
            msg = f"audited runtime trace chunk {expected_index} has an invalid chunk_index"
            raise TypeError(msg)
        if chunk_index != expected_index:
            msg = (
                f"audited runtime trace chunk indices must be contiguous; expected {expected_index}, "
                f"got {chunk_index!r}"
            )
            raise ValueError(msg)
        operations = chunk.get("operations")
        if not isinstance(operations, list):
            msg = f"audited runtime trace chunk {expected_index} operations must be a list"
            raise TypeError(msg)
        num_operations = chunk.get("num_operations")
        if isinstance(num_operations, bool) or not isinstance(num_operations, int):
            msg = f"audited runtime trace chunk {expected_index} has an invalid num_operations"
            raise TypeError(msg)
        if num_operations < 0 or num_operations != len(operations):
            msg = f"audited runtime trace chunk {expected_index} has an invalid operation count"
            raise ValueError(msg)
    terminal_positions = [index for index, chunk in enumerate(chunks) if chunk.get("stage") == "trace_complete"]
    if terminal_positions != [len(chunks) - 1]:
        msg = (
            "audited runtime trace must contain exactly one terminal trace_complete "
            f"chunk, as its last chunk; found terminal markers at positions {terminal_positions}"
        )
        raise ValueError(msg)
    terminal = chunks[-1]
    if terminal.get("operations") or terminal.get("lowered_quantum_ops") or terminal.get("named_result_traces"):
        msg = (
            "audited runtime trace terminal chunk must be empty; operations or "
            "results after the terminal marker cannot be certified"
        )
        raise ValueError(msg)


def named_result_traces_from_operation_trace(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return runtime `result(...)` provenance records from operation trace chunks."""
    traces: list[dict[str, Any]] = []
    for chunk in chunks:
        traces.extend(trace for trace in (chunk.get("named_result_traces") or []) if isinstance(trace, dict))
    return traces


def source_measurement_ids_from_operation_trace(chunks: list[dict[str, Any]]) -> list[int]:
    """Return pre-runtime QIS measurement ids in source execution order."""
    ids: list[int] = []
    for chunk in chunks:
        for operation in chunk.get("operations") or []:
            if not isinstance(operation, Mapping):
                continue
            quantum = operation.get("Quantum")
            if not isinstance(quantum, Mapping):
                continue
            measure = quantum.get("Measure")
            if not isinstance(measure, Sequence) or isinstance(measure, (str, bytes)) or len(measure) != 2:
                continue
            result_id = measure[1]
            if isinstance(result_id, bool) or not isinstance(result_id, int):
                msg = f"raw QIS measurement result id must be an integer, got {result_id!r}"
                raise TypeError(msg)
            if result_id < 0:
                msg = f"raw QIS measurement result id must be non-negative, got {result_id!r}"
                raise ValueError(msg)
            ids.append(result_id)
    if len(ids) != len(set(ids)):
        msg = "raw QIS operation trace contains duplicate measurement result ids"
        raise ValueError(msg)
    return ids


def _validate_trace_hosted_operations_if_requested(
    tick_circuit: object,
    *,
    require_hosted_operation_order: bool,
    max_hosted_tick_separation: int | None,
    context: str,
) -> None:
    if not require_hosted_operation_order and max_hosted_tick_separation is None:
        return
    validate_hosted_operations(
        tick_circuit,
        max_tick_separation=max_hosted_tick_separation,
        require_host_after_local=require_hosted_operation_order,
        require_unique_host_id=True,
        context=context,
    )
