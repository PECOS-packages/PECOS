# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Capture runtime QIS traces and replay them as PECOS circuits.

These helpers trace programs that PECOS can lower to its QIS execution path.
They execute one ideal shot through a Selene-compatible runtime and expose
either the structured operation trace or the corresponding runtime-lowered
:class:`~pecos.quantum.TickCircuit`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pecos.quantum import TickCircuit


def capture_qis_operation_trace(
    program: object,
    num_qubits: int,
    *,
    seed: int = 0,
    runtime: object | None = None,
) -> list[dict[str, Any]]:
    """Capture structured QIS operation-trace chunks from a program.

    The program is executed for one ideal shot with operation tracing enabled.
    Each returned dictionary is a framed ``pecos_qis_operation_trace_v1``
    chunk containing the source QIS operations, runtime-lowered quantum
    operations, and any named-result provenance emitted during that shot.

    Args:
        program: A Guppy, HUGR, or QIS program accepted by :func:`pecos.sim`,
            including a ``@guppy`` function or the :class:`pecos.Guppy`,
            :class:`pecos.Hugr`, and :class:`pecos.Qis` wrappers.
        num_qubits: Number of qubits to allocate.
        seed: Seed for the ideal trace execution.
        runtime: Optional Selene runtime selector or plugin. ``None`` selects
            the default runtime.

    Returns:
        The structured operation-trace chunks for one completed shot.
    """
    import pecos_rslib  # noqa: PLC0415

    import pecos  # noqa: PLC0415

    # Trace capture records runtime-lowered operations and provenance. Use a
    # permissive backend because no quantum-state evolution is needed here.
    sim_builder = (
        pecos.sim(program)
        .classical(pecos.selene_engine(runtime))
        .quantum(pecos_rslib.coin_toss())
        .qubits(num_qubits)
        .seed(seed)
    )
    return list(sim_builder.capture_operation_trace())


def _qis_operation_trace_to_tick_circuit(
    chunks: list[dict[str, Any]],
    *,
    measurement_crosstalk_topology: str | None = None,
    require_hosted_operation_order: bool = False,
    max_hosted_tick_separation: int | None = None,
    allow_raw_measurement_id_fallback: bool = False,
    context: str,
) -> TickCircuit:
    """Replay captured QIS operation-trace chunks into a ``TickCircuit``."""
    from pecos.qec.surface.decode import (  # noqa: PLC0415
        _replay_qis_trace_chunks_into_tick_circuit,
        _validate_trace_hosted_operations_if_requested,
        source_measurement_ids_from_operation_trace,
    )

    tick_circuit = _replay_qis_trace_chunks_into_tick_circuit(
        chunks,
        measurement_crosstalk_topology=measurement_crosstalk_topology,
        allow_raw_measurement_id_fallback=allow_raw_measurement_id_fallback,
    )
    tick_circuit.set_meta(
        "qis_source_measurement_ids",
        json.dumps(source_measurement_ids_from_operation_trace(chunks), separators=(",", ":")),
    )
    _validate_trace_hosted_operations_if_requested(
        tick_circuit,
        require_hosted_operation_order=require_hosted_operation_order,
        max_hosted_tick_separation=max_hosted_tick_separation,
        context=context,
    )
    return tick_circuit


def qis_operation_trace_to_tick_circuit(
    trace: list[dict[str, Any]],
    *,
    measurement_crosstalk_topology: str | None = None,
    require_hosted_operation_order: bool = False,
    max_hosted_tick_separation: int | None = None,
) -> TickCircuit:
    """Replay a completed QIS operation trace into a ``TickCircuit``.

    Args:
        trace: Framed ``pecos_qis_operation_trace_v1`` chunks, such as those
            returned by :func:`capture_qis_operation_trace`.
        measurement_crosstalk_topology: Optional measurement-crosstalk replay
            mode for global measurement-crosstalk payload markers.
        require_hosted_operation_order: Validate hosted-operation ordering
            metadata after replay.
        max_hosted_tick_separation: Optional maximum tick separation accepted
            by hosted-operation validation.

    Returns:
        A runtime-lowered ``TickCircuit``. Detector and observable metadata are
        not attached automatically. Runtime-emitted ``Idle`` durations are
        preserved as integer nanosecond :class:`pecos.TimeUnits`.

    Raises:
        TypeError: If trace framing, operation counts, or measurement metadata
            have invalid types.
        ValueError: If the trace is incomplete, mixes shots, lacks audited
            runtime-lowered operations, or cannot be replayed.
    """
    return _qis_operation_trace_to_tick_circuit(
        trace,
        measurement_crosstalk_topology=measurement_crosstalk_topology,
        require_hosted_operation_order=require_hosted_operation_order,
        max_hosted_tick_separation=max_hosted_tick_separation,
        context="qis_operation_trace_to_tick_circuit",
    )


def _trace_program_to_tick_circuit_with_result_traces(
    program: object,
    num_qubits: int,
    *,
    seed: int = 0,
    runtime: object | None = None,
    measurement_crosstalk_topology: str | None = None,
    require_hosted_operation_order: bool = False,
    max_hosted_tick_separation: int | None = None,
    allow_raw_measurement_id_fallback: bool = False,
) -> tuple[TickCircuit, list[dict[str, Any]]]:
    """Trace a program into a ``TickCircuit`` and result provenance.

    This is the provenance-preserving variant of
    :func:`trace_program_to_tick_circuit`. The second return value contains
    runtime ``result(...)`` records captured during the traced shot.
    """
    from pecos.qec.surface.decode import named_result_traces_from_operation_trace  # noqa: PLC0415

    chunks = capture_qis_operation_trace(program, num_qubits, seed=seed, runtime=runtime)
    tick_circuit = _qis_operation_trace_to_tick_circuit(
        chunks,
        measurement_crosstalk_topology=measurement_crosstalk_topology,
        require_hosted_operation_order=require_hosted_operation_order,
        max_hosted_tick_separation=max_hosted_tick_separation,
        allow_raw_measurement_id_fallback=allow_raw_measurement_id_fallback,
        context="_trace_program_to_tick_circuit_with_result_traces",
    )
    return tick_circuit, named_result_traces_from_operation_trace(chunks)


def trace_program_to_tick_circuit(
    program: object,
    num_qubits: int,
    *,
    seed: int = 0,
    runtime: object | None = None,
    measurement_crosstalk_topology: str | None = None,
    require_hosted_operation_order: bool = False,
    max_hosted_tick_separation: int | None = None,
) -> TickCircuit:
    """Trace a program's lowered runtime operations into a circuit.

    The program is run once through the selected Selene-compatible runtime.
    PECOS validates the framed, completed operation stream and replays its
    runtime-lowered gate batches into a :class:`pecos.quantum.TickCircuit`.

    Args:
        program: A Guppy, HUGR, or QIS program accepted by :func:`pecos.sim`,
            including a ``@guppy`` function or the :class:`pecos.Guppy`,
            :class:`pecos.Hugr`, and :class:`pecos.Qis` wrappers.
        num_qubits: Number of qubits to allocate.
        seed: Seed for the ideal trace execution.
        runtime: Optional Selene runtime selector or plugin. ``None`` selects
            the default runtime.
        measurement_crosstalk_topology: Optional measurement-crosstalk replay
            mode for global measurement-crosstalk payload markers.
        require_hosted_operation_order: Validate hosted-operation ordering
            metadata after replay.
        max_hosted_tick_separation: Optional maximum tick separation accepted
            by hosted-operation validation.

    Returns:
        A runtime-lowered ``TickCircuit``. Detector and observable metadata are
        not attached automatically. Runtime-emitted ``Idle`` durations are
        preserved as integer nanosecond :class:`pecos.TimeUnits`.

    Note:
        This represents one execution path. For static circuit analysis, do
        not use a trace from measurement-dependent branches or loops as though
        it represented all possible executions.
    """
    trace = capture_qis_operation_trace(program, num_qubits, seed=seed, runtime=runtime)
    return _qis_operation_trace_to_tick_circuit(
        trace,
        measurement_crosstalk_topology=measurement_crosstalk_topology,
        require_hosted_operation_order=require_hosted_operation_order,
        max_hosted_tick_separation=max_hosted_tick_separation,
        context="trace_program_to_tick_circuit",
    )


# Compatibility aliases for the original surface-code-internal names. These
# remain importable from ``pecos.qec.surface.decode`` but are intentionally not
# part of the new top-level public API.
capture_guppy_operation_trace = capture_qis_operation_trace
trace_guppy_into_tick_circuit = trace_program_to_tick_circuit
trace_guppy_into_tick_circuit_with_result_traces = _trace_program_to_tick_circuit_with_result_traces


__all__ = [
    "capture_qis_operation_trace",
    "qis_operation_trace_to_tick_circuit",
    "trace_program_to_tick_circuit",
]
