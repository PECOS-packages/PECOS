# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Capture runtime QIS traces and replay them as PECOS circuits.

These helpers trace programs that PECOS can lower to its QIS execution path.
They execute one ideal shot through a Selene-compatible runtime and expose
either the structured operation trace or the corresponding runtime-lowered
:class:`~pecos.quantum.TickCircuit`.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

import pecos_rslib
from pecos_rslib import (
    _qis_named_result_traces,
    _qis_source_measurement_ids,
    _replay_qis_trace,
)

from pecos.quantum import validate_hosted_operations

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pecos.quantum import TickCircuit


def _replay_qis_trace_chunks_into_tick_circuit(
    chunks: list[dict[str, Any]],
    *,
    measurement_crosstalk_topology: str | None = None,
) -> TickCircuit:
    """Replay audited stored chunks using the shared Rust implementation."""
    return _replay_qis_trace(
        chunks,
        measurement_crosstalk_topology,
    )


def named_result_traces_from_operation_trace(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return named output provenance in emission order."""
    return _qis_named_result_traces(chunks)


def source_measurement_ids_from_operation_trace(chunks: list[dict[str, Any]]) -> list[int]:
    """Return unique source measurement IDs in execution order."""
    return _qis_source_measurement_ids(chunks)


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


_RESULT_TRACE_COLLECTOR: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "pecos_result_trace_collector",
    default=None,
)


def _capture_qis_operation_traces(
    program: object,
    num_qubits: int,
    *,
    shots: int,
    seed: int = 0,
    runtime: object | None = None,
    batch_adapter_factory: object | None = None,
    clock_model: str | None = None,
) -> list[dict[str, Any]]:
    """Capture one or more QIS trace shots for internal certification."""
    if shots <= 0:
        msg = "trace shots must be greater than zero"
        raise ValueError(msg)
    import pecos  # noqa: PLC0415

    sim_builder = (
        pecos.sim(program)
        .classical(pecos.selene_engine(runtime, batch_adapter_factory=batch_adapter_factory, clock_model=clock_model))
        .quantum(pecos_rslib.coin_toss())
        .qubits(num_qubits)
        .seed(seed)
    )
    return list(sim_builder.capture_operation_trace(shots))


def capture_qis_operation_trace(
    program: object,
    num_qubits: int,
    *,
    seed: int = 0,
    runtime: object | None = None,
    batch_adapter_factory: object | None = None,
    clock_model: str | None = None,
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
        clock_model: Optional ``"continuous"`` or ``"shot_relative"`` native
            clock convention; defaults to the plugin declaration or continuous.
            Independent of custom-event handling and native instance reuse.
        batch_adapter_factory: Optional zero-argument factory returning a fresh
            event adapter with ``normalize(batch)``. Uses the same normalization
            as simulation; defaults to the runtime plugin's factory when present.

    Returns:
        The structured operation-trace chunks for one completed shot.
    """
    return _capture_qis_operation_traces(
        program,
        num_qubits,
        shots=1,
        seed=seed,
        runtime=runtime,
        batch_adapter_factory=batch_adapter_factory,
        clock_model=clock_model,
    )


def _qis_operation_trace_to_tick_circuit(
    chunks: list[dict[str, Any]],
    *,
    measurement_crosstalk_topology: str | None = None,
    require_hosted_operation_order: bool = False,
    max_hosted_tick_separation: int | None = None,
    context: str,
) -> TickCircuit:
    """Replay captured QIS operation-trace chunks into a ``TickCircuit``."""
    tick_circuit = _replay_qis_trace_chunks_into_tick_circuit(
        chunks,
        measurement_crosstalk_topology=measurement_crosstalk_topology,
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
        TypeError: If the input cannot be serialized as JSON.
        ValueError: If fields have invalid types, the trace is incomplete,
            mixes shots, lacks audited runtime-lowered operations, or cannot
            be replayed.
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
    batch_adapter_factory: object | None = None,
    clock_model: str | None = None,
    measurement_crosstalk_topology: str | None = None,
    require_hosted_operation_order: bool = False,
    max_hosted_tick_separation: int | None = None,
) -> tuple[TickCircuit, list[dict[str, Any]]]:
    """Trace a program into a ``TickCircuit`` and result provenance.

    This is the provenance-preserving variant of
    :func:`trace_program_to_tick_circuit`. The second return value contains
    runtime ``result(...)`` records captured during the traced shot.
    """
    chunks = capture_qis_operation_trace(
        program,
        num_qubits,
        seed=seed,
        runtime=runtime,
        batch_adapter_factory=batch_adapter_factory,
        clock_model=clock_model,
    )
    tick_circuit = _qis_operation_trace_to_tick_circuit(
        chunks,
        measurement_crosstalk_topology=measurement_crosstalk_topology,
        require_hosted_operation_order=require_hosted_operation_order,
        max_hosted_tick_separation=max_hosted_tick_separation,
        context="_trace_program_to_tick_circuit_with_result_traces",
    )
    return tick_circuit, named_result_traces_from_operation_trace(chunks)


@contextmanager
def _collect_program_result_traces() -> Iterator[list[dict[str, Any]]]:
    """Collect result provenance when tracing through the stable public helper."""
    result_traces: list[dict[str, Any]] = []
    token = _RESULT_TRACE_COLLECTOR.set(result_traces)
    try:
        yield result_traces
    finally:
        _RESULT_TRACE_COLLECTOR.reset(token)


def trace_program_to_tick_circuit(
    program: object,
    num_qubits: int,
    *,
    seed: int = 0,
    runtime: object | None = None,
    batch_adapter_factory: object | None = None,
    clock_model: str | None = None,
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
        clock_model: Optional ``"continuous"`` or ``"shot_relative"`` native
            clock convention; defaults to the plugin declaration or continuous.
            Independent of custom-event handling and native instance reuse.
        batch_adapter_factory: Optional zero-argument factory returning a fresh
            event adapter with ``normalize(batch)``. Uses the same normalization
            as simulation; defaults to the runtime plugin's factory when present.
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
    tick_circuit, result_traces = _trace_program_to_tick_circuit_with_result_traces(
        program,
        num_qubits,
        seed=seed,
        runtime=runtime,
        batch_adapter_factory=batch_adapter_factory,
        clock_model=clock_model,
        measurement_crosstalk_topology=measurement_crosstalk_topology,
        require_hosted_operation_order=require_hosted_operation_order,
        max_hosted_tick_separation=max_hosted_tick_separation,
    )
    collector = _RESULT_TRACE_COLLECTOR.get()
    if collector is not None:
        collector.extend(result_traces)
    return tick_circuit


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
