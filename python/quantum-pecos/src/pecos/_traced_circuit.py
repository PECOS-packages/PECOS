# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Generic utilities for runtime-traced PECOS TickCircuits.

These helpers intentionally have no dependency on QEC or surface-code
packages. They are internal building blocks for tracing, fault analysis, and
protocol-specific metadata binding.
"""

from __future__ import annotations


def measurement_ids_in_execution_order(tick_circuit: object) -> list[int]:
    """Return stable measurement ids in TickCircuit execution order."""
    measurement_ids: list[int] = []
    for tick_index in range(tick_circuit.num_ticks()):  # type: ignore[attr-defined]
        tick = tick_circuit.get_tick(tick_index)  # type: ignore[attr-defined]
        if tick is None:
            continue
        for gate in tick.gate_batches():
            gate_type = _gate_type_name(gate)
            if gate_type not in {"MZ", "MeasureFree"}:
                continue
            qubits = list(getattr(gate, "qubits", []))
            gate_measurement_ids = list(getattr(gate, "meas_ids", []))
            if len(gate_measurement_ids) != len(qubits):
                msg = (
                    f"traced measurement gate {gate_type} in tick {tick_index} carries "
                    f"{len(gate_measurement_ids)} MeasId(s) for {len(qubits)} qubit(s)"
                )
                raise ValueError(msg)
            measurement_ids.extend(int(measurement_id) for measurement_id in gate_measurement_ids)
    return measurement_ids


def normalize_traced_tick_circuit(
    tick_circuit: object,
    *,
    context: str = "traced-circuit fault analysis",
    simplify_single_qubit_clifford_chains: bool = True,
) -> None:
    """Normalize a runtime-traced TickCircuit before fault analysis.

    Runtime traces may contain parameterized Clifford rotations such as
    ``RZZ(pi/2)``. Fault analysis and replacement-branch noise models operate
    on named Clifford gates (``SZZ`` / ``SZZdg``), so callers should normalize
    traced circuits before converting them to a DAG.
    """
    _call_required_tick_circuit_method(tick_circuit, "lower_clifford_rotations", context)
    if simplify_single_qubit_clifford_chains:
        _call_required_tick_circuit_method(
            tick_circuit,
            "simplify_single_qubit_clifford_chains",
            context,
        )
    _call_required_tick_circuit_method(tick_circuit, "assign_missing_meas_ids", context)
    assert_tick_circuit_ready_for_fault_analysis(tick_circuit, context=context)


def assert_tick_circuit_ready_for_fault_analysis(
    tick_circuit: object,
    *,
    context: str = "traced-circuit fault analysis",
) -> None:
    """Fail loudly if raw runtime rotations survived normalization."""
    offenders = _raw_rzz_gates(tick_circuit, context=context)
    if not offenders:
        return

    preview = "; ".join(offenders[:5])
    suffix = f"; ... {len(offenders) - 5} more" if len(offenders) > 5 else ""
    msg = (
        f"{context}: traced circuit still contains raw RZZ gates after Clifford "
        "normalization. DEM/DAG analysis expects Clifford RZZ(pi/2) and "
        "RZZ(-pi/2) gates to be lowered to SZZ/SZZdg before noise attachment "
        "and fault propagation. Call normalize_traced_tick_circuit(...) "
        "before to_dag_circuit(), or extend lower_clifford_rotations() for the "
        f"runtime-emitted angle. First offending gates: {preview}{suffix}"
    )
    raise ValueError(msg)


def _call_required_tick_circuit_method(tick_circuit: object, method_name: str, context: str) -> None:
    method = getattr(tick_circuit, method_name, None)
    if not callable(method):
        msg = f"{context}: expected a TickCircuit with callable {method_name}()."
        raise TypeError(msg)
    method()


def _raw_rzz_gates(tick_circuit: object, *, context: str) -> list[str]:
    try:
        num_ticks = int(tick_circuit.num_ticks())  # type: ignore[attr-defined]
    except AttributeError as exc:
        msg = f"{context}: expected a TickCircuit with num_ticks() before DEM/DAG analysis."
        raise TypeError(msg) from exc

    offenders: list[str] = []
    for tick_index in range(num_ticks):
        try:
            tick = tick_circuit.get_tick(tick_index)  # type: ignore[attr-defined]
        except AttributeError as exc:
            msg = f"{context}: expected a TickCircuit with get_tick() before DEM/DAG analysis."
            raise TypeError(msg) from exc
        try:
            gate_batches = tick.gate_batches()
        except AttributeError as exc:
            msg = f"{context}: expected TickCircuit ticks with gate_batches() before DEM/DAG analysis."
            raise TypeError(msg) from exc
        for gate_index, gate in enumerate(gate_batches):
            if _gate_type_name(gate) != "RZZ":
                continue
            qubits = [int(q) for q in getattr(gate, "qubits", [])]
            offenders.append(
                f"tick={tick_index} gate={gate_index} qubits={qubits} angles={_gate_angles_for_message(gate)}",
            )
    return offenders


def _gate_type_name(gate: object) -> str:
    gate_type = getattr(gate, "gate_type", "")
    return str(getattr(gate_type, "name", str(gate_type).rsplit(".", maxsplit=1)[-1]))


def _format_gate_angle(angle: object) -> str:
    try:
        return repr(float(angle))
    except (TypeError, ValueError):
        return repr(angle)


def _gate_angles_for_message(gate: object) -> list[str]:
    angles = getattr(gate, "angles", None)
    if angles is None:
        angles = getattr(gate, "params", [])
    return [_format_gate_angle(angle) for angle in angles]
