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
    ``RZZ(pi/2)``. Lower these rotations, optionally simplify Clifford chains,
    and assign stable measurement ids before converting the trace to a DAG.
    """
    _call_required_tick_circuit_method(tick_circuit, "lower_clifford_rotations", context)
    if simplify_single_qubit_clifford_chains:
        _call_required_tick_circuit_method(
            tick_circuit,
            "simplify_single_qubit_clifford_chains",
            context,
        )
    _call_required_tick_circuit_method(tick_circuit, "assign_missing_meas_ids", context)


def _call_required_tick_circuit_method(tick_circuit: object, method_name: str, context: str) -> None:
    method = getattr(tick_circuit, method_name, None)
    if not callable(method):
        msg = f"{context}: expected a TickCircuit with callable {method_name}()."
        raise TypeError(msg)
    method()


def _gate_type_name(gate: object) -> str:
    gate_type = getattr(gate, "gate_type", "")
    return str(getattr(gate_type, "name", str(gate_type).rsplit(".", maxsplit=1)[-1]))
