# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Generic utilities for runtime-traced PECOS TickCircuits.

These helpers intentionally have no dependency on QEC or surface-code
packages. They are internal building blocks for tracing, fault analysis, and
protocol-specific metadata binding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos_rslib.quantum import TickCircuit


def measurement_ids_in_execution_order(tick_circuit: TickCircuit) -> list[int]:
    """Return stable measurement ids in TickCircuit emission order."""
    measurement_ids: list[int] = []
    for position, (_, measurement_id) in enumerate(tick_circuit.measurement_emission()):
        if measurement_id is None:
            msg = f"traced measurement at emission position {position} carries no MeasId"
            raise ValueError(msg)
        measurement_ids.append(measurement_id)
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
