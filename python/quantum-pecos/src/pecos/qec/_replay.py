# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Noiseless Clifford replay shared by QEC builders and testing oracles."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos_rslib import SparseStab, is_supported_noop_or_metadata_gate

if TYPE_CHECKING:
    from pecos_rslib.quantum import TickCircuit


def _replay_tick_circuit(tc: TickCircuit, num_ticks: int, seed: int) -> tuple[SparseStab, list[int]]:
    """Replay supported Clifford operations, rejecting unknown operations and angles."""
    if not 0 <= num_ticks <= tc.num_ticks():
        msg = "Replay tick count is outside the circuit"
        raise ValueError(msg)
    max_q = max(
        (int(q) for i in range(tc.num_ticks()) for g in tc.get_tick(i).gate_batches() for q in g.qubits),
        default=0,
    )
    sim = SparseStab(max_q + 1)
    sim.set_seed(seed)
    flat = []
    for i in range(num_ticks):
        for gate in tc.get_tick(i).gate_batches():
            name = gate.gate_type.name
            qubits = [int(q) for q in gate.qubits]
            if gate.angles:
                msg = f"Unsupported replay operation: {name} with angles"
                raise NotImplementedError(msg)
            if is_supported_noop_or_metadata_gate(gate.gate_type):
                continue
            if name in {"QAlloc", "PZ"}:
                sim.run_gate("PZ", set(qubits))
            elif name in {"MZ", "MeasureFree"}:
                flat.extend(sim.run_gate("MZ", {q}).get(q, 0) for q in qubits)
            elif gate.is_two_qubit():
                # Gate arity accessor: python/pecos-rslib/src/dag_circuit_bindings.rs.
                sim.run_gate(name, set(zip(qubits[::2], qubits[1::2], strict=True)))
            elif gate.is_single_qubit():
                # SparseStab.run_gate raises for unsupported symbols.
                sim.run_gate(name, set(qubits))
            else:
                msg = f"Unsupported replay operation: {name}"
                raise NotImplementedError(msg)
    return sim, flat


def _replay_measurements(tc: TickCircuit, expected_measurements: int, seed: int = 0) -> list[int]:
    """Replay the complete circuit and validate its measurement count before record indexing."""
    _, flat = _replay_tick_circuit(tc, tc.num_ticks(), seed)
    if len(flat) != expected_measurements:
        msg = "Replay measurement count disagrees with circuit metadata"
        raise ValueError(msg)
    return flat
