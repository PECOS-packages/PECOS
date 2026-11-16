# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Independent physical references for small, straight-line Guppy test programs."""

from pecos import trace_program_to_tick_circuit


def measurement_layout(program: object, num_qubits: int) -> tuple[dict[int, int], dict[int, int]]:
    """Return measurement IDs and record offsets keyed by physical qubit.

    These programs allocate all qubits up front and measure each exactly once.
    Read the executed quantum trace rather than the tag analyzer under test.
    Dict insertion order is the runtime measurement order.
    """
    emitted = trace_program_to_tick_circuit(program, num_qubits).measurement_emission()
    assert len(emitted) == num_qubits
    assert {qubit for qubit, _ in emitted} == set(range(num_qubits))
    assert all(mid is not None for _, mid in emitted)
    return dict(emitted), {qubit: index - len(emitted) for index, (qubit, _) in enumerate(emitted)}
