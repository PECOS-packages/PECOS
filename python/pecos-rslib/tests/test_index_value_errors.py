# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0.

"""Width and measurement indices reject bools and negative values consistently."""

import sys

import pytest
import pecos_rslib as rust


@pytest.mark.parametrize(("value", "error"), [(-1, ValueError), (True, TypeError)])
@pytest.mark.parametrize(
    "consumer",
    [
        lambda n: rust.X(0).to_dense_str(n),
        lambda n: rust.CliffordRep.identity(n),
        lambda n: rust.CliffordRep.random(n),
        lambda n: rust.quantum_info.Ptm.identity(n),
        lambda n: rust.quantum_info.ProcessTomographyDesign.matrix_unit(n),
        lambda n: rust.quantum_info.matrix_unit_basis(n),
        lambda n: rust.X(0).to_matrix(n),
        lambda n: rust.SymplecticMatrix.zeros(n, 1),
        lambda n: rust.SymplecticMatrix.zeros(1, n),
        lambda n: rust.ParityCheckMatrix.zeros(n, 1),
        lambda n: rust.DagCircuit().measured_pauli(n),
        lambda n: rust.DagCircuit().measured_paulis([n]),
        lambda n: rust.TickCircuit().measured_pauli(n),
        lambda n: rust.TickCircuit().measured_paulis([n]),
    ],
)
def test_width_and_id_types(consumer, value, error):
    with pytest.raises(error):
        consumer(value)


@pytest.mark.parametrize("width", [1, 13])
def test_dense_matrix_rejects_bad_width(width):
    with pytest.raises(ValueError, match="qubit 5|matrix width 13"):
        rust.X(5).to_matrix(width)


def test_dense_matrix_rejects_unrepresentable_natural_width():
    index = 2 * sys.maxsize + 1
    with pytest.raises(ValueError, match=f"qubit index {index}.*matrix width"):
        rust.X(index).to_matrix()


@pytest.mark.parametrize(
    ("mode", "gate_indices", "message"),
    [("replace", [], "circuit has no ticks"), ("update", [0], "request has no gates")],
)
def test_empty_tick_request_index_advice(mode, gate_indices, message):
    request = rust.TickCircuit()
    request.tick()
    with pytest.raises(IndexError, match=message):
        rust.TickCircuit()._apply_tick(request, 0, mode, gate_indices)
