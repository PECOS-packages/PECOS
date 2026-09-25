# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0

"""CPU-only checks of the exact pytket operation used by the MPS backend."""

import cmath
import math

import numpy as np
import pytest
from pecos import f64


@pytest.mark.parametrize(
    ("theta", "phi"),
    [(0.73, -0.41), (math.pi, math.pi), (math.pi / 2, math.pi / 2), (-0.73, 0.41)],
)
def test_mps_pytket_rxyxy2q_matrix(theta: float, phi: float) -> None:
    """PhasedXX matches every complex entry, including global phase, without CUDA."""
    circuit = pytest.importorskip("pytket.circuit")
    # Match the MPS binding's radians-to-half-turn conversion exactly.
    gate = circuit.Op.create(circuit.OpType.PhasedXX, [theta / f64.pi, phi / f64.pi])
    actual = gate.get_unitary()

    # Documented oracle in |00>, |01>, |10>, |11> order on (q0, q1).
    # _apply_two_qubit_matrix passes this matrix unchanged to apply_unitary
    # with [Qubit(q0), Qubit(q1)]: no transpose, permutation, or phase adjustment.
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    expected = np.array(
        [
            [c, 0, 0, -1j * cmath.exp(-2j * phi) * s],
            [0, c, -1j * s, 0],
            [0, -1j * s, c, 0],
            [-1j * cmath.exp(2j * phi) * s, 0, 0, c],
        ],
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
