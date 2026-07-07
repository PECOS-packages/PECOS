# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""Validate the reliable-observable prototype against the paper's examples.

Reproduces the fragile-observable example from Serra-Peralta et al.
(arXiv:2505.13599): two fragile observables whose product is reliable.
"""

from __future__ import annotations

import numpy as np
import pytest

stim = pytest.importorskip("stim")

from pecos.qec.reliable_observables import (  # noqa: E402
    _gf2_right_null_space,
    is_reliable,
    reliable_observables,
)


def test_gf2_right_null_space():
    a = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    basis = _gf2_right_null_space(a)
    # rank 2, 3 cols -> 1-d null space, spanned by [1,1,1].
    assert len(basis) == 1
    assert basis[0].tolist() == [1, 1, 1]
    for v in basis:
        assert np.all((a @ v) % 2 == 0)


def test_gf2_full_rank_has_trivial_null_space():
    a = np.eye(3, dtype=np.uint8)
    assert _gf2_right_null_space(a) == []


def _bell_circuit() -> stim.Circuit:
    # Paper's fragile example (eq. Bell_state_fragile_observables):
    # q0=|0>, q1=|+>, CX(1,0), measure both in Z. O0={M(q0)}, O1={M(q1)} are
    # each fragile; their product is reliable (and deterministic).
    c = stim.Circuit()
    c.append("RZ", [0])
    c.append("RX", [1])
    c.append("TICK")
    c.append("CX", [1, 0])
    c.append("TICK")
    c.append("M", [0])
    c.append("M", [1])
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-2)], 0)
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 1)
    return c


def test_bell_fragile_observables_product_is_reliable():
    """The paper's headline example: O0, O1 fragile; O0*O1 reliable."""
    c = _bell_circuit()
    assert reliable_observables(c) == [{0, 1}]
    assert is_reliable(c, {0, 1})
    assert not is_reliable(c, 0)
    assert not is_reliable(c, 1)


def test_memory_single_observable_is_reliable():
    c = stim.Circuit()
    c.append("RZ", [0])
    c.append("TICK")
    c.append("M", [0])
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
    assert reliable_observables(c) == [{0}]
    assert is_reliable(c, 0)


def test_runs_on_pecos_surface_memory_circuit():
    """End-to-end on a real PECOS-generated circuit: a surface-code Z memory has
    a single reliable logical observable."""
    from pecos.qec.surface import LogicalCircuitBuilder, SurfacePatch

    patch = SurfacePatch.create(distance=3)
    b = LogicalCircuitBuilder()
    b.add_patch(patch, "A")
    b.add_memory("A", 3, "Z")
    circuit = stim.Circuit(b.to_stim(p1=0.0, p2=0.0, p_meas=0.0, p_prep=0.0))
    assert circuit.num_observables >= 1
    rel = reliable_observables(circuit)
    # Every raw observable should be reliable on its own for a plain memory.
    for o in range(circuit.num_observables):
        assert is_reliable(circuit, o), f"observable {o} unexpectedly fragile"
    assert rel  # non-empty reliable basis


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
