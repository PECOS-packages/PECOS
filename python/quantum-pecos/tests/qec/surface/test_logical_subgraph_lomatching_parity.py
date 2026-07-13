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

"""Differential test: PECOS LogicalSubgraphDecoder observing-region selection
vs the reference algorithm shipped by lomatching.

PECOS's `coordinate_membership_from_dem` (exposed via
`LogicalSubgraphDecoder.observing_regions`) is the same boundary-edge +
stabilizer-coordinate vertex selection that lomatching ships for its MWPM
subgraphs (Serra-Peralta et al., arXiv:2505.13599). The papers justify the
construction via Clifford back-propagation, but the shipping decoder uses this
coordinate path -- so PECOS should agree with it detector-for-detector.

The oracle below is a faithful port of
``lomatching.util.get_detector_indices_for_subgraphs``
(``~/Repos/lomatching/lomatching/util.py:255-353``), depending only on stim so
the full lomatching decoder stack (numba/galois/ldpc/pymatching) is not required.
"""

from __future__ import annotations

import pytest
from pecos.qec.surface import LogicalCircuitBuilder, SurfacePatch
from pecos_rslib.qec import LogicalSubgraphDecoder

stim = pytest.importorskip("stim", reason="stim is the differential-test oracle dependency")


def _lomatching_reference_membership(dem_str: str, stab_coords) -> list[list[int]]:
    """Faithful port of lomatching's ``get_detector_indices_for_subgraphs``.

    Returns, per observable (sorted by observable index), the sorted detector
    ids in that observable's observing region. Reference:
    Serra-Peralta et al. arXiv:2505.13599; lomatching/util.py:255-353.
    """
    dem = stim.DetectorErrorModel(dem_str).flattened()

    det_to_coords = {d: tuple(map(float, c)) for d, c in dem.get_detector_coordinates().items()}
    coords_to_det = {c: d for d, c in det_to_coords.items()}

    # spatial coord (all but the trailing time element) -> (logical qubit, stab type)
    coords_to_stab: dict[tuple, tuple[int, str]] = {}
    for l_ind, qubit in enumerate(stab_coords):
        for stab_type in ("X", "Z"):
            for coord in qubit[stab_type]:
                coords_to_stab[tuple(map(float, coord))] = (l_ind, stab_type)

    # boundary edges = single-detector error mechanisms that flip an observable
    bd_edges_obs: dict[int, list[int]] = {o: [] for o in range(dem.num_observables)}
    for instr in dem:
        if instr.type != "error":
            continue
        dets = [t.val for t in instr.targets_copy() if t.is_relative_detector_id()]
        if len(dets) != 1:
            continue
        for o in (t.val for t in instr.targets_copy() if t.is_logical_observable_id()):
            bd_edges_obs[o] += dets

    # (logical qubit, stab type, time) seeds for each observable
    lst_obs: dict[int, set[tuple[int, str, float]]] = {o: set() for o in range(dem.num_observables)}
    for obs, dets in bd_edges_obs.items():
        for det in dets:
            coords = det_to_coords[det]
            l_ind, stab = coords_to_stab[coords[:-1]]
            lst_obs[obs].add((l_ind, stab, coords[-1]))

    # include every detector of those (qubit, stab, time) groups
    membership: list[list[int]] = []
    for obs in sorted(lst_obs):
        inds: list[int] = []
        for l_ind, stab, time in lst_obs[obs]:
            for c in stab_coords[l_ind][stab]:
                coord = (*map(float, c), time)
                inds.append(coords_to_det[coord])
        membership.append(sorted(inds))
    return membership


def _assert_parity(dem_str: str, stab_coords) -> None:
    decoder = LogicalSubgraphDecoder(dem_str, stab_coords, "pecos_uf:fast")
    pecos = [sorted(r) for r in decoder.observing_regions()]
    reference = _lomatching_reference_membership(dem_str, stab_coords)

    assert len(pecos) == len(reference), f"observable count mismatch: PECOS {len(pecos)} vs lomatching {len(reference)}"
    for obs, (p, r) in enumerate(zip(pecos, reference, strict=True)):
        assert p == r, f"observable {obs} membership differs:\n PECOS={p}\n lomatching={r}"


def test_parity_memory_z():
    """Single-patch Z memory: PECOS regions == lomatching regions."""
    b = LogicalCircuitBuilder()
    b.add_patch(SurfacePatch.create(distance=3), "A")
    b.add_memory("A", 3, "Z")
    dem_str = b.build_dem(p1=0.001, p2=0.001, p_meas=0.001)
    _assert_parity(dem_str, b.stab_coords())


def test_parity_transversal_cx():
    """Two-patch transversal CX (the hyperedge case): regions must agree."""
    patch = SurfacePatch.create(distance=3)
    nq = patch.geometry.num_data + patch.geometry.num_ancilla
    b = LogicalCircuitBuilder()
    b.add_patch(patch, "C", qubit_offset=0)
    b.add_patch(patch, "T", qubit_offset=nq)
    b.add_memory(["C", "T"], 3, "Z")
    b.add_transversal_cx("C", "T")
    b.add_memory(["C", "T"], 3, "Z")
    dem_str = b.build_dem(p1=0.001, p2=0.001, p_meas=0.001)
    _assert_parity(dem_str, b.stab_coords())
