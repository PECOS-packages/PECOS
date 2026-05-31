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

"""Compare observing-region constructions for the logical-subgraph decoder.

The decoder's accuracy depends on which detectors land in each observable's
subgraph. Two constructions:

- **coordinate group-fill** (the shipping path, == lomatching's
  ``get_detector_indices_for_subgraphs``): boundary edges seed
  ``(qubit, stab_type, time)`` groups, then ALL detectors of those groups are
  included.
- **back-propagation / co-flipped** (the papers' derivation, arXiv:2505.13587):
  detectors that share a fault with the observable -- i.e. lie in its
  detecting region. Computed here directly from the DEM ``L`` targets.

`LogicalSubgraphDecoder.from_membership` lets both feed the same decoder, so we
can compare. Finding (recorded in
``pecos-docs/design/logical-subgraph-backprop-region-builder.md``): the raw
back-prop set decodes much WORSE despite being larger, because it lacks the
group-fill structure that makes each subgraph cleanly matchable.
"""

from __future__ import annotations

import pytest
from pecos.qec.surface import LogicalCircuitBuilder, SurfacePatch
from pecos_rslib.qec import LogicalSubgraphDecoder, ParsedDem


def _coflip_membership_from_dem(dem_str: str, num_observables: int) -> list[list[int]]:
    """Back-propagation / co-flipped observing region, read straight off the DEM.

    For each observable O, the detectors of every error mechanism that flips O
    (an error in their shared support flips both -- O's detecting region).
    """
    regions: list[set[int]] = [set() for _ in range(num_observables)]
    for raw_line in dem_str.splitlines():
        line = raw_line.strip()
        if not line.startswith("error("):
            continue
        tokens = line[line.index(")") + 1 :].split()
        dets = [int(t[1:]) for t in tokens if t.startswith("D")]
        obs = [int(t[1:]) for t in tokens if t.startswith("L")]
        for o in obs:
            regions[o].update(dets)
    return [sorted(r) for r in regions]


def _cx_circuit():
    patch = SurfacePatch.create(distance=3)
    nq = patch.geometry.num_data + patch.geometry.num_ancilla
    b = LogicalCircuitBuilder()
    b.add_patch(patch, "C", qubit_offset=0)
    b.add_patch(patch, "T", qubit_offset=nq)
    b.add_memory(["C", "T"], 3, "Z")
    b.add_transversal_cx("C", "T")
    b.add_memory(["C", "T"], 3, "Z")
    return b


def test_from_membership_reproduces_coordinate_path():
    """Feeding the coordinate membership through from_membership reproduces the
    normal coordinate decoder exactly (the seam is behaviour-preserving)."""
    b = _cx_circuit()
    dem = b.build_dem(p1=0.001, p2=0.001, p_meas=0.001)
    sc = b.stab_coords()

    coord = LogicalSubgraphDecoder(dem, sc, "pecos_uf:fast")
    rebuilt = LogicalSubgraphDecoder.from_membership(
        dem,
        coord.observing_regions(),
        "pecos_uf:fast",
    )

    assert rebuilt.subgraph_sizes() == coord.subgraph_sizes()
    batch = ParsedDem.from_string(dem).to_dem_sampler().generate_samples(2000, seed=3)
    assert rebuilt.decode_count(batch) == coord.decode_count(batch)


def test_coordinate_region_beats_raw_backprop_region():
    """The coordinate group-fill region decodes far better than the raw
    back-prop / co-flipped detector set -- group-fill is essential, not
    cosmetic."""
    b = _cx_circuit()
    dem = b.build_dem(p1=0.001, p2=0.001, p_meas=0.001)
    sc = b.stab_coords()

    n = 20000
    batch = ParsedDem.from_string(dem).to_dem_sampler().generate_samples(n, seed=11)

    coord = LogicalSubgraphDecoder(dem, sc, "pecos_uf:fast")
    coflip_membership = _coflip_membership_from_dem(dem, coord.num_observables())
    backprop = LogicalSubgraphDecoder.from_membership(dem, coflip_membership, "pecos_uf:fast")

    coord_ler = coord.decode_count(batch) / n
    backprop_ler = backprop.decode_count(batch) / n

    # The coordinate group-fill region is dramatically better. The gap is large
    # and stable (~20x at d=3, p=0.001); assert a conservative margin.
    assert coord_ler < backprop_ler, (
        f"expected coordinate region to beat raw back-prop: "
        f"coord={coord_ler:.5f} backprop={backprop_ler:.5f}"
    )
    assert coord_ler * 5 < backprop_ler, (
        f"expected a large gap (group-fill essential): "
        f"coord={coord_ler:.5f} backprop={backprop_ler:.5f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
