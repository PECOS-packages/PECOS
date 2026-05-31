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


def _groupfill_membership(dem_str: str, stab_coords, *, seed_all: bool) -> list[list[int]]:
    """Coordinate group-fill membership, parameterized by the seed rule.

    ``seed_all=False`` seeds only 1-detector boundary edges (the shipping
    coordinate path / lomatching). ``seed_all=True`` seeds from every detector of
    any O-flipping mechanism (the back-propagation / detecting-region crossings),
    then group-fills the same way -- a strictly broader region.
    """
    from collections import defaultdict

    det_coords: dict[int, tuple[float, ...]] = {}
    mechs: list[tuple[list[int], list[int]]] = []
    for raw in dem_str.splitlines():
        ln = raw.strip()
        if ln.startswith("detector("):
            coords = tuple(float(x) for x in ln[ln.index("(") + 1 : ln.index(")")].split(","))
            for t in ln[ln.index(")") + 1 :].split():
                if t.startswith("D"):
                    det_coords[int(t[1:])] = coords
        elif ln.startswith("error("):
            toks = ln[ln.index(")") + 1 :].split()
            mechs.append(
                ([int(t[1:]) for t in toks if t.startswith("D")],
                 [int(t[1:]) for t in toks if t.startswith("L")]),
            )

    coords_to_stab: dict[tuple, tuple[int, str]] = {}
    for li, q in enumerate(stab_coords):
        for st in ("X", "Z"):
            for c in q[st]:
                coords_to_stab[tuple(map(float, c))] = (li, st)

    det_group: dict[int, tuple] = {}
    group_dets: dict[tuple, list[int]] = defaultdict(list)
    for d, c in det_coords.items():
        spatial, time = c[:-1], c[-1]
        if spatial in coords_to_stab:
            li, st = coords_to_stab[spatial]
            det_group[d] = (li, st, time)
            group_dets[(li, st, time)].append(d)

    nobs = 1 + max((o for _, obs in mechs for o in obs), default=-1)
    seeds: list[set] = [set() for _ in range(nobs)]
    for dets, obs in mechs:
        if not obs:
            continue
        seed_dets = dets if (seed_all or len(dets) == 1) else []
        for o in obs:
            for d in seed_dets:
                if d in det_group:
                    seeds[o].add(det_group[d])
    return [sorted({d for g in seeds[o] for d in group_dets[g]}) for o in range(nobs)]


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


def test_coordinate_seeding_reproduces_shipping_path():
    """The group-fill helper with boundary-edge seeding reproduces the shipping
    coordinate membership exactly (validates the helper used for the broader
    back-prop comparison)."""
    b = _cx_circuit()
    dem = b.build_dem(p1=0.001, p2=0.001, p_meas=0.001)
    sc = b.stab_coords()
    coord = LogicalSubgraphDecoder(dem, sc, "pecos_uf:fast")
    helper = _groupfill_membership(dem, sc, seed_all=False)
    assert helper == [sorted(r) for r in coord.observing_regions()]


def test_coordinate_beats_backprop_seeded_groupfill():
    """The faithful 'next step': seed the same group-fill from the operator's
    back-propagation crossings (all O-flipping mechanism detectors) instead of
    boundary edges. It is strictly broader than the coordinate region and
    decodes worse -- confirming the boundary-edge seeding IS the right (faithful)
    back-propagation region, and broadening hurts."""
    b = _cx_circuit()
    dem = b.build_dem(p1=0.001, p2=0.001, p_meas=0.001)
    sc = b.stab_coords()

    coord_membership = _groupfill_membership(dem, sc, seed_all=False)
    backprop_membership = _groupfill_membership(dem, sc, seed_all=True)

    # Coordinate region is a strict subset of the back-prop-seeded region.
    for c, bp in zip(coord_membership, backprop_membership, strict=True):
        assert set(c) <= set(bp)
    assert sum(len(bp) for bp in backprop_membership) > sum(len(c) for c in coord_membership)

    n = 20000
    batch = ParsedDem.from_string(dem).to_dem_sampler().generate_samples(n, seed=13)
    coord = LogicalSubgraphDecoder.from_membership(dem, coord_membership, "pecos_uf:fast")
    backprop = LogicalSubgraphDecoder.from_membership(dem, backprop_membership, "pecos_uf:fast")
    coord_ler = coord.decode_count(batch) / n
    backprop_ler = backprop.decode_count(batch) / n
    assert coord_ler < backprop_ler, (
        f"coord={coord_ler:.5f} backprop-seeds={backprop_ler:.5f}"
    )


def test_bp_inner_is_lower_ler_default():
    """The default inner decoder is the native UF+BP (`pecos_uf:bp`), which
    reaches exact-MWPM accuracy and decodes with lower LER than the older
    `pecos_uf:fast` union-find. The gap grows with distance; assert it at d=3."""
    b = _cx_circuit()
    dem = b.build_dem(p1=0.002, p2=0.002, p_meas=0.002)
    sc = b.stab_coords()

    n = 40000
    batch = ParsedDem.from_string(dem).to_dem_sampler().generate_samples(n, seed=4)

    default = LogicalSubgraphDecoder(dem, sc)  # no inner -> the new default
    bp = LogicalSubgraphDecoder(dem, sc, "pecos_uf:bp")
    fast = LogicalSubgraphDecoder(dem, sc, "pecos_uf:fast")
    exact = LogicalSubgraphDecoder(dem, sc, "pymatching")

    default_ler = default.decode_count(batch) / n
    bp_ler = bp.decode_count(batch) / n
    fast_ler = fast.decode_count(batch) / n
    exact_ler = exact.decode_count(batch) / n

    # The default IS pecos_uf:bp.
    assert default_ler == bp_ler
    # BP inner beats the old fast default...
    assert bp_ler < fast_ler, f"bp={bp_ler:.5f} fast={fast_ler:.5f}"
    # ...and matches exact MWPM (native UF+BP reaches the optimum on graphlike subgraphs).
    assert abs(bp_ler - exact_ler) <= 0.0005, f"bp={bp_ler:.5f} exact={exact_ler:.5f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
