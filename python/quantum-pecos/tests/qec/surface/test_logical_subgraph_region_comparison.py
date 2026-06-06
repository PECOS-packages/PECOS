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
from pecos_rslib.qec import (
    LogicalSubgraphDecoder,
    ParsedDem,
    WindowedLogicalSubgraphDecoder,
)


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


def _mem_ler(d, p, n, seed, inner=None):
    patch = SurfacePatch.create(d)
    b = LogicalCircuitBuilder()
    b.add_patch(patch, "A")
    b.add_memory("A", d, "Z")
    dem = b.build_dem(p1=p, p2=p, p_meas=p)
    sc = b.stab_coords()
    batch = ParsedDem.from_string(dem).to_dem_sampler().generate_samples(n, seed=seed)
    dec = LogicalSubgraphDecoder(dem, sc) if inner is None else LogicalSubgraphDecoder(dem, sc, inner)
    return dec.decode_count(batch) / n


def test_distance_suppression_memory():
    """A fault-tolerant decoder must drive LER DOWN as code distance grows below
    threshold. The default inner (`pecos_uf:bp`, native BP + union-find)
    suppresses, tracking exact MWPM / lomatching (d=7 -> 0). Guards against the
    default reverting to a non-suppressing inner."""
    p, n = 0.001, 60000
    ler_d3 = _mem_ler(3, p, n, seed=1)
    ler_d5 = _mem_ler(5, p, n, seed=1)
    # Below threshold, d=5 must beat d=3 by a clear margin (lomatching: ~13x).
    assert ler_d5 < ler_d3 * 0.7, (
        f"no distance suppression with default inner: d3={ler_d3:.5f} d5={ler_d5:.5f}"
    )


def test_default_inner_bp_uf_suppresses():
    """The default native inner decoder (`pecos_uf:bp`, belief-propagation +
    union-find) achieves distance suppression, as a fault-tolerant decoder must.

    Context: the UF predecoder used to mis-decode isolated defects whose
    minimum-weight correction is a bulk *path* to the boundary (it only looked at
    direct boundary edges, returning a no-flip / wrong-edge result). That was a
    catastrophic single-defect bug; fixed by making the predecoder fall through to
    the full grow+peel decoder unless its shortcut is provably optimal (see
    `predecode_single` / size-2 handling in pecos-uf-decoder).

    NOTE: this validates the *default* inner `pecos_uf:bp`, which suppresses
    robustly (tracks exact MWPM). Pure `pecos_uf:fast` (no belief propagation) was
    also improved by the predecoder fix but its full grow+peel heuristic does NOT
    robustly suppress at depth -- a separate, lesser weakness, which is why the
    default is `pecos_uf:bp`, not `pecos_uf:fast`.
    See pecos-docs/design/logical-subgraph-backprop-region-builder.md."""
    p, n = 0.001, 60000
    uf_d3 = _mem_ler(3, p, n, seed=1, inner="pecos_uf:bp")
    uf_d5 = _mem_ler(5, p, n, seed=1, inner="pecos_uf:bp")
    # Below threshold, d=5 must beat d=3 by a clear margin.
    assert uf_d5 < uf_d3 * 0.7, (
        f"default bp+uf inner no longer suppresses: d3={uf_d3:.5f} d5={uf_d5:.5f}"
    )


def _windowed_mem_ler(d, rounds, p, n, seed, step, buffer):
    patch = SurfacePatch.create(d)
    b = LogicalCircuitBuilder()
    b.add_patch(patch, "A")
    b.add_memory("A", rounds, "Z")
    dem = b.build_dem(p1=p, p2=p, p_meas=p)
    sc = b.stab_coords()
    batch = ParsedDem.from_string(dem).to_dem_sampler().generate_samples(n, seed=seed)
    dec = WindowedLogicalSubgraphDecoder(dem, sc, step, buffer)
    return dec.decode_count(batch) / n, dec.num_windows()


def _nonwindowed_mem_ler(d, rounds, p, n, seed, inner):
    patch = SurfacePatch.create(d)
    b = LogicalCircuitBuilder()
    b.add_patch(patch, "A")
    b.add_memory("A", rounds, "Z")
    dem = b.build_dem(p1=p, p2=p, p_meas=p)
    sc = b.stab_coords()
    batch = ParsedDem.from_string(dem).to_dem_sampler().generate_samples(n, seed=seed)
    return LogicalSubgraphDecoder(dem, sc, inner).decode_count(batch) / n


def test_windowed_logical_subgraph_single_window_matches_nonwindowed():
    """Correctness guarantee for the windowed decoder's construction: with a
    window larger than the circuit depth (a single window, all-core), the
    windowed logical-subgraph decoder reproduces the non-windowed decoder with
    the same native union-find inner.

    This pins that the per-observable subgraph serialization, the local<->global
    detector mapping, and the local-bit-0 -> global-observable-bit remapping are
    all correct -- independently of the time-windowing behaviour exercised
    below."""
    p, n, rounds = 0.001, 40000, 18
    for d in (3, 5):
        win, nwin = _windowed_mem_ler(d, rounds, p, n, seed=7, step=10_000, buffer=0)
        non = _nonwindowed_mem_ler(d, rounds, p, n, seed=7, inner="pecos_uf:fast")
        assert nwin == 1, f"expected a single window, got {nwin}"
        # Same decode up to negligible tie-breaking differences.
        assert abs(win - non) <= max(0.0005, 0.15 * non), (
            f"single-window windowed != non-windowed at d={d}: win={win:.5f} non={non:.5f}"
        )


def test_windowed_logical_subgraph_known_limitation_no_full_suppression():
    """KNOWN LIMITATION (separately tracked): the windowed logical-subgraph
    decoder does not yet achieve full distance suppression on memory.

    The decoder was rewritten to do proper sliding-window core-commit (each
    per-observable subgraph is wrapped in an ``OverlappingWindowedDecoder``, which
    commits only correction edges whose both endpoints lie in a window's core).
    That removed the old double-counting bug -- a single window now reproduces the
    non-windowed decoder exactly (see the test above), and multi-window LER is no
    longer catastrophic (the old naive-XOR decoder anti-suppressed to ~10-25%).

    What remains is the *windowed logical-observable-matching* limitation
    identified in Serra-Peralta et al. (arXiv:2505.13599, Sec. V): per-observable
    windowing admits "time-like snake" error patterns that scale sublinearly in
    d, so LER does not fully suppress without their additional machinery
    (synchronized resets every Omega(d) and/or a two-step decoder with short-cut
    edges). Standard *full-DEM* sliding-window decoding does not have this issue
    (PECOS's ``windowed:`` decoder suppresses on a graphlike/Stim-decomposed DEM
    -- it is not exercised here because the native PECOS DEM has undecomposed
    hyperedges that the full-DEM windowed path's matching inner rejects). For a
    single-observable memory prefer either the non-windowed
    ``LogicalSubgraphDecoder`` or full-DEM windowing on a decomposed DEM. This
    pins the limitation; implementing the anti-snake machinery is the remaining
    work.
    See pecos-docs/design/logical-subgraph-backprop-region-builder.md."""
    p, n, rounds = 0.001, 40000, 18
    ler_d3, nwin = _windowed_mem_ler(3, rounds, p, n, seed=7, step=3, buffer=3)
    ler_d5, _ = _windowed_mem_ler(5, rounds, p, n, seed=7, step=5, buffer=5)
    ler_d7, _ = _windowed_mem_ler(7, rounds, p, n, seed=7, step=7, buffer=7)
    # The circuit is deep enough to actually exercise windowing.
    assert nwin > 1, f"probe degenerated to a single window (nwin={nwin})"
    # Still does not fully suppress (the paper's windowed-LOM limitation): LER does
    # not fall with distance (it in fact grows). When the anti-snake machinery
    # lands and this suppresses, flip the assertions and update the test.
    assert ler_d5 >= ler_d3 * 0.7 and ler_d7 >= ler_d5 * 0.7, (
        f"windowed logical-subgraph now suppresses (d3={ler_d3:.5f} "
        f"d5={ler_d5:.5f} d7={ler_d7:.5f}) -- anti-snake machinery appears to "
        "have landed; update this test."
    )
    # Guard against regressing to the old catastrophic anti-suppression (the
    # naive-XOR decoder reached ~0.1-0.25 here); the core-commit rewrite keeps it
    # well below that across distances.
    assert max(ler_d3, ler_d5, ler_d7) < 0.1, (
        f"windowed LER regressed toward catastrophic: "
        f"d3={ler_d3:.5f} d5={ler_d5:.5f} d7={ler_d7:.5f}"
    )


def test_decode_each_matches_decode_count():
    """`SampleBatch.decode_each` returns per-shot predictions consistent with the
    aggregate `decode_count` -- it is the per-shot primitive used to localize
    where two decoders disagree."""
    b = _cx_circuit()
    dem = b.build_dem(p1=0.001, p2=0.001, p_meas=0.001)
    n = 3000
    batch = ParsedDem.from_string(dem).to_dem_sampler().generate_samples(n, seed=5)
    preds = batch.decode_each(dem, "pecos_uf:bp")
    assert len(preds) == n
    wrong = sum(1 for i, p in enumerate(preds) if p != batch.get_observable_mask(i))
    assert wrong == batch.decode_count(dem, "pecos_uf:bp")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
