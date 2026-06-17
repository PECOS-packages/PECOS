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

"""Rigorous inner-decoder study for ``LogicalSubgraphDecoder``.

Answers, with statistics that can actually separate the candidates:

* Fault tolerance / distance suppression -- does logical error rate (LER) fall
  as code distance ``d`` grows below threshold, for each inner decoder?
* Threshold -- where do the per-distance LER curves cross (so above it more
  distance hurts)? Estimated per inner.
* Lowest LER -- at fixed sub-threshold ``p``, which inner wins, and is the gap
  statistically real (non-overlapping Jeffreys intervals)?
* Speed -- decoder build cost vs per-shot decode throughput, separated.

Design choices that fix the under-powered earlier spot-check:

* PAIRED comparison: one sampled batch per (family, d, p, seed) is decoded by
  every inner, so decoder differences are not confounded by sampling noise.
* Sub-threshold ``p`` chosen so LER is large enough (~1e-3..1e-2) that 1e5 shots
  yield hundreds of failures -> tight intervals that resolve 2x differences.
* Multiple seeds for the headline cells (batch-to-batch stability).
* Jeffreys (Bayesian Beta(k+1/2, n-k+1/2)) intervals -- the project's preferred
  binomial CI -- computed via scipy as an analysis oracle (never a runtime dep).
* Build time (decoder construction) separated from decode time (decode_count).

Results are appended as JSON lines to ``results/inner_decoder_study_<phase>.jsonl``
so a run is resumable and analysable independently (see ``--phase analyze``).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from pecos.qec.surface import LogicalCircuitBuilder, SurfacePatch
from pecos_rslib.qec import LogicalSubgraphDecoder, ParsedDem

# Candidate inner decoders for the library default. fusion_blossom_serial is the
# current default (exact MWPM, bundled); pecos_uf:bp is the native option;
# belief_matching is BP+MWPM; pymatching/tesseract are external baselines.
CANDIDATES = [
    "fusion_blossom_serial",
    "pecos_uf:bp",
    "belief_matching",
    "pymatching",
    "tesseract",
]

RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class Cell:
    """One measured (family, d, p, seed, inner) point."""

    family: str
    distance: int
    rounds: int
    p: float
    seed: int
    inner: str
    num_shots: int
    num_errors: int
    ler: float
    build_seconds: float
    decode_seconds: float


# --------------------------------------------------------------------------- #
# Circuit families
# --------------------------------------------------------------------------- #


def _memory_builder(d: int, rounds: int) -> LogicalCircuitBuilder:
    patch = SurfacePatch.create(distance=d)
    b = LogicalCircuitBuilder()
    b.add_patch(patch, "A")
    b.add_memory("A", rounds, "Z")
    return b


def _cx_builder(d: int, rounds: int) -> LogicalCircuitBuilder:
    patch = SurfacePatch.create(distance=d)
    nq = patch.geometry.num_data + patch.geometry.num_ancilla
    b = LogicalCircuitBuilder()
    b.add_patch(patch, "C", qubit_offset=0)
    b.add_patch(patch, "T", qubit_offset=nq)
    b.add_memory(["C", "T"], rounds, "Z")
    b.add_transversal_cx("C", "T")
    b.add_memory(["C", "T"], rounds, "Z")
    return b


FAMILIES = {"memory": _memory_builder, "cx": _cx_builder}


# --------------------------------------------------------------------------- #
# Statistics (Jeffreys interval as an analysis oracle)
# --------------------------------------------------------------------------- #


def jeffreys_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided Jeffreys (Beta) credible interval for a binomial proportion.

    Posterior under the Jeffreys prior Beta(1/2, 1/2) is Beta(k+1/2, n-k+1/2).
    Endpoints clamped to (0, 1) at k=0 / k=n per the standard convention.
    """
    from scipy.stats import beta  # analysis-only oracle, not a PECOS runtime dep

    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k + 0.5, n - k + 0.5))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 0.5, n - k + 0.5))
    return lo, hi


def intervals_disjoint(a: Cell, b: Cell) -> bool:
    """True if the two cells' Jeffreys 95% intervals do not overlap."""
    a_lo, a_hi = jeffreys_ci(a.num_errors, a.num_shots)
    b_lo, b_hi = jeffreys_ci(b.num_errors, b.num_shots)
    return a_hi < b_lo or b_hi < a_lo


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #


def measure_cell(
    family: str,
    d: int,
    rounds: int,
    p: float,
    seed: int,
    inners: list[str],
    n: int,
    dem_source: str = "native",
) -> list[Cell]:
    """Sample ONE batch and decode it with every inner (paired comparison).

    ``dem_source="native"`` uses the PECOS-native ``build_dem`` pipeline (the
    main study). ``dem_source="stim"`` uses the exact DEM the production
    default decode path consumes (``LogicalCircuitBuilder.build_decoder`` with
    ``use_stim_dem=True``): the stim circuit's non-decomposed detector error
    model.
    """
    builder = FAMILIES[family](d, rounds)
    if dem_source == "stim":
        import stim  # analysis-only here; the production path already requires it

        stim_str = builder.to_stim(p1=p, p2=p, p_meas=p)
        dem = str(stim.Circuit(stim_str).detector_error_model(ignore_decomposition_failures=True))
    else:
        dem = builder.build_dem(p1=p, p2=p, p_meas=p)
    sc = builder.stab_coords()
    batch = ParsedDem.from_string(dem).to_dem_sampler().generate_samples(n, seed=seed)

    cells: list[Cell] = []
    for inner in inners:
        t0 = time.perf_counter()
        dec = LogicalSubgraphDecoder(dem, sc, inner)
        t1 = time.perf_counter()
        wrong = dec.decode_count(batch)
        t2 = time.perf_counter()
        cells.append(
            Cell(
                family=family,
                distance=d,
                rounds=rounds,
                p=p,
                seed=seed,
                inner=inner,
                num_shots=n,
                num_errors=wrong,
                ler=wrong / n,
                build_seconds=t1 - t0,
                decode_seconds=t2 - t1,
            ),
        )
    return cells


def _append(path: Path, cells: list[Cell]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for c in cells:
            fh.write(json.dumps(asdict(c)) + "\n")


def _load(path: Path) -> list[Cell]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(Cell(**json.loads(line)))
    return out


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #


def run_suppress(path: Path) -> None:
    """Distance suppression + decoder ranking with resolving statistics.

    Sub-threshold p, large n, multiple seeds; memory (1 obs) and transversal-CX
    (multi-obs, where the earlier spot-check saw fusion beat bp)."""
    done = {(c.family, c.distance, c.p, c.seed, c.inner) for c in _load(path)}
    plan = [
        ("memory", [3, 5, 7], [0.002, 0.003, 0.005], CANDIDATES, [1, 2, 3], 100_000),
        (
            "cx",
            [3, 5, 7],
            [0.002, 0.003, 0.005],
            ["fusion_blossom_serial", "pecos_uf:bp", "belief_matching"],
            [1, 2, 3],
            50_000,
        ),
    ]
    for family, ds, ps, inners, seeds, n in plan:
        for d in ds:
            for p in ps:
                for seed in seeds:
                    todo = [i for i in inners if (family, d, p, seed, i) not in done]
                    if not todo:
                        continue
                    t = time.perf_counter()
                    cells = measure_cell(family, d, d, p, seed, todo, n)
                    _append(path, cells)
                    best = min(cells, key=lambda c: c.ler)
                    print(
                        f"[suppress] {family:6s} d={d} p={p:.3f} seed={seed} n={n}: "
                        + " ".join(f"{c.inner.split(':')[0][:6]}={c.num_errors}" for c in cells)
                        + f"  best={best.inner.split(':')[0]}  ({time.perf_counter() - t:.1f}s)",
                        flush=True,
                    )


def run_threshold(path: Path) -> None:
    """Threshold sweep over p for the policy candidates -- locate the crossing."""
    done = {(c.family, c.distance, c.p, c.seed, c.inner) for c in _load(path)}
    inners = ["fusion_blossom_serial", "pecos_uf:bp", "pymatching"]
    ps = [0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.012]
    for d in [3, 5, 7]:
        for p in ps:
            todo = [i for i in inners if ("memory", d, p, 1, i) not in done]
            if not todo:
                continue
            t = time.perf_counter()
            cells = measure_cell("memory", d, d, p, 1, todo, 50_000)
            _append(path, cells)
            print(
                f"[threshold] memory d={d} p={p:.3f}: "
                + " ".join(f"{c.inner.split(':')[0][:6]}={c.ler:.4f}" for c in cells)
                + f"  ({time.perf_counter() - t:.1f}s)",
                flush=True,
            )


def run_hyperedge(path: Path) -> None:
    """Closes the graphlike-scope caveat: PECOS full DEMs are genuinely
    non-graphlike (weight-8 hyperedges, ~70% of CX errors weight>=3), so test
    whether hyperedge-aware inners beat plain MWPM on the per-observable-subgraph
    path NEAR threshold (where the 2026-04-24 audit saw a 23% hyperedge effect on
    the full memory DEM). If they merely tie, the LogicalSubgraphDecoder default
    is optimal even in the hyperedge regime, not just on graphlike DEMs."""
    done = {(c.family, c.distance, c.p, c.seed, c.inner) for c in _load(path)}
    inners = ["fusion_blossom_serial", "pymatching", "tesseract", "belief_matching", "belief_matching_correlated"]
    plan = [
        ("memory", 5, [0.006, 0.008], [1, 2], 30_000),
        ("memory", 7, [0.008], [1, 2], 20_000),
        ("cx", 5, [0.004], [1, 2], 30_000),
    ]
    for family, d, ps, seeds, n in plan:
        for p in ps:
            for seed in seeds:
                todo = [i for i in inners if (family, d, p, seed, i) not in done]
                if not todo:
                    continue
                cells = measure_cell(family, d, d, p, seed, todo, n)
                _append(path, cells)
                print(
                    f"[hyperedge] {family:6s} d={d} p={p:.3f} seed={seed}: "
                    + " ".join(f"{c.inner.split('_')[0][:6]}={c.num_errors}" for c in cells),
                    flush=True,
                )


def run_stim_spotcheck(path: Path) -> None:
    """Confirm the ranking transfers to the DEM generator the shipped default decodes.

    The production default path (``LogicalCircuitBuilder.build_decoder``,
    ``use_stim_dem=True``) consumes a STIM-generated DEM, while every main study
    cell used the PECOS-native ``build_dem``. This cell repeats the memory
    fusion-vs-bp contrast on the stim DEM (2026-06-11 review follow-up)."""
    done = {(c.family, c.distance, c.p, c.seed, c.inner) for c in _load(path)}
    inners = ["fusion_blossom_serial", "pecos_uf:bp", "pymatching"]
    p = 0.005
    for d in [3, 5]:
        for seed in [1, 2, 3]:
            todo = [i for i in inners if ("memory", d, p, seed, i) not in done]
            if not todo:
                continue
            t = time.perf_counter()
            cells = measure_cell("memory", d, d, p, seed, todo, 100_000, dem_source="stim")
            _append(path, cells)
            print(
                f"[stim_spotcheck] memory d={d} p={p:.3f} seed={seed}: "
                + " ".join(f"{c.inner.split(':')[0][:6]}={c.num_errors}" for c in cells)
                + f"  ({time.perf_counter() - t:.1f}s)",
                flush=True,
            )
    # Pooled verdict for the d=5 contrast (the cell where bp is dominated on
    # the native DEM): report Jeffreys intervals and disjointness.
    agg = _pool(_load(path))
    for d in [3, 5]:
        if ("memory", d, p, "fusion_blossom_serial") not in agg or ("memory", d, p, "pecos_uf:bp") not in agg:
            continue
        fk, fn = agg[("memory", d, p, "fusion_blossom_serial")]
        rk, rn = agg[("memory", d, p, "pecos_uf:bp")]
        flo, fhi = jeffreys_ci(fk, fn)
        rlo, rhi = jeffreys_ci(rk, rn)
        sep = "DISJOINT" if fhi < rlo or rhi < flo else "overlap"
        ratio = (rk / rn) / (fk / fn) if fk else float("inf")
        print(
            f"[stim_spotcheck] pooled d={d}: fusion {fk}/{fn} [{flo:.2e},{fhi:.2e}] vs "
            f"bp {rk}/{rn} [{rlo:.2e},{rhi:.2e}] -- {ratio:.2f}x, {sep}",
            flush=True,
        )


def run_speed(path: Path) -> None:
    """Per-shot decode throughput (build vs decode) at the costly d=7 point."""
    done = {(c.family, c.distance, c.p, c.seed, c.inner) for c in _load(path)}
    for family in ["memory", "cx"]:
        inners = CANDIDATES if family == "memory" else ["fusion_blossom_serial", "pecos_uf:bp", "belief_matching"]
        todo = [i for i in inners if (family, 7, 0.003, 1, i) not in done]
        if not todo:
            continue
        cells = measure_cell(family, 7, 7, 0.003, 1, todo, 50_000)
        _append(path, cells)
        for c in cells:
            us = c.decode_seconds / c.num_shots * 1e6
            print(
                f"[speed] {family:6s} d=7 {c.inner:24s}: build={c.build_seconds * 1e3:7.1f}ms "
                f"decode={c.decode_seconds:6.2f}s  {us:8.1f}us/shot",
                flush=True,
            )


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #


def _pool(cells: list[Cell]) -> dict:
    """Pool repeated seeds for the same (family,d,p,inner) into one binomial."""
    agg: dict[tuple, list[int]] = {}
    for c in cells:
        key = (c.family, c.distance, c.p, c.inner)
        k, n = agg.setdefault(key, [0, 0])
        agg[key] = [k + c.num_errors, n + c.num_shots]
    return agg


def analyze(out_dir: Path) -> str:
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    sup = _load(out_dir / "inner_decoder_study_suppress.jsonl")
    thr = _load(out_dir / "inner_decoder_study_threshold.jsonl")
    spd = _load(out_dir / "inner_decoder_study_speed.jsonl")

    w("# Inner-decoder study results")
    w()
    w("LER with Jeffreys 95% intervals (Beta(k+1/2, n-k+1/2)); seeds pooled into one")
    w("binomial per (family, d, p, inner). `k/n` = failures / shots.")
    w()

    if sup:
        agg = _pool(sup)
        families = sorted({k[0] for k in agg})
        inners = [i for i in CANDIDATES if any(k[3] == i for k in agg)]
        for fam in families:
            ps = sorted({k[2] for k in agg if k[0] == fam})
            ds = sorted({k[1] for k in agg if k[0] == fam})
            w(f"## {fam}: distance suppression + ranking")
            w()
            for p in ps:
                w(f"### p = {p}")
                w()
                w("| inner | " + " | ".join(f"d={d}" for d in ds) + " |")
                w("|---|" + "---|" * len(ds))
                for inner in inners:
                    cells = []
                    for d in ds:
                        kv = agg.get((fam, d, p, inner))
                        cells.append(kv)
                    row = [inner]
                    for kv in cells:
                        if kv is None:
                            row.append("--")
                            continue
                        k, n = kv
                        lo, hi = jeffreys_ci(k, n)
                        row.append(f"{k}/{n} {k / n:.2e} [{lo:.1e},{hi:.1e}]")
                    w("| " + " | ".join(row) + " |")
                w()
                # Decision-relevant contrast per distance: the best inner vs the
                # native pecos_uf:bp candidate (the MWPM-family members are
                # accuracy-tied on these graphlike DEMs, so best-vs-2nd is
                # uninformative -- best-vs-bp is the contrast that picks a default).
                for d in ds:
                    present = [(i, agg[(fam, d, p, i)]) for i in inners if (fam, d, p, i) in agg]
                    if len(present) < 2:
                        continue
                    present.sort(key=lambda t: t[1][0] / t[1][1])
                    bi, (bk, bn) = present[0]
                    bench = "pecos_uf:bp"
                    if (fam, d, p, bench) not in agg or bi == bench:
                        continue
                    rk, rn = agg[(fam, d, p, bench)]
                    blo, bhi = jeffreys_ci(bk, bn)
                    rlo, rhi = jeffreys_ci(rk, rn)
                    sep = "DISJOINT" if bhi < rlo else "overlap"
                    ratio = (rk / rn) / (bk / bn) if bk else float("inf")
                    w(
                        f"- d={d}: best **{bi}** {bk / bn:.2e} vs {bench} {rk / rn:.2e} "
                        f"({ratio:.1f}x) -- Jeffreys intervals {sep}",
                    )
                w()
            # Suppression check + exponent per inner (pooled across seeds).
            w(f"### {fam}: suppression exponent (LER ~ (p/p_th)^((d+1)/2))")
            w()
            for p in ps:
                for inner in inners:
                    seq = [(d, agg[(fam, d, p, inner)]) for d in ds if (fam, d, p, inner) in agg]
                    seq = [(d, kv) for d, kv in seq if kv[0] > 0]  # need nonzero to log
                    if len(seq) < 2:
                        continue
                    suppresses = all(
                        seq[i + 1][1][0] / seq[i + 1][1][1] < seq[i][1][0] / seq[i][1][1] for i in range(len(seq) - 1)
                    )
                    ratios = [
                        (seq[i][1][0] / seq[i][1][1]) / (seq[i + 1][1][0] / seq[i + 1][1][1])
                        for i in range(len(seq) - 1)
                    ]
                    tag = "suppresses" if suppresses else "NOT monotone"
                    w(f"- p={p} {inner}: {tag}; per-step LER ratio " + ", ".join(f"{r:.1f}x" for r in ratios))
            w()

    if thr:
        agg = _pool(thr)
        inners = sorted({k[3] for k in agg})
        ds = sorted({k[1] for k in agg})
        ps = sorted({k[2] for k in agg})
        w("## memory: threshold crossing")
        w()
        for inner in inners:
            w(f"### {inner}")
            w()
            w("| p | " + " | ".join(f"d={d}" for d in ds) + " |")
            w("|---|" + "---|" * len(ds))
            for p in ps:
                row = [f"{p:.3f}"]
                for d in ds:
                    kv = agg.get(("memory", d, p, inner))
                    row.append(f"{kv[0] / kv[1]:.2e}" if kv else "--")
                w("| " + " | ".join(row) + " |")
            # crossing estimate: smallest p where d=max no longer beats d=min
            cross = None
            d_lo, d_hi = ds[0], ds[-1]
            for p in ps:
                a = agg.get(("memory", d_lo, p, inner))
                b = agg.get(("memory", d_hi, p, inner))
                if a and b and b[0] / b[1] >= a[0] / a[1]:
                    cross = p
                    break
            w()
            w(
                f"- threshold estimate (d={d_hi} stops beating d={d_lo}): "
                + (f"~{cross}" if cross else f"above {ps[-1]} (not reached)"),
            )
            w()

    if spd:
        w("## speed (d=7, p=0.003, n per cell as sampled)")
        w()
        w("| family | inner | build ms | decode s | us/shot |")
        w("|---|---|---:|---:|---:|")
        for c in sorted(spd, key=lambda c: (c.family, c.decode_seconds)):
            us = c.decode_seconds / c.num_shots * 1e6
            w(f"| {c.family} | {c.inner} | {c.build_seconds * 1e3:.1f} | {c.decode_seconds:.2f} | {us:.1f} |")
        w()

    return "\n".join(lines)


# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--phase",
        required=True,
        choices=["suppress", "threshold", "hyperedge", "speed", "stim_spotcheck", "analyze", "smoke"],
    )
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()

    if args.phase == "smoke":
        cells = measure_cell("memory", 3, 3, 0.005, 1, ["fusion_blossom_serial", "pecos_uf:bp"], 2000)
        for c in cells:
            lo, hi = jeffreys_ci(c.num_errors, c.num_shots)
            print(
                f"smoke {c.inner}: {c.num_errors}/{c.num_shots} ler={c.ler:.4f} "
                f"CI=[{lo:.4f},{hi:.4f}] build={c.build_seconds * 1e3:.1f}ms decode={c.decode_seconds:.3f}s",
            )
        cx = measure_cell("cx", 3, 3, 0.005, 1, ["fusion_blossom_serial"], 2000)
        print(f"smoke cx: {cx[0].num_errors}/{cx[0].num_shots} ler={cx[0].ler:.4f}")
        return

    if args.phase == "analyze":
        report = analyze(args.out)
        print(report)
        (args.out / "inner_decoder_study_report.md").write_text(report + "\n")
        print(f"\n[written] {args.out / 'inner_decoder_study_report.md'}")
        return

    path = args.out / f"inner_decoder_study_{args.phase}.jsonl"
    {
        "suppress": run_suppress,
        "threshold": run_threshold,
        "hyperedge": run_hyperedge,
        "speed": run_speed,
        "stim_spotcheck": run_stim_spotcheck,
    }[args.phase](path)
    print(f"[done] {args.phase} -> {path}", flush=True)


if __name__ == "__main__":
    main()
