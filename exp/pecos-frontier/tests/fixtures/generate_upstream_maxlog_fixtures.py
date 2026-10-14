"""Golden-fixture generator for maxlog_int (frontier_lite) metric mode, run against upstream.

Orchestrator-owned oracle: this script and its JSON output are authored and
committed by the reviewer, not by the implementation. The implementation must
never edit them.

Usage (from a clone of the upstream repo):
  uv run python generate_upstream_maxlog_fixtures.py > upstream_maxlog_fixtures.json

Binary-mechanism models only: upstream restricts frontierLite/maxlog_int to the
native binary engine. Each fixture decodes every syndrome under
metric_mode="frontier_lite" at int_metric_scale values 1024 (the fast path) and
64 (the generic fixed-point path), in a wide configuration (K=10^9, Delta=1e6)
and a pruned configuration (K/Delta from the fixture).

Delta is kept finite everywhere: upstream quantizes Delta=inf to the negative
sentinel and clamps it to 0, which silently turns "no delta pruning" into
"prune everything below the best score" and can change the decoded label.
Verified live 2026-08-24 (hat flipped 1 -> 0 on the degeneracy model).

Expected values come from upstream FrontierResult: status, logical_hat,
log_evidence (the winning quantized max-log mass divided by the scale),
terminal_log_masses (per-label MAX-log route mass, not summed coset mass), and
terminal_top_log_mass_gap.
"""

from __future__ import annotations

import json
import math
import random
import sys

from frontier import FrontierModel, decode_frontier
from frontier.progressive import (
    FactorTransition,
    OutcomeTransition,
    build_frontier_layout,
    columns_from_factor_transitions,
)

WIDE_K = 10**9
WIDE_DELTA = 1.0e6
SCALES = (1024, 64)


def build_model(
    mechanisms: list,
    num_detectors: int,
    num_observables: int,
) -> FrontierModel:
    """Build an upstream FrontierModel from binary (p, detectors, observables) mechanisms."""
    factors = []
    for idx, (p, dets, obs) in enumerate(mechanisms):
        det_mask = 0
        for d in dets:
            det_mask |= 1 << d
        log_mask = 0
        for o in obs:
            log_mask |= 1 << o
        factors.append(
            FactorTransition(
                factor_id=idx,
                outcomes=(
                    OutcomeTransition(probability=1.0 - p, detector_mask=0, logical_mask=0),
                    OutcomeTransition(probability=p, detector_mask=det_mask, logical_mask=log_mask),
                ),
                instruction_offset=idx,
                label=f"f{idx}",
            ),
        )
    columns = tuple(columns_from_factor_transitions(tuple(factors)))
    return FrontierModel(
        columns=columns,
        layout=build_frontier_layout(list(columns), num_detectors=num_detectors),
        num_detectors=num_detectors,
        num_observables=num_observables,
    )


def decode_all(model: FrontierModel, syndromes: list, k: int, delta: float, scale: int) -> list:
    """Decode each syndrome under frontier_lite and record JSON-safe expected dicts."""
    out = []
    for syndrome in syndromes:
        r = decode_frontier(
            model,
            syndrome,
            K=k,
            Delta=delta,
            metric_mode="frontier_lite",
            int_metric_scale=scale,
        )
        gap = r.terminal_top_log_mass_gap
        out.append(
            {
                "syndrome": syndrome,
                "status": r.status,
                "logical_hat": r.logical_hat,
                "log_evidence": r.log_evidence if math.isfinite(r.log_evidence) else None,
                "terminal_log_masses": {
                    str(k_): v for k_, v in sorted(r.terminal_log_masses.items()) if math.isfinite(v)
                },
                "terminal_top_log_mass_gap": gap if math.isfinite(gap) else None,
                "engine": r.engine,
            },
        )
    return out


def random_mechanisms(
    rng: random.Random,
    num_mechs: int,
    num_detectors: int,
    num_observables: int,
) -> list:
    """Sample a seeded random binary-mechanism list (matches the binary generator's shape)."""
    mechs = []
    for _ in range(num_mechs):
        n_d = rng.choice([1, 1, 2, 2, 3])
        dets = sorted(rng.sample(range(num_detectors), min(n_d, num_detectors)))
        obs = sorted(rng.sample(range(num_observables), rng.choice([0, 0, 1, 1, 2])))
        p = rng.uniform(0.01, 0.4)
        mechs.append((round(p, 6), dets, obs))
    return mechs


def main() -> int:
    """Emit the fixture JSON to stdout."""
    fixtures = []

    # M1: the degeneracy model where max-log and coset-mass DISAGREE on the
    # winner. Float log-sum-exp picks label 0 (aggregate 0.24-mass routes);
    # max-log picks label 1 (single best route 0.2*0.85*0.85). This is the
    # fixture that proves the implementation really switched metrics.
    fixtures.append(
        {
            "name": "maxlog_metric_flips_winner",
            "mechanisms": [[0.20, [0], [0]], [0.15, [0], []], [0.15, [0], []]],
            "num_detectors": 1,
            "num_observables": 1,
            "syndromes": [0, 1],
            "pruned": {"k": 2, "delta": 100.0},
        },
    )

    # M2: chain with hyperedge, two observables (same corpus as the binary
    # float fixtures, re-decoded under the integer metric).
    fixtures.append(
        {
            "name": "maxlog_rep_chain_hyperedge",
            "mechanisms": [
                [0.05, [0], [0]],
                [0.04, [0, 1], []],
                [0.03, [1, 2], [1]],
                [0.06, [2, 3], []],
                [0.02, [3], [0, 1]],
                [0.07, [1, 2, 3], []],
            ],
            "num_detectors": 4,
            "num_observables": 2,
            "syndromes": list(range(16)),
            "pruned": {"k": 4, "delta": 30.0},
        },
    )

    # M3: wide observables under the integer metric.
    fixtures.append(
        {
            "name": "maxlog_wide_observable_70",
            "mechanisms": [
                [0.10, [0], [70]],
                [0.02, [0], [3]],
                [0.05, [1], [0, 70]],
                [0.03, [0, 1], []],
            ],
            "num_detectors": 2,
            "num_observables": 71,
            "syndromes": [0, 1, 2, 3],
            "pruned": {"k": 8, "delta": 100.0},
        },
    )

    # M4/M5: seeded random models.
    for seed, num_mechs, num_dets, num_obs in ((11, 8, 5, 3), (23, 12, 6, 2)):
        rng = random.Random(seed)
        fixtures.append(
            {
                "name": f"maxlog_random_seed{seed}",
                "mechanisms": random_mechanisms(rng, num_mechs, num_dets, num_obs),
                "num_detectors": num_dets,
                "num_observables": num_obs,
                "syndromes": sorted(rng.sample(range(1 << num_dets), 12)),
                "pruned": {"k": 3, "delta": 20.0},
            },
        )

    for fx in fixtures:
        model = build_model(fx["mechanisms"], fx["num_detectors"], fx["num_observables"])
        fx["expected"] = {}
        for scale in SCALES:
            fx["expected"][str(scale)] = {
                "wide": decode_all(model, fx["syndromes"], WIDE_K, WIDE_DELTA, scale),
                "pruned": decode_all(
                    model,
                    fx["syndromes"],
                    fx["pruned"]["k"],
                    fx["pruned"]["delta"],
                    scale,
                ),
            }

    json.dump(
        {"generator": "generate_upstream_maxlog_fixtures.py", "fixtures": fixtures},
        sys.stdout,
        indent=1,
        allow_nan=False,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
