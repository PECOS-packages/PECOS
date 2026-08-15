"""Ordering + committee golden-fixture generator, run against the upstream frontier package.

Orchestrator-owned oracle: this script and its JSON output are authored and
committed by the reviewer, not by the implementation. The implementation must
never edit them.

Usage (from a clone of the upstream repo with its venv built):
  .venv/bin/python generate_upstream_order_fixtures.py > upstream_order_fixtures.json

Per fixture: mechanisms in TIME order. Expected values from upstream:
- forward_ordering: optimize_column_order permutation over the time-ordered
  columns (deadline reorder). ordering[target] = source index.
- backward_ordering: reverse the forward-ordered columns, then
  optimize_column_order again (build_backward_deadline_ordered_family
  semantics), composed back to original mechanism indices.
- committee: decode_frontier_committee on the forward-ordered model per
  syndrome: status, logical_hat, direction, log_evidence.
"""

from __future__ import annotations

import json
import math
import random
import sys

from frontier import FrontierModel, decode_frontier_committee
from frontier.progressive import (
    FactorTransition,
    OutcomeTransition,
    build_frontier_layout,
    columns_from_factor_transitions,
    optimize_column_order,
)
from tools.frontier_progressive import _reverse_progressive_columns


def build_columns(mechanisms: list) -> list:
    """Build upstream progressive columns from binary mechanisms in time order."""
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
    return list(columns_from_factor_transitions(tuple(factors)))


def random_mechanisms(
    rng: random.Random,
    num_mechs: int,
    num_detectors: int,
    num_observables: int,
) -> list:
    """Sample a seeded random binary-mechanism list (detector-free allowed)."""
    mechs = []
    for _ in range(num_mechs):
        n_d = rng.choice([0, 1, 1, 2, 2, 3])
        dets = sorted(rng.sample(range(num_detectors), min(n_d, num_detectors)))
        obs = sorted(rng.sample(range(num_observables), rng.choice([0, 0, 1, 1, 2])))
        p = rng.uniform(0.01, 0.4)
        mechs.append((round(p, 6), dets, obs))
    return mechs


def main() -> int:
    """Emit the ordering/committee fixture JSON to stdout."""
    fixtures = []

    # F1: hand-built chain where time order != deadline order.
    fixtures.append(
        {
            "name": "chain_reorder",
            "mechanisms": [
                [0.05, [3], [0]],
                [0.04, [0, 1], []],
                [0.03, [0], [1]],
                [0.06, [2, 3], []],
                [0.02, [1, 2], [0, 1]],
                [0.07, [], [0]],
                [0.08, [3], []],
            ],
            "num_detectors": 4,
            "num_observables": 2,
            "syndromes": list(range(16)),
            "pruned": {"k": 3, "delta": 25.0},
        },
    )

    # F2/F3: seeded random models (include detector-free mechanisms).
    for seed, num_mechs, num_dets, num_obs in ((7, 10, 5, 2), (41, 14, 6, 3)):
        rng = random.Random(seed)
        fixtures.append(
            {
                "name": f"order_random_seed{seed}",
                "mechanisms": random_mechanisms(rng, num_mechs, num_dets, num_obs),
                "num_detectors": num_dets,
                "num_observables": num_obs,
                "syndromes": sorted(rng.sample(range(1 << num_dets), 10)),
                "pruned": {"k": 3, "delta": 25.0},
            },
        )

    for fx in fixtures:
        time_columns = build_columns(fx["mechanisms"])
        forward_columns, forward_ordering = optimize_column_order(
            list(time_columns),
            num_detectors=fx["num_detectors"],
        )
        fx["forward_ordering"] = [int(v) for v in forward_ordering]

        reversed_columns = _reverse_progressive_columns(forward_columns)
        _backward_columns, backward_ordering_local = optimize_column_order(
            list(reversed_columns),
            num_detectors=fx["num_detectors"],
        )
        # Compose back to original mechanism indices: reversed[i] came from
        # forward position len-1-i, which came from mechanism forward_ordering[...].
        n = len(forward_ordering)
        backward_in_original = [int(forward_ordering[n - 1 - int(local)]) for local in backward_ordering_local]
        fx["backward_ordering"] = backward_in_original

        model = FrontierModel(
            columns=tuple(forward_columns),
            layout=build_frontier_layout(list(forward_columns), num_detectors=fx["num_detectors"]),
            num_detectors=fx["num_detectors"],
            num_observables=fx["num_observables"],
        )
        committee = []
        for syndrome in fx["syndromes"]:
            r = decode_frontier_committee(
                model,
                syndrome,
                K=fx["pruned"]["k"],
                Delta=fx["pruned"]["delta"],
            )
            committee.append(
                {
                    "syndrome": syndrome,
                    "status": r.status,
                    "logical_hat": r.logical_hat,
                    "direction": r.direction,
                    "log_evidence": r.log_evidence if math.isfinite(r.log_evidence) else None,
                    "engine": r.engine,
                },
            )
        fx["expected_committee"] = committee

    json.dump(
        {"generator": "generate_upstream_order_fixtures.py", "fixtures": fixtures},
        sys.stdout,
        indent=1,
        allow_nan=False,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
