"""Golden-fixture generator for N-ary factor models, run against the upstream frontier package.

Orchestrator-owned oracle: this script and its JSON output are authored and
committed by the reviewer, not by the implementation. The implementation must
never edit them.

Usage (from a clone of the upstream repo):
  uv run python generate_upstream_nary_fixtures.py > upstream_nary_fixtures.json

Each fixture: a multi-outcome factor model given as factors, where every factor
is a list of outcomes [p, [detectors], [observables]] whose probabilities sum
to 1, in processing order (identical to the column order PECOS uses). Each
syndrome bitmask in `syndromes` is decoded unpruned (K=10^9, Delta=inf) and
pruned (K/Delta from the fixture). Expected values come from upstream
FrontierResult: status, logical_hat, log_evidence, and terminal_log_masses
(logical label -> log mass, unnormalized).

Oracle hygiene: every unpruned decode is cross-checked between the upstream
native choice engine and the pure-Python reference engine before it is
emitted; disagreement aborts generation. Pruned decodes record the engine
upstream's auto dispatch selected.
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

UNPRUNED_K = 10**9


def build_model(
    factors: list,
    num_detectors: int,
    num_observables: int,
) -> FrontierModel:
    """Build an upstream FrontierModel from N-ary factor outcome lists."""
    transitions = []
    for idx, outcomes in enumerate(factors):
        built = []
        for p, dets, obs in outcomes:
            det_mask = 0
            for d in dets:
                det_mask |= 1 << d
            log_mask = 0
            for o in obs:
                log_mask |= 1 << o
            built.append(
                OutcomeTransition(probability=p, detector_mask=det_mask, logical_mask=log_mask),
            )
        transitions.append(
            FactorTransition(
                factor_id=idx,
                outcomes=tuple(built),
                instruction_offset=idx,
                label=f"f{idx}",
            ),
        )
    columns = tuple(columns_from_factor_transitions(tuple(transitions)))
    return FrontierModel(
        columns=columns,
        layout=build_frontier_layout(list(columns), num_detectors=num_detectors),
        num_detectors=num_detectors,
        num_observables=num_observables,
    )


def result_to_dict(syndrome: int, r) -> dict:
    """Record one upstream FrontierResult as a JSON-safe dict."""
    # Keep the JSON strictly standard: no-path decodes report -inf
    # log_evidence upstream; emit null instead (allow_nan=False enforces).
    return {
        "syndrome": syndrome,
        "status": r.status,
        "logical_hat": r.logical_hat,
        "log_evidence": r.log_evidence if math.isfinite(r.log_evidence) else None,
        "terminal_log_masses": {
            str(k_): v for k_, v in sorted(r.terminal_log_masses.items()) if math.isfinite(v)
        },
        "engine": r.engine,
    }


def decode_all(model: FrontierModel, syndromes: list, k: int, delta: float, cross_check: bool) -> list:
    """Decode each syndrome; when cross_check, require native/python agreement."""
    out = []
    for syndrome in syndromes:
        r = decode_frontier(model, syndrome, K=k, Delta=delta)
        if cross_check:
            ref = decode_frontier(model, syndrome, K=k, Delta=delta, _engine="python")
            if ref.status != r.status or ref.logical_hat != r.logical_hat:
                raise AssertionError(
                    f"engine disagreement at syndrome {syndrome}: "
                    f"{r.engine} ({r.status}, {r.logical_hat}) vs "
                    f"python ({ref.status}, {ref.logical_hat})"
                )
            if math.isfinite(r.log_evidence) and abs(ref.log_evidence - r.log_evidence) > 1e-9:
                raise AssertionError(
                    f"log_evidence disagreement at syndrome {syndrome}: "
                    f"{r.log_evidence!r} vs {ref.log_evidence!r}"
                )
        out.append(result_to_dict(syndrome, r))
    return out


def random_factors(
    rng: random.Random,
    num_factors: int,
    num_detectors: int,
    num_observables: int,
) -> list:
    """Sample a seeded random N-ary factor list (2-4 outcomes per factor)."""
    factors = []
    for _ in range(num_factors):
        n_outcomes = rng.choice([2, 3, 3, 4])
        raw = [rng.uniform(0.05, 1.0) for _ in range(n_outcomes)]
        total = sum(raw)
        probs = [round(x / total, 6) for x in raw]
        # Absorb rounding drift into the largest outcome so the sum is exact.
        drift = 1.0 - sum(probs)
        probs[probs.index(max(probs))] = round(max(probs) + drift, 12)
        outcomes = []
        for i, p in enumerate(probs):
            if i == 0:
                # Baseline-style outcome: usually trivial, sometimes not.
                if rng.random() < 0.3:
                    dets = sorted(rng.sample(range(num_detectors), 1))
                else:
                    dets = []
                obs = []
            else:
                n_d = rng.choice([1, 1, 2])
                dets = sorted(rng.sample(range(num_detectors), min(n_d, num_detectors)))
                obs = sorted(rng.sample(range(num_observables), rng.choice([0, 1, 1])))
            outcomes.append([p, dets, obs])
        factors.append(outcomes)
    return factors


def main() -> int:
    """Emit the fixture JSON to stdout."""
    fixtures = []

    # N1: hand-built 3-outcome degeneracy -- coset-mass winner differs from the
    # single most likely outcome. Factor 0 has three outcomes: trivial baseline,
    # a det-0 flip carrying observable 0 (p=0.25), and a det-0 flip carrying no
    # observable (p=0.15). Factor 1 independently explains det 0 without an
    # observable flip (p=0.2 binary-shaped). Syndrome {det0}: label-0 mass
    # aggregates the 0.15 outcome and factor-1 routes; label-1 mass is the
    # single 0.25 route.
    fixtures.append(
        {
            "name": "nary_degeneracy_three_outcomes",
            "factors": [
                [[0.60, [], []], [0.25, [0], [0]], [0.15, [0], []]],
                [[0.80, [], []], [0.20, [0], []]],
            ],
            "num_detectors": 1,
            "num_observables": 1,
            "syndromes": [0, 1],
            "pruned": {"k": 2, "delta": 100.0},
        },
    )

    # N2: mixed binary-shaped and genuinely N-ary factors over a small chain,
    # two observables, includes a 4-outcome factor.
    fixtures.append(
        {
            "name": "nary_mixed_chain",
            "factors": [
                [[0.90, [], []], [0.10, [0], [0]]],
                [[0.70, [], []], [0.12, [0, 1], []], [0.10, [1], [1]], [0.08, [0], [0, 1]]],
                [[0.85, [], []], [0.09, [1, 2], []], [0.06, [2], [0]]],
                [[0.95, [], []], [0.05, [2, 3], [1]]],
                [[0.75, [], []], [0.15, [3], []], [0.10, [1, 3], [0]]],
            ],
            "num_detectors": 4,
            "num_observables": 2,
            "syndromes": list(range(16)),
            "pruned": {"k": 4, "delta": 30.0},
        },
    )

    # N3: degenerate outcome shapes -- a zero-probability outcome (dropped),
    # a single-outcome forced factor with nonzero masks (forced layer), and a
    # probability-1 outcome after a zero-probability sibling.
    fixtures.append(
        {
            "name": "nary_degenerate_outcomes",
            "factors": [
                [[1.0, [0], [0]]],
                [[0.60, [], []], [0.0, [1], []], [0.40, [0, 1], []]],
                [[0.0, [0], [1]], [1.0, [], []]],
                [[0.55, [], []], [0.30, [1], [1]], [0.15, [0], []]],
            ],
            "num_detectors": 2,
            "num_observables": 2,
            "syndromes": [0, 1, 2, 3],
            "pruned": {"k": 2, "delta": 50.0},
        },
    )

    # N4: two-outcome factors whose baseline masks are nonempty -- not
    # representable as a single binary DEM mechanism without a forced toggle.
    fixtures.append(
        {
            "name": "nary_nonempty_baseline_pairs",
            "factors": [
                [[0.80, [0], []], [0.20, [1], [0]]],
                [[0.65, [1], [1]], [0.35, [0, 2], []]],
                [[0.90, [], []], [0.10, [2], [0, 1]]],
            ],
            "num_detectors": 3,
            "num_observables": 2,
            "syndromes": list(range(8)),
            "pruned": {"k": 3, "delta": 40.0},
        },
    )

    # N5: wide observables -- winning labels can flip observable index 70.
    fixtures.append(
        {
            "name": "nary_wide_observable_70",
            "factors": [
                [[0.75, [], []], [0.15, [0], [70]], [0.10, [0], [3]]],
                [[0.85, [], []], [0.10, [1], [0, 70]], [0.05, [0, 1], []]],
            ],
            "num_detectors": 2,
            "num_observables": 71,
            "syndromes": [0, 1, 2, 3],
            "pruned": {"k": 8, "delta": 100.0},
        },
    )

    # N6/N7: seeded random N-ary models.
    for seed, num_factors, num_dets, num_obs in ((31, 7, 5, 3), (47, 10, 6, 2)):
        rng = random.Random(seed)
        fixtures.append(
            {
                "name": f"nary_random_seed{seed}",
                "factors": random_factors(rng, num_factors, num_dets, num_obs),
                "num_detectors": num_dets,
                "num_observables": num_obs,
                "syndromes": sorted(rng.sample(range(1 << num_dets), 12)),
                "pruned": {"k": 3, "delta": 20.0},
            },
        )

    for fx in fixtures:
        model = build_model(fx["factors"], fx["num_detectors"], fx["num_observables"])
        fx["expected_unpruned"] = decode_all(model, fx["syndromes"], UNPRUNED_K, math.inf, cross_check=True)
        fx["expected_pruned"] = decode_all(
            model,
            fx["syndromes"],
            fx["pruned"]["k"],
            fx["pruned"]["delta"],
            cross_check=False,
        )

    json.dump(
        {"generator": "generate_upstream_nary_fixtures.py", "fixtures": fixtures},
        sys.stdout,
        indent=1,
        allow_nan=False,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
