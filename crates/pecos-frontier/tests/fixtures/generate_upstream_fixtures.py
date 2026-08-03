# Golden-fixture generator for pecos-frontier, run against the upstream
# frontier package (github.com/aleverrier/frontier, arXiv:2606.20513).
#
# Orchestrator-owned oracle: this script and its JSON output are authored and
# committed by the reviewer, not by the implementation. The implementation must
# never edit them.
#
# Usage (from a clone of the upstream repo with its venv built):
#   .venv/bin/python generate_upstream_fixtures.py > upstream_fixtures.json
#
# Each fixture: a binary-fault model given as mechanisms [p, [detectors],
# [observables]] in processing order (identical to the column order PECOS uses),
# decoded for every syndrome bitmask in `syndromes`, unpruned (K=10^9,
# Delta=inf) and pruned (K/Delta from the fixture). Expected values come from
# upstream FrontierResult: status, logical_hat, log_evidence, and
# terminal_log_masses (logical label -> log mass, unnormalized).

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


def build_model(mechanisms, num_detectors, num_observables):
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
            )
        )
    columns = tuple(columns_from_factor_transitions(tuple(factors)))
    return FrontierModel(
        columns=columns,
        layout=build_frontier_layout(list(columns), num_detectors=num_detectors),
        num_detectors=num_detectors,
        num_observables=num_observables,
    )


def decode_all(model, syndromes, k, delta):
    out = []
    for syndrome in syndromes:
        r = decode_frontier(model, syndrome, K=k, Delta=delta)
        # Keep the JSON strictly standard: no-path decodes report -inf
        # log_evidence upstream; emit null instead (allow_nan=False enforces).
        out.append(
            {
                "syndrome": syndrome,
                "status": r.status,
                "logical_hat": r.logical_hat,
                "log_evidence": r.log_evidence if math.isfinite(r.log_evidence) else None,
                "terminal_log_masses": {
                    str(k_): v for k_, v in sorted(r.terminal_log_masses.items()) if math.isfinite(v)
                },
                "engine": r.engine,
            }
        )
    return out


def random_mechanisms(rng, num_mechs, num_detectors, num_observables):
    mechs = []
    for _ in range(num_mechs):
        n_d = rng.choice([1, 1, 2, 2, 3])
        dets = sorted(rng.sample(range(num_detectors), min(n_d, num_detectors)))
        obs = sorted(rng.sample(range(num_observables), rng.choice([0, 0, 1, 1, 2])))
        p = rng.uniform(0.01, 0.4)
        mechs.append((round(p, 6), dets, obs))
    return mechs


def main():
    fixtures = []

    # F1: hand-built degeneracy case -- logical-ML differs from most-likely-error.
    # Mechanism 0: p=0.20, flips detector 0, flips observable 0.
    # Mechanisms 1,2: p=0.15 each, flip detector 0, no observable flip.
    # Syndrome {det0}: MLE representative is mech 0 (0.20 > 0.15), but coset mass
    # of label 0 (mech1 alone + mech2 alone + all three) exceeds label 1's mass.
    fixtures.append(
        {
            "name": "degeneracy_ml_vs_mle",
            "mechanisms": [[0.20, [0], [0]], [0.15, [0], []], [0.15, [0], []]],
            "num_detectors": 1,
            "num_observables": 1,
            "syndromes": [0, 1],
            "pruned": {"k": 2, "delta": 100.0},
        }
    )

    # F2: repetition-code-like chain with hyperedge, two observables.
    fixtures.append(
        {
            "name": "rep_chain_hyperedge",
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
        }
    )

    # F3: wide observables -- winning label flips observable index 70.
    fixtures.append(
        {
            "name": "wide_observable_70",
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
        }
    )

    # F4/F5: seeded random models.
    for seed, num_mechs, num_dets, num_obs in ((11, 8, 5, 3), (23, 12, 6, 2)):
        rng = random.Random(seed)
        fixtures.append(
            {
                "name": f"random_seed{seed}",
                "mechanisms": random_mechanisms(rng, num_mechs, num_dets, num_obs),
                "num_detectors": num_dets,
                "num_observables": num_obs,
                "syndromes": sorted(rng.sample(range(1 << num_dets), 12)),
                "pruned": {"k": 3, "delta": 20.0},
            }
        )

    for fx in fixtures:
        model = build_model(fx["mechanisms"], fx["num_detectors"], fx["num_observables"])
        fx["expected_unpruned"] = decode_all(model, fx["syndromes"], UNPRUNED_K, math.inf)
        fx["expected_pruned"] = decode_all(
            model, fx["syndromes"], fx["pruned"]["k"], fx["pruned"]["delta"]
        )

    json.dump({"generator": "generate_upstream_fixtures.py", "fixtures": fixtures}, sys.stdout, indent=1, allow_nan=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
