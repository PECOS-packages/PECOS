# pecos-lp

Small dense linear-programming solver: a two-phase tableau simplex with
Bland's rule, built for the tiny LPs that arise in decoder relaxations (tens
to a couple hundred columns).

Properties:

- Deterministic by construction: fixed iteration order, no randomization;
  repeated solves are byte-identical.
- Loud about numerical trouble: every accepted solution passes a final
  primal-feasibility audit, and failures return `LpOutcome::InternalError`
  rather than a silently degraded optimum.
- Zero dependencies.

Used by the `highs` compatibility facade (`crates/pecos-highs`) that lets the
MWPF decoder build without cmake or a C++ toolchain.
