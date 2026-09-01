# Frontier port manual mutant checklist

The repository has no mutation runner. Apply each compiling edit separately,
run the named killer test(s), and restore the source before trying the next
row. This file tracks mutations owned by the Frontier identity layer; engine
mutations are tracked by `pecos-trellis`.

| Mutant | Exact edit (quoted old → new) | Expected killer test(s) or disposition |
|---|---|---|
| `remove_committee_forward_bonus` | `"let forward_bonus = if is_forward { 1.0 } else { 0.0 };"` → `"let forward_bonus = 0.0;"` | **EQUIVALENT.** The bonus is consulted only after all preceding rank components tie, and `FrontierCommittee::decode` already selects forward when `compare_committee_legs` returns `Equal`. Removing the bonus therefore leaves every binary-committee selection unchanged. |
| `remove_maxlog_committee_guard` | Delete the `MetricMode::MaxLogInt` rejection in `FrontierCommittee::from_sparse_dem`. | `committee_rejects_maxlog_metric` |
