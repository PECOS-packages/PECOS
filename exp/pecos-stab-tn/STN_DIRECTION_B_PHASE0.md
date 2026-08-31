# Direction B Phase 0: lazy compensation-frame shadow

## Outcome

**Verdict: kill Direction B as specified for the exact query path.**

The shadow model predicts positive pre-read time in all four campaign cells, but
the decisive captured n=64 replay does not preserve the eager conditional
state. The immediate forced-outcome probability agrees exactly, while the
squared normalized overlap on two deterministic iterative-amplitude samples is
only `0.371292573035989337`. Materializing the ten-operation B frame does not
repair the different truncation history; it costs another `5.208853620 s` and
leaves a different bond profile.

This is a correctness falsifier, not a throughput judgment. The favorable
model cannot justify building B on an exact path whose reference state it does
not reproduce.

## Method

The opt-in telemetry runs a `(shadow_tableau, ordered Vec<DeferredOp>)` beside
the real exact prefix query. The real tableau/MPS trajectory still owns every
returned probability and is never read from or mutated by the shadow.

The shadow pair is independently cloned at every trie branch. Before each
forced projection it uses the same deferred pre-reduction helper as the live
Lazy path, queues compensation CNOTs without touching an MPS, decomposes `Z_q`,
and reverse-conjugates the complete Pauli through the queue with the existing
deferred-operation conjugation helper. It then uses the live Lazy
outcome-dependent inverse/gauge helper to update its tableau and queue. Both
sibling outcomes get independent clones and records.

The campaign used CPU 2, release mode, 1 warmup plus 3 timed repetitions, 16
queries, and the existing projection-locality flag. Tables below select each
arm's median-query timed run. The four cells were:

- `sparse-n16-t2n`
- `sparse-n32-tn`
- `sparse-n32-t2n`
- `sparse-n64-t2n`

MAST was not exercised: these campaign cells construct `StabMps` and perform no
MAST injection during their queries, so a separate MAST lane would have no
events to observe.

## Support and queue distributions

All quintiles are `min/q25/median/q75/max`. "B same population" restricts B to
the eager direct-sum projector events, making the span comparison paired.
"B all" includes every shadow projection matched to an eager event.

| Cell | Eager direct span | B same-population span | B all span | Queue at projection | Queue after outcome update |
|---|---:|---:|---:|---:|---:|
| sparse-n16-t2n | 0/2/6/13/15 | 3/11/12/15/15 | 0/7/11/13/15 | 0/45/53/61/65 | 4/46/56/63/65 |
| sparse-n32-tn | 0/2/10/22/30 | 0/3/14/25/30 | 0/1/13/19/30 | 0/41/58/83/100 | 0/42/60/85/104 |
| sparse-n32-t2n | 0/5/14/24/30 | 0/0/14/29/31 | 0/0/14/29/31 | 1/77/131/168/194 | 4/85/141/170/197 |
| sparse-n64-t2n | 0/21/37/53/61 | 0/23/47/55/61 | 0/15/45/53/61 | 3/127/206/302/415 | 10/132/209/310/427 |

Event populations were 154 eager-direct / 208 all-shadow at n=16, 372/464
for sparse n=32 tn, 465/465 for sparse n=32 2tn, and 884/980 at n=64.

Queue growth is monotone along every branch because there were no
campaign-path flushes. The following are median queue lengths at
depth milestones (`at projection -> after outcome update`):

| Cell | d=0 | d=n/4 | d=n/2 | d=3n/4 | d=n-1 |
|---|---:|---:|---:|---:|---:|
| sparse-n16-t2n | 0 -> 4 | 30 -> 35 | 49 -> 50 | 61 -> 63 | 64 -> 64 |
| sparse-n32-tn | 0 -> 0 | 37 -> 40 | 54 -> 57 | 82 -> 85 | 96 -> 99 |
| sparse-n32-t2n | 1 -> 4 | 62 -> 75 | 118 -> 129 | 169 -> 169 | 190 -> 194 |
| sparse-n64-t2n | 3 -> 10 | 116 -> 123 | 206 -> 206 | 300 -> 308 | 410 -> 422 |

## Net-time model

The reported equation is

`Delta_T_B = T_prereduction_removed - (T_extra_projection + T_frame + T_flush_read)`.

`T_prereduction_removed` is the measured eager pre-reduction bucket. `T_frame`
is measured shadow algebra plus shadow-clone bookkeeping. `T_extra_projection`
is a **MODEL**, applying the Stage 5a event-calibrated chi-cubed walk/SVD cost
to the conjugated B support versus its paired eager projector. A negative term
means the modeled B projector is cheaper than eager for that event mix.
`T_flush_read` is a **MODEL**, calibrated from measured pre-reduction cost per
chi-cubed unit. No campaign query performed a read that required a flush, so it
is zero in these cells; the replay measures a materializing read separately.

| Cell | T pre-reduction removed, s (MEASURED) | T extra projection, s (MODEL) | T frame, s (MEASURED) | T flush read, s (MODEL) | Delta T B, s |
|---|---:|---:|---:|---:|---:|
| sparse-n16-t2n | 0.170219286 | 0.021293963 | 0.001373292 | 0.000000000 | +0.147552031 |
| sparse-n32-tn | 2.660737975 | 0.059249947 | 0.007315282 | 0.000000000 | +2.594172746 |
| sparse-n32-t2n | 76.831504719 | -2.184075784 | 0.010205934 | 0.000000000 | +79.005374569 |
| sparse-n64-t2n | 367.347816982 | 13.988169073 | 0.057774395 | 0.000000000 | +353.301873514 |

Every event was modeled: 208, 464, 465, and 980 events respectively, with zero
unmodeled events.

## Captured n=64 replay

The deterministic capture is the first surviving cap-saturated
pre-reduction plus direct-sum projection on the `sparse-n64-t2n` query path:
depth 0, outcome 0, 45 cap-saturated input bonds.

| Quantity | Eager | B virtual | B after comparison read |
|---|---:|---:|---:|
| Isolated wall time, s | 3.073647236 | 2.533559711 | 7.742413331 including flush |
| Immediate forced probability | 0.500002921639957409 | 0.500002921639957409 | unchanged by read |
| Pre-reduction SVDs | 61 | 0 | 61-SVD history is not recovered |
| Projection SVDs | 63 | 63 | flush adds long-range frame work |
| Pending frame operations | 0 | 10 | 0 |

The pre-read B replay is `0.540087525 s` faster. A required materializing read
costs `5.208853620 s`, making B plus read `4.668766095 s` slower than eager.

All 61 eager pre-reduction SVDs were `128x128 -> 64` and cap-binding. Eager and
B each performed 63 projection SVDs; their first differing retained ranks were
at projector bonds 55--57: eager `50,28,28`, B `48,24,24`. In the resulting
bond profile (zero-based profile indices), virtual B differs from eager at
indices 54--56 (`50,28,28` versus `48,24,24`). The comparison flush also changes
indices 48--49 from eager/B `64,64` to `60,60`; it does not restore the eager
tail.

State agreement used the allowed n=64 fallback: the existing iterative
amplitude machinery on two complementary deterministic bitstrings. The
squared normalized sample overlap was `0.371292573035989337`. The largest raw
phase-aligned residual was `5.236803306687918e-12`, but these amplitudes are
themselves tiny, so the normalized overlap is the meaningful comparison.

## Shadow ON/OFF A/B

Both arms retained the same eager projection diagnostics. Only the Direction B
shadow was toggled. All warmup and timed hashes agreed within each cell and
between ON/OFF.

| Cell | Shadow ON median query, s | Shadow OFF median query, s | ON - OFF, s | Relative | Hash |
|---|---:|---:|---:|---:|---:|
| sparse-n16-t2n | 0.447477572 | 0.462624376 | -0.015146804 | -3.274% | e672a5b669188f65 |
| sparse-n32-tn | 27.001181366 | 27.515402546 | -0.514221180 | -1.869% | f2856a191d28ffdd |
| sparse-n32-t2n | 184.412190290 | 193.742138935 | -9.329948645 | -4.816% | c94acae6a7e26ee5 |
| sparse-n64-t2n | 1087.714623969 | 1067.678479612 | +20.036144357 | +1.877% | 4b1424a30c76847f |

The directly measured shadow costs (1.37 ms, 7.32 ms, 10.21 ms, and 57.77 ms)
are far smaller than the independent-run ON/OFF spread. The negative apparent
overhead in three cells is therefore timing noise, not a claimed speedup.

## Mutation verification

Each mutation was applied alone, observed to fail, and restored before the
passing runs:

| Killed field or guard | Failure that observed the kill |
|---|---|
| Shadow-event recorder | expected 12 records, got 0 |
| Compensation-CNOT count | queue arithmetic mismatch (`1` versus `0`) |
| Reverse queue conjugation | independent support guard differed (`[1]` versus `[0,1]`) |
| Outcome inverse/gauge append | sibling outcome-update guard differed (`false` versus `true`) |
| Shadow clone-call counter | expected 6 clones, got 0 |
| Sibling-pair ID | clone-derived branch count was 1 instead of 6 |
| Flush-required boolean | `assert!(event.flush_read_required)` failed |
| Flush queue-length counter | expected 1, got 0 |
| Flush chi-cubed units | expected `1.0`, got `0.0` |
| Projection-SVD vector | expected length 2, got 0 |
| Projection-SVD timer | exact depth-bucket timer reproduction failed |
| Frame timer | global positive-frame-time guard failed |
| Shadow-clone timer | global positive-clone-time guard failed |
| Eager-event link | expected linked indices `[0,1]`, got `[]` |
| Projector span | expected 1, got 0 |
| Shadow-OFF selector | shadow-OFF emptiness assertion failed |

## Gates and deviations

- Full debug `cargo test -p pecos-stab-tn --all-targets`: library 354 passed / 8
  declared ignored; exact-default 9 passed / 20 ignored; verification 95 passed
  / 9 ignored; example tests 2 passed.
- Full release `cargo test --release -p pecos-stab-tn --all-targets`: library 355
  passed / 8 ignored; exact-default 26 passed / 20 ignored; verification 95
  passed / 9 ignored; example tests 2 passed.
- Feature-gated n=64 replay passed in release mode.
- `cargo clippy -p pecos-stab-tn --all-targets --features direction-b-phase0-test -- -D warnings` passed.
- `cargo fmt --all -- --check` and `git diff --check` passed.

Deviations and benign observations:

- MAST was skipped for the stated unexercised-lane reason above.
- A dense n=64 oracle is infeasible, so the replay used the review's permitted
  iterative-amplitude fallback.
- An exploratory four-bitstring extension hit an existing iterative-amplitude
  SVD non-convergence after `532.53 s`. The final harness uses the established
  convergent pair of complementary bitstrings; two are sufficient to compare
  relative amplitude, whereas one would make normalized sample overlap
  vacuous.
- Campaign `T_flush_read` is zero because no query-path read required a flush.
  The captured replay nevertheless measures the decisive read-flush cost
  directly.
- Independent ON/OFF median timing is noise-dominated, including negative
  apparent overhead in three cells. The in-shadow timer is the direct
  bookkeeping measurement.

No production Direction B mechanics were added. The only actual B execution is
feature-gated in the test/example harness. No commit was made.
