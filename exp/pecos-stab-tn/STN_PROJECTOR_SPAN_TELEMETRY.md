# STN projector-span telemetry (Stage 5a)

Date: 2026-08-29

Branch: `stn-projector-span-telemetry` at base `096fde0a5`
(`stn-prereduction-diagnosis-4b`)

Scope: measurement-only telemetry, a campaign/reporting extension, and a
test-only construction. No production projection, canonicalization,
compression, cutoff, or trajectory behavior was changed.

## Protocol and definitions

The three required sparse cells and optional n=64/T=2n cell used CPU 2, one
ordinary warmup, and three timed telemetry runs. Tables use the harness's
median-query run. All event populations and distributions were identical in
the three timed repetitions of a cell. The existing
`SATURATION_PROJECTION_LOCALITY` opt-in flag gates the new scans and
compression records.

The measurement point is after pre-reduction. Immediately before the Pauli is
applied, telemetry records the actual `flip_sites U sign_sites`; the reported
span is `s_max - s_min`. Compensation/gauge sites are a separate field and
never enter projector span. Scalar-scale and one-site local-block writes bypass
`Mps::add`; their counts are reported in the mix but only direct-sum events
enter span, joint, walk-model, and external-compaction denominators.

The entry bond profile and the first center snapshot are taken after
pre-reduction and before any projection positioning. A second center snapshot
is taken immediately before the projection write. Bond `b` is between sites
`b-1` and `b`; a bond is internal to `[s_min,s_max]` exactly when
`s_min < b <= s_max`. All other internal bonds are external.

The end-to-end trace put each observation at its owning layer: the projection
path records the post-reduction Pauli, centers, entry ranks, and gauge sites;
the MPS compressor records the actual SVD input/retained rank and discarded
weight; the projection layer alone classifies those bond decisions against its
support. Tests reconcile the two layers but are not used as the runtime
boundary.

Quintiles below are `min/q25/median/q75/max`, using indices
`0,(N-1)/4,(N-1)/2,3(N-1)/4,N-1` after sorting. The selected timing runs and
hashes were:

| Cell | Query range (s) | Selected run | Direct/scalar/local | Hash |
|---|---:|---:|---:|---|
| n=16/T=2n | 0.429294-0.431746 | 2 | 154/0/54 | `e672a5b669188f65` |
| n=32/T=n | 25.284842-25.635199 | 0 | 372/0/92 | `f2856a191d28ffdd` |
| n=32/T=2n | 171.769699-175.921140 | 0 | 465/0/0 | `c94acae6a7e26ee5` |
| n=64/T=2n | 1043.285294-1180.120122 | 2 | 884/0/96 | `4b1424a30c76847f` |

All warmup and timed hashes agreed within each cell. In particular, the
required n=32/T=n baseline reproduced `f2856a191d28ffdd` exactly.

## Measurement 1: actual post-pre-reduction projector population

### Projector, margins, centers, and entry ranks

| Cell | s_min | s_max | Span | Left margin | Right margin | Center before positioning | Center before write | Entry bond rank |
|---|---|---|---|---|---|---|---|---|
| n=16/T=2n | 0/2/4/6/13 | 3/10/11/13/15 | 0/2/6/13/15 | 0/2/4/6/13 | 0/2/4/5/12 | 10/13/15/15/15 | 10/13/15/15/15 | 1/2/4/8/64 |
| n=32/T=n | 0/7/9/16/29 | 3/15/25/30/31 | 0/2/10/22/30 | 0/7/9/16/29 | 0/1/6/16/28 | 9/27/31/31/31 | 9/27/31/31/31 | 1/2/4/16/64 |
| n=32/T=2n | 0/1/2/9/28 | 6/14/21/29/31 | 0/5/14/24/30 | 0/1/2/9/28 | 0/2/10/17/25 | 4/29/31/31/31 | 4/29/31/31/31 | 1/2/8/32/64 |
| n=64/T=2n | 1/4/7/18/45 | 4/43/51/62/63 | 0/21/37/53/61 | 1/4/7/18/45 | 0/1/12/20/59 | 20/53/63/63/63 | 20/53/63/63/63 | 1/2/16/64/64 |

The two center distributions are identical event-by-event, and all recorded
center claims validate. Pre-reduction, not projection construction, is what
has already moved some centers away from `n-1`. Relative to the projector
window, the centers were left/inside/right in 0/68/86 events at n=16/T=2n,
0/84/288 at n=32/T=n, 0/125/340 at n=32/T=2n, and 0/393/491 at
n=64/T=2n.

### Gauge compensation (kept separate)

| Cell | Events with gauge | Site count | Minimum site | Maximum site | Gauge span |
|---|---:|---|---|---|---|
| n=16/T=2n | 58/154 | 0/0/0/1/5 | 0/2/10/15/15 | 3/10/11/15/15 | 0/0/0/0/9 |
| n=32/T=n | 46/372 | 0/0/0/0/2 | 1/1/1/17/17 | 1/1/6/17/17 | 0/0/0/5/5 |
| n=32/T=2n | 260/465 | 0/0/1/2/5 | 0/1/6/12/25 | 0/6/13/18/30 | 0/0/4/12/24 |
| n=64/T=2n | 418/884 | 0/0/0/1/9 | 1/10/21/35/57 | 2/23/46/60/63 | 0/0/0/36/61 |

The min/max/span gauge quintiles exclude events with no gauge sites; the count
quintiles include all direct-sum events.

### Joint distribution of `(center before positioning, s_min, s_max)`

Counts are exact for the selected run. This is the load-bearing margin
structure; it is not reconstructed from independent quintiles.

| Cell | `(center,s_min,s_max): count` |
|---|---|
| n=16/T=2n | `(10,8,10):16`; `(12,4,15):12`; `(13,13,13):16`; `(15,0,3):2`; `(15,0,8):4`; `(15,0,13):16`; `(15,0,15):15`; `(15,2,11):16`; `(15,2,15):9`; `(15,4,8):16`; `(15,4,10):16`; `(15,6,6):16` |
| n=32/T=n | `(9,0,23):2`; `(12,1,26):14`; `(15,1,21):16`; `(25,15,25):16`; `(26,9,12):15`; `(26,11,13):16`; `(27,0,30):16`; `(28,2,30):4`; `(31,1,6):16`; `(31,3,3):7`; `(31,5,15):13`; `(31,7,15):16`; `(31,7,30):29`; `(31,8,30):32`; `(31,8,31):16`; `(31,10,10):16`; `(31,12,26):16`; `(31,13,25):16`; `(31,16,19):16`; `(31,17,17):16`; `(31,20,20):16`; `(31,21,30):16`; `(31,23,23):16`; `(31,29,29):16` |
| n=32/T=2n | `(4,4,10):12`; `(7,3,30):15`; `(21,1,30):16`; `(21,2,26):16`; `(21,5,26):13`; `(22,18,22):4`; `(29,0,9):16`; `(29,2,31):15`; `(29,9,29):16`; `(30,4,30):2`; `(31,0,14):16`; `(31,0,15):16`; `(31,0,29):16`; `(31,1,6):14`; `(31,1,13):32`; `(31,1,15):32`; `(31,1,21):16`; `(31,1,22):16`; `(31,1,31):16`; `(31,3,28):16`; `(31,6,30):7`; `(31,8,8):15`; `(31,8,19):16`; `(31,10,10):16`; `(31,19,19):16`; `(31,20,20):16`; `(31,23,23):16`; `(31,24,30):16`; `(31,26,26):16`; `(31,28,28):16` |
| n=64/T=2n | `(20,5,63):16`; `(21,21,21):12`; `(29,1,56):16`; `(30,7,54):16`; `(32,5,49):7`; `(40,3,59):16`; `(40,15,59):2`; `(41,35,63):16`; `(43,7,43):16`; `(46,15,63):16`; `(46,16,46):32`; `(51,7,63):16`; `(51,10,61):16`; `(51,18,51):16`; `(52,2,53):4`; `(53,2,58):16`; `(53,14,44):16`; `(57,4,50):32`; `(57,10,63):16`; `(58,6,24):16`; `(59,32,59):16`; `(60,3,60):16`; `(61,3,63):16`; `(61,21,61):16`; `(62,1,62):16`; `(63,1,62):48`; `(63,2,50):16`; `(63,3,40):16`; `(63,4,4):11`; `(63,4,57):32`; `(63,5,33):16`; `(63,6,6):16`; `(63,6,24):16`; `(63,7,28):32`; `(63,7,43):16`; `(63,7,51):16`; `(63,7,53):16`; `(63,9,9):16`; `(63,11,11):16`; `(63,11,34):16`; `(63,15,19):16`; `(63,16,63):16`; `(63,21,63):32`; `(63,23,44):16`; `(63,23,46):16`; `(63,31,43):16`; `(63,31,55):16`; `(63,36,36):16`; `(63,41,63):16`; `(63,44,46):16`; `(63,45,45):16` |

The corrected population does not support the v1 headroom number. Median
projector spans are 6, 10, 14, and 37 sites, but wide tails reach 15, 30, 30,
and 61; more importantly, the right-heavy joint centers and heterogeneous
entry ranks make span alone an inadequate walk-cost proxy.

## Measurement 2: bond-weighted walk model

These are **model numbers, not measured alternative implementations**. Each
walk across bond `b` costs `chi_b^3`, using that event's recorded entry rank.
For every direct-sum event:

- (a) is one current full-chain reversal, `sum_b chi_b^3`;
- (b) walks from the recorded center to the cheaper span edge by weighted
  cost, traverses the entire span, and stops at the opposite edge;
- (c) is (b), then walks from that terminal edge to site 0.

Each event's measured current post-projection QR wall time calibrates its
`chi^3` unit. The estimates are summed and divided by the full measured
`PostProjectionQr` bucket, which also contains the separately counted scalar
and local-block event time.

| Cell | QR bucket (s) | (a) current | (b) bounded/retain external rank | (c) bounded + mandatory 0 | c/a raw units |
|---|---:|---:|---:|---:|---:|
| n=16/T=2n | 0.103763 | 0.9363 | 0.9183 | 0.9373 | 1.0013 |
| n=32/T=n | 7.840242 | 0.9987 | 1.0084 | 1.1722 | 1.1711 |
| n=32/T=2n | 36.487586 | 1.0000 | 0.9584 | 1.0672 | 1.0658 |
| n=64/T=2n | 228.599490 | 0.9852 | 0.9149 | 1.0778 | 1.0933 |

Thus the bounded external-rank-retaining model saves only about 1.8%, 4.2%,
and 8.5% of the full QR bucket in the n=16/T=2n, n=32/T=2n, and n=64/T=2n
cells, and is 0.8% worse at n=32/T=n. This is substantially below the invalid
v1 site-count estimate.

The requested (c) `n-1` sanity identity is **not confirmed by the data**. Its
derivation, `(n-1-s_max) + span + s_min = n-1`, assumes the initial center is
`n-1` and starts at the right edge. Actual post-pre-reduction centers are often
inside the span and occasionally well left of `n-1`; choosing a nearest edge
can also make (c) revisit bonds. With bond weights this makes c/a 1.001-1.171,
not 1. This is a correction to the review's assumed event geometry, not a
measured implementation speedup or slowdown.

## Measurement 3: external-spectrum changes

Post-compression retained rank is compared with the pre-projection entry rank
on every external bond. Discarded weight is the compressor's existing relative
singular-value weight, summed only over those external bonds.

| Cell | Events with external rank change | Changed external bonds | Direction | Events with external discard | Summed external discarded weight |
|---|---:|---:|---|---:|---:|
| n=16/T=2n | 50/154 (32.47%) | 248/1,254 (19.78%) | 248 down, 0 up | 123/154 (79.87%) | 1.3731e-24 |
| n=32/T=n | 111/372 (29.84%) | 876/7,302 (12.00%) | 876 down, 0 up | 324/372 (87.10%) | 1.9603e-23 |
| n=32/T=2n | 143/465 (30.75%) | 884/8,044 (10.99%) | 884 down, 0 up | 431/465 (92.69%) | 5.4403e-23 |
| n=64/T=2n | 223/884 (25.23%) | 1,262/25,140 (5.02%) | 1,241 down, 21 up | 752/884 (85.07%) | 6.1548e-22 |

The retained-rank reductions total 608, 4,492, 7,335, and 10,065 dimensions;
per reduced-bond quintiles are respectively `1/1/2/2/32`, `1/1/4/4/32`,
`1/2/4/9/32`, and `1/2/3/10/41`.

Verdict: external compaction is not rare. It reduces at least one external
rank in 30-32% of the required-cell direct-sum events and 222/884 (25.11%) at
n=64. The discarded weights are numerically tiny, so retaining rank is cheap
in fidelity but not free in representation size. A windowed compressor with
external-rank retention would forego frequent real compaction; periodic
full-chain compaction or a proven rank-deficiency trigger remains necessary.

The optional n=64 cell also exposes a finite-precision caveat hidden by the
three smaller cells: five events retained a larger dimension on 21 external
bonds (205 added dimensions, `1/3/10/13/26` per-bond quintiles). These are not
a counterexample to exact Schmidt-rank monotonicity. They are the production
compressor's numerical retained dimensions after a cancellation-prone global
direct sum, under the configured cutoff and zero adaptive error budget. The
measurement therefore must not encode “never increases” as a representation-
level invariant; the observed up/down decisions are reported separately.

## Measurement 4: reproducible test-only windowed add

`mps::tests::test_only_windowed_add_matches_dense_pauli_sum_and_preserves_exterior`
runs 32 deterministic randomized six-site states/projectors across interior,
boundary-touching, full-chain, and one-site windows, with both `psi + P psi`
and `psi - P psi` branches. The helper is private to `#[cfg(test)]`, is marked
TEST ONLY, and deliberately selects no production compression policy.

The tests prove:

- every amplitude agrees with an independent dense Pauli oracle to `5e-12`;
- every exterior tensor is bit-identical and every exterior bond dimension is
  unchanged; only strictly internal window bonds double;
- the left edge is the exact horizontal block join and the right edge is the
  exact vertical block join, including chain-edge and one-site cases;
- all 20 multi-site cases invalidate the prior center claim, and none admits
  any valid one-site mixed-canonical center before repair; all 12 one-site
  cases retain a valid center at that site.

The edge-isometry result is therefore negative for raw multi-site windowed
addition: exact amplitudes and unchanged exterior tensors do **not** preserve a
global canonical-center claim. The joins have the expected algebraic block
form, but a canonicalization/claim-repair step is required. The tests preserve
no stale claim: multi-site results explicitly set `center=None`.

## Validation, A/B, mutation, and deviations

- `cargo test -p pecos-stab-tn --all-targets`: pass (344 passed/8 ignored in
  the library lane, 9 passed/20 ignored exact-default, 95 passed/9 ignored
  verification, 2 example tests).
- `cargo test --release -p pecos-stab-tn --all-targets`: pass in a clean
  isolated target (345/8 library, 26/20 exact-default, 95/9 verification, 2
  example tests).
- `cargo clippy -p pecos-stab-tn --all-targets -- -D warnings`: pass.
- `cargo fmt --all -- --check`: pass.
- `git diff --check`: pass.
- Mutation verification: four independent kills all failed the focused
  randomized guard: disabling the per-bond record failed the every-internal-
  bond assertion; zeroing the compression count failed `0 != n-1`; zeroing the
  external discarded-weight accumulator failed its bit-exact reconciliation;
  and zeroing per-event QR time failed exact reconstruction of the depth QR
  bucket. Each mutation was restored and the focused guard passed before the
  next kill. The full suites had run on the byte-identical final source before
  this later mutation audit; fmt, clippy, and diff checks passed again after
  restoration.
- Disabled-path ABBA on CPU 2, 10 samples per binary: base query median
  0.386706 s (range 0.383418-0.393128), modified median 0.386117 s (range
  0.381798-0.389629), -0.15%. Simulation medians were 0.107763 and 0.107177 s
  (-0.54%). All hashes were `e672a5b669188f65`. This is no regression above
  noise, not a speedup claim.
- The requested branch already existed, was clean, and pointed exactly at
  `stn-prereduction-diagnosis-4b` (`096fde0a5`); it was used as-is rather than
  deleting and recreating a user-visible reference.
- The design was not present at the stated in-repository `vault` path. The
  complete file, including Review outcome, was read from the sibling docs
  checkout at `/home/ciaranra/Repos/pecos-docs/design/stab-tn/projector-representation-v1.md`.
- Git worktree creation for the A/B baseline was unavailable because the
  shared Git metadata is read-only in this environment. An exact `git archive`
  of `stn-prereduction-diagnosis-4b` was built instead.
- The first release gate reused A/B target metadata and failed at compile time
  by pairing the archived base library with the modified example. Re-running
  from a fresh isolated `CARGO_TARGET_DIR` passed the entire release suite; this
  was build isolation contamination, not a source/test failure.
- The optional n=64 timed run 1 overlapped with several short debug mutation-
  test builds before the overlap was noticed; its 1180.120 s query was the
  slowest run. Runs 0 and 2 were kept free of build/test activity, and clean run
  2 was the 1079.586 s median selected for all reported n=64 model fractions.
  Geometry, rank decisions, discarded weights, and hashes were identical in
  all three repetitions.
- The first optional n=64 launch referenced a nonexistent isolated example
  binary and exited immediately, before warmup or measurement. Re-launching
  the existing modified binary completed the full 1-warmup/3-timed protocol
  reported here.
- Raw campaign and first mutation-kill logs are retained under
  `/tmp/stn-5a-*.log`; the later mutation failures were captured in the task
  transcript. The repository contains the reproducible harness/tests and this
  condensed report.
- No commit was created.

## Required final verdicts

- **Corrected span/margin distribution:** direct-sum median projector spans
  are 6, 10, 14, and 37, with full ranges 0-15, 0-30, 0-30, and 0-61. Exact
  joint center/support counts above show centers frequently inside the span
  and otherwise mostly to its right; independent span statistics cannot price
  the weighted walk.
- **Model fractions (a)/(b)/(c):** 0.9363/0.9183/0.9373,
  0.9987/1.0084/1.1722, 1.0000/0.9584/1.0672, and
  0.9852/0.9149/1.0778 of measured `PostProjectionQr`. These are calibrated
  `chi^3` model fractions, not timed alternative algorithms. Mandatory-to-zero
  does not satisfy the review's identity under the measured starting centers.
- **External-compaction frequency:** 25-32% of direct-sum events compact at
  least one external bond. Rank retention is fidelity-cheap at the observed
  discarded weights but gives up frequent representation compaction. At n=64,
  finite-precision retained dimensions also rise on 21 bonds, so exact rank
  monotonicity must not be mistaken for a numerical representation invariant.
- **Windowed-add evidence:** the construction is amplitude-exact, preserves
  exterior tensors and bond dimensions bit-for-bit, and has the asserted edge
  block joins. Raw multi-site joins preserve no valid one-site canonical
  center; the implementation must invalidate or repair the claim. One-site
  windows retain a valid center.
