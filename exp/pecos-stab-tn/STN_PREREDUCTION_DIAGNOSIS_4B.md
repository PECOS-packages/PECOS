# STN pre-reduction diagnosis (Stage 4b)

Date: 2026-08-28

Branch: `stn-prereduction-diagnosis-4b` at base `2d24799cc`

Scope: measurement-only telemetry and reporting. No replacement policy, gate
application, truncation, caching, early exit, or other algorithmic behavior was
changed.

## Protocol and definitions

Every campaign cell used CPU 2, one ordinary warmup, and three timed telemetry
runs. Timed tables use the existing harness's median-query run. The existing
`SATURATION_PROJECTION_LOCALITY` opt-in flag gates all new scans and records.
`SATURATION_MAX_BOND_DIM` was added only to the campaign example; it defaults to
the previous hard-coded cap of 64.

Replacement weight is exactly the current rule's
`row_x[id].len() + row_z[id].len()` (so a Y contributes to both stored sparse
rows). For replacement candidate `c`, nominal compensation cost is
`sum_(t != c) (2 * abs(c - t) - 1)` over every stabilizer anticommuting with
`Z_q`. The reported ratio is current cost divided by locality-optimal cost.
This is a count ceiling, not a wall-time claim.

The weight+1 result is reported in two forms to avoid hiding its semantic
choice:

- **weight+1-only** compares the current choice with the best candidate whose
  weight is exactly `minimum + 1`, only on events where such a candidate exists;
- **minimum-or-weight+1** chooses the cheaper candidate at either weight for
  every event. This is the usable ceiling if the maintainer permits the heavier
  choice.

`target_span` excludes the replacement/control site. `support_span` covers the
replacement and every target. SVD shape tables normalize orientation and print
`min(rows, columns) x max(rows, columns)`; raw telemetry retains the ordered
dimensions.

The timed ranges and hashes were:

| Cell | Query range (s) | Pre-reduction range (s) | Selected run | Hash |
|---|---:|---:|---:|---|
| n=16/T=2n, cap 64 | 0.400-0.408 | 0.156-0.159 | 1 | `e672a5b669188f65` |
| n=32/T=n, cap 64 | 23.524-23.620 | 2.368-2.401 | 0 | `f2856a191d28ffdd` |
| n=32/T=2n, cap 64 | 169.200-170.791 | 71.453-72.161 | 0 | `c94acae6a7e26ee5` |
| n=64/T=2n, cap 64 | 985.219-1019.676 | 338.404-349.770 | 2 | `4b1424a30c76847f` |
| n=32/T=2n, cap 128 | 791.949-874.312 | 327.099-364.958 | 2 | `ab51b088d6559d5e` |

All warmup and timed hashes agreed within each cell. All three cap-64 overlap
hashes reproduce Stage 4 exactly, including the specifically required
`f2856a191d28ffdd` baseline.

## Measurement 1: replacement-choice ceilings

### Exact minimum-weight ties

| Cell | Replacement events | Events with >1 min-weight candidate | Current cost | Locality-optimal cost | Ratio | Count saving |
|---|---:|---:|---:|---:|---:|---:|
| n=16/T=2n, cap 64 | 97 | 0 (0.00%) | 1,628 | 1,628 | 1.0000x | 0.00% |
| n=32/T=n, cap 64 | 189 | 16 (8.47%) | 2,330 | 2,298 | 1.0139x | 1.37% |
| n=32/T=2n, cap 64 | 212 | 45 (21.23%) | 11,166 | 11,166 | 1.0000x | 0.00% |
| n=64/T=2n, cap 64 | 473 | 48 (10.15%) | 38,142 | 36,862 | 1.0347x | 3.36% |
| n=32/T=2n, cap 128 | 212 | 45 (21.23%) | 11,166 | 11,166 | 1.0000x | 0.00% |

Ties are absent at n=16, uncommon at n=32/T=n, and not rare in the saturated
n=32 cell. Nevertheless, the mechanism is dead in the saturated n=32 cell:
all 45 tied events have the same aggregate optimum as the iteration-order
choice. Doubling the cap cannot change this tableau-only population and
reproduces the zero ceiling. The n=64 exact-tie ceiling is real but small.

### Weight+1

| Cell | Events with weight+1 candidate | Current cost on that subset | Weight+1-only optimum | Minimum-or-weight+1 total | Ratio | Count saving |
|---|---:|---:|---:|---:|---:|---:|
| n=16/T=2n, cap 64 | 22 | 400 | 304 | 1,532 | 1.0627x | 5.90% |
| n=32/T=n, cap 64 | 67 | 1,290 | 1,404 | 2,220 | 1.0495x | 4.72% |
| n=32/T=2n, cap 64 | 25 | 1,238 | 1,350 | 11,166 | 1.0000x | 0.00% |
| n=64/T=2n, cap 64 | 103 | 4,440 | 5,556 | 36,574 | 1.0429x | 4.11% |
| n=32/T=2n, cap 128 | 25 | 1,238 | 1,350 | 11,166 | 1.0000x | 0.00% |

A mandatory weight+1 policy is worse in every cell except n=16. Selective use
does find small wins at n=32/T=n and n=64, but the saturated n=32 cell again has
zero ceiling. This is not a large latent win for the target cell.

Fanout distributions (`min/q25/median/q75/max`) were:

| Cell | k | Target span | Full support span | Candidate count |
|---|---|---|---|---|
| n=16/T=2n, cap 64 | 1/1/2/2/3 | 0/0/5/7/8 | 2/7/8/8/9 | 2/2/3/3/4 |
| n=32/T=n, cap 64 | 1/1/1/2/3 | 0/0/0/6/25 | 1/1/3/11/26 | 2/2/2/3/4 |
| n=32/T=2n, either cap | 1/1/2/2/9 | 0/0/15/24/30 | 1/15/20/24/30 | 2/2/3/3/10 |
| n=64/T=2n, cap 64 | 1/1/2/2/7 | 0/0/1/22/54 | 1/10/26/34/58 | 2/2/3/3/8 |

The selected n=64 run requested 872 CNOTs, but 29 were structurally identity
and executed no SVD. Its nominal replacement cost (38,142) deliberately still
includes them, matching the requested `sum(2d-1)` definition.

## Measurement 2: n=64/T=2n at cap 64

### Pre-reduction phase attribution

| Cell | Pre-reduction (s) | Query share | SVD compute | QR/gauge | Tensor assembly | Bookkeeping |
|---|---:|---:|---:|---:|---:|---:|
| n=32/T=2n, cap 64 | 71.489 | 42.25% | 61.527 (86.07%) | 2.866 (4.01%) | 7.065 (9.88%) | 0.031 (0.04%) |
| **n=64/T=2n, cap 64** | **341.171** | **34.27%** | **299.283 (87.72%)** | **11.398 (3.34%)** | **30.363 (8.90%)** | **0.127 (0.04%)** |
| n=32/T=2n, cap 128 | 344.026 | 41.18% | 296.036 (86.05%) | 14.266 (4.15%) | 33.553 (9.75%) | 0.171 (0.05%) |

### n=64 depth bands

| Depth | Calls | Pre time (s) | SVD | QR | Tensor | Bookkeeping | Entry max/mean | Entry cap bonds | SVDs (capped) | CNOTs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-7 | 84 | 60.367 | 52.206 | 2.550 | 5.588 | 0.023 | 64/49.64 | 3,626/5,292 | 5,374 (4,755) | 104 |
| 8-15 | 128 | 216.056 | 192.021 | 4.871 | 19.123 | 0.041 | 64/47.24 | 5,616/8,064 | 19,280 (16,132) | 336 |
| 16-23 | 128 | 39.014 | 32.778 | 2.598 | 3.607 | 0.031 | 64/43.06 | 4,812/8,064 | 4,192 (2,964) | 112 |
| 24-31 | 128 | 14.187 | 11.875 | 1.214 | 1.084 | 0.013 | 64/32.21 | 3,356/8,064 | 1,664 (880) | 96 |
| 32-39 | 128 | 5.506 | 5.044 | 0.000 | 0.456 | 0.006 | 64/28.15 | 2,791/8,064 | 704 (363) | 32 |
| 40-47 | 128 | 5.623 | 5.007 | 0.145 | 0.461 | 0.010 | 64/24.04 | 2,073/8,064 | 1,424 (308) | 80 |
| 48-55 | 128 | 0.045 | 0.037 | 0.000 | 0.006 | 0.002 | 64/14.44 | 591/8,064 | 240 (0) | 16 |
| 56-63 | 128 | 0.372 | 0.314 | 0.020 | 0.037 | 0.002 | 64/4.00 | 18/8,064 | 3,945 (16) | 96 |

Depths 8-15 hold 63.33% of the n=64 bucket and 63.47% of its capped SVDs.
Conversely, depths 56-63 execute 3,945 SVDs but take only 0.11% of the bucket
because just 16 bind at the cap and the live ranks are small. This is a sharper
version of the Stage 4 result: raw SVD count is not enough; capped high-rank
SVD count is the cost marker.

Among work-bearing events, correlations with wall time were:

| Cell | Entry cap-bond count | SVD count | Capped-SVD count | CNOT count | Spearman entry-cap |
|---|---:|---:|---:|---:|---:|
| n=32/T=2n, cap 64 | 0.354 | 0.973 | 0.998 | 0.963 | 0.698 |
| **n=64/T=2n, cap 64** | **0.508** | **0.963** | **0.993** | **0.859** | **0.658** |
| n=32/T=2n, cap 128 | 0.375 | 0.960 | 0.998 | 0.961 | 0.634 |

Events entering with any cap bond hold 99.96% of n=64 pre-reduction time.
The n=64 result changes no Stage 4 conclusion: pre-reduction remains a large
saturated-query phase, SVD compute remains its dominant primitive, and capped
SVD count remains the best event-level correlate.

## Measurement 3: n=32/T=2n at cap 128

No second cap was configured by the campaign, so the required fallback was
used: 2x the current cap, 128.

| Cap | Query (s) | Pre (s) | Pre share | SVDs | Capped SVDs | CNOTs | Capped correlation |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 169.225 | 71.489 | 42.25% | 11,166 | 5,045 (45.18%) | 446 | 0.998 |
| 128 | 835.342 | 344.026 | 41.18% | 11,166 | 2,466 (22.09%) | 446 | 0.998 |

The geometry counts are cap-independent, but each surviving high-rank SVD is
far more expensive: the median query and pre-reduction bucket both grow by
about 4.8-4.9x. The internal ranking is unchanged (SVD compute, tensor work,
then QR/gauge), capped-SVD count remains the strongest wall-time correlate,
and both replacement-choice ceilings remain zero.

### Cap-128 depth bands

| Depth | Calls | Pre time (s) | SVD | QR | Tensor | Bookkeeping | Entry max/mean | Entry cap bonds | SVDs (capped) | CNOTs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-7 | 82 | 74.133 | 61.274 | 6.687 | 6.102 | 0.070 | 128/71.62 | 1,207/2,542 | 1,387 (577) | 87 |
| 8-15 | 127 | 260.962 | 227.331 | 7.081 | 26.455 | 0.094 | 128/50.89 | 885/3,937 | 6,755 (1,889) | 247 |
| 16-23 | 128 | 8.927 | 7.428 | 0.497 | 0.995 | 0.006 | 128/15.10 | 48/3,968 | 2,400 (0) | 96 |
| 24-31 | 128 | 0.004 | 0.003 | 0.000 | 0.001 | 0.000 | 16/3.21 | 0/3,968 | 624 (0) | 16 |

Depths 8-15 remain dominant (75.85% of the bucket), so the cap change does not
change the depth mechanism either.

## Measurement 4: fanout, SVD dimensions, and intermediate ranks

Every requested CNOT record contains its control/target/distance, input and
output full bond profiles, within-CNOT maximum retained rank, and the exact
slice of its event's SVD records. Every SVD record contains ordered input
dimensions, retained rank, and cap status. The mutation-verified test reconciles
these records with the existing SVD/CNOT counters and profile continuity.

Dominant SVD shapes in the median runs were:

| Cell | SVDs | Unique normalized shapes | Dominant capped shape | Count | Dominant non-capped shape | Count |
|---|---:|---:|---|---:|---|---:|
| n=32/T=2n, cap 64 | 11,166 | 54 | 128x128 -> rank 64 | 5,036 | 64x128 | 1,756 |
| n=64/T=2n, cap 64 | 36,823 | 326 | 128x128 -> rank 64 | 24,868 | 4x8 | 1,274 |
| n=32/T=2n, cap 128 | 11,166 | 73 | 256x256 -> rank 128 | 2,466 | 128x256 | 2,442 |

At n=32/cap64, 5,036 of 5,045 capped SVDs are the full 2x-cap by
2x-cap matrix. At n=32/cap128, all 2,466 capped SVDs are 256x256. At n=64,
24,868 of 25,418 capped SVDs are 128x128. Thus the cap status is not an
incidental equality: repeated swap-chain SVDs are actively reducing the local
matrix from the 2x-cap input dimension back to the configured retained rank.

For events with at least one compensation CNOT, maximum-rank distributions
(`min/q25/median/q75/max`) were:

| Cell | Events (any capped) | Entry max | Max retained within CNOTs | Exit max | Within peak > exit |
|---|---:|---|---|---|---:|
| n=32/T=2n, cap 64 | 212 (146) | 4/64/64/64/64 | 4/64/64/64/64 | 4/64/64/64/64 | 0 |
| n=64/T=2n, cap 64 | 473 (369) | 2/64/64/64/64 | 2/64/64/64/64 | 2/64/64/64/64 | 3 |
| n=32/T=2n, cap 128 | 212 (105) | 4/64/128/128/128 | 4/128/128/128/128 | 4/64/128/128/128 | 16 |

For multi-CNOT events, the profiles between successive CNOTs were:

| Cell | Multi-CNOT events | Entry/intermediate/exit max median | Entry/intermediate/exit bond-sum median | Intermediate sum > both entry and exit | Intermediate max at cap |
|---|---:|---|---|---:|---:|
| n=32/T=2n, cap 64 | 115 | 64/64/64 | 1,160/1,160/1,160 | 0 | 115 |
| n=64/T=2n, cap 64 | 237 | 64/64/64 | 2,975/2,975/2,975 | 22 | 191 |
| n=32/T=2n, cap 128 | 115 | 128/128/128 | 1,608/1,608/1,608 | 0 | 83 |

The boundary profiles look flat because the truncation is doing its job: the
dominant capped SVD repeatedly maps a 2x-cap matrix back to rank `cap`. This is
load-bearing intermediate compression even when entry and exit profiles match.
The n=64 transient bond-sum cases and cap-128 events whose within-chain peak
exceeds the exit give direct additional evidence of re-compression. Therefore
the Chesterton verdict is **yes**: removing the swap-chain's intermediate SVDs
would remove repeated rank clamping, so SVD-count savings cannot be converted
into a wall-time ceiling without an end-to-end fanout implementation and
trajectory/accuracy test.

## Validation, A/B, and deviations

- `cargo test -p pecos-stab-tn --all-targets`: pass (343 library, 9
  exact-default, 95 verification, 2 example tests; repository-declared ignored
  lanes remained ignored).
- `cargo test --release -p pecos-stab-tn --all-targets`: pass (344 library, 26
  exact-default, 95 verification, 2 example tests; declared ignored lanes
  remained ignored).
- `cargo clippy -p pecos-stab-tn --all-targets -- -D warnings`: pass.
- `cargo fmt --all -- --check`: pass.
- `git diff --check`: pass.
- Mutation verification: deleting the per-SVD diagnostic record failed the
  focused randomized guard immediately (`0` records versus `1` existing SVD
  count). Restoring it passed, and the full suites then passed.
- Disabled-path ABBA on CPU 2, 10 samples per binary: base query median
  0.406807 s (range 0.404842-0.411655), modified median 0.406080 s (range
  0.400352-0.408270), -0.18%. Simulation medians were 0.112884 and 0.111755 s.
  All hashes were `e672a5b669188f65`. This is no regression above noise, not a
  speedup claim.
- The requested branch already existed, was checked out, clean, and pointed
  exactly at `stn-prereduction-diagnosis` (`2d24799cc`). It was used as-is
  rather than deleting and recreating a user-visible branch reference.
- The campaign had no second configured cap, so cap 128 was used as the
  required 2x fallback.
- The requested n=64 expectation of about 11 minutes per run was optimistic on
  this machine. Warmup was 968.3 s and timed runs were 985.2-1019.7 s; these
  agree with the earlier Stage 1 ~972 s baseline. The protocol was not reduced.
- Cap 128 changes the truncation trajectory and therefore its hash; the stable
  `ab51b088d6559d5e` hash is not expected to match cap 64.
- Raw logs were retained under `/tmp/stn-4b-*.log`; the repository contains the
  reproducible telemetry and this condensed report, not 40 MB of generated log
  output.
- No commit was created.

## Required final verdicts

- **Exact-tie replacement ceiling:** 0.00% in n=16/T=2n, 1.37% in n=32/T=n,
  and 0.00% in the saturated n=32/T=2n target (ratios 1.0000x, 1.0139x,
  1.0000x). It is not a useful target-cell mechanism.
- **Weight+1 ceiling:** selective minimum-or-weight+1 gives 5.90%, 4.72%, and
  0.00% respectively (1.0627x, 1.0495x, 1.0000x). It is also dead in the
  saturated target cell. n=64 ceilings are 3.36% exact-tie and 4.11% with
  weight+1, too small to lead the work.
- **n=64 conclusion:** unchanged and strengthened. Pre-reduction is 34.27% of
  query time, 87.72% of that bucket is SVD compute, and capped-SVD count remains
  the strongest correlate (0.993).
- **Second-cap ranking:** unchanged. SVD compute remains first, then tensor,
  then QR/gauge; capped-SVD count remains the strongest correlate (0.998), and
  replacement-choice ceilings remain zero.
- **Chesterton verdict:** intermediate swap-chain compression is load-bearing.
  The flat retained profiles are produced by thousands of cap-binding SVDs
  repeatedly reducing 2x-cap inputs back to the cap. A direct/fanout design must
  measure its new rank and truncation trajectory end to end; count arithmetic
  alone is not a safe wall-time ceiling.
