# STN pre-reduction diagnosis (Stage 4)

Date: 2026-08-28

Branch: `stn-prereduction-diagnosis` at base `4725901e2`

Scope: diagnosis and runtime-gated telemetry only. No optimization, caching,
early exit, or algorithm change is included.

## Protocol and traced path

The affected query path is:

`prob_bitstrings` prefix trie -> `PrefixProjectionState::project_z` ->
`project_forced_z_with_update_impl` -> compensated
`pre_reduce_for_measurement` -> zero or more virtual CNOTs ->
`Mps::apply_long_range_two_site_gate` -> QR/LQ gauge setup plus
SWAP-in/gate/SWAP-out two-site contractions and truncating SVDs.

Each nontrivial compensated pre-reduction snapshots the tableau and full MPS
for transactional rollback. The new diagnostic timers measure SVD compute,
QR/gauge work, and tensor contraction/assembly in the MPS owner. The owning
phase's uninstrumented residual is bookkeeping (rollback snapshots,
structural tests, tableau composition, modified-site vectors, and timer
overhead).

Measurements used CPU 2, one ordinary warmup, and three diagnostic repetitions.
The reported run is the existing harness's median-query run. The complete
timed ranges were:

| Cell | Query range (s) | Pre-reduction range (s) | Reported run/output hash |
|---|---:|---:|---|
| sparse n=16, T=2n | 0.428-0.433 | 0.167-0.169 | run 0 / `e672a5b669188f65` |
| sparse n=32, T=n | 24.470-25.334 | 2.467-2.552 | run 0 / `f2856a191d28ffdd` |
| sparse n=32, T=2n | 169.464-173.968 | 71.440-73.509 | run 1 / `c94acae6a7e26ee5` |

The required n=32/T=n baseline hash reproduced exactly in both warmup and
diagnostic runs.

## 1. What the bucket does

| Cell | Pre-reduction (s) | Query share | SVD compute | QR/gauge | Tensor assembly | Bookkeeping |
|---|---:|---:|---:|---:|---:|---:|
| n=16, T=2n | 0.167 | 39.01% | 0.142 (84.88%) | 0.003 (2.02%) | 0.021 (12.63%) | 0.0008 (0.46%) |
| n=32, T=n | 2.509 | 10.07% | 1.723 (68.68%) | 0.526 (20.95%) | 0.256 (10.18%) | 0.0045 (0.18%) |
| n=32, T=2n | 73.012 | 42.27% | 62.800 (86.01%) | 2.914 (3.99%) | 7.262 (9.95%) | 0.0349 (0.05%) |

SVD compute dominates every cell. It is especially dominant in the saturated
n=32/T=2n cell; bookkeeping is not the cause.

| Cell/depth band | Calls | Pre time (s; bucket share) | SVD | QR | Tensor | Bookkeeping |
|---|---:|---:|---:|---:|---:|---:|
| n16/T=2n, 0-7 | 80 | 0.144 (86.45%) | 85.51% | 2.08% | 12.18% | 0.23% |
| n16/T=2n, 8-15 | 128 | 0.023 (13.55%) | 80.85% | 1.66% | 15.55% | 1.94% |
| n32/T=n, 0-7 | 81 | 0.945 (37.66%) | 57.21% | 35.09% | 7.51% | 0.20% |
| n32/T=n, 8-15 | 127 | 1.504 (59.93%) | 75.81% | 12.22% | 11.86% | 0.11% |
| n32/T=n, 16-23 | 128 | 0.059 (2.34%) | 70.94% | 17.77% | 10.36% | 0.93% |
| n32/T=n, 24-31 | 128 | 0.002 (0.07%) | 59.68% | 5.25% | 13.98% | 21.10% |
| n32/T=2n, 0-7 | 82 | 11.301 (15.48%) | 81.67% | 9.43% | 8.82% | 0.08% |
| n32/T=2n, 8-15 | 127 | 55.579 (76.12%) | 87.28% | 2.47% | 10.22% | 0.04% |
| n32/T=2n, 16-23 | 128 | 6.128 (8.39%) | 82.59% | 7.79% | 9.55% | 0.06% |
| n32/T=2n, 24-31 | 128 | 0.003 (0.005%) | 75.54% | 0.71% | 17.62% | 6.14% |

## 2. Why it is depth-concentrated

| Cell/depth band | Entry max/mean bond | Entry bonds at cap | SVDs (capped) | Compensating CNOTs |
|---|---:|---:|---:|---:|
| n16/T=2n, 0-7 | 64 / 7.86 | 4/1,200 (0.33%) | 684 (0) | 66 |
| n16/T=2n, 8-15 | 16 / 3.04 | 0/1,920 | 944 (0) | 112 |
| n32/T=n, 0-7 | 64 / 22.88 | 546/2,511 (21.74%) | 684 (16) | 52 |
| n32/T=n, 8-15 | 64 / 14.34 | 198/3,937 (5.03%) | 1,006 (30) | 78 |
| n32/T=n, 16-23 | 16 / 6.93 | 0/3,968 | 336 (0) | 80 |
| n32/T=n, 24-31 | 4 / 2.08 | 0/3,968 | 304 (0) | 48 |
| n32/T=2n, 0-7 | 64 / 41.23 | 1,439/2,542 (56.61%) | 1,387 (739) | 87 |
| n32/T=2n, 8-15 | 64 / 36.50 | 1,808/3,937 (45.92%) | 6,755 (4,105) | 247 |
| n32/T=2n, 16-23 | 64 / 14.17 | 272/3,968 (6.85%) | 2,400 (201) | 96 |
| n32/T=2n, 24-31 | 16 / 3.00 | 0/3,968 | 624 (0) | 16 |

The entry-cap candidate is a strong regime marker but is refuted as the
sufficient explanation for the depth concentration:

- n=32/T=2n depth 0-7 has more cap-saturated entry bonds than depth 8-15
  (56.61% versus 45.92%) but only 15.48% of bucket time versus 76.12%.
- Depth 8-15 instead executes 6,755 SVDs, including 4,105 capped SVDs, from
  247 compensating CNOTs. Depth 0-7 executes 1,387/739 from 87 CNOTs.
- Among work-bearing n=32/T=2n calls, Pearson correlation with wall time is
  0.354 for entry cap-bond count, 0.973 for SVD count, 0.999 for capped-SVD
  count, and 0.963 for compensating-CNOT count.
- At cell level, calls with any cap-saturated entry bond hold 99.85% of
  n=32/T=2n pre-reduction time and 96.62% of n=32/T=n time, but only 0.0002%
  at n=16/T=2n. Spearman correlation between cap-bond count and call time is
  0.676, 0.366, and -0.050 respectively.

Verdict: saturation makes the numerical work expensive, but the mid-depth
explosion is the product of how many long-range compensated CNOT/SVD steps are
requested and how many of those SVDs bind at the cap. Entry cap count alone
does not predict it.

## 3. Redundancy across trie siblings

The cheap fingerprint is `(chain length, full internal-bond profile,
accumulated projector count)`. It is intentionally not a tensor/tableau hash.

| Cell | Same-depth repeated fingerprints | Structural-only ceiling | Exact sibling calls | Matching sibling fingerprints | Exact sibling-share ceiling |
|---|---:|---:|---:|---:|---:|
| n=16, T=2n | 187/208 (89.90%) | 86.93% | 30 (15 pairs) | 30/30 | 26.56% |
| n=32, T=n | 388/464 (83.62%) | 82.08% | 30 (15 pairs) | 30/30 | 12.89% |
| n=32, T=2n | 365/465 (78.49%) | 91.62% | 30 (15 pairs) | 30/30 | 4.45% |

The broad repeated-fingerprint ceiling is not a caching ceiling: distinct trie
states frequently collide because only their bond shapes are fingerprinted.
Actual binary siblings are identified separately by a trie-parent pair ID.
All 15 actual pairs match, as expected for two outcome projections from one
parent, but they cover only 4.45% of the saturated bucket by cost. Exact
sibling sharing is real but too small to explain or remove the saturated gap.

## 4. Bond-profile no-ops and guard cost

| Cell | Output profile unchanged | Its bucket time | Zero-compensation calls | Their bucket time | Entry profile scan/call |
|---|---:|---:|---:|---:|---:|
| n=16, T=2n | 171/208 (82.21%) | 24.51% | 111/208 | 0.00212% | 0.147 us |
| n=32, T=n | 452/464 (97.41%) | 72.54% | 275/464 | 0.00058% | 0.437 us |
| n=32, T=2n | 447/465 (96.13%) | 99.12% | 253/465 | 0.000022% | 0.615 us |

The rank-profile definition labels a material fraction as no-op, but it does
not support an early exit. In the saturated cell, 194 unchanged-profile calls
still execute 421 required compensating CNOTs and 10,455 SVDs (5,039 capped),
consuming 72.372 s. The phase changes the generator basis and coefficient MPS
even when every resulting bond rank is unchanged. Skipping it based on the
input profile would change the represented state.

The genuinely empty calls (`compensating_cnot_count == 0`) already take the
existing immediate return and collectively consume only 0.000022% of the
saturated bucket. The existing per-CNOT structural identity/X guards fired
zero times in all three cells. Scanning an entry profile is cheap, but it cannot
prove that a required virtual CNOT may be skipped. Safe early-exit ceiling: effectively zero.

## Validation and deviations

- `cargo test -p pecos-stab-tn --all-targets`: pass (343 library, 9
  exact-default, 95 verification, 2 example tests; repository-declared ignored
  lanes remained ignored).
- `cargo test --release -p pecos-stab-tn --all-targets`: pass (344 library, 26
  exact-default, 95 verification, 2 example tests; declared ignored lanes
  remained ignored).
- `cargo clippy -p pecos-stab-tn --all-targets -- -D warnings`: pass.
- `cargo fmt --all -- --check`: pass.
- `git diff --check`: pass.
- Runtime-gate mutation check: forcing diagnostics through the disabled path
  failed the focused randomized query test immediately; restoring the guard
  passed it.
- Disabled-path ABBA on CPU 2, 10 samples per binary: preserved base median
  0.42034 s, modified median 0.40489 s (-3.68%). This is reported only as no
  regression above run/order noise, not as a speedup. All hashes matched.
- The requested branch already existed and was checked out at the exact
  `stn-projection-qr-locality` head, so no second branch could be created.
- n=64 was not run: the required n=32/T=2n warmup plus three repetitions took
  about eleven minutes and already supplied the saturation contrast; n=64 was
  therefore outside the optional time budget.
- No commit was created.

## Ranked measured mechanisms

Ceilings below are for the key n=32/T=2n pre-reduction bucket and overlap; they
must not be added.

1. **Cap-bearing SVD compute:** 85.32% ceiling (62.291/73.012 s). All SVD
   compute has an 86.01% ceiling. This is the only mechanism large enough to
   address the bucket.
2. **Tensor contraction/assembly:** 9.95% ceiling (7.262/73.012 s).
3. **Exact trie-sibling sharing:** 4.45% ceiling (3.249/73.012 s), using the
   cheaper measured member of each identical sibling pair as avoidable time.
4. **QR/gauge work:** 3.99% ceiling (2.914/73.012 s).
5. **Safe early exit / existing structural skip:** 0.000022% for already-empty
   calls; zero observed CNOT-level skips. The apparent 99.12% unchanged-profile
   ceiling is not semantically safe.

This is not a null result: the data localizes the dominant cost to cap-bearing
SVD compute. It is a null result for broad trie caching and bond-profile-based
early exit as explanations or high-ceiling remedies.
