# PECOS Trellis Decoder

Trellis dynamic-programming engine for logical coset-mass decoding.
The `frontier` and `bp_trellis` modules provide the Frontier parity identity
and PECOS's BP-guided preset, respectively.

## PECOS Frontier Decoder

Native Rust implementation of the Frontier approximate logical maximum-likelihood
decoder (Leverrier & Urbanke, arXiv:2606.20513). Not a wrap of the upstream
`frontier` package; the upstream implementation is used as a verification oracle.

**Experimental** (`exp/`): the algorithm core is enumeration- and upstream-verified
(per-shot parity on matched models), but the crate has not yet accumulated real-user
mileage. Python exposes `pecos_rslib_exp.frontier()` for parallel `SampleBatch.decode`
and `DemSampler.decode` execution. The experimental extension is optional;
standard `pecos-rslib` and `pecos-decoders` do not depend on this crate. The
detailed per-shot API remains available as `pecos_rslib_exp.FrontierDecoder`.
Graduation to `crates/` awaits broader real-world use.

Pruning ranks accumulated prefix log mass plus a `score_alpha`-weighted
suffix-compatibility estimate. Unpruned results are exact and upstream-verified.

Deterministic ordering and tie-breaking are bitwise reproducible for a fixed
build and platform. The platform's `ln` and `exp` implementations may differ
across platforms.

## PECOS BP Trellis Decoder

PECOS's degeneracy-aware BP-guided trellis decoder is an approximate logical
maximum-likelihood decoder that is exact in the unpruned limit. Its optimality
is relative to the supplied detector error model, not the underlying physics,
and pruned results have no certified bound on discarded posterior mass. Belief
propagation guides only which states pruning retains; it does not change branch
probabilities or mass arithmetic. This is not a wrap or port of an external
project.

**Experimental** (`exp/`): the defaults and optional no-path escalation ladder
remain provisional pending broader validation. The shared trellis engine lives
in `pecos-trellis`; this crate contains PECOS's configuration and decoder
facade.

The optional `pecos-rslib-exp` package provides `pecos_rslib_exp.bp_trellis(...)`
for `SampleBatch.decode` and `DemSampler.decode`, including parallel Rust workers.
All seven configuration options are exposed. Standard `pecos-rslib` and
`pecos-decoders` do not depend on this crate. The direct
`pecos_rslib_exp.BpTrellisDecoder` API additionally returns detailed per-shot
confidence and retry telemetry.
