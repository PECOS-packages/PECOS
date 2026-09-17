# PECOS BP Trellis Decoder

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
