# PECOS Frontier Decoder

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
