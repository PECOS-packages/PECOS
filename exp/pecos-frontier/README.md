# PECOS Frontier Decoder

Native Rust implementation of the Frontier approximate logical maximum-likelihood
decoder (Leverrier & Urbanke, arXiv:2606.20513). Not a wrap of the upstream
`frontier` package; the upstream implementation is used as a verification oracle.

**Experimental** (`exp/`): the algorithm core is enumeration- and upstream-verified
(per-shot parity on matched models), but the crate has not yet accumulated real-user
mileage. Graduation to `crates/` and registration in the `pecos-decoders` meta-crate
are planned once it has been exercised more broadly (larger code families, Python
bindings, human users).

Pruning ranks accumulated prefix log mass plus a `score_alpha`-weighted
suffix-compatibility estimate. Unpruned results are exact and upstream-verified.

Deterministic ordering and tie-breaking are bitwise reproducible for a fixed
build and platform. The platform's `ln` and `exp` implementations may differ
across platforms.
