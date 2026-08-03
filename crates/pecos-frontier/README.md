# PECOS Frontier Decoder

Native Rust implementation of the Frontier approximate logical maximum-likelihood decoder.

`k` and `delta` operate on prefix log mass only (upstream's `score_alpha`
suffix-compatibility scoring is a planned follow-up); pruned results and K/Delta
values are not directly comparable to upstream until then. Unpruned results are
exact and upstream-verified.

Deterministic ordering and tie-breaking are bitwise reproducible for a fixed
build and platform. The platform's `ln` and `exp` implementations may differ
across platforms.
