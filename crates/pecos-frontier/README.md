# PECOS Frontier Decoder

Native Rust implementation of the Frontier approximate logical maximum-likelihood decoder.

Pruning ranks accumulated prefix log mass plus a `score_alpha`-weighted
suffix-compatibility estimate. Unpruned results are exact and upstream-verified.

Deterministic ordering and tie-breaking are bitwise reproducible for a fixed
build and platform. The platform's `ln` and `exp` implementations may differ
across platforms.
