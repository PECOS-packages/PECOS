# pecos-relay-bp

Relay BP decoder for PECOS.

## Purpose

Provides PECOS native Relay BP decoding for quantum low-density parity-check (qLDPC) codes. Relay BP enhances standard min-sum belief propagation with disordered memory strengths, ensembling, and relaying for improved convergence on codes like bivariate bicycle codes.

## Key Types

- `RelayBpDecoder` - Relay BP ensemble decoder
- `MinSumBpDecoder` - Plain min-sum BP decoder
- `RelayConfig` - Relay ensemble configuration
- `MinSumConfig` - Min-sum BP configuration
- `RelayBpBuilder` / `MinSumBpBuilder` - Builder patterns for decoder construction

Relay runs may use sampled memory strengths or reproducible explicit gamma
sets. Configure the latter with `RelayConfig::explicit_gammas` or
`RelayBpBuilder::explicit_gammas`; public row zero is the first relay leg,
each set has one finite value per check-matrix column, and rows are reused
cyclically when necessary.

Results are a deterministic function of `(seed, syndrome)`: decode order,
batch splitting and worker count cannot change a result. `gamma0=None` makes
leg zero memoryless; relay legs retain sampled or explicit memory strengths.
Use `pre_iter` and `set_max_iter` for Relay iteration limits. Min-sum retains
`max_iter`, runs one native memory-BP leg, and is plain BP when `gamma0=None`.
`alpha=None` means unit scaling; `alpha=Some(0.0)` enables dynamic alpha with
a finite, strictly positive `alpha_iteration_scaling_factor`.

## Acknowledgements

This crate is a thin facade over `pecos-bp`, implementing the Relay-BP algorithm of Müller et al.

The frozen parity fixtures were generated with the Apache-2.0 reference implementation [trmue/relay](https://github.com/trmue/relay).

**Paper:**
- Müller, T., Alexander, T., Beverland, M. E., Buehler, M., Johnson, B. R., Maurer, T., & Vandeth, D. (2025). "Improved Belief Propagation Is Sufficient for Real-Time Decoding of Quantum Memory." [arXiv:2506.01779](https://arxiv.org/abs/2506.01779)
