# pecos-relay-bp

Relay BP decoder for PECOS.

## Purpose

Wraps the Relay BP decoder for quantum low-density parity-check (qLDPC) code decoding. Relay BP enhances standard min-sum belief propagation with disordered memory strengths, ensembling, and relaying for improved convergence on codes like bivariate bicycle codes.

## Key Types

- `RelayBpDecoder` - Relay BP ensemble decoder
- `MinSumBpDecoder` - Plain min-sum BP decoder
- `RelayConfig` - Relay ensemble configuration
- `MinSumConfig` - Min-sum BP configuration
- `RelayBpBuilder` / `MinSumBpBuilder` - Builder patterns for decoder construction

## Acknowledgements

This crate wraps [relay-bp](https://github.com/trmue/relay), a Relay BP decoder developed by Tristan Mueller, Thomas Alexander, Michael E. Beverland, Markus Buehler, Blake R. Johnson, Thilo Maurer, and Drew Vandeth.

**Paper:**
- Mueller, T., Alexander, T., Beverland, M. E., Buehler, M., Johnson, B. R., Maurer, T., & Vandeth, D. (2025). "Improved Belief Propagation Is Sufficient for Real-Time Decoding of Quantum Memory." [arXiv:2506.01779](https://arxiv.org/abs/2506.01779)
