# pecos-ldpc-decoders

LDPC decoder implementations for PECOS.

## Purpose

Provides LDPC (Low-Density Parity-Check) based decoders including belief propagation and related algorithms.

## Available Decoders

- `BpOsdDecoder` - Belief Propagation with Ordered Statistics Decoding
- `BpLsdDecoder` - Belief Propagation with Localised Statistics Decoding
- `SoftInfoBpDecoder` - Soft Information BP decoder
- `FlipDecoder` - Bit-flipping decoder
- `UnionFindDecoder` - Union-Find decoder
- `BeliefFindDecoder` - BP + Union-Find hybrid
- `MbpDecoder` - MBP decoder for quantum codes

## Acknowledgements

This crate wraps [ldpc](https://github.com/quantumgizmos/ldpc), a high-performance LDPC decoder library developed by Joschka Roffe and collaborators.

**Paper:**
- Roffe, J., White, D. R., Burton, S., & Campbell, E. T. (2020). "Decoding Across the Quantum LDPC Code Landscape." Physical Review Research, 2(4), 043423. [arXiv:2005.07016](https://arxiv.org/abs/2005.07016)
