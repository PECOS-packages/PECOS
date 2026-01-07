# pecos-pymatching

PyMatching MWPM decoder for PECOS.

## Purpose

Wraps the PyMatching minimum-weight perfect matching decoder for quantum error correction.

## Key Types

- `PyMatchingDecoder` - Main decoder interface
- `PyMatchingBuilder` - Builder pattern for construction
- `CheckMatrix` - Parity check matrix representation
- `PyMatchingConfig` - Decoder configuration

## Features

- Batch decoding support
- Zero-copy decode buffers
- Petgraph integration for graph construction

## Acknowledgements

This crate wraps [PyMatching](https://github.com/oscarhiggott/PyMatching), a fast MWPM decoder developed by Oscar Higgott.

**Papers:**
- Higgott, O. (2022). "PyMatching: A Python package for decoding quantum codes with minimum-weight perfect matching." ACM Transactions on Quantum Computing. [arXiv:2105.13082](https://arxiv.org/abs/2105.13082)
- Higgott, O. & Gidney, C. (2023). "Sparse Blossom: correcting a million errors per core second with minimum-weight matching." [arXiv:2303.15933](https://arxiv.org/abs/2303.15933)
