# pecos-fusion-blossom

Fusion Blossom MWPM decoder for PECOS.

## Purpose

Wraps the Fusion Blossom minimum-weight perfect matching decoder for quantum error correction.

## Key Types

- `FusionBlossomDecoder` - Main decoder interface
- `FusionBlossomConfig` - Decoder configuration
- `SyndromeData` - Syndrome input format
- `StandardCode` - Standard code definitions

## Acknowledgements

This crate wraps [Fusion Blossom](https://github.com/yuewuo/fusion-blossom), a fast MWPM decoder developed by Yue Wu, Namitha Liyanage, and Lin Zhong at Yale University.

**Paper:**
- Wu, Y., Liyanage, N., & Zhong, L. (2023). "Fusion Blossom: Fast MWPM Decoders for QEC." [arXiv:2305.08307](https://arxiv.org/abs/2305.08307)
