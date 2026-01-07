# pecos-chromobius

Chromobius color code decoder for PECOS.

## Purpose

Wraps the Chromobius decoder for color code quantum error correction. Uses Mobius matching for efficient syndrome decoding.

## Key Types

- `ChromobiusDecoder` - Main decoder interface
- `ChromobiusConfig` - Decoder configuration

## Acknowledgements

This crate wraps [Chromobius](https://github.com/quantumlib/chromobius), a color code decoder developed by Craig Gidney and Cody Jones at Google Quantum AI.

**Paper:**
- Gidney, C. & Jones, C. (2023). "New circuits and an open source decoder for the color code." [arXiv:2312.08813](https://arxiv.org/abs/2312.08813)
