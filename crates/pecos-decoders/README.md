# pecos-decoders

Unified decoder meta-crate for PECOS.

## Purpose

Provides a unified interface to all PECOS decoders through feature-gated re-exports.

## Features

Enable the appropriate features to include specific decoder families:

- `ldpc` - LDPC decoders (BP-OSD, BP-LSD, Union-Find, etc.)
- `fusion-blossom` - Fusion Blossom MWPM decoder
- `pymatching` - PyMatching MWPM decoder
- `tesseract` - Tesseract search-based decoder
- `chromobius` - Chromobius color code decoder
- `all` - Enable all decoders

## Key Types

Re-exports from `pecos-decoder-core`:
- `Decoder` trait - Interface for QEC decoders
- `BatchDecoder` trait - Batch decoding interface
- `CssDecoder` trait - CSS code specific decoding
- `SoftDecoder` trait - Soft information decoding
