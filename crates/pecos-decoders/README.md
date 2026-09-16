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
- `relay-bp` - Relay BP decoder for qLDPC codes
- `all` - Enable all decoders

## Key Types

Re-exports from `pecos-decoder-core`:
- `Decoder` trait - Interface for QEC decoders
- `BatchDecoder` trait - Batch decoding interface
- `CssDecoder` trait - CSS code specific decoding
- `SoftDecoder` trait - Soft information decoding

## Frontier

Enable the `frontier` feature to build the experimental native Rust Frontier
decoder through `DecoderSpec::Frontier(FrontierConfig::default())` or
`DecoderSpec::parse("frontier")`. It accepts raw DEMs with hyperedges and
arbitrary-width observables. Python builds enable this feature and expose
`pecos.decoders.frontier()` for sequential or parallel batch decoding.

## BP-Trellis

Enable the `bp-trellis` feature to build PECOS's experimental BP-guided trellis
decoder through `DecoderSpec::BpTrellis(BpTrellisConfig::default())` or
`DecoderSpec::parse("bp_trellis")`. Python exposes `pecos.decoders.bp_trellis()`
with all configuration options, including optional no-path escalation widths,
for sequential or parallel batch decoding.
