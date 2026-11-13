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

BP+LSD defaults are defined in `pecos_ldpc_decoders::bp_lsd_defaults`: minimum-sum
BP with scaling 0.625, at most 30 iterations, parallel scheduling, one bit per
cluster-growth step, and LSD-0 (order 0, no reprocessing). `BpLsdConfig` uses
`ms_scaling_factor: Some(0.625)` for minimum-sum; `None` selects product-sum.
`bits_per_step: 0` grows all candidate bits each step. The native backend remains
optional: `BpLsdConfig` mirrors these defaults without the `ldpc` feature;
`spec_and_raw_builder_use_shared_paper_defaults` checks that the values agree.
