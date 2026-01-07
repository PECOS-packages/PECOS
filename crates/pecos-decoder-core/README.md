# pecos-decoder-core

Core decoder traits and types for PECOS.

## Purpose

Defines the fundamental decoder traits used across all decoder implementations. Separated from `pecos-decoders` to avoid circular dependencies.

## Key Traits

- `Decoder` - Core trait all decoders implement
- `BatchDecoder` - Batch decoding interface
- `CssDecoder` - CSS code specific decoding
- `SoftDecoder` - Soft information (LLR) decoding

## Additional Types

- `DecoderError` - Unified error types
- `DecodingResultTrait` - Result trait
- `CheckMatrixConfig` - Matrix configuration
