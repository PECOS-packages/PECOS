# pecos-tesseract

Tesseract search-based decoder for PECOS.

## Purpose

Wraps the Tesseract search-based decoder for quantum error correction. Uses A* search with pruning heuristics to find the most likely error configuration. Also wraps upstream's trellis-mode decoder, which sums probability mass over a beam of partial syndromes and reports a per-shot observable probability.

## Key Features

- A* search with Dijkstra algorithm
- Trellis-mode beam decoder with observable probabilities (one observable per model, as upstream)
- Support for Stim circuits and Detector Error Models (DEM)
- Parallel decoding with multithreading
- Beam search for efficiency

## Key Types

- `TesseractDecoder` - A* decoder interface
- `TesseractConfig` - A* decoder configuration
- `TesseractTrellisDecoder` - Trellis-mode decoder interface
- `TesseractTrellisConfig` - Trellis-mode configuration (`beam_width`, `beam_eps`, ranking mode)
- `TesseractTrellisResult` - Trellis-mode result with `observable_probability`

## Native dependencies

`pecos.toml` pins the upstream Tesseract commit together with the headers it
compiles against (Stim, Boost `dynamic_bitset`, and nlohmann/json at the
commit Tesseract's own `MODULE.bazel` pins). The build script fetches them into
`~/.pecos/deps/` on first use.

## Acknowledgements

This crate wraps [Tesseract](https://github.com/quantumlib/tesseract-decoder), a search-based decoder developed at Google Quantum AI.

**Paper:**
- Aghababaie Beni, L., Higgott, O., & Shutty, N. (2025). "Tesseract: A Search-Based Decoder for Quantum Error Correction." [arXiv:2503.10988](https://arxiv.org/abs/2503.10988)
