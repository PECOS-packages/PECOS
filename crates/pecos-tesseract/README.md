# pecos-tesseract

Tesseract search-based decoder for PECOS.

## Purpose

Wraps the Tesseract search-based decoder for quantum error correction. Uses A* search with pruning heuristics to find the most likely error configuration.

## Key Features

- A* search with Dijkstra algorithm
- Support for Stim circuits and Detector Error Models (DEM)
- Parallel decoding with multithreading
- Beam search for efficiency

## Key Types

- `TesseractDecoder` - Main decoder interface
- `TesseractConfig` - Decoder configuration

## Acknowledgements

This crate wraps [Tesseract](https://github.com/quantumlib/tesseract-decoder), a search-based decoder developed at Google Quantum AI.

**Paper:**
- Beni, N., Higgott, O., & Shutty, N. (2025). "Tesseract: A Search-Based Decoder for Quantum Error Correction." [arXiv:2503.10988](https://arxiv.org/abs/2503.10988)
