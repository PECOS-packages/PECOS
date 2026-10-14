# pecos-uf-decoder

Pure Rust Union-Find decoding for surface codes and other quantum error correction
codes with a matching-graph structure. Decoding operates on syndrome graphs, where
nodes represent detectors and edges represent error mechanisms.

The crate provides `UfDecoder`, belief-propagation-assisted decoding through
`BpUfDecoder`, CSS decoding through `CssUfDecoder`, and windowed decoding variants.
It integrates with the decoder traits in `pecos-decoder-core` and reuses scratch
storage across shots.

See the [crate documentation](src/lib.rs) and [tests](tests/) for the available
decoders and examples of graph construction and decoding.
