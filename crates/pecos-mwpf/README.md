# pecos-mwpf

PECOS wrapper for the [MWPF (Minimum-Weight Parity Factor)](https://github.com/yuewuo/mwpf) hypergraph decoder by Yue Wu (Yale).

Unlike MWPM decoders (PyMatching, Fusion Blossom), MWPF handles hyperedges natively. This means it can decode Y errors, depolarizing noise, color codes, and small QLDPC codes with higher accuracy than graph-based decoders that must decompose hyperedges.

The tradeoff is a heavier worst-case latency tail. MWPF is best suited for offline benchmarks, correlated-noise studies, and accuracy-first decoding.

## Key configuration

- `cluster_node_limit` (default 50): Controls accuracy vs speed. Lower values are faster.
- `timeout`: Optional solver timeout in seconds.

## License

MWPF is MIT-licensed. This wrapper is Apache-2.0-licensed as part of PECOS.
