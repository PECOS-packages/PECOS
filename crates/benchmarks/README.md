# benchmarks

`benchmarks` is an **internal crate** to provide benchmarking of Rust code.

This is not intended for external use.

## Running Benchmarks

For best performance, use the pecos CLI which enables native CPU optimizations (AVX2, etc.):

```bash
# Run all benchmarks
pecos rust bench

# Run specific benchmarks
pecos rust bench "SoA Comparison"
pecos rust bench "DOD"
```

Alternatively, run manually with:

```bash
RUSTFLAGS="-C target-cpu=native" cargo bench -p benchmarks
```
