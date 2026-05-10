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

## Fault Catalog Benchmarks

The fault-catalog suite covers rotated surface-code memory circuits at
distances 3, 5, 7, 9, and 11. It measures structural catalog construction,
noise re-parameterization, raw-mechanism materialization, and noise-sweep
strategies.

```bash
# Full fault-catalog suite
just bench native "" "fault_catalog/"

# Structural construction only
just bench native "" "fault_catalog/from_circuit"

# Compare direct rebuild, cloned parameterization, and mutable with_noise sweeps
just bench native "" "fault_catalog/noise_sweep"
```

For parameter sweeps that do not need to keep independent catalog snapshots,
prefer building one structural catalog and calling `with_noise()` for each
noise point. Use `parameterized()` when the code needs independent catalogs
alive at the same time.
