# Performance: Large-Scale Simulations

## Scaling to 1M+ Qubits

`NoiseContext` uses bit vectors internally for O(1) qubit state lookups,
enabling efficient simulation of large qubit counts:

```rust
use pecos_neo::noise::NoiseContext;

// Pre-allocate for large simulations (optional but recommended)
let ctx = NoiseContext::with_capacity(1_000_000);
```

## Benchmarks

Performance (measured on typical hardware):

| Qubits | Time |
|--------|------|
| 1K     | 0.09 ms |
| 10K    | 0.8 ms  |
| 100K   | 70 ms   |
| 1M     | 1.0 s   |

See the `large_scale` example for a benchmark.
