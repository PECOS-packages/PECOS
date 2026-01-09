# pecos-gpu-sims

Cross-platform GPU-accelerated quantum state vector simulator using [wgpu](https://wgpu.rs/).

## Supported Backends

- Vulkan (Linux, Windows)
- Metal (macOS, iOS)
- DirectX 12 (Windows)
- WebGPU (browsers via WASM)

## Requirements

A GPU with Vulkan, Metal, or DX12 support. Check availability with:

```bash
cargo run -p pecos-gpu-sims --bin gpu-check
```

Or via the PECOS CLI:

```bash
pecos gpu check
```

## Usage

```rust
use pecos_gpu_sims::GpuStateVec;

let mut sim = GpuStateVec::new(4)?;  // 4 qubits
sim.h(0);                             // Hadamard
sim.cx(0, 1);                         // CNOT
let result = sim.measure(0);          // Measure
```

## Supported Gates

| Gate | Method | Description |
|------|--------|-------------|
| H | `h(q)` | Hadamard |
| X, Y, Z | `x(q)`, `y(q)`, `z(q)` | Pauli gates |
| S, Sdg | `sz(q)`, `szdg(q)` | Phase gates |
| T, Tdg | `t(q)`, `tdg(q)` | T gates |
| RX, RY, RZ | `rx(θ,q)`, `ry(θ,q)`, `rz(θ,q)` | Rotation gates |
| CX, CZ | `cx(c,t)`, `cz(c,t)` | Two-qubit gates |
| RZZ | `rzz(θ,q1,q2)` | ZZ rotation |

## Error Handling

If no GPU is available, `GpuStateVec::new()` returns `Err(GpuError::NoAdapter)`. Use a CPU-based simulator like `StateVec` as a fallback.
