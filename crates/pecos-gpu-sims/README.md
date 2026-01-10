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

## Development

### Current Optimizations

- **Dynamic uniform buffer offsets**: Avoids per-gate bind group creation by using a single persistent bind group with dynamic offsets into a pre-allocated parameter buffer.
- **Batched buffer writes**: Gate parameters are accumulated on the CPU and written to GPU memory in a single transfer per batch, reducing driver overhead.
- **2D dispatch**: Workgroup counts exceeding the 65535 limit (at 24+ qubits) use 2D dispatch with dynamic linear index computation in shaders.
- **Adapter-based limits**: Queries the GPU adapter for maximum buffer sizes rather than using hardcoded values, enabling support for larger qubit counts on capable hardware.

### Potential Future Optimizations

These optimizations could provide additional performance gains but require more substantial engineering effort:

- **Workgroup-local memory**: Cache portions of the state vector in workgroup shared memory to reduce global memory bandwidth. Most beneficial for gates that access nearby amplitude pairs.
- **Double buffering**: Overlap compute and memory transfers by using ping-pong buffers, allowing the next batch of parameters to upload while the current batch executes.
- **Kernel fusion**: Combine multiple single-qubit gates acting on different qubits into a single dispatch, reducing kernel launch overhead and memory round-trips.
- **Sparse state representation**: For circuits that maintain low entanglement, a sparse representation could reduce memory requirements and computation for large qubit counts.
