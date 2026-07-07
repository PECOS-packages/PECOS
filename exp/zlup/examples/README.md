# Zlup Examples

This directory contains example programs demonstrating Zlup features and common quantum algorithms.

## Running Examples

```bash
# Compile to SLR-AST JSON
zlup compile examples/bell_state.zlp -o bell_state.json

# Check syntax and semantics
zlup check examples/bell_state.zlp
```

## Examples

### Basic Quantum States

| File | Description |
|------|-------------|
| `bell_state.zlp` | Creates the Bell state (|00⟩ + |11⟩) / sqrt(2) |
| `ghz_state.zlp` | Creates an N-qubit GHZ state |

### Quantum Algorithms

| File | Description |
|------|-------------|
| `teleportation.zlp` | Quantum teleportation protocol |
| `grover_2qubit.zlp` | Grover's search algorithm for 2 qubits |
| `qft_3qubit.zlp` | Quantum Fourier Transform on 3 qubits |

### Error Correction

| File | Description |
|------|-------------|
| `simple_qec.zlp` | 3-qubit bit-flip code with syndrome measurement |

### Testing

| File | Description |
|------|-------------|
| `test_lsp.zlp` | Test file for LSP functionality |

## Learning Path

1. Start with `bell_state.zlp` to understand basic gates and entanglement
2. Move to `ghz_state.zlp` to see loops and multiple qubits
3. Try `teleportation.zlp` for a complete protocol
4. Explore `simple_qec.zlp` for error correction concepts
5. Study `grover_2qubit.zlp` and `qft_3qubit.zlp` for algorithms
