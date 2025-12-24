# Command Line Interface

PECOS provides a command-line interface for running quantum simulations without writing code.

## Installation

```bash
# Install with default features (recommended)
cargo install pecos

# Install with LLVM/QIS support (requires LLVM 14)
cargo install pecos --features llvm

# Install with all simulator backends
cargo install pecos --features full
```

## Quick Start

```bash
# Check your installation
pecos info

# Run a built-in example
pecos examples bell --run

# Run your own circuit
pecos run my_circuit.qasm -s 1000
```

## Commands

### `pecos run`

Run a quantum program with configurable simulation parameters.

```bash
pecos run <PROGRAM> [OPTIONS]
```

**Supported formats:**

- `.qasm` - OpenQASM 2.0 circuits
- `.phir.json` or `.json` - PHIR/JSON programs
- `.ll` - QIS programs in LLVM IR format (requires `--features llvm`)

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `-s, --shots <N>` | Number of simulation shots | 1 |
| `-w, --workers <N>` | Parallel workers | 1 |
| `-d, --seed <N>` | Random seed for reproducibility | random |
| `-S, --sim <TYPE>` | Simulator: `statevector` or `stabilizer` | statevector |
| `-m, --model <TYPE>` | Noise model: `depolarizing` or `general` | depolarizing |
| `-p, --noise <P>` | Noise probability (0-1) | none |
| `-o, --output <FILE>` | Output file path | stdout |
| `-f, --format <FMT>` | Result format: `decimal`, `binary`, `hex` | decimal |

**Examples:**

```bash
# Run 1000 shots of a QASM circuit
pecos run circuit.qasm -s 1000

# Reproducible simulation with fixed seed
pecos run bell.qasm -s 100 -d 42

# Use stabilizer simulator for Clifford circuits (faster)
pecos run clifford.qasm -S stabilizer -s 10000

# Add 1% depolarizing noise
pecos run circuit.qasm -s 1000 -p 0.01

# Parallel execution with 4 workers
pecos run large_circuit.qasm -s 10000 -w 4

# Save results to file in binary format
pecos run circuit.qasm -s 1000 -o results.json -f binary

# Use general noise model with per-operation rates
# Format: prep,meas_0,meas_1,single_qubit,two_qubit
pecos run circuit.qasm -s 1000 -m general -p 0.01,0.02,0.02,0.05,0.1
```

### `pecos compile`

Compile a QIS program to native code (requires LLVM feature).

```bash
pecos compile <PROGRAM>
```

This pre-compiles QIS programs for faster subsequent execution.

### `pecos info`

Display version, compiled features, and system information.

```bash
$ pecos info
PECOS - Quantum Error Correction Simulator
Version: 0.1.1

Compiled Features:
  [x] qasm     - OpenQASM 2.0 circuit support
  [x] phir     - PHIR/JSON program support
  [x] selene   - Selene QIS runtime
  [x] wasm     - WebAssembly foreign objects
  [ ] llvm     - LLVM/QIS compilation
  [ ] quest    - QuEST simulator backend
  [ ] qulacs   - Qulacs simulator backend

Simulators:
  statevector  - Full quantum state simulation (default)
  stabilizer   - Efficient Clifford circuit simulation

Noise Models:
  depolarizing - Uniform error probability (default)
  general      - Configurable per-operation error rates
```

### `pecos doctor`

Check installation and diagnose common issues.

```bash
$ pecos doctor
Checking PECOS installation...

  [OK] PECOS CLI: v0.1.1
  [OK] QASM support: available
  [OK] PHIR/JSON support: available
  [OK] Selene runtime: available
  [!!] LLVM/QIS support: not compiled (optional)
  [OK] LLVM 14: 14.0.6 at /home/user/.pecos/llvm
  [OK] Test circuit: execution successful

Suggestions:
  - LLVM support not compiled. To enable: cargo install pecos --features llvm

All checks passed! PECOS is ready to use.
```

### `pecos examples`

List and run example quantum circuits.

```bash
# List available examples
pecos examples

# Show an example circuit
pecos examples bell

# Run an example (100 shots)
pecos examples bell --run

# Copy example to current directory
pecos examples bell --copy
```

**Available examples:**

| Name | Description |
|------|-------------|
| `bell` | Bell state - entangle two qubits |
| `ghz` | GHZ state - three-qubit entanglement |
| `teleport` | Quantum teleportation protocol |
| `superposition` | Simple superposition with Hadamard gate |
| `phase` | Phase kickback demonstration |

### `pecos completions`

Generate shell completion scripts.

```bash
# Bash
pecos completions bash > ~/.local/share/bash-completion/completions/pecos

# Zsh (add ~/.zfunc to fpath in .zshrc)
pecos completions zsh > ~/.zfunc/_pecos

# Fish
pecos completions fish > ~/.config/fish/completions/pecos.fish

# PowerShell
pecos completions powershell >> $PROFILE
```

## Simulators

### State Vector Simulator

The default simulator that maintains the full quantum state. Supports all quantum gates including arbitrary rotations.

```bash
pecos run circuit.qasm -S statevector
```

Best for:

- Small to medium circuits (up to ~25 qubits)
- Circuits with non-Clifford gates (T, Rx, Ry, Rz, etc.)
- Highest accuracy simulations

### Stabilizer Simulator

An optimized simulator for Clifford circuits that uses the stabilizer formalism.

```bash
pecos run clifford_circuit.qasm -S stabilizer
```

Best for:

- Large Clifford circuits (100+ qubits)
- Circuits using only H, S, CNOT, and Pauli gates
- Error correction simulations

## Noise Models

### Depolarizing Noise

Applies uniform error probability to all operations.

```bash
# 1% error rate on all operations
pecos run circuit.qasm -s 1000 -p 0.01
```

### General Noise Model

Allows different error rates for different operation types.

```bash
# Format: prep,meas_0,meas_1,single_qubit,two_qubit
pecos run circuit.qasm -s 1000 -m general -p 0.001,0.01,0.01,0.001,0.01
```

Parameters:

- `prep` - State preparation error probability
- `meas_0` - Measurement error for |0⟩ state
- `meas_1` - Measurement error for |1⟩ state
- `single_qubit` - Single-qubit gate error probability
- `two_qubit` - Two-qubit gate error probability

## Output Formats

Results are output as JSON with measurement outcomes:

```bash
# Decimal format (default)
pecos run bell.qasm -s 5
# {"c": [0, 3, 0, 3, 3]}

# Binary format
pecos run bell.qasm -s 5 -f binary
# {"c": ["00", "11", "00", "11", "11"]}

# Hexadecimal format
pecos run bell.qasm -s 5 -f hex
# {"c": ["0x0", "0x3", "0x0", "0x3", "0x3"]}
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `RUST_LOG` | Set logging level (`error`, `warn`, `info`, `debug`, `trace`) |
| `PECOS_HOME` | Override PECOS cache directory |

```bash
# Enable debug logging
RUST_LOG=debug pecos run circuit.qasm

# Use custom cache directory
PECOS_HOME=/tmp/pecos pecos run circuit.qasm
```

## See Also

- [Getting Started](getting-started.md) - Python API introduction
- [QASM Simulation](qasm-simulation.md) - OpenQASM format details
- [Simulators](simulators.md) - Simulator backends in depth
- [Noise Model Builders](noise-model-builders.md) - Advanced noise configuration
