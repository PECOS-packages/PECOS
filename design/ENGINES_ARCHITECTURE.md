# Engines Architecture: Simulation Framework

This document describes the architecture of the `pecos-engines` crate, which provides the simulation framework for PECOS. It explains how quantum programs are executed, how classical and quantum components interact, and how the system enables mid-circuit measurements with classical feedback.

## Design Philosophy

PECOS serves two complementary roles:

**As a Framework** - A complete, extendable environment for studying QEC and hybrid quantum-classical computation. Users can plug in custom components (error models, decoders, machines) and run full simulations with the `sim()` API or `HybridEngine`.

**As a Library** - A collection of well-designed, independent components that users can pick and choose for their own projects. Need just a fast stabilizer simulator? Use `pecos-simulators::SparseStab`. Need deterministic seeding? Use `pecos-core::derive_seed()`. The crates are designed to be useful standalone.

```
┌─────────────────────────────────────────────────────────────────┐
│  PECOS as Framework                                             │
│  - sim(program).quantum(sparse_stab()).run(1000)                │
│  - HybridEngine with custom components                          │
│  - Full QEC simulation pipelines                                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  PECOS as Library (pick what you need)                          │
│                                                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │  pecos-simulators  │ │  pecos-core  │ │  pecos-random   │            │
│  │  SparseStab  │ │  QubitId     │ │  PecosRng    │            │
│  │  StateVec    │ │  derive_seed │ │              │            │
│  │  Gateable    │ │  GateType    │ │              │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
│                                                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │ pecos-gpu-   │ │   pecos-     │ │    pecos-    │            │
│  │    sims      │ │  clifford-   │ │   engines    │            │
│  │ GpuSampler   │ │   gates      │ │ ByteMessage  │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

This dual nature means:
- Researchers can quickly prototype QEC experiments using the framework
- Library authors can integrate specific PECOS components into their own tools
- The same battle-tested code serves both use cases

## Overview

The `pecos-engines` crate orchestrates quantum simulation through a layered architecture:

1. **User API Layer** - `sim()` function and `SimBuilder` for configuration
2. **Parallelization Layer** - `MonteCarloEngine` for multi-shot execution
3. **Execution Layer** - `HybridEngine` for single-shot orchestration
4. **Component Layer** - Classical engines, quantum systems, and noise models

```
┌─────────────────────────────────────────────────────────────┐
│                      User API (sim_builder)                 │
│  sim(program).quantum(sparse_stab()).noise(...).run(1000)   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │   MonteCarloEngine   │ (parallel orchestration)
          │  (num_workers, seed) │
          └──────────┬───────────┘
                     │
          ┌──────────┴───────────┐
          │ (parallel workers)   │
          ▼                      ▼
     ┌──────────────────────────────────────────┐
     │      HybridEngine (per worker)           │
     │  - Cloned with derived seed              │
     │  - Reset between shots                   │
     └────────┬─────────────────────────────────┘
              │
    ┌─────────┴──────────┐
    │                    │
    ▼                    ▼
┌────────────────┐   ┌──────────────────────┐
│ ClassicalEngine│   │    QuantumSystem     │
│                │   │ ┌────────────────┐   │
│ - generate_    │   │ │   NoiseModel   │   │
│   commands()   │   │ │  (transforms   │   │
│ - handle_      │   │ │   operations)  │   │
│   measurements │   │ └───────┬────────┘   │
│ - get_results()│   │         ▼            │
└────────┬───────┘   │ ┌────────────────┐   │
         │           │ │ QuantumEngine  │   │
         │           │ │ (StateVec or   │   │
         │           │ │  SparseStab)   │   │
         │           │ └────────────────┘   │
         │           └──────────────────────┘
         │                    │
         └────ByteMessage─────┘
              (binary protocol)
```

## Core Concepts

### The Engine Trait

All components in the system implement the base `Engine` trait:

```rust
pub trait Engine<Input, Output> {
    fn process(&mut self, input: Input) -> Result<Output>;
    fn reset(&mut self) -> Result<()>;
}
```

This simple interface enables composition - engines can delegate to other engines.

### Control Flow with EngineStage

The `EngineStage` enum enables feedback loops between components:

```rust
pub enum EngineStage<I, O> {
    NeedsProcessing(I),  // "Send this input to the controlled engine"
    Complete(O),         // "Processing finished, here's the result"
}
```

This is used by `ControlEngine` implementations (like `ClassicalEngine` and `NoiseModel`) to orchestrate execution with another engine.

### ByteMessage Protocol

Components communicate using `ByteMessage`, a binary protocol for quantum commands and measurement results:

```rust
// Commands from classical to quantum
ByteMessage: [H(0), CX(0,1), MZ(0), MZ(1)]

// Results from quantum to classical
ByteMessage: [MZ(0)=1, MZ(1)=1]
```

This allows efficient batching of operations and decouples the classical and quantum components.

## The Classical-Quantum Feedback Loop

The key architectural feature is the **feedback loop** between classical and quantum components. This enables:

- Mid-circuit measurements
- Classical control based on measurement outcomes
- Repeat-until-success protocols
- QEC syndrome decoding and correction

### Single Shot Execution Flow

Inside `HybridEngine::run_shot()`:

```
┌─────────────────┐                      ┌─────────────────┐
│ ClassicalEngine │                      │  QuantumSystem  │
└────────┬────────┘                      └────────┬────────┘
         │                                        │
         │  1. start()                            │
         │  ─────────────────────────────────►    │
         │     ByteMessage: [H(0), CX(0,1), MZ(0)]│
         │                                        │
         │  2. process() → execute gates          │
         │                                        │
         │  3. measurement results                │
         │  ◄─────────────────────────────────    │
         │     ByteMessage: [MZ(0) = 1]           │
         │                                        │
         │  4. continue_processing()              │
         │     // Decide next action based on     │
         │     // measurement result              │
         │                                        │
         │  5. More commands (if needed)          │
         │  ─────────────────────────────────►    │
         │     ByteMessage: [X(1), MZ(1)]         │
         │                                        │
         │  6. process()                          │
         │                                        │
         │  7. final measurements                 │
         │  ◄─────────────────────────────────    │
         │                                        │
         │  8. Complete(Shot)                     │
         ▼                                        ▼
```

### The Loop in Code

```rust
// Simplified HybridEngine::run_shot()
fn run_shot(&mut self) -> Result<Shot> {
    // Reset both engines for fresh shot
    self.classical_engine.reset()?;
    self.quantum_system.reset()?;

    // Start execution - classical engine generates first batch of commands
    let mut stage = self.classical_engine.start()?;

    loop {
        match stage {
            EngineStage::NeedsProcessing(commands) => {
                // Send commands to quantum system
                let measurements = self.quantum_system.process(commands)?;

                // Classical engine processes measurements and decides next action
                stage = self.classical_engine.continue_processing(measurements)?;
            }
            EngineStage::Complete(shot) => {
                // Done - return results
                return Ok(shot);
            }
        }
    }
}
```

### Concrete Example: QasmEngine

The `QasmEngine` is a good example to understand how the feedback loop works in practice. Consider this QASM program with conditional logic:

```qasm
h q[0];
measure q[0] -> c[0];
if (c==1) x q[0];
measure q[0] -> c[1];
```

Here's exactly what happens:

**Round 1 - Start:**
```
QasmEngine.start()
  └─ process_program_impl()
       ├─ Process: h q[0]  → add H gate to batch
       └─ Process: measure q[0] → BREAK! Must wait for result
                                   Return NeedsProcessing([H(0), MZ(0)])
```

**Round 1 - Quantum System:**
```
QuantumSystem.process([H(0), MZ(0)])
  ├─ NoiseModel transforms operations (maybe adds errors)
  ├─ QuantumEngine executes H, then measures
  └─ Return ByteMessage([MZ(0) = 1])  // measured |1⟩
```

**Round 2 - Continue:**
```
QasmEngine.continue_processing([MZ(0) = 1])
  ├─ handle_measurements(): store c[0] = 1
  └─ process_program_impl()
       ├─ Process: if (c==1) x q[0]
       │    └─ c[0] is 1, so add X gate to batch
       └─ Process: measure q[0] → BREAK!
                                   Return NeedsProcessing([X(0), MZ(0)])
```

**Round 2 - Quantum System:**
```
QuantumSystem.process([X(0), MZ(0)])
  ├─ Execute X gate (flips |1⟩ back to |0⟩)
  └─ Return ByteMessage([MZ(0) = 0])
```

**Round 3 - Finish:**
```
QasmEngine.continue_processing([MZ(0) = 0])
  ├─ handle_measurements(): store c[1] = 0
  └─ process_program_impl()
       └─ No more operations → Return Complete(Shot { c: [1, 0] })
```

**Key insight:** QasmEngine breaks the batch on every measurement because:
1. The measurement result might be needed by the next operation (`if` statement)
2. It can't know the result until the quantum system actually measures
3. So it must pause, get the result, store it in classical registers, then continue

This is what makes mid-circuit measurement possible - the classical engine is in control, asking for quantum operations in batches and making decisions based on results.

### Why This Matters

This architecture enables **adaptive quantum circuits** where the program flow depends on measurement outcomes:

```
Example: Repeat-until-success

Round 1:
  Classical: "Apply H, measure"
  Quantum:   executes, returns measurement = 0
  Classical: "Wrong outcome, try again"

Round 2:
  Classical: "Reset, apply H, measure"
  Quantum:   executes, returns measurement = 1
  Classical: "Success! Done."
```

Without this feedback loop, you'd need to know all operations upfront, making adaptive protocols impossible.

## Component Details

### ClassicalEngine Trait

The classical engine controls program flow:

```rust
pub trait ClassicalEngine {
    /// Compile/prepare the program
    fn compile(&mut self) -> Result<()>;

    /// Generate quantum commands to execute
    fn generate_commands(&mut self) -> ByteMessage;

    /// Process measurement results from quantum system
    fn handle_measurements(&mut self, measurements: ByteMessage);

    /// Get final results after execution completes
    fn get_results(&self) -> Shot;

    /// Number of qubits needed
    fn num_qubits(&self) -> usize;

    /// Reset for next shot
    fn reset(&mut self) -> Result<()>;
}
```

Different classical engines implement different program formats:
- `QasmEngine` - OpenQASM circuits
- `QisEngine` - QIS/LLVM IR programs (via Helios)
- `HugrEngine` - HUGR graphs (via Guppy)

### QuantumEngine Trait

The quantum engine executes gates:

```rust
pub trait QuantumEngine {
    /// Process a batch of quantum commands, return measurement results
    fn process(&mut self, commands: ByteMessage) -> ByteMessage;

    /// Set RNG seed for reproducibility
    fn set_seed(&mut self, seed: u64);

    /// Reset quantum state
    fn reset(&mut self);
}
```

Built-in implementations:
- `StateVecEngine` - Full state vector simulation (universal)
- `SparseStabEngine` - Stabilizer simulation (Clifford circuits only)

### NoiseModel Trait

Noise models transform operations before execution:

```rust
pub trait NoiseModel: ControlEngine<ByteMessage, ByteMessage, ByteMessage, ByteMessage> {
    // Inherits from ControlEngine:
    // - start(commands) -> EngineStage<modified_commands, output>
    // - continue_processing(measurements) -> EngineStage<more_commands, output>
}
```

The noise model sits between classical and quantum engines:

```
Classical → NoiseModel → QuantumEngine
                ↑
         May add noise gates
         May flip measurement results
```

Built-in noise models:
- `PassThroughNoiseModel` - No noise (default)
- `DepolarizingNoiseModel` - Depolarizing noise on gates
- `BiasedDepolarizingNoiseModel` - Gate noise + measurement errors
- `GeneralNoiseModel` - Customizable per-gate noise

### QuantumSystem

`QuantumSystem` combines a noise model and quantum engine. The noise model "wraps" the quantum engine and can transform operations before they reach the simulator:

```rust
pub struct QuantumSystem {
    noise_model: Box<dyn NoiseModel>,
    quantum_engine: Box<dyn QuantumEngine>,
}
```

**The flow through QuantumSystem:**

```
                    QuantumSystem
┌─────────────────────────────────────────────────────┐
│                                                     │
│  ByteMessage: [H(0), CX(0,1), MZ(0)]               │
│         │                                           │
│         ▼                                           │
│  ┌─────────────────────────────────────────────┐   │
│  │            NoiseModel                        │   │
│  │  - May add depolarizing errors after gates   │   │
│  │  - May flip measurement outcomes             │   │
│  │  - Returns EngineStage::NeedsProcessing      │   │
│  │    with modified commands                    │   │
│  └─────────────────┬───────────────────────────┘   │
│                    │                                │
│         [H(0), X(0), CX(0,1), Z(1), MZ(0)]         │
│         (original ops + injected errors)           │
│                    │                                │
│                    ▼                                │
│  ┌─────────────────────────────────────────────┐   │
│  │          QuantumEngine                       │   │
│  │  (StateVec or SparseStab)                    │   │
│  │  - Executes all gates on quantum state       │   │
│  │  - Performs measurements                     │   │
│  │  - Returns raw measurement results           │   │
│  └─────────────────┬───────────────────────────┘   │
│                    │                                │
│         ByteMessage: [MZ(0) = 1]                   │
│                    │                                │
│                    ▼                                │
│  ┌─────────────────────────────────────────────┐   │
│  │            NoiseModel (again)                │   │
│  │  - May flip measurement results              │   │
│  │  - Returns EngineStage::Complete             │   │
│  │    with final measurements                   │   │
│  └─────────────────┬───────────────────────────┘   │
│                    │                                │
│         ByteMessage: [MZ(0) = 0]  (flipped!)       │
│                    │                                │
└────────────────────┼────────────────────────────────┘
                     │
                     ▼
           Back to ClassicalEngine
```

The NoiseModel is itself a `ControlEngine` - it can return `NeedsProcessing` to send modified commands to the quantum engine, and the loop continues until it returns `Complete`. This allows noise models to:
- Inject error gates before/after operations
- Transform gate parameters
- Flip measurement outcomes
- Add multiple rounds of noise injection if needed

## Parallelization with MonteCarloEngine

`MonteCarloEngine` distributes shots across worker threads:

```rust
pub struct MonteCarloEngine {
    /// Template engine (cloned for each worker)
    template: HybridEngine,

    /// Number of parallel workers
    num_workers: usize,

    /// Base seed for reproducibility
    seed: u64,
}
```

### Execution Flow

```rust
fn run(&mut self, num_shots: usize) -> Result<ShotVec> {
    // Distribute shots across workers
    let shots_per_worker = num_shots / self.num_workers;

    // Parallel execution with rayon
    let results: Vec<Shot> = (0..self.num_workers)
        .into_par_iter()
        .flat_map(|worker_id| {
            // Clone template with derived seed
            let mut engine = self.template.clone();
            engine.set_seed(derive_seed(self.seed, worker_id));

            // Run assigned shots
            (0..shots_per_worker)
                .map(|_| engine.run_shot())
                .collect::<Vec<_>>()
        })
        .collect();

    Ok(ShotVec::from(results))
}
```

### Seed Derivation for Reproducibility

Seeds are derived deterministically to ensure reproducible results:

```
Base seed (42)
    ├── Worker 0: derive_seed(42, 0)
    │     ├── Classical: derive_seed(..., 0)
    │     └── Quantum:   derive_seed(..., 1)
    │           ├── NoiseModel: derive_seed(..., 0)
    │           └── QuantumEngine: derive_seed(..., 1)
    ├── Worker 1: derive_seed(42, 1)
    │     └── ...
    ...
```

This ensures:
- Same seed always produces same results
- Different workers have uncorrelated random streams
- Different components have uncorrelated random streams

## User API: sim() and SimBuilder

The `sim()` function provides a fluent API for configuration:

```rust
// Basic usage
let results = sim(my_program)
    .quantum(sparse_stab())
    .run(1000)?;

// With noise
let results = sim(my_program)
    .quantum(state_vec())
    .noise(DepolarizingNoise { p: 0.01 })
    .seed(42)
    .workers(4)
    .run(10000)?;

// Reusable engine
let mut engine = sim_builder()
    .classical(qasm_engine().qasm("H q[0]; measure q[0] -> c[0];"))
    .quantum(sparse_stab())
    .build()?;

let batch1 = engine.run(1000)?;
let batch2 = engine.run(2000)?;  // Reuse same engine
```

### SimBuilder Configuration

| Method | Purpose |
|--------|---------|
| `.classical(builder)` | Set classical engine (program source) |
| `.quantum(builder)` | Set quantum simulator |
| `.noise(model)` | Set noise model |
| `.seed(u64)` | Set RNG seed for reproducibility |
| `.workers(n)` | Set number of parallel workers |
| `.build()` | Build reusable engine |
| `.run(shots)` | Build and run immediately |

## Results: Shot and ShotVec

### Shot

A `Shot` represents results from a single execution:

```rust
pub struct Shot {
    /// Named results (e.g., "outcome" -> 1)
    results: BTreeMap<String, Data>,
}

pub enum Data {
    U32(u32),
    I64(i64),
    F64(f64),
    Bool(bool),
    BitVec(BitVec),
    Json(serde_json::Value),
}
```

### ShotVec

A `ShotVec` aggregates results from multiple shots in columnar format:

```rust
let results: ShotVec = engine.run(1000)?;

// Access as columns
let outcomes: &[i64] = results.get_i64("outcome")?;
// outcomes = [0, 1, 1, 0, 1, ...] (1000 values)

// Convert to HashMap
let map: HashMap<String, Vec<i64>> = results.to_map();
```

## Crate Dependencies

```
pecos-engines (orchestration)
    │
    ├── pecos-simulators
    │   ├── StateVec (state vector simulator)
    │   ├── SparseStab (stabilizer simulator)
    │   └── CliffordGateable, ArbitraryRotationGateable traits
    │
    ├── pecos-core
    │   ├── QubitId (qubit identification)
    │   ├── GateType, Gate (gate definitions)
    │   ├── derive_seed() (deterministic seed derivation)
    │   └── PecosError (error handling)
    │
    ├── pecos-random
    │   └── PecosRng (parallel-safe RNG)
    │
    └── byte_message/ (internal module)
        ├── message.rs (parsing/serialization)
        ├── protocol.rs (binary format definitions)
        └── builder.rs (message construction)

pecos-qis-ffi (C ABI for external programs)
    │
    ├── QIS-style exports (__quantum__qis__*)
    ├── Runtime functions (__quantum__rt__*)
    ├── Dynamic circuit support (___lazy_measure, etc.)
    └── ExecutionContext (thread-local isolation)

selene-plugins/ (simulator plugins)
    │
    ├── pecos-selene-statevec
    └── pecos-selene-stabilizer
```

## ByteMessage: Binary Protocol for FFI and Plugins

The `ByteMessage` protocol is a cornerstone of PECOS's extensibility. Beyond decoupling internal components, it enables Foreign Function Interface (FFI) support and a plugin architecture.

### Binary Format

ByteMessage uses a 4-byte aligned binary format stored in `Vec<u32>`:

```rust
pub struct ByteMessage {
    data: Vec<u32>,      // Binary format with 4-byte alignment
    byte_len: usize,     // Track actual byte length
}
```

**Message Structure:**
- **Batch Header (16 bytes):** Magic number (`0x50_45_43_53` = "PECS"), protocol version, flags, message count, total size
- **Per-Message:** Message header (8 bytes) + payload
- **Payload:** Gate operations with encoded qubit indices and floating-point parameters
- **Alignment:** All boundaries padded to 4-byte alignment for FFI safety using `bytemuck`

### FFI Support (pecos-qis-ffi)

The `pecos-qis-ffi` crate exports C ABI functions following QIS (Quantum Instruction Set) standards:

```rust
// Gate operations exported with #[no_mangle] extern "C"
__quantum__qis__h__body(qubit: *mut Qubit)
__quantum__qis__cx__body(control: *mut Qubit, target: *mut Qubit)
__quantum__qis__rz__body(theta: f64, qubit: *mut Qubit)
__quantum__qis__mz__body(qubit: *mut Qubit, result: *mut Result)

// Runtime functions
__quantum__rt__qubit_allocate() -> *mut Qubit
__quantum__rt__qubit_release(qubit: *mut Qubit)
```

**Dynamic Circuit Support:**

For mid-circuit measurement with classical feedback across FFI:

```rust
// Lazy measurement returns a future ID
___lazy_measure(qubit: i64) -> i64

// Blocks until measurement result is available
___read_future_bool(future_id: i64) -> bool

// Control dynamic execution mode
pecos_enable_dynamic_mode()
pecos_disable_dynamic_mode()
```

Thread-local `ExecutionContext` enables per-execution isolation for parallel Monte Carlo simulations:

```rust
pub struct ExecutionContext {
    pub dynamic_mode_active: AtomicBool,
    pub waiting_for_result: AtomicU64,
    pub sync_state: Mutex<DynamicSyncState>,
    pub measurement_results: Mutex<BTreeMap<u64, bool>>,
}
```

### Plugin Architecture (selene-plugins)

Plugins implement the `SimulatorInterface` trait:

```rust
pub trait SimulatorInterface {
    fn shot_start(&mut self, shot_id: u64, seed: u64) -> Result<()>;
    fn shot_end(&mut self) -> Result<()>;
    fn rxy(&mut self, qubit: u64, theta: f64, phi: f64) -> Result<()>;
    fn rz(&mut self, qubit: u64, theta: f64) -> Result<()>;
    fn czz(&mut self, q1: u64, q2: u64) -> Result<()>;
    fn measure(&mut self, qubit: u64) -> Result<bool>;
    // ... additional gate methods
}
```

**Available Plugins:**
- `pecos-selene-statevec` - State vector simulator
- `pecos-selene-stabilizer` - Stabilizer simulator

### Python Bindings

ByteMessage is exposed to Python via `pecos-rslib`:

```python
from pecos import ByteMessage

# Build a message
builder = ByteMessage.quantum_operations_builder()
builder.h([0])
builder.cx([(0, 1)])
builder.mz([0])
message = builder.build()

# Parse operations
gates = message.parse_quantum_operations()  # Returns list of dicts
raw = message.as_bytes()  # Raw binary for network/storage
```

### Design Benefits

The ByteMessage protocol provides:

- **Decouples components** - Classical and quantum engines don't need to know about each other's internals
- **Enables batching** - Multiple operations sent in one message
- **FFI-safe** - 4-byte alignment and simple binary format works across language boundaries
- **Plugin extensibility** - New simulators can be added without modifying core code
- **Network-ready** - Messages can be serialized for distributed simulation
- **Python integration** - Full access to simulation infrastructure from Python

## Key Design Decisions

### Why ControlEngine Pattern?

The `ControlEngine` pattern (start/continue_processing) enables:
- **Feedback loops** - Essential for mid-circuit measurements
- **Lazy evaluation** - Only generate commands as needed
- **State management** - Controller maintains state across rounds
- **Composability** - Controllers can wrap other engines (e.g., NoiseModel wraps QuantumEngine)

### Why Clone-per-Worker?

Each worker gets a clone of the `HybridEngine`:
- **Thread safety** - No shared mutable state between workers
- **Independence** - Workers can't interfere with each other
- **Simplicity** - No complex synchronization needed
- **Reproducibility** - Each worker has deterministic behavior

## Example: QEC Simulation Flow

Here's how the architecture supports a QEC simulation:

```
1. Classical engine: Generate data qubit initialization
   → ByteMessage: [H(d0), H(d1), ...]

2. Quantum system: Execute initialization
   → ByteMessage: [] (no measurements)

3. Classical engine: Generate syndrome extraction circuit
   → ByteMessage: [CX(d0,a0), CX(d1,a0), ..., MZ(a0), MZ(a1), ...]

4. Quantum system: Execute, return syndrome measurements
   → ByteMessage: [MZ(a0)=1, MZ(a1)=0, ...]

5. Classical engine: Decode syndrome, generate corrections
   → ByteMessage: [X(d0), Z(d2)]  // Corrections based on decoder

6. Quantum system: Apply corrections
   → ByteMessage: []

7. Classical engine: Generate next round or final measurements
   → ...

8. Complete: Return Shot with logical measurement results
```

The feedback loop is essential here - the corrections depend on the syndrome measurements.

## Python Extensibility

PECOS is designed for Python users to write custom components while leveraging Rust performance for the heavy lifting.

### Protocol-Based Architecture

Python components implement Protocol classes (structural typing):

```python
from __future__ import annotations

from typing import Any, Callable, Protocol

BitArray = Any  # placeholder for bit-array type
Correction = Any  # placeholder for correction type


class MachineProtocol(Protocol):
    """Interface for hardware models (connectivity, leakage, etc.)."""

    leaked_qubits: set[int]
    lost_qubits: set[int]

    def process(self, op_buffer: list) -> list: ...


class ErrorModelProtocol(Protocol):
    """Interface for custom error/noise models."""

    error_params: dict

    def init(self, num_qubits: int, machine: MachineProtocol | None = None) -> None: ...
    def process(self, qops: list, call_back: Callable | None = None) -> list | None: ...
    def reset(self) -> None: ...


class Decoder(Protocol):
    """Interface for QEC decoders."""

    def decode(self, syndrome: BitArray) -> Correction: ...
```

### Writing Custom Components in Python

Users can implement any protocol in pure Python:

```python
import random


class MyCustomErrorModel:
    """Custom error model - just implement the protocol methods."""

    def __init__(self, error_rate: float):
        self.error_params = {"p": error_rate}
        self.num_qubits = None

    def init(self, num_qubits: int, machine=None) -> None:
        self.num_qubits = num_qubits

    def process(self, qops: list, call_back=None) -> list:
        noisy_ops = []
        for op in qops:
            noisy_ops.append(op)
            if random.random() < self.error_params["p"]:
                noisy_ops.append(("X", op))  # simplified noise
        return noisy_ops

    def reset(self) -> None:
        pass


model = MyCustomErrorModel(0.01)
model.init(num_qubits=5)
```

Usage with the hybrid engine:

```hidden-python
import random

from pecos.engines import HybridEngine


class MyCustomErrorModel:
    def __init__(self, error_rate):
        self.error_params = {"p": error_rate}
        self.num_qubits = None

    def init(self, num_qubits, machine=None):
        self.num_qubits = num_qubits

    def process(self, qops, call_back=None):
        noisy_ops = []
        for op in qops:
            noisy_ops.append(op)
        return noisy_ops

    def reset(self):
        pass

    def shot_reinit(self):
        pass


program = {
    "format": "PHIR/JSON",
    "version": "0.1.0",
    "metadata": {"num_qubits": 2},
    "ops": [
        {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
        {"data": "cvar_define", "data_type": "i32", "variable": "m", "size": 2},
        {"qop": "H", "args": [["q", 0]]},
        {"qop": "CX", "args": [[["q", 0], ["q", 1]]]},
        {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
        {"qop": "Measure", "args": [["q", 1]], "returns": [["m", 1]]},
    ],
}
```

```python
from pecos.engines import HybridEngine

# Use with HybridEngine
engine = HybridEngine(
    error_model=MyCustomErrorModel(0.01),  # Python error model (flexible)
)
results = engine.run(program, shots=100)
```

### Two-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Python Layer                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Python HybridEngine                     │    │
│  │  - Orchestrates Python-defined components            │    │
│  │  - Custom ErrorModel, Machine, Decoder in Python     │    │
│  │  - Flexible experimentation                          │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         │                                    │
│  ┌──────────────────────┴──────────────────────────────┐    │
│  │              PyO3 Bindings (pecos-rslib)             │    │
│  │  - SparseStab, StateVec exposed to Python             │    │
│  │  - WasmForeignObject for classical co-processors     │    │
│  │  - Engine builders for Rust-native pipelines         │    │
│  └──────────────────────┬──────────────────────────────┘    │
└─────────────────────────┼───────────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────────┐
│                    Rust Layer                                │
│  ┌──────────────────────┴──────────────────────────────┐    │
│  │              Rust HybridEngine                       │    │
│  │  - High-performance orchestration                    │    │
│  │  - ByteMessage protocol                              │    │
│  │  - Parallel Monte Carlo                              │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Rust Simulators                         │    │
│  │  - SparseStab (stabilizer)                           │    │
│  │  - StateVec (state vector)                           │    │
│  │  - GPU backends                                      │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### Use Cases

| Scenario | Approach |
|----------|----------|
| **Prototyping new error model** | Write in Python, use Rust simulator |
| **Custom QEC decoder** | Python decoder with Rust stabilizer sim |
| **Production simulation** | Full Rust pipeline via `sim()` API |
| **Research flexibility** | Mix Python and Rust components freely |
| **Classical co-processing** | WASM modules via WasmForeignObject |

### Foreign Objects (Classical Co-Processors)

For computationally intensive classical logic (decoders, lookup tables), PECOS supports WASM:

```python
import tempfile
from pathlib import Path

from pecos import WasmForeignObject

# Create a minimal WASM module with a decode function
wat = """(module
  (func $init)
  (func $decode_syndrome (param i32) (result i32) (local.get 0))
  (memory (;0;) 1)
  (export "init" (func $init))
  (export "decode_syndrome" (func $decode_syndrome))
  (export "memory" (memory 0))
)"""
wat_path = Path(tempfile.mktemp(suffix=".wat"))
wat_path.write_text(wat)

# Load and use
decoder_wasm = WasmForeignObject.from_file(str(wat_path))
decoder_wasm.init()
result = decoder_wasm.exec("decode_syndrome", [42])
```

This allows writing performance-critical classical code in Rust/C/C++ compiled to WASM, while keeping the orchestration in Python.

## Summary

The `pecos-engines` architecture provides:

- **Modularity** - Swap simulators, noise models, or program formats independently
- **Composability** - Engines delegate to other engines via well-defined interfaces
- **Parallelism** - Automatic multi-threaded shot execution
- **Reproducibility** - Deterministic seed derivation
- **Flexibility** - Support for adaptive circuits via classical-quantum feedback
- **Extensibility** - FFI support and plugin architecture via ByteMessage protocol
- **Cross-language** - Python bindings and C ABI exports for external integration

The key insights are:
1. The `EngineStage` pattern enables feedback loops between classical and quantum components, making mid-circuit measurements and classical control possible
2. The `ByteMessage` binary protocol provides a clean FFI boundary, enabling plugins, Python integration, and potential distributed simulation
3. The two-layer architecture (Python + Rust) allows users to prototype custom components in Python while leveraging Rust performance for simulation
