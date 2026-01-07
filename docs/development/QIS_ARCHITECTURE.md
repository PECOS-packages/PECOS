# QIS Architecture: Interface, Runtime, and Engine

This document describes the architecture of the Quantum Instruction Set (QIS) system in PECOS, focusing on how quantum programs are compiled, executed, and simulated.

## Overview

The QIS architecture consists of three main components:

1. **Interface Layer** - Compiles quantum programs and collects operations
2. **Runtime Layer** - Executes collected quantum operations
3. **Engine Layer** - Orchestrates interface and runtime

```
┌─────────────────────────────────────────────────────────────┐
│                         QisEngine                           │
│                       (pecos-qis)                           │
│                                                             │
│  ┌─────────────────────┐      ┌──────────────────────┐      │
│  │   QisInterface      │      │     QisRuntime       │      │
│  │  (Interface Impl)   │──────│   (Runtime Impl)     │      │
│  └─────────────────────┘      └──────────────────────┘      │
│           │                              │                  │
└───────────┼──────────────────────────────┼──────────────────┘
            │                              │
            ▼                              ▼
    Compile & Collect                Execute Operations
      Operations                    (Quantum Simulation)
```

## 1. Interface Architecture

The **Interface Layer** is responsible for taking a quantum program (in various formats) and extracting the quantum operations from it.

### Interface Trait

Defined in `pecos-qis/src/qis_interface.rs`:

```rust
pub trait QisInterface {
    /// Load a quantum program
    fn load_program(&mut self, program_bytes: &[u8], format: ProgramFormat)
        -> Result<(), InterfaceError>;

    /// Collect operations from the loaded program
    fn collect_operations(&mut self) -> Result<OperationCollector, InterfaceError>;

    /// Execute with pre-set measurement results (for conditional operations)
    fn execute_with_measurements(&mut self, measurements: HashMap<usize, bool>)
        -> Result<OperationCollector, InterfaceError>;

    /// Get interface metadata
    fn metadata(&self) -> HashMap<String, String>;

    /// Interface name
    fn name(&self) -> &'static str;

    /// Reset the interface state
    fn reset(&mut self) -> Result<(), InterfaceError>;
}
```

### Helios Interface Implementation

The **Helios Interface** (`QisHeliosInterface` in `pecos-qis`) is the primary interface implementation. It works by:

1. **Compilation**: Linking quantum program bitcode with Selene's Helios library
2. **Dynamic Execution**: Loading and executing the compiled program in-process
3. **Operation Collection**: Capturing quantum operations via FFI interception

#### Helios Interface Flow

```
User provides QIS bitcode/LLVM IR
         ↓
QisHeliosInterface.load_program()
         ↓
    Compile with clang:
    program.bc + libhelios.a → program.so
         ↓
QisHeliosInterface.collect_operations()
         ↓
    Load libraries with RTLD_GLOBAL:
    1. libpecos_qis_ffi.so (provides __quantum__rt__*)
    2. libpecos_selene.so (provides selene_*)
    3. program.so (calls selene_*)
         ↓
    Execute: qmain() or main()
         ↓
    Collect operations from thread-local storage
         ↓
    Return OperationCollector
```

### Symbol Resolution Chain

When a quantum program executes, function calls are resolved through multiple layers:

```
program.so: qmain()
  ↓ calls ___qalloc()

libhelios.a (linked into program.so)
  ↓ calls selene_qalloc()

libpecos_selene.so (C shim, loaded with RTLD_GLOBAL)
  │ File: pecos-qis/src/c/selene_shim.c
  │ Purpose: Adapts Selene interface to PECOS FFI
  ↓ calls __quantum__rt__qubit_allocate()

libpecos_qis_ffi.so (Rust cdylib, loaded with RTLD_GLOBAL)
  │ Crate: pecos-qis-ffi
  │ Purpose: Provides QIS FFI functions
  ↓ records operation

OperationCollector (thread-local storage)
  │ Records: AllocateQubit, H, CX, Measure, etc.
  ↓ retrieved by

QisHeliosInterface
  │ Returns operations to QisEngine
```

### The Shim Layer (libpecos_selene.so)

**Purpose**: Bridges Selene's C interface to PECOS Rust FFI

**Location**: Built by `pecos-qis/build_selene.rs` from `src/c/selene_shim.c`

**Example** (from `selene_shim.c`):
```c
selene_u64_result_t selene_qalloc(SeleneInstance *instance) {
    (void)instance;  // Unused - we use thread-local storage
    int64_t qubit_id = __quantum__rt__qubit_allocate();
    return SUCCESS_VAL(selene_u64_result_t, (uint64_t)qubit_id);
}

selene_void_result_t selene_rxy(SeleneInstance *instance,
                                  uint64_t q, double theta, double phi) {
    (void)instance;
    __quantum__qis__r1xy__body(theta, phi, (int64_t)q);
    return SUCCESS(selene_void_result_t);
}
```

**Why it exists**: Selene's Helios compiler expects functions with specific signatures (e.g., `selene_qalloc`). The shim provides these functions and forwards calls to our Rust FFI layer.

### The FFI Layer (libpecos_qis_ffi.so)

**Purpose**: Provides `__quantum__rt__*` and `__quantum__qis__*` symbols that record operations

**Crate**: `pecos-qis-ffi`

**Example** (from `pecos-qis-ffi/src/ffi.rs`):
```rust
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__rt__qubit_allocate() -> i64 {
    with_interface(|interface| {
        let id = interface.allocate_qubit();
        interface.queue_operation(Operation::AllocateQubit { id });
        i64::try_from(id).expect("Qubit ID too large for i64")
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__h__body(qubit: i64) {
    let qubit_id = i64_to_usize(qubit);
    with_interface(|interface| {
        interface.queue_operation(QuantumOp::H(qubit_id).into());
    });
}
```

**Thread-local storage**: Operations are collected in thread-local `OperationCollector` that can be retrieved after execution.

### Operation Collector

The `OperationCollector` (in `pecos-qis-ffi`) stores:

```rust
pub struct OperationCollector {
    /// Allocated qubit IDs
    pub allocated_qubits: Vec<usize>,

    /// Allocated result IDs
    pub allocated_results: Vec<usize>,

    /// Sequence of quantum operations
    pub operations: Vec<Operation>,

    /// Measurement results (for conditional execution)
    measurement_results: HashMap<usize, bool>,
}
```

Operations include:
- `AllocateQubit`, `ReleaseQubit`
- `AllocateResult`
- Quantum gates: `H`, `X`, `Y`, `Z`, `S`, `T`, `CX`, `CY`, `CZ`, etc.
- Rotations: `RX`, `RY`, `RZ`, `RXY`, `RZZ`, etc.
- Measurements: `Measure`, `Reset`

## 2. Runtime Architecture

The **Runtime Layer** takes collected quantum operations and executes them using a quantum simulator.

### Runtime Trait

Defined in `pecos-qis/src/runtime.rs`:

```rust
pub trait QisRuntime: Send + Sync + DynClone {
    /// Execute quantum operations and return results
    fn execute(&mut self, operations: &OperationCollector)
        -> Result<RuntimeResult, RuntimeError>;

    /// Runtime name
    fn name(&self) -> &'static str;

    /// Clone the runtime
    fn clone_box(&self) -> Box<dyn QisRuntime>;
}
```

### Selene Runtime Implementation

The **Selene Runtime** wraps Selene's quantum simulator library (.so files).

**Location**: `pecos-qis/src/selene_runtime.rs`

#### Selene Runtime Types

Selene provides multiple runtime variants (all are .so files):

1. **Simple Runtime** (`libselene_simple_runtime.so`):
   - State vector simulation
   - Full quantum state tracking
   - Function: `selene_simple_runtime()?`

2. **Soft-Rz Runtime** (`libselene_soft_rz_runtime.so`):
   - Optimized for Rz-heavy circuits
   - Function: `selene_soft_rz_runtime()?`

#### Runtime Wrapper Structure

```rust
pub struct QisSeleneRuntime {
    /// Path to the Selene runtime .so file
    runtime_lib_path: PathBuf,

    /// Loaded runtime library
    runtime_lib: Option<Library>,

    /// Runtime metadata
    metadata: HashMap<String, String>,
}
```

#### Runtime Execution Flow

```
QisEngine calls runtime.execute(operations)
         ↓
QisSeleneRuntime.execute()
         ↓
    Load libselene_*_runtime.so
         ↓
    Initialize Selene instance
         ↓
    For each operation in OperationCollector:
        - Translate to Selene API call
        - Call runtime function via FFI
        - Track quantum state in Selene
         ↓
    Perform measurements (if any)
         ↓
    Extract results from Selene
         ↓
    Return RuntimeResult
```

#### Selene Runtime Functions

Selene runtimes expose functions like:

```c
// State management
SeleneInstance* selene_new_instance(void);
void selene_free_instance(SeleneInstance*);

// Qubit operations
selene_u64_result_t selene_qalloc(SeleneInstance*);
selene_void_result_t selene_qfree(SeleneInstance*, uint64_t qubit);

// Quantum gates
selene_void_result_t selene_rxy(SeleneInstance*, uint64_t q, double theta, double phi);
selene_void_result_t selene_rz(SeleneInstance*, uint64_t q, double theta);

// Measurements
selene_bool_result_t selene_qubit_measure(SeleneInstance*, uint64_t qubit);
```

The `QisSeleneRuntime` wrapper calls these functions via `libloading` FFI.

### Runtime Results

The `RuntimeResult` contains:

```rust
pub struct RuntimeResult {
    /// Measurement outcomes (result_id → bool)
    pub measurements: HashMap<usize, bool>,

    /// Runtime-specific metadata
    pub metadata: HashMap<String, String>,
}
```

## 3. Engine Architecture (QisEngine)

The **QisEngine** orchestrates the interface and runtime to provide a complete quantum program execution pipeline.

**Location**: `pecos-qis/src/ccengine.rs`

### QisEngine Structure

```rust
pub struct QisEngine {
    /// Interface implementation (e.g., QisHeliosInterface)
    interface: Box<dyn QisInterface>,

    /// Runtime implementation (e.g., QisSeleneRuntime)
    runtime: Box<dyn QisRuntime>,

    /// Number of qubits in the current program
    num_qubits: usize,

    /// Number of classical results
    num_results: usize,
}
```

### Engine Builder Pattern

Users construct a `QisEngine` using the builder pattern:

```rust
use pecos_qis::{qis_engine, helios_interface_builder, selene_simple_runtime};

let engine = qis_engine()
    .interface(helios_interface_builder())     // Set interface
    .runtime(selene_simple_runtime()?)         // Set runtime
    .program(qis_program)                      // Load program
    .build()?;                                 // Build engine
```

**Builder location**: `pecos-qis/src/engine_builder.rs`

### QisEngine Execution Flow

#### 1. Initialization (build time)

```rust
QisEngineBuilder::build()
    ↓
Interface: load_program(program_bytes)
    ↓ (compiles program)
Interface: collect_operations()
    ↓ (executes program, collects ops)
Store operations and metadata
    ↓
Return QisEngine
```

#### 2. Execution (run time)

```rust
engine.run(options)
    ↓
For each shot:
    ↓
    Runtime: execute(operations)
        ↓ (simulates quantum circuit)
        ↓ (performs measurements)
        ↓
    Return RuntimeResult
    ↓
Aggregate results across shots
    ↓
Return SimulationResult
```

### Engine Responsibilities

The `QisEngine` mediates between interface and runtime:

1. **Initialization**:
   - Uses interface to compile and collect operations
   - Stores program metadata (num_qubits, num_results)

2. **Execution**:
   - Passes operations to runtime for each shot
   - Handles multi-shot simulations
   - Aggregates measurement results

3. **Classical Control** (implements `ClassicalEngine` trait):
   - Supports conditional operations based on measurements
   - Manages measurement result storage
   - Enables dynamic circuit execution

## 4. Complete Example Flow

Let's trace a complete example: executing a Bell state program.

### Step 1: User Code

```rust
use pecos_qis::{qis_engine, helios_interface_builder, selene_simple_runtime};
use pecos_programs::Qis;
use pecos_engines::{ClassicalControlEngineBuilder, ClassicalEngine};

// Load Bell state program
let qis_program = Qis::from_file("bell.ll")?;

// Build engine
let mut engine = qis_engine()
    .interface(helios_interface_builder())
    .runtime(selene_simple_runtime()?)
    .program(qis_program)
    .build()?;

// Run simulation
let result = engine.run(&sim_options)?;
```

### Step 2: Interface Processing (during build)

```
QisEngineBuilder::build()
    ↓
QisHeliosInterface::load_program(bell.ll)
    ↓
    Compile: clang bell.ll + libhelios.a → bell.so
    Store: temp file bell.so
    ↓
QisHeliosInterface::collect_operations()
    ↓
    Load: libpecos_qis_ffi.so (RTLD_GLOBAL)
    Load: libpecos_selene.so (RTLD_GLOBAL)
    Load: bell.so
    ↓
    Execute: qmain(0)
        ↓ calls ___qalloc() [twice]
        ↓ calls ___h() [once on qubit 0]
        ↓ calls ___cx() [once: control=0, target=1]
    ↓
    Operations recorded in thread-local:
        - AllocateQubit { id: 0 }
        - AllocateQubit { id: 1 }
        - H(0)
        - CX(0, 1)
    ↓
    Return OperationCollector
    ↓
QisEngine stores:
    - operations: [AllocateQubit(0), AllocateQubit(1), H(0), CX(0,1)]
    - num_qubits: 2
```

### Step 3: Runtime Execution (during run)

```
engine.run(sim_options)
    ↓
For shot in 0..num_shots:
    ↓
    QisSeleneRuntime::execute(operations)
        ↓
        Load: libselene_simple_runtime.so
        Init: instance = selene_new_instance()
        ↓
        Process operations:
            AllocateQubit(0) → q0 = selene_qalloc(instance)
            AllocateQubit(1) → q1 = selene_qalloc(instance)
            H(0)             → selene_rxy(instance, q0, π, 0)
            CX(0, 1)         → (implemented via Rxy+Rz+Rxy+Rz)
        ↓
        Measurements (if any):
            Measure(0, 0) → result = selene_qubit_measure(instance, q0)
            Measure(1, 1) → result = selene_qubit_measure(instance, q1)
        ↓
        Cleanup: selene_free_instance(instance)
        ↓
        Return RuntimeResult {
            measurements: {0: false, 1: false} (or {0: true, 1: true})
        }
    ↓
Aggregate across shots:
    - Count: |00⟩ and |11⟩ states
    - Expected: ~50% each for Bell state
    ↓
Return SimulationResult
```

## 5. Architecture Benefits

This three-layer architecture provides:

### Separation of Concerns

- **Interface**: Handles program compilation and operation extraction
- **Runtime**: Handles quantum simulation
- **Engine**: Orchestrates and provides unified API

### Flexibility

- **Multiple Interfaces**: Can implement JIT, AOT, or other compilation strategies
- **Multiple Runtimes**: Can swap Selene for other simulators (QuEst, Qulacs, etc.)
- **Mix and Match**: Any interface can work with any runtime

### Extensibility

Adding a new interface:
```rust
pub struct MyCustomInterface { /* ... */ }

impl QisInterface for MyCustomInterface {
    fn load_program(&mut self, program: &[u8], format: ProgramFormat)
        -> Result<(), InterfaceError> {
        // Custom compilation logic
    }

    fn collect_operations(&mut self) -> Result<OperationCollector, InterfaceError> {
        // Custom operation collection
    }
    // ... other methods
}
```

Adding a new runtime:
```rust
pub struct MyCustomRuntime { /* ... */ }

impl QisRuntime for MyCustomRuntime {
    fn execute(&mut self, operations: &OperationCollector)
        -> Result<RuntimeResult, RuntimeError> {
        // Custom simulation logic
    }
    // ... other methods
}
```

### Testability

- Interface and runtime can be tested independently
- Mock implementations for unit testing
- Real implementations for integration testing

## 6. Key Design Decisions

### Why Dynamic Loading?

The Helios interface uses dynamic loading (`dlopen`/`libloading`) because:

1. **Symbol Resolution**: LLVM-compiled programs need `__quantum__rt__*` symbols available globally
2. **Flexibility**: Programs are compiled at runtime, not build time
3. **Interception**: We can intercept operations before they reach the simulator

### Why Thread-Local Storage?

Operation collection uses thread-local storage because:

1. **Simplicity**: No need to pass context through C FFI calls
2. **Safety**: Each thread has independent operation collector
3. **Performance**: Thread-local access is fast

### Why Separate Shim and FFI?

We have both `libpecos_selene.so` (C shim) and `libpecos_qis_ffi.so` (Rust FFI) because:

1. **Compatibility**: Helios expects specific C function signatures (`selene_*`)
2. **Type Safety**: Rust FFI provides safe operation collection
3. **Reusability**: FFI layer can be used by other interfaces, not just Helios

## 7. Crate Organization

```
pecos-qis/                    # Main QIS crate (with optional selene feature)
├── src/
│   ├── lib.rs                # Re-exports, prelude
│   ├── ccengine.rs           # QisEngine
│   ├── engine_builder.rs     # QisEngineBuilder
│   ├── qis_interface.rs      # QisInterface trait
│   ├── runtime.rs            # QisRuntime trait
│   ├── executor.rs           # QisHeliosInterface (selene feature)
│   ├── selene_runtime.rs     # SeleneRuntime (selene feature)
│   ├── selene_runtimes.rs    # Runtime discovery (selene feature)
│   ├── shim.rs               # Path to libpecos_selene.so (selene feature)
│   └── c/
│       └── selene_shim.c     # C shim implementation (selene feature)
├── build.rs                  # Main build script
├── build_selene.rs           # Selene build logic (selene feature)
└── Cargo.toml
│
pecos-qis-ffi/                # FFI layer (cdylib)
├── src/
│   ├── lib.rs                # OperationCollector, thread-local
│   ├── ffi.rs                # __quantum__rt__* and __quantum__qis__* exports
│   └── operations.rs         # Operation types
└── Cargo.toml                # crate-type = ["rlib", "cdylib"]
```

## 8. Future Directions

Potential extensions to this architecture:

1. **Additional Interfaces**:
   - JIT interface using LLVM Orc
   - Ahead-of-time (AOT) compiled interface
   - Direct QASM→operations interface

2. **Additional Runtimes**:
   - Native PECOS runtime (no Selene dependency)
   - GPU-accelerated runtime (QuEst, Qulacs)
   - Distributed runtime for large-scale simulation

3. **Optimizations**:
   - Operation fusion (combine multiple gates)
   - Circuit optimization passes
   - Lazy evaluation of operations

4. **Features**:
   - Noise models in runtime layer
   - State vector inspection
   - Intermediate measurements with classical control

## Summary

The QIS architecture provides a clean separation between:

- **Interface** (compilation & operation collection)
- **Runtime** (quantum simulation)
- **Engine** (orchestration & API)

This design enables flexibility, extensibility, and maintainability while supporting complex quantum program execution with features like conditional operations and multi-shot simulations.
