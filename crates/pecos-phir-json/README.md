# PECOS High-level Intermediate Representation JSON Format (PHIR-JSON)

This crate provides parsing and execution capabilities for PHIR-JSON, the JSON serialization format for the PECOS High-level Intermediate Representation (PHIR), used for representing quantum programs in the PECOS quantum simulator framework.

## Overview

PHIR-JSON is designed to:

- Provide a human-readable representation of quantum circuits
- Support a mix of quantum and classical operations
- Allow for deterministic execution of quantum programs
- Serve as an intermediate layer between high-level languages and lower-level simulators

## Usage

### Basic Example

`PhirJsonEngine` is a classical controller. Standalone `process(())` executes
programs that generate no quantum commands; it returns an error as soon as a
non-empty command batch is produced. Quantum programs need
`start()`/`continue_processing()` with a quantum engine, or the simulation builder:

```rust
use pecos_engines::{ClassicalControlEngineBuilder, PecosError, StateVectorEngineBuilder};
use pecos_phir_json::phir_json_engine;
use std::collections::BTreeSet;

fn main() -> Result<(), PecosError> {
    let bell = r#"{
      "format": "PHIR/JSON",
      "version": "0.1.0",
      "ops": [
        {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
        {"data": "cvar_define", "data_type": "i32", "variable": "m", "size": 2},
        {"qop": "H", "args": [["q", 0]]},
        {"qop": "CX", "args": [["q", 0], ["q", 1]]},
        {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
        {"qop": "Measure", "args": [["q", 1]], "returns": [["m", 1]]},
        {"cop": "Result", "args": ["m"], "returns": ["c"]}
      ]
    }"#;

    let results = phir_json_engine()
        .json(bell)?
        .to_sim()
        .quantum(StateVectorEngineBuilder::default())
        .seed(42)
        .run(100)?;

    let outcomes: BTreeSet<_> = results
        .shots
        .iter()
        .map(|shot| shot.data["c"].as_u32().unwrap())
        .collect();
    assert_eq!(outcomes, BTreeSet::from([0, 3]));
    println!("{}", results.to_compact_json());
    Ok(())
}
```

### Using with Automatic Version Detection

```rust
use pecos_engines::{DepolarizingNoiseModel, MonteCarloEngine, PecosError};
use pecos_phir_json::setup_phir_json_engine;
use std::path::Path;

fn main() -> Result<(), PecosError> {
    // Create a classical engine from a PHIR program file; the version is detected
    // from the file. The path is relative to this crate's directory.
    let classical_engine =
        setup_phir_json_engine(Path::new("../../examples/phir/bell.phir.json"))?;

    // Run the program with a noise model
    let noise_model = Box::new(DepolarizingNoiseModel::new_uniform(0.01));
    let results = MonteCarloEngine::run_with_noise_model(
        classical_engine,
        noise_model,
        100,      // shots
        2,        // workers
        Some(42), // seed
    )?;
    assert_eq!(results.shots.len(), 100);

    println!("{results}");
    Ok(())
}
```

### Explicit Version Selection

```rust
use pecos_engines::PecosError;
use pecos_phir_json::setup_phir_json_v0_1_engine;
use std::path::Path;

fn main() -> Result<(), PecosError> {
    // Explicitly use the v0.1 implementation
    let engine = setup_phir_json_v0_1_engine(Path::new("../../examples/phir/bell.phir.json"))?;
    assert_eq!(engine.num_qubits(), 2);
    Ok(())
}
```

## PHIR File Format

PHIR files are JSON documents with the following structure:

```json
{
  "format": "PHIR/JSON",
  "version": "0.1.0",
  "metadata": {
    "description": "Example PHIR program"
  },
  "ops": [
    {
      "data": "qvar_define",
      "data_type": "qubits",
      "variable": "q",
      "size": 2
    },
    {
      "data": "cvar_define",
      "data_type": "i64",
      "variable": "m",
      "size": 2
    },
    {"qop": "H", "args": [["q", 0]]},
    {"qop": "CX", "args": [["q", 0], ["q", 1]]},
    {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
    {"qop": "Measure", "args": [["q", 1]], "returns": [["m", 1]]},
    {"cop": "Result", "args": ["m"], "returns": ["c"]}
  ]
}
```

See the [specification](specification/v0.1/spec.md) for more details.

## Validation and Execution

This crate provides:

1. **Validation**: Rust-based parsing and validation of PHIR programs against the specification
2. **Execution**: Full integration with PECOS for running PHIR programs on quantum simulators
3. **Error Handling**: Detailed error messages for both validation and runtime errors

For alternative validation, the [Python Pydantic PHIR validator](https://github.com/Quantinuum/phir) is also available.

### Testing with Inline JSON

A program can be passed as a JSON string with `phir_json_engine().json(...)`, as in the
[Basic Example](#basic-example), which keeps the test data and verification code together in
one place. This crate's own integration tests also use the `run_phir_simulation_from_json`
helper in `tests/common/phir_test_utils.rs`, which adds noise-model and worker options.

> **Note**: Work is currently in progress to extend the PhirJsonEngine to support the full PHIR specification. Some
> advanced features may not be fully implemented yet. The specification itself is also evolving - the "Result"
> command for exporting measurement results is being added as part of a v0.1.1 specification update.

## Supported Operations

### Quantum Operations

The native engine and JSON-to-module converter support every quantum operation
and alias in [PHIR v0.1 Table II](specification/v0.1/spec.md#table-ii---quantum-operations):

- Single-qubit gates: `I`, `X`, `Y`, `Z`, `H`, `SX`, `SXdg`, `SY`, `SYdg`,
  `SZ` (`S`), `SZdg` (`Sdg`), `F`, `Fdg`, `T`, `Tdg`
- Single-qubit rotations: `RX`, `RY`, `RZ`, `R1XY` (`U1q`)
- Two-qubit gates: `CX` (`CNOT`), `CY`, `CZ`, `SXX`, `SXXdg`, `SYY`, `SYYdg`,
  `SZZ` (`ZZ`, `ZZMax`), `SZZdg`, `SWAP`
- Two-qubit rotations: `RXX`, `RYY`, `RZZ` (`ZZPhase`), `R2XXYYZZ` (`RXXYYZZ`)
- Preparation and measurement: `Init`, `Measure`

Each operation broadcasts in argument order over its qubits or pairs. Qubit IDs
must be unique within an operation, and its angle count must match Table II.
Angles use `rad`, `pi`, or `deg`; every broadcast target receives the same angles.
Compatibility spellings `Prep`, `RXY1Q`, `RXXRYYRZZ` and the extensions `U` and
`RXYXY2Q` are also available. Gates use the existing exact byte-message representation;
backend support still determines which simulators can execute a given gate.

### Classical Operations

- Variable operations: `=` (assignment), arithmetic (+, -, *, /, etc.), comparisons (==, !=, <, >, etc.)
- Control flow: Conditional execution with `if` blocks
- Foreign function calls: `ffcall` for calling WebAssembly functions
- Export: `Result` for exporting measurement results

### Machine Operations

- `Idle`: Specify qubits to idle for a specific duration
- `Delay`: Insert a specific delay for qubits
- `Transport`: Move qubits from one location to another
- `Timing`: Synchronize operations in time
- `Reset`: Reset qubits to |0⟩ state
- `Skip`: No-op placeholder

See [Machine Operations Documentation](src/v0_1/README.md) for more details.

## Versioning

This crate implements a versioning strategy to handle multiple versions of the PHIR specification. See
[VERSIONING.md](VERSIONING.md) for details on how versions are managed.

### Available Versions

- **v0.1**: The initial version, supporting basic quantum operations, variable definitions, and classical exports.
  - Specification: [specification/v0.1/spec.md](specification/v0.1/spec.md)
  - Feature flag: `v0_1` (enabled by default)

### Feature Flags

You can control which PHIR versions are included in your build using Cargo feature flags:

```toml
# Default: only include v0.1
pecos-phir-json = { version = "0.1" }

# Explicitly select a specific version
pecos-phir-json = { version = "0.1", default-features = false, features = ["v0_1"] }

# Include all available versions
pecos-phir-json = { version = "0.1", features = ["all-versions"] }
```

## Conversion Architecture

This crate provides a streamlined conversion architecture:

- **PHIR-JSON → PHIR Module**: Direct conversion from JSON to native PHIR Module structures
- **PHIR Module ↔ PHIR-RON**: Bidirectional serialization for debugging and persistence

The conversion paths are:
1. **Input**: PHIR-JSON (human-readable JSON format) → PHIR Module (in-memory representation)
2. **Debug/Export**: PHIR Module → PHIR-RON (Rusty Object Notation for inspection)

### Converting PHIR-JSON to PHIR Module

Broadcast quantum operations become one instruction per qubit or qubit pair,
including preparations and measurements. Each instruction receives the same
rotation angles; measurement destinations retain their argument order.
The converter uses the native angle decoder (`rad`, `pi`, and `deg`).
Two-qubit argument groups must contain exactly two qubits in the native engine,
Rust and Python classical interpreters and converter. They share validation of
group shapes, qubit uniqueness and gate signatures. RXYXY2Q requires paired
arguments; its gate validation runs during execution/conversion, after structural
AST decoding.

JSON-to-module conversion supports register declarations, quantum operations
and Result exports with whole-register names or indexed `[name, bit]` references.
Result args and returns must have matching nonzero lengths. Each pair becomes a
named Result instruction; copies retain their source-order values even when
subsequent measurements overwrite a register. Indexed sources extract one bit;
indexed destinations replace one bit
without losing the others. Result cannot overwrite quantum registers or read
unknown sources. Measurement destinations must be declared classical registers,
and their bit indices must be below the declared size (excluding the sign bit).
All classical storage types (`i8`–`i64`, `u8`–`u64`, and `bool`) support
measurements. Declaration sizes use one shared Rust decoder: only an absent
field means omitted. JSON null, negative, string, fractional and boolean size
values are rejected with a named input error. Classical sizes are inferred when
omitted; quantum declarations still require a positive size. Bool has one-bit
backing storage: use size 1 or omit size; declarations with size greater than 1
are rejected on every path. Assignment and
ffcall destinations also require declared classical registers, with indexed
writes below the declared size. Whole signed Result copies sign-extend to i64
before destination-width masking, independently of the source backing type,
matching `=`. Malformed Result entries report the offending JSON value; Python
uses TypeError for wrong types and ValueError for invalid integer indices.
Rust classifies these as `PecosError::Input`. The Python interpreter validates
Result reference syntax and declarations at load. It checks Result counts when
that Result executes, like native and the Rust interpreter, and caches successful
reference decoding until the next init. Later shots do not repeat JSON/FFI
validation. Measurement destinations are checked against declarations when results
arrive. Shot resets do not repeat declaration checks. PyPHIR's classical-input
validation does not serialize unrelated metadata in parsed Python programs.
Exports use U32 through 32 bits and U64 for wider registers.
Without Result mappings, all declared classical registers are exported.
CPhase is not a PHIR v0.1 gate and must be lowered before emitting PHIR JSON.

Known limitation ([#1323](https://github.com/PECOS-packages/PECOS/issues/1323)):
`cvar_export` remains unsupported. The native engine currently ignores it and
exports all classical registers unless Result mappings select them. Lowering it
to module Result instructions would select different registers.

The converter rejects blocks such as `qparallel` and `if`, machine/meta operations,
unsupported data/classical operations and unrecognized operations with a named
error. Use the native PHIR-JSON engine for supported PHIR programs containing
these constructs. Although the module IR has structural parallel and conditional
forms, its current execution path does not execute
those regions; the converter does not emit programs that would lose their work.
PHIR comments carry no execution semantics and are ignored.

The legacy Python `pecos_rslib.PhirJsonEngine` (also exported through
`pecos.engines`) delegates command generation, register updates, exports, and
execution state to this native engine. It no longer invokes a Python interpreter
fallback or changes outcomes when pytest is installed. `process_program()` returns
the next batch as command dictionaries. Supply every measurement outcome in order
using `handle_measurement(0)` or `handle_measurement(1)` before requesting the next
batch. `get_results()` and `results_dict` expose native results as u32 dictionaries;
values outside that representation raise `ValueError` instead of being truncated.
Use the simulation results API for wider values. The compatibility constructor
`create_with_validation_disabled()` bypasses no native validation; neither
constructor uses the Python schema validator.

PyPHIR also interprets `rad` unchanged, `pi` as multiples of pi, and `deg` as
multiples of pi/180. Unknown unit strings raise `ValueError`; nonstring units raise
`TypeError`.

```rust
use pecos_engines::PecosError;
use pecos_phir_json::phir_json_to_module;

fn main() -> Result<(), PecosError> {
    // Convert PHIR-JSON string directly to PHIR Module
    let json_str = r#"{
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 1},
            {"qop": "H", "args": [["q", 0]]}
        ]
    }"#;

    let module = phir_json_to_module(json_str)?;
    println!("{module:?}");
    Ok(())
}
```

### Example: PHIR-JSON to Module Converter

The crate includes an example tool for converting PHIR-JSON files:

```bash
# Convert and display module info
cargo run --example phir_json_to_module input.phir.json

# Convert and export to PHIR-RON for debugging
cargo run --example phir_json_to_module input.phir.json output.ron
```

## License

This crate is licensed under the Apache License, Version 2.0.
