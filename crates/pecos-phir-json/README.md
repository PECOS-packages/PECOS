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

```rust
use pecos_phir_json::PhirJsonEngine;
use pecos_engines::core::shot_results::OutputFormat;
use std::path::Path;

// Load a PHIR program from a file (v0.1 implementation)
let engine = PhirJsonEngine::new(Path::new("examples/bell.phir.json"))?;

// Process the program
let results = engine.process(())?;

// Format the results
let formatted_results = engine.get_formatted_results(OutputFormat::PrettyJson)?;
println!("{}", formatted_results);
```

### Using with Automatic Version Detection

```rust
use pecos_phir_json::setup_phir_json_engine;
use pecos_engines::{MonteCarloEngine, engines::noise::DepolarizingNoiseModel};
use std::path::Path;

// Create a classical engine from a PHIR program file
// The version will be automatically detected from the file
let classical_engine = setup_phir_json_engine(Path::new("examples/bell.phir.json"))?;

// Run the program with a noise model
let noise_model = Box::new(DepolarizingNoiseModel::new_uniform(0.01));
let results = MonteCarloEngine::run_with_noise_model(
    classical_engine,
    noise_model,
    100, // shots
    2,   // workers
    None // seed
)?;

println!("{}", results);
```

### Explicit Version Selection

```rust
// For specific version implementations
use pecos_phir_json::setup_phir_json_v0_1_engine;
use std::path::Path;

// Explicitly use v0.1 implementation
let engine = setup_phir_json_v0_1_engine(Path::new("examples/bell.phir.json"))?;
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

For testing PHIR programs, you can use the `run_phir_simulation_from_json` helper function to run a simulation directly from a JSON string:

```rust
use pecos_core::errors::PecosError;
use pecos_engines::PassThroughNoiseModel;

// Import helpers from common module
use crate::common::phir_test_utils::run_phir_simulation_from_json;

#[test]
fn test_bell_state_with_inline_json() -> Result<(), PecosError> {
    // Define the Bell state PHIR program directly in the test
    let phir_json = r#"{
      "format": "PHIR/JSON",
      "version": "0.1.0",
      "metadata": {"description": "Bell state preparation"},
      "ops": [
        {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
        {"data": "cvar_define", "data_type": "i32", "variable": "m", "size": 2},
        {"qop": "H", "args": [["q", 0]]},
        {"qop": "CX", "args": [["q", 0], ["q", 1]]},
        {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
        {"qop": "Measure", "args": [["q", 1]], "returns": [["m", 1]]},
        {"cop": "Result", "args": ["m"], "returns": ["output"]}
      ]
    }"#;

    // Run with a single shot and no noise using the full simulation pipeline
    let results = run_phir_simulation_from_json(
        phir_json,
        1,  // shots
        1,  // workers
        None,  // No specific seed
        None::<PassThroughNoiseModel>,  // No noise model
    )?;

    // Process the results...
    Ok(())
}
```

This approach makes tests more readable and maintainable by keeping the test data and verification code together in one place.

> **Note**: Work is currently in progress to extend the PhirJsonEngine to support the full PHIR specification. Some
> advanced features may not be fully implemented yet. The specification itself is also evolving - the "Result"
> command for exporting measurement results is being added as part of a v0.1.1 specification update.

## Supported Operations

### Quantum Operations

- Single-qubit gates: `H`, `X`, `Y`, `Z`
- Rotations: `RZ`, `R1XY`
- Two-qubit gates: `CX` (CNOT), `SZZ` (ZZ interaction)
- Measurement: `Measure`

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

```rust
use pecos_phir_json::phir_json_to_module;

// Convert PHIR-JSON string directly to PHIR Module
let json_str = r#"{
    "format": "PHIR/JSON",
    "version": "0.1.0",
    "ops": [...]
}"#;

let module = phir_json_to_module(json_str)?;
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
