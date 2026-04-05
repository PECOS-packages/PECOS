# Engine Selection

This guide covers working with PECOS engines in both compile-time and runtime contexts.

Source: `crates/pecos/src/engine_type.rs`

PECOS provides two main tools for engine selection:

- `EngineType`: An enumeration of all available engine types
- `DynamicEngineBuilder`: A type-erased wrapper for runtime engine selection

## Overview

PECOS provides multiple classical control engines (QASM, LLVM, Selene) for
executing quantum programs. Normally, you work with these engines directly:

```rust
use pecos_qasm::qasm_engine;
use pecos_engines::sim;
use pecos_programs::Qasm;

// Compile-time engine selection - best performance
let qasm_code = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
h q[0];
measure q[0] -> c[0];
"#;
let results = sim(qasm_engine().program(Qasm::from_string(qasm_code)))
    .seed(42)
    .run(10)?;

// Verify results
assert_eq!(results.len(), 10);
let shot_map = results.try_as_shot_map().unwrap();
let values = shot_map.try_bits_as_u64("c").unwrap();
// H gate creates superposition, so we should see both 0 and 1
assert!(values.iter().any(|&v| v == 0) || values.iter().any(|&v| v == 1));
```

However, sometimes you need to select an engine at runtime based on user input,
configuration files, or other dynamic conditions. The `DynamicEngineBuilder`
provides the tools to do that.

## Dynamic Engine Selection

The `DynamicEngineBuilder` type uses trait objects to enable runtime engine
selection while maintaining the same API:

```rust
use pecos::{EngineType, DynamicEngineBuilder, sim_dynamic};
use pecos_qasm::qasm_engine;
use pecos_programs::Qasm;

// Runtime engine selection based on user input
let user_input = "qasm";
let engine_type = match user_input {
    "qasm" => EngineType::Qasm,
    "llvm" => EngineType::Llvm,
    "selene" => EngineType::Selene,
    _ => panic!("Unknown engine type"),
};

// For this example, we'll just use QASM
let qasm_code = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
h q[0];
measure q[0] -> c[0];
"#;
let builder = DynamicEngineBuilder::new(qasm_engine().program(Qasm::from_string(qasm_code)));

// Use the same API regardless of engine type
let results = sim_dynamic(builder).seed(42).run(10)?;
assert_eq!(results.len(), 10);
```

## Performance Considerations

Dynamic engine selection has a small runtime overhead due to trait object
indirection. For performance-critical code where the engine type is known
at compile time, prefer using the concrete engine builders directly.

## Feature Flags

The availability of engines depends on which features are enabled:

- `qasm`: Enables QASM engine support
- `llvm`: Enables LLVM engine support
- `selene`: Enables Selene engine support
