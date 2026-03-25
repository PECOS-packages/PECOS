# WebAssembly Foreign Objects

This guide covers using WebAssembly (WASM) modules for classical computation within PECOS quantum simulations. WASM foreign objects allow you to execute custom classical logic alongside quantum operations in QASM and PHIR programs.

## What You'll Learn

- When and why to use WASM foreign objects
- How to load WASM modules from files or bytes
- WASM module requirements and conventions
- Integrating WASM functions with quantum programs
- Configuration options (timeout, memory limits)
- Serialization for distributed execution

## Overview

WASM foreign objects enable hybrid quantum/classical computation by allowing quantum programs to call classical functions implemented in WebAssembly. This is useful for:

- **Classical preprocessing/postprocessing**: Compute values needed for quantum operations
- **Conditional logic**: Make decisions based on measurement outcomes
- **Complex arithmetic**: Perform calculations that would be cumbersome in QASM
- **Reusable libraries**: Share classical logic across multiple quantum programs

## Loading WASM Modules

PECOS supports loading WebAssembly modules from files (`.wasm` binary or `.wat` text format) or directly from bytes in memory.

### From a File

Use `from_file()` to load a WASM module from disk:

=== ":fontawesome-brands-python: Python"

    ```hidden-python
    import tempfile
    import os
    import shutil
    from pathlib import Path as _Path
    from pecos_rslib import WasmForeignObject
    import pickle

    # Use the shared test WAT file from docs/assets/test-data/
    _orig_cwd = os.getcwd()
    _test_wat_src = _Path(_orig_cwd) / "docs/assets/test-data/math.wat"

    _tmpdir = tempfile.mkdtemp()
    os.chdir(_tmpdir)

    # Copy test WAT file with various names used in examples
    for _name in ["math_functions.wasm", "math_functions.wat", "math.wasm", "math.wat",
                  "compute.wat", "simple.wasm", "stateful.wasm"]:
        shutil.copy(_test_wat_src, _name)
    ```

    ```python
    from pecos_rslib import WasmForeignObject
    from pathlib import Path

    # From a string path
    wasm = WasmForeignObject.from_file("math_functions.wat")

    # From a pathlib.Path
    wasm = WasmForeignObject.from_file(Path("math_functions.wasm"))

    # With custom timeout (5 seconds instead of default 1 second)
    wasm = WasmForeignObject.from_file("math_functions.wasm", timeout=5.0)

    # With memory limit (10 MB)
    wasm = WasmForeignObject.from_file("math_functions.wasm", timeout=5.0, memory_size=10 * 1024 * 1024)
    ```

=== ":fontawesome-brands-rust: Rust"

    ```hidden-rust
    use pecos::wasm::WasmForeignObject;
    use std::fs;

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        let tmpdir = tempfile::tempdir()?;
        let wasm_path = tmpdir.path().join("math_functions.wasm");
        let wat = r#"(module
          (global $accumulator (mut i32) (i32.const 0))
          (func $init)
          (func $shot_reinit (i32.const 0) (global.set $accumulator))
          (func $add (param i32 i32) (result i32) (local.get 0) (local.get 1) (i32.add))
          (func $mul (param i32 i32) (result i32) (local.get 0) (local.get 1) (i32.mul))
          (memory (;0;) 1)
          (export "init" (func $init))
          (export "shot_reinit" (func $shot_reinit))
          (export "add" (func $add))
          (export "mul" (func $mul))
          (export "memory" (memory 0))
        )"#;
        fs::write(&wasm_path, wat)?;

        // CODE
        Ok(())
    }
    ```

    ```rust
    use pecos::wasm::WasmForeignObject;

    // From a file path with default timeout (1 second)
    let wasm = WasmForeignObject::new(&wasm_path)?;

    // With custom timeout
    let wasm = WasmForeignObject::with_timeout(&wasm_path, 5.0)?;

    // With custom timeout and memory limit
    let wasm = WasmForeignObject::with_limits(
        wasm_path.to_str().unwrap(),
        5.0,                      // timeout in seconds
        Some(10 * 1024 * 1024),   // memory limit in bytes
    )?;
    ```

### From Bytes

Use `from_bytes()` when you have the WASM binary in memory. This is useful for:

- Downloaded WASM modules
- Embedded/bundled WASM binaries
- Dynamically generated WASM

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos_rslib import WasmForeignObject

    # Load WASM bytes from a file
    with open("math_functions.wasm", "rb") as f:
        wasm_bytes = f.read()

    wasm = WasmForeignObject.from_bytes(wasm_bytes)

    # With configuration options
    wasm = WasmForeignObject.from_bytes(wasm_bytes, timeout=5.0, memory_size=10 * 1024 * 1024)
    ```

=== ":fontawesome-brands-rust: Rust"

    ```hidden-rust
    use pecos::wasm::WasmForeignObject;

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        let wasm_bytes: Vec<u8> = br#"(module
          (func $init)
          (func $add (param i32 i32) (result i32) (local.get 0) (local.get 1) (i32.add))
          (memory (;0;) 1)
          (export "init" (func $init))
          (export "add" (func $add))
          (export "memory" (memory 0))
        )"#.to_vec();

        // CODE
        Ok(())
    }
    ```

    ```rust
    use pecos::wasm::WasmForeignObject;

    // From bytes with default timeout
    let wasm = WasmForeignObject::from_bytes(&wasm_bytes)?;

    // With custom timeout
    let wasm = WasmForeignObject::from_bytes_with_timeout(&wasm_bytes, 5.0)?;

    // With custom timeout and memory limit
    let wasm = WasmForeignObject::from_bytes_with_limits(
        &wasm_bytes,
        5.0,
        Some(10 * 1024 * 1024),
    )?;
    ```

## WASM Module Requirements

For a WASM module to work with PECOS, it must follow these conventions:

### Required: `init()` Function

Every WASM module **must** export an `init()` function. This is called once when the module is initialized:

```wat
(module
  (func $init)
  (export "init" (func $init))
)
```

### Optional: `shot_reinit()` Function

If your module maintains state that should be reset between shots, export a `shot_reinit()` function:

```wat
(module
  (global $counter (mut i32) (i32.const 0))

  (func $init)

  (func $shot_reinit
    ;; Reset counter to 0 before each shot
    i32.const 0
    global.set $counter)

  (export "init" (func $init))
  (export "shot_reinit" (func $shot_reinit))
)
```

### Supported Types

WASM functions can use:

- **Parameters**: `i32` or `i64` integers
- **Return values**: `i32` or `i64` integers (single or multiple)

### Reserved Function Names

The following function names are reserved and **cannot** be overridden by WASM modules:

- `sin`, `cos`, `tan`
- `exp`, `ln`
- `sqrt`

### Example: Complete WASM Module

Here's a complete example of a WASM module with multiple functions:

```wat
(module
  ;; Global state
  (global $accumulator (mut i32) (i32.const 0))

  ;; Required init function
  (func $init)

  ;; Optional shot reset
  (func $shot_reinit
    i32.const 0
    global.set $accumulator)

  ;; Add two numbers
  (func $add (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.add)

  ;; Multiply two numbers
  (func $mul (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.mul)

  ;; Accumulate a value and return the total
  (func $accumulate (param i32) (result i32)
    local.get 0
    global.get $accumulator
    i32.add
    global.set $accumulator
    global.get $accumulator)

  ;; Exports
  (export "init" (func $init))
  (export "shot_reinit" (func $shot_reinit))
  (export "add" (func $add))
  (export "mul" (func $mul))
  (export "accumulate" (func $accumulate))
)
```

## Using WASM with Quantum Programs

### Direct Execution

You can execute WASM functions directly:

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos_rslib import WasmForeignObject

    wasm = WasmForeignObject.from_file("math.wasm")
    wasm.init()

    # Execute functions
    result = wasm.exec("add", [5, 3])
    print(f"5 + 3 = {result}")  # Output: 5 + 3 = 8

    # List available functions
    print(wasm.get_funcs())  # ['init', 'add', 'mul', ...]
    ```

=== ":fontawesome-brands-rust: Rust"

    ```hidden-rust
    use pecos::wasm::{WasmForeignObject, ForeignObject};
    use std::fs;

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        let tmpdir = tempfile::tempdir()?;
        let wat_path = tmpdir.path().join("math.wat");
        let wat = r#"(module
          (func $init)
          (func $add (param i32 i32) (result i32) (local.get 0) (local.get 1) (i32.add))
          (func $mul (param i32 i32) (result i32) (local.get 0) (local.get 1) (i32.mul))
          (memory (;0;) 1)
          (export "init" (func $init))
          (export "add" (func $add))
          (export "mul" (func $mul))
          (export "memory" (memory 0))
        )"#;
        fs::write(&wat_path, wat)?;

        // CODE
        Ok(())
    }
    ```

    ```rust
    use pecos::wasm::{WasmForeignObject, ForeignObject};

    let mut wasm = WasmForeignObject::new(&wat_path)?;
    wasm.init()?;

    // Execute functions
    let result = wasm.exec("add", &[5, 3])?;
    println!("5 + 3 = {:?}", result);  // Output: 5 + 3 = [8]

    // List available functions
    println!("{:?}", wasm.get_funcs());
    ```

### Integration with QASM

WASM functions can be called from QASM programs using the foreign function syntax:

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos import sim, Qasm
    from pecos_rslib import WasmForeignObject

    # QASM code that uses a foreign function
    qasm_code = """
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[2];
        creg c[2];
        creg result[8];

        // Call WASM function and store result
        result = add(3, 5);

        // Use result in quantum operations (if needed)
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    """

    # Create WASM foreign object from WAT file
    wasm = WasmForeignObject.from_file("math.wat")

    # Run simulation with foreign object
    results = sim(Qasm(qasm_code)).foreign_object(wasm).run(10)
    ```

=== ":fontawesome-brands-rust: Rust"

    ```hidden-rust
    use pecos::prelude::*;
    use std::fs;

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        let tmpdir = tempfile::tempdir()?;
        let wat_path = tmpdir.path().join("math_add.wat");
        let wat = r#"(module
          (func $init)
          (func $add (param i32 i32) (result i32) (local.get 0) (local.get 1) (i32.add))
          (memory (;0;) 1)
          (export "init" (func $init))
          (export "add" (func $add))
          (export "memory" (memory 0))
        )"#;
        fs::write(&wat_path, wat)?;

        // CODE
        Ok(())
    }
    ```

    ```rust
    use pecos::prelude::*;

    let qasm_code = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[2];
        creg c[2];
        creg result[8];

        result = add(3, 5);

        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    // Build engine with WASM foreign functions
    let results = qasm_engine()
        .qasm(qasm_code)
        .wasm(wat_path.to_str().unwrap())  // Load WASM module for foreign functions
        .to_sim()
        .run(100)?;
    ```

### Conditional Quantum Operations

WASM functions enable conditional logic based on classical computation:

```qasm
OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c[3];
creg threshold[8];

// Compute threshold from external parameters
threshold = compute_threshold(100, 50);

// Prepare state
h q[0];
cx q[0], q[1];

// Measure
measure q[0] -> c[0];

// Conditional operation based on measurement and threshold
if (c[0] == 1) x q[2];

measure q -> c;
```

## Configuration Options

### Timeout

WASM execution has a configurable timeout (default: 1 second) to prevent infinite loops:

=== ":fontawesome-brands-python: Python"

    ```python
    # 5 second timeout
    wasm = WasmForeignObject.from_file("compute.wat", timeout=5.0)

    # Very short timeout for quick operations
    wasm = WasmForeignObject.from_file("simple.wasm", timeout=0.1)
    ```

=== ":fontawesome-brands-rust: Rust"

    ```hidden-rust
    use pecos::wasm::WasmForeignObject;
    use std::fs;

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        let tmpdir = tempfile::tempdir()?;
        let wat_path = tmpdir.path().join("compute.wat");
        let wat = r#"(module
          (func $init)
          (memory (;0;) 1)
          (export "init" (func $init))
          (export "memory" (memory 0))
        )"#;
        fs::write(&wat_path, wat)?;

        // CODE
        Ok(())
    }
    ```

    ```rust
    // 5 second timeout
    let wasm = WasmForeignObject::with_timeout(&wat_path, 5.0)?;
    ```

If execution exceeds the timeout, a `RuntimeError` (Python) or `PecosError::Processing` (Rust) is raised.

### Memory Limits

You can limit the memory available to WASM modules:

=== ":fontawesome-brands-python: Python"

    ```python
    # Limit to 10 MB
    wasm = WasmForeignObject.from_file("compute.wat", memory_size=10 * 1024 * 1024)

    # No limit (default)
    wasm = WasmForeignObject.from_file("compute.wat", memory_size=None)
    ```

=== ":fontawesome-brands-rust: Rust"

    ```hidden-rust
    use pecos::wasm::WasmForeignObject;
    use std::fs;

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        let tmpdir = tempfile::tempdir()?;
        let wat_path = tmpdir.path().join("compute.wat");
        let wat = r#"(module
          (func $init)
          (memory (;0;) 1)
          (export "init" (func $init))
          (export "memory" (memory 0))
        )"#;
        fs::write(&wat_path, wat)?;

        // CODE
        Ok(())
    }
    ```

    ```rust
    // Limit to 10 MB
    let wasm = WasmForeignObject::with_limits(
        wat_path.to_str().unwrap(),
        1.0,                      // timeout
        Some(10 * 1024 * 1024),   // memory limit
    )?;

    // No limit
    let wasm = WasmForeignObject::with_limits(wat_path.to_str().unwrap(), 1.0, None)?;
    ```

## Serialization and Pickling

WASM foreign objects support Python pickling for distributed execution:

```python
import pickle
from pecos_rslib import WasmForeignObject

# First, create a simple WASM module
math_wat = """
(module
  (func (export "add") (param i64 i64) (result i64)
    local.get 0
    local.get 1
    i64.add)
  (func (export "init"))
)
"""
with open("math.wat", "w") as f:
    f.write(math_wat)

# Create and configure
wasm = WasmForeignObject.from_file("math.wat", timeout=5.0)
wasm.init()

# Serialize
data = pickle.dumps(wasm)

# Deserialize (e.g., on another worker)
wasm_restored = pickle.loads(data)
wasm_restored.init()

# Use normally
result = wasm_restored.exec("add", [1, 2])
assert result == 3
```

You can also use the explicit `to_dict()` and `from_dict()` methods:

```python
from pecos_rslib import WasmForeignObject

# Create WASM object (using math.wat from previous example)
wasm = WasmForeignObject.from_file("math.wat")

# Serialize to dict
state = wasm.to_dict()

# Restore from dict
wasm_restored = WasmForeignObject.from_dict(state)
```

## Accessing WASM Bytes

You can retrieve the compiled WASM bytes from a foreign object:

=== ":fontawesome-brands-python: Python"

    ```python
    wasm = WasmForeignObject.from_file("math.wat")  # Load from WAT

    # Get the compiled WASM bytes
    wasm_bytes = wasm.wasm_bytes

    # Save to a .wasm file
    with open("math.wasm", "wb") as f:
        f.write(wasm_bytes)

    # Or create a new instance from the bytes
    wasm2 = WasmForeignObject.from_bytes(wasm_bytes)
    ```

=== ":fontawesome-brands-rust: Rust"

    ```hidden-rust
    use pecos::wasm::WasmForeignObject;
    use std::fs;

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        let tmpdir = tempfile::tempdir()?;
        let wat_path = tmpdir.path().join("clone_test.wat");
        let wat = r#"(module
          (func $init)
          (memory (;0;) 1)
          (export "init" (func $init))
          (export "memory" (memory 0))
        )"#;
        fs::write(&wat_path, wat)?;

        // CODE
        Ok(())
    }
    ```

    ```rust
    let wasm = WasmForeignObject::new(&wat_path)?;

    // Get the WASM bytes
    let bytes = wasm.wasm_bytes();

    // Create a new instance from bytes
    let wasm2 = WasmForeignObject::from_bytes(bytes)?;
    ```

## Lifecycle Management

### Initialization Flow

1. **Load**: `from_file()` or `from_bytes()` compiles the WASM module
2. **Init**: `init()` creates an instance and calls the module's `init` function
3. **Execute**: `exec()` calls functions on the instance
4. **Reset** (optional): `shot_reinit()` resets state between shots
5. **Teardown**: `teardown()` cleans up resources (called automatically on drop)

### Resetting State

For simulations with multiple shots, call `shot_reinit()` to reset module state:

```python
wasm = WasmForeignObject.from_file("stateful.wasm")
wasm.init()

for shot in range(1000):
    wasm.shot_reinit()  # Reset state for this shot
    # ... run quantum simulation ...
```

### Creating Fresh Instances

To completely reset a module (re-run `init`):

```python
from pecos_rslib import WasmForeignObject

# Create WASM object
wasm = WasmForeignObject.from_file("math.wat")
wasm.init()

# Later, to completely reset:
wasm.new_instance()  # Creates a fresh WASM instance
wasm.init()  # Re-initialize
```

## Error Handling

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos import WasmForeignObject

    # File not found
    try:
        wasm = WasmForeignObject.from_file("nonexistent.wasm")
    except FileNotFoundError as e:
        print(f"File error: {e}")

    # Compilation error (invalid WASM)
    try:
        wasm = WasmForeignObject.from_bytes(b"invalid wasm")
    except RuntimeError as e:
        print(f"Compilation error: {e}")
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    use pecos::wasm::WasmForeignObject;

    // File not found error - this intentionally fails
    match WasmForeignObject::new("nonexistent.wasm") {
        Err(e) => println!("Expected error: {}", e),
        Ok(_) => println!("Unexpected success"),
    }
    ```

## Best Practices

1. **Keep WASM modules focused**: Each module should do one thing well
2. **Use `shot_reinit()`**: If your module has state, implement reset logic
3. **Set appropriate timeouts**: Prevent runaway computations
4. **Limit memory**: Protect against memory exhaustion
5. **Test independently**: Verify WASM functions work before integrating with quantum code
6. **Use `from_bytes()` for embedded modules**: Avoid file system dependencies in production

## Further Reading

- [QASM Simulations](qasm-simulation.md) - Using WASM with QASM programs
- [WebAssembly Text Format](https://webassembly.github.io/spec/core/text/index.html) - WAT syntax reference
- [Wasmtime Documentation](https://docs.wasmtime.dev/) - The WASM runtime used by PECOS
