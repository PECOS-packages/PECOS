# pecos-qasm

OpenQASM 2.0 parser and execution engine.

## Purpose

Parses and executes OpenQASM 2.0 programs, implementing a classical control engine for the PECOS simulation framework.

## Key Types

- `QasmEngine` - Classical control engine for QASM programs
- `QasmEngineBuilder` - Builder pattern for engine construction
- `qasm_engine()` - Convenience function to start building

## Usage

```rust
use pecos_qasm::qasm_engine;
use pecos_programs::Qasm;

let qasm = Qasm::from_string(r#"
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    h q[0];
    cx q[0], q[1];
"#);

let engine = qasm_engine().program(qasm);
```
