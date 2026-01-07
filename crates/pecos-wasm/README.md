# pecos-wasm

`pecos-wasm` provides WebAssembly foreign object support for PECOS.

This crate enables execution of WebAssembly modules for classical computations within the PECOS quantum error correction framework, with configurable timeout and memory limits.

## Features

- Thread-safe WASM execution via Wasmtime
- Configurable execution timeouts (default: 1 second)
- Configurable memory limits (default: unlimited)
- Support for both .wasm and .wat files

## Usage

This is an **internal crate** used by:
- `pecos-qasm` - QASM program execution with WASM foreign objects
- `pecos-phir-json` - PHIR program execution with WASM foreign objects
- `pecos-rslib` - Python bindings exposing WASM functionality

## Acknowledgements

This crate uses [Wasmtime](https://github.com/bytecodealliance/wasmtime), a standalone WebAssembly runtime developed by the Bytecode Alliance.
