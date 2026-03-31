# pecos-cppsparsestab

C++ sparse stabilizer simulator bindings for PECOS.

This crate provides Rust FFI bindings to a C++ implementation of a sparse stabilizer tableau simulator. It implements the same interface as the pure Rust sparse stabilizer simulator but uses an optimized C++ backend.

## Features

- Efficient sparse representation of stabilizer tableaux
- Support for all Clifford gates
- Compatible with PECOS quantum simulation framework

## Usage

This crate is primarily used as a backend for PECOS simulations and is not intended for direct use. See the main PECOS documentation for usage examples.

## License

Licensed under the Apache License, Version 2.0. See LICENSE for details.
