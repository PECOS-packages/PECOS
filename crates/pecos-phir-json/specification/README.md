# PHIR-JSON Specification

This directory contains specifications for the JSON serialization format (PHIR-JSON) of the PECOS High-level Intermediate Representation (PHIR).

## Overview

PHIR-JSON is the JSON serialization format for PHIR, an intermediate representation for quantum programs in the PECOS ecosystem. It's designed to:

- Express quantum circuits combined with classical control
- Support deterministic execution of quantum programs
- Provide a human-readable and machine-processable format
- Bridge high-level languages and PECOS simulators

The current implementation uses a JSON-based format. Future versions may support additional serialization formats for
different use cases and performance requirements.

## Motivation

Quantum programs often combine quantum operations with classical control and processing. PHIR-JSON provides a standardized
JSON-based way to express these hybrid quantum-classical programs with a focus on:

1. **Readability**: JSON format is human-readable and easily inspectable
2. **Simplicity**: Direct mapping between operations and simulator capabilities
3. **Determinism**: Clear execution semantics for reproducible results
4. **Extensibility**: Versioned specification that can evolve over time

## Versioning

The PHIR-JSON specification follows a versioning scheme where each version resides in its own subdirectory:

- [v0.1/](v0.1/): Initial specification version
- Future versions will be added in similarly named directories (v0.2/, etc.)

For details on how versions evolve and are supported in the implementation, see the [LANGUAGE_EVOLUTION.md](../LANGUAGE_EVOLUTION.md)
document.

## Implementation

The primary implementation of PHIR-JSON is in Rust, providing both validation and execution capabilities:

- **Validation**: Type checking and semantic validation of PHIR-JSON programs
- **Execution**: Integration with PECOS for simulating quantum programs
- **Multi-version support**: Concurrent support for multiple specification versions

## Usage

PHIR-JSON can be used as:

1. A serialization format for quantum programs
2. An interchange format between tools in the PECOS ecosystem
3. A debugging representation for quantum circuits
4. A target for compilation from higher-level languages

## Related Resources

- [Python PHIR Validator](https://github.com/Quantinuum/phir): A Pydantic-based validator for PHIR-JSON documents
- [PECOS](https://github.com/PECOS-packages/PECOS): The PECOS quantum simulation framework
