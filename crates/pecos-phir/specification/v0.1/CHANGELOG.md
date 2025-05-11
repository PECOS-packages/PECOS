# PHIR v0.1 Specification Changelog

This document tracks changes and additions to the PHIR v0.1.x specification series.

## v0.1.1

### Added
- **Result Command**: Added the `Result` classical operation for exporting measurement results
  ```json
  {
    "cop": "Result",
    "args": [...],    // Source registers or bits to export
    "returns": [...]  // Names under which to export the results
  }
  ```
  This operation maps internal measurement results to exported values in the final output. It allows programs to clearly
  specify which measurement results should be exposed to users and how they should be named.

  The command supports:
  - Single bit export: `[["m", 0]]` for a specific bit
  - Entire register export: `["m"]` for the whole register
  - Multiple variable export: `["m1", "m2"]` and `["c1", "c2"]` for mapping multiple variables at once

  This flexible approach follows the general pattern of other PHIR commands and allows for concise expression of which
  values should be included in program outputs.

## v0.1.0

Initial release of the PHIR specification with:
- Basic program structure with format, version, metadata, and operations
- Quantum variable definitions
- Classical variable definitions
- Single-qubit gates: H, X, Y, Z
- Rotations: RZ, R1XY
- Two-qubit gates: CX (CNOT), SZZ (ZZ)
- Measurement operations
