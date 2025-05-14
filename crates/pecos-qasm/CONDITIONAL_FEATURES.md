# Conditional Statement Features in PECOS QASM

## Overview

PECOS QASM supports standard OpenQASM 2.0 conditional statements by default, with optional extended features available via a configuration flag.

## Standard OpenQASM 2.0 Conditionals (Default)

By default, PECOS QASM follows the OpenQASM 2.0 specification for conditional statements:

```qasm
OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[4];
creg d[1];

// Valid standard conditionals
if (c == 2) h q[0];      // Register compared to integer constant
if (c != 0) x q[1];      // Register compared to integer constant
if (c > 1) h q[0];       // Register compared to integer constant
if (c <= 3) x q[1];      // Register compared to integer constant

d[0] = 1;
if (d[0] == 1) x q[1];   // Bit compared to integer constant
```

### Supported Comparison Operators
- `==` (equals)
- `!=` (not equals)
- `<` (less than)
- `>` (greater than)
- `<=` (less than or equal)
- `>=` (greater than or equal)

### Limitations in Standard Mode
- Only register or bit compared to integer constants
- No complex expressions in conditionals
- No register-to-register comparisons

## Extended Conditionals (Feature Flag)

PECOS QASM provides extended conditional functionality that can be enabled via a feature flag:

```rust
use pecos_qasm::engine::QASMEngine;

let mut engine = QASMEngine::new().unwrap();
engine.set_allow_complex_conditionals(true);  // Enable extended features
```

With the flag enabled, you can use:

```qasm
OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg a[4];
creg b[4];

a = 2;
b = 3;

// Extended conditionals (require feature flag)
if (a < b) h q[0];                  // Register compared to register
if ((a + b) == 5) x q[1];          // Expression compared to integer
if (a[0] & b[0] == 0) h q[0];      // Bitwise operation in condition
if ((a * 2) > b) x q[1];           // Complex expression
```

## Error Messages

When attempting to use extended features without the flag:

```
Complex conditionals are not allowed. Only register/bit compared to integer 
is supported in standard OpenQASM 2.0. Enable allow_complex_conditionals 
to use general expressions.
```

## Implementation Details

### Type System
- Uses signed 64-bit integers (`i64`) to handle arithmetic operations
- Prevents underflow issues with subtraction operations
- Supports all standard arithmetic and bitwise operations

### Parser Architecture
- Parses all conditional expressions as general expressions
- Engine validates expressions based on configuration
- Clean separation between parsing and semantic validation

## Examples

### Standard Mode (Default)
```rust
let qasm = r#"
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    creg c[4];
    c = 2;
    if (c == 2) h q[0];  // Works in standard mode
"#;

let program = QASMParser::parse_str(qasm)?;
let mut engine = QASMEngine::new()?;
engine.load_program(program)?;
engine.generate_commands()?;  // Success
```

### Extended Mode
```rust
let qasm = r#"
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    creg a[2];
    creg b[2];
    if (a < b) h q[0];  // Requires feature flag
"#;

let program = QASMParser::parse_str(qasm)?;
let mut engine = QASMEngine::new()?;
engine.set_allow_complex_conditionals(true);  // Enable feature
engine.load_program(program)?;
engine.generate_commands()?;  // Success
```

## Testing

Comprehensive test coverage ensures:
- Standard conditionals work by default
- Extended features fail without the flag
- Extended features work with the flag
- Clear error messages guide users
- All comparison operators function correctly
- Bit indexing works in conditionals

## Future Extensions

The architecture supports future additions:
- More complex boolean expressions
- Multiple conditions with logical operators
- Function calls in conditionals
- Pattern matching