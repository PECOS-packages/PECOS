# PECOS QASM Extensions

PECOS implements a **superset** of OpenQASM 2.0, providing additional features and flexibility while maintaining backward compatibility with standard OpenQASM programs.

## Extensions to OpenQASM 2.0

### 1. Extended Gate Body Operations
While OpenQASM 2.0 typically restricts gate bodies to unitary operations, PECOS allows:

- **Barriers in gate definitions**
  ```qasm
  gate my_gate a, b {
      h a;
      barrier a, b;  // PECOS extension
      cx a, b;
  }
  ```

- **Reset operations in gate definitions**
  ```qasm
  gate reset_and_prepare a {
      reset a;       // PECOS extension
      h a;
  }
  ```

### 2. Conditional Expressions (Feature Flag)
PECOS supports extended conditional expressions when feature flags are enabled:

- Complex comparisons: `if ((a + b) > c) h q[0];`
- Expression support in conditions

### 3. Native Hardware Gates
PECOS treats additional gates as native for performance:

- `H`, `X`, `Y`, `Z` (uppercase variants)
- `RZ`, `RZZ`, `SZZ` (hardware-optimized)
- Direct mapping to quantum hardware capabilities

### 4. Classical Operations
Enhanced classical computation support:

- Bitwise operations: `&`, `|`, `^`, `~`, `<<`, `>>`
- Arithmetic: `+`, `-`, `*`, `/`
- Register-wide operations: `c = a & b;`

## Compatibility Note

All standard OpenQASM 2.0 programs will run unchanged in PECOS. The extensions are:
- Optional - you don't have to use them
- Backward compatible - existing programs work as expected
- Performance-oriented - designed for real quantum hardware

## Philosophy

PECOS QASM follows a "be liberal in what you accept" philosophy:
- If an operation makes sense and can be executed, we allow it
- Extensions are driven by practical hardware needs
- Clear semantics are maintained for all operations

## Usage Guidelines

1. **For OpenQASM 2.0 compatibility**: Stick to standard operations
2. **For PECOS features**: Use extensions where they provide value
3. **For hardware optimization**: Leverage native gates and barriers

The permissive approach allows researchers and developers to experiment while maintaining a path to standard compliance when needed.