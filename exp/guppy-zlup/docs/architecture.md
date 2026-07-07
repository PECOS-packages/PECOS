# Architecture

This document describes the internal architecture of guppy-zlup.

## Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           guppy-zlup                                     │
│                                                                          │
│  Python Source (.py)                                                     │
│        │                                                                 │
│        ▼ rustpython-parser                                               │
│  Python AST (rustpython_parser::ast)                                     │
│        │                                                                 │
│        ├──────────────────────┬──────────────────────┐                   │
│        ▼                      ▼                      ▼                   │
│  Lint Rules (rules/*.rs)   linter/lower.rs       ir::emit_ir             │
│        │                      │                      │                   │
│        ▼                      ▼                      ▼                   │
│  Diagnostics            Guppy AST             Guppy IR (JSON)            │
│                    (linter::ast)                     │                   │
│                                                      │                   │
│                                                      ▼                   │
│                                              ir::validate_ir ◄── IR Validation
│                                                      │                   │
│                                                      ▼                   │
│                                              compiler/parser.rs          │
│                                                      │                   │
│                                                      ▼                   │
│                                              compiler/transform.rs       │
│                                              (with invariant checks)     │
│                                                      │                   │
│                                                      ▼                   │
│                                              Zlup AST (zlup::ast)        │
│                                                      │                   │
│                                                      ▼                   │
│                                              zlup::pretty::pretty_print  │
│                                                      │                   │
│                                                      ▼                   │
│                                              Zlup Source (.zlp)          │
│                                                      │                   │
│                                                      ▼                   │
│                                              validate_zlup ◄── Output Validation
│                                              (parse + semantic analysis) │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

## AST Layers

The toolchain uses multiple AST representations, each serving a specific purpose:

### 1. Python AST (`rustpython_parser::ast`)

The raw AST from parsing Python source code. This is a general-purpose Python
AST that includes all Python constructs, many of which aren't valid in Guppy.

- **Source**: rustpython-parser crate
- **Used by**: `linter/lower.rs`, `ir.rs`

### 2. Guppy AST (`linter::ast`)

A clean, Guppy-specific AST that:
- Represents only constructs valid in Guppy
- Has first-class support for quantum operations (gates, measurements, qalloc)
- Isolates the codebase from Python parser API changes

Key types:
```rust
pub struct Module { functions: Vec<Function>, ... }
pub struct Function { name: String, params: Vec<Param>, body: Vec<Stmt>, ... }
pub enum Stmt { Qalloc, Gate, Measure, For, While, If, ... }
pub enum Expr { IntLit, Name, BinOp, Call, ... }
pub enum GateKind { H, X, Y, Z, Cx, Cz, ... }
```

- **Source**: `src/linter/ast.rs`
- **Used by**: Lint rules

### 3. Guppy IR (JSON)

A serializable intermediate representation for tool interoperability:
- Can be consumed by any tool that reads JSON
- Enables caching of validated programs
- Allows alternative frontends to target the Zlup backend

See [ir-format.md](./ir-format.md) for the full schema.

- **Source**: `src/ir.rs`
- **Used by**: `emit` command (output), `compile --ir` command (input)

### 4. Zlup AST (`zlup::ast`)

The target language's AST. This is the canonical representation of Zlup
programs, used by the Zlup compiler for:
- Code generation (QASM, PHIR, HUGR)
- Optimization passes
- Semantic analysis

- **Source**: `../zlup/src/ast.rs`
- **Used by**: `compiler/transform.rs`, Zlup pretty printer

## Validation Layers

The toolchain includes multiple validation layers to catch errors early and ensure correctness:

### 1. IR Validation (`ir::validate_ir`)

Validates the intermediate representation before transformation:
- **Schema validation**: Required fields present for each statement/expression kind
- **Semantic validation**: Variables defined before use, allocators exist before gate use
- **Gate arity**: Correct number of targets for each gate type
- **Operator validation**: Only known operators allowed

```rust
let result = ir::validate_ir(&ir);
if !result.is_valid() {
    // Handle errors
}
```

### 2. Transform Invariants

Debug assertions during IR → Zlup transformation:
- Allocators must be registered before use in gates
- Variable scope tracking for assignments vs new bindings
- Consistency between qalloc_sizes and declared_vars

These fire as panics in debug builds, catching internal bugs early.

### 3. Output Validation (`validate_zlup`)

Validates generated Zlup source by:
- Parsing it back through Zlup's parser
- Running Zlup's semantic analyzer (permissive mode)

### 4. Round-Trip Validation (`validate_zlup_roundtrip`)

Optional stricter validation:
- Compares original AST with re-parsed AST
- Verifies function counts, names, and parameter counts match
- Catches pretty-printer bugs

## Design Decisions

### Why a separate Guppy AST?

1. **Parser isolation** - rustpython-parser API changes don't ripple through
   the codebase
2. **Quantum-first** - Gates, measurements, and qalloc are first-class
   constructs, not function calls
3. **Linting** - Easier to write lint rules against a clean AST

### Why JSON for the IR?

1. **Tool interop** - Other languages can emit Guppy IR
2. **Debugging** - Human-readable intermediate format
3. **Caching** - Save validated IR to disk

### Why use Zlup's pretty printer?

1. **Consistency** - Output matches Zlup's canonical style
2. **Maintenance** - One formatter to maintain, not two
3. **Correctness** - Zlup's formatter handles edge cases

## Module Structure

```
src/
├── lib.rs              # Library entry point, public API
├── main.rs             # CLI entry point
├── ir.rs               # Guppy IR types and emit_ir()
├── compiler.rs         # Compiler module
├── compiler/
│   ├── parser.rs       # JSON → GuppyIR
│   └── transform.rs    # GuppyIR → zlup::ast::Program
├── linter.rs           # Linter module
└── linter/
    ├── ast.rs          # Guppy AST definitions
    ├── config.rs       # Configuration
    ├── diagnostic.rs   # Error/warning types
    ├── engine.rs       # Lint orchestration
    ├── lower.rs        # Python AST → Guppy AST
    ├── rules.rs        # Rules module
    └── rules/          # Lint rule implementations
        ├── zlup001.rs  # Unbounded loops
        ├── zlup002.rs  # Recursion
        ├── zlup003.rs  # Dynamic allocation
        ├── zlup004.rs  # Dynamic dispatch
        ├── zlup005.rs  # Unchecked errors
        ├── zlup006.rs  # Missing types
        ├── zlup007.rs  # Complex control flow
        ├── zlup008.rs  # Deep call nesting
        ├── zlup009.rs  # Missing assertions
        └── zlup010.rs  # Global mutable state
```

## Key Transformations

### Entry Function Returns

In Guppy (following Quantinuum's design), entry/main functions must return `None`.
Values are emitted to the quantum runtime via explicit `result()` calls:

**Guppy:**
```python
def main() -> None:
    q = qubit[4]
    h(q[0])
    cx(q[0], q[1])
    m = measure(q)
    result("measurements", m)  # Emit to runtime
```

**Zlup:**
```zlup
fn main() -> unit {
    mut q := qalloc(4);
    h q[0];
    cx (q[0], q[1]);
    m := mz([4]u1) q;
    result("measurements", m);
    return;
}
```

This pattern:
- Entry functions return `None` (Guppy) / `unit` (Zlup)
- Results are explicitly tagged via `result(tag, value)`
- The quantum runtime collects all `result()` emissions

## Extending the Toolchain

### Adding a new lint rule

1. Create `src/linter/rules/zlupNNN.rs`
2. Implement the `LintRule` trait
3. Register in `src/linter/rules.rs`
4. Add tests

### Supporting a new Guppy construct

1. Add the construct to `linter::ast`
2. Update `linter/lower.rs` to convert from Python AST
3. Update `ir.rs` to serialize/deserialize
4. Update `compiler/transform.rs` to emit Zlup

### Alternative frontends

Any tool can target guppy-zlup by emitting Guppy IR JSON. See
[ir-format.md](./ir-format.md) for the schema.
