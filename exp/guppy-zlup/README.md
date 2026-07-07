# guppy-zlup

> **Note:** Zlup is an experimental toy language for exploring quantum programming
> language design concepts. This toolchain is for research and experimentation.

A compiler toolchain for transforming Guppy quantum programs into Zlup, with
static analysis based on NASA's Power of 10 coding guidelines.

## Overview

guppy-zlup is a unified tool that:

- **Validates** Guppy programs using guppylang (semantic validation)
- **Lints** against NASA Power of 10 safety rules (ZLUP001-010)
- **Emits** intermediate representation (IR) as JSON
- **Compiles** Guppy source or IR to Zlup source code
- **Verifies** generated Zlup is syntactically and semantically valid

```
Guppy Source (.py)
    │
    ▼
┌─────────────┐
│ guppy-zlup  │
│   check     │──▶ Diagnostics (errors, warnings)
└─────────────┘
    │
    ▼
┌─────────────┐
│ guppy-zlup  │
│    emit     │──▶ Guppy IR (.json)
└─────────────┘
    │
    ▼
┌─────────────┐
│ guppy-zlup  │
│   compile   │
└─────────────┘
    │
    ▼
Zlup Source (.zlp)
```

## Installation

```bash
cargo install --path . --features cli
```

## Quick Start

```bash
# Validate a Guppy file using guppylang (requires Python)
guppy-zlup validate program.py

# Check a Guppy file for lint violations
guppy-zlup check program.py

# Check with JSON output (for CI/tooling)
guppy-zlup check program.py --format json

# Check with SARIF output (for GitHub Actions)
guppy-zlup check program.py --format sarif

# Watch mode - re-lint on file changes
guppy-zlup check program.py --watch

# Emit IR as JSON
guppy-zlup emit program.py -o program.json

# Compile Guppy source directly to Zlup
guppy-zlup compile program.py -o program.zlp

# Compile with guppylang validation first
guppy-zlup compile --validate program.py -o program.zlp

# Or compile from existing IR JSON
guppy-zlup compile --ir program.json -o program.zlp

# Analyze parallelism opportunities
guppy-zlup analyze program.py

# Analyze with JSON output
guppy-zlup analyze program.py --format json

# Compile and analyze in one step
guppy-zlup compile program.py --analyze
```

## Example

Given a Guppy program `bell.py`:

```python
def bell() -> None:
    q = qubit[2]
    h(q[0])
    cx(q[0], q[1])
    m = measure(q)
    result("measurements", m)
```

Run the toolchain:

```bash
$ guppy-zlup check bell.py
No issues found.
All checks passed!

$ guppy-zlup compile bell.py --stdout
fn bell() -> unit {
    mut q := qalloc(2);
    h q[0];
    cx (q[0], q[1]);
    m := mz([2]u1) q;
    result("measurements", m);
    return;
}
```

## Lint Rules

Based on NASA's Power of 10 coding guidelines for safety-critical systems:

| Rule    | Severity | Description                          |
|---------|----------|--------------------------------------|
| ZLUP001 | Error    | Unbounded loops (while True)         |
| ZLUP002 | Error    | Recursive function calls             |
| ZLUP003 | Error    | Dynamic memory allocation in loops   |
| ZLUP004 | Error    | Dynamic dispatch (eval, getattr)     |
| ZLUP005 | Warning  | Unchecked error-prone operations     |
| ZLUP006 | Warning  | Missing type annotations             |
| ZLUP007 | Warning  | Excessive control flow complexity    |
| ZLUP008 | Warning  | Deep call nesting (>4 levels)        |
| ZLUP009 | Info     | Missing assertions in large functions|
| ZLUP010 | Warning  | Global mutable state                 |

### Suppressing Warnings

Use `# noqa` comments to suppress specific warnings:

```python
x = eval("expression")  # noqa: ZLUP004
```

## Validation Pipeline

guppy-zlup performs multi-stage validation:

1. **Syntax Check** - Python syntax via rustpython_parser
2. **Guppylang Validation** - Semantic validation via guppylang (optional, `--validate`)
3. **Lint Rules** - NASA Power of 10 safety checks (ZLUP001-010)
4. **IR Validation** - Schema and semantic checks on intermediate representation
5. **Transform Invariants** - Debug assertions during IR→Zlup transform
6. **Output Validation** - Generated Zlup is parsed and analyzed by Zlup's semantic analyzer

This ensures both the input Guppy code and output Zlup code are valid.

## Requirements

- **Rust** 1.70+ for building
- **Python** 3.10+ with guppylang for validation (optional)

Install guppylang for validation support:
```bash
uv pip install guppylang
# or
pip install guppylang
```

## Documentation

- [docs/index.md](docs/index.md) - Full documentation index
- [docs/architecture.md](docs/architecture.md) - Pipeline design
- [docs/rules.md](docs/rules.md) - Lint rules reference
- [docs/ir-format.md](docs/ir-format.md) - IR JSON schema
- [docs/examples/](docs/examples/) - Example walkthroughs

## Library Usage

```rust
use guppy_zlup::{lint_source, compile, lint_and_compile, validate_ir, compile_with_roundtrip};

// Lint source code
let result = lint_source("def main(): pass", None);
if result.has_errors {
    for diag in result.diagnostics {
        println!("{}", diag);
    }
}

// Compile IR to Zlup
let zlup_source = compile(ir_json)?;

// Full pipeline: lint + emit + validate IR + compile + validate output
let zlup_source = lint_and_compile(guppy_source, Some("example.py"))?;

// Compile with round-trip validation (stricter)
let zlup_source = compile_with_roundtrip(ir_json)?;

// Validate IR separately
let ir = guppy_zlup::ir::emit_ir(source, None)?;
let validation = validate_ir(&ir);
if !validation.is_valid() {
    for error in &validation.errors {
        println!("IR error: {}", error);
    }
}
```

## License

Apache-2.0
