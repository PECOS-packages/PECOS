# guppy-zlup

> **Note:** Zlup is an experimental toy language for exploring quantum programming
> language design concepts. This toolchain is for research and experimentation.

A compiler toolchain for transforming Guppy quantum programs into Zlup, with
static analysis based on NASA's Power of 10 coding guidelines.

## Overview

guppy-zlup is a unified tool that:

- **Validates** Guppy programs against safety rules (ZLUP001-010)
- **Analyzes** parallelism opportunities in compiled Zlup code
- **Emits** intermediate representation (IR) as JSON
- **Compiles** Guppy source or IR to Zlup source code

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

## Quick Start

### Installation

```bash
cd exp/guppy-zlup
cargo build --features cli
```

### Basic Usage

```bash
# Check a Guppy file for violations
guppy-zlup check program.py

# Emit IR as JSON
guppy-zlup emit program.py -o program.json

# Compile Guppy source directly to Zlup (lint + emit + compile)
guppy-zlup compile program.py -o program.zlp

# Or compile from existing IR JSON
guppy-zlup compile --ir program.json -o program.zlp
```

### Example

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

## CLI Reference

### `guppy-zlup check`

Validate a Guppy file against lint rules.

```bash
guppy-zlup check <files>... [OPTIONS]

Options:
  -W, --warnings-as-errors  Treat warnings as errors
      --config <path>       Path to config file (pyproject.toml)
  -D, --disable <rule>      Disable specific rules (can be repeated)
      --max-complexity <n>  Maximum complexity for ZLUP007
  -f, --format <fmt>        Output format: text (default), json, or sarif
  -w, --watch               Watch for file changes and re-lint
```

### `guppy-zlup emit`

Emit validated IR as JSON.

```bash
guppy-zlup emit <file> [OPTIONS]

Options:
  -o, --output <path>  Output file (default: ir.json)
      --skip-lint      Skip lint check
      --stdout         Print to stdout instead of file
```

### `guppy-zlup compile`

Compile Guppy source or IR to Zlup.

```bash
guppy-zlup compile <file> [OPTIONS]

Options:
  -o, --output <path>  Output file (default: <input>.zlp)
      --stdout         Print to stdout instead of file
      --ir             Input is IR JSON (skip linting)
      --validate       Validate with guppylang before compiling (requires Python)
      --analyze        Run parallelism analysis after compilation
```

### `guppy-zlup analyze`

Analyze parallelism opportunities in generated Zlup code.

```bash
guppy-zlup analyze <file> [OPTIONS]

Options:
      --ir             Input is IR JSON (skip linting)
  -f, --format <fmt>   Output format: text (default) or json
  -v, --verbose        Show detailed dependency information
```

## Documentation

- [Architecture](./architecture.md) - Pipeline design, AST layers, internals
- [Lint Rules](./rules.md) - ZLUP001-010 reference with examples
- [IR Format](./ir-format.md) - Guppy IR JSON schema
- [Examples](./examples/bell-state.md) - Walkthrough of common patterns

### Future / Design Notes

- [Parallelism](./future/parallelism.md) - Scope-aware parallelism analysis

## Why These Rules?

Quantum programs run on hardware with strict constraints:

1. **Bounded execution** - No infinite loops; hardware has finite coherence time
2. **Static resource allocation** - Qubit counts must be known at compile time
3. **Predictable control flow** - Dynamic dispatch complicates timing analysis
4. **Type safety** - Quantum operations require precise type information

The lint rules (ZLUP001-010) enforce these constraints at the source level,
catching violations before they reach the quantum hardware.

## License

Apache-2.0
