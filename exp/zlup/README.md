# Zlup

> **EXPERIMENTAL / EXPLORATORY** - Zlup is a research experiment, not a production language.

A quantum programming language for QEC research: simple, low-level, and predictable by design.

## Overview

Zlup is the **low-level complement to Guppy** in the PECOS ecosystem. While Guppy provides a high-level, Pythonic experience with linear types for safety, Zlup explores a different point in the design space:

| | Guppy | Zlup |
|---|---|---|
| **Philosophy** | High-level, Pythonic | Low-level, explicit |
| **Safety mechanism** | Linear type system | Constraints make unsafe impossible |
| **Target users** | QEC researchers | Systems programmers |

**Design principles:** Simple. Explicit. No magic.

- Zig semantics with Rust/Python syntax
- NASA Power of 10 compliance for reliability
- Safe by constraint (no recursion, no escaping references)

## Quick Example

```zig
pub fn main() -> unit {
    q := qalloc(4);
    pz q;

    // Create GHZ state
    h q[0];
    cx (q[0], q[1]);
    cx (q[1], q[2]);
    cx (q[2], q[3]);

    // Measure all qubits
    results: [4]u1 = mz([4]u1) [q[0], q[1], q[2], q[3]];

    return;
}
```

## Building

```bash
# Build the compiler
cargo build --features cli

# Run tests
cargo test --features cli

# Compile a program
cargo run --features cli -- compile program.zlp

# Check without compiling
cargo run --features cli -- check program.zlp
```

Or use the Justfile:

```bash
just build      # Build compiler
just test       # Run tests
just docs       # View documentation
```

## Documentation

Full documentation is available in the `docs/` directory:

| Document | Description |
|----------|-------------|
| [Tutorial](docs/tutorial.md) | Getting started guide |
| [Language Syntax](docs/syntax.md) | Complete language reference |
| [CLI Reference](docs/cli.md) | Command-line interface |
| [Standard Library](docs/stdlib.md) | Standard library reference |
| [Error Messages](docs/errors.md) | Error messages guide |
| [Design Philosophy](docs/design.md) | Why Zlup exists |
| [Rust Integration](docs/rust-integration.md) | FFI and native backends |
| [IDE Setup](docs/ide-setup.md) | Editor configuration |

To view docs locally:

```bash
just docs  # Starts server at http://127.0.0.1:8000
```

## Status

Zlup is **highly experimental**—an exploration of language design, not a production tool. The goal is to learn from building it, not to replace existing tools.

See the [Design Philosophy](docs/design.md) for more on why Zlup exists.

## License

Apache-2.0
