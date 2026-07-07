# CLI Reference

Zlup provides a command-line interface for compiling, checking, and running Zlup programs.

## Project Commands

### Initialize a Project

```bash
# Create a new project
zlup init my-project

# Create project in current directory
zlup init my-project --here
```

### Build a Project

```bash
# Build using zlup.toml configuration
zlup build

# Build with strict mode override
zlup build --strict true

# Build to different target
zlup build --target hugr
```

## Direct Compilation

### Compile

```bash
# Compile to SLR-AST JSON (default)
zlup compile program.zlp

# Compile to specific target
zlup compile --target slr program.zlp
zlup compile --target qasm program.zlp
zlup compile --target hugr program.zlp

# Read from stdin
echo 'fn main() -> unit { return; }' | zlup compile -
```

### Check

```bash
# Check syntax and semantics
zlup check program.zlp

# Strict mode (NASA Power of 10 checks)
zlup check --strict program.zlp
```

## Formatting

Format Zlup source files to a consistent style.

```bash
# Format and print to stdout
zlup fmt program.zlp

# Format in place
zlup fmt --write program.zlp
zlup fmt -w program.zlp

# Check formatting (exit 1 if not formatted)
zlup fmt --check program.zlp

# Read from stdin
cat program.zlp | zlup fmt -
```

## Linting

Zlup includes a linter with automatic fix capabilities, similar to ruff for Python.

### Basic Usage

```bash
# Run linter (strict mode by default)
zlup lint program.zlp

# Run with relaxed or minimal rules
zlup lint --level relaxed program.zlp
zlup lint --level minimal program.zlp
```

### Auto-Fix

```bash
# Preview fixes without applying (shows diff)
zlup lint --diff program.zlp

# Apply safe fixes automatically
zlup lint --fix program.zlp

# Apply both safe and unsafe fixes
zlup lint --fix --unsafe-fixes program.zlp
```

### Output Options

```bash
# Show statistics only
zlup lint --statistics program.zlp

# Treat warnings as errors
zlup lint --deny-warnings program.zlp

# Output formats
zlup lint --format pretty program.zlp   # Human-readable (default)
zlup lint --format json program.zlp     # JSON for tooling
zlup lint --format compact program.zlp  # One line per diagnostic
```

### Fix Safety Levels

| Safety | Description | Examples |
|--------|-------------|----------|
| Safe | Preserves semantics, guaranteed correct | Prefix unused vars with `_`, decimal to fraction |
| Unsafe | Probably safe, may need manual verification | Complex refactorings |

### Common Lint Rules

| Rule | Description |
|------|-------------|
| `unused_variable` | Detects unused variables (fixable: prefix with `_`) |
| `unused_function` | Detects unused non-public functions |
| `function_naming` | Enforces snake_case for function names |
| `variable_naming` | Enforces snake_case for variable names |
| `type_naming` | Enforces PascalCase for type names |
| `deep_nesting` | Warns on excessive nesting depth (NASA PoT Rule 1) |
| `low_assertion_density` | Warns when code lacks assertions (NASA PoT Rule 5) |
| `prefer_fraction_turns` | Suggests exact fractions over decimals (fixable) |
| `prefer_turns_over_radians` | Suggests turns over radians for precision |

## Parallelism Analysis

Analyze Zlup programs for parallelism opportunities:

```bash
# Analyze a program
zlup analyze program.zlp

# JSON output for tooling
zlup analyze program.zlp --format json

# Verbose output with dependency graph
zlup analyze program.zlp --verbose

# Read from stdin
cat program.zlp | zlup analyze -
```

The analyzer identifies:
- Qubit allocator lifetimes and scopes
- Operation dependencies (qubit and data dependencies)
- Parallel execution layers (operations that can run simultaneously)

## Expression Evaluation

Quick evaluation of expressions for experimentation:

```bash
# Evaluate arithmetic
zlup eval "2 + 3 * 4"      # 14

# Evaluate comparisons
zlup eval "10 > 5"         # true

# Strings
zlup eval '"hello"'        # "hello"

# Read from stdin
echo "100 / 4" | zlup eval -

# Verbose mode (show AST)
zlup eval --verbose "1 + 2"
```

## Project Configuration

Zlup projects use a `zlup.toml` file for configuration:

```toml
[package]
name = "my-quantum-program"
version = "0.1.0"
entry = "main.zlp"
description = "A quantum application"
authors = ["Alice"]

[build]
strict = false          # Enable NASA Power of 10 strict checks
target = "slr"          # Default: "slr" or "hugr"
output_dir = "build"    # Output directory
```

## Compilation Targets

Zlup can compile to multiple targets:

```
Zlup Source (.zlp)
        |
        v
    Zlup AST
        |
    +---+---+
    |       |
    v       v
SLR-AST   HUGR
    |       |
    v       v
 Guppy   Native
 QASM    Execution
```

| Target | Description |
|--------|-------------|
| `slr` | SLR-AST JSON for Python/PECOS bridge (default) |
| `qasm` | OpenQASM 2.0 output |
| `hugr` | HUGR IR for native execution |

## Debugging

### Parse AST

Dump the parsed AST for debugging and development:

```bash
# Debug format (Rust Debug output)
zlup parse program.zlp

# JSON format
zlup parse --format json program.zlp
zlup parse -f json program.zlp

# Read from stdin
echo 'fn main() -> unit { return; }' | zlup parse -
```
