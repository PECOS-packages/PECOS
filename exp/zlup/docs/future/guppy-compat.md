# Guppy-Zlup Compatibility Layer

This document outlines the strategy for Guppy ↔ Zlup interoperability.

## Key Insight: The Linter is Valuable Independently

Even if Zlup never sees wide adoption, a **Guppy linter enforcing NASA Power of 10 constraints** would be valuable on its own. It would:

- Improve reliability of production Guppy code
- Catch common bugs (unbounded loops, recursion, dynamic allocation)
- Establish best practices for QEC code quality
- Create a "reliable Guppy" subset for mission-critical applications

**The linter is the contribution, Zlup conversion is a bonus.**

## The Challenge

- Guppy is Python-embedded, Zlup is Rust-native
- Direct FFI between Python and Zlup is complex
- Zlup is more restrictive than Guppy (NASA Power of 10)
- Not all Guppy programs can become Zlup programs

## Strategy: NASA Power of 10 Guppy (with Optional Zlup Conversion)

Rather than trying to make arbitrary Guppy work with Zlup, we define a **constrained subset of Guppy** that maps cleanly to Zlup. This is enforced by a linter.

```
┌─────────────────────────────────────────────────────────────┐
│                    Full Guppy                               │
│  (Python-embedded, linear types, full flexibility)          │
└─────────────────────────┬───────────────────────────────────┘
                          │ guppy-lint --zlup-compat
┌─────────────────────────▼───────────────────────────────────┐
│              "Zlup-Compatible Guppy"                        │
│  • Bounded loops only (no while True)                       │
│  • No recursion                                             │
│  • Explicit resource management                             │
│  • Fixed allocations at function entry                      │
│  • No dynamic Python features                               │
└─────────────────────────┬───────────────────────────────────┘
                          │ guppy-to-zlup (mechanical transform)
┌─────────────────────────▼───────────────────────────────────┐
│                       Zlup                                  │
└─────────────────────────────────────────────────────────────┘
```

## Zlup-Compatible Guppy Constraints

### 1. Bounded Loops Only

```python
# ❌ NOT Zlup-compatible
while condition:
    do_something()

# ❌ NOT Zlup-compatible
while True:
    if done: break

# ✅ Zlup-compatible
for i in range(100):
    do_something()

# ✅ Zlup-compatible (with early exit)
for i in range(100):
    if done: break
```

**Lint message:**
```
error[ZLUP001]: unbounded loop not Zlup-compatible
  --> circuit.py:42
   |
42 |     while condition:
   |     ^^^^^ use bounded `for i in range(N)` instead
   |
help: replace with bounded loop
   |
42 |     for _ in range(MAX_ITERATIONS):
43 |         if not condition: break
```

### 2. No Recursion

```python
# ❌ NOT Zlup-compatible
def factorial(n):
    if n <= 1: return 1
    return n * factorial(n - 1)

# ✅ Zlup-compatible
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
```

**Lint message:**
```
error[ZLUP002]: recursive function not Zlup-compatible
  --> math.py:5
   |
 5 |     return n * factorial(n - 1)
   |                ^^^^^^^^^ recursive call
   |
help: convert to iterative form with bounded loop
```

### 3. Fixed Allocations

```python
# ❌ NOT Zlup-compatible (dynamic allocation in loop)
for round in range(100):
    qubits = allocate(compute_size(round))  # Size varies!

# ✅ Zlup-compatible (fixed allocation)
qubits = allocate(MAX_SIZE)
for round in range(100):
    use_qubits(qubits[:needed_size(round)])
```

### 4. No Dynamic Python Features

```python
# ❌ NOT Zlup-compatible
gate_name = "h" if condition else "x"
getattr(circuit, gate_name)(qubit)  # Dynamic dispatch

# ✅ Zlup-compatible
if condition:
    circuit.h(qubit)
else:
    circuit.x(qubit)
```

## Linter Implementation

The linter would be a Guppy plugin or standalone tool:

```bash
# Check if Guppy code is Zlup-compatible
guppy-lint --zlup-compat circuit.py

# Generate Zlup from compatible Guppy
guppy-to-zlup circuit.py -o circuit.zlp
```

### Lint Rules

| Rule | Description | Severity |
|------|-------------|----------|
| ZLUP001 | Unbounded loop | Error |
| ZLUP002 | Recursive function | Error |
| ZLUP003 | Dynamic allocation in loop | Error |
| ZLUP004 | Dynamic dispatch | Error |
| ZLUP005 | Unchecked error | Warning |
| ZLUP006 | Missing type annotation | Warning |
| ZLUP007 | Complex control flow | Warning |

## Transformation Errors

When `guppy-to-zlup` cannot transform code, it provides actionable feedback:

```
error: cannot transform to Zlup
  --> syndrome.py:78
   |
78 |     while defects:
   |     ^^^^^ unbounded loop
   |
help: this pattern often indicates MWPM-style iteration
      consider moving this logic to a Rust decoder:

      1. Create a Rust function implementing the algorithm
      2. Export via zlup-ffi with #[zlup_export]
      3. Call from Zlup: `correction := mwpm_decode(syndrome);`

      See: docs/rust-integration.md
```

## HUGR as Intermediate

For cases where direct transformation isn't possible, HUGR provides an intermediate:

```
Guppy → HUGR → (optimize) → Zlup
```

This allows:
- Guppy-native optimizations before Zlup conversion
- Shared optimization passes between Guppy and Zlup
- Gradual migration path

## Benefits

1. **Gradual Adoption**: Teams can lint existing Guppy code toward Zlup compatibility
2. **Clear Errors**: When code can't be converted, users know exactly why
3. **Best of Both**: Research in Guppy, production in Zlup
4. **NASA Power of 10 for Guppy**: The linter brings reliability principles to Python QEC code

## Standalone Value: guppy-lint Without Zlup

The linter is useful **even without any Zlup conversion**:

```bash
# Just lint for reliability, no Zlup involved
guppy-lint --strict circuit.py
```

### Use Cases

| Use Case | Zlup Needed? | Value |
|----------|--------------|-------|
| Catch unbounded loops | No | Prevent hangs in production |
| Flag recursion | No | Predictable stack usage |
| Detect dynamic allocation in hot paths | No | Consistent memory usage |
| Enforce explicit error handling | No | Robust fault tolerance |
| Require type annotations | No | Better static analysis |
| **Convert to Zlup** | Yes | Rust-native execution |

### "Reliable Guppy" Subset

The linter effectively defines a **"Reliable Guppy"** subset:

```
┌─────────────────────────────────────────────────────────────┐
│                    Full Guppy                               │
│  (All Python flexibility, linear types)                     │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │            "Reliable Guppy"                           │  │
│  │  (Bounded, predictable, NASA Power of 10)             │  │
│  │                                                       │  │
│  │  • Production QEC systems                             │  │
│  │  • Mission-critical code                              │  │
│  │  • Code that needs formal analysis                    │  │
│  │                                                       │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │        Zlup-Convertible                         │  │  │
│  │  │  (Can mechanically transform to Zlup)           │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

Teams can choose their level:
1. **Full Guppy**: Maximum flexibility for research/prototyping
2. **Reliable Guppy**: NASA Power of 10 for production reliability
3. **Zlup-Convertible**: When Rust-native execution is needed

### Integration with Guppy Ecosystem

The linter could be:
- A **Guppy plugin** (`guppy.lint.nasa_p10`)
- A **pre-commit hook** for CI/CD
- An **IDE extension** for real-time feedback
- A **standalone tool** for auditing

This makes reliability opt-in and gradual - teams can adopt constraints incrementally without committing to Zlup.

## Implementation Roadmap

### Phase 1: Standalone Guppy Linter
- Implement lint rules for NASA Power of 10 constraints
- Good error messages with actionable suggestions
- Integration with existing Python tooling (pylint, flake8 plugin?)

### Phase 2: Guppy → Zlup Conversion (Optional)
- Mechanical transformation for compliant code
- Clear errors when conversion isn't possible
- HUGR as intermediate where helpful

### Phase 3: Round-Trip (Future)
- Zlup → Guppy for interop
- Shared HUGR representation
