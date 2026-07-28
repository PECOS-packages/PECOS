# Zlup Error Messages Guide

This guide explains common error messages and how to fix them.

## Parse Errors

Parse errors occur when the code doesn't match the expected syntax.

### "expected an identifier"

**Cause:** A name (variable, function, type) was expected but something else was found.

```zlup_nocheck
// Bad
fn 123() -> unit { }    // Numbers can't be function names

// Good
fn my_func() -> unit { }
```

### "expected a type"

**Cause:** A type annotation was expected after `:`.

```zlup_nocheck
// Bad
x: = 42;               // Missing type

// Good
x := 42;               // Infer type
x: u32 = 42;           // Explicit type
```

### "expected ';'"

**Cause:** Statements must end with semicolons.

```zlup_nocheck
// Bad
x := 42
y := 10

// Good
x := 42;
y := 10;
```

### "expected '{'"

**Cause:** Blocks require curly braces.

```zlup_nocheck
// Bad
if condition
    do_something();

// Good
if condition {
    do_something();
}
```

### "expected ')'" / "expected ']'"

**Cause:** Unmatched parentheses or brackets.

```zlup_nocheck
// Bad
result := add(1, 2;
array := [1, 2, 3;

// Good
result := add(1, 2);
array := [1, 2, 3];
```

### "expected ':=' for binding or '=' for assignment"

**Cause:** Incorrect binding syntax.

```zlup_nocheck
// Bad
x = 42;                // This is assignment, not binding

// Good
x := 42;               // Binding (creates new variable)
mut y := 0;
y = 42;                // Assignment (to existing variable)
```

### "expected function parameter (name: Type)"

**Cause:** Function parameters must have names and types.

```zlup_nocheck
// Bad
fn add(u32, u32) -> u32 { }

// Good
fn add(a: u32, b: u32) -> u32 { }
```

### "expected '-> T' return type"

**Cause:** Function return type syntax uses arrow.

```zlup_nocheck
// Bad
fn add(a: u32, b: u32) u32 { }

// Good
fn add(a: u32, b: u32) -> u32 { }
```

---

## Semantic Errors

Semantic errors occur when code is syntactically valid but logically incorrect.

### "undefined symbol 'name'"

**Cause:** Using a variable, function, or type that hasn't been declared.

```zlup_nocheck
// Bad
fn main() -> unit {
    y := x + 1;        // x not defined
    return;
}

// Good
fn main() -> unit {
    x := 10;
    y := x + 1;
    return;
}
```

### "symbol 'name' already defined"

**Cause:** Declaring the same name twice in the same scope.

```zlup_nocheck
// Bad
x := 1;
x := 2;                // Can't redeclare

// Good
x := 1;
y := 2;                // Different name

// Or use mut for reassignment
mut x := 1;
x = 2;                 // Assignment, not redeclaration
```

### "cannot assign to immutable variable 'name'"

**Cause:** Trying to reassign a variable that wasn't declared with `mut`.

```zlup_nocheck
// Bad
x := 1;
x = 2;                 // x is immutable!

// Good
mut x := 1;
x = 2;                 // x is mutable
```

### "unsafe blocks are forbidden"

**Cause:** Using an `unsafe` block without the `--allow-unsafe` flag.

```zlup_nocheck
// This will fail without --allow-unsafe
fn example() -> unit {
    unsafe {
        // ...
    }
    return;
}
```

**Solution:** Either:
1. Remove the unsafe block and rewrite using safe constructs
2. Compile with `--allow-unsafe` flag (for development/testing)

Production code should avoid unsafe blocks. They exist as an escape hatch for expert use cases.

### "type mismatch: expected T, found U"

**Cause:** Using a value of the wrong type.

```zlup_nocheck
// Bad
x: u32 = "hello";      // String is not u32

// Good
x: u32 = 42;
s: []const u8 = "hello";
```

### "cannot infer type for 'name'"

**Cause:** Type cannot be determined from context.

```zlup_nocheck
// Bad
x := undefined;        // What type?

// Good
x: u32 = undefined;    // Explicit type
```

---

## Gate Errors

### "gate 'G' requires N qubits, got M"

**Cause:** Wrong number of qubits for the gate.

```zlup_nocheck
// Bad - too few qubits
cx q[0];               // CX needs 2 qubits
ccx (q[0], q[1]);      // CCX (Toffoli) needs 3 qubits

// Bad - too many qubits
h (q[0], q[1]);        // H is single-qubit gate

// Good
h q[0];                // Single qubit
cx (q[0], q[1]);       // Two qubits (control, target)
ccx (q[0], q[1], q[2]); // Three qubits (Toffoli)
```

### "ambiguous target for multi-qubit gate"

**Cause:** Multi-qubit gates need explicit tuple or set syntax.

```zlup_nocheck
// Bad
cx q;                  // Which qubits?

// Good
cx (q[0], q[1]);       // Explicit pair
cx {(q[0], q[1]), (q[2], q[3])};  // Batch
```

### "invalid gate syntax: use 'gate target' instead"

**Cause:** Gates use space-separated syntax, not function call syntax.

```zlup_nocheck
// Bad (old syntax)
h(q[0]);
cx(q[0], q[1]);

// Good (current syntax)
h q[0];
cx (q[0], q[1]);
```

---

## Qubit Errors

### "qubit 'alloc[i]' is not prepared"

**Cause:** Using a qubit before preparing it, or after measurement without re-preparing.

```zlup_nocheck
// Bad - never prepared
q := qalloc(2);
h q[0];                // Qubit not prepared!

// Bad - used after measurement
q := qalloc(2);
pz q;
h q[0];
r := mz(u1) q[0];      // Measurement resets qubit state
h q[0];                // Error: qubit no longer prepared!

// Good - prepare before use
q := qalloc(2);
pz q;                  // Prepare first
h q[0];

// Good - re-prepare after measurement
q := qalloc(2);
pz q;
h q[0];
r := mz(u1) q[0];
pz q[0];               // Re-prepare after measurement
h q[0];                // Now OK
```

### "qubit 'alloc[i]' is already prepared"

**Cause:** Preparing an already-prepared qubit (in strict mode).

```zlup_nocheck
// Bad
pz q[0];
pz q[0];               // Already prepared

// Good
pz q[0];
// ... use qubit ...
// Reset only if needed
```

### "qubit index N out of bounds for allocator (capacity: M)"

**Cause:** Accessing a qubit index beyond the allocator's capacity.

```zlup_nocheck
// Bad
q := qalloc(4);
h q[10];               // Only 0-3 available

// Good
q := qalloc(4);
h q[3];                // Index 0-3 valid
```

### "cannot call .child() on immutable allocator"

**Cause:** Parent allocator must be mutable to create children.

```zlup_nocheck
// Bad
q := qalloc(10);
data := q.child(5);    // q is immutable

// Good
mut q := qalloc(10);   // Make mutable
data := q.child(5);
```

### "qubit used multiple times within tick block"

**Cause:** Same qubit appears in multiple operations within one tick.

```zlup_nocheck
// Bad
tick {
    h q[0];
    x q[0];            // Same qubit!
}

// Good
tick { h q[0]; }
tick { x q[0]; }       // Different ticks
```

---

## Measurement Errors

### "invalid measurement type"

**Cause:** Measurement type must be u1, u8, u64, or arrays thereof.

```zlup_nocheck
// Bad
r: f64 = mz(f64) q[0];  // Float not valid

// Good
r: u1 = mz(u1) q[0];
```

### "measurement type mismatch: declared [N]T but measuring M qubits"

**Cause:** Array size doesn't match number of qubits.

```zlup_nocheck
// Bad
r: [4]u1 = mz([4]u1) [q[0], q[1]];  // Only 2 qubits

// Good
r: [2]u1 = mz([2]u1) [q[0], q[1]];
```

### "single qubit measurement requires scalar type"

**Cause:** Measuring one qubit requires u1 (or similar), not array.

```zlup_nocheck
// Bad
r: [1]u1 = mz([1]u1) q[0];  // Single qubit

// Good
r: u1 = mz(u1) q[0];
```

### "multiple qubit measurement requires array type"

**Cause:** Measuring multiple qubits requires array type.

```zlup_nocheck
// Bad
r: u1 = mz(u1) [q[0], q[1]];  // Multiple qubits

// Good
r: [2]u1 = mz([2]u1) [q[0], q[1]];
```

### "pack mode: type T has N bits but measuring M qubits"

**Cause:** Pack type doesn't have enough bits.

```zlup_nocheck
// Bad
r: u4 = mz(pack u4) [q[0], q[1], q[2], q[3], q[4]];  // 5 qubits, only 4 bits

// Good
r: u8 = mz(pack u8) [q[0], q[1], q[2], q[3], q[4]];  // 8 bits >= 5 qubits
```

---

## Control Flow Errors

### "unbounded loop detected"

**Cause:** Loops must have bounded iteration (NASA Power of 10).

```zlup_nocheck
// Bad
while condition { }    // while loops not allowed

// Good
for i in 0..100 {      // Bounded iteration
    if !condition { break; }
}
```

### "loop bound too large"

**Cause:** Loop iteration count exceeds maximum (in strict mode).

```zlup_nocheck
// Bad (if max is 1000)
for i in 0..1000000 { }

// Good
for i in 0..1000 { }
```

### "recursion detected in function"

**Cause:** Recursive calls are not allowed in strict mode (NASA Power of 10).

```zlup_nocheck
// Bad
fn factorial(n: u32) -> u32 {
    if n <= 1 { return 1; }
    return n * factorial(n - 1);  // Recursion!
}

// Good - use iteration
fn factorial(n: u32) -> u32 {
    mut result: u32 = 1;
    for i in 1..n+1 {
        result *= i;
    }
    return result;
}

// Alternative - use unsafe block (requires --allow-unsafe)
fn factorial(n: u32) -> u32 {
    unsafe {
        if n <= 1 { return 1; }
        return n * factorial(n - 1);
    }
}
```

### "mutual recursion detected"

**Cause:** Two or more functions call each other, forming a cycle.

```zlup_nocheck
// Bad
fn is_even(n: u32) -> bool {
    if n == 0 { return true; }
    return is_odd(n - 1);  // Calls is_odd
}

fn is_odd(n: u32) -> bool {
    if n == 0 { return false; }
    return is_even(n - 1);  // Calls is_even - cycle!
}

// Good - use iteration or combine into one function
fn is_even(n: u32) -> bool {
    return n % 2 == 0;
}
```

### "break outside of loop" / "continue outside of loop"

**Cause:** Using break/continue outside a loop context.

```zlup_nocheck
// Bad
fn main() -> unit {
    break;             // Not in a loop
    return;
}

// Good
fn main() -> unit {
    for i in 0..10 {
        if i == 5 { break; }
    }
    return;
}
```

### "missing return statement"

**Cause:** Function doesn't return on all paths.

```zlup_nocheck
// Bad
fn get_value(x: bool) -> u32 {
    if x {
        return 1;
    }
    // Missing return for else case!
}

// Good
fn get_value(x: bool) -> u32 {
    if x {
        return 1;
    } else {
        return 0;
    }
}
```

---

## Module Errors

### "module not found: name"

**Cause:** Imported module doesn't exist or isn't in search path.

```zlup_nocheck
// Bad
utils := @import("nonexistent.zlup");

// Solutions:
// 1. Check file exists
// 2. Set ZLUP_STDLIB_PATH for std imports
// 3. Check relative path is correct
```

---

## Tips for Debugging

1. **Read the full error message** - It often includes the exact location and suggestion.

2. **Check the line number** - The error points to where the problem was detected, which may be after where it was caused.

3. **Look for typos** - Common issues: `:=` vs `=`, missing semicolons, wrong brackets.

4. **Use `zlup check`** - Runs semantic analysis without full compilation.

5. **Try `zlup eval`** - Test small expressions interactively.

6. **Enable verbose mode** - Some commands have `--verbose` for more details.

7. **Simplify** - If you can't find the error, try removing code until it compiles, then add back piece by piece.
