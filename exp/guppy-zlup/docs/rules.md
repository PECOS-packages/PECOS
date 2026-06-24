# Lint Rules Reference

The Guppy linter enforces rules derived from NASA's Power of 10 coding
guidelines, adapted for quantum computing. These rules ensure programs are
safe, predictable, and suitable for execution on quantum hardware.

## Rule Summary

| Rule    | Severity | Description                          |
|---------|----------|--------------------------------------|
| ZLUP001 | Error    | Unbounded loops                      |
| ZLUP002 | Error    | Recursive function calls             |
| ZLUP003 | Error    | Dynamic memory allocation in loops   |
| ZLUP004 | Error    | Dynamic dispatch                     |
| ZLUP005 | Warning  | Unchecked error-prone operations     |
| ZLUP006 | Warning  | Missing type annotations             |
| ZLUP007 | Warning  | Excessive control flow complexity    |
| ZLUP008 | Warning  | Deep call nesting (>4 levels)        |
| ZLUP009 | Info     | Missing assertions in large functions|
| ZLUP010 | Warning  | Global mutable state                 |

---

## ZLUP001: Unbounded Loops

**Severity:** Error

**Rationale:** Quantum hardware has finite coherence time. Programs must
terminate within a bounded time, which requires all loops to have statically
determinable bounds.

### Triggers

- `while True:` loops
- `while 1:` loops
- While loops without a clear termination condition

### Examples

```python
# BAD: Unbounded loop
def bad():
    while True:  # ZLUP001: unbounded loop
        do_something()

# BAD: Effectively unbounded
def also_bad():
    while some_condition():  # ZLUP001: cannot prove termination
        do_something()

# GOOD: Bounded by break with finite iterations
def acceptable():
    for i in range(100):
        if done():
            break
        do_something()

# GOOD: Clear termination condition
def good():
    for i in range(n):  # Bounded by n
        do_something()
```

### Fix

Replace unbounded loops with bounded `for` loops or add explicit iteration limits.

---

## ZLUP002: Recursion

**Severity:** Error

**Rationale:** Recursive calls can lead to unbounded stack growth and
unpredictable execution time. Quantum programs must have statically
determinable resource usage.

### Triggers

- Direct recursion (function calls itself)
- Indirect recursion (A calls B, B calls A)

### Examples

```python
# BAD: Direct recursion
def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)  # ZLUP002: recursive call

# BAD: Indirect recursion
def ping(n: int) -> int:
    if n <= 0:
        return 0
    return pong(n - 1)  # ZLUP002: mutual recursion

def pong(n: int) -> int:
    return ping(n - 1)  # ZLUP002: mutual recursion

# GOOD: Iterative version
def factorial(n: int) -> int:
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
```

### Fix

Convert recursive algorithms to iterative versions using explicit loops.

---

## ZLUP003: Dynamic Allocation in Loops

**Severity:** Error

**Rationale:** Dynamic memory allocation inside loops can lead to unbounded
memory growth. For quantum programs, qubit allocation must be statically
determinable.

### Triggers

- `qubit[n]` allocation inside loop bodies
- List creation (`[]`, `list()`) inside loops
- `.append()` calls inside loops
- List comprehensions that grow unboundedly

### Examples

```python
# BAD: Qubit allocation in loop
def bad():
    for i in range(10):
        q = qubit[2]  # ZLUP003: allocation in loop
        h(q[0])

# BAD: List growth in loop
def also_bad():
    results = []
    for i in range(n):
        results.append(measure(q))  # ZLUP003: dynamic growth

# GOOD: Pre-allocate outside loop
def good():
    q = qubit[20]  # Allocate once
    for i in range(10):
        h(q[i * 2])

# GOOD: Fixed-size collection
def also_good(n: int):
    q = qubit[n]
    results = [0] * n  # Pre-sized
    for i in range(n):
        results[i] = measure(q[i])
```

### Fix

Move allocations outside loops. Pre-size collections before loop entry.

---

## ZLUP004: Dynamic Dispatch

**Severity:** Error

**Rationale:** Dynamic dispatch (runtime method resolution) makes control flow
unpredictable and complicates timing analysis on quantum hardware.

### Triggers

- `getattr()` calls
- `eval()` calls
- `exec()` calls
- Subscript-based function calls (`funcs[i]()`)

### Examples

```python
# BAD: Dynamic attribute access
def bad(obj, method_name: str):
    func = getattr(obj, method_name)  # ZLUP004: dynamic dispatch
    func()

# BAD: eval
def also_bad(code: str):
    eval(code)  # ZLUP004: eval

# BAD: Subscript dispatch
def dispatch(funcs, i: int):
    funcs[i]()  # ZLUP004: dynamic dispatch

# GOOD: Static dispatch
def good(obj):
    obj.known_method()  # Static, known at compile time

# GOOD: Match/if for dispatch
def also_good(choice: int):
    if choice == 0:
        func_a()
    elif choice == 1:
        func_b()
```

### Fix

Use static method calls or explicit if/match statements for dispatch.

---

## ZLUP005: Unchecked Error-Prone Operations

**Severity:** Warning

**Rationale:** Operations that can fail at runtime should be wrapped in
error handling to prevent unexpected crashes.

### Triggers

- Division operations not in try/except blocks
- `int()` conversions not in try/except blocks
- Other operations that may raise exceptions

### Examples

```python
# WARNING: Division might fail
def bad(a: int, b: int) -> int:
    return a / b  # ZLUP005: possible division by zero

# OK: Division by literal
def ok(a: int) -> int:
    return a / 2  # Literal cannot be zero

# GOOD: Wrapped in try/except
def good(a: int, b: int) -> int:
    try:
        return a / b
    except ZeroDivisionError:
        return 0

# GOOD: Explicit check
def also_good(a: int, b: int) -> int:
    if b == 0:
        return 0
    return a / b
```

### Fix

Wrap error-prone operations in try/except or add explicit validation.

---

## ZLUP006: Missing Type Annotations

**Severity:** Warning

**Rationale:** Type annotations enable static analysis and catch errors at
compile time rather than runtime. They're especially important for quantum
programs where runtime debugging is difficult.

### Triggers

- Functions without return type annotations
- Parameters without type annotations
- Exception: `self` parameter (implicitly typed)

### Examples

```python
# WARNING: Missing return type
def bad(x: int):  # ZLUP006: missing return type
    return x + 1

# WARNING: Missing parameter type
def also_bad(x) -> int:  # ZLUP006: missing parameter type
    return x + 1

# GOOD: Fully annotated
def good(x: int) -> int:
    return x + 1

# OK: self doesn't need annotation
class Foo:
    def method(self, x: int) -> int:  # self is fine
        return x
```

### Fix

Add type annotations to all function parameters and return types.

---

## ZLUP007: Excessive Control Flow Complexity

**Severity:** Warning

**Rationale:** Complex control flow is harder to analyze, test, and reason
about. High cyclomatic complexity correlates with bugs.

### Triggers

- Functions with cyclomatic complexity > 10 (configurable)
- Deeply nested control structures

### Complexity Calculation

Each of the following adds 1 to complexity:
- `if` statement
- `elif` clause
- `for` loop
- `while` loop
- `except` handler
- `and` operator
- `or` operator
- Conditional expression (`x if cond else y`)
- Comprehension with `if` clause

### Examples

```python
# WARNING: Too complex
def bad(x: int, y: int, z: int) -> int:
    if x > 0:
        if y > 0:
            if z > 0:
                return 1
            else:
                if x > y:
                    return 2
                else:
                    return 3
        else:
            for i in range(x):
                if i > y:
                    return 4
    # ... continues with more nesting
    # ZLUP007: complexity > 10

# GOOD: Refactored into smaller functions
def handle_positive_z(x: int, y: int) -> int:
    return 2 if x > y else 3

def handle_positive_y(x: int, y: int, z: int) -> int:
    if z > 0:
        return 1
    return handle_positive_z(x, y)

def good(x: int, y: int, z: int) -> int:
    if x <= 0:
        return 0
    if y > 0:
        return handle_positive_y(x, y, z)
    return handle_negative_y(x, y)
```

### Fix

Break complex functions into smaller, focused helper functions.

---

## ZLUP008: Deep Call Nesting

**Severity:** Warning

**Rationale:** Deeply nested function calls (calls within calls within calls)
make code hard to verify and debug. NASA Power of 10 Rule 5 recommends keeping
assertions and expressions simple and verifiable.

### Triggers

- Function calls nested more than 4 levels deep (configurable)
- Chains like `a(b(c(d(e()))))`

### Examples

```python
# WARNING: Call depth exceeds 4
def bad():
    result = a(b(c(d(e()))))  # ZLUP008: too deep

# GOOD: Intermediate variables
def good():
    e_result = e()
    d_result = d(e_result)
    c_result = c(d_result)
    b_result = b(c_result)
    result = a(b_result)
```

### Fix

Extract nested calls into named intermediate variables. This improves
readability, debuggability, and makes assertions on intermediate values possible.

---

## ZLUP009: Missing Assertions

**Severity:** Info

**Rationale:** NASA Power of 10 Rule 5 requires assertions to check invariants.
Non-trivial functions should contain assertions to validate preconditions,
postconditions, and invariants.

### Triggers

- Functions with 5+ statements that contain no `assert` statements
- Exception: test functions (`test_*`) and dunder methods (`__*__`)

### Examples

```python
# INFO: Function has 6 statements but no assertions
def process(data):
    x = 1
    y = 2
    z = 3
    a = x + y
    b = y + z
    return a + b  # ZLUP009: no assertions

# GOOD: Has assertion for precondition
def process(data):
    assert data is not None, "data must not be None"
    x = 1
    y = 2
    z = 3
    a = x + y
    b = y + z
    return a + b
```

### Fix

Add `assert` statements to validate preconditions, postconditions, or
loop invariants. Even simple assertions like `assert data is not None`
help document assumptions and catch errors early.

---

## ZLUP010: Global Mutable State

**Severity:** Warning

**Rationale:** Mutable global state creates hidden dependencies between
functions, makes programs harder to reason about, and can cause issues
in concurrent execution (relevant for quantum-classical hybrid programs).

### Triggers

- Module-level variable assignments (lowercase names)
- Use of `global` keyword inside functions
- Exception: UPPER_CASE constants are allowed

### Examples

```python
# WARNING: Mutable global state
counter = 0  # ZLUP010: lowercase module-level variable

def increment():
    global counter  # ZLUP010: global keyword
    counter += 1

# GOOD: Constants allowed
MAX_SIZE = 100
DEFAULT_VALUE = 42

# GOOD: Pass state explicitly
def increment(counter: int) -> int:
    return counter + 1
```

### Fix

Use UPPER_CASE names for true constants. For mutable state, pass values
explicitly through function parameters and return values.

---

## Configuration

Rules can be configured in `pyproject.toml`:

```toml
[tool.guppy-zlup]
# Disable specific rules
disabled_rules = ["ZLUP005"]

# Adjust complexity threshold
max_complexity = 15

# Treat warnings as errors
warnings_as_errors = true
```

Or via command line:

```bash
guppy-zlup check program.py --disable ZLUP005 --max-complexity 15
```
