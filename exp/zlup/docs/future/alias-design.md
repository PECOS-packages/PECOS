# Alias Design Notes

> **Status:** MVP implemented (February 2026)

This document explores adding an `alias` keyword to Zlup for creating named views
into existing data structures, particularly qubit registers.

## Motivation

In QEC code, it's common to partition a qubit register into logical regions:

```zlup_nocheck
// Current approach: use slices directly
q := qalloc(9);
data := q[0..1];       // Is this a copy or a view? Unclear at a glance
x_ancilla := q[1..5];
z_ancilla := q[5..9];
```

Problems with the current approach:
1. **Intent unclear** - Is `data` a view or a copy?
2. **No overlap checking** - Nothing prevents `q[0..3]` and `q[2..5]` aliases
3. **Lifetime implicit** - Relationship to source not explicit in syntax

## Proposed Syntax

```zlup_nocheck
// Immutable alias (default)
alias data := q[0..1];

// Mutable alias (explicit)
mut alias x_ancilla := q[1..5];

// Multiple aliases
alias {
    data := q[0..1],
    x_ancilla := q[1..5],
    z_ancilla := q[5..9],
}
```

## Semantics

### Immutability by Default

```zlup_nocheck
q := qalloc(4);
alias view := q[0..2];

h view[0];           // OK: quantum ops through immutable alias
view = q[2..4];      // ERROR: cannot reassign immutable alias
```

Mutable aliases can be reassigned:

```zlup_nocheck
mut alias current := q[0..2];
h current[0];
current = q[2..4];   // OK: mutable alias
h current[0];        // Now operates on q[2]
```

### Lifetime Binding

Aliases are bound to their source's lifetime:

```zlup_nocheck
fn bad() -> alias []u8 {
    arr := [1, 2, 3, 4];
    alias view := arr[0..2];
    return view;         // ERROR: alias would outlive source
}

fn ok(arr: []u8) -> alias []u8 {
    alias view := arr[0..2];
    return view;         // OK: source outlives function
}
```

### Overlap Checking

**Option A: Error on overlap (strict)**
```zlup_nocheck
q := qalloc(4);
alias a := q[0..2];
alias b := q[1..3];  // ERROR: overlaps with 'a'
```

**Option B: Warning on overlap (permissive)**
```zlup_nocheck
q := qalloc(4);
alias a := q[0..2];
alias b := q[1..3];  // WARNING: overlaps with 'a', parallel ops may conflict
```

**Option C: Error only for mutable overlap**
```zlup_nocheck
q := qalloc(4);
alias a := q[0..2];       // immutable
alias b := q[1..3];       // immutable - OK, both read-only

mut alias c := q[0..2];   // mutable
mut alias d := q[1..3];   // ERROR: mutable overlap with 'c'
```

**Recommendation:** Start with Option C - mutable overlap is an error, immutable
overlap is allowed. This matches Rust's borrowing rules conceptually.

## Difference from Slices

| Aspect | Slice (`x := arr[0..2]`) | Alias (`alias x := arr[0..2]`) |
|--------|--------------------------|--------------------------------|
| Intent | Ambiguous | Explicitly a view |
| Overlap check | None | Static analysis possible |
| Lifetime | Implicit | Explicit in type system |
| Reassignment | Always mutable | Immutable by default |

An alias IS a slice underneath, but with:
1. Explicit "this is a view" semantics
2. Compiler tracking for overlap analysis
3. Immutability by default

## AST Representation

```rust
/// Alias binding - creates a named view into existing data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AliasBinding {
    /// Name of the alias
    pub name: String,
    /// The source expression (must be slice-able)
    pub source: Expr,
    /// Whether this alias is mutable (can be reassigned)
    pub is_mutable: bool,
    /// Optional type annotation
    pub ty: Option<TypeExpr>,
    pub location: Option<SourceLocation>,
}

// In Stmt enum:
pub enum Stmt {
    // ... existing variants ...
    Alias(AliasBinding),
    AliasGroup(Vec<AliasBinding>),  // For grouped aliases
}
```

## Semantic Analysis

### Alias Tracking

The semantic analyzer tracks aliases and their source relationships:

```rust
struct AliasInfo {
    name: String,
    source: String,           // Name of source variable
    range: Option<Range>,     // Static range if known
    is_mutable: bool,
}

// In SemanticAnalyzer:
aliases: BTreeMap<String, AliasInfo>,
```

### Overlap Detection

For static ranges, detect overlaps at compile time:

```rust
fn ranges_overlap(a: &Range, b: &Range) -> bool {
    // [a.start, a.end) overlaps [b.start, b.end)?
    a.start < b.end && b.start < a.end
}

fn check_alias_overlap(&self, new_alias: &AliasInfo) -> Option<SemanticError> {
    for existing in self.aliases.values() {
        if existing.source == new_alias.source {
            if let (Some(r1), Some(r2)) = (&existing.range, &new_alias.range) {
                if ranges_overlap(r1, r2) {
                    if existing.is_mutable || new_alias.is_mutable {
                        return Some(SemanticError::OverlappingMutableAlias { ... });
                    }
                }
            }
        }
    }
    None
}
```

### Dynamic Ranges

For runtime-computed ranges, overlap checking happens at runtime or is skipped:

```zlup_nocheck
fn partition(q: [n]qubit, split: usize) -> unit {
    alias left := q[0..split];      // Range not known at compile time
    alias right := q[split..n];     // Could overlap if split > n
    // Runtime check or trust the programmer?
}
```

Options:
1. **Require comptime ranges** for overlap checking
2. **Insert runtime checks** for dynamic ranges
3. **Trust programmer** for dynamic ranges (document the risk)

## Integration with Parallelism Analysis

Aliases provide explicit partitioning information to the parallelism analyzer:

```zlup_nocheck
q := qalloc(9);
alias {
    data := q[0..1],
    x_ancilla := q[1..5],
    z_ancilla := q[5..9],
}

// Analyzer knows these are disjoint, can parallelize:
h data[0];           // Independent
h x_ancilla[0];      // Independent
h z_ancilla[0];      // Independent
```

The `alias` block makes the partitioning explicit and verifiable.

## Examples

### QEC Syndrome Extraction

```zlup_nocheck
fn surface_code_round(q: [13]qubit) -> unit {
    // Explicit partitioning with aliases
    alias {
        data := q[0..9],
        x_stabilizers := q[9..11],
        z_stabilizers := q[11..13],
    }

    // Operations on disjoint regions - can parallelize
    prepare_data(data);
    extract_x_syndrome(data, x_stabilizers);
    extract_z_syndrome(data, z_stabilizers);

    // Measure stabilizers
    x_syndrome := mz([2]u1) x_stabilizers;
    z_syndrome := mz([2]u1) z_stabilizers;

    result("syndrome/x", x_syndrome);
    result("syndrome/z", z_syndrome);
}
```

### Sliding Window

```zlup_nocheck
fn sliding_window(arr: []u8, window_size: usize) -> unit {
    for i in 0..(arr.len() - window_size) {
        alias window := arr[i..i+window_size];
        process(window);
    }
}
```

### Temporary Mutable View

```zlup_nocheck
fn initialize_register(q: [8]qubit) -> unit {
    // Mutable alias for initialization phase
    mut alias current := q[0..4];
    initialize_block(current);

    current = q[4..8];
    initialize_block(current);
}
```

## Open Questions

1. **Overlap policy:** Error, warning, or context-dependent?

2. **Runtime checks:** For dynamic ranges, should we insert bounds checks?

3. **Alias of alias:** Should this be allowed?
   ```zlup_nocheck
   alias a := q[0..4];
   alias b := a[0..2];  // Alias of alias?
   ```

4. **Alias in function signatures:**
   ```zlup_nocheck
   fn process(alias data: []qubit) -> unit { ... }
   // vs
   fn process(data: &[]qubit) -> unit { ... }
   ```

5. **Interaction with borrowing:** How does `alias` relate to `&` and `&mut`?

## Minimal First Version (MVP)

Start with a constrained version to validate the concept before adding complexity.

### MVP Scope

**Include:**
- `alias name := slice_expr;` - immutable only, no `mut alias`
- Static range slices only: `alias x := q[0..4];`
- Same-scope only (no passing aliases to/from functions)
- Overlap checking within same source variable
- Error on any overlap (simpler than mutable-only rule)

**Exclude (for later):**
- `mut alias` (mutable aliases)
- `alias { }` grouped syntax
- Dynamic ranges
- Alias as function parameter/return type
- Alias of alias

### MVP Syntax

```zlup_nocheck
fn example() -> unit {
    q := qalloc(8);

    // Simple aliases with static ranges
    alias data := q[0..4];
    alias ancilla := q[4..8];

    // Use like slices
    h data[0];
    cx (data[0], ancilla[0]);

    // Overlap is an error
    alias overlap := q[2..6];  // ERROR: overlaps with 'data' and 'ancilla'

    return;
}
```

### MVP Semantics

1. **Immutable binding** - Cannot reassign: `data = q[0..2];` is an error
2. **View semantics** - Alias is a view, not a copy
3. **Lifetime** - Alias lives until end of scope (same as source)
4. **Static only** - Range bounds must be comptime-known for overlap checking

### MVP Error Messages

```
error: overlapping alias
  --> example.zlp:8:5
   |
 5 |     alias data := q[0..4];
   |           ---- first alias covers q[0..4]
 8 |     alias overlap := q[2..6];
   |           ^^^^^^^ overlaps with 'data' at indices 2..4
   |
   = help: use non-overlapping ranges or access q directly
```

### MVP Implementation Estimate

- Parser: ~50 lines (new `alias` statement)
- AST: ~20 lines (AliasBinding struct)
- Semantic: ~100 lines (overlap checking, symbol tracking)
- Tests: ~100 lines

Total: ~270 lines, relatively low risk.

### Graduation Criteria

Expand beyond MVP when:
1. MVP is stable and tested
2. Real QEC code shows need for `mut alias` or function passing
3. User feedback indicates grouped syntax would help

---

## Full Implementation Plan

1. **Phase 1: Parser**
   - Add `alias` keyword to reserved words
   - Parse `alias name := expr;` and `mut alias name := expr;`
   - Parse `alias { ... }` block syntax

2. **Phase 2: AST**
   - Add `AliasBinding` struct
   - Add `Stmt::Alias` and `Stmt::AliasGroup` variants

3. **Phase 3: Semantic Analysis**
   - Track aliases in symbol table (new `SymbolKind::Alias`)
   - Implement lifetime checking (alias can't outlive source)
   - Implement overlap detection for static ranges

4. **Phase 4: Code Generation**
   - Aliases compile to slice references
   - No runtime overhead for immutable aliases

5. **Phase 5: Parallelism Analysis Integration**
   - Use alias info for more precise dependency tracking
   - Alias blocks provide explicit partitioning hints

## Alternatives Considered

### Just Use Slices
Keep current behavior, document that slices are views.
- Pro: No new syntax
- Con: Intent unclear, no overlap checking

### Borrow Syntax (`&`)
Use Rust-style borrowing more explicitly.
- Pro: Familiar to Rust users
- Con: More complex, might not fit Zlup's simpler model

### Named Regions in Allocator
```zlup_nocheck
q := qalloc(9) {
    data: 0..1,
    x_ancilla: 1..5,
    z_ancilla: 5..9,
};
```
- Pro: Declaration and partitioning together
- Con: Only works for allocators, not general slices

## Summary

The `alias` keyword provides:
1. **Explicit intent** - "this is a view, not a copy"
2. **Static safety** - overlap detection for mutable aliases
3. **Immutability by default** - matches Zlup's philosophy
4. **Parallelism hints** - explicit partitioning aids analysis

It's essentially slices with semantic meaning and compiler support for safety checks.
