# Development Notes

This document tracks recent development work on Zlup, implementation details, and context for contributors. Read this to understand recent changes, current test status, and suggested next tasks.

## What is Zlup?

Zlup is a quantum programming language with Zig-inspired semantics and Rust/Python-flavored syntax. It compiles to multiple backends (SLR-AST, HUGR; QASM and PHIR planned) and features:
- Static type checking with quantum-aware semantics
- Qubit state tracking (prepared/unprepared lifecycle)
- NASA Power of 10 compliance options (strict mode)
- Comptime evaluation
- Batch-oriented quantum gate API

## Recent Work Completed

### Error Handling Documentation (February 2026)

Created comprehensive error handling tutorial at `docs/tutorial-error-handling.md`:
- Faults vs Errors distinction with practical explanations
- Error sets and fault sets definitions and usage
- Error union syntax (`E!T` and `[]E!T`)
- `try` (collect mode) vs `try!` (propagate mode) blocks
- Try functions with both modes
- The explicit handling philosophy (no `?` operator)
- Four practical QEC examples:
  - Basic syndrome extraction
  - Full QEC round with threshold
  - Promoting faults to errors
  - Rich fault context inspection
- Quick reference and best practices

Updated `docs/tutorial.md` to reference the new detailed guide.

### Alias MVP Implementation (February 2026)

Added the `alias` keyword for creating safe slice views with overlap detection:

**Grammar changes:**
- Added `alias` to reserved keywords
- Added `alias_stmt = { "alias" ~ identifier ~ ":=" ~ expr ~ ";" }`

**AST changes:**
- Added `AliasBinding` struct with name, source, location
- Added `Stmt::Alias(AliasBinding)` variant

**Semantic analysis:**
- Added `AliasInfo` struct for tracking alias metadata (name, source, range, location)
- Added `aliases: BTreeMap<String, AliasInfo>` to SemanticAnalyzer
- Implemented `analyze_alias()` with overlap detection
- Added `extract_alias_source_info()` for parsing slice expressions
- Added `ranges_overlap()` helper for range intersection testing
- New error types: `OverlappingAlias`, `AliasSourceNotSlice`, `AliasRangeNotComptime`

**MVP scope:**
- Immutable aliases only (no `mut alias`)
- Static ranges only (must be comptime-evaluable)
- Error on any overlap (simpler than mutable-only rule)

**Files modified:**
- `src/zluppy.pest` - grammar rules
- `src/ast.rs` - AliasBinding, Stmt::Alias
- `src/parser.rs` - parse_alias_stmt()
- `src/semantic.rs` - AliasInfo, analyze_alias(), overlap detection
- `src/pretty.rs` - print_alias_binding()
- `src/comptime.rs` - eval alias as source expression

### Phase 3: Comptime Features (February 2026)

Completed the full comptime implementation plan with four features:

**1. Inline For Loop Unrolling:**
- `inline for i in 0..N { }` unrolls at compile time in `optimize.rs`
- Semantic validation in `semantic.rs`:
  - `InlineForRangeNotComptime` - range must be comptime-evaluable
  - `BreakInInlineFor` - break disallowed in inline for
  - `ContinueInInlineFor` - continue disallowed in inline for
- Recursive unrolling for nested inline for loops
- Variable substitution replaces loop variable with concrete values

**2. Advanced Builtins (snake_case naming):**
- `@type_info(T)` - returns TypeInfo struct with kind, name, fields, variants
- `@field_names(T)` - returns array of struct field name strings
- `@enum_fields(T)` - returns array of enum variant name strings
- `@type_from_info(info)` - constructs Type from TypeInfo (reverse of @type_info)
- Both snake_case and camelCase names supported; snake_case preferred

**3. Generic Type Instantiation:**
- Functions with `comptime` params get specialized versions at call sites
- Cache: `generic_instantiations: BTreeMap<(String, String), String>` in SemanticAnalyzer
- Mangled names: `make_array__u32_4` for `make_array(u32, 4)`
- Original declarations stored for cloning and substitution

**4. Comptime Function Memoization:**
- `memo_cache: BTreeMap<(String, String), ComptimeValue>` in ComptimeEvaluator
- Structural type serialization for cache keys (handles anonymous structs correctly)
- Avoids redundant evaluation when same function called with same args

**Files modified:**
- `src/comptime.rs` - memoization, advanced builtins, TypeInfoKind enum
- `src/semantic.rs` - inline for validation, generic instantiation, error types
- `src/optimize.rs` - inline for unrolling pass
- `src/zluppy.pest` - `comptime_modifier` rule for parser
- `src/parser.rs` - comptime parameter detection

### Error Handling Improvements (February 2026)

Enhanced the error handling type system with several improvements:

**Grammar Rule Ordering Fix:**
- Fixed parser ambiguity where `error_set_decl` and `fault_set_decl` were being parsed as bindings
- Reordered `top_level_decl` rules to try error/fault set declarations before general bindings
- Now `GateFaults := fault { ... }` correctly parses as `FaultSetDecl`, not `Binding`

**Error/Fault Set Union Support:**
- Error sets can be combined with `|` operator: `IoError | NetworkError`
- Fault sets can be combined similarly: `GateFaults | MeasurementFaults`
- Unions combine variant names, deduplicating common variants
- Type checking verifies operands are compatible error/fault sets

**Associated Data Types:**
- Error and fault variants can now have associated data types
- Example: `FileError := error { NotFound: struct { path: []u8 }, PermissionDenied };`
- Types are resolved during semantic analysis and stored in `Type::ErrorSet`/`Type::FaultSet`
- Imported modules store variant names without associated types (resolved locally)

**Try Block Type Inference:**
- `try {}` (collect mode) returns `[]AnyError!T` - slice of error unions
- `try! {}` (propagate mode) returns `AnyError!T` - single error union
- With catch clause, the result type is based on the body type
- Uses `Type::AnyError` as a conservative error type (full inference is future work)

### Parallelism Analysis Module (February 2026)

Added `src/analysis.rs` providing constraint-based parallelism analysis:

**Analysis Passes:**
- `AllocatorAnalysis`: Tracks allocator lifetimes and accessibility scopes
- `OperationTagger`: Tags each quantum operation with resources it touches
- `DependencyGraph`: Builds edges between dependent operations
- `parallel_layers()`: Extracts maximal independent operation sets

**CLI Integration:**
- `zlup analyze program.zlp` - Analyze a program for parallelism
- `--format json` - Machine-readable output
- `--verbose` - Show dependency graph

**Design Philosophy:**
Parallelism follows from type signatures and allocator ownership:
- No allocator param → pure classical → parallelizable with any quantum
- Different allocator params → different qubits → independent
- Scopes act as implicit barriers

### Safe-by-Constraint Memory Model (February 2026)

Implemented unconditional safety checks that don't require strict mode:

**Escape Analysis:**
- Functions cannot return references/slices to local variables
- `check_no_local_escape()` runs on all return statements
- Parameters are safe (caller owns the data)

**Recursion Prevention:**
- All recursion is rejected (direct and mutual)
- Tracking always enabled, not just in strict mode
- `RecursionTracker` with `enter_function()`/`exit_function()`

**@swap Validation:**
- Requires exactly 2 pointer arguments
- Types must match
- Proper error messages

### Slice Type System (February 2026)

**Type Distinction:**
- `[N]T` = Array (fixed size, known at compile time)
- `[]T` = Slice (dynamic view into memory)
- These are distinct types - arrays don't implicitly coerce to slices

**Slice Syntax:**
- `arr[0..5]` - slice from index 0 to 4
- `arr[2..]` - slice from index 2 to end
- `arr[..5]` - slice from start to index 4
- `arr[..]` - full slice (converts array to slice)

**Re-slicing:**
- `slice[0..2]` on a slice returns a slice
- `slice[0]` on a slice returns the element type
- Chained slicing works: `arr[0..10][2..5][0..2]`

**Implementation:**
- Parser: `parse_range_expr()` for range syntax
- Semantic: `[]T` parsed as `Type::Slice`, not `Type::Array { size: None }`
- Type checking: `is_slice_op` detection in Index expression

### Gate Naming Cleanup (February 2026)

**Removed:**
- `s` and `sdg` gate names (replaced with `sz`/`szdg` per PECOS naming conventions)
- `GateKind::S` and `GateKind::Sdg` from AST

**Correct Names:**
- `sz` = S gate (sqrt of Z)
- `szdg` = S-dagger gate

**Updated Files:**
- `ast.rs` - removed enum variants
- `parser.rs` - updated gate name mapping
- `semantic.rs` - updated get_gate_info()
- `optimize.rs` - updated cancellation rules
- All codegen files (slr.rs, qasm.rs, phir.rs)
- Documentation (syntax.md, tutorial.md, README.md)

### Deterministic Compilation (February 2026)

- Replaced `HashMap`/`HashSet` with `BTreeMap`/`BTreeSet` throughout
- Ensures consistent output order across compilations
- Files: semantic.rs, module.rs, comptime.rs, optimize.rs, linter.rs, build.rs, all codegens

### Duplicate Qubit Detection in Measurements (February 2026)

- Extended tick block duplicate detection to include `Expr::Measure`
- Measurements on the same qubit in parallel are now caught
- Added to `collect_qubit_ids_from_expr()`

### Boolean Literal Parsing Fix

Fixed a parser issue where boolean literals on the LEFT side of `and`/`or` operators failed to parse (e.g., `true and y` failed but `y and true` worked).

**Root cause**: Pest's implicit `WHITESPACE` rule inserted whitespace matching between elements, which interfered with keyword-based operator matching when the left operand was a keyword-like literal (`true`, `false`).

**Solution**: Created a compound-atomic `atom` rule in `zluppy.pest` to wrap leaf expressions (literals, identifiers) that don't recursively contain `expr`. This prevents implicit whitespace from interfering with keyword operator matching while keeping recursive expressions (like `paren_expr`) outside the compound-atomic context.

**Files modified:**
- `zluppy.pest`: Added `atom` rule with compound-atomic modifier (`${ }`)
- `parser.rs`: Added `Rule::atom` handling
- `optimize.rs`: Removed outdated comments about the parsing limitation

### Advanced Optimization Framework

Added a comprehensive optimization framework with QEC-aware barriers:

**Optimization passes in `src/optimize.rs`:**
- Constant folding (arithmetic, boolean, comparison expressions)
- Dead code elimination (unreachable code after return/break)
- Gate cancellation (self-inverse gates like H·H = I, X·X = I)

**Optimization barriers for QEC:**
- `@preserve` - Prevent any optimization on marked operations
- `@timing` - Preserve timing relationships
- `@identity` - Keep intentional identity operations
- `@noopt` - Disable all optimizations in scope
- `@round(n)` - QEC round tracking, prevents cross-round gate cancellation

**Tick blocks as barriers:**
- `tick {}` blocks always act as optimization barriers
- Operations cannot be moved across tick boundaries
- Nested tick blocks are now disallowed (semantic error)

### Build System Infrastructure

Added `build.zlp` infrastructure in `src/build.rs`:
- `Target` enum: x86_64, aarch64, wasm32, etc.
- `Optimize` enum: Debug, ReleaseSafe, ReleaseFast, ReleaseSmall
- `Build` struct with addExecutable, addLibrary, addTest, addStep methods
- `BuildRunner` for executing build steps
- Support for `-Dname=value` options

### 1. Numeric Literal Type Suffixes
- Added support for type suffixes on numeric literals: `42u32`, `1.5f32`, `0xFF_u16`
- Files modified: `zluppy.pest`, `parser.rs`, `ast.rs`, `semantic.rs`
- Suffixes are parsed and used for type inference in expressions

### 2. Strict Mode Qubit Duplicate Detection in Tick Blocks
- Added `DuplicateQubitInTick` error for detecting when the same qubit is used multiple times in a tick block
- Tick blocks represent parallel quantum operations - same qubit can't be targeted twice
- Only enforced in strict mode (NASA Power of 10 compliance)
- Added helper methods: `check_duplicate_qubits_in_tick()`, `collect_qubit_ids_from_stmt()`, `collect_qubit_ids_from_expr()`

### 3. Break/Continue Loop Validation
- Added `loop_depth` tracking to `SemanticAnalyzer`
- Added `BreakContinueOutsideLoop` error for break/continue statements outside loops
- Properly tracks nested loop depth for for loops

### 4. Type Inference from Range Expressions in For Loops
- Added `infer_for_range_type()` helper method
- For loops now infer the loop variable type from the range expression
- `for i in 0u32..10u32` correctly infers `i` as `u32`
- Supports both `ForRange::Range` and `ForRange::Collection`

### 5. Comptime Evaluation of Array Sizes
- Array type sizes are now evaluated at compile time
- `[10]u32` correctly resolves to `Type::Array { size: Some(10) }`
- Uses `ComptimeEvaluator` for evaluation

### 6. Const Propagation for Array Sizes
- Constants are now evaluated at comptime and stored in `comptime_values` HashMap
- Array sizes can reference values: `N := 10; mut arr: [N]u32`
- Supports chained references: `A := 4; B := A; mut arr: [B]u32`

## Current Test Status

All tests pass:
- 621 library tests (including alias, comptime, inline-for tests)
- 16 main binary tests
- 48 CLI integration tests
- 174 proptest integration tests
- 9 doctests
- **Total: 868 tests**

## Remaining TODOs in Code

### semantic.rs
1. ~~**Line ~1940**: `// TODO: Implement proper error set type`~~ **FIXED**
   - ~~Error values currently return `Type::Unknown`~~
   - ~~Should track actual error set types for better type safety~~
   - Error values now properly return `Type::ErrorSet` with the correct error set name and variants
   - Module-exported error sets now include variant names for proper lookup
   - Type checking now verifies error value assignments match expected error union types

2. **Line ~2835**: `// TODO: Extract actual function signature from AST`
   - Imported module functions have empty signatures
   - Should extract actual parameter and return types

### parser.rs
1. **Line ~1855**: `// TODO: Parse struct body for type definition`
   - Struct types used in declarations don't parse their full body

2. ~~**Lines ~2228, ~2236**: `// TODO: parse sentinel for [*:0] pointers`~~ **FIXED**
   - ~~Sentinel-terminated pointer syntax not fully implemented~~
   - Now supports arbitrary sentinel values: `[*:expr]T` for pointers, `[N:expr]T` for arrays
   - Added grammar rule `pointer_prefix` for flexible sentinel parsing
   - Examples: `[*:0]u8`, `[*:255]u8`, `[10:0]u8`, `[:0]u8`

## Roadmap

### Current State (February 2026)

**Completed:**
- Phase 1 (Core Language): Parser, basic types, control flow, quantum ops ✓
- Phase 2 (Type System): Type checking, allocator validation, error unions ✓
- Phase 3 (Comptime): Inline for, advanced builtins, generic instantiation, memoization ✓
- Alias MVP: Safe slice views with overlap detection ✓
- Tooling: LSP, formatter, linter ✓
- Codegens: SLR-AST, QASM, HUGR ✓

**In Progress:**
- Phase 4 (Integration): PyO3 bindings working, PHIR planned
- Phase 5 (Tooling): Doc generator and test runner planned

### Immediate Priorities (Polish & Stability)

1. **Parser error recovery** - Unknown gate names currently panic; should return proper errors

2. **Module function signature extraction** - Extract actual function signatures from imported modules for better type checking

3. **Improve type display names** - Better error messages for complex types (arrays, slices, functions)

### Short-term (Feature Completion)

4. **Array bounds checking** - Add compile-time bounds checking when array size is known

5. **Documentation generator** - Generate docs from `///` comments

6. **Built-in test runner** - `zlup test` command for running test blocks

### Architectural Decision: Custom Gates

The [Custom Gates Design](future/custom-gates-design.md) proposes a significant rethink of how gates work:

- **All gates become target-provided** (including `h`, `cx`, `pz`, `mz`)
- **`std.gates` becomes declarations**, not built-ins
- **Composite gates** allow full subroutines (prep, measurement, control flow)
- **Compile-time target validation** against target gate sets
- **IDE support** via project config and `@import("target")`

**Impact if pursued:**
- Changes how every quantum operation works
- Requires target definition system
- Affects IDE/LSP significantly
- More flexible but more explicit (requires imports)

**Decision needed:** Is this the right direction? If yes, it becomes high priority and affects subsequent work.

### Medium-term (Post-Decision)

If custom gates design is adopted:
- Implement target definition system
- Refactor `std.gates` as declarations
- Update IDE/LSP for target awareness
- Update all examples and docs

If not adopted:
- Continue with current built-in gate model
- Focus on PHIR codegen
- Guppy compatibility work

### Longer-term

- **PHIR codegen** - For PECOS simulator targeting
- **Guppy compatibility** - Linter for "Reliable Guppy" subset, mechanical conversion
- **Full stdlib** - math, bits, mem, qec, testing, ffi modules

### Lower Priority (Nice to Have)

- **Array-to-slice coercion** - Consider implicit array→slice in function arguments
- **Struct body parsing** - Parse full struct bodies when used as types

## Completed Features

### FFI Support (External Functions)
- Added `extern "C" fn name(params) -> type;` syntax for declaring external functions
- Supports calling conventions: `"C"` (C ABI), `"Rust"` (Rust ABI)
- Library linking via `@link("libname")` attribute
- Useful for integrating classical decoders (MWPM, Union-Find, etc.)
- Grammar: `extern_fn_decl` in `zluppy.pest`
- AST: `ExternFnDecl` struct in `ast.rs`
- SLR codegen generates `ExternDecl` and `ExternCall` nodes
- C-compatible type mapping: primitives (`u8`, `u32`, etc.), pointers (`[*]T`), arrays

### Gate Extensions
- Added SWAP, ISWAP, and CCX (Toffoli) gates to `GateKind` enum
- Updated SLR codegen to support new gates
- Three-qubit gate support (CCX has arity 3)

## Key Files

| File | Purpose |
|------|---------|
| `src/zluppy.pest` | PEG grammar definition |
| `src/parser.rs` | Parser implementation |
| `src/ast.rs` | AST node definitions |
| `src/semantic.rs` | Semantic analysis, type checking, symbol table |
| `src/comptime.rs` | Compile-time evaluation |
| `src/optimize.rs` | AST optimization passes (constant folding, DCE, gate cancellation) |
| `src/analysis.rs` | Parallelism analysis (allocators, dependencies, layers) |
| `src/build.rs` | Build system infrastructure for `build.zlp` |
| `src/codegen/slr.rs` | SLR-AST code generation |
| `src/codegen/qasm.rs` | OpenQASM code generation |
| `src/codegen/hugr.rs` | HUGR code generation |
| `src/main.rs` | CLI entry point |

## Building and Testing

```bash
# Build with CLI features
cargo build --features cli

# Run all tests
cargo test --features cli

# Run specific test
cargo test --features cli test_const_propagation
```

## Quantum Gate API

The API uses DSL-style syntax (gate followed by target) with batch operations using set literals:
```zlup_nocheck
// Single-qubit gates (space-separated syntax)
h q[0];
x q[1];
rz(pi/4) q[0];

// Batch operations with set literals
h {q[0], q[1], q[2]};
cx {(q[0], q[1]), (q[2], q[3])};

// Typed measurements
r: u1 = mz(u1) q[0];                              // Single qubit
results: [2]u1 = mz([2]u1) [q[0], q[1]];          // Multiple qubits -> [2]u1
syndrome: u8 = mz(pack u8) [q[0], q[1], ...];     // Pack into integer
```

## Example Program

```zlup
pub fn main() -> unit {
    q := qalloc(4);
    pz q;

    // Apply Hadamard to all qubits
    h {q[0], q[1], q[2], q[3]};

    // Create entanglement
    cx {(q[0], q[1]), (q[2], q[3])};

    // Measure all qubits
    results: [4]u1 = mz([4]u1) [q[0], q[1], q[2], q[3]];

    return;
}
```

## Important Notes for Future Sessions

### Variable Naming
Single-letter names like `s`, `h`, `x`, `y`, `z`, `t` are gate names. When writing tests or examples with slice/array parameters, use names like `data`, `arr`, `items` to avoid parsing conflicts.

### Type System Key Points
- `[]T` and `[N]T` are distinct types
- Use `arr[..]` to convert array to slice
- Escape analysis prevents returning slices of local variables
- Parameters are safe to slice and return (caller owns data)

### Safety Model
The language is "safe by constraint" - no recursion, no dangling references, no escaping locals. These checks are always enabled, not just in strict mode.

---
*Last updated: February 2026*
