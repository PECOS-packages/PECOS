# Rust Classical Interpreter -- Known Issues and Suspected Bugs

## Expression evaluation type context (Rust and Python differ)

**Status:** Both Rust and Python have this issue, but they disagree on the behavior.

**Description:** The `ExpressionEvaluator` evaluates all arithmetic at `i64`/`u64` width, regardless of the operand's declared type. The Python side evaluates through PECOS dtype arithmetic which wraps at the operand's type width.

**Example:**
```
v2: u32 size=2, value=0
v1: u64 size=46

v1 = v2 - 12
```

- Python: `u32(0) - 12` wraps at 32 bits -> `u32(4294967284)` = `0xFFFFFFF4`. Assigned to u64 size=46 -> `0x00000000FFFFFFF4` (46-bit view has leading zeros).
- Rust: evaluates `0u64 - 12` at 64 bits -> `0xFFFFFFFFFFFFFFF4`. Masked to 46 bits -> `0x3FFFFFFFFFFF4` (46-bit view has leading ones).

**Root cause:** Rust `ExpressionEvaluator` promotes all values to i64/u64 for computation, losing the source operand's type width. Python PECOS dtypes preserve type width through arithmetic operations.

**Impact:** Affects underflow/overflow in mixed-type expressions where the source operand is narrower than i64/u64. Rare in real programs (most PHIR programs use same-type expressions).

**Fuzz result:** 1 out of 971 random programs triggered this. 99.9% identical.

**Fix:** The `ExpressionEvaluator` would need to track the operand's `DataType` through evaluation and apply type-width wrapping after each operation. This is a deeper change to expression.rs.

---

## Signed types not masked to register size (Python behavior, arguably spec-incorrect)

**Status:** Deliberate mismatch with PHIR spec to match Python behavior.

**Description:** The PHIR spec says "assigning 5 to a 2-bit variable stores only the lower 2 bits" -- this should apply to both signed and unsigned types. However, the Python `PhirClassicalInterpreter.assign_int()` only masks unsigned types:

```python
if type(cval) not in signed_data_types.values():
    size = self.cvar_meta[cid].size
    cval &= (1 << size) - 1
```

For signed types, no masking occurs. The value is stored at the full type width (e.g., i64 stores all 64 bits regardless of `size`).

**Our approach:** We match Python's behavior. For signed types, `BitValue` uses the type width (64 for i64, 32 for i32) as the storage width. For unsigned types, it uses the register's declared `size`.

**Impact:** Signed variables with `size` < type width store more bits than the spec suggests. This could matter for programs that rely on narrow signed registers wrapping at `size` bits.

**Fix:** Change both Python and Rust to mask signed types to `size` bits. This would be a behavioral change that needs careful testing across the full test suite.

---

## `program.ops` returns None on Rust interpreter

**Status:** Cosmetic difference, not a behavioral issue.

**Description:** `RustPhirClassicalInterpreter().program.ops` returns `None` instead of the actual operation list. The Python side returns a `PyPHIR.ops` list. This matters if user code inspects `cinterp.program.ops` directly (not just passes it to `execute()`).

**Impact:** Low. `HybridEngine` passes `cinterp.program.ops` to `cinterp.execute()`, but our `execute()` ignores the argument and uses its own internal ops. No existing test accesses `program.ops` for inspection.

---

## `program` type differs

**Status:** Cosmetic.

**Description:** `type(cinterp.program)` is `PyPHIR` for Python, `_PhirProgramWrapper` for Rust. Code that does `isinstance(cinterp.program, PyPHIR)` would fail with the Rust interpreter.

**Impact:** Low. No existing code checks the type of `program`.

---

## `run_phir_sim` full-Rust path uses different seeding

**Status:** By design, documented.

**Description:** The `run_phir_sim()` function uses the existing `PhirJsonEngine` + `MonteCarloEngine` pipeline which has its own seed derivation mechanism. This produces different random sequences than the Python `HybridEngine` loop for the same seed value.

**Impact:** The `_can_use_full_rust` auto-detection is currently disabled to avoid surprising users. `run_phir_sim()` is available as an explicit opt-in for users who want the 3.6x speedup and don't need seed-compatible results with the Python path.

---

## `run_phir_sim` result format differences

**Status:** Known, inherent to existing PhirJsonEngine.

**Description:** The existing `PhirJsonEngine` used by `run_phir_sim` has some result format differences from the Python `PhirClassicalInterpreter`:
- Bit ordering may differ for some operations
- The existing engine's expression evaluation has its own implementation separate from our new `PhirClassicalInterpreter`

**Impact:** Users of `run_phir_sim()` get correct quantum simulation results but the exact bit patterns in classical registers may differ from the Python path for complex programs with heavy classical computation.
