# Python PhirClassicalInterpreter -- Suspected Bugs

Found during the Rust reimplementation and fuzz testing.

## 1. Signed types not masked to register size

**File:** `python/quantum-pecos/src/pecos/classical_interpreters/phir_classical_interpreter.py`
**Line:** 345-349

**Description:** `assign_int()` only masks unsigned types to the register's declared `size`. Signed types are stored at the full dtype width, ignoring `size`.

```python
if type(cval) not in signed_data_types.values():
    # mask off bits given the size of the register
    # (only valid for unsigned data types)
    size = self.cvar_meta[cid].size
    cval &= (1 << size) - 1
```

**PHIR spec says:** "assigning 5 to a 2-bit variable stores only the lower 2 bits" -- no unsigned-only qualifier.

**Example:** An `i64` register with `size: 14` can store value 712872 (which needs 20 bits). The register should only hold the lower 14 bits (712872 & 0x3FFF = 8360).

**Impact:** Signed registers with `size` < type width hold more information than declared. Classical expressions operating on these may produce unexpected results.

---

## 2. Expression evaluation respects operand dtype width but not register size

**File:** `python/quantum-pecos/src/pecos/classical_interpreters/phir_classical_interpreter.py`
**Line:** 265-317 (`eval_expr`)

**Description:** Expression evaluation operates through PECOS dtype arithmetic. Intermediate results wrap at the dtype's width (32 bits for u32, 64 bits for i64), but do NOT consider the register's declared `size`.

**Example:**
```
v: u32 size=3, value=7 (all bits set in 3-bit register)
result = v + 1
```
Python evaluates as `u32(7) + 1 = u32(8)`. The value 8 doesn't fit in 3 bits, but the addition happens at u32 width (32 bits). The masking to 3 bits only happens later in `assign_int`.

This means intermediate expression values can exceed the register's `size`. Whether this is a bug depends on spec interpretation -- the spec defines register size for storage, not necessarily for arithmetic.

---

## 3. Overflow rejected for values that fit the register but not the dtype

**File:** `python/quantum-pecos/src/pecos/classical_interpreters/phir_classical_interpreter.py`  
**Line:** 336 (`val = dtype(val)`)

**Description:** `assign_int` converts the value through the PECOS dtype constructor (`dtype(val)`) before masking to register size. If the value exceeds the dtype's range but would fit in the register's `size`, Python throws `OverflowError`.

**Example:**
```
c: u32 size=31
c = 8589934591  (= 2^33 - 1, fits in 31 bits after masking, but > u32::MAX)
```
Python: `u32(8589934591)` -> `OverflowError: out of range integral type conversion attempted`

The PHIR spec says this should work: the value should be masked to `size=31` bits, giving `0x7FFFFFFF`.

**Impact:** Programs that assign large literal values to narrow registers fail unnecessarily.

---

## 4. `PhirModel.model_validate` rejects valid PHIR programs with `Result` cop

**File:** `python/quantum-pecos/src/pecos/classical_interpreters/phir_classical_interpreter.py`  
**Line:** 101-102

**Description:** When `phir_validate=True` (default), the interpreter validates programs through `PhirModel.model_validate()` from the `phir` pydantic package. This validator rejects the `Result` classical operation, which is a valid PECOS-specific extension used in many test programs.

**Example:** Programs with `{"cop": "Result", "args": ["m"], "returns": ["c"]}` fail pydantic validation even though they execute correctly.

**Impact:** Users must set `phir_validate=False` to run programs with `Result` operations when using the Python interpreter. The Rust interpreter's serde parser handles these correctly.

---

## 5. `_internal_cinterp` hardcoded to Python

**File:** `python/quantum-pecos/src/pecos/engines/hybrid_engine.py`  
**Line:** 94 (before our changes)

**Description:** (Now fixed in our branch.) `HybridEngine.__init__` previously hardcoded `_internal_cinterp = PhirClassicalInterpreter()` regardless of what `cinterp` was. This meant the inner interpreter was always Python even when the user chose a different interpreter.

**Status:** Fixed in this branch -- the inner interpreter now matches the outer.
