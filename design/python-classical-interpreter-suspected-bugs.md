# Python PhirClassicalInterpreter -- Suspected Bugs

Found during the Rust reimplementation and fuzz testing.

Bugs #1 and #2 are the same class of issue as
[PECOS-packages/PECOS#213](https://github.com/PECOS-packages/PECOS/issues/213):
PECOS dtype constructors reject values outside the type range instead of
masking/wrapping. PR #214 fixed the specific operator overload case but the
underlying dtype overflow issue remains.

## 1. Overflow rejected for values that fit the register but not the dtype

**Confidence:** High
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

## 2. Bitwise NOT overflows when assigning cross-type

**Confidence:** High
**Found via:** Edge case testing related to issue #213

**Description:** `~m` where `m` is `u32 size=1` (measurement register) produces `u32(4294967295)`. Assigning to an `i32` variable does `i32(4294967295)` which throws `OverflowError` because 4294967295 > `i32::MAX`. The masking to register size happens AFTER the dtype conversion, so it never gets a chance to mask.

**Rust behavior:** Evaluates `~0 = 0xFFFFFFFFFFFFFFFF`, constrains to i32 width (`-1`), stores fine.

---

## 3. `PhirModel.model_validate` rejects valid PHIR programs with `Result` cop

**Confidence:** High (but in the `phir` pydantic model, not the interpreter itself)
**File:** `python/quantum-pecos/src/pecos/classical_interpreters/phir_classical_interpreter.py`
**Line:** 101-102

**Description:** When `phir_validate=True` (default), the interpreter validates programs through `PhirModel.model_validate()` from the `phir` pydantic package. This validator rejects the `Result` classical operation, which is a valid PECOS-specific extension used in many test programs.

**Example:** Programs with `{"cop": "Result", "args": ["m"], "returns": ["c"]}` fail pydantic validation even though they execute correctly.

**Impact:** Users must set `phir_validate=False` to run programs with `Result` operations when using the Python interpreter. The Rust interpreter's serde parser handles these correctly.

---

## Design questions (not clear-cut bugs)

### Signed types not masked to register size

**File:** `python/quantum-pecos/src/pecos/classical_interpreters/phir_classical_interpreter.py`
**Line:** 345-349

`assign_int()` only masks unsigned types to the register's declared `size`. Signed types are stored at the full dtype width. The code has a comment: `# (only valid for unsigned data types)` -- suggesting this was a deliberate choice.

The PHIR spec says "assigning 5 to a 2-bit variable stores only the lower 2 bits" with no unsigned-only qualifier, but there may be good reasons for treating signed types differently (sign-extension from narrow widths is lossy).

### Shift by type width is a no-op

`u32(1) << 32` gives `u32(1)` instead of `u32(0)`. The PECOS dtype uses native hardware shift semantics (shift amount modulo type width). This matches x86/ARM behavior but not mathematical semantics. Whether the PHIR spec expects hardware or mathematical shift behavior is unclear.
