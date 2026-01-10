# Gate Naming: Future Considerations

This document captures naming conventions being considered for future adoption. These are **not yet implemented** but are recorded here for discussion and planning.

For current naming conventions, see [Gate Naming Conventions](gate-naming-conventions.md).

## Nth Root Gates

A systematic way to denote arbitrary roots of Pauli gates:

| Potential Name | Meaning | Current Name |
|----------------|---------|--------------|
| S2Z or SZ | 2nd root (square root) of Z | SZ |
| S4Z | 4th root of Z | T |
| S8Z | 8th root of Z | (none) |
| S2X | 2nd root of X | SX |

**Considerations**:
- More systematic and extensible
- Conflicts with existing `S` prefix meaning "square root"
- Would require `S` to mean "root" with a numeric qualifier

### Fraction-Based Naming

An alternative approach using fractions to denote the rotation as a fraction of the full rotation:

| Potential Name | Meaning | Rotation | Current Name |
|----------------|---------|----------|--------------|
| S1o2Z | Z^(1/2) | RZ(π/2) | SZ |
| S1o4Z | Z^(1/4) | RZ(π/4) | T |
| S1o8Z | Z^(1/8) | RZ(π/8) | (none) |
| S3o4Z | Z^(3/4) | RZ(3π/4) | (none) |
| S1o2X | X^(1/2) | RX(π/2) | SX |

Where `o` represents "over" (division), making `1o2` = 1/2.

**Alternative separators:**

| Style | Example | Notes |
|-------|---------|-------|
| S1o2Z | "one over two" | Compact |
| S1_2Z | underscore | Clear separation |
| S1d2Z | "divided by" | Alternative to `o` |

**Considerations**:
- Keeps `S` prefix consistent with current square root naming
- More intuitive: "S one over two Z" = square root of Z
- Supports arbitrary fractions, not just roots (e.g., S3o4Z for 3/4 rotation)
- `SZ` could be shorthand for `S1o2Z` (the common case)

## Power/Exponent Notation (Documentation Only)

Mathematical notation for roots (not valid as code identifiers):

| Math Notation | Meaning | Valid Code Name |
|---------------|---------|-----------------|
| Z^(1/2) | Square root of Z | SZ or S2Z |
| Z^(1/4) | Fourth root of Z | T or S4Z |
| X^(1/2) | Square root of X | SX or S2X |

**Note:** Power notation is useful in documentation and papers but cannot be used as function/variable names.

## Controlled Gate Extensions

Systematic naming for multi-controlled gates:

| Potential Name | Meaning | Notes |
|----------------|---------|-------|
| CCX | Doubly-controlled X | Already used (Toffoli) |
| CCCX | Triply-controlled X | Extends pattern |
| C2X | 2-controlled X | Alternative to CCX |
| C3X | 3-controlled X | More scalable |
| MCX | Multi-controlled X | Generic form |

## Parameterized Controlled Gates

Naming for controlled rotation gates:

| Potential Name | Meaning |
|----------------|---------|
| CRX(θ) | Controlled RX rotation |
| CRY(θ) | Controlled RY rotation |
| CRZ(θ) | Controlled RZ rotation |

**Note**: CRZ is already implemented in PECOS.

## Single-Qubit Plane Rotations vs Two-Qubit Interactions

There is an ambiguity with rotation gates that reference two axes:

| Name | Could Mean | Context |
|------|------------|---------|
| RXY | Single-qubit rotation in XY plane | Bloch sphere interpretation |
| RXY | Two-qubit exp(-i θ XY/2) | Following RXX/RYY/RZZ pattern |

Currently, PECOS uses `R1XY` for the single-qubit XY-plane rotation, where the `1` indicates single-qubit. However, this is not immediately obvious.

**Potential naming schemes:**

| Single-Qubit (XY plane) | Two-Qubit (XY interaction) | Notes |
|------------------------|---------------------------|-------|
| R1XY | RXY | Current PECOS convention |
| R1qXY | R2qXY | Explicit qubit count |
| RPXY | RXY | "P" for plane |
| RPLXY | RXY | "PL" for plane |

**Considerations:**
- The `RXX/RYY/RZZ` pattern strongly suggests two-qubit interactions
- Single-qubit plane rotations are less common, so longer names may be acceptable
- `R1XY` works but the `1` prefix is unique to this gate
- Names must be valid identifiers (no special characters, case-insensitive)

**Current behavior:**
```
R1XY(θ, φ) = RZ(-φ + π/2) · RY(θ) · RZ(φ - π/2)
```
This rotates by angle θ around an axis in the XY plane specified by φ.

## Root of Two-Qubit Gates

Extending the root notation to two-qubit interactions:

| Current | Meaning | Potential Systematic |
|---------|---------|---------------------|
| SZZ | Square root of ZZ | S2ZZ |
| (none) | Fourth root of ZZ | S4ZZ or TZZ |
| RZZ(π/2) | Same as SZZ | - |

## Global Phase Conventions

Currently, many gate equivalences hold only "up to global phase." A convention for phase-exact definitions:

| Notation | Meaning |
|----------|---------|
| =ₚ | Equal up to global phase |
| = | Exactly equal (including phase) |

Example: `T² =ₚ SZ` (equal up to global phase)

## Systematic Clifford Naming

The current Hadamard (H, H2-H6) and Face (F, F2-F4) naming is historical but not self-documenting. You cannot tell from `H5` or `F3` what the gate actually does without looking it up.

### The Problem

A single-qubit Clifford is fully determined by where it sends X and Z (since Y = iXZ). The current numbered variants don't encode this information:

| Current Name | X → | Z → | Why "H5"? |
|--------------|-----|-----|-----------|
| H | Z | X | Historical |
| H5 | -X | Y | Arbitrary numbering |
| F | Y | X | Historical |
| F3 | Y | -X | Arbitrary numbering |

### Potential: Transformation-Based Names (K Notation)

Encode the Pauli destinations directly in the name using `K` prefix (K for Clifford, to avoid confusion with C for controlled):

| Format | Example | Meaning |
|--------|---------|---------|
| K\[X_dest\]\[Z_dest\] | KZX | X→Z, Z→X (Hadamard) |
| K\[XI_dest\]\_\[ZI_dest\]\_\[IX_dest\]\_\[IZ_dest\] | KXX_ZI_IX_ZZ | CX gate |

The qubit count is implied by the structure (single Paulis vs two-Pauli terms with I).

Where destinations use: X, Y, Z, NX (=-X), NY (=-Y), NZ (=-Z), and for multi-qubit: XI, IX, ZZ, etc.

### Single-Qubit Examples

| Current | Proposed | Transformation |
|---------|----------|----------------|
| I | KXZ | X→X, Z→Z |
| X | KXNZ | X→X, Z→-Z |
| Y | KNXNZ | X→-X, Z→-Z |
| Z | KNXZ | X→-X, Z→Z |
| H | KZX | X→Z, Z→X |
| H2 | KNZNX | X→-Z, Z→-X |
| SZ | KYZ | X→Y, Z→Z |
| SX | KXY | X→X, Z→Y |
| F | KYX | X→Y, Z→X |

**Considerations:**
- Self-documenting: can read the transformation from the name
- Systematic: covers all 24 single-qubit Cliffords
- Compact: `KYX` is only one character longer than `F`
- Uses K (not C) to avoid confusion with "controlled" prefix
- Learning curve: users familiar with H, S, T would need to adapt
- Could coexist as aliases

### Extension to Multi-Qubit Cliffords

For n-qubit Cliffords, we need to track where the 2n generators map:
- 1-qubit: X, Z (2 generators)
- 2-qubit: XI, ZI, IX, IZ (4 generators)
- 3-qubit: XII, ZII, IXI, IZI, IIX, IIZ (6 generators)

A two-qubit Clifford is fully determined by the 4 generator mappings:

| Current | XI → | ZI → | IX → | IZ → |
|---------|------|------|------|------|
| CX | XX | ZI | IX | ZZ |
| CZ | XZ | ZI | ZX | IZ |
| SWAP | IX | IZ | XI | ZI |
| iSWAP | -ZY | IZ | YZ | ZI |
| SZZ | YZ | ZI | ZY | IZ |

### Two-Qubit Naming Scheme

Using underscore-separated destinations in fixed order (XI, ZI, IX, IZ):

```
KXX_ZI_IX_ZZ  → CX  (XX for XI→, ZI for ZI→, IX for IX→, ZZ for IZ→)
KXZ_ZI_ZX_IZ  → CZ
KIX_IZ_XI_ZI  → SWAP
```

The qubit count is implied: single Paulis = 1-qubit, two-Pauli terms = 2-qubit, etc.

### The Verbosity Problem

| Qubits | Generators | Full encoding length |
|--------|------------|---------------------|
| 1 | 2 | 4 chars (KZX) |
| 2 | 4 | 15 chars (KXX_ZI_IX_ZZ) |
| 3 | 6 | ~27 chars |

For common two-qubit gates, traditional names (CX, CZ, SWAP) are likely preferable, with systematic names as documentation or for unusual Cliffords.

### Practical Compromise

- Use systematic names (K notation) for single-qubit Cliffords where H2-H6, F2-F4 are unclear
- Keep traditional names for common two-qubit gates
- Use systematic notation in documentation/comments to clarify behavior
- Reserve full systematic names for programmatically-generated or unusual Cliffords

### Alternative: Grouped Systematic Names

Keep the H/F grouping but use systematic suffixes based on transformation properties:

- **H-type** (Hadamard-like): Exchange two axes
  - HZX, HYX, HZY for the three planes (X→Z, Z→X means HZX)
  - HNZX, HNZNX for sign variants (X→-Z, Z→X vs X→-Z, Z→-X)

- **F-type** (Face-like): Cyclic permutation
  - FYX (X→Y, Z→X)
  - FZY (X→Z, Z→Y)
  - With sign variants (FNYX, FNYNX, etc.)

### Clifford Group Structure

The 24 single-qubit Cliffords can be organized as:
- 1 identity
- 3 Pauli gates (X, Y, Z)
- 6 Hadamard-type (exchange two axes, 3 planes × 2 signs)
- 6 Face-type (cyclic permutation, 2 directions × 3 starting points... approximately)
- 8 composed operations (S-gates and variants)

A systematic naming would make this structure explicit.

## Design Principles

When extending PECOS gate names, these principles guide decisions:

1. **Consistency**: New names should follow existing patterns where possible
2. **Discoverability**: Names should be guessable from the pattern
3. **Compatibility**: Maintain QASM and literature compatibility for common gates
4. **Brevity**: Prefer short names for frequently-used gates
5. **Precision**: Avoid ambiguity in meaning

## See Also

- [Gate Naming Conventions](gate-naming-conventions.md) - Current naming conventions
- [Gate Reference](gates.md) - Complete gate documentation with matrices
