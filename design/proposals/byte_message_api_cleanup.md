# ByteMessageBuilder API Cleanup

## Changes Made

### Rust API

Two-qubit gates now take `&[(usize, usize)]` (slice of pairs) instead of separate slices:

```rust
// Before:
builder.cx(&[0], &[1]);
builder.rzz(theta, &[0], &[1]);

// After:
builder.cx(&[(0, 1)]);
builder.rzz(theta, &[(0, 1)]);

// Batch:
builder.cx(&[(0, 1), (2, 3)]);
```

Affected methods: `cx`, `cy`, `cz`, `szz`, `szzdg`, `rzz`

### Rename: `mz` -> `mz`

The method name now matches the gate name (MZ).

### Python API

Single-qubit gates take lists: `h([0, 1, 2])`
Two-qubit gates take lists of tuples: `cx([(0, 1), (2, 3)])`
Measurements renamed: `mz([0, 1])`

## Status

Done. Both Rust and Python APIs updated.
