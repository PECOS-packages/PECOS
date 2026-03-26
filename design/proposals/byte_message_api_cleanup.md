# ByteMessageBuilder API Cleanup

## Changes Made

### Rust API

Two-qubit gates now take `&[(usize, usize)]` (slice of pairs) instead of separate slices:

```rust
// Before:
builder.add_cx(&[0], &[1]);
builder.add_rzz(theta, &[0], &[1]);

// After:
builder.add_cx(&[(0, 1)]);
builder.add_rzz(theta, &[(0, 1)]);

// Batch:
builder.add_cx(&[(0, 1), (2, 3)]);
```

Affected methods: `add_cx`, `add_cy`, `add_cz`, `add_szz`, `add_szzdg`, `add_rzz`

### Rename: `add_measurements` -> `add_mz`

The method name now matches the gate name (MZ).

### Python API

Single-qubit gates take lists: `add_h([0, 1, 2])`
Two-qubit gates take lists of tuples: `add_cx([(0, 1), (2, 3)])`
Measurements renamed: `add_mz([0, 1])`

## Status

Done. Both Rust and Python APIs updated.
