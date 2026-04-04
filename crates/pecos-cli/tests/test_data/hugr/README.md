# HUGR Test Files

This directory contains HUGR test files generated from guppy quantum circuits.

## Files

- `bell_state.hugr` - Bell State
- `single_hadamard.hugr` - Single Hadamard
- `ghz_state.hugr` - Ghz State

## File Format

The `.hugr` files use HUGR's current "binary" format, which is actually a 10-byte header followed by JSON data. This makes them git-friendly despite the binary extension. If HUGR moves to a true binary format in the future, we may need to reconsider storing these files in git.

## Regenerating Files

To regenerate these files, run:
```bash
uv run python scripts/generate_hugr_test_files.py
```

Note: This requires guppylang to be installed.
