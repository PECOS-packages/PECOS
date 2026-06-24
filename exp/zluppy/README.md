# Zluppy

**EXPERIMENTAL** - Python bindings for the Zlup quantum programming language.

Zlup is an experimental language exploring alternative syntax for quantum programs.
It complements Guppy (the primary quantum programming language in PECOS) and may
serve as a compilation target or alternative syntax for certain workflows.

## Installation

```bash
pip install zluppy
```

## Usage

```python
import zluppy

# Compile Zluppy source to SLR-AST (returns dict)
ast = zluppy.compile_to_slr("""
    fn main() -> void {
        var q = qalloc(2);
        h(q[0]);
        cx(q[0], q[1]);
    }
""")

# Compile to SLR-AST JSON string
json_str = zluppy.compile_to_slr_json(source)

# Check source for errors
zluppy.check(source)  # Raises ZluppyError on failure
zluppy.check(source, strict=True)  # NASA Power of 10 mode

# Build programs programmatically
# Note: SlrProgram uses uppercase gate names (SLR-AST convention)
# while Zlup source code uses lowercase (h, cx, etc.)
prog = zluppy.SlrProgram("main")
prog.add_allocator("q", 2)
prog.add_gate("H", [("q", 0)])
prog.add_gate("CX", [("q", 0), ("q", 1)])
json_str = prog.to_json()
```

## License

Apache-2.0
