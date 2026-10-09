# Documentation Code Testing

<!--skip: Examples show marker syntax, not runnable code-->

PECOS automatically tests code examples in documentation to ensure they remain correct as the codebase evolves.

## Overview

The documentation testing system:

1. Extracts Python and Rust code blocks from Markdown files
2. Generates pytest test files from the extracted code
3. Runs the tests as part of CI

## Running Documentation Tests

```bash
# Generate tests from documentation
uv run python scripts/docs/generate_doc_tests.py

# Run all documentation tests
uv run pytest python/quantum-pecos/tests/docs/generated -v

# Run tests for a specific document
uv run pytest python/quantum-pecos/tests/docs/generated/user-guide/test_getting_started.py -v
```

## Marker Reference

Markers are HTML comments placed immediately before code blocks to control test behavior.

### Skip Markers

#### Block-level Skip

Skip a single code block:

```markdown
<!--skip-->
\```python
# This code won't be tested
from some_unavailable_module import thing
\```
```

Skip with a reason (shown in test output):

```markdown
<!--skip: requires external hardware-->
\```python
connect_to_quantum_computer()
\```
```

#### Document-level Skip

Skip all code blocks in a document by placing a skip marker at the beginning of the file (after the heading):

```markdown
# Document Title

<!--skip: All examples require external files-->

Content here...
```

### Conditional Skip

Skip tests when CUDA is not available:

```markdown
<!--skip-if-no-cuda-->
\```python
from pecos.simulators import CudaSim
sim = CudaSim()
\```
```

### Expected Errors

Test that code raises a specific error:

```markdown
<!--expect-error: TypeError-->
\```python
"string" + 123  # This will raise TypeError
\```
```

The test passes if the code raises an error containing the specified text.

### Expected Output

Verify code produces specific output:

```markdown
<!--expect-output: Hello, World!-->
\```python
print("Hello, World!")
\```
```

### Test Names

Give a test a specific name for easier identification:

```markdown
<!--test-name: bell_state_example-->
\```python
# This test will be named test_bell_state_example
\```
```

### Pytest Marks

Add pytest marks to tests:

```markdown
<!--mark.slow-->
\```python
# This test will have @pytest.mark.slow
\```
```

### Test Data Files

For unskipped Rust examples that use external crates and run in the unified cargo
crate, list comma-separated bare filenames from `docs/assets/test-data/`:

```markdown
<!--test-data: repetition_code.hugr-->
\```rust
use pecos::prelude::*;

fn main() {
    let data = std::fs::read("repetition_code.hugr").unwrap();
}
\```
```

The generator copies these files directly into
`python/quantum-pecos/tests/docs/rust_crate/`, the package root that Cargo uses as
the working directory for integration tests. Code opens them by bare relative name.
Continuation tests inherit every data filename named earlier in their chain.

Without a marker, no data is staged. A named source file must exist; otherwise
generation fails and reports the Markdown file, block number, and missing filename.
Markers on non-Rust, skipped, non-cargo, hidden, setup, or teardown blocks are
authoring errors reported with the marker's file and line number. Every literal
test-data marker must be attached directly to an eligible block; an intervening
comment detaches it. A document with a document-level skip is exempt from this
check, since none of its blocks run. Only the immediately preceding comment supplies block markers.
Use one comment when combining continuation with test-data (for example, put
`continuation` before `test-data:` in that comment).

Use bare filenames, not paths. `Cargo.toml`, `Cargo.lock`, `build.rs`,
`rust-toolchain`, `rust-toolchain.toml`, and all names starting with `.` are
reserved. Unrelated existing crate files and symlink destinations cannot be overwritten.

Every generation refreshes the copies and removes stale copies listed in
`.test-data-files.json`; unrelated files are left alone. When no files are requested,
the manifest is removed too. Commit both the copies
and this generated ownership manifest with the crate so the existing CI drift
check covers them.

## Hidden Preambles

Use hidden code blocks to provide imports or setup code that should be included in tests but not shown in documentation:

```markdown
\```hidden-python
from pecos.slr import Main, QReg, CReg
from pecos.slr.qeclib import qubit as qb
\```

Now the visible code can use these imports:

\```python
prog = Main(
    q := QReg("q", 2),
    qb.H(q[0]),
)
\```
```

The hidden preamble is prepended to all subsequent code blocks in the document until a `<!--preamble-reset-->` marker is encountered.

### Preamble Reset

Reset the accumulated preamble:

```markdown
<!--preamble-reset-->
\```python
# This code starts fresh without any preamble
\```
```

## Code Block Languages

### Python

Python code blocks are tested using `exec()`:

```markdown
\```python
from pecos import sim
result = sim(Qasm("...")).run(10)
\```
```

### Rust

Rust code blocks can be tested in two ways:

1. **Simple Rust** (rustc): Code with `fn main()` that doesn't use external crates
2. **Cargo Rust**: Code using `pecos` crates (detected by `use pecos*::`)

Incomplete Rust snippets must be completed or explicitly marked skipped (with a
`<!--skip: reason-->` comment or a `,skip`, `,ignore`, `,no_run`, or `,notest` fence
suffix). Otherwise generation fails with the Markdown file and block number, for
both the unified crate and pytest paths.

#### Rust with Cargo Dependencies

For Rust code that uses PECOS crates:

```markdown
\```rust
use pecos::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let results = sim(Qasm::from_string("...")).run(10)?;
    Ok(())
}
\```
```

This creates a temporary Cargo project with the necessary dependencies.

## Fence-level Skip Markers

Standard Rust documentation markers are also supported:

```markdown
\```rust,skip
// This code is skipped
\```

\```rust,ignore
// This code is also skipped
\```

\```rust,no_run
// This code is also skipped
\```
```

## Tips

### Do

- Add hidden preambles for common imports to keep visible code focused
- Use meaningful skip reasons to explain why code can't be tested
- Test error conditions with `<!--expect-error-->` markers
- Keep code examples self-contained when possible

### Don't

- Don't skip code unless necessary
- Don't rely on state from previous code blocks (each block is tested independently)
- Don't include real API keys or credentials in examples
- Don't use `exec()` in your actual code examples (it's only used internally for testing)

## Generated Files

The test generator creates files in:

```
python/quantum-pecos/tests/docs/generated/
  conftest.py          # Shared fixtures
  test_README.py       # Tests from docs/README.md
  user-guide/          # Tests from docs/user-guide/
  development/         # Tests from docs/development/
```

These files are gitignored and regenerated before tests run.

## Debugging Failed Tests

When a documentation test fails:

1. Check the test docstring for the source file and line number
2. The test name includes the source file and block number
3. Run the specific test with `-v` for verbose output
4. Check if the code requires imports that should be in a hidden preamble

Example:

```bash
# Run with verbose output
uv run pytest python/quantum-pecos/tests/docs/generated/user-guide/test_gates.py::test_gates_block_5 -v

# The test docstring shows the source:
# """Test from docs/user-guide/gates.md:142."""
```

## Adding New Documentation

When adding new documentation:

1. Add code examples in fenced code blocks
2. Add hidden preambles for common imports if needed
3. Add skip markers for code that can't be tested
4. Run `uv run python scripts/docs/generate_doc_tests.py` to generate tests
5. Run the generated tests to verify examples work
6. Commit the documentation (generated tests are gitignored)
