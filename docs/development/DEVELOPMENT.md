# Basic development steps

## Requirements

**Full development** (Python + Rust, recommended):

- [Python 3.10+](https://www.python.org/downloads/)
- [Rust](https://www.rust-lang.org/tools/install) (stable toolchain)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) - Python package manager
- [just](https://github.com/casey/just) - Command runner
- [pecos](https://crates.io/crates/pecos) - PECOS dev tools CLI
- [ripgrep](https://github.com/BurntSushi/ripgrep#installation) - Required by the dependency integrity checks run by `just lint` and `just security-check`. Install it with `pecos install ripgrep`, or manually with `cargo install ripgrep --locked`, `brew install ripgrep`, `apt install ripgrep`, or `winget install BurntSushi.ripgrep`.
- **Windows**: [Git for Windows](https://git-scm.com/download/win) (provides Git Bash, required by Justfile recipes) or WSL

**Pure Rust development** (Rust crates only):

If you're only working on Rust crates (e.g., `pecos-core`, `pecos-engines`), you can use `cargo` directly without Python:

```sh
cargo build -p pecos-core
cargo test -p pecos-core
cargo clippy -p pecos-core
cargo clean  # Clean Rust artifacts only
```

## Setup Steps

For developers who want to contribute or modify PECOS:

1. Make sure you have [Python](https://www.python.org/downloads/) and [Rust](https://www.rust-lang.org/tools/install) installed for your system.

2. Install the pre-clone dev tools from crates.io:
   ```sh
   cargo install --locked uv just
   ```

   This installs:
   - `uv` - Python package manager
   - `just` - Command runner for build tasks

3. Clone the repository:
   ```sh
   git clone https://github.com/PECOS-packages/PECOS.git
   cd PECOS
   ```

4. Install the PECOS developer CLI from the repo:
   ```sh
   cargo install --path crates/pecos-cli
   ```

   This installs the `pecos` binary (llvm, cuda, cuquantum, rust, python, deps commands).

5. Create the development environment:
   ```sh
   uv sync
   ```

   `uv sync` installs the default `dev` and `test` groups (lint, build,
   docs, and pytest tooling). Optional groups you may want to add:

   | Group         | When to enable                                              | Command                       |
   |---------------|-------------------------------------------------------------|-------------------------------|
   | `examples`    | Running notebooks under `examples/` or DataFrame benchmarks | `uv sync --group examples`    |
   | `numpy-compat`| Verifying older NumPy/SciPy minimums                        | `uv sync --group numpy-compat`|
   | `cuda13`      | GPU simulators on CUDA 13 (Turing/CC 7.5+; requires CUDA toolkit) | `uv sync --group cuda13`  |
   | `cuda12`      | GPU simulators on CUDA 12 (incl. V100/Volta)               | `uv sync --group cuda12`      |

   Combine groups with multiple `--group` flags
   (e.g. `uv sync --group examples --group cuda13`). Pick one CUDA major --
   `cuda12` and `cuda13` are mutually exclusive.

6. **LLVM 21.1 Setup (Required for LLVM IR/QIS Support)**

   PECOS requires LLVM version 21.1 for LLVM IR execution features.

   **Quick setup:**
   ```sh
   pecos install llvm
   cargo build
   ```

   `pecos install llvm` is the managed shared-LLVM path on supported
   Debian/Ubuntu-compatible Linux systems. For macOS, Windows, and other Linux
   distributions, see the [**LLVM Setup Guide**](../user-guide/llvm-setup.md).

7. You may wish to explicitly activate the environment for development. To do so:

    === "Linux/Mac"
        ```sh
        source .venv/bin/activate
        ```

    === "Windows"
        ```sh
        .\.venv\Scripts\activate
        ```

8. Build the project in editable mode
    ```sh
   just build
   ```
   Other build options: `just build-release` (optimized), `just build-native` (optimized for your CPU).

9. Run all Python and Rust tests:
   ```sh
   just test
   ```
   Note: Make sure you have run a build command before running tests.

10. Run linters using pre-commit (after [installing it](https://pre-commit.com/)) to make sure all everything is properly linted/formated
    ```sh
    just lint
    ```

11. Run dependency and security policy checks when touching dependency manifests, lockfiles, GitHub Actions workflows, or security policy:
    ```sh
    just security-check
    ```

    For Rust-only dependency changes, `just cargo-deny` runs the same `cargo-deny` checks that CI applies to the root workspace and the standalone native benchmark crate.

12. To deactivate your development venv:
    ```sh
    deactivate
    ```

Before pull requests are merged, they must pass linting, tests, and dependency/security checks. The local pre-PR gate is:

```sh
just check-all
```

Note: For the Rust side of the project, you can use `cargo` to run tests, benchmarks, formatting, etc.

## Native Numeric Migration Pattern

Runtime code imports numeric primitives from the public `pecos` surface. NumPy remains in tests as an oracle.

```python
from pecos import Array, array, asarray, dtypes, sum as array_sum, zeros
```

Map the dtype used by the migrated code as follows:

| NumPy dtype | PECOS dtype |
|-------------|-------------|
| `np.uint8`  | `dtypes.uint8` |
| `np.float64` or `float` | `dtypes.float64` |

Use `Array` in annotations and enter the native layer with `asarray()` when an external API returns an array-like
object. Constructors and casts take PECOS dtypes, for example `zeros(size, dtype=dtypes.uint8)` and
`array(values, dtype=dtypes.uint8)`.

`Array.flatten()` preserves row-major order and returns an independent copy:

```python
from pecos import array, dtypes

values = array([[1, 0], [0, 1]], dtype=dtypes.uint8)
flat = values.flatten().tolist()
assert flat == [1, 0, 0, 1]
```

`Array` deliberately differs from NumPy at three boundaries:

- `ravel()` and `reshape()` always return copies. NumPy may return views, but `Array` owns its
  buffer and has no view representation, so an honest copy is preferred over a silently-copying
  "view".
- `reshape()` infers a dimension only from the literal `-1`. Other negative dimensions are
  rejected, even though NumPy treats any single negative dimension as inferred.
- `fill()` does not coerce values by truthiness or parse numeric strings. Convert values
  explicitly instead of relying on behavior such as a non-empty string becoming `True`.

Two more migration boundaries currently need local workarounds:

- Elementwise `Array ^ Array` is not available yet (#458). For arrays already known to contain only
  binary values, elementwise inequality has the same result; do not use that substitution for general integers.
- Bit shifts, unsigned arithmetic, and bitwise boolean operators are not available on `Array` (#458).
  For binary values, cast to `int64` and double instead of shifting; use `where()` for boolean
  selection. `Array * Array` is matrix multiplication, so use `elemwise_mul()` when the intended
  operation is elementwise.
- Spell NumPy's `dtype=float` as `dtype=dtypes.float64`, especially with `asarray()`, to preserve
  NumPy's 64-bit width explicitly.

The decoder-side bit packing and boolean-selection patterns remain explicit and exact:

```python
from pecos import any as array_any, array, dtypes, sum as array_sum, where

low = array([1, 0], dtype=dtypes.uint8).astype(dtypes.int64)
high = array([0, 1], dtype=dtypes.uint8).astype(dtypes.int64)
packed = (low + high * 2).astype(dtypes.uint8)
assert packed.tolist() == [1, 2]

observable_bits = array([[1, 0], [1, 1]], dtype=dtypes.int64)
weights = array([1, 2], dtype=dtypes.int64)
observable_masks = array_sum(observable_bits.elemwise_mul(weights), axis=1)
assert list(observable_masks) == [1, 3]

active = array([True, False], dtype=dtypes.bool_)
values = array([0, 2], dtype=dtypes.uint8)
assert array_any(where(active, False, values != 0))
```

Elementwise comparison returns a boolean `Array`, and `array_sum()` accepts it directly -- counting
mismatches needs no cast:

```python
from pecos import array, dtypes, sum as array_sum

predicted = array([1, 0, 1], dtype=dtypes.uint8)
expected = array([1, 1, 1], dtype=dtypes.uint8)
logical_errors = int(array_sum(predicted != expected))
assert logical_errors == 1
```

Seed `pecos.random` immediately before a reproducible draw. Its stream deliberately differs from
NumPy's for the same seed, so migrate statistical tests by re-baselining pinned samples and retaining
distributional invariants instead of asserting cross-library sample equality:

```python
from pecos import random

random.seed(458)
first = random.binomial(20, 0.25, size=8)
random.seed(458)
repeated = random.binomial(20, 0.25, size=8)
assert first.tolist() == repeated.tolist()
```

## Dependency and Security Checks

Use the Justfile recipes below so local checks match CI:

| Command | When to run | What it checks |
|---------|-------------|----------------|
| `just security-check` | Dependency, lockfile, GitHub Actions, cache, or security-policy changes | Runs the dependency integrity script and both `cargo-deny` checks |
| `just cargo-deny` | Rust dependency or Cargo lockfile changes | Checks advisories, banned dependency patterns, and allowed dependency sources |
| `just cargo-deny-workspace` | Root workspace Rust dependency changes | Runs `cargo-deny` on the root Rust workspace |
| `just cargo-deny-native-bench` | Native benchmark crate dependency changes | Runs `cargo-deny` on `scripts/native_bench/bench_pecos/Cargo.toml` |
| `just dependency-integrity-check` | CI workflow, lockfile policy, action pinning, or cache posture changes | Checks lock discipline, action pinning, cache write posture, dependency review coverage, and package-worm indicators |
| `just check-all` | Before opening or updating a PR with broad changes | Runs clean, release build, release tests, lint, and dependency/security checks |

`cargo-deny` is not installed by `uv sync`. To run the Rust dependency policy checks locally, install the same version used by CI:

```sh
cargo install --locked --version 0.19.6 cargo-deny
```

The first `cargo-deny` run may update the local advisory database under `~/.cargo`. CI runs these checks on every relevant Cargo manifest, lockfile, `deny.toml`, or cargo-deny workflow change, and also on the scheduled security lane.

## Cleaning Build Artifacts

Clean commands are cross-platform (Windows, macOS, Linux):

```sh
just clean              # Clean project build artifacts (includes selene)
just clean cache        # Clean ~/.pecos/cache/ and ~/.pecos/tmp/
just clean deps         # Clean ~/.pecos/deps/ (LLVM, CUDA, cuQuantum)
just clean all          # Everything above
just clean dry-run      # Preview what would be cleaned
```

You can also run the cleaning script directly:

```sh
uv run python scripts/clean.py --help
uv run python scripts/clean.py --dry-run  # Preview what would be deleted
```

For day-to-day Rust development, `cargo clean` handles the `target/` directory. The `~/.pecos/` directory (LLVM, CUDA, C++ dependencies) rarely needs cleaning - it contains installed dependencies rather than build artifacts.

## PECOS Home Directory

PECOS uses `~/.pecos/` to store external dependencies and build artifacts that cannot be managed through Cargo.toml:

```
~/.pecos/
├── deps/llvm-21.1/  # LLVM 21.1 installation (for QIR/LLVM IR execution)
├── deps/       # Downloaded C++ dependencies (Stim, etc.)
└── cache/      # Build artifacts and intermediate files
```

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `PECOS_HOME` | Override entire home directory | `~/.pecos/` |
| `PECOS_DEPS_DIR` | Override deps location | `$PECOS_HOME/deps/` |
| `PECOS_CACHE_DIR` | Override cache location | `$PECOS_HOME/cache/` |

These can be set via shell environment or in `.cargo/config.toml`:

```toml
[env]
PECOS_HOME = { value = "/custom/path", force = true }
```

## Development Guides

For specific development topics, see:

- [Parallel Blocks and Optimization](parallel-blocks-and-optimization.md) - Guide to using and extending the Parallel block construct and optimizer
- [Documentation Code Testing](doc-testing.md) - Guide to testing code examples in documentation
