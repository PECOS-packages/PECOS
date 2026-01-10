# Basic development steps

## Requirements

**Full development** (Python + Rust, recommended):

- [Python 3.10+](https://www.python.org/downloads/)
- [Rust](https://www.rust-lang.org/tools/install) (stable toolchain)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) - Python package manager
- [just](https://github.com/casey/just) - Command runner
- [pecos](https://crates.io/crates/pecos) - PECOS dev tools CLI

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

2. Install all dev tools with a single command:
   ```sh
   cargo install --locked uv just pecos
   ```

   This installs:
   - `uv` - Python package manager
   - `just` - Command runner for build tasks
   - `pecos` - PECOS dev tools (llvm, cuda, rust, python commands)

3. Clone the repository:
   ```sh
   git clone https://github.com/PECOS-packages/PECOS.git
   cd PECOS
   ```

4. Create the development environment:
   ```sh
   uv sync
   ```

5. **LLVM 14 Setup (Required for LLVM IR/QIS Support)**

   PECOS requires LLVM version 14 for LLVM IR execution features.

   **Quick setup:**
   ```sh
   pecos llvm install
   cargo build
   ```

   For detailed installation instructions for all platforms (macOS, Linux, Windows), see the [**LLVM Setup Guide**](../user-guide/llvm-setup.md).

6. You may wish to explicitly activate the environment for development. To do so:

    === "Linux/Mac"
        ```sh
        source .venv/bin/activate
        ```

    === "Windows"
        ```sh
        .\.venv\Scripts\activate
        ```

6. Build the project in editable mode
    ```sh
   just build
   ```
   Other build options: `just build-release` (optimized), `just build-native` (optimized for your CPU).

7. Run all Python and Rust tests:
   ```sh
   just test
   ```
   Note: Make sure you have run a build command before running tests.

8. Run linters using pre-commit (after [installing it](https://pre-commit.com/)) to make sure all everything is properly linted/formated
   ```sh
   just lint
   ```

9. To deactivate your development venv:
    ```sh
    deactivate
    ```

Before pull requests are merged, they must pass linting and the test.

Note: For the Rust side of the project, you can use `cargo` to run tests, benchmarks, formatting, etc.

## Cleaning Build Artifacts

Clean commands are cross-platform (Windows, macOS, Linux):

```sh
just clean              # Clean project build artifacts
just clean-selene       # Clean Selene plugin artifacts only
just clean-cache        # Clean ~/.pecos/cache/ and ~/.pecos/tmp/
just clean-deps         # Clean ~/.pecos/deps/
just clean-all          # Clean project + cache + deps
just clean-everything   # Nuclear option: includes LLVM and CUDA
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
├── llvm/       # LLVM-14 installation (for QIR/LLVM IR execution)
├── deps/       # Downloaded C++ dependencies (Stim, QuEST, Qulacs, etc.)
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
