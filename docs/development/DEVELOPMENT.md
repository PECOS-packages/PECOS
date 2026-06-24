# Basic development steps

## Requirements

**Full development** (Python + Rust, recommended):

- [Python 3.10+](https://www.python.org/downloads/)
- [Rust](https://www.rust-lang.org/tools/install) (stable toolchain)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) - Python package manager
- [just](https://github.com/casey/just) - Command runner
- [pecos](https://crates.io/crates/pecos) - PECOS dev tools CLI
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
   | `cuda`        | Building/running GPU simulators (requires CUDA toolkit)     | `uv sync --group cuda`        |

   Combine groups with multiple `--group` flags
   (e.g. `uv sync --group examples --group cuda`).

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
