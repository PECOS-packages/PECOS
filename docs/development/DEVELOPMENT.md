# Basic development steps

For developers who want to contribute or modify PECOS:

1. Make sure you have [Python](https://www.python.org/downloads/) and [Rust](https://www.rust-lang.org/tools/install) installed for you system (although you can get away with developing in one or the other).

2. Clone the repository:
   ```sh
   git clone https://github.com/PECOS-packages/PECOS.git
   cd PECOS
   ```

3. [Install `uv` for your system](https://docs.astral.sh/uv/getting-started/installation/).
   And run the following at the root of the project to create a development environment, which will be stored in `.venv/`:
   ```sh
   uv sync
   ```

4. **LLVM 14 Setup (Required for LLVM IR/QIS Support)**

   PECOS requires LLVM version 14 for LLVM IR execution features.

   **Quick setup:**
   ```sh
   cargo run -p pecos-dev -- llvm install
   cargo build
   ```

   For detailed installation instructions for all platforms (macOS, Linux, Windows), see the [**LLVM Setup Guide**](../user-guide/llvm-setup.md).

5. You may wish to explicitly activate the environment for development. To do so:

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
   make build-dev
   ```
   Other build options: `make build-release` (optimized), `make build-native` (optimized for your CPU).

7. Run all Python and Rust tests:
   ```sh
   make test
   ```
   Note: Make sure you have run a build command before running tests.

8. Run linters using pre-commit (after [installing it](https://pre-commit.com/)) to make sure all everything is properly linted/formated
   ```sh
   make lint
   ```

9. To deactivate your development venv:
    ```sh
    deactivate
    ```

Before pull requests are merged, they must pass linting and the test.

Note: For the Rust side of the project, you can use `cargo` to run tests, benchmarks, formatting, etc.

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

For more details, see [PECOS Home Directory Plan](PECOS_HOME_PLAN.md).

## Development Guides

For specific development topics, see:

- [Parallel Blocks and Optimization](parallel-blocks-and-optimization.md) - Guide to using and extending the Parallel block construct and optimizer
- [PECOS Home Directory Plan](PECOS_HOME_PLAN.md) - External dependency management architecture
