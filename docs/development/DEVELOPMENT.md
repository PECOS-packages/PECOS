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
   cargo run -p pecos-llvm-utils --bin pecos-llvm -- install
   cargo build
   ```

   For detailed installation instructions for all platforms (macOS, Linux, Windows), see the [**LLVM Setup Guide**](../user-guide/getting-started.md#llvm-for-qis-support) in the Getting Started documentation.

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
   make build
   ```
   See other build options in the `Makefile`.

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

## Development Guides

For specific development topics, see:

- [Parallel Blocks and Optimization](parallel-blocks-and-optimization.md) - Guide to using and extending the Parallel block construct and optimizer
