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

4. You may wish to explicitly activate the environment for development. To do so:

    On Linux/Mac:
    ```sh
    source .venv/bin/activate
    ```

    On Windows:
    ```sh
    .\.venv\Scripts\activate
    ```

5. Build the project in editable mode
    ```sh
   make build
   ```
   See other build options in the `Makefile`.

6. Run all Python and Rust tests:
   ```sh
   make test
   ```
   Note: Make sure you have run a build command before running tests.

7. Run linters using pre-commit (after [installing it](https://pre-commit.com/)) to make sure all everything is properly linted/formated
   ```sh
   make lint
   ```

8. To deactivate your development venv:
    ```sh
    deactivate
    ```

Before pull requests are merged, they must pass linting and the test.

Note: For the Rust side of the project, you can use `cargo` to run tests, benchmarks, formatting, etc.
