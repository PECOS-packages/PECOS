# Basic development steps

For developers who want to contribute or modify PECOS:

1. Make sure you have [Python](https://www.python.org/downloads/) and [Rust](https://www.rust-lang.org/tools/install) installed for you system (although you can get away with developing in one or the other).

2. Clone the repository:
   ```sh
   git clone https://github.com/PECOS-packages/PECOS.git
   cd PECOS
   ```

3. It is recommended to [install uv for your system](https://docs.astral.sh/uv/getting-started/installation/).
   After installing `uv`, run following from the root of the project (this will create a `.venv/` that will house the virtual environment):

   ```sh
   uv sync
   ```

   Alternatively, you can use the `Makefile` to use your system's Python to install uv and set up the virtual environment (Note: for Windows to use the `Makefile` you may need to use a shell that has access to Linux commands such as utilizing [git bash](https://gitforwindows.org/)):
   ```sh
    make venv
   ```

4. Once setup, you can use the virtual environment to develop. To so activate it as follows:

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
