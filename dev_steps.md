# Basic development steps

For developers who want to contribute or modify PECOS:

1. Make sure you have [Python](https://www.python.org/downloads/) and [Rust](https://www.rust-lang.org/tools/install) installed for you system (although you can get away with developing in one or the other).

2. Clone the repository:
   ```sh
   git clone https://github.com/PECOS-packages/PECOS.git
   cd PECOS
   ```

3. Set up the development environment either using the `Makefile` (Note: for Windows to use the `Makefile` you may need to use a shell that has access to Linux commands such as utilizing [git bash](https://gitforwindows.org/)):
   ```sh
   make venv
   ```
    or manually following [these steps](venv_setup.md).

    When developing in the development environment, be sure to activate the venv:

    On Linux/Mac:
    ```sh
    source .venv/bin/activate
    ```

    On Windows:
    ```sh
    .\venv\Scripts\activate
    ```

4. Build the project in editable mode
    ```sh
   make build
   ```
   See other build options in the `Makefile`.

5. Run all Python and Rust tests:
   ```sh
   make test
   ```
   Note: Make sure you have ran a build command before running tests.

6. Run pre-commit (after [installing it](https://pre-commit.com/)) to make sure all everything is properly linted/formated
   ```sh
   make pre-commit
   ```

7. To deactivate your development venv:
    ```sh
    deactivate
    ```

Note: For the Rust side of the project, you can use `cargo` to run tests, benchmarks, formatting, etc.
