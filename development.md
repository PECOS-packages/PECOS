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
   <details>
   <summary>or manually set up a Python virtual environment for develop of this project's code.</summary>

   On Linux/Mac:

   ```sh
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip setuptools
   pip install -r python/quantum-pecos/requirements.txt
   make metadeps
   pre-commit install
   ```

   On Windows:

   ```sh
   python -m venv .venv
   .\venv\Scripts\activate
   pip install -U pip setuptools
   pip install -r python/quantum-pecos/requirements.txt
   make metadeps
   pre-commit install
   ```
   </details>


4. When developing in the development environment, be sure to activate the venv:

    On Linux/Mac:
    ```sh
    source .venv/bin/activate
    ```

    On Windows:
    ```sh
    .\venv\Scripts\activate
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

Note: For the Rust side of the project, you can use `cargo` to run tests, benchmarks, formatting, etc.
