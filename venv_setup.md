# Development virtual environment setup

To manually set up a Python virtual environment for develop of this project's code, do the follow:

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

See `Makefile` for other useful commands.
