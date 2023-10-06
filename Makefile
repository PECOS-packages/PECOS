.PHONY: install dev tests lint docs clean build metadeps updatereqs

install:
	pip install .

install-all:
	pip install .[all]

dev:
	pip install -e .

dev-all:
	pip install -e .[all]

tests: install
	pytest tests

lint:
	pre-commit run --all-files

docs: install
	cd docs && make clean && make html && cd -

clean:
	rm -rf *.egg-info dist build docs/_build

build: clean
	python -m build --sdist --wheel -n

metadeps:
	pip install -U build pip-tools pre-commit sphinx_rtd_theme wheel

updatereqs:
	pip install -U pip-tools
	rm requirements.txt
	pip-compile --extra=tests --no-annotate --no-emit-index-url --output-file=requirements.txt --strip-extras pyproject.toml
