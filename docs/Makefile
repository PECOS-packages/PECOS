# Documentation Makefile

.PHONY: build
build:  ## Build the documentation
	uv run mkdocs build --clean

.PHONY: serve
serve:  ## Serve the documentation locally (port 9000)
	uv run mkdocs serve --dev-addr=127.0.0.1:9000

.PHONY: serve-default
serve-default:  ## Serve the documentation on default port (8000)
	uv run mkdocs serve

.PHONY: test
test:  ## Test all code examples in documentation
	uv run python test_code_examples.py

.PHONY: test-working
test-working:  ## Test only working code examples
	uv run python test_working_examples.py

.PHONY: lint
lint:  ## Check for broken links in the documentation
	uv run mkdocs build --clean --strict

.PHONY: clean
clean:  ## Clean the build directory
	@echo "Cleaning documentation build directory"
	@rm -rf site

.PHONY: install-deps
install-deps:  ## Install documentation dependencies
	@echo "Installing documentation dependencies..."
	uv pip install -r requirements.txt

.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help