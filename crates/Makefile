.DEFAULT_GOAL := help

SHELL=/bin/bash

.PHONY: fmt
fmt:  ## Format Rust
	cargo fmt --all


.PHONY: check
check:  ## Check the features of the main crate
	cargo check -p pecos --all-features

.PHONY: clippy
clippy:  ## Run clippy with all features
	cargo clippy -p pecos --all-features -- -W clippy::dbg_macro

.PHONY: clippy-default
clippy-default:  ## Run clippy with default features
	cargo clippy -p pecos -- -W clippy::dbg_macro

.PHONY: pre-commit
pre-commit: fmt clippy clippy-default  ## Run autoformatting and linting

.PHONY: test
test:  ## Run tests
	cargo test --all-features \
		-p pecos-eng \
		-p pecos-sim \
		-p pecos-noise \
		-- \
		--test-threads=4

.PHONY: publish
publish:  ## Publish Polars crates
	cargo publish --allow-dirty -p pecos-noise
	cargo publish --allow-dirty -p pecos-sim
	cargo publish --allow-dirty -p pecos-eng
	cargo publish --allow-dirty -p pecos

# Help
# ----

help:  ## Show the help menu
	@echo "Available make commands:"
	@echo ""
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
