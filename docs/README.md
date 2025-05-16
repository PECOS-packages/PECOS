# PECOS Documentation

This directory contains the unified documentation for PECOS, implemented using MkDocs with the Material theme.

## Documentation Structure

- `mkdocs.yml`: Configuration file for MkDocs
- `source/`: Markdown source files
- `assets/`: Static assets (images, CSS, JavaScript)
## Setting Up Documentation Development Environment

To work on the documentation, you'll need to install MkDocs and its dependencies:

```bash
# From the repository root
make docs-deps
```

## Building and Previewing Documentation

You can build and preview the documentation in two ways:

### Using the Root Makefile

```bash
# From the repository root
# Clean, install deps, and build documentation
make docs-build  # or simply 'make docs' (alias)

# Serve the built documentation (port 9000)
# (assumes docs have been built)
make docs-serve

# Serve on the default port (8000)
# (assumes docs have been built)
make docs-serve-default

# Typical workflow:
make docs-build && make docs-serve
```

### Using the Docs Makefile

```bash
# From the docs directory
cd docs

# Install dependencies
make install-deps

# Build the documentation
make build

# Serve the documentation (port 9000)
make serve

# Serve on default MkDocs port (8000)
make serve-default

# Test code examples
make test

# Test only working examples (ones marked with ```python executable)
make test-working

# Build and serve in one step
make build-and-serve
```

Then open your browser to http://127.0.0.1:9000 to view the documentation.

## Documentation Organization

The documentation is organized as follows:

- **Home**: Main landing page
- **Getting Started**: Installation and first steps
- **User Guide**: Conceptual documentation
- **API Reference**: Links to language-specific API docs
  - Python API (Sphinx): https://quantum-pecos.readthedocs.io/
  - Rust API (docs.rs): https://docs.rs/pecos/
- **Development**: Contributing to PECOS
- **Releases**: Version history

## Why Separate Source and Assets?

Following Polars' best practices, we separate source Markdown files and assets to:

1. Prevent infinite reloading loops when using `mkdocs serve`
2. Keep the source files clean and focused on content
3. Make maintenance easier

## Makefile Organization

We maintain two Makefiles for documentation:

1. Root `/Makefile` - Contains high-level documentation commands that integrate with the project build system
2. `/docs/Makefile` - Contains specific documentation-related commands with more options

The root Makefile delegates to the docs Makefile for most documentation tasks, providing convenient access to documentation operations from the project root. This approach keeps the documentation build system modular while ensuring it integrates properly with the overall project build.

Key differences:
- Root Makefile commands use the prefix `docs-mkdocs-*` (e.g., `docs-mkdocs-build`)
- Docs Makefile uses direct command names (e.g., `build`)
- All commands in docs/Makefile are accessible through the root Makefile

## Contributing to Documentation

When contributing to the documentation:

1. Place new Markdown files in the `source/` directory
2. Place new images, CSS, or JavaScript in the appropriate subdirectory of `assets/`
3. Update `mkdocs.yml` if adding new pages to the navigation
4. Preview your changes locally before submitting
5. Test your code examples using `make test-working` or `make docs-mkdocs-test-working`

## Building for Production

To build the documentation for production deployment:

```bash
# From the repository root (preferred - includes code testing)
make docs-mkdocs-build-test

# Or just build without testing
make docs-mkdocs-build

# From the docs directory
cd docs && make deploy
```

This will generate a `site/` directory containing the static HTML site. The `-test` variants will first test all the code examples marked as executable to ensure they work correctly.

## Documentation Maintenance

The documentation includes tools for maintenance and quality control:

- `make lint`: Check for broken links and other issues
- `make fix-links`: Analyze and suggest fixes for broken links
- `make fix-nav`: Fix navigation structure and common link patterns
- `make fix-direct`: Apply direct fixes for specific broken links

These tools help resolve warnings shown when building the documentation. The root Makefile also provides access to these commands with the `docs-mkdocs-fix-links`, `docs-mkdocs-fix-nav`, and `docs-mkdocs-fix-direct` targets. For convenience, you can run all fixes with `make docs-mkdocs-fix-all`.