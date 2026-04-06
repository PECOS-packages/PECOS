# pecos-build

Build utilities and dependency management for PECOS.

## Purpose

Used by build scripts (`build.rs`) to manage external dependencies. Handles downloading, caching, and locating libraries.

## Key Features

- **LLVM 14 management**: Install, configure, and find LLVM 14
- **Dependency downloads**: QuEST, Qulacs, Stim, Eigen, etc.
- **Tool finding**: `find_tool("llvm-as")`, `find_llvm_14()`
- **Manifest parsing**: Load `pecos.toml` for dependency versions

## PECOS Home Directory

All dependencies managed under `~/.pecos/`:

```
~/.pecos/
├── cache/   # Downloaded archives
├── deps/    # Extracted source trees
├── llvm/    # LLVM installation
└── tmp/     # Temporary files
```

## Usage

```rust
// In build.rs
use pecos_build::{ensure_dep_ready, Manifest, find_tool};

let manifest = Manifest::find_and_load()?;
let quest_path = ensure_dep_ready("quest", &manifest)?;
let llvm_as = find_tool("llvm-as");
```
