# pecos-build

Build utilities and dependency management for PECOS.

## Purpose

Used by build scripts (`build.rs`) to manage external dependencies. Handles downloading, caching, and locating libraries.

## Key Features

- **LLVM 21.1 management**: Install where PECOS can provide shared LLVM, configure, and find LLVM 21.1
- **Dependency downloads**: QuEST, Qulacs, Stim, Eigen, etc.
- **Tool finding**: `find_tool("llvm-as")`, `find_llvm()`
- **Manifest parsing**: Load `pecos.toml` for dependency versions

## PECOS Home Directory

All dependencies managed under `~/.pecos/`:

```
~/.pecos/
├── cache/   # Downloaded archives
├── deps/    # Extracted toolchains and source trees, including llvm-21.1/
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
