# QoL: Installation & Dependency Management Polish

Tracked improvements to the build setup, dependency management, and CLI developer experience.

## 1. Consolidate deps under ~/.pecos/deps/

**Status:** DONE

Currently LLVM, CUDA, and cuQuantum live at the top level of `~/.pecos/` while other C++ dependencies (QuEST, Qulacs, Eigen, etc.) live under `~/.pecos/deps/`. Move everything under `deps/` for consistency:

```
~/.pecos/
├── deps/
│   ├── llvm-14/
│   ├── cuda-12.6/
│   ├── cuquantum-25.11/
│   ├── quest-v4.1.0/
│   └── ...
├── cache/
└── tmp/
```

Requires:
- Update `home.rs` functions (`get_llvm_dir`, etc.) to point to new paths
- Add migration: detect old `~/.pecos/llvm/` and either move or symlink
- Update all detection code (`find_llvm_14`, `find_cuda`, `find_cuquantum`) to check new paths
- Update installer code to install to new paths
- Update `.cargo/config.toml` auto-configuration

## 2. Add confirmation prompt to uninstall

**Status:** TODO

`pecos uninstall --all` deletes multi-GB installations with no warning. Use the new `confirm()` utility to ask before destroying. Show what will be removed and how much space.

Requires:
- Add `--yes` flag to uninstall for scripted/CI usage
- Show sizes before confirming
- Reuse `pecos_build::prompt::confirm`

## 3. Fix LLVM config.toml parser fragility

**Status:** TODO

`crates/pecos-build/src/llvm/config.rs` lines 94-142 use hand-rolled string parsing to read `.cargo/config.toml`. The `toml` crate is already a workspace dependency. Use it instead.

## 4. Clean up Justfile naming inconsistencies

**Status:** TODO

- `install-cuda` vs `setup-cuda` is confusing now that `pecos setup` exists
- Old individual install/check recipes (`install-llvm`, `check-llvm`, `configure-llvm`, `install-cuda`, `check-cuda`, `validate-cuda`, `install-cuda-python`, `setup-cuda`) overlap with `pecos setup` and `pecos install`
- Consider deprecating or removing the individual recipes in favor of `just setup` / `pecos install <target>`

## 5. Improve error message consistency

**Status:** TODO

Some commands print helpful context on failure, others just say "not available". Audit `llvm_cmd.rs`, `cuda_cmd.rs`, `cuquantum_cmd.rs` check commands and make error output consistently actionable (what failed, why, what to do).

## 6. Add post-install guidance

**Status:** TODO

After `pecos install llvm` or `pecos setup` completes, print a summary of what to do next:
- "Run `just build` to build PECOS"
- "Run `pecos llvm check` to verify"

Currently the output is terse ("All done.") with no next steps.

## 7. Show disk usage in `pecos list`

**Status:** TODO

`pecos list` shows what's installed but not how much space each takes. Useful when deciding what to clean up.
