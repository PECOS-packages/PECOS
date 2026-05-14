# cmake Setup Guide

!!! note "Python wheel users"
    **Skip this guide.** Pre-built Python wheels already include the MWPF decoder, so cmake is never needed at install time.

This guide is for **anyone building PECOS from source** who wants the optional MWPF (Minimum-Weight Parity Factor) decoder.

## When is cmake Needed?

cmake is **optional**. It is only required when building PECOS with the `mwpf` cargo feature enabled, because MWPF pulls in `highs-sys` (a C++ LP solver) which uses cmake.

Without cmake, every other PECOS decoder still builds and works. The `mwpf` feature is simply off.

## Install Paths

### Path A — PECOS dev using `just` / `pecos setup` (recommended)

`pecos setup` will detect a missing cmake and prompt you:

```text
Install cmake 3.31.12? (~50MB download to ~/.pecos/deps/cmake-3.31.12/, enables the optional MWPF decoder) [Y/n]
```

Answering yes downloads a Kitware-signed binary into PECOS's managed dependency directory and verifies its SHA-256. This is the same pattern PECOS uses for LLVM. If you'd rather drive it directly:

```bash
pecos install cmake          # install the PECOS-managed copy
pecos install cmake --force  # reinstall / upgrade
```

Decline the prompt and `just build` will continue to build everything *except* MWPF.

### Path B — Rust crate user installing from crates.io

If you're consuming `pecos-decoders` (or another PECOS crate) from crates.io and you want `--features mwpf`, you have two options:

1. **Install the PECOS CLI first**, then use it as above:
   ```bash
   cargo install pecos-cli
   pecos install cmake
   cargo add pecos-decoders --features mwpf
   cargo build
   ```

2. **Use your own cmake.** Any cmake ≥ 3.13 on `PATH` works:

    === "macOS"
        ```bash
        brew install cmake
        ```

    === "Debian / Ubuntu"
        ```bash
        sudo apt-get install cmake
        ```

    === "Fedora / RHEL"
        ```bash
        sudo dnf install cmake
        ```

    === "Arch"
        ```bash
        sudo pacman -S cmake
        ```

    === "Windows"
        ```powershell
        winget install --id Kitware.CMake -e
        # Reopen your terminal so cmake is on PATH for the current shell.
        ```

    Or grab a tarball directly from <https://cmake.org/download/>.

We do not auto-download cmake from `build.rs` scripts — that surprises users with hidden network I/O during `cargo build`. The PECOS-managed install is explicit (you ran `pecos install cmake`); the system install is whatever you already did.

### Path C — CI / automation

Use `pecos install cmake` (or `pecos setup --yes`, which accepts every prompt including the cmake one). Mirrors the LLVM install pattern PECOS workflows already use.

## Verifying

```bash
just doctor
```

Should report:

```text
Optional decoders:
  [OK] cmake: 3.31.12 (MWPF decoder available)
```

`pecos python build` will detect cmake automatically and pass `--features mwpf` to maturin. To check the decoder from Python:

```python
from pecos_rslib.qec import ObservableSubgraphDecoder  # MWPF-capable decoder

# Construct with a real DEM + stabilizer coords:
#   decoder = ObservableSubgraphDecoder(dem_str, stab_coords, inner_decoder="mwpf")
```

Set `PECOS_BUILD_MWPF=0` to force MWPF off even when cmake is present (useful for reproducing the lean build locally). `PECOS_BUILD_MWPF=1` forces it on, which is what CI sets.

## Windows Notes

cmake alone is not sufficient on Windows — `highs-sys`'s build needs a working C++ toolchain (MSVC Build Tools). If you don't already have it:

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools -e `
  --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

This is a multi-GB install, so `pecos setup` does not offer to do it for you.
