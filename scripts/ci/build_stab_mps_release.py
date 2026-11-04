# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Build the StabMps release library with Cargo.lock before Hatch packages it.

Called by cibuildwheel inside the target build environment. Prebuilding makes
the existing Hatch hook collect the locked library rather than invoke Cargo
without --locked. Each release runner starts from a clean checkout.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Build and stage the native library for the existing Hatch hook."""
    root = Path(__file__).resolve().parents[2]
    cargo = shutil.which("cargo")
    if cargo is None:
        msg = "Cargo is required to build the StabMps release library"
        raise RuntimeError(msg)
    subprocess.run(
        [cargo, "+stable", "build", "--locked", "--release", "-p", "pecos-selene-stab-mps"],
        cwd=root,
        check=True,
    )
    names = {
        "linux": "libpecos_selene_stab_mps.so",
        "darwin": "libpecos_selene_stab_mps.dylib",
        "win32": "pecos_selene_stab_mps.dll",
    }
    name = names[sys.platform]
    destination = root / "python/selene-plugins/pecos-selene-stab-mps/python/pecos_selene_stab_mps/_dist/lib"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "target/release" / name, destination / name)


if __name__ == "__main__":
    main()
