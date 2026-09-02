"""Build and bundle the native Selene error-model plugin."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from packaging.tags import sys_tags


class PecosSeleneGeneralNoiseBuildHook(BuildHookInterface):
    """Compile the Rust cdylib and add it to the platform wheel."""

    def _set_wheel_tag(self, build_data: dict[str, Any]) -> None:
        build_data["pure_python"] = False
        tag = next(tag for tag in sys_tags() if "manylinux" not in tag.platform and "musllinux" not in tag.platform)
        target_platform = tag.platform
        if sys.platform == "darwin":
            from hatchling.builders.macos import process_macos_plat_tag

            target_platform = process_macos_plat_tag(target_platform, compat=False)
        build_data["tag"] = f"py3-none-{target_platform}"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Build the native library unless an existing artifact is available."""
        root = Path(self.root)
        dist_dir = root / "python" / "pecos_selene_general_noise" / "_dist"
        lib_dir = dist_dir / "lib"

        system = platform.system()
        if system == "Linux":
            prefix, suffix = "lib", ".so"
        elif system == "Darwin":
            prefix, suffix = "lib", ".dylib"
        elif system == "Windows":
            prefix, suffix = "", ".dll"
        else:
            message = f"Unsupported platform: {system}"
            raise RuntimeError(message)

        workspace_root = root.parent.parent.parent
        subprocess.run(
            ["cargo", "build", "--release", "--package", "pecos-selene-general-noise"],
            cwd=workspace_root,
            check=True,
        )
        filename = f"{prefix}pecos_selene_general_noise{suffix}"
        source = workspace_root / "target" / "release" / filename
        lib_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, lib_dir / filename)
        build_data["artifacts"] += [
            artifact.relative_to(root).as_posix() for artifact in dist_dir.rglob("*") if artifact.is_file()
        ]
        self._set_wheel_tag(build_data)
