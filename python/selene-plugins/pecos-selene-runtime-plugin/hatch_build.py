"""Build the standard runtime plugin from this PECOS workspace."""

import json
import platform
import shutil
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from packaging.tags import sys_tags


class RuntimeBuildHook(BuildHookInterface):
    """Build and package the workspace runtime library."""

    def initialize(self, version: str, build_data: dict) -> None:
        """Compile the plugin and tag the wheel for the current platform."""
        root = Path(self.root)
        workspace = root.parents[2]
        subprocess.run(
            ["cargo", "build", "--locked", "--release", "-p", "pecos-selene-runtime-plugin"],
            cwd=workspace,
            check=True,
        )
        metadata_json = subprocess.check_output(
            ["cargo", "metadata", "--format-version=1", "--no-deps"],
            cwd=workspace,
        )
        metadata = json.loads(metadata_json)
        filename = {
            "Darwin": "libpecos_selene_runtime_plugin.dylib",
            "Linux": "libpecos_selene_runtime_plugin.so",
            "Windows": "pecos_selene_runtime_plugin.dll",
        }[platform.system()]
        dest = root / "python" / "pecos_selene_runtime_plugin" / "_dist" / "lib" / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(metadata["target_directory"]) / "release" / filename, dest)
        build_data["artifacts"].append(dest.relative_to(root).as_posix())
        build_data["pure_python"] = False
        tag = next(t for t in sys_tags() if "manylinux" not in t.platform and "musllinux" not in t.platform)
        target = tag.platform
        if platform.system() == "Darwin":
            from hatchling.builders.macos import process_macos_plat_tag

            target = process_macos_plat_tag(target, compat=False)
        build_data["tag"] = f"py3-none-{target}"
