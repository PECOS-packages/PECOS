# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""Custom hatch build hook to compile and include the Rust shared library."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from packaging.tags import sys_tags


def is_cuda_available() -> bool:
    """Check if CUDA is available on the system."""
    # Check for nvcc (CUDA compiler)
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        return True

    # Check common CUDA installation paths
    cuda_paths = [
        Path("/usr/local/cuda/bin/nvcc"),
        Path("/opt/cuda/bin/nvcc"),
    ]
    for path in cuda_paths:
        if path.exists():
            return True

    # Check CUDA_HOME environment variable
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        nvcc = Path(cuda_home) / "bin" / "nvcc"
        if nvcc.exists():
            return True

    return False


class PecosSeleneQuestBuildHook(BuildHookInterface):
    """Build hook that compiles the Rust plugin and copies it to the Python package."""

    def _set_wheel_tag(self, build_data: dict[str, Any]) -> None:
        """Set platform-specific wheel tags.

        This ensures the wheel is marked as platform-specific (not pure Python).
        We use py3-none-{platform} since we don't bind to Python ABI directly.
        """
        build_data["pure_python"] = False

        # Get the appropriate platform tag
        tag = next(
            iter(
                t
                for t in sys_tags()
                if "manylinux" not in t.platform and "musllinux" not in t.platform
            ),
        )
        target_platform = tag.platform
        if sys.platform == "darwin":
            from hatchling.builders.macos import process_macos_plat_tag

            target_platform = process_macos_plat_tag(target_platform, compat=False)
        build_data["tag"] = f"py3-none-{target_platform}"

        self.app.display_info(f"Wheel tag: {build_data['tag']}")

    def initialize(
        self,
        version: str,
        build_data: dict[str, Any],
    ) -> None:
        """Build the Rust library and include it as an artifact."""
        # Get the root directory (where pyproject.toml is)
        root = Path(self.root)

        # Check if library already exists (e.g., from `make build-selene`)
        # If so, skip building and just collect artifacts
        dist_dir = root / "python" / "pecos_selene_quest" / "_dist"
        lib_dir = dist_dir / "lib"
        if lib_dir.exists() and any(lib_dir.iterdir()):
            self.app.display_info("Library already built, skipping cargo build...")
            # Collect artifacts
            artifacts = []
            for artifact in dist_dir.rglob("*"):
                if artifact.is_file():
                    rel_path = artifact.relative_to(root)
                    artifacts.append(str(rel_path.as_posix()))
            if artifacts:
                self.app.display_info("Found existing artifacts:")
                for a in artifacts:
                    self.app.display_info(f"    {a}")
                build_data["artifacts"] += artifacts
                self._set_wheel_tag(build_data)
                return

        # Determine library extension based on platform
        system = platform.system()
        if system == "Linux":
            lib_prefix = "lib"
            lib_suffix = ".so"
        elif system == "Darwin":
            lib_prefix = "lib"
            lib_suffix = ".dylib"
        elif system == "Windows":
            lib_prefix = ""
            lib_suffix = ".dll"
        else:
            msg = f"Unsupported platform: {system}"
            raise RuntimeError(msg)

        lib_name = "pecos_selene_quest"
        cargo_package = "pecos-selene-quest"

        # Check if CUDA is available for CUDA support
        cuda_available = is_cuda_available()
        features = []
        if cuda_available:
            features.append("cuda")
            self.app.display_info(
                f"Building {cargo_package} with CUDA support...",
            )
        else:
            self.app.display_info(
                f"Building {cargo_package} (CPU only, CUDA not detected)...",
            )

        # Run cargo build from the PECOS workspace root
        workspace_root = root.parent.parent  # Go up to PECOS root
        cargo_cmd = [
            "cargo",
            "build",
            "--release",
            "--package",
            cargo_package,
        ]
        if features:
            cargo_cmd.extend(["--features", ",".join(features)])

        result = subprocess.run(
            cargo_cmd,
            check=False,
            cwd=workspace_root,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            self.app.display_error(f"Failed to build {cargo_package}:")
            self.app.display_error(result.stderr)
            msg = f"Cargo build failed for {cargo_package}"
            raise RuntimeError(msg)

        # Find the compiled library
        lib_filename = f"{lib_prefix}{lib_name}{lib_suffix}"
        source_lib = workspace_root / "target" / "release" / lib_filename

        if not source_lib.exists():
            msg = f"Built library not found: {source_lib}"
            raise RuntimeError(msg)

        # Copy to the _dist/lib directory in the Python package
        dest_dir = root / "python" / "pecos_selene_quest" / "_dist" / "lib"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_lib = dest_dir / lib_filename

        self.app.display_info(f"Copying {source_lib} -> {dest_lib}")
        shutil.copy2(source_lib, dest_lib)

        # Also copy the QuEST CUDA backend if it exists (built when --features cuda is used)
        # This backend library is loaded at runtime via dlopen, allowing the wheel to work
        # on systems both with and without NVIDIA CUDA installed.
        cuda_backend_filename = f"{lib_prefix}pecos_quest_cuda{lib_suffix}"
        source_cuda_backend = (
            workspace_root / "target" / "release" / cuda_backend_filename
        )
        if source_cuda_backend.exists():
            dest_cuda_backend = dest_dir / cuda_backend_filename
            self.app.display_info(
                f"Copying QuEST CUDA backend {source_cuda_backend} -> {dest_cuda_backend}",
            )
            shutil.copy2(source_cuda_backend, dest_cuda_backend)
        elif cuda_available:
            # CUDA was requested but backend wasn't built - this is unexpected
            self.app.display_warning(
                f"CUDA detected but QuEST CUDA backend not found at {source_cuda_backend}. "
                "CUDA acceleration may not be available.",
            )

        # Collect artifacts
        artifacts = []
        dist_dir = root / "python" / "pecos_selene_quest" / "_dist"
        for artifact in dist_dir.rglob("*"):
            if artifact.is_file():
                rel_path = artifact.relative_to(root)
                artifacts.append(str(rel_path.as_posix()))

        self.app.display_info("Found artifacts:")
        for a in artifacts:
            self.app.display_info(f"    {a}")

        build_data["artifacts"] += artifacts
        self._set_wheel_tag(build_data)
