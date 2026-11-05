#!/usr/bin/env python3
# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Build, verify and collect the explicitly selected Selene release wheels."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from check_wheel_platform_tags import platform_tags

ROOT = Path(__file__).resolve().parents[2]
PLATFORMS = ("manylinux_2_28_x86_64", "macosx_13_0_arm64", "macosx_13_0_x86_64", "win_amd64")


def release_plugins() -> list[str]:
    """Read the release selection, ignoring blank and comment lines."""
    path = ROOT / "python/selene-plugins/release-plugins.txt"
    plugins = [
        line.strip() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    if not plugins:
        message = f"No Selene release plugins listed in {path}"
        raise SystemExit(message)
    return plugins


def run(*command: str, cwd: Path = ROOT) -> None:
    """Run a release step, failing immediately on errors."""
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def validate(wheels: list[Path], plugins: list[str], platforms: tuple[str, ...]) -> dict[Path, str]:
    """Require exactly one wheel per selected distribution and platform."""
    matches: dict[tuple[str, str], list[Path]] = {(p, t): [] for p in plugins for t in platforms}
    names = {p.replace("-", "_"): p for p in plugins}
    result = {}
    for wheel in wheels:
        plugin = names.get(wheel.name.split("-", 1)[0])
        tags = platform_tags(wheel) or set()
        expected = tags.intersection(platforms)
        if not wheel.is_file() or wheel.suffix != ".whl" or plugin is None or len(expected) != 1:
            message = f"Unexpected release wheel: {wheel}"
            raise SystemExit(message)
        matches[plugin, expected.pop()].append(wheel)
        result[wheel] = plugin
    for (plugin, tag), files in matches.items():
        if len(files) != 1:
            message = f"Expected exactly one wheel for {plugin} / {tag}; found {files}"
            raise SystemExit(message)
    print(f"Validated {len(wheels)} Selene release wheels", flush=True)
    return result


def build(output: Path, plugins: list[str]) -> None:
    """Stage separately built cdylibs so Hatch never invokes unlocked Cargo."""
    output.mkdir(parents=True, exist_ok=False)
    prefix, suffix = {"linux": ("lib", "so"), "darwin": ("lib", "dylib"), "win32": ("", "dll")}[sys.platform]
    for plugin in plugins:
        package = plugin.replace("-", "_")
        directory = ROOT / "python/selene-plugins" / plugin
        # Separate invocations prevent feature unification across shipped libraries.
        run("cargo", "build", "--locked", "--release", "-p", plugin)
        stage = directory / "python" / package / "_dist"
        if stage.exists():
            shutil.rmtree(stage)
        library = stage / "lib"
        library.mkdir(parents=True)
        target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
        shutil.copy2(target / "release" / f"{prefix}{package}.{suffix}", library)
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            run(sys.executable, "-m", "build", "--wheel", "--outdir", str(raw), str(directory))
            wheels = list(raw.glob("*.whl"))
            if len(wheels) != 1:
                message = f"Expected one built wheel for {plugin}: {wheels}"
                raise SystemExit(message)
            if sys.platform == "linux":
                run("auditwheel", "repair", "--plat", PLATFORMS[0], "-w", str(output), str(wheels[0]))
            else:
                shutil.copy2(wheels[0], output)


def verify(output: Path, plugins: list[str], tag: str) -> None:
    """Check tags and load each installed wheel in its own clean environment."""
    wheels = validate(list(output.iterdir()), plugins, (tag,))
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        constraints = work / "constraints.txt"
        run("uv", "export", "--frozen", "--no-hashes", "--all-packages", "--no-emit-workspace", "-o", str(constraints))
        for wheel, plugin in wheels.items():
            run(
                sys.executable,
                str(ROOT / "scripts/ci/check_wheel_platform_tags.py"),
                "--expected-platform-tag",
                tag,
                str(wheel),
            )
            if sys.platform == "darwin":
                run(
                    sys.executable,
                    str(ROOT / "scripts/ci/check_macos_wheel_minos.py"),
                    "--maximum-version",
                    os.environ["MACOS_WHEEL_DEPLOYMENT_TARGET"],
                    str(wheel),
                )
            venv = work / plugin
            run("uv", "venv", str(venv), "--python", sys.executable)
            run("uv", "pip", "install", "--python", str(venv), "-c", str(constraints), str(wheel))
            python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            run(str(python), str(ROOT / "scripts/ci/smoke_selene_wheel.py"), plugin.replace("-", "_"), cwd=work)


def collect(artifacts: Path, output: Path, plugins: list[str]) -> None:
    """Validate the complete input before moving any wheels into the bundle."""
    directories = list(artifacts.glob("selene-release-wheels-*"))
    if len(directories) != len(PLATFORMS) or any(not p.is_dir() for p in directories):
        message = f"Expected four Selene release artifact directories: {directories}"
        raise SystemExit(message)
    wheels = validate([p for directory in directories for p in directory.iterdir()], plugins, PLATFORMS)
    for plugin in plugins:
        (output / plugin).mkdir(parents=True, exist_ok=False)
    for wheel, plugin in wheels.items():
        shutil.move(str(wheel), output / plugin / wheel.name)


def main() -> None:
    """Dispatch one release task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify", "collect", "summary"))
    parser.add_argument("--output", type=Path, default=Path("selene-release-wheels"))
    parser.add_argument("--artifacts", type=Path, default=Path("temp-artifacts"))
    parser.add_argument("--platform", choices=PLATFORMS)
    args = parser.parse_args()
    plugins = release_plugins()
    output = args.output.resolve()
    if args.command == "build":
        build(output, plugins)
    elif args.command == "verify":
        if not args.platform:
            parser.error("verify requires --platform")
        verify(output, plugins, args.platform)
    elif args.command == "collect":
        collect(args.artifacts.resolve(), output, plugins)
    else:
        for plugin in plugins:
            wheels = sorted((output / plugin).iterdir())
            validate(wheels, [plugin], PLATFORMS)
            print(f"=== {plugin}: {len(wheels)} wheels ===")
            for wheel in wheels:
                print(wheel.name)


if __name__ == "__main__":
    main()
