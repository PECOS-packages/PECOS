# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Regression suite for scripts/ci/check_selene_wheels.py.

The combined artifact must contain exactly one wheel for every discovered
plugin and release runner before it can be uploaded.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

CHECKER = Path(__file__).with_name("check_selene_wheels.py")
ROOT = Path(__file__).resolve().parents[2]


def wheel_set(tmp_path: Path, linux_tag: str = "manylinux_2_28_x86_64") -> dict[tuple[str, str], Path]:
    plugins = sorted(path.name for path in (ROOT / "python/selene-plugins").glob("pecos-selene-*/") if path.is_dir())
    assert plugins, "No Selene plugins discovered"
    platforms = {
        "ubuntu-latest": linux_tag,
        "macos-latest": "macosx_14_0_arm64",
        "macos-15-intel": "macosx_15_0_x86_64",
        "windows-2022": "win_amd64",
    }
    wheels = {}
    for plugin in plugins:
        for runner, tag in platforms.items():
            wheel = tmp_path / f"{plugin.replace('-', '_')}-1.0-py3-none-{tag}.whl"
            wheel.touch()
            wheels[plugin, runner] = wheel
    return wheels


def run(tmp_path: Path) -> tuple[int, str]:
    done = subprocess.run([sys.executable, str(CHECKER), str(tmp_path)], capture_output=True, text=True, check=False)
    return done.returncode, done.stdout + done.stderr


@pytest.mark.parametrize("linux_tag", ["manylinux_2_28_x86_64", "linux_x86_64"])
def test_complete_set_passes(tmp_path: Path, linux_tag: str) -> None:
    # uv build actually emits py3-none-linux_x86_64 for these Linux plugins.
    wheels = wheel_set(tmp_path, linux_tag)
    code, output = run(tmp_path)
    assert code == 0, output
    assert f"Validated {len(wheels)} wheels" in output


def test_missing_wheel_names_plugin_and_runner(tmp_path: Path) -> None:
    wheels = wheel_set(tmp_path)
    (plugin, runner), wheel = next(iter(wheels.items()))
    wheel.unlink()
    code, output = run(tmp_path)
    assert code == 1, output
    assert f"Missing wheel: {plugin} on {runner}" in output


def test_duplicate_wheel_fails(tmp_path: Path) -> None:
    wheels = wheel_set(tmp_path)
    (plugin, runner), wheel = next(iter(wheels.items()))
    wheel.with_name(wheel.name.replace("-1.0-", "-2.0-")).touch()
    code, output = run(tmp_path)
    assert code == 1, output
    assert f"Expected exactly one wheel: {plugin} on {runner}" in output


@pytest.mark.parametrize("unexpected", ["unknown_distribution", "unmatched_platform"])
def test_unexpected_wheel_fails(tmp_path: Path, unexpected: str) -> None:
    wheels = wheel_set(tmp_path)
    plugin, _ = next(iter(wheels))
    filename = (
        "unknown_distribution-1.0-py3-none-linux_x86_64.whl"
        if unexpected == "unknown_distribution"
        else f"{plugin.replace('-', '_')}-1.0-py3-none-any.whl"
    )
    (tmp_path / filename).touch()
    code, output = run(tmp_path)
    assert code == 1, output
    assert f"Unexpected wheel: {filename}" in output
