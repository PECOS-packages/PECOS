# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Exercise release completeness failures before a bundle can be published."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from release_selene_wheels import PLATFORMS, collect, release_plugins, validate


class TestReleaseWheels:
    """Check the explicit release selection and its artifact boundary."""

    def test_complete_and_invalid_sets(self) -> None:
        """Reject omissions, duplicates, unlisted plugins and bare Linux tags."""
        plugins = release_plugins()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheels = []
            for plugin in plugins:
                for tag in PLATFORMS:
                    wheel = root / f"{plugin.replace('-', '_')}-1.0-py3-none-{tag}.whl"
                    wheel.touch()
                    wheels.append(wheel)
            assert len(validate(wheels, plugins, PLATFORMS)) == len(plugins) * len(PLATFORMS)
            with pytest.raises(SystemExit, match="Expected exactly one wheel"):
                validate(wheels[1:], plugins, PLATFORMS)
            with pytest.raises(SystemExit, match="Expected exactly one wheel"):
                validate([*wheels, wheels[0]], plugins, PLATFORMS)
            for filename in (
                "pecos_selene_general_noise-1.0-py3-none-win_amd64.whl",
                "pecos_selene_stabilizer-1.0-py3-none-linux_x86_64.whl",
                "unexpected.txt",
            ):
                extra = root / filename
                extra.touch()
                with pytest.raises(SystemExit, match="Unexpected release wheel"):
                    validate([*wheels, extra], plugins, PLATFORMS)

    def test_collection_validates_before_moving(self) -> None:
        """An incomplete collection must leave all inputs in place."""
        plugins = release_plugins()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = []
            for tag in PLATFORMS:
                directory = root / f"selene-release-wheels-{tag}"
                directory.mkdir()
                for plugin in plugins:
                    wheel = directory / f"{plugin.replace('-', '_')}-1.0-py3-none-{tag}.whl"
                    wheel.touch()
                    files.append(wheel)
            missing = files.pop()
            missing.unlink()
            output = root / "dist"
            with pytest.raises(SystemExit, match="Expected exactly one wheel"):
                collect(root, output, plugins)
            assert not output.exists()
            assert all(wheel.exists() for wheel in files)
            missing.touch()
            collect(root, output, plugins)
            for plugin in plugins:
                assert len(list((output / plugin).glob("*.whl"))) == len(PLATFORMS)
