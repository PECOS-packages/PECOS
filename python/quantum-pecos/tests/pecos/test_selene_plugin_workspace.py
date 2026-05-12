"""Checks for Selene plugin workspace consistency."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _nonempty_selene_plugin_dirs(repo_root: Path) -> tuple[list[str], list[str]]:
    """Return real plugin dirs and stray non-empty dirs."""
    plugin_root = repo_root / "python" / "selene-plugins"
    real_dirs: list[str] = []
    stray_dirs: list[str] = []

    for path in sorted(plugin_root.glob("pecos-selene-*")):
        if not path.is_dir():
            continue
        children = list(path.iterdir())
        if not children:
            continue
        # Cargo.toml / pyproject.toml members use forward slashes on all platforms;
        # normalize via as_posix() so the comparison does not fail on Windows.
        rel = path.relative_to(repo_root).as_posix()
        if (path / "Cargo.toml").is_file() and (path / "pyproject.toml").is_file():
            real_dirs.append(rel)
        else:
            stray_dirs.append(rel)
    return real_dirs, stray_dirs


def test_selene_plugin_workspace_members_are_explicit_and_complete() -> None:
    """Workspace manifests should enumerate exactly the real Selene plugin packages."""
    repo_root = _repo_root()

    cargo_toml = _load_toml(repo_root / "Cargo.toml")
    pyproject_toml = _load_toml(repo_root / "pyproject.toml")

    cargo_members = cargo_toml["workspace"]["members"]
    uv_members = pyproject_toml["tool"]["uv"]["workspace"]["members"]

    cargo_plugin_members = [member for member in cargo_members if member.startswith("python/selene-plugins/")]
    uv_plugin_members = [member for member in uv_members if member.startswith("python/selene-plugins/")]

    assert all(
        "*" not in member for member in cargo_plugin_members
    ), "Cargo workspace should list Selene plugins explicitly instead of using a wildcard"
    assert all(
        "*" not in member for member in uv_plugin_members
    ), "uv workspace should list Selene plugins explicitly instead of using a wildcard"

    actual_plugin_dirs, stray_dirs = _nonempty_selene_plugin_dirs(repo_root)
    msg = f"Found non-empty pecos-selene-* directories that are not real plugin packages: {stray_dirs}"
    assert stray_dirs == [], msg
    assert cargo_plugin_members == actual_plugin_dirs, (
        "Cargo workspace Selene plugin members are out of sync with the actual plugin packages on disk: "
        f"{cargo_plugin_members} vs {actual_plugin_dirs}"
    )
    assert uv_plugin_members == actual_plugin_dirs, (
        "uv workspace Selene plugin members are out of sync with the actual plugin packages on disk: "
        f"{uv_plugin_members} vs {actual_plugin_dirs}"
    )
