"""Wheel packaging must use Cargo's selected artifacts and valid integrity records."""

from __future__ import annotations

import base64
import csv
import importlib.util
import io
import json
import zipfile
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "pecos_build_backend", Path(__file__).parents[1] / "pecos_build_backend.py"
)
assert _SPEC is not None
assert _SPEC.loader is not None
backend = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(backend)


_LIBRARIES = frozenset({"pecos_qis_ffi", "selene_simple_runtime", "selene_soft_rz_runtime"})


def _cargo_output(command: list[str], messages: list[dict]) -> SimpleNamespace:
    if command[1] == "metadata":
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "packages": [
                        {
                            "name": "pecos-qis",
                            "metadata": {"pecos": {"runtime-libraries": sorted(_LIBRARIES)}},
                        }
                    ]
                }
            )
        )
    return SimpleNamespace(stdout="\n".join(map(json.dumps, messages)))


def test_bundle_replaces_stale_libraries_and_updates_record(tmp_path: Path) -> None:
    """Only the current Cargo payload survives, and every wheel hash verifies."""
    wheel = tmp_path / "test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("pecos_rslib/__init__.py", "# package\n")
        archive.writestr("pecos_rslib/_native/stale.so", b"old")
        archive.writestr("test.dist-info/RECORD", "obsolete")
    library = tmp_path / "runtime.so"
    library.write_bytes(b"current runtime")

    backend._bundle_runtime(wheel, [library])

    with zipfile.ZipFile(wheel) as archive:
        assert "pecos_rslib/_native/stale.so" not in archive.namelist()
        assert archive.read("pecos_rslib/_native/runtime.so") == b"current runtime"
        rows = list(csv.reader(io.StringIO(archive.read("test.dist-info/RECORD").decode())))
        assert {row[0] for row in rows} == set(archive.namelist())
        for name, digest, size in rows:
            if name.endswith("/RECORD"):
                assert (digest, size) == ("", "")
                continue
            data = archive.read(name)
            assert digest == "sha256=" + base64.urlsafe_b64encode(sha256(data).digest()).rstrip(b"=").decode()
            assert int(size) == len(data)


def test_runtime_artifacts_use_cargo_messages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hash-suffixed artifacts are selected without guessing from directory contents."""
    messages = []
    expected = []
    for name in sorted(_LIBRARIES):
        path = tmp_path / f"lib{name}-current.so"
        expected.append(path)
        (tmp_path / f"lib{name}-stale.so").write_bytes(b"stale")
        messages.append(
            {
                "reason": "compiler-artifact",
                "target": {"name": name},
                "filenames": [str(path)],
            }
        )
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda command, **kwargs: _cargo_output(command, messages),
    )
    assert backend._runtime_artifacts({"build-args": "--profile dev"}, editable=False) == expected


def test_runtime_artifacts_reject_incomplete_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful Cargo command without runtime outputs cannot produce a wheel."""
    monkeypatch.setattr(backend.subprocess, "run", lambda command, **kwargs: _cargo_output(command, []))
    with pytest.raises(RuntimeError, match="Cargo did not produce required QIS runtime libraries"):
        backend._runtime_artifacts(None, editable=False)


def test_windows_bundle_requires_matching_import_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """QIS compilation on Windows needs the import library beside the runtime DLL."""
    messages = [
        {
            "reason": "compiler-artifact",
            "target": {"name": name},
            "filenames": [str(tmp_path / f"{name}.dll")],
        }
        for name in sorted(_LIBRARIES)
    ]
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda command, **kwargs: _cargo_output(command, messages),
    )
    with pytest.raises(RuntimeError, match="QIS runtime import library"):
        backend._runtime_artifacts(None, editable=False)
    import_lib = tmp_path / "pecos_qis_ffi.dll.lib"
    import_lib.write_bytes(b"import library")
    assert import_lib in backend._runtime_artifacts(None, editable=False)


@pytest.mark.parametrize("key", ["build-args", "maturin.build-args"])
def test_build_options_preserve_target_and_profile(key: str) -> None:
    """Native artifacts follow the frontend's cross-target and custom profile."""
    options = backend._cargo_options(
        {key: "--target aarch64-apple-darwin --target-dir custom-target --profile native --locked"},
        editable=False,
    )
    assert options == [
        "--profile",
        "native",
        "--target",
        "aarch64-apple-darwin",
        "--target-dir",
        "custom-target",
        "--locked",
    ]


@pytest.mark.parametrize("flags", ["--locked", "--offline --frozen"])
def test_metadata_uses_manifest_and_cargo_policy(flags: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Metadata is queried through Cargo with the frontend's resolution policy."""
    commands = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        commands.append(command)
        return _cargo_output(command, [])

    monkeypatch.setattr(backend.subprocess, "run", run)
    assert backend._runtime_libraries(flags.split()) == _LIBRARIES
    assert commands == [
        [
            "cargo",
            "metadata",
            "--no-deps",
            "--format-version",
            "1",
            "--manifest-path",
            str(backend._ROOT / "Cargo.toml"),
            *flags.split(),
        ]
    ]


@pytest.mark.parametrize("metadata", [None, {}, {"pecos": {}}, {"pecos": {"runtime-libraries": []}}])
def test_metadata_without_runtime_libraries_names_the_manifest_key(
    metadata: dict | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing or empty library declaration is a named error, not a KeyError."""
    output = json.dumps({"packages": [{"name": "pecos-qis", "metadata": metadata}]})
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(stdout=output),
    )
    with pytest.raises(RuntimeError, match=r"\[package\.metadata\.pecos\] runtime-libraries"):
        backend._runtime_libraries([])
